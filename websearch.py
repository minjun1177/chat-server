"""Web search that is allowed to come back empty.

The old implementation handed whatever DuckDuckGo returned straight to the
model. For a query like "ollama num_ctx meaning" DDG drops the discriminative
term and answers the entity instead: ollama.com, the Windows download page,
five Korean install blogs. None of them contain "num_ctx" anywhere, but the
model received them as if they were the answer and wrote a confident, wrong
reply.

So this module does three things the old one did not:

1. Asks several complementary free sources, not one. A general web index is bad
   at code identifiers; GitHub and Stack Exchange are good at exactly that, and
   Wikipedia is good at concepts. All are keyless.
2. Reads the actual pages instead of trusting 150-character snippets, and ranks
   passages with BM25 whose IDF comes from the candidate pool itself, so a term
   that shows up in every candidate carries no weight and a rare one dominates.
3. Refuses to answer. If the rare terms of the query appear in nothing that came
   back, the result says so. An honest miss costs one retry; a plausible wrong
   answer is spent as a fact.

Free and keyless throughout. Everything except the optional SearXNG instance
runs against public no-auth endpoints; point config.SEARXNG_URL at a local
SearXNG and the pipeline becomes fully self-hosted for candidate generation too.
"""

import concurrent.futures as futures
import html
import math
import re
import time

import requests
from bs4 import BeautifulSoup

import config

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
}

_HANGUL = re.compile(r"[가-힣]+")
_WORD = re.compile(r"[A-Za-z0-9_]+")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+|[a-z]+[A-Z][A-Za-z0-9]*")
_SPLIT_ID = re.compile(r"[_]+|(?<=[a-z0-9])(?=[A-Z])")

# words that carry no retrieval signal; they only dilute BM25
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "of", "to", "in", "on",
    "for", "and", "or", "what", "how", "why", "when", "where", "which", "who",
    "does", "do", "did", "can", "i", "my", "it", "its", "this", "that", "with",
    "meaning", "means", "about", "please", "tell", "me", "explain", "example",
    "뭐야", "뭔가요", "무엇", "알려줘", "방법", "사용법", "어떻게", "why",
}


# ---------------------------------------------------------------------------
# tokenising
# ---------------------------------------------------------------------------

def _tokenize(text):
    """Tokens for BM25.

    Latin words are lowercased; an identifier like `num_ctx` is kept whole *and*
    split, so it matches both `num_ctx` and prose that says "num" and "ctx".
    Korean has no spaces between a word and its particle, so Hangul runs become
    character bigrams - the standard CJK trick, and it needs no analyzer.
    """
    out = []
    for word in _WORD.findall(text or ""):
        low = word.lower()
        out.append(low)
        if "_" in word or _IDENTIFIER.fullmatch(word):
            out.extend(p.lower() for p in _SPLIT_ID.split(word) if p)
    for run in _HANGUL.findall(text or ""):
        if len(run) == 1:
            out.append(run)
        else:
            out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return out


def distill(query):
    """Strip conversational filler, keep the terms that discriminate.

    "ollama num_ctx meaning" -> "ollama num_ctx". Keyword APIs like GitHub and
    Stack Exchange match this well and match the raw sentence badly: sending the
    full phrase to GitHub returns generic ollama issues, sending the distilled
    form returns the issues that actually discuss num_ctx.
    """
    kept = [w for w in re.split(r"\s+", (query or "").strip())
            if w and w.lower().strip("?!.,") not in _STOP]
    return " ".join(kept) or (query or "").strip()


def _query_terms(query):
    """(scoring terms, identifier terms the answer must contain)."""
    terms = [t for t in _tokenize(distill(query)) if t not in _STOP and len(t) > 1]
    must = [t.lower() for t in _IDENTIFIER.findall(query or "")]
    return terms, must


def _discriminative(terms, must_ids, doc_freq, n_docs):
    """Terms a genuine hit has to contain.

    Identifiers always count. Beyond those, let the candidate pool decide: a
    term carried by few of the passages is what the query is really about, while
    one in nearly all of them ("python", "ollama") separates nothing. This is
    what catches `asyncio`, which no identifier pattern would match.
    """
    if must_ids:
        # an explicit identifier is never incidental - require every one of them
        return list(dict.fromkeys(must_ids))
    rare = [t for t in dict.fromkeys(terms)
            if len(t) >= 3 and not _HANGUL.match(t)
            and doc_freq.get(t, 0) <= max(1, n_docs * 0.4)]
    rare.sort(key=lambda t: doc_freq.get(t, 0))
    return rare[:1]


def _is_code_query(query):
    if _IDENTIFIER.search(query or ""):
        return True
    return bool(re.search(r"[`(){}\[\]]|\.\w+\(|error|exception|traceback", query or "", re.I))


def _has_hangul(text):
    return bool(_HANGUL.search(text or ""))


def strip_html(raw):
    """HTML to readable text. Shared with the get_url tool."""
    soup = BeautifulSoup(raw or "", "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe", "svg", "form"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


# ---------------------------------------------------------------------------
# candidate sources - every one of these is free and needs no API key
# ---------------------------------------------------------------------------

def _src_searxng(query):
    """A local SearXNG, if the user runs one. Best quality, fully self-hosted."""
    base = (config.SEARXNG_URL or "").rstrip("/")
    if not base:
        return []
    resp = requests.get(
        f"{base}/search",
        params={"q": query, "format": "json", "safesearch": 1},
        headers=UA, timeout=config.SEARCH_SOURCE_TIMEOUT,
    )
    resp.raise_for_status()
    out = []
    for item in resp.json().get("results", [])[:config.SEARCH_CANDIDATES]:
        if item.get("url"):
            out.append({
                "url": item["url"], "title": item.get("title", ""),
                "snippet": item.get("content", ""), "text": "", "source": "searxng",
            })
    return out


def _src_ddg(query):
    """General web. Broad coverage, weak precision - the pool, not the answer."""
    try:
        from duckduckgo_search import DDGS  # pinned in requirements.txt
    except ImportError:
        from ddgs import DDGS               # the package's new name
    region = "kr-kr" if _has_hangul(query) else "wt-wt"
    results = DDGS().text(query, region=region, safesearch="moderate",
                          max_results=config.SEARCH_CANDIDATES)
    out = []
    for item in results or []:
        if item.get("href"):
            out.append({
                "url": item["href"], "title": item.get("title", ""),
                "snippet": item.get("body", ""), "text": "", "source": "ddg",
            })
    return out


def _src_wikipedia(query):
    """Encyclopedic backstop. Official API, no key."""
    lang = "ko" if _has_hangul(query) else "en"
    resp = requests.get(
        f"https://{lang}.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": query,
                "format": "json", "srlimit": 3},
        headers=UA, timeout=config.SEARCH_SOURCE_TIMEOUT,
    )
    resp.raise_for_status()
    out = []
    for item in resp.json().get("query", {}).get("search", []):
        title = item.get("title", "")
        out.append({
            "url": f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "title": title,
            "snippet": re.sub(r"<[^>]+>", "", item.get("snippet", "")),
            "text": "", "source": "wikipedia",
        })
    return out


def _src_stackexchange(query):
    """Technical Q&A. Keyless quota is 300/day; bodies come back inline."""
    resp = requests.get(
        "https://api.stackexchange.com/2.3/search/advanced",
        params={"order": "desc", "sort": "relevance", "q": distill(query),
                "site": "stackoverflow", "filter": "withbody", "pagesize": 5},
        headers=UA, timeout=config.SEARCH_SOURCE_TIMEOUT,
    )
    resp.raise_for_status()
    out = []
    for item in resp.json().get("items", []):
        body = strip_html(item.get("body", ""))[:config.SEARCH_PAGE_CHARS]
        out.append({
            "url": item.get("link", ""),
            "title": html.unescape(item.get("title", "")),
            "snippet": body[:300], "text": body, "source": "stackexchange",
        })
    return out


def _src_github(query):
    """Where code identifiers actually live. Keyless issue search, 10/min."""
    resp = requests.get(
        "https://api.github.com/search/issues",
        params={"q": distill(query), "per_page": 5, "sort": "reactions", "order": "desc"},
        headers={**UA, "Accept": "application/vnd.github+json"},
        timeout=config.SEARCH_SOURCE_TIMEOUT,
    )
    resp.raise_for_status()
    out = []
    for item in resp.json().get("items", []):
        body = (item.get("body") or "")[:config.SEARCH_PAGE_CHARS]
        out.append({
            "url": item.get("html_url", ""),
            "title": item.get("title", ""),
            "snippet": body[:300], "text": body, "source": "github",
        })
    return out


def _pick_sources(query):
    """Spend requests where the query shape says the answer lives."""
    chosen = []
    if config.SEARXNG_URL:
        chosen.append(("searxng", _src_searxng))
    chosen.append(("ddg", _src_ddg))
    if _is_code_query(query):
        chosen.append(("github", _src_github))
        chosen.append(("stackexchange", _src_stackexchange))
    else:
        chosen.append(("wikipedia", _src_wikipedia))
        chosen.append(("stackexchange", _src_stackexchange))
    return chosen


def _gather(query):
    """Run the chosen sources concurrently; one failing must not sink the rest."""
    candidates, notes = [], []
    sources = _pick_sources(query)
    with futures.ThreadPoolExecutor(max_workers=len(sources)) as pool:
        running = {pool.submit(fn, query): name for name, fn in sources}
        try:
            for future in futures.as_completed(running, timeout=config.SEARCH_TOTAL_TIMEOUT):
                name = running[future]
                try:
                    found = future.result()
                    notes.append(f"{name}:{len(found)}")
                    candidates.extend(found)
                except Exception as e:
                    notes.append(f"{name}:failed({type(e).__name__})")
        except futures.TimeoutError:
            # keep whatever already came back rather than losing the whole search
            for future, name in running.items():
                if not future.done():
                    notes.append(f"{name}:timeout")

    seen, unique = set(), []
    for cand in candidates:
        key = cand["url"].rstrip("/")
        if key and key not in seen:
            seen.add(key)
            unique.append(cand)
    return unique, notes


# ---------------------------------------------------------------------------
# fetching and ranking
# ---------------------------------------------------------------------------

def _fetch(cand):
    if cand["text"]:
        return cand
    try:
        resp = requests.get(cand["url"], headers=UA, timeout=config.SEARCH_FETCH_TIMEOUT)
        resp.raise_for_status()
        if "html" in resp.headers.get("Content-Type", "").lower():
            cand["text"] = strip_html(resp.text)[:config.SEARCH_PAGE_CHARS]
        else:
            cand["text"] = resp.text[:config.SEARCH_PAGE_CHARS]
    except Exception:
        cand["text"] = ""
    return cand


def _fetch_all(candidates):
    """API sources already carry their text; spend the fetch budget on the rest."""
    ready = [c for c in candidates if c["text"]]
    targets = [c for c in candidates if not c["text"]][:config.SEARCH_FETCH_PAGES]
    if not targets:
        return ready
    with futures.ThreadPoolExecutor(max_workers=min(8, len(targets))) as pool:
        return ready + list(pool.map(_fetch, targets))


def _passages(candidates):
    """Split each page into passages so a long page cannot win on length alone."""
    out = []
    for cand in candidates:
        text = cand["text"] or cand["snippet"]
        if not text:
            continue
        buf, size = [], 0
        for line in text.split("\n"):
            if not line.strip():
                continue
            buf.append(line)
            size += len(line)
            if size >= config.SEARCH_PASSAGE_CHARS:
                out.append((cand, " ".join(buf)))
                buf, size = [], 0
        if buf:
            out.append((cand, " ".join(buf)))
    return out


def _rank(passages, terms):
    """BM25 with IDF taken from the candidate pool.

    Pool-local IDF is the point: for "ollama num_ctx", every candidate says
    "ollama" so it scores ~0, while "num_ctx" appears in few and dominates.
    No external corpus, no model, no network.
    """
    if not passages or not terms:
        return [], {}, 0

    docs = [_tokenize(text) for _, text in passages]
    n_docs = len(docs)
    avg_len = sum(len(d) for d in docs) / max(1, n_docs)

    doc_freq = {}
    for doc in docs:
        for term in set(doc):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    k1, b = 1.5, 0.75
    scored = []
    for (cand, text), doc in zip(passages, docs):
        freq = {}
        for term in doc:
            freq[term] = freq.get(term, 0) + 1
        score = 0.0
        for term in terms:
            n_t = doc_freq.get(term, 0)
            if not n_t:
                continue
            idf = math.log(1 + (n_docs - n_t + 0.5) / (n_t + 0.5))
            f = freq.get(term, 0)
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * len(doc) / max(1, avg_len)))
        if score > 0:
            scored.append((score, cand, text))
    scored.sort(key=lambda x: -x[0])
    return scored, doc_freq, n_docs


def _covers(text, must):
    """Every required term must actually appear in the passage."""
    low = text.lower()
    return all(t in low for t in must)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def _attempt(query, terms, must_ids):
    """One retrieval round: gather, read, rank, and decide what a hit must contain."""
    candidates, notes = _gather(query)
    if not candidates:
        return [], [], notes, []
    fetched = _fetch_all(candidates)
    scored, doc_freq, n_docs = _rank(_passages(fetched), terms)
    must = _discriminative(terms, must_ids, doc_freq, n_docs)
    if must:
        scored = [row for row in scored if _covers(row[2], must)]
    return scored, fetched, notes, must


def search_web(query, max_results=None):
    query = (query or "").strip()
    if not query:
        return "[Error] Empty search query."

    limit = max_results or config.SEARCH_MAX_RESULTS
    started = time.time()
    terms, must_ids = _query_terms(query)

    try:
        scored, fetched, notes, must = _attempt(query, terms, must_ids)

        # A natural-language phrase can bury the terms that matter. If the first
        # round found nothing that mentions them, ask again with just those.
        short = distill(query)
        if not scored and short and short != query:
            retry_scored, retry_fetched, retry_notes, retry_must = _attempt(short, terms, must_ids)
            notes = notes + [f"retry({short})"] + retry_notes
            if retry_scored:
                scored, fetched, must = retry_scored, retry_fetched, retry_must
            else:
                fetched = fetched + retry_fetched
    except Exception as e:
        return f"[Error] Search failed: {type(e).__name__}: {e}"

    if not fetched:
        return (f"[Search] No results for '{query}'. Sources tried: {', '.join(notes) or 'none'}.\n"
                "Every source returned nothing - treat this as no information, not as a negative answer.")

    # The relevance floor. Returning the closest junk is what made the old
    # search wrong; saying nothing was found is recoverable.
    if not scored:
        urls = "\n".join(f"  - {c['url']}" for c in fetched[:6])
        wanted = ", ".join(repr(t) for t in must) if must else "the query terms"
        return (f"[Search] No relevant results for '{query}'.\n"
                f"Sources: {', '.join(notes)}; read {sum(1 for c in fetched if c['text'])} pages.\n"
                f"None of them mention {wanted}. Pages checked:\n{urls}\n\n"
                "Do NOT answer from these pages - they cover the general topic, not the specific "
                "term asked about. Either retry with different wording, use get_url on official "
                "documentation, or tell the user the search found nothing.")

    blocks, used_urls, budget = [], set(), config.SEARCH_RESULT_CHARS
    for score, cand, text in scored:
        if cand["url"] in used_urls:
            continue
        used_urls.add(cand["url"])
        block = (f"{len(blocks) + 1}. {cand['title'] or '(no title)'}  [{cand['source']}, score {score:.1f}]\n"
                 f"   URL: {cand['url']}\n"
                 f"   {text[:config.SEARCH_PASSAGE_CHARS]}")
        if budget - len(block) < 0:
            break
        budget -= len(block)
        blocks.append(block)
        if len(blocks) >= limit:
            break

    header = (f"[Search] '{query}' - {len(blocks)} relevant passage(s) "
              f"({', '.join(notes)}), {time.time() - started:.1f}s"
              + (f", matched on {', '.join(must)}" if must else "") + ".\n"
              "Passages are page text ranked locally, not search-engine snippets. "
              "Answer only from what appears below and cite the URL you used.")
    return header + "\n\n" + "\n\n".join(blocks)


if __name__ == "__main__":
    print("This file can not run directly.")
