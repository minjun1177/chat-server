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
runs against public no-auth endpoints; point config.SEARXNG_URL
at a local SearXNG and the pipeline becomes fully self-hosted for candidate
generation too. The keyless engines are scraped, so they rate-limit a bot that
searches often - that, not ranking, is what an unconfigured setup loses first.
"""

import concurrent.futures as futures
import html
import json
import math
import re
import threading
import time
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlsplit, urlunsplit

import lxml.etree
import lxml.html
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
# snake_case or camelCase; the lookbehind stops "TypeScript" yielding "ypeScript"
_IDENTIFIER = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+|[a-z]+[A-Z][A-Za-z0-9]*)")
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


# letters and digits in one token: call signs, model and part numbers, error codes
_CODE = re.compile(r"(?<![A-Za-z0-9])(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{3,}(?![A-Za-z0-9])")


def _query_terms(query):
    """(scoring terms, identifier and code terms the answer must contain)."""
    terms = [t for t in _tokenize(distill(query)) if t not in _STOP and len(t) > 1]
    must = [t.lower() for t in _IDENTIFIER.findall(query or "") + _CODE.findall(query or "")]
    return terms, list(dict.fromkeys(must))


def _discriminative(terms, must_ids, doc_freq, n_docs):
    """Terms a genuine hit has to contain: only identifiers and codes.

    This used to also require the single rarest plain word in the pool. That
    word is often incidental to how the answer is phrased: for "discord.py
    external scheduled event required arguments" it picked `arguments`, which
    the documentation calls "Parameters", and every passage that held the answer
    was filtered out. Plain words are weighed softly by `_apply_coverage`
    instead; on the evaluation pool that changed no other result.
    """
    return list(dict.fromkeys(must_ids))


def _is_code_query(query):
    if _IDENTIFIER.search(query or ""):
        return True
    return bool(re.search(r"[`(){}\[\]]|\.\w+\(|error|exception|traceback", query or "", re.I))


def _has_hangul(text):
    return bool(_HANGUL.search(text or ""))


_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_\-]+)""", re.I)


def decode_body(resp):
    """A response body as text, in the encoding the page is actually written in.

    `requests` falls back to ISO-8859-1 for any text/* response whose header
    names no charset, which turns every Korean page served as plain
    "text/html" (Google Patents, many blogs) into mojibake that neither the
    ranker nor the model can read. The header wins when it has a charset;
    otherwise the page's own <meta charset>, then UTF-8 if the bytes are valid
    UTF-8, then a statistical guess. Shared with the get_url tool.
    """
    if "charset=" not in resp.headers.get("Content-Type", "").lower():
        found = _META_CHARSET.search(resp.content[:4096])
        encoding = found.group(1).decode("ascii") if found else ""
        if not encoding:
            try:
                resp.content.decode("utf-8")
                encoding = "utf-8"
            except UnicodeDecodeError:
                encoding = resp.apparent_encoding or "utf-8"
        resp.encoding = encoding
    try:
        return resp.text
    except LookupError:                         # a <meta charset> naming no real codec
        resp.encoding = "utf-8"
        return resp.text


_XML_DECLARATION = re.compile(r"^\s*<\?xml[^>]*\?>")
_NOT_CONTENT = ("//script|//style|//nav|//footer|//header|//noscript|//iframe|//svg"
                "|//button|//select|//textarea")
_LARGE_HTML = 1_000_000                 # characters; parsing one costs tens of MB
_large_parses = threading.BoundedSemaphore(2)


def strip_html(raw):
    """HTML to readable text. Shared with the get_url tool.

    lxml rather than BeautifulSoup's html.parser: on the 4.6 MB discord.py API
    reference it is 6.5x faster and peaks 34% lower, with word-for-word the
    same text on 34 ordinary pages. Parsing several such pages at once is what
    ran an 8 GB machine out of memory, so large documents also take turns.

    Form *controls* go, not <form> itself: ASP.NET-style and many older sites
    wrap the whole page in one form, and dropping it threw the content away
    (qrz.com kept 928 of its 7,985 characters).
    """
    raw = _XML_DECLARATION.sub("", raw or "")   # lxml refuses str input that declares an encoding
    if not raw.strip():
        return ""
    if len(raw) < _LARGE_HTML:
        return _html_text(raw)
    with _large_parses:
        return _html_text(raw)


_BLOCK_TAGS = frozenset((
    "address", "article", "aside", "blockquote", "body", "br", "caption", "dd", "details", "dialog",
    "div", "dl", "dt", "fieldset", "figcaption", "figure", "form", "h1", "h2", "h3", "h4", "h5", "h6",
    "head", "hr", "html", "li", "main", "ol", "p", "pre", "section", "summary", "table", "tbody", "td",
    "tfoot", "th", "thead", "title", "tr", "ul",
))
_SPACES = re.compile(r"\s+")


def _html_text(raw):
    """Text with line breaks where the layout has them - at block elements only.

    Breaking at every tag split anything written with inline markup: Sphinx
    renders `asyncio.Semaphore(value=1)` as separate spans, so the page never
    appeared to contain "asyncio.semaphore" and lost to blogs that merely
    mention it. Walks the tree iteratively; a deep DOM must not hit the
    recursion limit.
    """
    try:
        doc = lxml.html.document_fromstring(raw)
    except (lxml.etree.ParserError, ValueError):
        return ""
    for element in doc.xpath(_NOT_CONTENT):
        element.drop_tree()                     # keeps the text that follows the element

    parts, pre_depth = [], 0
    for event, element in lxml.etree.iterwalk(doc, events=("start", "end", "comment", "pi")):
        tag = element.tag if isinstance(element.tag, str) else ""   # comments have a function here
        if event in ("comment", "pi"):
            # never a start/end pair, but the text after one is still page text
            if element.tail:
                parts.append(element.tail if pre_depth else _SPACES.sub(" ", element.tail))
        elif event == "start":
            if tag == "pre":
                pre_depth += 1
            if tag in _BLOCK_TAGS:
                parts.append("\n")
            if tag and element.text:
                parts.append(element.text if pre_depth else _SPACES.sub(" ", element.text))
        else:
            if tag in _BLOCK_TAGS:
                parts.append("\n")
            if tag == "pre":
                pre_depth -= 1
            if element.tail:
                parts.append(element.tail if pre_depth else _SPACES.sub(" ", element.tail))
    lines = (line.strip() for line in "".join(parts).split("\n"))
    return "\n".join(line for line in lines if line)


THIN_PAGE_CHARS = 500           # below this, the HTML probably is not where the content is
_MAX_FRAMES = 2


def _content_frames(soup, page_url):
    """Same-site frames that may hold the page's real content.

    Blog platforms that load a post into an iframe (Naver blog, Naver cafe) and
    old framesets keep it on their own host. Cross-site frames are ads,
    trackers and video embeds, so they are not followed.
    """
    host = urlparse(page_url).hostname
    frames = []
    for tag in soup.find_all(["iframe", "frame"]):
        src = urljoin(page_url, (tag.get("src") or "").strip())
        parsed = urlparse(src)
        if parsed.scheme in ("http", "https") and parsed.hostname == host and src != page_url:
            frames.append(src)
    return list(dict.fromkeys(frames))[:_MAX_FRAMES]


def _page_metadata(soup):
    """What a JavaScript-rendered page still says about itself in its HTML."""
    found = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except ValueError:
            continue
        for item in data if isinstance(data, list) else [data]:
            if isinstance(item, dict):
                found += [item.get("headline"), item.get("articleBody"), item.get("description")]
    for attrs in ({"property": "og:title"}, {"property": "og:description"}, {"name": "description"}):
        tag = soup.find("meta", attrs=attrs)
        if tag:
            found.append(tag.get("content"))
    texts = [t.strip() for t in found if isinstance(t, str) and t.strip()]
    return "\n".join(dict.fromkeys(texts))


def read_page(url, timeout, headers=None):
    """A URL's readable text, including pages whose HTML is only a shell.

    When the stripped page is thin, the content is usually somewhere the plain
    HTML does not show: a same-site frame, or metadata a JavaScript app ships
    for crawlers. Both are tried generically - no per-site rules. Raises for
    HTTP errors like requests does. Shared with the get_url tool.
    """
    headers = headers or UA
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    body = decode_body(resp)
    if "html" not in resp.headers.get("Content-Type", "").lower():
        return body
    text = strip_html(body)
    if len(text) >= THIN_PAGE_CHARS:
        return text

    soup = BeautifulSoup(body, "html.parser")
    for frame_url in _content_frames(soup, resp.url or url):
        try:
            inner = requests.get(frame_url, headers=headers, timeout=timeout)
            inner.raise_for_status()
            inner_text = strip_html(decode_body(inner))
        except Exception:                       # noqa: BLE001 - the outer page still counts
            continue
        if len(inner_text) > len(text):
            text = inner_text
    if len(text) >= THIN_PAGE_CHARS:
        return text

    meta = _page_metadata(soup)
    parts = [meta] if meta else []
    if text and text not in meta:
        parts.append(text)
    return "\n".join(parts)


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
    # SearXNG already merged several engines into ~30 ranked results for this
    # one request; cutting to SEARCH_CANDIDATES dropped official pages it ranked
    # 10th-25th (docs.python.org, PEPs) for no saving at all.
    for item in resp.json().get("results", [])[:config.SEARCH_CANDIDATES * 3]:
        if item.get("url"):
            out.append({
                "url": item["url"], "title": item.get("title", ""),
                "snippet": item.get("content", ""), "text": "", "source": "searxng",
            })
    return out


def _src_ddg(query):
    """General web. Broad coverage, weak precision - the pool, not the answer.

    `ddgs` has to come first. The old `duckduckgo_search` package (8.x) quietly
    hard-codes Bing as its only backend and ignores `region`, and Bing answers a
    cookie-less scraper with pages unrelated to the query - a multi-word search
    came back about its first word at best, so it looked as if everything after
    the first space had been dropped. `ddgs` rotates across several engines.
    """
    try:
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException
    except ImportError:
        from duckduckgo_search import DDGS  # legacy fallback, Bing-only results
        from duckduckgo_search.exceptions import DuckDuckGoSearchException as DDGSException
    region = "kr-kr" if _has_hangul(query) else "us-en"
    try:
        results = DDGS().text(query, region=region, safesearch="moderate",
                              max_results=config.SEARCH_CANDIDATES, backend="auto")
    except DDGSException as error:
        # ddgs reports an empty page as an exception; that is a result, not a failure.
        if "no results" in str(error).lower():
            return []
        raise
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
    wanted = set(_tokenize(distill(query)))
    out = []
    for item in resp.json().get("query", {}).get("search", []):
        title = item.get("title", "")
        # srsearch always returns something, and on Korean Wikipedia the fuzzy
        # match is often an unrelated long article ("김일성" for a mushroom
        # festival) that costs a fetch and skews the pool's term statistics.
        # An article whose title shares no term with the query is not the one.
        if not wanted & set(_tokenize(title)):
            continue
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


_cache: dict = {}                       # key -> (stored_at, value)
_cache_lock = threading.Lock()


def _cache_get(key):
    ttl = config.SEARCH_CACHE_TTL
    if ttl <= 0:
        return None
    with _cache_lock:
        hit = _cache.get(key)
    return hit[1] if hit and time.time() - hit[0] < ttl else None


_CACHE_MAX_CHARS = 20_000_000          # page text held across all cached pages


def _cache_put(key, value):
    """Remember a result. Only non-empty ones - an empty answer is often a rate
    limit, and caching it would keep the search broken after the limit lifts.

    Page text is kept whole (a long page is focused per query later), so the
    cache is bounded by total characters, oldest entries leaving first.
    """
    ttl = config.SEARCH_CACHE_TTL
    if ttl <= 0 or not value:
        return
    now = time.time()
    with _cache_lock:
        for stale in [k for k, (at, _) in _cache.items() if now - at >= ttl]:
            del _cache[stale]
        _cache[key] = (now, value)
        size = sum(len(v) for _, v in _cache.values() if isinstance(v, str))
        for old in sorted(_cache, key=lambda k: _cache[k][0]):
            if size <= _CACHE_MAX_CHARS:
                break
            if isinstance(_cache[old][1], str) and old != key:
                size -= len(_cache.pop(old)[1])


def _gather(query):
    """Candidates for a query, reused for a while.

    The model often searches the same thing again a turn or two later, and
    every repeat is another request to engines that throttle scrapers.
    """
    cached = _cache_get(("gather", query))
    if cached is not None:
        candidates, notes = cached
        return [dict(c) for c in candidates], notes + ["cached"]
    candidates, notes = _gather_live(query)
    _cache_put(("gather", query), ([dict(c) for c in candidates], notes) if candidates else None)
    return candidates, notes


_NOISE_PARAMS = re.compile(r"^(utm_\w*|highlight|fbclid|gclid|ref_src|spm)$", re.I)


def _url_key(url):
    """The page a URL names, for de-duplication.

    Engines return the same page under several spellings - Sphinx's
    `?highlight=` search terms, tracking parameters, a #fragment - and each copy
    took a slot in the fetch budget (the 788,000-character discord.py reference
    was fetched twice for one query).
    """
    parts = urlsplit(url or "")
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                       if not _NOISE_PARAMS.match(k)])
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), query, ""))


def _gather_live(query):
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

    # Interleave the sources by their own rank. Concatenating in completion
    # order let whichever source answered first take the whole fetch budget.
    by_source = {}
    for cand in candidates:
        by_source.setdefault(cand["source"], []).append(cand)
    queues = list(by_source.values())
    interleaved = []
    for depth in range(max((len(q) for q in queues), default=0)):
        interleaved.extend(q[depth] for q in queues if depth < len(q))

    seen, unique = set(), []
    for cand in interleaved:
        key = _url_key(cand["url"])
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
    cached = _cache_get(("page", cand["url"]))
    if cached is not None:
        cand["text"] = cached
        return cand
    try:
        cand["text"] = read_page(cand["url"], config.SEARCH_FETCH_TIMEOUT)[:MAX_PAGE_TEXT]
    except Exception:
        cand["text"] = ""
    _cache_put(("page", cand["url"]), cand["text"])
    return cand


MAX_PAGE_TEXT = 2_000_000               # characters read from one page before focusing


def _focus(text, query):
    """Cut a long page down to SEARCH_PAGE_CHARS by keeping what the query is about.

    Taking the first N characters lost exactly the pages that hold the answer
    deep inside: on the discord.py API reference `create_scheduled_event` sits
    at character 320,144 of 805,846. The page is cut into passage-sized chunks,
    each scored by which query words it contains (rarer across the page counts
    more), and the best chunks are kept in their original order - plus the
    first one, which usually names the page.
    """
    limit = config.SEARCH_PAGE_CHARS
    if len(text) <= limit:
        return text
    words = _query_words(query) or [t for t in _tokenize(query) if len(t) > 1]
    chunks, buf, size = [], [], 0
    for line in text.split("\n"):
        buf.append(line)
        size += len(line) + 1
        if size >= config.SEARCH_PASSAGE_CHARS:
            chunks.append("\n".join(buf))
            buf, size = [], 0
    if buf:
        chunks.append("\n".join(buf))

    lows = [chunk.lower() for chunk in chunks]
    present = {w: [_has_word(w, low) for low in lows] for w in words}
    weight = {w: math.log(1 + len(chunks) / (1 + sum(hits))) for w, hits in present.items()}
    scores = [sum(weight[w] for w in words if present[w][i]) for i in range(len(chunks))]

    keep, used = {0}, len(chunks[0])
    for i in sorted(range(1, len(chunks)), key=lambda i: -scores[i]):
        if scores[i] <= 0 or used + len(chunks[i]) > limit:
            break
        keep.add(i)
        used += len(chunks[i]) + 1
    return "\n".join(chunks[i] for i in sorted(keep))


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


COVERAGE_FLOOR = 0.5


def _query_words(query):
    """The words the user typed that carry meaning, whole - not BM25's bigrams."""
    words = []
    for raw in distill(query).split():
        word = raw.strip("\"'()[]{}<>,.?!:;").lower()
        if len(word) > 1 and word not in _STOP:
            words.append(word)
    return list(dict.fromkeys(words))


def _has_word(word, low):
    if _HANGUL.search(word):
        return word in low                  # particles attach after: 구리시의, 구리시에서
    return re.search(r"(?<![a-z0-9])" + re.escape(word), low) is not None


def _apply_coverage(scored, pages, query):
    """Demote, and past a floor drop, pages that match only part of the query.

    BM25 over Hangul bigrams cannot tell 구리시 (the city) from 구리 (copper), so
    for "구리시 진주 비취 매입 업체" copper-scrap dealers in 진주 scored well on the
    common words alone. Coverage is counted per page on whole query words, and
    each word is weighted by how few candidate pages carry it: missing the rare
    word the query is really about costs far more than missing "업체". A word no
    page carries at all gets the lowest weight - it cannot tell candidates apart.
    """
    words = _query_words(query)
    if len(words) < 2 or not scored:
        return scored
    lows = {c["url"]: (c["title"] + " " + (c["text"] or c["snippet"])).lower() for c in pages}
    n_pages = len(lows)
    page_freq = {w: sum(_has_word(w, low) for low in lows.values()) for w in words}
    weight = {w: math.log(1 + (n_pages + 1) / (f + 0.5)) for w, f in page_freq.items() if f}
    floor_weight = min(weight.values()) if weight else 1.0
    for w in words:
        weight.setdefault(w, floor_weight)
    total = sum(weight.values())
    coverage = {url: sum(weight[w] for w in words if _has_word(w, low)) / total
                for url, low in lows.items()}
    kept = [(score * coverage.get(cand["url"], 0) ** 2, cand, text)
            for score, cand, text in scored
            if coverage.get(cand["url"], 0) >= COVERAGE_FLOOR]
    kept.sort(key=lambda row: -row[0])
    return kept


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

# Measured on a 12-query answer set: one passage per page answered 11, a second
# passage at >= 0.6 of the page's best answered 12, and at 0.4 the extra text
# crowded out the fifth page and it fell back to 11.
_KO_GROUP = r"\d{1,3}(?:,\d{3})+|\d{1,4}"
_KO_COUNTER = r"원|명|개|건|달러|위안|엔|유로|가구|대|회|표|톤|주|곳|채"
_KO_NUMBER = re.compile(
    rf"(?<![\d,.])"
    rf"(?:(?P<jo>{_KO_GROUP})\s*조\s*)?"
    rf"(?:(?P<eok>{_KO_GROUP})\s*억\s*)?"
    rf"(?:(?P<man>{_KO_GROUP})\s*만)?"
    # possessive (?+): once the number and its counter are read, backtracking
    # must not drop them to annotate a shorter, wrong reading
    rf"(?:\s*(?P<rest>{_KO_GROUP})(?=\s*(?:{_KO_COUNTER})))?+"
    rf"(?P<unit>\s*(?:{_KO_COUNTER}))?+"
    rf"(?!\s*[\d천백(])"
)


def normalize_korean_numbers(text):
    """Write Korean mixed-unit amounts as plain figures: 1만 320원 -> 10,320원.

    A 8B model read the official "시간급 1만 320원" as 13,200원 and answered the
    minimum wage wrong from a correct source. Replaying that exact search result
    six times each: as written, 0 correct; with the figure appended in
    brackets, 3; with the amount replaced, 6. Only exact cases are touched: an
    amount with 천/백 inside or a number right after it is left as it is, and a
    trailing number joins only when a counter like 원 or 명 follows it.
    """
    def replace(match):
        groups = [match.group(g) for g in ("jo", "eok", "man")]
        if not any(groups):
            return match.group(0)
        value = 0
        for group, scale in zip(groups, (10 ** 12, 10 ** 8, 10 ** 4)):
            if group:
                value += int(group.replace(",", "")) * scale
        rest = match.group("rest")
        if rest:
            rest_value = int(rest.replace(",", ""))
            if rest_value >= 10 ** 4:
                return match.group(0)
            value += rest_value
        unit = (match.group("unit") or "").strip()
        lead = match.group(0)[:len(match.group(0)) - len(match.group(0).lstrip())]
        return f"{lead}{value:,}{unit}"
    return _KO_NUMBER.sub(replace, text)


PASSAGES_PER_URL = 2
SECOND_PASSAGE_RATIO = 0.6


def _render_blocks(scored, limit):
    """One block per page, best pages first, within the result budget.

    A page may contribute a second passage when it scores close to its best one
    (SECOND_PASSAGE_RATIO of it): on a long reference page the passage that
    wins on the query's words is not always the one holding the answer.
    """
    by_url = {}
    for score, cand, text in scored:
        by_url.setdefault(cand["url"], []).append((score, cand, text))
    blocks, budget = [], config.SEARCH_RESULT_CHARS
    for rows in by_url.values():                # insertion order = best score first
        best_score, cand, text = rows[0]
        passages = [text[:config.SEARCH_PASSAGE_CHARS]]
        for score, _, extra in rows[1:PASSAGES_PER_URL]:
            if score >= best_score * SECOND_PASSAGE_RATIO:
                passages.append(extra[:config.SEARCH_PASSAGE_CHARS])
        passages = [normalize_korean_numbers(p) if _has_hangul(p) else p for p in passages]
        title = normalize_korean_numbers(cand["title"] or "(no title)")
        block = (f"{len(blocks) + 1}. {title}  [{cand['source']}, score {best_score:.1f}]\n"
                 f"   URL: {cand['url']}\n"
                 "   " + "\n   ...\n   ".join(passages))
        if budget - len(block) < 0:
            if len(passages) == 1:
                break
            block = block[:block.index("\n   ...\n")]       # drop the extra passage before the page
            if budget - len(block) < 0:
                break
        budget -= len(block)
        blocks.append(block)
        if len(blocks) >= limit:
            break
    return blocks


def _attempt(query, terms, must_ids):
    """One retrieval round: gather, read, rank, and decide what a hit must contain."""
    candidates, notes = _gather(query)
    if not candidates:
        return [], [], notes, []
    fetched = _fetch_all(candidates)
    for cand in fetched:
        cand["text"] = _focus(cand["text"], query)
    scored, doc_freq, n_docs = _rank(_passages(fetched), terms)
    must = _discriminative(terms, must_ids, doc_freq, n_docs)
    if must:
        scored = [row for row in scored if _covers(row[2], must)]
    scored = _apply_coverage(scored, fetched, query)
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
                "Every source returned nothing - treat this as no information, not as a negative answer.\n"
                "The free engines throttle repeated searches, so an empty result right after other "
                "searches is often a rate limit: do not fire several rephrasings in a row. Try at most "
                "one clearly different query, or tell the user search is unavailable right now.")

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

    blocks = _render_blocks(scored, limit)
    header = (f"[Search] '{query}' - {len(blocks)} relevant passage(s) "
              f"({', '.join(notes)}), {time.time() - started:.1f}s"
              + (f", matched on {', '.join(must)}" if must else "") + ".\n"
              "Passages are page text ranked locally, not search-engine snippets. "
              "Answer only from what appears below and cite the URL you used.")
    return header + "\n\n" + "\n\n".join(blocks)


if __name__ == "__main__":
    print("This file can not run directly.")
