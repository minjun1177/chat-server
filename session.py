import os
import re
import json
import datetime
from contextvars import ContextVar

import config
import logs
import providers
from config import ttlp


# Where this conversation's memory and transcripts live. The terminal uses the
# defaults (memory.json / sessions/); a Discord conversation sets its own so
# one person's memories and history never land in another's file.
_store: ContextVar[tuple] = ContextVar("session_store", default=(None, None))


def use_store(memory_file: str = "", session_dir: str = ""):
    """Point memory and transcripts somewhere else for this task. Returns a token."""
    return _store.set((memory_file or None, session_dir or None))


def reset_store(token) -> None:
    _store.reset(token)


def memory_path() -> str:
    return _store.get()[0] or config.MEMORY_FILE


def session_dir() -> str:
    return _store.get()[1] or config.SESSION_DIR


def load_memory() -> dict:
    if os.path.exists(memory_path()):
        try:
            with open(memory_path(), "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def save_memory(memory: dict) -> None:
    parent = os.path.dirname(memory_path())
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(memory_path(), "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def handle_write_memory(memory_id: str, content: str) -> str:
    if not memory_id:
        return "[Error] Memory ID is required."
    memory = load_memory()
    memory[memory_id] = {
        "content": content,
        "created_at": datetime.datetime.now().isoformat()
    }
    save_memory(memory)
    return f"[Success] Memory saved: '{memory_id}'"

def handle_get_memory_list() -> str:
    memory = load_memory()
    if not memory:
        return "[Memory] No memories stored."
    lines = []
    for i, (mid, data) in enumerate(memory.items(), 1):
        created = data.get("created_at", "unknown")
        preview = data.get("content", "")[:50]
        lines.append(f"{i}. {mid} ({created}) - {preview}")
    return "\n".join(lines)

def handle_read_memory(memory_id: str) -> str:
    if not memory_id:
        return "[Error] Memory ID is required."
    memory = load_memory()
    if memory_id not in memory:
        return f"[Error] Memory '{memory_id}' not found."
    data = memory[memory_id]
    return f"[Memory: {memory_id}]\nContent: {data.get('content', '')}\nCreated: {data.get('created_at', 'unknown')}"

def handle_delete_memory(memory_id: str) -> str:
    if not memory_id:
        return "[Error] Memory ID is required."
    memory = load_memory()
    if memory_id not in memory:
        return f"[Error] Memory '{memory_id}' not found."
    del memory[memory_id]
    save_memory(memory)
    return f"[Success] Memory deleted: '{memory_id}'"

def handle_edit_memory(memory_id: str, new_content: str) -> str:
    if not memory_id:
        return "[Error] Memory ID is required."
    memory = load_memory()
    if memory_id not in memory:
        return f"[Error] Memory '{memory_id}' not found."
    memory[memory_id]["content"] = new_content
    save_memory(memory)
    return f"[Success] Memory edited: '{memory_id}'"


_FS_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_WINDOWS_RESERVED = ({"CON", "PRN", "AUX", "NUL", "CLOCK$"}
                     | {f"COM{i}" for i in range(1, 10)}
                     | {f"LPT{i}" for i in range(1, 10)})


def _timestamp_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def clean_title(raw: str) -> str:
    """Squeeze a model reply (or a user's /title text) into one usable title line."""
    text = re.sub(r'<think>.*?</think>', '', raw or '', flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    title = lines[-1]
    for _ in range(2):  # "**Title: x**" needs the markers peeled on both sides of the prefix
        title = re.sub(r'^[\s\-*•#]+', '', title)
        title = re.sub(r'[\s*#]+$', '', title)
        title = re.sub(r'^(?:title|제목)\s*[:：]\s*', '', title, flags=re.IGNORECASE)
    title = title.strip('`').strip('"\'“”‘’').strip()
    title = re.sub(r'\s+', ' ', title).strip(' .。!?')
    return title[:config.SESSION_TITLE_MAX_LEN].strip()


def slugify_title(title: str) -> str:
    """Turn a human title into a filesystem-safe session id ("" if nothing survives)."""
    slug = _FS_UNSAFE.sub(" ", title or "")
    slug = re.sub(r'[^\w\s-]', '', slug, flags=re.UNICODE)
    slug = re.sub(r'\s+', '-', slug.strip())
    slug = re.sub(r'-{2,}', '-', slug).strip('-._')
    slug = slug[:config.SESSION_SLUG_MAX_LEN].strip('-._').lower()
    if not slug:
        return ""
    if slug.split('.')[0].upper() in _WINDOWS_RESERVED:
        slug = f"{slug}-session"
    return slug


def _unique_session_id(base: str, current_id: str = None) -> str:
    """Append a counter until the id stops colliding with another session file."""
    if not base:
        base = _timestamp_id()
    candidate, n = base, 2
    while True:
        path = os.path.join(session_dir(), f"{candidate}.json")
        if candidate == current_id or not os.path.exists(path):
            return candidate
        candidate = f"{base}-{n}"
        n += 1


def save_session(messages: list[dict], session_id: str) -> str:
    if not config.SAVE_CHAT_HISTORY:
        return session_id
    if not os.path.exists(session_dir()):
        os.makedirs(session_dir())
    if not session_id:
        # An untitled session starts on a timestamp id; the title (from the model
        # or from /title) renames the file as soon as there is one.
        session_id = _unique_session_id(slugify_title(config.SESSION_TITLE))
    filepath = os.path.join(session_dir(), f"{session_id}.json")

    data = {
        "version": 3,
        "title": config.SESSION_TITLE,
        "model": config.MODEL,
        "persona": config.CUSTOM_PERSONA,
        "token_history": config.token_history,
        "messages": config.repair_messages(messages),
        "updated_at": datetime.datetime.now().isoformat()
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logs.error(f"failed to save the session: {e}")
    return session_id


def rename_session(session_id: str, new_title: str) -> str:
    """Retitle a session and move its file to match. Returns the (possibly new) id."""
    new_title = clean_title(new_title)
    if not new_title:
        return session_id

    config.SESSION_TITLE = new_title
    old_path = os.path.join(session_dir(), f"{session_id}.json") if session_id else ""
    if not old_path or not os.path.exists(old_path):
        return session_id

    new_id = _unique_session_id(slugify_title(new_title), current_id=session_id)
    try:
        with open(old_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data["title"] = new_title
            with open(old_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        if new_id != session_id:
            os.replace(old_path, os.path.join(session_dir(), f"{new_id}.json"))
        return new_id
    except Exception as e:
        logs.error(f"failed to rename the session: {e}")
        return session_id


async def generate_session_title(messages: list[dict]) -> str:
    """Ask the model to name the conversation. Returns "" if it cannot."""
    convo = []
    for m in messages:
        if m["role"] == "system":
            continue
        content = re.sub(r'<tool_call>.*?</tool_call>', '', m.get("content", ""), flags=re.DOTALL).strip()
        if not content or content.startswith("[Tool Result"):
            continue
        role = "User" if m["role"] == "user" else "Assistant"
        convo.append(f"[{role}]: {content[:600]}")
        if len(convo) >= 4:
            break
    if not convo:
        return ""

    prompt = ttlp() + "\n\n".join(convo)
    try:
        text = await providers.complete(
            [{"role": "user", "content": prompt}], max_tokens=40)
        return clean_title(text)
    except Exception:
        return ""


def load_session(session_id: str) -> dict | list:
    filepath = os.path.join(session_dir(), f"{session_id}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            pass
    return None


def list_sessions() -> list[tuple[str, str, str]]:
    """Return (session_id, title, meta), newest first."""
    if not os.path.exists(session_dir()):
        return []
    sessions = []
    for f in os.listdir(session_dir()):
        if not f.endswith(".json"):
            continue
        sid = f[:-len(".json")]
        filepath = os.path.join(session_dir(), f)
        title, meta_str, updated = "", "", ""
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, dict) and data.get("version") in (2, 3):
                title = data.get("title", "")
                updated = data.get("updated_at", "")
                m = data.get("model", "unknown")
                msgs = len(data.get("messages", []))
                tokens = sum(t.get("prompt", 0) + t.get("completion", 0) for t in data.get("token_history", []))
                meta_str = f"[{m}] {msgs} msgs, {tokens} tokens"
            elif isinstance(data, list):
                meta_str = f"[legacy] {len(data)} msgs"
        except Exception:
            meta_str = "[error loading info]"
        if not updated:
            try:
                updated = datetime.datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
            except OSError:
                updated = ""
        sessions.append((sid, title, meta_str, updated))
    sessions.sort(key=lambda x: x[3], reverse=True)
    return [(sid, title, meta) for sid, title, meta, _ in sessions]


def find_sessions(query: str) -> list[tuple[str, str, str]]:
    """Look a session up by id or title: exact id, then exact title, then substring."""
    query = (query or "").strip()
    if not query:
        return []
    sessions = list_sessions()
    for entry in sessions:
        if entry[0] == query:
            return [entry]
    q = query.lower()
    exact = [e for e in sessions if e[1].lower() == q]
    if exact:
        return exact
    return [e for e in sessions if q in e[0].lower() or q in e[1].lower()]
