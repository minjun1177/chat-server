"""Model providers: Ollama, Anthropic, OpenAI (and compatible), Google Gemini.

The harness asks for tool calls as `<tool_call>` text rather than through each
vendor's function-calling API, so a provider only has to do one thing: turn a
list of messages into a stream of text. That is why each of these is a few
dozen lines and needs no SDK - the wire formats differ, the job does not.

Every provider yields the same events, whatever it is talking to:

    {"text": "..."}                        a piece of the reply
    {"thinking": "..."}                    reasoning, where the provider sends it
    {"done": True, "prompt_tokens": N, "completion_tokens": M,
     "total_seconds": f, "eval_seconds": f}

Credentials come from the environment first, then `~/.localchat/providers.json`,
which is written with owner-only permissions. Nothing is ever stored in the
project directory - that is a place people commit from.
"""

import asyncio
import json
import os
import threading
import time

import env
from sse import iter_sse

env.load()          # so a key in .env is found by the lookups below


CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".localchat")
CONFIG_PATH = os.path.join(CONFIG_DIR, "providers.json")

HTTP_TIMEOUT = (10, 600)          # (connect, read): a long reply is not a hang


def _requests():
    import requests
    return requests


# ---------------------------------------------------------------------------
# message shaping
# ---------------------------------------------------------------------------

def split_system(messages: list) -> tuple:
    """Separate the system prompt from the conversation.

    Ollama and OpenAI take it as a message; Anthropic and Gemini take it as its
    own field.
    """
    system = "\n\n".join(m.get("content", "") for m in messages
                         if m.get("role") == "system").strip()
    rest = [m for m in messages if m.get("role") != "system"]
    return system, rest


def merge_runs(messages: list) -> list:
    """Collapse consecutive same-role messages into one.

    The harness appends every tool result as its own user message, so a turn
    routinely ends up with several in a row. Some APIs reject that.
    """
    merged = []
    for message in messages:
        content = message.get("content", "")
        if merged and merged[-1]["role"] == message.get("role"):
            merged[-1]["content"] += "\n\n" + content
        else:
            merged.append({"role": message.get("role", "user"), "content": content})
    return merged


def _as_stream(make_chunks):
    """Turn a blocking generator into an async one, off the event loop.

    Without this the spinner would freeze and the tokens-per-second counter
    would stop while the reply streams in.
    """
    async def stream():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        done = object()

        def worker():
            try:
                for item in make_chunks():
                    loop.call_soon_threadsafe(queue.put_nowait, item)
            except BaseException as error:               # noqa: BLE001 - re-raised below
                loop.call_soon_threadsafe(queue.put_nowait, error)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, done)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = await queue.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    return stream()


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------

class Provider:
    name = ""
    label = ""
    key_env: tuple = ()
    key_help = ""
    default_base_url = ""
    needs_key = True

    def __init__(self, settings: dict | None = None):
        self.settings = settings or {}

    # -- configuration -----------------------------------------------------

    @property
    def api_key(self) -> str:
        for variable in self.key_env:
            value = os.environ.get(variable)
            if value:
                return value.strip()
        return str(self.settings.get("api_key") or "").strip()

    @property
    def base_url(self) -> str:
        return str(self.settings.get("base_url") or self.default_base_url).rstrip("/")

    @property
    def model(self) -> str:
        return str(self.settings.get("model") or "")

    @property
    def key_source(self) -> str:
        for variable in self.key_env:
            if os.environ.get(variable):
                return f"${variable}"
        return "saved" if self.settings.get("api_key") else ""

    def ready(self) -> str:
        """"" when this provider can be used, otherwise what is missing."""
        if self.needs_key and not self.api_key:
            return f"no API key ({' or '.join(self.key_env)})"
        if not self.model:
            return "no model chosen"
        return ""

    # -- what a subclass implements ---------------------------------------

    def list_models(self) -> list:
        """[{"name": ..., "detail": ...}], newest/most useful first."""
        raise NotImplementedError

    def stream(self, messages: list, max_tokens: int | None = None):
        raise NotImplementedError

    # -- shared helpers ----------------------------------------------------

    def _get_json(self, url: str, headers: dict) -> dict:
        response = _requests().get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()

    def _post_sse(self, url: str, headers: dict, payload: dict):
        response = _requests().post(url, headers=headers,
                                    data=json.dumps(payload).encode("utf-8"),
                                    stream=True, timeout=HTTP_TIMEOUT)
        if response.status_code >= 400:
            detail = (response.text or "")[:400].strip()
            raise RuntimeError(f"{self.label} returned HTTP {response.status_code}: {detail}")
        return response


class OllamaProvider(Provider):
    name = "ollama"
    label = "Ollama"
    needs_key = False
    key_help = "Runs locally - no API key needed."

    @property
    def model(self) -> str:
        # Someone who never runs /connect keeps the model from config.py.
        import config
        return str(self.settings.get("model") or config.MODEL or "")

    @property
    def host(self) -> str:
        return str(self.settings.get("base_url") or "").rstrip("/")

    def _client(self):
        import ollama
        return ollama.AsyncClient(host=self.host) if self.host else ollama.AsyncClient()

    def list_models(self) -> list:
        import ollama
        listing = ollama.list() if not self.host else ollama.Client(host=self.host).list()
        entries = listing.get("models", []) if isinstance(listing, dict) \
            else getattr(listing, "models", [])
        models = []
        for entry in entries:
            name = entry.get("model", entry.get("name")) if isinstance(entry, dict) \
                else getattr(entry, "model", None)
            size = entry.get("size", 0) if isinstance(entry, dict) else getattr(entry, "size", 0)
            if not name:
                continue
            detail = f"{size / 1024 ** 3:.1f}GB" if size >= 1024 ** 3 else \
                (f"{size / 1024 ** 2:.0f}MB" if size else "")
            models.append({"name": name, "detail": detail})
        return models

    async def stream(self, messages: list, max_tokens: int | None = None):
        import config
        options = {"num_ctx": config.NUM_CTX,
                   "num_predict": max_tokens or config.NUM_PREDICT}
        response = await self._client().chat(model=self.model, messages=messages,
                                             stream=True, options=options)
        async for chunk in response:
            message = chunk.get("message") or {}
            text = message.get("content", "") or ""
            thinking = message.get("thinking", "") or ""
            if thinking:
                yield {"thinking": thinking}
            if text:
                yield {"text": text}
            if chunk.get("done"):
                yield {"done": True,
                       "prompt_tokens": chunk.get("prompt_eval_count", 0) or 0,
                       "completion_tokens": chunk.get("eval_count", 0) or 0,
                       "total_seconds": (chunk.get("total_duration", 0) or 0) / 1e9,
                       "eval_seconds": (chunk.get("eval_duration", 0) or 0) / 1e9}


class AnthropicProvider(Provider):
    name = "anthropic"
    label = "Anthropic"
    key_env = ("ANTHROPIC_API_KEY",)
    key_help = "console.anthropic.com -> API keys"
    default_base_url = "https://api.anthropic.com"
    api_version = "2023-06-01"

    def _headers(self) -> dict:
        return {"x-api-key": self.api_key,
                "anthropic-version": self.api_version,
                "content-type": "application/json"}

    def list_models(self) -> list:
        data = self._get_json(f"{self.base_url}/v1/models?limit=100", self._headers())
        return [{"name": item["id"], "detail": item.get("display_name", "")}
                for item in data.get("data", []) if item.get("id")]

    def stream(self, messages: list, max_tokens: int | None = None):
        import config
        system, conversation = split_system(messages)
        payload = {
            "model": self.model,
            # Anthropic requires max_tokens; it is a cap, not a target.
            "max_tokens": max_tokens or config.NUM_PREDICT,
            "messages": merge_runs(conversation) or [{"role": "user", "content": "."}],
            "stream": True,
        }
        if system:
            payload["system"] = system

        def chunks():
            response = self._post_sse(f"{self.base_url}/v1/messages",
                                      self._headers(), payload)
            prompt_tokens = completion_tokens = 0
            started = time.time()
            for _event, data in iter_sse(response):
                if not data.strip():
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue          # a bare value: nothing to read from it
                kind = event.get("type")
                if kind == "message_start":
                    usage = (event.get("message") or {}).get("usage") or {}
                    prompt_tokens = usage.get("input_tokens", 0) or 0
                elif kind == "content_block_delta":
                    delta = event.get("delta") or {}
                    if delta.get("type") == "thinking_delta":
                        yield {"thinking": delta.get("thinking", "")}
                    elif delta.get("text"):
                        yield {"text": delta["text"]}
                elif kind == "message_delta":
                    completion_tokens = (event.get("usage") or {}).get(
                        "output_tokens", completion_tokens) or completion_tokens
                elif kind == "error":
                    detail = (event.get("error") or {}).get("message", "unknown error")
                    raise RuntimeError(f"Anthropic: {detail}")
            elapsed = time.time() - started
            yield {"done": True, "prompt_tokens": prompt_tokens,
                   "completion_tokens": completion_tokens,
                   "total_seconds": elapsed, "eval_seconds": elapsed}

        return _as_stream(chunks)


class OpenAIProvider(Provider):
    name = "openai"
    label = "OpenAI"
    key_env = ("OPENAI_API_KEY",)
    key_help = "platform.openai.com -> API keys"
    default_base_url = "https://api.openai.com/v1"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    def list_models(self) -> list:
        data = self._get_json(f"{self.base_url}/models", self._headers())
        names = sorted(item["id"] for item in data.get("data", []) if item.get("id"))
        return [{"name": name, "detail": ""} for name in names]

    def stream(self, messages: list, max_tokens: int | None = None):
        import config
        payload = {
            "model": self.model,
            "messages": merge_runs(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": max_tokens or config.NUM_PREDICT,
        }

        def chunks():
            try:
                response = self._post_sse(f"{self.base_url}/chat/completions",
                                          self._headers(), payload)
            except RuntimeError as error:
                # Older models and most OpenAI-compatible servers only know the
                # original parameter name.
                if "max_completion_tokens" not in str(error):
                    raise
                retry = dict(payload)
                retry["max_tokens"] = retry.pop("max_completion_tokens")
                response = self._post_sse(f"{self.base_url}/chat/completions",
                                          self._headers(), retry)

            prompt_tokens = completion_tokens = 0
            started = time.time()
            for _event, data in iter_sse(response):
                data = data.strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue          # a bare value: nothing to read from it
                if event.get("error"):
                    raise RuntimeError(f"OpenAI: {event['error'].get('message', 'error')}")
                usage = event.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens) or prompt_tokens
                    completion_tokens = usage.get("completion_tokens",
                                                  completion_tokens) or completion_tokens
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("reasoning_content"):
                        yield {"thinking": delta["reasoning_content"]}
                    if delta.get("content"):
                        yield {"text": delta["content"]}
            elapsed = time.time() - started
            yield {"done": True, "prompt_tokens": prompt_tokens,
                   "completion_tokens": completion_tokens,
                   "total_seconds": elapsed, "eval_seconds": elapsed}

        return _as_stream(chunks)


class GeminiProvider(Provider):
    name = "gemini"
    label = "Google Gemini"
    key_env = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    key_help = "aistudio.google.com -> Get API key"
    default_base_url = "https://generativelanguage.googleapis.com"

    def _headers(self) -> dict:
        return {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

    def list_models(self) -> list:
        data = self._get_json(f"{self.base_url}/v1beta/models?pageSize=200",
                              self._headers())
        models = []
        for item in data.get("models", []):
            name = str(item.get("name", "")).removeprefix("models/")
            methods = item.get("supportedGenerationMethods") or []
            if name and (not methods or "generateContent" in methods):
                models.append({"name": name, "detail": item.get("displayName", "")})
        return models

    def stream(self, messages: list, max_tokens: int | None = None):
        import config
        system, conversation = split_system(messages)
        contents = [{"role": "model" if m["role"] == "assistant" else "user",
                     "parts": [{"text": m.get("content", "")}]}
                    for m in merge_runs(conversation)]
        payload = {
            "contents": contents or [{"role": "user", "parts": [{"text": "."}]}],
            "generationConfig": {"maxOutputTokens": max_tokens or config.NUM_PREDICT},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        url = (f"{self.base_url}/v1beta/models/{self.model}"
               ":streamGenerateContent?alt=sse")

        def chunks():
            response = self._post_sse(url, self._headers(), payload)
            prompt_tokens = completion_tokens = 0
            started = time.time()
            for _event, data in iter_sse(response):
                if not data.strip():
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue          # a bare value: nothing to read from it
                if event.get("error"):
                    raise RuntimeError(f"Gemini: {event['error'].get('message', 'error')}")
                usage = event.get("usageMetadata") or {}
                prompt_tokens = usage.get("promptTokenCount", prompt_tokens) or prompt_tokens
                completion_tokens = usage.get("candidatesTokenCount",
                                              completion_tokens) or completion_tokens
                for candidate in event.get("candidates") or []:
                    for part in (candidate.get("content") or {}).get("parts") or []:
                        text = part.get("text")
                        if not text:
                            continue
                        if part.get("thought"):
                            yield {"thinking": text}
                        else:
                            yield {"text": text}
            elapsed = time.time() - started
            yield {"done": True, "prompt_tokens": prompt_tokens,
                   "completion_tokens": completion_tokens,
                   "total_seconds": elapsed, "eval_seconds": elapsed}

        return _as_stream(chunks)


PROVIDERS = {p.name: p for p in
             (OllamaProvider, AnthropicProvider, OpenAIProvider, GeminiProvider)}


# ---------------------------------------------------------------------------
# what is connected
# ---------------------------------------------------------------------------

_state: dict = {}
_active: Provider | None = None


def load_state(force: bool = False) -> dict:
    global _state
    if _state and not force:
        return _state
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("active", "ollama")
    data.setdefault("providers", {})
    _state = data
    return _state


def save_state() -> str:
    """Write the config with owner-only permissions - it holds API keys."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(_state, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except Exception:
        pass
    return CONFIG_PATH


def settings_for(name: str) -> dict:
    return load_state()["providers"].setdefault(name, {})


def build(name: str) -> Provider | None:
    factory = PROVIDERS.get(name)
    return factory(settings_for(name)) if factory else None


def current() -> Provider:
    """The provider in use. Falls back to Ollama, which needs no account."""
    global _active
    if _active is None:
        _active = build(load_state().get("active", "ollama")) or OllamaProvider({})
    return _active


def connect(name: str, model: str = "", api_key: str = "",
            base_url: str = "") -> tuple:
    """Point the harness at a provider. Returns (provider, problem)."""
    global _active
    if name not in PROVIDERS:
        return None, f"unknown provider '{name}' - try {', '.join(PROVIDERS)}"

    settings = settings_for(name)
    if api_key:
        settings["api_key"] = api_key
    if base_url:
        settings["base_url"] = base_url
    if model:
        settings["model"] = model

    provider = build(name)
    load_state()["active"] = name
    _active = provider
    _sync_config(provider)
    save_state()
    return provider, provider.ready()


def _sync_config(provider: Provider) -> None:
    """Keep `config.MODEL` in step - the TUI and session files both read it."""
    try:
        import config
        config.MODEL = provider.model or config.MODEL
    except Exception:
        pass


async def complete(messages: list, max_tokens: int) -> str:
    """One short non-streamed answer - used for session titles and summaries."""
    pieces = []
    async for chunk in current().stream(messages, max_tokens=max_tokens):
        if chunk.get("text"):
            pieces.append(chunk["text"])
    return "".join(pieces).strip()


def status_line() -> str:
    provider = current()
    problem = provider.ready()
    if problem:
        return f"{provider.label} ({problem})"
    where = f" · key {provider.key_source}" if provider.key_source else ""
    return f"{provider.label} · {provider.model}{where}"


def apply_startup() -> None:
    """Restore the last connection when the app starts."""
    _sync_config(current())


if __name__ == "__main__":
    print("This file can not run directly.")
