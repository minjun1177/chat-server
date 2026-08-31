"""MCP (Model Context Protocol) client: attach external tool servers.

A server is declared in `.mcp.json` next to the app (or `~/.localchat/mcp.json`
for personal ones):

    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "D:/work"]
        },
        "github": {
          "url": "https://api.githubcopilot.com/mcp/",
          "headers": {"Authorization": "Bearer ${GITHUB_TOKEN}"}
        }
      }
    }

At startup every enabled server is connected, its tool list is fetched and
merged into the system prompt as `mcp__<server>__<tool>`, and calls are routed
back through `dispatch_tool`. Three transports are supported: `stdio` (a local
subprocess speaking newline-delimited JSON-RPC), `http` (streamable HTTP), and
`sse` (the deprecated 2024-11-05 HTTP+SSE transport).

MCP is plain JSON-RPC 2.0, so this is a self-contained implementation rather
than a dependency on the official SDK - it keeps the harness installable with
nothing but the stdlib, and keeps the call path synchronous like every other
tool in `tools.py`.

Stdlib only at module level on purpose: `systemprompt.py` imports this module,
and `config.py` imports `systemprompt`, so importing `config` here at the top
would create a cycle. Settings are read lazily through `_cfg()`, which also
tolerates `config` being half-initialised during that first import.
"""

import atexit
import itertools
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from sse import iter_sse

import env

env.load()   # ${VAR} in .mcp.json may come from .env


PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "localchat", "version": "1.0.0"}

TOOL_PREFIX = "mcp__"
PROJECT_CONFIG_FILES = (".mcp.json", "mcp.json")
USER_CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".localchat", "mcp.json")

STDERR_KEEP_LINES = 60
LIST_PAGE_LIMIT = 20            # pagination safety valve for tools/resources/prompts
DESC_MAX_LENGTH = 500           # per tool description, in the system prompt
PARAM_DESC_MAX_LENGTH = 260
INSTRUCTIONS_MAX_LENGTH = 1200


class MCPError(Exception):
    """A server was unreachable, spoke badly, or answered with a JSON-RPC error."""


def _cfg(name: str, default):
    """Read a setting from `config`, tolerating a partially-initialised module."""
    import config
    return getattr(config, name, default)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(text)).strip("-") or "server"


_ENV_PATTERN = re.compile(r"\$\{(?:env:)?([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value):
    """Substitute ${VAR} / ${env:VAR} from the environment, recursively."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# server-side event stream
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# transports
# ---------------------------------------------------------------------------

class StdioTransport:
    """A local subprocess exchanging newline-delimited JSON-RPC on stdin/stdout."""

    kind = "stdio"

    def __init__(self, spec: dict):
        self.command = str(spec.get("command") or "")
        self.args = [str(a) for a in (spec.get("args") or [])]
        self.env = spec.get("env") or {}
        self.cwd = spec.get("cwd") or None
        self.proc = None
        self.stderr_tail = deque(maxlen=STDERR_KEEP_LINES)
        self._on_message = None
        self._write_lock = threading.Lock()

    @property
    def description(self) -> str:
        return " ".join([self.command] + self.args)

    def _argv(self) -> list[str]:
        resolved = shutil.which(self.command) or self.command
        argv = [resolved] + self.args
        # CreateProcess cannot run .cmd/.bat directly, and npx/uvx are exactly
        # that on Windows - the single most common way an MCP server fails to
        # start there.
        if platform.system() == "Windows" and resolved.lower().endswith((".cmd", ".bat")):
            argv = ["cmd", "/c"] + argv
        return argv

    def start(self, on_message) -> None:
        if not self.command:
            raise MCPError("no 'command' given for a stdio server")
        self._on_message = on_message

        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in self.env.items()})
        env.setdefault("PYTHONIOENCODING", "utf-8")

        kwargs = {}
        if platform.system() == "Windows":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            self.proc = subprocess.Popen(
                self._argv(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self.cwd,
                env=env,
                **kwargs,
            )
        except FileNotFoundError:
            raise MCPError(f"command not found: {self.command}")
        except Exception as e:
            raise MCPError(f"failed to start '{self.command}': {e}")

        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self) -> None:
        stream = self.proc.stdout
        try:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    # Servers that print banners to stdout are non-conformant but
                    # common; keep the noise for diagnostics instead of crashing.
                    self.stderr_tail.append(f"(stdout) {line[:400]}")
                    continue
                if self._on_message:
                    try:
                        self._on_message(message)
                    except Exception:
                        pass
        except Exception:
            pass

    def _read_stderr(self) -> None:
        try:
            for line in self.proc.stderr:
                line = line.rstrip()
                if line:
                    self.stderr_tail.append(line[:400])
        except Exception:
            pass

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def send(self, message: dict) -> None:
        if not self.alive():
            tail = "; ".join(list(self.stderr_tail)[-3:])
            raise MCPError(f"server process is not running{f' - {tail}' if tail else ''}")
        payload = json.dumps(message, ensure_ascii=False) + "\n"
        with self._write_lock:
            try:
                self.proc.stdin.write(payload)
                self.proc.stdin.flush()
            except Exception as e:
                raise MCPError(f"failed to write to server: {e}")

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=3)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None


class HTTPTransport:
    """Streamable HTTP: every message is POSTed; replies arrive inline or as SSE."""

    kind = "http"

    def __init__(self, spec: dict):
        self.url = str(spec.get("url") or "")
        self.headers = {str(k): str(v) for k, v in (spec.get("headers") or {}).items()}
        self.session_id = ""
        self.protocol_version = ""
        self._session = None
        self._on_message = None

    @property
    def description(self) -> str:
        return self.url

    def start(self, on_message) -> None:
        if not self.url:
            raise MCPError("no 'url' given for an http server")
        try:
            import requests
        except ImportError:
            raise MCPError("the 'requests' package is required for http transports")
        self._on_message = on_message
        self._session = requests.Session()

    def _request_headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            # An SSE reply is read straight off the raw stream, so it must not
            # arrive compressed.
            "Accept-Encoding": "identity",
        }
        headers.update(self.headers)
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if self.protocol_version:
            headers["MCP-Protocol-Version"] = self.protocol_version
        return headers

    def alive(self) -> bool:
        return self._session is not None

    def send(self, message: dict) -> None:
        if self._session is None:
            raise MCPError("transport is closed")
        timeout = _cfg("MCP_HTTP_TIMEOUT", 60)
        try:
            response = self._session.post(
                self.url,
                data=json.dumps(message, ensure_ascii=False).encode("utf-8"),
                headers=self._request_headers(),
                timeout=timeout,
                stream=True,
            )
        except Exception as e:
            raise MCPError(f"HTTP request failed: {e}")

        session_id = response.headers.get("Mcp-Session-Id") or response.headers.get("mcp-session-id")
        if session_id:
            self.session_id = session_id

        if response.status_code >= 400:
            body = (response.text or "")[:300].strip()
            raise MCPError(f"HTTP {response.status_code}{f' - {body}' if body else ''}")

        content_type = (response.headers.get("Content-Type") or "").lower()
        if "text/event-stream" in content_type:
            for _event, data in iter_sse(response):
                if not data.strip():
                    continue
                try:
                    self._deliver(json.loads(data))
                except json.JSONDecodeError:
                    continue
            return

        body = response.content
        if not body:
            return              # 202 Accepted: the reply to a notification
        try:
            self._deliver(json.loads(body.decode("utf-8", "replace")))
        except json.JSONDecodeError:
            raise MCPError("server returned a non-JSON body")

    def _deliver(self, message) -> None:
        if self._on_message:
            self._on_message(message)

    def close(self) -> None:
        if self._session is None:
            return
        # Best effort: tell the server the session is over, then drop it.
        if self.session_id:
            try:
                self._session.delete(self.url, headers=self._request_headers(), timeout=5)
            except Exception:
                pass
        try:
            self._session.close()
        except Exception:
            pass
        self._session = None


class SSETransport:
    """The deprecated HTTP+SSE transport: a long-lived GET plus POSTed messages."""

    kind = "sse"

    def __init__(self, spec: dict):
        self.url = str(spec.get("url") or "")
        self.headers = {str(k): str(v) for k, v in (spec.get("headers") or {}).items()}
        self.post_url = ""
        self._session = None
        self._response = None
        self._on_message = None
        self._endpoint_ready = threading.Event()
        self._failure = ""

    @property
    def description(self) -> str:
        return self.url

    def start(self, on_message) -> None:
        if not self.url:
            raise MCPError("no 'url' given for an sse server")
        try:
            import requests
        except ImportError:
            raise MCPError("the 'requests' package is required for sse transports")
        self._on_message = on_message
        self._session = requests.Session()

        headers = {"Accept": "text/event-stream", "Cache-Control": "no-cache",
                   "Accept-Encoding": "identity"}
        headers.update(self.headers)
        try:
            self._response = self._session.get(self.url, headers=headers, stream=True, timeout=(10, None))
            self._response.raise_for_status()
        except Exception as e:
            raise MCPError(f"failed to open SSE stream: {e}")

        threading.Thread(target=self._read_stream, daemon=True).start()
        if not self._endpoint_ready.wait(timeout=_cfg("MCP_STARTUP_TIMEOUT", 30)):
            raise MCPError(self._failure or "server never sent its 'endpoint' event")

    def _read_stream(self) -> None:
        from urllib.parse import urljoin
        try:
            for event, data in iter_sse(self._response):
                if event == "endpoint":
                    self.post_url = urljoin(self.url, data.strip())
                    self._endpoint_ready.set()
                elif data.strip():
                    try:
                        message = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if self._on_message:
                        try:
                            self._on_message(message)
                        except Exception:
                            pass
        except Exception as e:
            self._failure = str(e)
        finally:
            self._endpoint_ready.set()

    def alive(self) -> bool:
        return self._session is not None and bool(self.post_url)

    def send(self, message: dict) -> None:
        if not self.post_url:
            raise MCPError("SSE endpoint is not ready")
        headers = {"Content-Type": "application/json"}
        headers.update(self.headers)
        try:
            response = self._session.post(
                self.post_url,
                data=json.dumps(message, ensure_ascii=False).encode("utf-8"),
                headers=headers,
                timeout=_cfg("MCP_HTTP_TIMEOUT", 60),
            )
        except Exception as e:
            raise MCPError(f"HTTP request failed: {e}")
        if response.status_code >= 400:
            raise MCPError(f"HTTP {response.status_code} - {(response.text or '')[:200]}")

    def close(self) -> None:
        try:
            if self._response is not None:
                self._response.close()
        except Exception:
            pass
        try:
            if self._session is not None:
                self._session.close()
        except Exception:
            pass
        self._session = None
        self._response = None
        self.post_url = ""


def _build_transport(spec: dict):
    kind = str(spec.get("type") or spec.get("transport") or "").strip().lower()
    if not kind:
        kind = "stdio" if spec.get("command") else ("http" if spec.get("url") else "")
    if kind in ("stdio", "local", "process"):
        return StdioTransport(spec)
    if kind in ("http", "streamable-http", "streamablehttp", "streamable_http"):
        return HTTPTransport(spec)
    if kind == "sse":
        return SSETransport(spec)
    raise MCPError(f"unknown transport '{kind or 'none'}' - give a 'command' or a 'url'")


# ---------------------------------------------------------------------------
# a single server
# ---------------------------------------------------------------------------

class MCPServer:
    def __init__(self, name: str, spec: dict, source: str):
        self.name = name
        self.slug = _slug(name)
        self.spec = spec
        self.source = source
        self.transport = None
        self.state = "idle"                 # idle | connected | failed | disabled
        self.error = ""
        self.info: dict = {}
        self.capabilities: dict = {}
        self.instructions = ""
        self.protocol_version = ""
        self.tools: list[dict] = []
        self.resources: list[dict] = []
        self.prompts: list[dict] = []
        self.logs = deque(maxlen=40)
        self.stale = False

        self._ids = itertools.count(1)
        self._pending: dict[int, queue.Queue] = {}
        self._lock = threading.Lock()

    # -- plumbing ----------------------------------------------------------

    @property
    def target(self) -> str:
        if self.transport is not None:
            return self.transport.description
        return str(self.spec.get("command") or self.spec.get("url") or "")

    @property
    def timeout(self) -> float:
        try:
            return float(self.spec.get("timeout") or _cfg("MCP_CALL_TIMEOUT", 120))
        except (TypeError, ValueError):
            return float(_cfg("MCP_CALL_TIMEOUT", 120))

    def _dispatch(self, message) -> None:
        """Route one incoming message. Runs on the transport's reader thread."""
        if isinstance(message, list):
            for item in message:
                self._dispatch(item)
            return
        if not isinstance(message, dict):
            return

        mid = message.get("id")
        method = message.get("method")

        if mid is not None and method is None:
            with self._lock:
                waiter = self._pending.pop(mid, None)
            if waiter is not None:
                waiter.put(message)
            return

        if mid is not None and method:
            # A server-initiated request. We advertise no capabilities, so the
            # only thing we owe an answer to is ping - everything else gets a
            # proper error rather than being left to hang.
            if method == "ping":
                reply = {"jsonrpc": "2.0", "id": mid, "result": {}}
            else:
                reply = {"jsonrpc": "2.0", "id": mid,
                         "error": {"code": -32601, "message": f"{method} is not supported by this client"}}
            try:
                self.transport.send(reply)
            except Exception:
                pass
            return

        if method == "notifications/tools/list_changed":
            self.stale = True
        elif method in ("notifications/resources/list_changed", "notifications/prompts/list_changed"):
            self.stale = True
        elif method == "notifications/message":
            params = message.get("params") or {}
            data = params.get("data")
            text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
            self.logs.append(f"[{params.get('level', 'info')}] {text[:300]}")

    def request(self, method: str, params: dict | None = None, timeout: float | None = None) -> dict:
        if self.transport is None:
            raise MCPError("not connected")
        mid = next(self._ids)
        waiter: queue.Queue = queue.Queue(maxsize=1)
        with self._lock:
            self._pending[mid] = waiter

        payload = {"jsonrpc": "2.0", "id": mid, "method": method}
        if params is not None:
            payload["params"] = params

        try:
            self.transport.send(payload)
        except Exception:
            with self._lock:
                self._pending.pop(mid, None)
            raise

        wait_for = timeout if timeout is not None else self.timeout
        try:
            reply = waiter.get(timeout=wait_for)
        except queue.Empty:
            with self._lock:
                self._pending.pop(mid, None)
            raise MCPError(f"'{method}' timed out after {wait_for:g}s")

        if "error" in reply:
            error = reply.get("error") or {}
            raise MCPError(f"{error.get('message', 'unknown error')} (code {error.get('code', '?')})")
        result = reply.get("result")
        return result if isinstance(result, dict) else {}

    def notify(self, method: str, params: dict | None = None) -> None:
        if self.transport is None:
            return
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        try:
            self.transport.send(payload)
        except Exception:
            pass

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> bool:
        self.error = ""
        try:
            self.transport = _build_transport(self.spec)
            self.transport.start(self._dispatch)

            result = self.request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": CLIENT_INFO,
                },
                timeout=float(_cfg("MCP_STARTUP_TIMEOUT", 30)),
            )
            self.info = result.get("serverInfo") or {}
            self.capabilities = result.get("capabilities") or {}
            self.protocol_version = str(result.get("protocolVersion") or PROTOCOL_VERSION)
            self.instructions = str(result.get("instructions") or "").strip()
            if isinstance(self.transport, HTTPTransport):
                self.transport.protocol_version = self.protocol_version

            self.notify("notifications/initialized")
            self.refresh()
            self.state = "connected"
            return True
        except Exception as e:
            self.state = "failed"
            self.error = str(e)
            detail = self.stderr_tail()
            if detail:
                self.error += f" | stderr: {detail}"
            self.close()
            return False

    def _list_all(self, method: str, key: str) -> list[dict]:
        items: list[dict] = []
        cursor = None
        for _ in range(LIST_PAGE_LIMIT):
            result = self.request(method, {"cursor": cursor} if cursor else {})
            page = result.get(key)
            if isinstance(page, list):
                items.extend(x for x in page if isinstance(x, dict))
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return items

    def refresh(self) -> None:
        """Re-read what the server exposes. Failures leave the old lists alone."""
        if "tools" in self.capabilities:
            try:
                self.tools = self._list_all("tools/list", "tools")
            except MCPError:
                pass
        if "resources" in self.capabilities:
            try:
                self.resources = self._list_all("resources/list", "resources")
            except MCPError:
                pass
        if "prompts" in self.capabilities:
            try:
                self.prompts = self._list_all("prompts/list", "prompts")
            except MCPError:
                pass
        self.stale = False

    def stderr_tail(self, lines: int = 3) -> str:
        if isinstance(self.transport, StdioTransport) and self.transport.stderr_tail:
            return " / ".join(list(self.transport.stderr_tail)[-lines:])
        return ""

    def close(self) -> None:
        if self.transport is not None:
            try:
                self.transport.close()
            except Exception:
                pass
        self.transport = None
        with self._lock:
            self._pending.clear()

    # -- operations --------------------------------------------------------

    def get_tool(self, tool_name: str) -> dict | None:
        for tool in self.tools:
            if tool.get("name") == tool_name:
                return tool
        lowered = tool_name.lower()
        for tool in self.tools:
            if str(tool.get("name", "")).lower() == lowered:
                return tool
        return None

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        return self.request("tools/call", {"name": tool_name, "arguments": arguments or {}})

    def read_resource(self, uri: str) -> dict:
        return self.request("resources/read", {"uri": uri})

    def get_prompt(self, prompt_name: str, arguments: dict | None = None) -> dict:
        return self.request("prompts/get", {"name": prompt_name, "arguments": arguments or {}})


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

_servers: dict[str, MCPServer] = {}
_config_files: list[str] = []
_loaded = False


def config_paths() -> list[tuple[str, str]]:
    """(source label, path) pairs, highest precedence first."""
    paths = [("project", os.path.abspath(name)) for name in PROJECT_CONFIG_FILES]
    paths.append(("user", USER_CONFIG_FILE))
    return paths


def _read_config(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        raise MCPError(f"{path}: invalid JSON ({e})")
    except Exception as e:
        raise MCPError(f"{path}: {e}")

    servers = data.get("mcpServers") or data.get("servers") or {}
    return servers if isinstance(servers, dict) else {}


def load_servers(force: bool = False) -> dict[str, MCPServer]:
    """Read the config files and build (but do not connect) the server objects."""
    global _loaded
    if _loaded and not force:
        return _servers

    if force:
        shutdown()

    _servers.clear()
    _config_files.clear()
    errors = []

    for source, path in config_paths():
        if not os.path.isfile(path):
            continue
        try:
            entries = _read_config(path)
        except MCPError as e:
            errors.append(str(e))
            continue
        _config_files.append(path)
        for name, raw in entries.items():
            if name in _servers or not isinstance(raw, dict):
                continue          # first file wins: project overrides user
            spec = _expand(raw)
            server = MCPServer(name, spec, source)
            disabled = spec.get("disabled") is True or spec.get("enabled") is False
            if disabled:
                server.state = "disabled"
            _servers[name] = server

    load_servers.errors = errors
    _loaded = True
    return _servers


load_servers.errors = []


def all_servers() -> list[MCPServer]:
    return list(load_servers().values())


def connected_servers() -> list[MCPServer]:
    return [s for s in load_servers().values() if s.state == "connected"]


def get_server(name: str) -> MCPServer | None:
    if not name:
        return None
    servers = load_servers()
    if name in servers:
        return servers[name]
    lowered = name.lower()
    for server in servers.values():
        if server.name.lower() == lowered or server.slug.lower() == lowered:
            return server
    return None


def connect_all(progress=None) -> list[MCPServer]:
    """Connect every enabled server in parallel. Returns the ones that came up."""
    if not _cfg("MCP_ENABLED", True):
        return []

    servers = [s for s in load_servers().values() if s.state in ("idle", "failed")]
    if not servers:
        return connected_servers()

    def _connect(server: MCPServer) -> MCPServer:
        server.connect()
        if progress is not None:
            try:
                progress(server)
            except Exception:
                pass
        return server

    with ThreadPoolExecutor(max_workers=min(8, len(servers))) as pool:
        list(pool.map(_connect, servers))
    return connected_servers()


def reconnect(name: str = "") -> list[MCPServer]:
    """Reconnect one server by name, or every configured server."""
    if name:
        server = get_server(name)
        if server is None:
            return []
        server.close()
        if server.state != "disabled":
            server.state = "idle"
            server.connect()
        return [server]

    load_servers(force=True)
    connect_all()
    return all_servers()


def shutdown() -> None:
    for server in _servers.values():
        server.close()
        if server.state == "connected":
            server.state = "idle"


atexit.register(shutdown)


# ---------------------------------------------------------------------------
# tool naming and lookup
# ---------------------------------------------------------------------------

def qualified_name(server: MCPServer, tool_name: str) -> str:
    return f"{TOOL_PREFIX}{server.slug}__{tool_name}"


def is_mcp_tool(name: str) -> bool:
    if not name:
        return False
    if str(name).startswith(TOOL_PREFIX):
        return True
    return resolve_tool(name) is not None


def resolve_tool(name: str) -> tuple[MCPServer, dict] | None:
    """Find the server and tool a call refers to, forgiving small name mangling.

    Small local models drop the prefix or swap the separator often enough that
    strict matching would turn working calls into 'unknown tool'.
    """
    if not name:
        return None
    raw = str(name).strip()

    for server in connected_servers():
        for tool in server.tools:
            if qualified_name(server, str(tool.get("name", ""))) == raw:
                return server, tool

    stripped = raw[len(TOOL_PREFIX):] if raw.startswith(TOOL_PREFIX) else raw
    for separator in ("__", "::", ".", "/", ":"):
        if separator not in stripped:
            continue
        server_part, _, tool_part = stripped.partition(separator)
        server = get_server(server_part)
        if server is not None and server.state == "connected":
            tool = server.get_tool(tool_part)
            if tool is not None:
                return server, tool

    # Bare tool name, accepted only when exactly one server offers it.
    matches = []
    for server in connected_servers():
        tool = server.get_tool(stripped)
        if tool is not None:
            matches.append((server, tool))
    if len(matches) == 1:
        return matches[0]
    return None


def tool_count() -> int:
    return sum(len(s.tools) for s in connected_servers())


def has_resources() -> bool:
    return any(s.resources or "resources" in s.capabilities for s in connected_servers())


def has_prompts() -> bool:
    return any(s.prompts for s in connected_servers())


# ---------------------------------------------------------------------------
# rendering the system prompt section
# ---------------------------------------------------------------------------

def _type_label(spec: dict) -> str:
    kind = spec.get("type")
    if isinstance(kind, list):
        kind = "|".join(str(k) for k in kind if k != "null") or "any"
    if not kind:
        for key in ("anyOf", "oneOf", "allOf"):
            variants = spec.get(key)
            if isinstance(variants, list):
                labels = [_type_label(v) for v in variants if isinstance(v, dict)]
                labels = [l for l in labels if l and l != "any"]
                if labels:
                    return "|".join(dict.fromkeys(labels))
        if "enum" in spec:
            return "string"
        return "any"
    if kind == "array":
        items = spec.get("items")
        if isinstance(items, dict):
            inner = _type_label(items)
            if inner != "any":
                return f"array of {inner}"
        return "array"
    return str(kind)


def describe_schema(schema: dict | None) -> dict:
    """Flatten a JSON Schema into the `{param: description}` shape the prompt uses."""
    if not isinstance(schema, dict):
        return {}
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return {}

    required = schema.get("required")
    required = set(required) if isinstance(required, list) else set()

    described = {}
    for key, spec in properties.items():
        if not isinstance(spec, dict):
            spec = {}
        head = f"{_type_label(spec)}, {'required' if key in required else 'optional'}."
        text = " ".join(str(spec.get("description") or "").split())
        parts = [head]
        if text:
            parts.append(text)
        enum = spec.get("enum")
        if isinstance(enum, list) and enum:
            allowed = ", ".join(json.dumps(v, ensure_ascii=False) for v in enum[:12])
            parts.append(f"Allowed values: {allowed}.")
        if "default" in spec:
            parts.append(f"Default: {json.dumps(spec['default'], ensure_ascii=False)}.")
        joined = " ".join(parts)
        if len(joined) > PARAM_DESC_MAX_LENGTH:
            joined = joined[:PARAM_DESC_MAX_LENGTH - 1].rstrip() + "…"
        described[str(key)] = joined
    return described


def _tool_entries() -> list[dict]:
    limit = int(_cfg("MCP_MAX_TOOLS_PER_SERVER", 40))
    entries = []
    for server in connected_servers():
        for tool in server.tools[:limit]:
            name = str(tool.get("name") or "").strip()
            if not name:
                continue
            description = " ".join(str(tool.get("description") or "").split())
            annotations = tool.get("annotations") if isinstance(tool.get("annotations"), dict) else {}
            title = annotations.get("title") or tool.get("title")
            if title and title not in description:
                description = f"{title}. {description}".strip()
            if not description:
                description = f"Tool '{name}' provided by the '{server.name}' MCP server."
            if len(description) > DESC_MAX_LENGTH:
                description = description[:DESC_MAX_LENGTH - 1].rstrip() + "…"
            if annotations.get("readOnlyHint") is True:
                description += " (read-only)"
            if annotations.get("destructiveHint") is True:
                description += " (destructive - confirm with the user first)"
            entries.append({
                "name": qualified_name(server, name),
                "description": f"[{server.name}] {description}",
                "parameters": describe_schema(tool.get("inputSchema") or tool.get("input_schema")),
            })
    return entries


def _resource_tool_entries() -> list[dict]:
    if not has_resources():
        return []
    return [
        {
            "name": "list_mcp_resources",
            "description": "List the resources (files, records, documents) exposed by the attached MCP servers, with the URI needed to read each one.",
            "parameters": {
                "server": "(Optional) Only list resources from this MCP server name."
            },
        },
        {
            "name": "read_mcp_resource",
            "description": "Read one resource from an MCP server by its URI. Use list_mcp_resources first to find the URI.",
            "parameters": {
                "uri": "The resource URI exactly as reported by list_mcp_resources.",
                "server": "(Optional) The MCP server that owns the URI. Inferred when omitted.",
            },
        },
    ]


def mcp_tools_prompt() -> str:
    """The MCP section of the system prompt. Empty when nothing is connected."""
    if not _cfg("MCP_ENABLED", True):
        return ""
    entries = _tool_entries()
    if not entries:
        return ""
    entries.extend(_resource_tool_entries())

    lines = [
        "\n### MCP TOOLS:",
        "These tools come from MCP servers attached to this session. Call them",
        "exactly like the built-in tools, using the full name shown below - the",
        "`mcp__<server>__<tool>` name is the real tool name, do not shorten it.",
        "Their results are authoritative: report what they return, never what you",
        "assume they would return.",
        "",
        json.dumps(entries, indent=2, ensure_ascii=False),
        "",
    ]

    notes = []
    for server in connected_servers():
        if server.instructions:
            text = server.instructions
            if len(text) > INSTRUCTIONS_MAX_LENGTH:
                text = text[:INSTRUCTIONS_MAX_LENGTH].rstrip() + "…"
            notes.append(f"- {server.name}: {text}")
    if notes:
        lines.append("Server notes (written by the servers themselves):")
        lines.extend(notes)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# calling
# ---------------------------------------------------------------------------

def _block_to_text(block) -> str:
    if not isinstance(block, dict):
        return str(block)
    kind = block.get("type")
    if kind == "text":
        return str(block.get("text") or "")
    if kind in ("image", "audio"):
        data = block.get("data") or ""
        return f"[{kind}: {block.get('mimeType', 'unknown type')}, {len(data)} base64 chars - not displayable here]"
    if kind == "resource_link":
        return f"[resource link: {block.get('uri', '')} - read it with read_mcp_resource]"
    if kind == "resource":
        resource = block.get("resource") or {}
        uri = resource.get("uri", "")
        if resource.get("text") is not None:
            return f"[resource {uri}]\n{resource['text']}"
        return f"[resource {uri}: {resource.get('mimeType', 'binary')} - binary content omitted]"
    return json.dumps(block, ensure_ascii=False)


def format_tool_result(result: dict) -> str:
    """Turn an MCP tools/call result into the plain text the model reads."""
    content = result.get("content")
    text = "\n".join(t for t in (_block_to_text(b) for b in content) if t) if isinstance(content, list) else ""

    if not text and result.get("structuredContent") is not None:
        text = json.dumps(result["structuredContent"], ensure_ascii=False, indent=2)
    if not text:
        text = "(the tool returned no content)"

    if result.get("isError") is True:
        text = f"[Error] The MCP tool reported a failure:\n{text}"

    limit = int(_cfg("MCP_RESULT_CHARS", 8000))
    if len(text) > limit:
        text = text[:limit] + f"\n...[MCP result truncated - {len(text) - limit} chars omitted]"
    return text


def auto_approved(server: MCPServer, tool_name: str) -> bool:
    """Whether this call may skip the approval prompt."""
    trusted = _cfg("MCP_TRUSTED_SERVERS", [])
    if server.name in trusted or server.spec.get("trust") is True:
        return True
    allow = server.spec.get("autoApprove") or server.spec.get("auto_approve") or []
    if isinstance(allow, list) and tool_name in allow:
        return True
    if _cfg("MCP_AUTO_APPROVE_READONLY", False):
        tool = server.get_tool(tool_name) or {}
        annotations = tool.get("annotations")
        if isinstance(annotations, dict) and annotations.get("readOnlyHint") is True:
            return True
    return False


def call_tool(server: MCPServer, tool_name: str, arguments: dict) -> str:
    try:
        result = server.call_tool(tool_name, arguments)
    except MCPError as e:
        return f"[Error] MCP server '{server.name}' failed to run '{tool_name}': {e}"
    except Exception as e:
        return f"[Error] MCP call to '{server.name}.{tool_name}' failed: {e}"
    return format_tool_result(result)


def available_tool_names() -> list[str]:
    names = []
    for server in connected_servers():
        for tool in server.tools:
            name = str(tool.get("name") or "")
            if name:
                names.append(qualified_name(server, name))
    return names


# ---------------------------------------------------------------------------
# resources
# ---------------------------------------------------------------------------

def list_resources_text(server_name: str = "") -> str:
    servers = connected_servers()
    if server_name:
        server = get_server(server_name)
        if server is None or server.state != "connected":
            return f"[Error] No connected MCP server named '{server_name}'."
        servers = [server]

    lines = []
    for server in servers:
        if server.stale:
            server.refresh()
        if not server.resources:
            continue
        lines.append(f"[{server.name}] {len(server.resources)} resource(s)")
        for resource in server.resources:
            uri = resource.get("uri", "")
            title = resource.get("name") or resource.get("title") or ""
            mime = resource.get("mimeType") or ""
            description = " ".join(str(resource.get("description") or "").split())
            detail = " · ".join(p for p in (title, mime) if p)
            lines.append(f"  - {uri}{f'  ({detail})' if detail else ''}")
            if description:
                lines.append(f"      {description[:200]}")

    if not lines:
        return "No MCP resources are available from the connected servers."
    return "\n".join(lines)


def read_resource_text(uri: str, server_name: str = "") -> str:
    if not uri:
        return "[Error] 'uri' is required. Run list_mcp_resources first."

    candidates: list[MCPServer] = []
    if server_name:
        server = get_server(server_name)
        if server is None or server.state != "connected":
            return f"[Error] No connected MCP server named '{server_name}'."
        candidates = [server]
    else:
        for server in connected_servers():
            if any(r.get("uri") == uri for r in server.resources):
                candidates.append(server)
        if not candidates:
            candidates = [s for s in connected_servers() if "resources" in s.capabilities]

    if not candidates:
        return "[Error] No connected MCP server exposes resources."

    last_error = ""
    for server in candidates:
        try:
            result = server.read_resource(uri)
        except MCPError as e:
            last_error = f"{server.name}: {e}"
            continue

        parts = []
        for item in result.get("contents") or []:
            if not isinstance(item, dict):
                continue
            if item.get("text") is not None:
                parts.append(str(item["text"]))
            elif item.get("blob") is not None:
                parts.append(f"[binary content: {item.get('mimeType', 'unknown type')}, "
                             f"{len(str(item['blob']))} base64 chars - not displayable here]")
        text = "\n".join(parts) if parts else "(the resource is empty)"

        limit = int(_cfg("MCP_RESULT_CHARS", 8000))
        if len(text) > limit:
            text = text[:limit] + f"\n...[resource truncated - {len(text) - limit} chars omitted]"
        return f"[{server.name}] {uri}\n{text}"

    return f"[Error] Could not read '{uri}'.{f' {last_error}' if last_error else ''}"


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

def prompt_to_text(result: dict) -> str:
    lines = []
    description = str(result.get("description") or "").strip()
    if description:
        lines.append(description)
    for message in result.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role", "user")
        content = message.get("content")
        if isinstance(content, list):
            text = "\n".join(_block_to_text(b) for b in content)
        else:
            text = _block_to_text(content)
        lines.append(f"[{role}] {text}".strip())
    return "\n\n".join(l for l in lines if l) or "(the prompt is empty)"


# ---------------------------------------------------------------------------
# status, for the TUI
# ---------------------------------------------------------------------------

def status_summary() -> str:
    servers = all_servers()
    if not servers:
        return "none"
    connected = [s for s in servers if s.state == "connected"]
    failed = [s for s in servers if s.state == "failed"]
    disabled = [s for s in servers if s.state == "disabled"]

    parts = [f"{len(connected)}/{len(servers) - len(disabled)} connected"]
    tools = tool_count()
    if tools:
        parts.append(f"{tools} tool{'s' if tools != 1 else ''}")
    if failed:
        parts.append(f"{len(failed)} failed")
    if disabled:
        parts.append(f"{len(disabled)} disabled")
    return " · ".join(parts)


if __name__ == "__main__":
    print("This file can not run directly.")
