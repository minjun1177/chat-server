"""Three layers of log, each one including the layer above it.

    LOG_LEVEL=error     something failed, or was refused
    LOG_LEVEL=tool      + every tool call: who asked, what it did, how it ended
    LOG_LEVEL=request   + every message and command that reached the bot
    LOG_LEVEL=debug     + the harness talking to itself

The layers nest on screen as well as in verbosity. Everything a single exchange
does carries the same four-character turn id and its tool lines are indented
under the request that caused them, so a busy channel still reads as separate
conversations rather than one interleaved stream:

    14:22:01 REQ  a3f2  » alice in #general · "이 서버 정보 알려줘"
    14:22:01 TOOL a3f2    ▸ discord_server_info
    14:22:02 TOOL a3f2    ✓ discord_server_info · 812 chars · 0.21s
    14:22:07 REQ  a3f2  « done · 1 tool · 1420 tokens · 6.1s

The tool layer doubles as the audit trail: a bot that can run shell commands
should leave a record of who asked it to, and of every approval granted or
refused. That is why it sits below errors rather than beside debug output.
"""

import logging
import os
import random
import sys
import time
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler

import env


TOOL = 25                       # between INFO and WARNING
logging.addLevelName(TOOL, "TOOL")

LAYERS = {
    "error": logging.WARNING,
    "tool": TOOL,
    "request": logging.INFO,
    "debug": logging.DEBUG,
}

FILE_BYTES = 2 * 1024 * 1024
FILE_KEEP = 3

_SHORT = {
    logging.ERROR: "ERR ",
    logging.WARNING: "WARN",
    TOOL: "TOOL",
    logging.INFO: "REQ ",
    logging.DEBUG: "DBG ",
}

_turn: ContextVar[str] = ContextVar("log_turn", default="")
_log = logging.getLogger("chatserver")
_level_name = "tool"


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

class _TurnId(logging.Filter):
    def filter(self, record):
        record.turn = _turn.get() or "----"
        return True


class _Line(logging.Formatter):

    def __init__(self, color: bool):
        super().__init__()
        self.color = color

    def format(self, record) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        level = _SHORT.get(record.levelno, "LOG ")
        text = record.getMessage()
        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)
        line = f"{stamp} {level} {getattr(record, 'turn', '----')}  {text}"
        if not self.color:
            return line
        from config import S
        tint = {logging.ERROR: S.ERR, logging.WARNING: S.WARN,
                TOOL: S.INFO, logging.INFO: S.WHITE}.get(record.levelno, S.MUTED)
        return f"{S.MUTED}{stamp}{S.R} {tint}{level}{S.R} {S.MUTED}{getattr(record, 'turn', '----')}{S.R}  {text}"


def setup(level: str = "", path: str = "") -> str:
    """Wire the handlers. Returns the layer actually in use."""
    global _level_name

    wanted = (level or env.text("LOG_LEVEL", "tool")).strip().lower()
    if wanted not in LAYERS:
        unknown, wanted = wanted, "tool"
    else:
        unknown = ""
    _level_name = wanted

    _log.setLevel(LAYERS[wanted])
    _log.propagate = False
    for handler in list(_log.handlers):
        _log.removeHandler(handler)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(_Line(color=sys.stdout.isatty()))
    stream.addFilter(_TurnId())
    _log.addHandler(stream)

    path = path or env.text("LOG_FILE", "")
    if path:
        folder = os.path.dirname(os.path.abspath(path))
        if folder:
            os.makedirs(folder, exist_ok=True)
        rotating = RotatingFileHandler(path, maxBytes=FILE_BYTES, backupCount=FILE_KEEP,
                                       encoding="utf-8")
        rotating.setFormatter(_Line(color=False))
        rotating.addFilter(_TurnId())
        _log.addHandler(rotating)

    # discord.py is chatty about gateway internals; it only joins in at debug.
    logging.getLogger("discord").setLevel(
        logging.DEBUG if wanted == "debug" else logging.WARNING)

    if unknown:
        warn(f"LOG_LEVEL={unknown!r} is not one of {', '.join(LAYERS)} - using 'tool'")
    return wanted


def level() -> str:
    return _level_name


# ---------------------------------------------------------------------------
# one exchange
# ---------------------------------------------------------------------------

def begin(identity: str = ""):
    """Tag everything this task logs from here on. Returns (id, token)."""
    identity = identity or f"{random.getrandbits(16):04x}"
    return identity, _turn.set(identity)


def end(token) -> None:
    _turn.reset(token)


def turn() -> str:
    return _turn.get()


# ---------------------------------------------------------------------------
# the layers
# ---------------------------------------------------------------------------

def preview(text: str, limit: int = 0) -> str:
    """What a message looks like in the log. LOG_PREVIEW=0 hides content."""
    limit = limit or env.integer("LOG_PREVIEW", 120)
    if limit <= 0:
        return f"({len(text or '')} chars)"
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def request(actor: str, where: str, text: str, extra: str = "") -> None:
    tail = f" {extra}" if extra else ""
    _log.info(f"» {actor} in {where} · \"{preview(text)}\"{tail}")


def command(actor: str, where: str, name: str, detail: str = "") -> None:
    _log.info(f"» {actor} in {where} · /{name}{(' ' + detail) if detail else ''}")


def turn_done(summary: str) -> None:
    _log.info(f"« done · {summary}")


def tool_call(name: str, arguments: dict, actor: str = "") -> None:
    who = f" for {actor}" if actor else ""
    _log.log(TOOL, f"  ▸ {name}{who} {_arguments(arguments)}")


def tool_result(name: str, ok: bool, chars: int, seconds: float) -> None:
    mark = "✓" if ok else "✗"
    _log.log(TOOL, f"  {mark} {name} · {chars} chars · {seconds:.2f}s")


def approval(action: str, actor: str, granted: bool, rule: str = "") -> None:
    """Who allowed what. This is the line an audit actually needs."""
    verdict = "granted" if granted else "refused"
    detail = f" · {rule}" if rule else ""
    _log.log(TOOL, f"  ⚠ approval {verdict} by {actor} · {action}{detail}")


def blocked(reason: str) -> None:
    _log.warning(f"  ⛔ {reason}")


def error(message: str, exc: bool = False) -> None:
    _log.error(f"  ✗ {message}", exc_info=exc)


def warn(message: str) -> None:
    _log.warning(message)


def info(message: str) -> None:
    _log.info(message)


def debug(message: str) -> None:
    _log.debug(message)


def _arguments(arguments: dict, limit: int = 160) -> str:
    if not arguments:
        return ""
    parts = []
    for key, value in arguments.items():
        text = value if isinstance(value, str) else repr(value)
        text = " ".join(str(text).split())
        if len(text) > 60:
            text = text[:59] + "…"
        parts.append(f"{key}={text}")
    joined = ", ".join(parts)
    return joined if len(joined) <= limit else joined[:limit - 1] + "…"
