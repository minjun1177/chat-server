"""The seam between the agent loop and whoever is watching it.

The loop does not print and does not call `input()`. It hands what it produces
to a `Sink` and asks its questions through one, which is what lets the same
`chat_turn` stream into a Discord message and take its approvals from buttons.

`DiscordSink` (in `discord_ui.py`) is the one that matters. `ConsoleSink` is the
fallback for code that runs with no conversation bound - it writes to the bot's
log and refuses every question, because there is nobody at a keyboard to answer
one and a silent "yes" is the wrong default for a tool that can delete files.
"""

from contextvars import ContextVar

import config
import logs


# ---------------------------------------------------------------------------
# tool policy
# ---------------------------------------------------------------------------

class ToolPolicy:
    """Which tools this conversation may use at all.

    Distinct from `permissions.py`: that decides allow/ask/deny for a *call*,
    this decides whether the tool exists for this actor. A blocked tool is never
    dispatched and is left out of the system prompt, so the model does not spend
    turns calling something it cannot have.
    """

    def __init__(self, allowed: set[str] | None = None, label: str = ""):
        self.allowed = allowed          # None = everything
        self.label = label

    def permits(self, name: str) -> bool:
        if self.allowed is None:
            return True
        if name in self.allowed:
            return True
        # MCP tools are allowed as a family: "mcp__*" covers every server.
        if name.startswith("mcp__") and "mcp__*" in self.allowed:
            return True
        return False

    def refusal(self, name: str) -> str:
        where = f" ({self.label})" if self.label else ""
        return (f"[System] The tool '{name}' is not available in this conversation{where}. "
                "Do not retry it; tell the user it is unavailable here and continue with "
                "the tools you do have.")


ALLOW_ALL = ToolPolicy()


# ---------------------------------------------------------------------------
# the sink
# ---------------------------------------------------------------------------

class Sink:
    """Where a turn's output goes and where its questions are answered.

    The streaming half is async and runs on the event loop. The asking half is
    synchronous, because it is called from inside tool handlers - which a bot
    runs in a worker thread, and the terminal runs inline.
    """

    tool_policy: ToolPolicy = ALLOW_ALL
    show_thinking = None            # None = follow config.SHOW_THINKING
    aborted = False                 # the user stopped this turn
    tools_run = 0                   # how many tools this exchange has called

    # -- streaming ---------------------------------------------------------
    async def turn_start(self) -> None: ...
    async def text(self, delta: str) -> None: ...
    async def thinking(self, delta: str) -> None: ...
    async def mark_tool_call(self) -> None: ...
    async def turn_end(self, prompt_tokens: int, completion_tokens: int,
                       total_seconds: float, eval_seconds: float) -> None: ...

    async def close(self) -> None:
        """The whole exchange is over.

        `turn_end` fires once per streamed reply, and a turn that calls three
        tools streams four times. Anything that should happen once - retiring a
        stop button, stopping a typing indicator - belongs here.
        """

    # -- tools -------------------------------------------------------------
    async def tool_call(self, name: str, arguments: dict): return None
    async def tool_result(self, handle, name: str, result: str) -> None: ...

    async def run_tool(self, fn, name: str, arguments: dict):
        """Run one (synchronous) tool handler. Inline here; threaded for a bot."""
        return fn(name, arguments)

    async def notice(self, text: str, level: str = "info") -> None: ...

    # -- asking (synchronous: called from tool code) ------------------------
    def ask_approval(self, label: str, details: list[tuple[str, str]], rule: str = "") -> bool:
        return False

    def ask_choice(self, question: str, options: list[str], index: int, total: int) -> str | None:
        return None

    def ask_plan(self, context_discovered: str, diff_blueprint: str, verification_steps: str) -> str:
        return "[System] Plan approval is not available in this context."

    def confirm(self, question: str) -> bool:
        return False


# ---------------------------------------------------------------------------
# the terminal sink - the original behaviour, moved
# ---------------------------------------------------------------------------

class ConsoleSink(Sink):
    """Log the turn; refuse anything that needs a person.

    Nothing should normally run under this - every turn binds a `DiscordSink` -
    but a tool reached with no sink bound must still behave, and behaving means
    saying no.
    """

    def __init__(self, label: str = "console"):
        self.label = label
        self._line = ""

    # -- streaming ---------------------------------------------------------

    async def text(self, delta: str) -> None:
        self._line += delta
        while "\n" in self._line:
            line, self._line = self._line.split("\n", 1)
            if line.strip():
                logs.debug(f"  {line}")

    async def turn_end(self, prompt_tokens: int, completion_tokens: int,
                       total_seconds: float, eval_seconds: float) -> None:
        if self._line.strip():
            logs.debug(f"  {self._line}")
        self._line = ""
        logs.debug(f"  tokens: {prompt_tokens} in · {completion_tokens} out · "
                   f"{total_seconds:.1f}s")

    # -- tools -------------------------------------------------------------

    async def tool_call(self, name: str, arguments: dict):
        logs.tool_call(name, arguments)
        return None

    async def tool_result(self, handle, name: str, result: str) -> None:
        failed = (result or "").lstrip().startswith("[Error]")
        logs.tool_result(name, not failed, len(result or ""), 0.0)

    async def notice(self, text: str, level: str = "info") -> None:
        (logs.warn if level in ("warn", "error") else logs.info)(text)

    # -- asking: there is nobody to ask ------------------------------------

    def ask_approval(self, label: str, details: list[tuple[str, str]], rule: str = "") -> bool:
        logs.approval(label, "nobody (unattended)", False, rule)
        return False

    def ask_choice(self, question: str, options: list[str], index: int, total: int) -> str | None:
        logs.warn(f"a question had no one to answer it: {question}")
        return None

    def ask_plan(self, context_discovered: str, diff_blueprint: str,
                 verification_steps: str) -> str:
        return "[System] Plan approval is not available in this context."

    def confirm(self, question: str) -> bool:
        return False


def _preview(arguments: dict, limit: int = 120) -> str:
    if not arguments:
        return ""
    text = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    return text if len(text) <= limit else text[:limit - 1] + "…"


# ---------------------------------------------------------------------------
# the approval seam
# ---------------------------------------------------------------------------

def approval_prompt(action_label: str, details: list[tuple[str, str]], rule: str = "") -> bool:
    """Ask whoever is driving this conversation whether a call may run.

    Every tool that needs consent calls this. Where the question is actually
    asked is the sink's business - on Discord it is three buttons under a
    message, and nobody clicking counts as a refusal.
    """
    # A permission rule already said yes, or /automode is on.
    if config.AUTO_ALLOW or config.policy_auto_allow():
        return True
    return current().ask_approval(action_label, details, rule)


# ---------------------------------------------------------------------------
# the active sink
# ---------------------------------------------------------------------------

_default = ConsoleSink()
_current: ContextVar[Sink] = ContextVar("ui_sink", default=_default)


def current() -> Sink:
    return _current.get()


def set_current(sink: Sink):
    """Bind a sink for this task. Returns the token `reset()` takes."""
    return _current.set(sink)


def reset(token) -> None:
    _current.reset(token)


def policy() -> ToolPolicy:
    return current().tool_policy or ALLOW_ALL
