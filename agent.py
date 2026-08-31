"""Conversations, one per place the bot is talking.

`app.py` keeps its single `messages` list as a local and stores everything else
in `config` globals. That is fine for one terminal and wrong for a bot, where
three channels can be mid-turn at once. An `AgentSession` holds what used to be
global - the history, the loaded skills, the title, the token counters - and
`run_turn` swaps them into `config` for the duration of a turn, under a lock
that keeps one channel's turns in order.

The heavy lifting is still `llm_client.chat_turn`; nothing here re-implements
the agent loop.
"""

import asyncio
import contextlib
import os
import re
import time
from dataclasses import dataclass, field

import config
import session as session_store
import ui
from context import manage_context
from llm_client import chat_turn
from session import (generate_session_title, load_session, save_session,
                     slugify_title)
from systemprompt import systemprompt as build_system_prompt


_SUMMARY = re.compile(r'\n\n<SUMMARY>(.*?)</SUMMARY>', re.DOTALL)


def compose_system_prompt(tool_filter=None, discord: bool = False, summary: str = "") -> str:
    base = build_system_prompt(tool_filter=tool_filter, discord=discord)
    if config.CUSTOM_PERSONA:
        base = config.CUSTOM_PERSONA + "\n\n" + base
    return base + summary


def extract_summary(system_content: str) -> str:
    """Keep a compression summary across a system-prompt rebuild."""
    match = _SUMMARY.search(system_content or "")
    return f"\n\n<SUMMARY>{match.group(1)}</SUMMARY>" if match else ""


@dataclass
class AgentSession:
    """One thread of conversation, wherever it is happening."""

    key: str
    messages: list = field(default_factory=list)
    session_id: str = ""
    title: str = ""
    loaded_skills: list = field(default_factory=list)
    token_history: list = field(default_factory=list)
    memory_file: str = ""
    session_dir: str = ""
    workspace: str = ""
    tool_filter: set | None = None
    discord: bool = False
    last_used: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    tasks: set = field(default_factory=set)
    meta: dict = field(default_factory=dict)

    # -- prompt ------------------------------------------------------------

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt()}]
        self.loaded_skills = []
        self.token_history = []
        self.session_id = ""
        self.title = ""

    def system_prompt(self, summary: str = "") -> str:
        return compose_system_prompt(self.tool_filter, self.discord, summary)

    def refresh_system_prompt(self) -> None:
        """Rebuild the system message in place, keeping persona and summary."""
        if not self.messages:
            self.reset()
            return
        summary = extract_summary(self.messages[0].get("content", ""))
        self.messages[0]["content"] = self.system_prompt(summary)

    @property
    def busy(self) -> bool:
        return self.lock.locked()

    def track(self, task) -> None:
        """Remember a running turn so it can be cancelled later."""
        self.tasks = {t for t in self.tasks if not t.done()}
        self.tasks.add(task)

    def running(self) -> list:
        return [t for t in self.tasks if not t.done()]

    def cancel(self) -> int:
        """Cancel every turn in flight. Returns how many were stopped."""
        live = self.running()
        for task in live:
            task.cancel()
        return len(live)

    def token_total(self) -> int:
        return sum(t.get("prompt", 0) + t.get("completion", 0) for t in self.token_history)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

_sessions: dict[str, AgentSession] = {}
_turn_slots: asyncio.Semaphore | None = None


def _slots() -> asyncio.Semaphore:
    global _turn_slots
    if _turn_slots is None:
        _turn_slots = asyncio.Semaphore(max(1, config.DISCORD_MAX_CONCURRENT_TURNS))
    return _turn_slots


def get(key: str) -> AgentSession | None:
    return _sessions.get(key)


def get_or_create(key: str, **fields) -> AgentSession:
    sess = _sessions.get(key)
    if sess is None:
        sess = AgentSession(key=key, **fields)
        sess.reset()
        restore(sess)
        _sessions[key] = sess
    sess.last_used = time.time()
    return sess


def drop(key: str) -> None:
    _sessions.pop(key, None)


def all_sessions() -> list[AgentSession]:
    return list(_sessions.values())


def prune(max_idle: int | None = None) -> None:
    """Forget conversations nobody has touched in a while. Files stay on disk."""
    limit = config.DISCORD_SESSION_IDLE if max_idle is None else max_idle
    cutoff = time.time() - limit
    for key, sess in list(_sessions.items()):
        if sess.last_used < cutoff and not sess.busy:
            _sessions.pop(key, None)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def bound(sess: AgentSession, memory_file: str = ""):
    """Make `sess` the current conversation for everything that reads globals.

    `config` scopes the three per-conversation settings to whichever task is
    asking, so two channels streaming at the same time cannot swap titles,
    loaded skills or token counts. The memory file and the session folder are
    redirected the same way.

    `memory_file` overrides the session's own for one turn: a channel's history
    is shared, but memories belong to whoever is speaking, and the speaker can
    change between the moment a turn is queued and the moment it runs.
    """
    store_token = session_store.use_store(memory_file or sess.memory_file,
                                          sess.session_dir)
    scope_token = config.push_conversation({
        "LOADED_SKILLS": sess.loaded_skills,
        "SESSION_TITLE": sess.title,
        "token_history": sess.token_history,
    })
    try:
        yield sess
    finally:
        state = config.conversation_state()
        sess.loaded_skills = state.get("LOADED_SKILLS", sess.loaded_skills)
        sess.title = state.get("SESSION_TITLE", sess.title)
        sess.token_history = state.get("token_history", sess.token_history)
        config.pop_conversation(scope_token)
        session_store.reset_store(store_token)


def _file_id(sess: AgentSession) -> str:
    return slugify_title(sess.key.replace(":", "-")) or "conversation"


def restore(sess: AgentSession) -> bool:
    """Reload a conversation saved before the bot last restarted."""
    if not config.SAVE_CHAT_HISTORY or not sess.session_dir:
        return False
    with bound(sess):
        data = load_session(_file_id(sess))
    if not data:
        return False
    messages = data.get("messages") if isinstance(data, dict) else data
    if not isinstance(messages, list) or not messages:
        return False

    sess.messages = messages
    sess.session_id = _file_id(sess)
    if isinstance(data, dict):
        sess.title = data.get("title", "") or ""
        sess.token_history = data.get("token_history", []) or []
    # The tool set, the MCP servers and the skills may all have changed while
    # the bot was down, so the stored system prompt is not to be trusted.
    if sess.messages and sess.messages[0].get("role") == "system":
        sess.refresh_system_prompt()
    else:
        sess.messages.insert(0, {"role": "system", "content": sess.system_prompt()})
    # Skills are re-declared by the messages themselves; manage_context prunes
    # the list on the next turn, so start it empty rather than stale.
    sess.loaded_skills = []
    return True


def _save_bound(sess: AgentSession) -> None:
    """Write the transcript. Only valid inside `bound(sess)`."""
    if not config.SAVE_CHAT_HISTORY or not sess.session_dir:
        return
    save_session(sess.messages, _file_id(sess))
    sess.session_id = _file_id(sess)


def persist(sess: AgentSession) -> None:
    """Write the transcript from outside a turn."""
    if not config.SAVE_CHAT_HISTORY or not sess.session_dir:
        return
    with bound(sess):
        _save_bound(sess)


# ---------------------------------------------------------------------------
# running a turn
# ---------------------------------------------------------------------------

async def run_turn(sess: AgentSession, user_text: str, sink: ui.Sink,
                   autotitle: bool = True, memory_file: str = "") -> str:
    """One user message through the agent loop, with `sink` receiving output."""
    ui_token = ui.set_current(sink)
    try:
        async with sess.lock:
            async with _slots():
                sess.last_used = time.time()
                with bound(sess, memory_file):
                    try:
                        sess.messages.append({"role": "user",
                                              "content": config.safe_text(user_text)})
                        config.repair_messages(sess.messages)
                        await manage_context(sess.messages)

                        reply = await chat_turn(sess.messages)

                        if autotitle and config.AUTO_TITLE and not config.SESSION_TITLE:
                            title = await generate_session_title(sess.messages)
                            if title:
                                config.SESSION_TITLE = title
                        return reply
                    finally:
                        # Saving belongs inside `bound`, where the title and the
                        # session folder are this conversation's - and it has to
                        # happen even when the turn was stopped half way, or the
                        # exchange would be lost on the next restart.
                        sess.last_used = time.time()
                        _save_bound(sess)
    except asyncio.CancelledError:
        sink.aborted = True
        raise
    finally:
        # A cancelled turn still has to put its stop button away and stop the
        # typing indicator, and every await inside a cancelled task raises
        # again - so closing runs as its own task and is only shielded here.
        closing = asyncio.ensure_future(sink.close())
        try:
            await asyncio.shield(closing)
        except asyncio.CancelledError:
            pass                                # it finishes on its own
        except Exception:                       # noqa: BLE001 - never mask the turn's own error
            pass
        ui.reset(ui_token)


def workspace_for(sess: AgentSession) -> str:
    path = sess.workspace or os.path.join(config.DISCORD_WORKSPACE, _file_id(sess))
    os.makedirs(path, exist_ok=True)
    return path
