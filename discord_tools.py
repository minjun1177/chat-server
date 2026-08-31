"""Tools that only exist while the harness is talking through Discord.

Reading a channel's own history and describing the server it sits in are things
the model cannot get any other way - there is no file to read and no URL to
fetch. They are handled here so `tools.py` stays a pure local-machine toolbox.

Every handler is synchronous, because that is what the dispatch table expects,
while discord.py is async and single-loop. The bridge is
`asyncio.run_coroutine_threadsafe`, which is safe precisely because the bot
sink runs tool handlers in a worker thread.
"""

import asyncio
import json
import os
from contextvars import ContextVar
from dataclasses import dataclass, field

import config


TOOL_NAMES = (
    "discord_read_messages",
    "discord_server_info",
    "discord_user_info",
    "discord_send_file",
)

CALL_TIMEOUT = 30       # seconds to wait for one Discord API round trip


# ---------------------------------------------------------------------------
# per-turn context
# ---------------------------------------------------------------------------

@dataclass
class DiscordContext:
    """Where the current turn is happening."""
    client: object = None           # discord.Client
    loop: object = None             # the bot's event loop
    channel: object = None
    guild: object = None
    author: object = None
    workspace: str = ""
    is_user_app: bool = False       # invoked where the bot itself is not installed
    extra: dict = field(default_factory=dict)


_ctx: ContextVar[DiscordContext | None] = ContextVar("discord_ctx", default=None)


def set_context(ctx: DiscordContext):
    return _ctx.set(ctx)


def reset_context(token) -> None:
    _ctx.reset(token)


def context() -> DiscordContext | None:
    return _ctx.get()


def _run(coro, timeout: int = CALL_TIMEOUT):
    """Run a coroutine on the bot's loop from whatever thread we are on."""
    ctx = context()
    if ctx is None or ctx.loop is None:
        raise RuntimeError("no Discord context")
    future = asyncio.run_coroutine_threadsafe(coro, ctx.loop)
    return future.result(timeout)


def is_discord_tool(name: str) -> bool:
    return name in TOOL_NAMES


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------

def _channel_ref(text: str) -> str:
    """Strip the shapes Discord writes a channel in: #general, <#123…>."""
    text = (text or "").strip()
    if text.startswith("<#") and text.endswith(">"):
        text = text[2:-1]
    return text.lstrip("#").strip()


async def _resolve_channel(ctx: DiscordContext, reference: str):
    """Find a channel by name, mention or id.

    Names are what people actually type, so `discord_read_messages` takes
    "general" as readily as `123456789012345678` - an id is only what Discord
    happens to store underneath.
    """
    reference = _channel_ref(reference)
    if not reference:
        return ctx.channel

    if reference.isdigit():
        channel_id = int(reference)
        found = None
        if ctx.guild is not None:
            found = ctx.guild.get_channel(channel_id)
        if found is None:
            found = ctx.client.get_channel(channel_id)
        if found is None:
            found = await ctx.client.fetch_channel(channel_id)
        return found

    guild = ctx.guild
    if guild is None:
        raise RuntimeError(f"there is no server here to look up a channel named "
                           f"'{reference}' - pass a channel id instead")

    wanted = reference.casefold()
    readable = [c for c in guild.channels if hasattr(c, "history")]
    for match in (lambda c: c.name.casefold() == wanted,
                  lambda c: c.name.casefold().replace("-", " ") == wanted.replace("-", " "),
                  lambda c: wanted in c.name.casefold()):
        hits = [c for c in readable if match(c)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            names = ", ".join(f"#{c.name}" for c in hits[:10])
            raise RuntimeError(f"'{reference}' matches several channels ({names}) - "
                               "use the exact name")

    names = ", ".join(f"#{c.name}" for c in readable[:20]) or "(none readable)"
    raise RuntimeError(f"no channel named '{reference}' here. Readable channels: {names}")


def _embed_to_text(embed) -> str:
    """Flatten one embed, including the ones this bot wrote itself.

    Tool results are shown as embeds, so without this the model reading back a
    channel would see its own past work as blank messages.
    """
    parts = []
    if getattr(embed, "author", None) and getattr(embed.author, "name", None):
        parts.append(f"[embed author] {embed.author.name}")
    if embed.title:
        parts.append(f"[embed] {embed.title}")
    if embed.description:
        parts.append(embed.description)
    for field_ in getattr(embed, "fields", []) or []:
        name = (field_.name or "").strip()
        value = (field_.value or "").strip()
        parts.append(f"  - {name}: {value}" if name else f"  - {value}")
    if getattr(embed, "footer", None) and getattr(embed.footer, "text", None):
        parts.append(f"[embed footer] {embed.footer.text}")
    return "\n".join(p for p in parts if p)


def _message_to_text(message) -> str:
    stamp = message.created_at.strftime("%m-%d %H:%M") if message.created_at else "??"
    author = getattr(message.author, "display_name", None) or str(message.author)
    tag = " [BOT]" if getattr(message.author, "bot", False) else ""
    head = f"[{stamp}] {author}{tag} (id={message.author.id}):"

    body = []
    if message.content:
        body.append(message.content)
    for embed in message.embeds or []:
        text = _embed_to_text(embed)
        if text:
            body.append(text)
    for attachment in message.attachments or []:
        body.append(f"[attachment] {attachment.filename} ({attachment.size} bytes) {attachment.url}")
    if getattr(message, "reference", None) and message.reference.message_id:
        body.append(f"[reply to message {message.reference.message_id}]")
    if not body:
        body.append("(no text content)")

    indented = "\n".join("    " + line for chunk in body for line in chunk.split("\n"))
    return f"{head}\n{indented}"


# ---------------------------------------------------------------------------
# discord_read_messages
# ---------------------------------------------------------------------------

def handle_read_messages(limit=30, channel: str = "", before: str = "",
                         include_embeds=True) -> str:
    ctx = context()
    if ctx is None:
        return "[Error] discord_read_messages only works while the harness is running as a Discord bot."

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 30
    limit = max(1, min(limit, config.DISCORD_HISTORY_MAX))

    async def read():
        target = await _resolve_channel(ctx, channel)
        if target is None or not hasattr(target, "history"):
            raise RuntimeError(
                "this conversation has no readable channel - a user-installed "
                "app in someone else's server cannot read history")

        kwargs = {"limit": limit}
        if before:
            import discord
            kwargs["before"] = discord.Object(id=int(before))
        return [m async for m in target.history(**kwargs)], target

    try:
        messages, target = _run(read())
    except Exception as error:                      # noqa: BLE001 - reported to the model
        return _api_error("read this channel's history", error)

    messages.reverse()                              # oldest first reads better
    lines = []
    for message in messages:
        if not include_embeds and not message.content:
            continue
        lines.append(_message_to_text(message))

    if not lines:
        return "(the channel has no readable messages)"

    where = getattr(target, "name", None) or "this conversation"
    header = f"Last {len(lines)} message(s) in #{where} (oldest first):"
    body = "\n".join(lines)
    limit_chars = config.DISCORD_HISTORY_CHARS
    if len(body) > limit_chars:
        body = "...[older messages trimmed]\n" + body[-limit_chars:]
    oldest = messages[0].id if messages else ""
    footer = f"\n\n(to page further back, call again with before=\"{oldest}\")" if oldest else ""
    return f"{header}\n{body}{footer}"


# ---------------------------------------------------------------------------
# discord_server_info
# ---------------------------------------------------------------------------

def handle_server_info(server: str = "") -> str:
    ctx = context()
    if ctx is None:
        return "[Error] discord_server_info only works while the harness is running as a Discord bot."

    async def collect():
        guild = ctx.guild
        if server:
            reference = server.strip()
            if reference.isdigit():
                guild = ctx.client.get_guild(int(reference))
                if guild is None:
                    guild = await ctx.client.fetch_guild(int(reference))
            else:
                wanted = reference.casefold()
                hits = [g for g in ctx.client.guilds if g.name.casefold() == wanted] or \
                       [g for g in ctx.client.guilds if wanted in g.name.casefold()]
                if not hits:
                    known = ", ".join(g.name for g in ctx.client.guilds[:20]) or "(none)"
                    raise RuntimeError(f"no server named '{reference}'. Known servers: {known}")
                guild = hits[0]
        if guild is None:
            raise RuntimeError(
                "this conversation is not in a server (a DM, a group DM, or a "
                "server where this app is user-installed only)")

        info = {
            "name": guild.name,
            "id": str(guild.id),
            "description": guild.description or "",
            "created_at": guild.created_at.strftime("%Y-%m-%d") if guild.created_at else "",
            "member_count": guild.member_count or len(getattr(guild, "members", []) or []),
            "boost_tier": getattr(guild, "premium_tier", None),
            "boosts": getattr(guild, "premium_subscription_count", None),
            "locale": str(getattr(guild, "preferred_locale", "")),
            "features": list(getattr(guild, "features", []) or []),
        }
        owner = getattr(guild, "owner", None)
        if owner is None and getattr(guild, "owner_id", None):
            info["owner"] = f"id={guild.owner_id}"
        elif owner is not None:
            info["owner"] = f"{owner.display_name} (id={owner.id})"

        channels = {}
        for channel in getattr(guild, "channels", []) or []:
            kind = channel.__class__.__name__.replace("Channel", "").lower()
            if kind == "category":
                continue
            category = getattr(channel, "category", None)
            bucket = category.name if category else "(no category)"
            channels.setdefault(bucket, []).append(f"#{channel.name} [{kind}, id={channel.id}]")
        info["channels"] = channels

        info["roles"] = [
            f"{role.name} (id={role.id}, members={len(getattr(role, 'members', []) or [])})"
            for role in reversed(getattr(guild, "roles", []) or [])
            if role.name != "@everyone"
        ][:40]
        info["emoji_count"] = len(getattr(guild, "emojis", []) or [])
        return info

    try:
        info = _run(collect())
    except Exception as error:                      # noqa: BLE001
        return _api_error("read this server's information", error)

    return json.dumps(info, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# discord_user_info
# ---------------------------------------------------------------------------

def handle_user_info(user_id: str = "") -> str:
    ctx = context()
    if ctx is None:
        return "[Error] discord_user_info only works while the harness is running as a Discord bot."

    async def collect():
        target = ctx.author
        if user_id:
            uid = int(str(user_id).strip("<@!>"))
            target = None
            if ctx.guild is not None:
                target = ctx.guild.get_member(uid)
            if target is None:
                target = ctx.client.get_user(uid) or await ctx.client.fetch_user(uid)
        if target is None:
            raise RuntimeError("no such user is visible from here")

        info = {
            "name": str(target),
            "display_name": getattr(target, "display_name", str(target)),
            "id": str(target.id),
            "bot": bool(getattr(target, "bot", False)),
            "account_created": target.created_at.strftime("%Y-%m-%d") if target.created_at else "",
        }
        joined = getattr(target, "joined_at", None)
        if joined:
            info["joined_server"] = joined.strftime("%Y-%m-%d")
        roles = getattr(target, "roles", None)
        if roles:
            info["roles"] = [r.name for r in roles if r.name != "@everyone"]
        return info

    try:
        info = _run(collect())
    except Exception as error:                      # noqa: BLE001
        return _api_error("read that user's information", error)

    return json.dumps(info, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# discord_send_file
# ---------------------------------------------------------------------------

def handle_send_file(path: str, comment: str = "") -> str:
    ctx = context()
    if ctx is None:
        return "[Error] discord_send_file only works while the harness is running as a Discord bot."
    if not path:
        return "[Error] No path was given."

    target = os.path.abspath(path)
    workspace = os.path.abspath(ctx.workspace or config.DISCORD_WORKSPACE)
    if os.path.commonpath([target, workspace]) != workspace:
        return (f"[Error] Only files under {workspace} can be uploaded. "
                "Write the file there first, then send it.")
    if not os.path.isfile(target):
        return f"[Error] File not found: {target}"
    size = os.path.getsize(target)
    if size > 8 * 1024 * 1024:
        return f"[Error] {os.path.basename(target)} is {size} bytes; Discord's plain upload limit is 8MB."

    async def send():
        import discord
        await ctx.channel.send(content=comment or None,
                               file=discord.File(target, filename=os.path.basename(target)))

    try:
        _run(send(), timeout=120)
    except Exception as error:                      # noqa: BLE001
        return _api_error("upload that file", error)
    return f"[System] Uploaded {os.path.basename(target)} ({size} bytes) to the channel."


def _api_error(what: str, error: Exception) -> str:
    name = error.__class__.__name__
    if name == "Forbidden":
        return f"[Error] The bot is not allowed to {what} here (missing Discord permission)."
    if name in ("NotFound", "ValueError"):
        return f"[Error] Could not {what}: {error}"
    if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
        return f"[Error] Discord did not answer in time while trying to {what}."
    return f"[Error] Could not {what}: {name}: {error}"


# ---------------------------------------------------------------------------
# dispatch + prompt
# ---------------------------------------------------------------------------

def dispatch(name: str, arguments: dict) -> str | None:
    if name == "discord_read_messages":
        return handle_read_messages(
            arguments.get("limit", 30),
            str(arguments.get("channel", "") or arguments.get("channel_name", "")
                or arguments.get("channel_id", "") or ""),
            str(arguments.get("before", "") or ""),
            arguments.get("include_embeds", True) not in (False, "false", "False"),
        )
    if name == "discord_server_info":
        return handle_server_info(str(arguments.get("server", "") or arguments.get("guild", "")
                                      or arguments.get("guild_id", "")
                                      or arguments.get("server_id", "") or ""))
    if name == "discord_user_info":
        return handle_user_info(str(arguments.get("user_id", "") or arguments.get("user", "") or ""))
    if name == "discord_send_file":
        return handle_send_file(arguments.get("path", "") or arguments.get("filepath", ""),
                                arguments.get("comment", ""))
    return None


_ENTRIES = [
    {
        "name": "discord_read_messages",
        "description": (
            "Read recent messages from the current Discord channel, oldest first. "
            "Embeds are included and flattened to text - that is how you read back your "
            "own tool-result cards and anything another bot posted. Use it whenever the "
            "user refers to what was said earlier ('summarise the last hour', "
            "'what did they decide?'), "
            "because your conversation history holds only the turns addressed to you."),
        "parameters": {
            "limit": "How many messages to read (default 30, maximum 100).",
            "channel": "(Optional) Another channel, by NAME as the user says it (\"general\", \"#dev-talk\") or by id. Omit for the current channel.",
            "before": "(Optional) A message id; reads the messages posted before it, for paging further back.",
            "include_embeds": "(Optional) false to skip messages that carry only embeds. Default true.",
        },
    },
    {
        "name": "discord_server_info",
        "description": (
            "Describe the Discord server this conversation is in: name, owner, creation "
            "date, member count, boost tier, features, the channel list grouped by "
            "category, roles and emoji count. Returns JSON."),
        "parameters": {
            "server": "(Optional) Another server, by name or id. Omit for the current one.",
        },
    },
    {
        "name": "discord_user_info",
        "description": (
            "Look up a Discord user: display name, id, account creation date, when they "
            "joined this server and which roles they hold. Omit user_id for whoever is "
            "talking to you."),
        "parameters": {
            "user_id": "(Optional) The user id or mention. Omit for the current speaker.",
        },
    },
    {
        "name": "discord_send_file",
        "description": (
            "Upload a file from this conversation's workspace folder to the channel as an "
            "attachment. Use it when you have produced a file the user should be able to "
            "download. Files outside the workspace folder are refused."),
        "parameters": {
            "path": "Path of the file to upload. It must be inside the conversation's workspace folder.",
            "comment": "(Optional) A short message to send with the file.",
        },
    },
]


def discord_tools_prompt(allowed: set[str] | None = None) -> str:
    """The Discord section of the system prompt. Only added when running as a bot."""
    entries = [e for e in _ENTRIES if allowed is None or e["name"] in allowed]
    if not entries:
        return ""
    return "\n".join([
        "\n### DISCORD TOOLS:",
        "You are talking through Discord. These tools read the Discord side of the",
        "conversation - use them instead of guessing what a channel or server contains.",
        "",
        json.dumps(entries, indent=2, ensure_ascii=False),
        "",
        "### DISCORD REPLY STYLE:",
        "- Discord messages are capped at 2000 characters and long answers get split, so",
        "  be concise and put the answer first.",
        "- Discord markdown supports **bold**, *italic*, `code`, ```code blocks```, lists",
        "  and > quotes. It has no tables and no headings larger than ###; do not draw",
        "  ASCII tables, use short lists instead.",
        "- Mention someone as <@user_id> only when you actually need their attention.",
        "- Never paste a whole file into a message; post the relevant lines instead."
        + (" When the user should be able to download it, write the file into the\n"
           "  workspace folder and send it with discord_send_file."
           if any(e["name"] == "discord_send_file" for e in entries) else ""),
        "",
    ])
