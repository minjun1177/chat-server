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
from datetime import datetime, timezone

import config


TOOL_NAMES = (
    "discord_read_messages",
    "discord_server_info",
    "discord_user_info",
    "discord_send_file",
    # Admin actions - the bot must itself hold the matching Discord permission.
    "discord_list_events",
    "discord_create_event",
    "discord_cancel_event",
    "discord_manage_role",
    "discord_create_channel",
    "discord_edit_event",
    "discord_kick_member",
    "discord_ban_member",
    "discord_unban_member",
    "discord_list_bans",
    "discord_timeout_member",
    "discord_set_nickname",
    "discord_move_member",
    "discord_purge_messages",
    "discord_delete_message",
    "discord_pin_message",
    "discord_edit_channel",
    "discord_delete_channel",
    "discord_lock_channel",
    "discord_create_role",
    "discord_edit_role",
    "discord_delete_role",
    "discord_create_invite",
    "discord_audit_log",
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
    # Built at the start of every turn, so this marks where the bot's own live
    # cards begin - bulk deletes stop here instead of eating them mid-turn.
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


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


def _parse_time(value: str) -> datetime:
    """Turn a human-written timestamp into a timezone-aware UTC datetime.

    Discord stores scheduled events as absolute instants, so a bare local time
    is ambiguous. We accept ISO 8601 - "2026-09-20T18:00", "2026-09-20 18:00",
    or with an offset like "2026-09-20T18:00+09:00" - and treat anything without
    an offset as UTC, which is what the model is told in the tool description.
    """
    text = (value or "").strip().replace("Z", "+00:00")
    if not text:
        raise ValueError("no time was given")
    try:
        when = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(
            f"could not read '{value}' as a time - use ISO 8601 like "
            "2026-09-20T18:00 or 2026-09-20T18:00+09:00")
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when


async def _resolve_voice_channel(ctx: DiscordContext, reference: str):
    """Find a voice or stage channel by id or name, for scheduled events."""
    reference = _channel_ref(reference)
    guild = ctx.guild
    if guild is None:
        raise RuntimeError("there is no server here to look up a channel in")

    pool = list(getattr(guild, "voice_channels", []) or []) + \
        list(getattr(guild, "stage_channels", []) or [])
    if reference.isdigit():
        wanted_id = int(reference)
        for channel in pool:
            if channel.id == wanted_id:
                return channel
        raise RuntimeError(f"no voice or stage channel with id {reference} here")

    wanted = reference.casefold()
    hits = [c for c in pool if c.name.casefold() == wanted] or \
        [c for c in pool if wanted in c.name.casefold()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        names = ", ".join(c.name for c in hits[:10])
        raise RuntimeError(f"'{reference}' matches several voice channels ({names}) - use the exact name")
    names = ", ".join(c.name for c in pool[:20]) or "(none)"
    raise RuntimeError(f"no voice or stage channel named '{reference}' here. Voice channels: {names}")


def _resolve_role(guild, reference: str):
    """Find a role by id or name."""
    reference = (reference or "").strip()
    if reference.startswith("<@&") and reference.endswith(">"):
        reference = reference[3:-1]
    if not reference:
        raise RuntimeError("no role was given")
    roles = [r for r in getattr(guild, "roles", []) or [] if r.name != "@everyone"]
    if reference.isdigit():
        wanted_id = int(reference)
        for role in roles:
            if role.id == wanted_id:
                return role
        raise RuntimeError(f"no role with id {reference} here")
    wanted = reference.casefold()
    hits = [r for r in roles if r.name.casefold() == wanted] or \
        [r for r in roles if wanted in r.name.casefold()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        names = ", ".join(r.name for r in hits[:10])
        raise RuntimeError(f"'{reference}' matches several roles ({names}) - use the exact name")
    names = ", ".join(r.name for r in roles[:20]) or "(none)"
    raise RuntimeError(f"no role named '{reference}' here. Roles: {names}")


def _snowflake(value, what: str = "id") -> int:
    """Read a Discord id out of a raw id or a <@…>/<#…>/<@&…> mention."""
    text = str(value or "").strip().strip("<>").lstrip("@#&!")
    if not text.isdigit():
        raise ValueError(f"'{value}' is not a valid {what} - give the numeric id or a mention")
    return int(text)


async def _fetch_member(ctx: DiscordContext, user_id: str):
    """Resolve a guild member by id or mention, via REST so no member cache is needed."""
    uid = _snowflake(user_id, "user id")
    member = ctx.guild.get_member(uid) if ctx.guild is not None else None
    if member is None:
        member = await ctx.guild.fetch_member(uid)
    return member


def _check_target(ctx: DiscordContext, member, verb: str) -> None:
    """Refuse moderation Discord would reject anyway, or that nobody should automate.

    The model reads channel history, so a message can try to talk it into
    turning on someone. The server owner, the harness owners and the bot itself
    are therefore never targets, whatever the conversation says.
    """
    guild = ctx.guild
    if member.id == getattr(ctx.client.user, "id", None):
        raise ValueError(f"the bot will not {verb} itself")
    if member.id == guild.owner_id:
        raise ValueError(f"cannot {verb} the server owner")
    if member.id in config.DISCORD_OWNER_IDS:
        raise ValueError(f"cannot {verb} {member.display_name} - they are an owner of this harness")
    me = guild.me
    top = getattr(member, "top_role", None)
    if me is not None and top is not None and top >= me.top_role:
        raise ValueError(
            f"cannot {verb} {member.display_name}: their top role ({top.name}) is not "
            f"below the bot's ({me.top_role.name})")


def _check_role(ctx: DiscordContext, role, verb: str) -> None:
    if role.managed:
        raise ValueError(f"cannot {verb} {role.name} - it is managed by an integration or bot")
    me = ctx.guild.me
    if me is not None and role >= me.top_role:
        raise ValueError(f"cannot {verb} {role.name}: it is not below the bot's top role ({me.top_role.name})")


async def _guild_channel(ctx: DiscordContext, reference: str):
    """`_resolve_channel`, but only ever a channel in the current server."""
    channel = await _resolve_channel(ctx, reference)
    if channel is None or getattr(getattr(channel, "guild", None), "id", None) != ctx.guild.id:
        raise ValueError("that channel is not in this server")
    return channel


async def _find_event(ctx: DiscordContext, reference: str):
    reference = (reference or "").strip()
    if not reference:
        raise ValueError("name or id of the event is required")
    events = await ctx.guild.fetch_scheduled_events()
    if reference.isdigit():
        match = next((e for e in events if e.id == int(reference)), None)
    else:
        wanted = reference.casefold()
        hits = [e for e in events if e.name.casefold() == wanted] or \
            [e for e in events if wanted in e.name.casefold()]
        if len(hits) > 1:
            names = ", ".join(e.name for e in hits[:10])
            raise ValueError(f"'{reference}' matches several events ({names}) - use the id")
        match = hits[0] if hits else None
    if match is None:
        raise ValueError(f"no scheduled event matching '{reference}'")
    return match


def _flag(value, default=None):
    """Read a yes/no argument; None when it was not given at all."""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _colour(value: str):
    import discord
    text = (value or "").strip().lstrip("#")
    try:
        return discord.Colour(int(text, 16))
    except ValueError:
        raise ValueError(f"'{value}' is not a hex colour like #5865F2")


def _reason(ctx: DiscordContext, detail: str = "") -> str:
    """Audit-log reason attached to every admin action the bot takes."""
    who = getattr(ctx.author, "display_name", None) or str(getattr(ctx.author, "id", "?"))
    base = f"Requested by {who} through the AI harness"
    return f"{base}: {detail}"[:512] if detail else base


def _guild_action(tool: str, what: str, build) -> str:
    """Run one admin action: check we are in a server, run it, word the failure.

    `build(ctx)` returns the coroutine to run on the bot's loop, which returns
    the result text. ValueError/RuntimeError carry messages written for the
    model; anything else is a Discord failure and goes through `_api_error`.
    """
    ctx = context()
    if ctx is None:
        return f"[Error] {tool} only works while the harness is running as a Discord bot."
    if ctx.guild is None:
        return "[Error] There is no server here - this is a server-admin action."
    try:
        return _run(build(ctx))
    except (ValueError, RuntimeError) as error:
        return f"[Error] {error}"
    except Exception as error:                      # noqa: BLE001
        return _api_error(what, error)


def _event_line(event) -> str:
    start = event.start_time.strftime("%Y-%m-%d %H:%M UTC") if event.start_time else "?"
    where = event.location or (f"#{event.channel.name}" if event.channel else "?")
    guild_id = getattr(getattr(event, "guild", None), "id", None)
    url = f"https://discord.com/events/{guild_id}/{event.id}" if guild_id else ""
    count = getattr(event, "user_count", None)
    tail = f" · {count} interested" if count is not None else ""
    return f"- {event.name} (id={event.id}) — {start} @ {where}{tail}\n  {url}"


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


# ---------------------------------------------------------------------------
# admin actions (the bot account must hold the matching Discord permission)
# ---------------------------------------------------------------------------

def handle_list_events() -> str:
    ctx = context()
    if ctx is None:
        return "[Error] discord_list_events only works while the harness is running as a Discord bot."
    if ctx.guild is None:
        return "[Error] There is no server here - scheduled events live in a server."

    async def collect():
        return await ctx.guild.fetch_scheduled_events()

    try:
        events = _run(collect())
    except Exception as error:                      # noqa: BLE001
        return _api_error("list this server's scheduled events", error)

    if not events:
        return "(this server has no scheduled events)"
    events = sorted(events, key=lambda e: e.start_time or datetime.now(timezone.utc))
    lines = "\n".join(_event_line(e) for e in events)
    return f"{len(events)} scheduled event(s) in {ctx.guild.name}:\n{lines}"


def handle_create_event(name: str, start_time: str, end_time: str = "",
                        location: str = "", channel: str = "",
                        description: str = "") -> str:
    ctx = context()
    if ctx is None:
        return "[Error] discord_create_event only works while the harness is running as a Discord bot."
    if ctx.guild is None:
        return "[Error] There is no server here - scheduled events live in a server."
    if not (name or "").strip():
        return "[Error] An event needs a name."

    try:
        start = _parse_time(start_time)
    except ValueError as error:
        return f"[Error] start_time: {error}"
    end = None
    if end_time:
        try:
            end = _parse_time(end_time)
        except ValueError as error:
            return f"[Error] end_time: {error}"

    async def create():
        import discord
        kwargs = {
            "name": name.strip(),
            "start_time": start,
            "privacy_level": discord.PrivacyLevel.guild_only,
            "reason": _reason(ctx),
        }
        if description:
            kwargs["description"] = description
        if channel:
            target = await _resolve_voice_channel(ctx, channel)
            kwargs["channel"] = target
            kwargs["entity_type"] = (
                discord.EntityType.stage_instance
                if target.__class__.__name__ == "StageChannel"
                else discord.EntityType.voice)
            if end is not None:
                kwargs["end_time"] = end
        elif location:
            # External events must carry both a location and an end time.
            if end is None:
                raise ValueError("an event at a location needs an end_time")
            kwargs["entity_type"] = discord.EntityType.external
            kwargs["location"] = location
            kwargs["end_time"] = end
        else:
            raise ValueError("give either a location (an external event) or a "
                             "channel (a voice/stage event)")
        return await ctx.guild.create_scheduled_event(**kwargs)

    try:
        event = _run(create())
    except ValueError as error:
        return f"[Error] {error}"
    except Exception as error:                      # noqa: BLE001
        return _api_error("create that scheduled event", error)

    url = f"https://discord.com/events/{ctx.guild.id}/{event.id}"
    return (f"[System] Created scheduled event \"{event.name}\" (id={event.id}).\n"
            f"Share this link so members can mark interest: {url}")


def handle_cancel_event(event: str) -> str:
    async def cancel(ctx):
        match = await _find_event(ctx, event)
        await match.delete(reason=_reason(ctx))
        return f"[System] Cancelled scheduled event \"{match.name}\"."

    return _guild_action("discord_cancel_event", "cancel that scheduled event", cancel)


def handle_manage_role(user_id: str, role: str, action: str = "add") -> str:
    ctx = context()
    if ctx is None:
        return "[Error] discord_manage_role only works while the harness is running as a Discord bot."
    if ctx.guild is None:
        return "[Error] There is no server here - roles live in a server."
    action = (action or "add").strip().lower()
    if action not in ("add", "remove"):
        return "[Error] action must be \"add\" or \"remove\"."
    if not (user_id or "").strip():
        return "[Error] A user id or mention is required."

    async def apply():
        member = await _fetch_member(ctx, user_id)
        target_role = _resolve_role(ctx.guild, role)
        _check_role(ctx, target_role, "assign" if action == "add" else "remove")
        if action == "add":
            await member.add_roles(target_role, reason=_reason(ctx))
        else:
            await member.remove_roles(target_role, reason=_reason(ctx))
        return member.display_name, target_role.name

    try:
        who, role_name = _run(apply())
    except (ValueError, RuntimeError) as error:
        return f"[Error] {error}"
    except Exception as error:                      # noqa: BLE001
        return _api_error("change that member's roles", error)
    verb = "Gave" if action == "add" else "Removed"
    join = "to" if action == "add" else "from"
    return f"[System] {verb} role \"{role_name}\" {join} {who}."


def handle_create_channel(name: str, kind: str = "text", category: str = "",
                          topic: str = "") -> str:
    ctx = context()
    if ctx is None:
        return "[Error] discord_create_channel only works while the harness is running as a Discord bot."
    if ctx.guild is None:
        return "[Error] There is no server here - channels live in a server."
    if not (name or "").strip():
        return "[Error] A channel name is required."
    kind = (kind or "text").strip().lower()
    if kind not in ("text", "voice"):
        return "[Error] kind must be \"text\" or \"voice\"."

    async def create():
        parent = None
        if category:
            wanted = category.strip().casefold()
            cats = getattr(ctx.guild, "categories", []) or []
            hits = ([c for c in cats if str(c.id) == category.strip()]
                    or [c for c in cats if c.name.casefold() == wanted]
                    or [c for c in cats if wanted in c.name.casefold()])
            if not hits:
                names = ", ".join(c.name for c in cats[:20]) or "(none)"
                raise ValueError(f"no category named '{category}'. Categories: {names}")
            parent = hits[0]
        reason = _reason(ctx)
        if kind == "voice":
            return await ctx.guild.create_voice_channel(
                name.strip(), category=parent, reason=reason)
        kwargs = {"category": parent, "reason": reason}
        if topic:
            kwargs["topic"] = topic
        return await ctx.guild.create_text_channel(name.strip(), **kwargs)

    try:
        channel = _run(create())
    except ValueError as error:
        return f"[Error] {error}"
    except Exception as error:                      # noqa: BLE001
        return _api_error("create that channel", error)
    return f"[System] Created {kind} channel #{channel.name} (id={channel.id})."


# --- events ----------------------------------------------------------------

EVENT_STATUSES = {"start": "active", "end": "completed", "cancel": "cancelled"}


def handle_edit_event(event: str, name: str = "", description: str = "",
                      start_time: str = "", end_time: str = "",
                      location: str = "", status: str = "") -> str:
    async def edit(ctx):
        import discord
        match = await _find_event(ctx, event)
        kwargs = {}
        if name:
            kwargs["name"] = name.strip()
        if description:
            kwargs["description"] = description
        if start_time:
            kwargs["start_time"] = _parse_time(start_time)
        if end_time:
            kwargs["end_time"] = _parse_time(end_time)
        if location:
            if match.entity_type != discord.EntityType.external:
                raise ValueError("only an event at an external location has a location - "
                                 "cancel it and create a new one to change its kind")
            kwargs["location"] = location
        if status:
            key = status.strip().lower()
            if key not in EVENT_STATUSES:
                raise ValueError("status must be \"start\", \"end\" or \"cancel\"")
            kwargs["status"] = getattr(discord.EventStatus, EVENT_STATUSES[key])
        if not kwargs:
            raise ValueError("nothing to change - give at least one field")
        updated = await match.edit(reason=_reason(ctx), **kwargs)
        return (f"[System] Updated scheduled event \"{(updated or match).name}\" "
                f"({', '.join(kwargs)}).")

    return _guild_action("discord_edit_event", "edit that scheduled event", edit)


# --- members -----------------------------------------------------------------

def handle_kick_member(user_id: str, reason: str = "") -> str:
    async def kick(ctx):
        member = await _fetch_member(ctx, user_id)
        _check_target(ctx, member, "kick")
        await member.kick(reason=_reason(ctx, reason))
        return f"[System] Kicked {member.display_name} (id={member.id})."

    return _guild_action("discord_kick_member", "kick that member", kick)


def handle_ban_member(user_id: str, delete_message_hours=0, reason: str = "") -> str:
    async def ban(ctx):
        import discord
        uid = _snowflake(user_id, "user id")
        # Someone who already left can still be banned, so a missing member is fine;
        # a present one goes through the same guard as a kick.
        member = ctx.guild.get_member(uid)
        if member is None:
            try:
                member = await ctx.guild.fetch_member(uid)
            except discord.NotFound:
                member = None
        if member is not None:
            _check_target(ctx, member, "ban")
        elif uid in config.DISCORD_OWNER_IDS or uid == ctx.guild.owner_id:
            raise ValueError("cannot ban an owner")
        try:
            hours = max(0, min(int(delete_message_hours or 0), 24 * 7))
        except (TypeError, ValueError):
            hours = 0
        await ctx.guild.ban(discord.Object(id=uid), reason=_reason(ctx, reason),
                            delete_message_seconds=hours * 3600)
        who = member.display_name if member is not None else f"user {uid}"
        wiped = f", deleting their last {hours}h of messages" if hours else ""
        return f"[System] Banned {who} (id={uid}){wiped}."

    return _guild_action("discord_ban_member", "ban that user", ban)


def handle_unban_member(user_id: str, reason: str = "") -> str:
    async def unban(ctx):
        import discord
        uid = _snowflake(user_id, "user id")
        await ctx.guild.unban(discord.Object(id=uid), reason=_reason(ctx, reason))
        return f"[System] Unbanned user {uid}."

    return _guild_action("discord_unban_member", "unban that user", unban)


def handle_list_bans(limit=50) -> str:
    try:
        limit = max(1, min(int(limit), 200))
    except (TypeError, ValueError):
        limit = 50

    async def collect(ctx):
        entries = [entry async for entry in ctx.guild.bans(limit=limit)]
        if not entries:
            return "(nobody is banned from this server)"
        lines = [f"- {e.user} (id={e.user.id})" + (f" — {e.reason}" if e.reason else "")
                 for e in entries]
        return f"{len(entries)} ban(s) in {ctx.guild.name}:\n" + "\n".join(lines)

    return _guild_action("discord_list_bans", "read this server's ban list", collect)


MAX_TIMEOUT_MINUTES = 28 * 24 * 60      # Discord's ceiling


def handle_timeout_member(user_id: str, minutes=10, reason: str = "") -> str:
    async def timeout(ctx):
        from datetime import timedelta
        member = await _fetch_member(ctx, user_id)
        _check_target(ctx, member, "time out")
        try:
            amount = int(float(minutes))
        except (TypeError, ValueError):
            raise ValueError(f"'{minutes}' is not a number of minutes")
        if amount <= 0:
            await member.timeout(None, reason=_reason(ctx, reason))
            return f"[System] Removed the timeout on {member.display_name}."
        amount = min(amount, MAX_TIMEOUT_MINUTES)
        await member.timeout(timedelta(minutes=amount), reason=_reason(ctx, reason))
        return f"[System] Timed out {member.display_name} for {amount} minute(s)."

    return _guild_action("discord_timeout_member", "time out that member", timeout)


def handle_set_nickname(user_id: str = "", nickname: str = "") -> str:
    async def rename(ctx):
        me = ctx.guild.me
        if not (user_id or "").strip() or _snowflake(user_id, "user id") == me.id:
            member = me                             # the bot renaming itself needs no guard
        else:
            member = await _fetch_member(ctx, user_id)
            _check_target(ctx, member, "rename")
        nick = nickname.strip() or None
        await member.edit(nick=nick, reason=_reason(ctx))
        return (f"[System] Set {member.name}'s nickname to \"{nick}\"." if nick
                else f"[System] Reset {member.name}'s nickname.")

    return _guild_action("discord_set_nickname", "change that nickname", rename)


def handle_move_member(user_id: str, channel: str = "") -> str:
    async def move(ctx):
        member = await _fetch_member(ctx, user_id)
        _check_target(ctx, member, "move")
        if getattr(member, "voice", None) is None or member.voice.channel is None:
            raise ValueError(f"{member.display_name} is not in a voice channel")
        if not channel:
            await member.move_to(None, reason=_reason(ctx))
            return f"[System] Disconnected {member.display_name} from voice."
        target = await _resolve_voice_channel(ctx, channel)
        await member.move_to(target, reason=_reason(ctx))
        return f"[System] Moved {member.display_name} to {target.name}."

    return _guild_action("discord_move_member", "move that member", move)


# --- messages ----------------------------------------------------------------

def handle_purge_messages(limit=10, channel: str = "", user_id: str = "",
                          contains: str = "") -> str:
    try:
        limit = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        return f"[Error] '{limit}' is not a number of messages."

    async def purge(ctx):
        target = await _guild_channel(ctx, channel)
        if not hasattr(target, "purge"):
            raise ValueError(f"#{target.name} does not hold messages that can be bulk-deleted")
        author_id = _snowflake(user_id, "user id") if user_id else None
        needle = contains.casefold()

        def wanted(message):
            if author_id is not None and message.author.id != author_id:
                return False
            return not needle or needle in (message.content or "").casefold()

        # `limit` counts messages scanned, not deleted, once a filter is set.
        deleted = await target.purge(limit=limit, check=wanted, before=ctx.started_at,
                                     reason=_reason(ctx))
        return f"[System] Deleted {len(deleted)} message(s) in #{target.name}."

    return _guild_action("discord_purge_messages", "delete messages in that channel", purge)


def handle_delete_message(message_id: str, channel: str = "") -> str:
    async def delete(ctx):
        target = await _guild_channel(ctx, channel)
        message = await target.fetch_message(_snowflake(message_id, "message id"))
        await message.delete()
        return f"[System] Deleted message {message.id} by {message.author} in #{target.name}."

    return _guild_action("discord_delete_message", "delete that message", delete)


def handle_pin_message(message_id: str, channel: str = "", action: str = "pin") -> str:
    action = (action or "pin").strip().lower()
    if action not in ("pin", "unpin"):
        return "[Error] action must be \"pin\" or \"unpin\"."

    async def pin(ctx):
        target = await _guild_channel(ctx, channel)
        message = await target.fetch_message(_snowflake(message_id, "message id"))
        if action == "pin":
            await message.pin(reason=_reason(ctx))
        else:
            await message.unpin(reason=_reason(ctx))
        return f"[System] {action.capitalize()}ned message {message.id} in #{target.name}."

    return _guild_action("discord_pin_message", f"{action} that message", pin)


# --- channels ----------------------------------------------------------------

def handle_edit_channel(channel: str = "", name: str = "", topic: str = "",
                        slowmode_seconds="", nsfw="", category: str = "") -> str:
    async def edit(ctx):
        target = await _guild_channel(ctx, channel)
        kwargs = {}
        if name:
            kwargs["name"] = name.strip()
        if topic:
            kwargs["topic"] = topic
        if slowmode_seconds not in ("", None):
            try:
                kwargs["slowmode_delay"] = max(0, min(int(slowmode_seconds), 21600))
            except (TypeError, ValueError):
                raise ValueError(f"'{slowmode_seconds}' is not a number of seconds")
        if _flag(nsfw) is not None:
            kwargs["nsfw"] = _flag(nsfw)
        if category:
            wanted = category.strip().casefold()
            cats = getattr(ctx.guild, "categories", []) or []
            hits = ([c for c in cats if str(c.id) == category.strip()]
                    or [c for c in cats if c.name.casefold() == wanted])
            if not hits:
                names = ", ".join(c.name for c in cats[:20]) or "(none)"
                raise ValueError(f"no category named '{category}'. Categories: {names}")
            kwargs["category"] = hits[0]
        if not kwargs:
            raise ValueError("nothing to change - give at least one field")
        await target.edit(reason=_reason(ctx), **kwargs)
        return f"[System] Updated #{target.name} ({', '.join(kwargs)})."

    return _guild_action("discord_edit_channel", "edit that channel", edit)


def handle_delete_channel(channel: str) -> str:
    if not (channel or "").strip():
        return "[Error] Name the channel to delete explicitly - there is no default for this."

    async def delete(ctx):
        target = await _guild_channel(ctx, channel)
        if ctx.channel is not None and target.id == ctx.channel.id:
            raise ValueError("refusing to delete the channel this conversation is in - "
                             "the reply would have nowhere to go. Ask from another channel.")
        label = target.name
        await target.delete(reason=_reason(ctx))
        return f"[System] Deleted channel #{label}."

    return _guild_action("discord_delete_channel", "delete that channel", delete)


def handle_lock_channel(channel: str = "", action: str = "lock") -> str:
    action = (action or "lock").strip().lower()
    if action not in ("lock", "unlock"):
        return "[Error] action must be \"lock\" or \"unlock\"."

    async def lock(ctx):
        target = await _guild_channel(ctx, channel)
        everyone = ctx.guild.default_role
        overwrite = target.overwrites_for(everyone)
        # Unlock restores "inherit" rather than forcing allow, so category
        # permissions keep deciding what they decided before the lock.
        value = False if action == "lock" else None
        overwrite.send_messages = value
        overwrite.send_messages_in_threads = value
        overwrite.create_public_threads = value
        await target.set_permissions(everyone, overwrite=overwrite, reason=_reason(ctx))
        return f"[System] {'Locked' if action == 'lock' else 'Unlocked'} #{target.name} for @everyone."

    return _guild_action("discord_lock_channel", f"{action} that channel", lock)


def handle_create_invite(channel: str = "", max_age_hours=24, max_uses=0) -> str:
    async def invite(ctx):
        target = await _guild_channel(ctx, channel)
        try:
            hours = max(0, min(int(max_age_hours), 24 * 7))
            uses = max(0, min(int(max_uses), 100))
        except (TypeError, ValueError):
            raise ValueError("max_age_hours and max_uses must be whole numbers")
        created = await target.create_invite(max_age=hours * 3600, max_uses=uses,
                                             unique=True, reason=_reason(ctx))
        age = f"expires in {hours}h" if hours else "never expires"
        use = f"{uses} use(s)" if uses else "unlimited uses"
        return f"[System] Invite to #{target.name}: {created.url} ({age}, {use})."

    return _guild_action("discord_create_invite", "create an invite", invite)


# --- roles ---------------------------------------------------------------------

def handle_create_role(name: str, color: str = "", hoist="", mentionable="") -> str:
    if not (name or "").strip():
        return "[Error] A role name is required."

    async def create(ctx):
        # Deliberately no `permissions` argument: a role is created with none, so
        # no conversation can mint an Administrator role through this tool.
        kwargs = {"name": name.strip(), "reason": _reason(ctx)}
        if color:
            kwargs["colour"] = _colour(color)
        if _flag(hoist) is not None:
            kwargs["hoist"] = _flag(hoist)
        if _flag(mentionable) is not None:
            kwargs["mentionable"] = _flag(mentionable)
        role = await ctx.guild.create_role(**kwargs)
        return f"[System] Created role \"{role.name}\" (id={role.id}) with no permissions."

    return _guild_action("discord_create_role", "create that role", create)


def handle_edit_role(role: str, name: str = "", color: str = "", hoist="",
                     mentionable="") -> str:
    async def edit(ctx):
        target = _resolve_role(ctx.guild, role)
        _check_role(ctx, target, "edit")
        kwargs = {}
        if name:
            kwargs["name"] = name.strip()
        if color:
            kwargs["colour"] = _colour(color)
        if _flag(hoist) is not None:
            kwargs["hoist"] = _flag(hoist)
        if _flag(mentionable) is not None:
            kwargs["mentionable"] = _flag(mentionable)
        if not kwargs:
            raise ValueError("nothing to change - give at least one field")
        await target.edit(reason=_reason(ctx), **kwargs)
        return f"[System] Updated role \"{target.name}\" ({', '.join(kwargs)})."

    return _guild_action("discord_edit_role", "edit that role", edit)


def handle_delete_role(role: str) -> str:
    async def delete(ctx):
        target = _resolve_role(ctx.guild, role)
        _check_role(ctx, target, "delete")
        label = target.name
        await target.delete(reason=_reason(ctx))
        return f"[System] Deleted role \"{label}\"."

    return _guild_action("discord_delete_role", "delete that role", delete)


# --- audit log -----------------------------------------------------------------

def handle_audit_log(limit=20, user_id: str = "", action: str = "") -> str:
    try:
        limit = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        limit = 20

    async def collect(ctx):
        import discord
        kwargs = {"limit": limit}
        if user_id:
            kwargs["user"] = discord.Object(id=_snowflake(user_id, "user id"))
        if action:
            key = action.strip().lower().replace(" ", "_")
            found = getattr(discord.AuditLogAction, key, None)
            if not isinstance(found, discord.AuditLogAction):
                raise ValueError(f"unknown audit log action '{action}' - e.g. ban, kick, "
                                 "member_role_update, channel_delete, message_delete")
            kwargs["action"] = found
        entries = [entry async for entry in ctx.guild.audit_logs(**kwargs)]
        if not entries:
            return "(no matching audit log entries)"
        lines = []
        for entry in entries:
            stamp = entry.created_at.strftime("%m-%d %H:%M")
            target = getattr(entry.target, "name", None) or getattr(entry.target, "id", None) or "?"
            reason = f" — {entry.reason}" if entry.reason else ""
            lines.append(f"- [{stamp}] {entry.user} → {entry.action.name} on {target}{reason}")
        return f"Last {len(lines)} audit log entr(ies) in {ctx.guild.name} (newest first):\n" + \
            "\n".join(lines)

    return _guild_action("discord_audit_log", "read the audit log", collect)


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
    if name == "discord_list_events":
        return handle_list_events()
    if name == "discord_create_event":
        return handle_create_event(
            str(arguments.get("name", "") or arguments.get("title", "") or ""),
            str(arguments.get("start_time", "") or arguments.get("start", "") or ""),
            str(arguments.get("end_time", "") or arguments.get("end", "") or ""),
            str(arguments.get("location", "") or arguments.get("place", "") or ""),
            str(arguments.get("channel", "") or arguments.get("channel_name", "") or ""),
            str(arguments.get("description", "") or arguments.get("details", "") or ""),
        )
    if name == "discord_cancel_event":
        return handle_cancel_event(str(arguments.get("event", "") or arguments.get("name", "")
                                       or arguments.get("event_id", "") or ""))
    if name == "discord_manage_role":
        return handle_manage_role(
            str(arguments.get("user_id", "") or arguments.get("user", "") or ""),
            str(arguments.get("role", "") or arguments.get("role_name", "")
                or arguments.get("role_id", "") or ""),
            str(arguments.get("action", "add") or "add"),
        )
    if name == "discord_create_channel":
        return handle_create_channel(
            str(arguments.get("name", "") or arguments.get("channel_name", "") or ""),
            str(arguments.get("kind", "text") or arguments.get("type", "text") or "text"),
            str(arguments.get("category", "") or ""),
            str(arguments.get("topic", "") or ""),
        )
    handler = _ADMIN_DISPATCH.get(name)
    if handler is not None:
        return handler(arguments)
    return None


def _arg(arguments: dict, *keys: str, default=""):
    """First non-empty value among several spellings of one argument."""
    for key in keys:
        value = arguments.get(key)
        if value not in (None, ""):
            return value
    return default


_ADMIN_DISPATCH = {
    "discord_edit_event": lambda a: handle_edit_event(
        str(_arg(a, "event", "event_id")), str(_arg(a, "name", "title")),
        str(_arg(a, "description")), str(_arg(a, "start_time", "start")),
        str(_arg(a, "end_time", "end")), str(_arg(a, "location")), str(_arg(a, "status"))),
    "discord_kick_member": lambda a: handle_kick_member(
        str(_arg(a, "user_id", "user")), str(_arg(a, "reason"))),
    "discord_ban_member": lambda a: handle_ban_member(
        str(_arg(a, "user_id", "user")), _arg(a, "delete_message_hours", default=0),
        str(_arg(a, "reason"))),
    "discord_unban_member": lambda a: handle_unban_member(
        str(_arg(a, "user_id", "user")), str(_arg(a, "reason"))),
    "discord_list_bans": lambda a: handle_list_bans(_arg(a, "limit", default=50)),
    "discord_timeout_member": lambda a: handle_timeout_member(
        str(_arg(a, "user_id", "user")), _arg(a, "minutes", "duration", default=10),
        str(_arg(a, "reason"))),
    "discord_set_nickname": lambda a: handle_set_nickname(
        str(_arg(a, "user_id", "user")), str(_arg(a, "nickname", "nick"))),
    "discord_move_member": lambda a: handle_move_member(
        str(_arg(a, "user_id", "user")), str(_arg(a, "channel", "channel_name"))),
    "discord_purge_messages": lambda a: handle_purge_messages(
        _arg(a, "limit", "count", default=10), str(_arg(a, "channel", "channel_name")),
        str(_arg(a, "user_id", "user")), str(_arg(a, "contains"))),
    "discord_delete_message": lambda a: handle_delete_message(
        str(_arg(a, "message_id", "message")), str(_arg(a, "channel", "channel_name"))),
    "discord_pin_message": lambda a: handle_pin_message(
        str(_arg(a, "message_id", "message")), str(_arg(a, "channel", "channel_name")),
        str(_arg(a, "action", default="pin"))),
    "discord_edit_channel": lambda a: handle_edit_channel(
        str(_arg(a, "channel", "channel_name")), str(_arg(a, "name", "new_name")),
        str(_arg(a, "topic")), _arg(a, "slowmode_seconds", "slowmode"),
        _arg(a, "nsfw"), str(_arg(a, "category"))),
    "discord_delete_channel": lambda a: handle_delete_channel(
        str(_arg(a, "channel", "channel_name", "channel_id"))),
    "discord_lock_channel": lambda a: handle_lock_channel(
        str(_arg(a, "channel", "channel_name")), str(_arg(a, "action", default="lock"))),
    "discord_create_invite": lambda a: handle_create_invite(
        str(_arg(a, "channel", "channel_name")), _arg(a, "max_age_hours", default=24),
        _arg(a, "max_uses", default=0)),
    "discord_create_role": lambda a: handle_create_role(
        str(_arg(a, "name", "role_name")), str(_arg(a, "color", "colour")),
        _arg(a, "hoist"), _arg(a, "mentionable")),
    "discord_edit_role": lambda a: handle_edit_role(
        str(_arg(a, "role", "role_id", "role_name")), str(_arg(a, "name", "new_name")),
        str(_arg(a, "color", "colour")), _arg(a, "hoist"), _arg(a, "mentionable")),
    "discord_delete_role": lambda a: handle_delete_role(
        str(_arg(a, "role", "role_id", "role_name"))),
    "discord_audit_log": lambda a: handle_audit_log(
        _arg(a, "limit", default=20), str(_arg(a, "user_id", "user")), str(_arg(a, "action"))),
}


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
    {
        "name": "discord_list_events",
        "description": (
            "List this server's scheduled events (the ones under the server's "
            "Events tab): name, id, start time in UTC and where they happen. Use it "
            "before creating or cancelling an event so you act on the right one."),
        "parameters": {},
    },
    {
        "name": "discord_create_event",
        "description": (
            "Create a scheduled event in this server (an admin action - the bot must "
            "hold the Manage Events permission). An event is EITHER at an external "
            "location (give `location`, and `end_time` is then required) OR inside a "
            "voice/stage channel (give `channel`). Times are ISO 8601; a time without "
            "a timezone offset is read as UTC, so prefer an explicit offset like "
            "2026-09-20T18:00+09:00. Returns a link members can use to mark interest."),
        "parameters": {
            "name": "The event's title.",
            "start_time": "When it starts, ISO 8601 (e.g. 2026-09-20T18:00+09:00). Bare times are UTC.",
            "end_time": "(Required for a location event, optional for a channel event) When it ends, ISO 8601.",
            "location": "(For an external event) A free-text place, e.g. \"Zoom\" or \"Main Hall\".",
            "channel": "(For a voice/stage event) The voice or stage channel, by name or id. Use this OR location, not both.",
            "description": "(Optional) A longer description shown on the event page.",
        },
    },
    {
        "name": "discord_cancel_event",
        "description": (
            "Cancel (delete) a scheduled event in this server (an admin action). "
            "Identify it by id from discord_list_events, or by name."),
        "parameters": {
            "event": "The event's id (preferred) or its name.",
        },
    },
    {
        "name": "discord_manage_role",
        "description": (
            "Give a member a role or take one away (an admin action - the bot must "
            "hold Manage Roles and its own top role must sit above the target role). "
            "Cannot grant a role higher than the bot's own."),
        "parameters": {
            "user_id": "The member's id or mention.",
            "role": "The role, by name or id.",
            "action": "\"add\" to grant the role or \"remove\" to take it away. Default \"add\".",
        },
    },
    {
        "name": "discord_create_channel",
        "description": (
            "Create a new text or voice channel in this server (an admin action - the "
            "bot must hold Manage Channels)."),
        "parameters": {
            "name": "The new channel's name.",
            "kind": "\"text\" or \"voice\". Default \"text\".",
            "category": "(Optional) The category to place it under, by name or id.",
            "topic": "(Optional, text channels) The channel topic.",
        },
    },
    {
        "name": "discord_edit_event",
        "description": (
            "Change a scheduled event, or start/end it (admin, Manage Events). Only the "
            "fields you give change."),
        "parameters": {
            "event": "The event's id (preferred) or name.",
            "name": "(Optional) New title.",
            "description": "(Optional) New description.",
            "start_time": "(Optional) New start, ISO 8601 with an offset.",
            "end_time": "(Optional) New end, ISO 8601 with an offset.",
            "location": "(Optional) New location - external events only.",
            "status": "(Optional) \"start\" to begin it now, \"end\" to finish it, \"cancel\" to cancel it.",
        },
    },
    {
        "name": "discord_kick_member",
        "description": (
            "Kick a member from the server (admin, Kick Members). They can rejoin with an "
            "invite. The server owner, harness owners and the bot itself are never targets."),
        "parameters": {
            "user_id": "The member's id or mention.",
            "reason": "(Optional) Why - recorded in the audit log.",
        },
    },
    {
        "name": "discord_ban_member",
        "description": (
            "Ban a user from the server (admin, Ban Members). Works on users who already "
            "left. Prefer discord_timeout_member for a temporary problem."),
        "parameters": {
            "user_id": "The user's id or mention.",
            "delete_message_hours": "(Optional) Also delete their messages from the last N hours, 0-168. Default 0.",
            "reason": "(Optional) Why - recorded in the audit log.",
        },
    },
    {
        "name": "discord_unban_member",
        "description": "Lift a ban (admin, Ban Members). Find the id with discord_list_bans.",
        "parameters": {
            "user_id": "The banned user's id.",
            "reason": "(Optional) Why - recorded in the audit log.",
        },
    },
    {
        "name": "discord_list_bans",
        "description": "List the users banned from this server, with ban reasons (admin, Ban Members).",
        "parameters": {
            "limit": "(Optional) How many to list, up to 200. Default 50.",
        },
    },
    {
        "name": "discord_timeout_member",
        "description": (
            "Time a member out - they cannot talk, react or join voice until it ends "
            "(admin, Moderate Members). minutes=0 removes an existing timeout."),
        "parameters": {
            "user_id": "The member's id or mention.",
            "minutes": "How long, in minutes (maximum 40320 = 28 days). 0 lifts the timeout. Default 10.",
            "reason": "(Optional) Why - recorded in the audit log.",
        },
    },
    {
        "name": "discord_set_nickname",
        "description": (
            "Set or reset a member's server nickname (admin, Manage Nicknames). Omit "
            "user_id to rename the bot itself."),
        "parameters": {
            "user_id": "(Optional) The member's id or mention. Omit for the bot.",
            "nickname": "The new nickname. Empty resets it to their username.",
        },
    },
    {
        "name": "discord_move_member",
        "description": (
            "Move a member who is in voice to another voice channel, or disconnect them "
            "(admin, Move Members)."),
        "parameters": {
            "user_id": "The member's id or mention.",
            "channel": "(Optional) Target voice channel by name or id. Omit to disconnect them.",
        },
    },
    {
        "name": "discord_purge_messages",
        "description": (
            "Bulk-delete recent messages in a channel (admin, Manage Messages). Only "
            "messages from before the current request are touched. With a filter, "
            "`limit` is how many messages are scanned, not deleted. Read the channel "
            "first so you know what you are about to remove."),
        "parameters": {
            "limit": "How many recent messages to scan, 1-100. Default 10.",
            "channel": "(Optional) Channel by name or id. Omit for the current one.",
            "user_id": "(Optional) Only delete messages by this user.",
            "contains": "(Optional) Only delete messages containing this text.",
        },
    },
    {
        "name": "discord_delete_message",
        "description": "Delete one message by id (admin, Manage Messages for others' messages).",
        "parameters": {
            "message_id": "The message id (discord_read_messages shows them).",
            "channel": "(Optional) Channel by name or id. Omit for the current one.",
        },
    },
    {
        "name": "discord_pin_message",
        "description": "Pin or unpin a message (admin, Pin Messages).",
        "parameters": {
            "message_id": "The message id.",
            "channel": "(Optional) Channel by name or id. Omit for the current one.",
            "action": "\"pin\" or \"unpin\". Default \"pin\".",
        },
    },
    {
        "name": "discord_edit_channel",
        "description": (
            "Rename a channel or change its topic, slowmode, NSFW flag or category "
            "(admin, Manage Channels). Only the fields you give change."),
        "parameters": {
            "channel": "(Optional) Channel by name or id. Omit for the current one.",
            "name": "(Optional) New name.",
            "topic": "(Optional) New topic.",
            "slowmode_seconds": "(Optional) Seconds between messages per user, 0-21600. 0 turns slowmode off.",
            "nsfw": "(Optional) true or false.",
            "category": "(Optional) Move under this category, by name or id.",
        },
    },
    {
        "name": "discord_delete_channel",
        "description": (
            "Permanently delete a channel and everything in it (admin, Manage Channels). "
            "The channel must be named explicitly, and the channel you are talking in "
            "cannot be deleted."),
        "parameters": {
            "channel": "The channel by name or id.",
        },
    },
    {
        "name": "discord_lock_channel",
        "description": (
            "Lock a channel so @everyone cannot send messages or open threads, or unlock "
            "it again (admin, Manage Roles). Roles with their own allow still can."),
        "parameters": {
            "channel": "(Optional) Channel by name or id. Omit for the current one.",
            "action": "\"lock\" or \"unlock\". Default \"lock\".",
        },
    },
    {
        "name": "discord_create_invite",
        "description": "Create an invite link to a channel (Create Invite).",
        "parameters": {
            "channel": "(Optional) Channel by name or id. Omit for the current one.",
            "max_age_hours": "(Optional) Hours until it expires, 0-168; 0 never expires. Default 24.",
            "max_uses": "(Optional) Maximum uses, 0-100; 0 is unlimited. Default 0.",
        },
    },
    {
        "name": "discord_create_role",
        "description": (
            "Create a role (admin, Manage Roles). It is created with NO permissions - this "
            "tool cannot grant permissions; tell the user to set them in server settings."),
        "parameters": {
            "name": "The role's name.",
            "color": "(Optional) Hex colour like #5865F2.",
            "hoist": "(Optional) true to list its members separately in the sidebar.",
            "mentionable": "(Optional) true to let anyone @mention it.",
        },
    },
    {
        "name": "discord_edit_role",
        "description": (
            "Rename a role or change its colour, hoist or mentionable flag (admin, Manage "
            "Roles). The role must be below the bot's top role."),
        "parameters": {
            "role": "The role by name or id.",
            "name": "(Optional) New name.",
            "color": "(Optional) New hex colour.",
            "hoist": "(Optional) true or false.",
            "mentionable": "(Optional) true or false.",
        },
    },
    {
        "name": "discord_delete_role",
        "description": "Delete a role (admin, Manage Roles). The role must be below the bot's top role.",
        "parameters": {
            "role": "The role by name or id.",
        },
    },
    {
        "name": "discord_audit_log",
        "description": (
            "Read the server's audit log - who kicked, banned, deleted or changed what "
            "(admin, View Audit Log). Newest first."),
        "parameters": {
            "limit": "(Optional) How many entries, up to 100. Default 20.",
            "user_id": "(Optional) Only actions taken by this user.",
            "action": "(Optional) One action type, e.g. ban, kick, member_role_update, channel_delete, message_delete.",
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
