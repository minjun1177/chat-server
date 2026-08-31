"""The harness, as a Discord bot.

    DISCORD_BOT_TOKEN=... python bot.py

It answers three ways: mention it, reply to something it said, or use `/call`.
`/call` is registered as a user-installable command too, so once you install
the app on your account it comes with you into servers the bot itself is not
in, and into DMs and group DMs.

Everything below is plumbing around `agent.run_turn`: work out who is asking,
decide which tools they may reach, build a `DiscordSink`, and hand the message
to the same loop the terminal uses.

Safety: the harness's tools reach the whole machine, so the bot ships closed.
Only ids in `config.DISCORD_OWNER_IDS` can touch anything that writes, and with
no allow list at all nobody else can talk to it.
"""

import asyncio
import datetime
import io
import os
import sys
import traceback

import discord
from discord import app_commands

import agent
import config
import discord_tools
import env
import mcp_client
import permissions
import providers
import session as session_store
import ui
from discord_ui import DiscordSink, workspace_path


# ---------------------------------------------------------------------------
# access control
# ---------------------------------------------------------------------------

OWNER, GUEST, NOBODY = "owner", "guest", None


def access_level(user_id: int, guild_id: int | None, is_dm: bool) -> str | None:
    if user_id in config.DISCORD_OWNER_IDS:
        return OWNER
    if is_dm and not config.DISCORD_ALLOW_DM:
        return NOBODY
    if guild_id is not None and config.DISCORD_ALLOWED_GUILDS \
            and guild_id not in config.DISCORD_ALLOWED_GUILDS:
        return NOBODY
    if user_id in config.DISCORD_ALLOWED_USERS:
        return GUEST
    if config.DISCORD_ALLOW_EVERYONE:
        return GUEST
    return NOBODY


def tool_policy_for(level: str) -> ui.ToolPolicy:
    """Which tools exist for this person - see config.DISCORD_SAFE_TOOLS."""
    if level == OWNER and config.DISCORD_OWNER_IDS:
        allowed = set(config.DISCORD_SAFE_TOOLS) | set(config.DISCORD_OWNER_TOOLS)
        return ui.ToolPolicy(allowed, "owner")
    allowed = set(config.DISCORD_SAFE_TOOLS)
    if config.DISCORD_MCP_FOR_EVERYONE:
        allowed.add("mcp__*")
    return ui.ToolPolicy(allowed, "guest")


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------

def session_key(guild_id: int | None, channel_id: int, user_id: int) -> str:
    scope = config.DISCORD_SESSION_SCOPE
    where = f"g{guild_id}-c{channel_id}" if guild_id else f"dm-c{channel_id}"
    if scope == "user":
        return f"u{user_id}"
    if scope == "channel+user":
        return f"{where}-u{user_id}"
    return where


def memory_file_for(user_id: int) -> str:
    return os.path.join(config.DISCORD_MEMORY_DIR, f"discord-{user_id}.json")


def session_dir_for(user_id: int) -> str:
    base = os.path.join(config.SESSION_DIR, config.DISCORD_SESSION_SUBDIR)
    if config.DISCORD_SESSION_SCOPE in ("user", "channel+user"):
        return os.path.join(base, f"u{user_id}")
    return base


def get_session(guild_id, channel_id, user_id, level: str) -> agent.AgentSession:
    agent.prune()               # conversations nobody has touched in hours
    key = session_key(guild_id, channel_id, user_id)
    policy = tool_policy_for(level)
    sess = agent.get_or_create(
        key,
        tool_filter=policy.allowed,
        discord=True,
        session_dir=session_dir_for(user_id),
        memory_file=memory_file_for(user_id),
    )
    # Memory follows the speaker even when the conversation belongs to a channel,
    # and the tool set follows whoever is talking right now.
    sess.memory_file = memory_file_for(user_id)
    if sess.tool_filter != policy.allowed:
        sess.tool_filter = policy.allowed
        sess.refresh_system_prompt()
    sess.workspace = os.path.join(config.DISCORD_WORKSPACE, key)
    return sess


# ---------------------------------------------------------------------------
# the client
# ---------------------------------------------------------------------------

class HarnessBot(discord.Client):

    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        register_commands(self)
        await self.tree.sync()
        print("[bot] application commands synced")

    async def on_ready(self):
        print(f"[bot] logged in as {self.user} (id={self.user.id})")
        print(f"[bot] provider: {providers.status_line()}")
        if not config.DISCORD_OWNER_IDS:
            print("[bot] WARNING: DISCORD_OWNER_IDS is empty in .env - every "
                  "conversation is limited to the safe, read-only tool set.")

    async def on_message(self, message: discord.Message):
        if message.author.bot or message.author.id == self.user.id:
            return

        is_dm = message.guild is None
        mentioned = self.user in message.mentions
        replied_to_me = False
        if message.reference is not None and message.reference.resolved is not None:
            resolved = message.reference.resolved
            replied_to_me = getattr(getattr(resolved, "author", None), "id", None) == self.user.id
        elif message.reference is not None and message.reference.message_id:
            try:
                parent = await message.channel.fetch_message(message.reference.message_id)
                replied_to_me = parent.author.id == self.user.id
            except (discord.HTTPException, AttributeError):
                replied_to_me = False

        if not (mentioned or replied_to_me or is_dm):
            return

        level = access_level(message.author.id, message.guild.id if message.guild else None, is_dm)
        if level is NOBODY:
            try:
                await message.add_reaction("🔒")
            except discord.HTTPException:
                pass
            return

        content = message.content
        for mention in (f"<@{self.user.id}>", f"<@!{self.user.id}>"):
            content = content.replace(mention, "")
        content = content.strip()

        sess = get_session(message.guild.id if message.guild else None,
                           message.channel.id, message.author.id, level)
        prompt = await build_prompt(content, message.attachments, sess, message.author)
        if not prompt:
            await message.channel.send("Ask me something and I'll get to work - a question, or a task to run.")
            return

        if sess.busy:
            try:
                await message.add_reaction("⏳")
            except discord.HTTPException:
                pass

        sink = DiscordSink(self, message.channel, message.author,
                           tool_policy=tool_policy_for(level),
                           workspace=sess.workspace,
                           owner_ids=config.DISCORD_OWNER_IDS,
                           on_cancel=lambda: cancel_session(sess))
        sink.show_thinking = sess.meta.get("show_thinking", False)
        await launch_turn(self, sess, prompt, sink, message.channel,
                          message.guild, message.author, is_user_app=False)


# ---------------------------------------------------------------------------
# running a turn
# ---------------------------------------------------------------------------

def cancel_session(sess: agent.AgentSession) -> int:
    """Stop whatever this conversation is doing. A queued turn counts too."""
    return sess.cancel()


async def build_prompt(text: str, attachments, sess: agent.AgentSession, author) -> str:
    """The user's message, plus where any attachments were saved."""
    parts = [text] if text else []
    if attachments:
        saved = []
        base = agent.workspace_for(sess)
        for attachment in attachments:
            path = workspace_path(base, attachment.filename)
            try:
                await attachment.save(path)
                saved.append(f"[attachment saved: {os.path.abspath(path)} "
                             f"({attachment.content_type or 'unknown'}, {attachment.size} bytes)]")
            except (discord.HTTPException, OSError) as error:
                saved.append(f"[attachment {attachment.filename} could not be saved: {error}]")
        if saved:
            parts.append("\n".join(saved)
                         + "\nRead a saved file with read_file when it is text or code.")
    speaker = getattr(author, "display_name", None) or str(author)
    if parts:
        parts.insert(0, f"[from Discord user {speaker} (id={author.id})]")
    return "\n".join(parts).strip()


async def launch_turn(client, sess, prompt, sink, channel, guild, author, is_user_app: bool):
    """Run one turn as a task so `/stop` and the Stop button can cancel it."""

    # Captured now, not read off the session later: a channel's conversation is
    # shared, its memories are not, and someone else may speak before this turn
    # gets its slot.
    memory_file = memory_file_for(author.id)

    async def runner():
        ctx = discord_tools.DiscordContext(
            client=client, loop=asyncio.get_running_loop(), channel=channel,
            guild=guild, author=author, workspace=sess.workspace, is_user_app=is_user_app)
        token = discord_tools.set_context(ctx)
        try:
            await agent.run_turn(sess, prompt, sink, memory_file=memory_file)
        except asyncio.CancelledError:
            with_suppressed(channel.send("⏹ Stopped."))
            raise
        except Exception as error:                      # noqa: BLE001
            traceback.print_exc()
            try:
                await channel.send(f"⚠ Something went wrong: `{type(error).__name__}: {error}`")
            except discord.HTTPException:
                pass
        finally:
            discord_tools.reset_context(token)

    task = asyncio.create_task(runner())
    sess.track(task)
    try:
        await task
    except asyncio.CancelledError:
        pass


def with_suppressed(coro):
    """Fire a coroutine we do not want to await inside a cancellation path."""
    asyncio.ensure_future(_suppress(coro))


async def _suppress(coro):
    try:
        await coro
    except Exception:                                   # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# slash commands
# ---------------------------------------------------------------------------

def register_commands(client: HarnessBot) -> None:
    tree = client.tree

    def anywhere(command):
        """A command that works in servers, DMs, group DMs, and as a user app."""
        app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)(command)
        app_commands.allowed_installs(guilds=True, users=True)(command)
        return command

    async def resolve(interaction: discord.Interaction):
        """(level, session, channel) for an interaction, or (None, None, None)."""
        guild_id = interaction.guild_id
        is_dm = guild_id is None
        level = access_level(interaction.user.id, guild_id, is_dm)
        if level is NOBODY:
            await interaction.response.send_message(
                "You are not on this bot's allow list. Ask its owner for access.", ephemeral=True)
            return None, None, None
        channel = interaction.channel
        if channel is None:
            channel = client.get_channel(interaction.channel_id)
        sess = get_session(guild_id, interaction.channel_id, interaction.user.id, level)
        return level, sess, channel

    # -- the one that matters ---------------------------------------------

    @tree.command(name="call", description="Ask the AI agent - works anywhere the app is installed")
    @app_commands.describe(prompt="What to ask or do", file="A file to hand over (optional)")
    @anywhere
    async def call(interaction: discord.Interaction, prompt: str,
                   file: discord.Attachment | None = None):
        level, sess, channel = await resolve(interaction)
        if sess is None:
            return
        await interaction.response.defer(thinking=True)

        text = await build_prompt(prompt, [file] if file else [], sess, interaction.user)
        sink = DiscordSink(client, channel, interaction.user,
                           tool_policy=tool_policy_for(level),
                           workspace=sess.workspace,
                           owner_ids=config.DISCORD_OWNER_IDS,
                           interaction=interaction,
                           on_cancel=lambda: cancel_session(sess))
        sink.show_thinking = sess.meta.get("show_thinking", False)
        is_user_app = interaction.guild is None and interaction.guild_id is not None
        await launch_turn(client, sess, text, sink, channel, interaction.guild,
                          interaction.user, is_user_app=is_user_app)

    # -- conversation management -------------------------------------------

    @tree.command(name="new", description="Start this conversation over")
    @anywhere
    async def new(interaction: discord.Interaction):
        level, sess, _ = await resolve(interaction)
        if sess is None:
            return
        if sess.busy:
            await interaction.response.send_message("Still working on the previous request.", ephemeral=True)
            return
        agent.persist(sess)
        _archive(sess)
        sess.reset()
        await interaction.response.send_message("🆕 Fresh conversation. The previous transcript was kept on disk.")

    @tree.command(name="session", description="Show this conversation's state")
    @anywhere
    async def session_info(interaction: discord.Interaction):
        level, sess, _ = await resolve(interaction)
        if sess is None:
            return
        turns = sum(1 for m in sess.messages if m.get("role") == "user")
        embed = discord.Embed(title="Conversation", color=discord.Color.blurple())
        embed.add_field(name="Title", value=sess.title or "(not named yet)", inline=False)
        embed.add_field(name="Scope", value=f"`{sess.key}` ({config.DISCORD_SESSION_SCOPE})", inline=False)
        embed.add_field(name="Messages", value=f"{len(sess.messages)} ({turns} from users)")
        embed.add_field(name="Tokens", value=f"{sess.token_total()}")
        embed.add_field(name="Model", value=providers.status_line())
        embed.add_field(name="Access", value=level)
        embed.set_footer(text=f"your memories: {memory_file_for(interaction.user.id)}")
        await interaction.response.send_message(embed=embed)

    @tree.command(name="export", description="Download this conversation as Markdown")
    @anywhere
    async def export(interaction: discord.Interaction):
        level, sess, _ = await resolve(interaction)
        if sess is None:
            return
        lines = [f"# {sess.title or sess.key}", ""]
        for message in sess.messages:
            role = message.get("role", "")
            if role == "system":
                continue
            who = {"user": "User", "assistant": "Assistant"}.get(role, role)
            lines.append(f"## {who}\n\n{message.get('content', '')}\n")
        data = io.BytesIO("\n".join(lines).encode("utf-8"))
        name = f"{(sess.title or sess.key).replace(' ', '-')}.md"
        await interaction.response.send_message(
            file=discord.File(data, filename=name))

    @tree.command(name="stop", description="Stop whatever is running")
    @anywhere
    async def stop(interaction: discord.Interaction):
        level, sess, _ = await resolve(interaction)
        if sess is None:
            return
        stopped = cancel_session(sess)
        if not stopped:
            await interaction.response.send_message("Nothing is running.", ephemeral=True)
            return
        await interaction.response.send_message(
            f"⏹ Asked {stopped} running turn(s) to stop.")

    # -- what the bot can do ------------------------------------------------

    @tree.command(name="help", description="How to use this bot")
    @anywhere
    async def help_command(interaction: discord.Interaction):
        level = access_level(interaction.user.id, interaction.guild_id, interaction.guild_id is None)
        embed = discord.Embed(
            title="AI harness",
            description=("**How to reach me**\n"
                         "• Mention me\n"
                         "• Reply to one of my messages\n"
                         "• `/call <prompt>` - anywhere, once you install the app on your account\n"
                         "• In a DM, just talk"),
            color=discord.Color.blurple())
        embed.add_field(
            name="Commands",
            value=("`/call` ask · `/new` reset · `/session` state · `/export` download\n"
                   "`/stop` cancel · `/tools` what I can use · `/memory` your notes\n"
                   "`/mcp` servers · `/think` show reasoning · `/model` provider (owner)\n"
                   "`/perms` permission rules (owner)"),
            inline=False)
        embed.add_field(name="Your access",
                        value={OWNER: "owner - every tool, each behind an approval button",
                               GUEST: "guest - the read-only safe tools"}.get(level, "no access"),
                        inline=False)
        embed.set_footer(text=providers.status_line())
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @tree.command(name="tools", description="List the tools available to you here")
    @anywhere
    async def tools_command(interaction: discord.Interaction):
        level, sess, _ = await resolve(interaction)
        if sess is None:
            return
        policy = tool_policy_for(level)
        names = sorted(policy.allowed) if policy.allowed else ["(everything)"]
        mcp_names = mcp_client.available_tool_names() if policy.permits("mcp__x") else []
        embed = discord.Embed(title=f"Available tools ({len(names)})",
                              description="`" + "` `".join(names) + "`",
                              color=discord.Color.blurple())
        if mcp_names:
            embed.add_field(name=f"MCP ({len(mcp_names)})",
                            value="`" + "` `".join(sorted(mcp_names)[:40]) + "`", inline=False)
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @tree.command(name="memory", description="List your own stored memories")
    @anywhere
    async def memory_command(interaction: discord.Interaction):
        level, sess, _ = await resolve(interaction)
        if sess is None:
            return
        from session import handle_get_memory_list
        token = session_store.use_store(memory_file_for(interaction.user.id),
                                        sess.session_dir)
        try:
            listing = handle_get_memory_list()
        finally:
            session_store.reset_store(token)
        await interaction.response.send_message(
            f"```\n{listing[:1900]}\n```", ephemeral=True)

    @tree.command(name="think", description="Show or hide a reasoning model's thinking")
    @app_commands.describe(state="on or off")
    @anywhere
    async def think(interaction: discord.Interaction, state: str):
        level, sess, _ = await resolve(interaction)
        if sess is None:
            return
        on = state.strip().lower() in ("on", "true", "yes", "1")
        sess.meta["show_thinking"] = on
        await interaction.response.send_message(
            f"Reasoning is now {'shown' if on else 'hidden'} here.", ephemeral=True)

    @tree.command(name="mcp", description="Show MCP server status")
    @app_commands.describe(action="empty for status, 'tools' to list them, 'reload' to reconnect (owner)")
    @anywhere
    async def mcp_command(interaction: discord.Interaction, action: str = ""):
        level, sess, _ = await resolve(interaction)
        if sess is None:
            return
        action = (action or "").strip().lower()
        if action == "reload":
            if level != OWNER:
                await interaction.response.send_message("Only an owner can reconnect the servers.", ephemeral=True)
                return
            await interaction.response.defer(thinking=True)
            await asyncio.to_thread(mcp_client.load_servers, True)
            await asyncio.to_thread(mcp_client.connect_all)
            for other in agent.all_sessions():
                other.refresh_system_prompt()
            await interaction.followup.send("MCP servers reconnected.")
            return

        servers = mcp_client.load_servers()
        embed = discord.Embed(title="MCP servers", color=discord.Color.blurple())
        if not servers:
            embed.description = "No servers are configured (`.mcp.json`)."
        for name, server in servers.items():
            detail = [f"state: {server.state}"]
            if getattr(server, "error", ""):
                detail.append(f"error: {str(server.error)[:200]}")
            tools = [t.get("name", "") for t in getattr(server, "tools", []) or []]
            detail.append(f"{len(tools)} tool(s)")
            if action == "tools" and tools:
                detail.append("`" + "` `".join(tools[:30]) + "`")
            embed.add_field(name=name, value="\n".join(detail)[:1000], inline=False)
        await interaction.response.send_message(embed=embed, ephemeral=True)

    # -- owner only ---------------------------------------------------------

    @tree.command(name="model", description="Show the provider and model, or switch them (owner)")
    @app_commands.describe(provider="ollama, anthropic, openai, gemini", model="Model name")
    @anywhere
    async def model_command(interaction: discord.Interaction, provider: str = "", model: str = ""):
        level = access_level(interaction.user.id, interaction.guild_id, interaction.guild_id is None)
        if not provider and not model:
            await interaction.response.send_message(
                f"Currently: **{providers.status_line()}**", ephemeral=True)
            return
        if level != OWNER:
            await interaction.response.send_message("Only an owner can switch the model.", ephemeral=True)
            return
        await interaction.response.defer(thinking=True, ephemeral=True)
        name = provider or providers.current().name
        try:
            provider, problem = await asyncio.to_thread(providers.connect, name, model)
        except Exception as error:                      # noqa: BLE001
            await interaction.followup.send(f"Could not connect: `{error}`", ephemeral=True)
            return
        if provider is None or problem:
            await interaction.followup.send(f"❌ {problem or 'could not connect'}", ephemeral=True)
            return
        for other in agent.all_sessions():
            other.refresh_system_prompt()
        await interaction.followup.send(f"✅ {providers.status_line()}", ephemeral=True)

    @tree.command(name="perms", description="Show or add the bot's permission rules (owner)")
    @app_commands.describe(action="allow or deny", rule="e.g. run_cmd(git *)")
    @anywhere
    async def perms_command(interaction: discord.Interaction, action: str = "", rule: str = ""):
        level = access_level(interaction.user.id, interaction.guild_id, interaction.guild_id is None)
        if level != OWNER:
            await interaction.response.send_message("This command is owner-only.", ephemeral=True)
            return
        action = (action or "").strip().lower()
        if action in ("allow", "deny") and rule:
            saved, where = permissions.add_rule(rule, action)
            permissions.load_rules(force=True)
            await interaction.response.send_message(
                (f"✅ `{rule}` → {action} ({where})" if saved else f"❌ {where}"), ephemeral=True)
            return
        allow = [f"{t}({p})" if p else t for t, p in permissions.rules_for("allow")]
        deny = [f"{t}({p})" if p else t for t, p in permissions.rules_for("deny")]
        embed = discord.Embed(title="Permission rules", color=discord.Color.blurple())
        embed.add_field(name=f"allow ({len(allow)})", value="`" + "` `".join(allow[:40]) + "`" if allow else "(none)", inline=False)
        embed.add_field(name=f"deny ({len(deny)})", value="`" + "` `".join(deny[:40]) + "`" if deny else "(none)", inline=False)
        embed.set_footer(text=f"profile: {permissions.profile() or '(default)'} · saved to {permissions.write_target()}")
        await interaction.response.send_message(embed=embed, ephemeral=True)


def _archive(sess: agent.AgentSession) -> None:
    """Keep the transcript of a conversation `/new` is about to replace."""
    if not sess.session_dir or not sess.session_id:
        return
    current = os.path.join(sess.session_dir, f"{sess.session_id}.json")
    if not os.path.isfile(current):
        return
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        os.replace(current, os.path.join(sess.session_dir, f"{sess.session_id}-{stamp}.json"))
    except OSError as error:
        print(f"[bot] could not archive the session: {error}")


# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------

def start_mcp() -> None:
    if not config.MCP_ENABLED:
        return
    pending = [s for s in mcp_client.load_servers().values() if s.state != "disabled"]
    if not pending:
        return
    print(f"[bot] connecting {len(pending)} MCP server(s)…")
    mcp_client.connect_all()
    for server in pending:
        if server.state == "failed":
            print(f"[bot] MCP server '{server.name}' failed: {str(server.error)[:200]}")


def main() -> int:
    for problem in env.audit():
        print(f"[bot] ⚠ {problem}")

    token = os.environ.get(config.DISCORD_TOKEN_ENV, "").strip()
    if not token:
        where = ", ".join(env.sources()) or ".env"
        print(f"No bot token. Put {config.DISCORD_TOKEN_ENV}=… in {where} "
              f"(start from .env.example) or set it in the environment.")
        return 2
    if env.sources():
        print(f"[bot] settings from {', '.join(env.sources())}")

    providers.apply_startup()
    problem = providers.current().ready()
    if problem:
        print(f"[bot] the model provider is not usable yet: {problem}")
        return 2

    # Rules from .permissions.discord.json apply on top of the usual ones, so
    # the bot can be stricter than the terminal without touching its rules.
    permissions.use_profile("discord")
    permissions.load_rules(force=True)

    start_mcp()
    os.makedirs(config.DISCORD_WORKSPACE, exist_ok=True)
    os.makedirs(config.DISCORD_MEMORY_DIR, exist_ok=True)

    client = HarnessBot()
    try:
        client.run(token, log_handler=None)
    except discord.LoginFailure:
        print("[bot] Discord rejected the token.")
        return 2
    except discord.PrivilegedIntentsRequired:
        print("[bot] Enable the MESSAGE CONTENT intent for this app in the Discord "
              "developer portal (Bot → Privileged Gateway Intents).")
        return 2
    finally:
        mcp_client.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
