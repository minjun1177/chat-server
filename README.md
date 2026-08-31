# chat-server

A Discord bot that gives a language model real tools - a shell, your files, the
web, and any MCP server you attach - and shows every one of them being used.

## 1. What It Does

Mention it, reply to it, or `/call` it from anywhere, and it answers in the
channel while it works: prose streams into the message as the model writes it,
each tool call appears as its own card, and anything that touches your machine
waits behind an approval button that only you can press.

The model does not have to be a hosted one. The harness underneath speaks to
Ollama on your own machine as readily as to Anthropic, OpenAI or Gemini,
because tool calls travel as text rather than through a vendor's
function-calling API - so a small local model gets the same 31 tools a frontier
one does.

---

## 2. Features

- **Any Provider**: Ollama, Anthropic, OpenAI (or anything OpenAI-compatible), or Google Gemini, switched with `/model`. Tool calls travel as text, so changing provider changes nothing else about how the harness works.
- **Live Tool Cards**: Every call - built-in or from an MCP server - is posted as an embed the moment it starts and edited in place when it returns. Nothing runs invisibly.
- **Approval Buttons**: Shell commands, file writes and network calls wait for **Approve** / **Deny** / **Always allow** from the person who asked. Nobody clicking counts as a refusal.
- **Dynamic Context Compression**: Monitors active token counts and conversation length to automatically condense conversation history when nearing model limits, tailored to model size.
- **Per-User Memory**: A long-term key-value store the model reads and writes on its own, kept in a separate file per Discord user - one person's notes never surface in another's answers, even in a shared channel.
- **Sessions That Survive a Restart**: One conversation per channel, saved as it goes and reloaded when the bot comes back. `/new` starts over, `/export` downloads it as Markdown.
- **Named Sessions**: The model titles each conversation after its first exchange, so transcripts are filed under something readable instead of a timestamp.
- **Hashline Line-Level Hashing**: File reading appends 2-character MD5 hashes to line numbers (`LINE_NUM:HASH|`) allowing the agent to target precise code locations when making edits.
- **Multi-Source Web Search**: Queries several keyless sources (DuckDuckGo, Wikipedia, Stack Exchange, GitHub, optional self-hosted SearXNG), reads the actual pages, ranks passages locally with BM25, and reports "no relevant results" rather than returning off-topic pages.
- **Agent Skills**: Folder-based instruction packs (`skills/<name>/SKILL.md`) that the model loads on demand. Only each skill's name and description sit in the system prompt, so a large library stays cheap until a skill is actually needed.
- **Tool Permissions**: Rules in `.permissions.json` decide what runs without asking and what never runs at all, filling the gap between prompting for everything and `/automode` allowing everything. Answering `a` at any approval prompt saves a rule.
- **Reasoning Model Support**: A model's `<think>` blocks (and Ollama's separate `thinking` field) are kept out of the answer, off the screen by default, and out of the conversation history - so scratch work never eats the context budget.
- **Layered Logs**: `error`, `tool` and `request`, each including the one above it, tied together by a per-exchange id. The tool layer doubles as an audit trail of who ran what and which approvals were granted.
- **Nothing Personal in the Repository**: keys, tokens and the account ids allowed to use the bot live in a git-ignored `.env`; `config.py` holds only publishable defaults, and startup warns if a `.env` is ever staged for commit.
- **Discord Bot**: `python bot.py` puts the same harness in Discord - mention it, reply to it, or `/call` it from anywhere as a user-installed app. Tool calls show up as live embeds, approvals become buttons, and a tool policy keeps everything that writes to the machine in the owner's hands.
- **MCP Servers**: Any Model Context Protocol server declared in `.mcp.json` is started with the app, and its tools join the built-in ones as `mcp__<server>__<tool>`. Local subprocesses (stdio) and remote endpoints (streamable HTTP, legacy SSE) are all supported, with the same approval prompt guarding every call.

---

## 3. Setup

### Prerequisites
- **Python**: 3.10 or higher (tested on 3.14)
- **A Discord application**: with the Message Content intent on - see *Setting the app up on Discord* below
- **A model**: either [Ollama](https://ollama.com) running locally (default `http://localhost:11434`) or an API key for Anthropic, OpenAI or Gemini

### Installation Steps

1. **Install Required Packages**:
   ```bash
   pip install -r requirments.txt
   ```

2. **Pull an Ollama Model**:
   ```bash
   ollama pull gemma4:e4b
   ```

3. **Set up your own settings** (see below):
   ```bash
   cp .env.example .env && chmod 600 .env
   ```

4. **Run the bot**:
   ```bash
   python bot.py
   ```

### Configuration and secrets

`config.py` is committed, so nothing personal lives in it. Everything that
differs between two people running this - API keys, the Discord token, the
account ids allowed to use the bot, where sessions are written - goes in a
**`.env`** file that git ignores, and `config.py` keeps only the default.

```bash
DISCORD_BOT_TOKEN=...                 # from the developer portal
DISCORD_OWNER_IDS=123456789012345678  # full tool access
ANTHROPIC_API_KEY=...                 # only the provider you use
LOCALCHAT_MODEL=gemma4:e4b
```

`.env.example` lists every variable with its default. Three rules:

- **A real environment variable always wins** over the file, so a container or
  a systemd unit overrides it without editing anything.
- **`./.env` first, then `~/.localchat/.env`** - keep personal settings outside
  the repository entirely if you work in several checkouts.
- **The values are exported into the environment**, so `${VAR}` in `.mcp.json`
  and the provider key lookups read them too - one file for everything.

There is no dependency for this: `env.py` parses `KEY=value`, quotes, `export`
prefixes and comments in about eighty lines.

At startup both entry points check the obvious ways a secret escapes and say so:
a `.env` that git is tracking (with the `git rm --cached` line to fix it), and
one that is readable by other users on the machine.

---

## 4. Tool Capabilities

The bot equips the model with 31 function-calling tools, plus every tool each
attached MCP server exposes. Which of them a given person can reach is decided
by the tool policy - see *Who may use it* under Discord Bot.

### Tool Call Format

The model calls a tool by emitting a `<tool_call>` block. Anything a parameter
can hold in one line goes in the JSON; a file body does not:

```
<tool_call>
{"name": "write_file", "arguments": {"filepath": "game.py"}}
<content>
import random

print("Guess the number!")
</content>
</tool_call>
```

Escaping a whole source file into a JSON string is the single thing small local
models get wrong most often - a bare quote inside `print("hi")`, a lost
backslash before a line continuation, one uncounted brace - and any of them used
to throw the entire generation away. A raw block removes the requirement: the
text is written exactly as it belongs on disk, with no escaping at all. `<content>`
feeds `write_file`; `<old_content>` and `<new_content>` feed `edit_file`.

Plain JSON still works. When it arrives damaged, the parser repairs what is
unambiguously safe - unclosed brackets, parameters the model put beside `name`
instead of inside `arguments`, a payload whose quotes broke the JSON around it -
and says so. What it refuses to repair is a reply that stopped early: closing
the brackets there would invent arguments that were never sent, and `write_file`
would happily write the empty result over a real file. Those are reported, and
the model is asked to send the call again.

### Web & Network Tools
- `search_web`: Multi-source search with local relevance ranking (see Web Search below).
- `get_url`: Fetch web page contents and strip raw HTML down to readable text.
- `call_api`: Execute HTTP requests (GET, POST, PUT, PATCH, DELETE) with custom headers and JSON/text payloads.

### File System & Workspace Tools
- `read_file`: Read contents of a local file formatted with line numbers and line MD5 hashes.
- `write_file`: Create new files or overwrite existing file content. The body comes in a `<content>` raw block.
- `edit_file`: Find and replace specific content snippets within an existing file, via `<old_content>` / `<new_content>` raw blocks.
- `delete_file`: Remove a file from disk.
- `copy_file`: Copy a file to a new location.
- `create_dir`: Create a new directory path.
- `list_dir`: Display directory contents.
- `search_in_file`: Search workspace files for string or regex patterns (grep functionality).

### System & Git Management
- `run_cmd`: Execute system shell commands (requires approval). Stays connected to the command and reports when it is waiting for input.
- `send_input`: Answer a running command's prompt and read what it prints next.
- `end_process`: Stop a command left running by `run_cmd`.
- `get_system_info`: Retrieve system CPU, memory usage, disk statistics, and top memory-consuming processes.
- `git_status`: Check current git repository status.
- `git_diff`: View current git working directory modifications.

### Memory & Interaction Tools
- `write_memory`: Save key information to persistent JSON storage.
- `read_memory`: Retrieve content of a specific stored memory item.
- `get_memory_list`: List stored memory IDs with timestamp and preview.
- `edit_memory`: Update content of an existing memory record.
- `delete_memory`: Remove a memory entry from disk.
- `get_user_input`: Ask the user one or more questions, each with its own list of options plus a free-text choice.

### MCP Tools
Present only when an MCP server is attached (see MCP Servers below).
- `mcp__<server>__<tool>`: Every tool each connected server exposes, listed in the system prompt with its own parameters.
- `list_mcp_resources`: List the resources the connected servers expose, with the URI needed to read each one.
- `read_mcp_resource`: Read one resource by URI.

### Discord Tools
Present only when the harness is running as a Discord bot (see Discord Bot below).
- `discord_read_messages`: Read the channel's recent messages, oldest first - **including the embeds the bot itself wrote**, flattened back into text, so it can read its own past tool cards. Another channel is named the way a person names it (`general`, `#dev-talk`, `<#123…>`, or an id); an unknown name comes back with the list of channels that do exist. Pages further back with `before`.
- `discord_server_info`: The server's name, owner, creation date, member count, boost tier, features, channels grouped by category, roles and emoji count, as JSON. Takes a server name or id.
- `discord_user_info`: One user's display name, id, account age, join date and roles.
- `discord_send_file`: Upload a file from the conversation's workspace folder as an attachment. Paths outside it are refused.

### Workflow Tools
- `use_skill`: Load the full instructions of a skill listed in the system prompt.
- `submit_plan_for_approval`: Present a task plan and diff blueprint for approval before executing (plan mode).
- `get_code_skeleton`: Return a JSON outline of a source file's structure via Tree-sitter.
- `query_ast_node`: Search a source file for Tree-sitter S-expression patterns.

---

## 5. Providers

The bot starts on Ollama and stays there until told otherwise. An owner moves
it with `/model`:

```
/model                              show the provider and model in use
/model provider:anthropic           switch provider, keeping its saved model
/model provider:openai model:gpt-4o connect straight to a model
```

The provider is process-wide, so `/model` changes it for every channel at once.

| Provider | Endpoint | Key from |
| :--- | :--- | :--- |
| `ollama` | local, `OLLAMA_HOST` or a `base_url` | none needed |
| `anthropic` | `api.anthropic.com` | `ANTHROPIC_API_KEY` |
| `openai` | `api.openai.com/v1`, or any compatible `base_url` | `OPENAI_API_KEY` |
| `gemini` | `generativelanguage.googleapis.com` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |

Because `base_url` is settable, the `openai` entry also reaches anything that
speaks the same protocol - a local vLLM or llama.cpp server, OpenRouter, Groq,
Together.

Keys come from `.env` (or the real environment), and `~/.localchat/providers.json`
is read as a fallback - it is written with owner-only permissions, never into
the project directory, which is a place people commit from.

**Why this is small.** The harness asks for tool calls as `<tool_call>` text
rather than through each vendor's function-calling API, so a provider only has
to turn messages into a stream of text. That is a few dozen lines each and no
SDK: `providers.py` normalises the three wire formats into one event shape, and
everything downstream - streaming, `<think>` filtering, tool parsing, titles,
context compression - is untouched by which provider is answering.

The shapes that do differ are handled in one place: Anthropic and Gemini take
the system prompt as its own field rather than a message, Gemini calls the
assistant role `model`, and both want consecutive same-role messages merged -
which the harness produces constantly, since every tool result is its own user
message.

---

## 6. Web Search

A single general search engine answers the *entity* in a query and drops the
term that matters. Asked for `ollama num_ctx meaning` it returns ollama.com, the
Windows download page, and install blogs - none of which contain `num_ctx`. The
old implementation passed those straight to the model, which then answered
confidently and wrongly.

`websearch.py` fixes that in three stages:

1. **Candidates from complementary sources**, run concurrently. A general web
   index is weak on code identifiers, so Stack Exchange and GitHub are queried
   for those and Wikipedia for concepts. Keyword APIs receive a distilled query
   (`ollama num_ctx meaning` becomes `ollama num_ctx`); web engines get the
   original. If a source fails or times out, the rest still return.
2. **The pages are read**, not just their snippets, and split into passages
   ranked by BM25 whose IDF comes from the candidate pool itself - so a term in
   every candidate scores near zero and a rare one dominates. No model, no
   corpus, no network.
3. **A relevance floor.** The query's discriminative terms - explicit
   identifiers, or the rarest term the pool exposes - must actually appear in a
   passage. If nothing clears it, the tool reports that the search found nothing
   and names the pages it rejected, instead of handing over the closest junk.

Everything is free and keyless. To make candidate generation fully local too,
run a [SearXNG](https://github.com/searxng/searxng) instance and point
`SEARXNG_URL` in `.env` at it (e.g. `http://localhost:8080`); it is then used as
the primary source and the public ones stay as backup.

Tuning knobs live in `config.py`: `SEARCH_CANDIDATES`, `SEARCH_FETCH_PAGES`,
`SEARCH_PASSAGE_CHARS`, `SEARCH_RESULT_CHARS`, and the three timeouts.

---

## 7. Skills

A skill is an instruction pack stored on disk that the model pulls in only when
it is relevant. This keeps the system prompt small no matter how many skills
exist: only `name` and `description` are always loaded, and the body arrives
when the model calls `use_skill`.

```
skills/
  git-commit/
    SKILL.md          <- required
    types.md          <- optional bundled files, listed with absolute paths
  quick-skill.md      <- one-file skill
```

Skills are searched in `./skills/` first, then `~/.localchat/skills/`; the first
match on a name wins, so a project skill overrides a personal one.

`SKILL.md` opens with YAML frontmatter:

```markdown
---
name: git-commit
description: Use when the user asks for a commit message. Triggers on "commit", "커밋".
allowed-tools: git_status, git_diff, run_cmd
---

Instructions in plain markdown.
```

- `name` — optional; the folder or file name is used when it is missing.
- `description` — the only thing the model sees before loading, so write it as
  *when to use this* and include the words a user would actually type.
- `allowed-tools` — optional; shown to the model as the tool set the skill
  expects. Advisory, not enforced by the harness.

Two skills ship with the repo: `git-commit` and `code-review`. See
`skills/README.md` for the full format reference.

---

## 8. MCP Servers

[MCP](https://modelcontextprotocol.io) is the standard way to hand an assistant
tools it did not ship with - a filesystem browser, a database, an issue tracker.
Declare a server once and its tools appear alongside the built-in ones.

### Declaring a server

Servers are read from `./.mcp.json` first, then `~/.localchat/mcp.json`; a
project entry wins over a personal one with the same name. Copy
`.mcp.json.example` to get started.

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "D:/work"]
    },
    "github": {
      "type": "http",
      "url": "https://api.githubcopilot.com/mcp/",
      "headers": {"Authorization": "Bearer ${GITHUB_TOKEN}"}
    }
  }
}
```

| Key | Meaning |
| :--- | :--- |
| `command`, `args`, `env`, `cwd` | Start a local server over **stdio**. |
| `url`, `headers` | Talk to a remote server over **streamable HTTP**. |
| `type` | `stdio`, `http`, or `sse` for the deprecated HTTP+SSE transport. Inferred from `command`/`url` when omitted. |
| `disabled` | `true` keeps the entry but does not start it. |
| `timeout` | Seconds to wait for a tool call from this server. |
| `autoApprove` | Tool names on this server that may run without an approval prompt. |
| `trust` | `true` auto-approves every tool on this server. |

`${VAR}` and `${env:VAR}` anywhere in a server entry are replaced with the
matching environment variable, so tokens stay out of the file.

### How it behaves

Every enabled server is started when the app launches, in parallel. Its tools
are fetched and written into the system prompt as `mcp__<server>__<tool>`, with
each tool's JSON Schema flattened into the same parameter list the built-in
tools use. A server that fails to start is reported and skipped - the app runs
without it. Server-declared `instructions` are passed through to the model, and
`readOnlyHint` / `destructiveHint` annotations are surfaced in the tool
description.

Calls go through the same approval prompt as `run_cmd` and file edits, so
nothing runs on an attached server without a `y` - unless `/automode on`,
`autoApprove`, or `trust` says otherwise.

Tuning knobs live in `config.py`: `MCP_ENABLED`, `MCP_STARTUP_TIMEOUT`,
`MCP_CALL_TIMEOUT`, `MCP_HTTP_TIMEOUT`, `MCP_RESULT_CHARS`,
`MCP_MAX_TOOLS_PER_SERVER`, `MCP_TRUSTED_SERVERS`, and
`MCP_AUTO_APPROVE_READONLY`.

### Inspecting and controlling

`/mcp` shows every configured server, its transport, what it exposes, and the
error behind any failure. `/mcp tools` expands the tool list, `/mcp resources`
lists readable resources, `/mcp reload` re-reads the config files and
reconnects, and `/mcp prompt <server> <name> key=value` runs a prompt template
the server offers as your next message.

The protocol client is a self-contained JSON-RPC implementation in
`mcp_client.py` - no SDK dependency, and no new packages to install.

---

## 9. Discord Bot

```bash
pip install -r requirements.txt
export DISCORD_BOT_TOKEN=...          # Windows: set DISCORD_BOT_TOKEN=...
python bot.py
```

`bot.py` does not implement an agent loop of its own. It binds a `Sink` (see
Architecture) and hands the message to `chat_turn`, which knows nothing about
Discord - which is why tools, MCP servers, skills, permissions and providers all
behave the same whatever is watching.

### Talking to it

| Trigger | Where |
| :--- | :--- |
| Mention the bot | any channel it can read |
| Reply to one of its messages | any channel |
| `/call <prompt>` | anywhere, including servers the bot is not in |
| Just type | in a DM |

`/call` is registered as a **user-installable** command
(`allowed_installs(guilds=True, users=True)`), so installing the app on your
account carries it into other people's servers, DMs and group DMs. In those
places Discord gives the app the interaction and nothing else: there is no
channel to read, so `discord_read_messages` and `discord_server_info` say so
instead of guessing.

While the model works you see Discord's own **typing indicator** - not a
placeholder message that has to be cleaned up afterwards. It comes back between
tool calls and stops when the exchange ends.

Answers stream: the reply is posted as soon as the first words arrive and the
message is edited about once a second. Past 2000 characters it continues in the
next message, and the break lands on a **line boundary** - a blank line if there
is a sensible one, otherwise the last newline, and a mid-word cut only for a
single line longer than the whole limit. An open code fence is closed before the
break and reopened after it. Past `DISCORD_MAX_MESSAGES` the remainder arrives
as a `.md` attachment instead.

A **Stop** button rides on the first thing each exchange posts - a message or a
tool card, whichever comes first - and is taken off again when the exchange is
over, so old replies do not sit there with a dead button.

Attachments you send are saved under `workspace/discord/<key>/` and their paths
are handed to the model, which reads them with `read_file`; files the model
writes there can come back with `discord_send_file`.

### Tool calls, in the open

Every tool call becomes an embed posted the moment the call starts and edited
in place when it returns - name, arguments, the first ~900 characters of the
result, and a colour for running / done / failed. MCP tools appear the same way
under their `mcp__<server>__<tool>` names. Nothing runs invisibly.

Approval is three buttons: **Approve**, **Deny**, and **Always allow**, which
writes the rule to `.permissions.discord.json` so the same call never asks
again. Only the person who asked (or an owner) can press them, and a prompt
nobody answers inside `DISCORD_APPROVAL_TIMEOUT` counts as a refusal.
`get_user_input`'s questions become a button per choice plus a text modal, and
`submit_plan_for_approval` becomes an approve / deny / revise card.

### Who may use it

The toolbox reaches the whole machine, so the bot ships closed: with nothing
configured, nobody can talk to it. Put the ids in `.env` - never in `config.py`,
which is a file people commit and publish:

```bash
DISCORD_OWNER_IDS=123456789012345678      # full toolset, behind the approval buttons
DISCORD_ALLOWED_USERS=987654321098765432  # safe, read-only tools
DISCORD_ALLOWED_GUILDS=                   # empty = every guild the bot is in
DISCORD_ALLOW_EVERYONE=false              # true = anyone, still safe tools only
```

(Turn on Developer Mode in Discord, then right-click a user or server → Copy ID.)

Two gates, not one. **A tool policy** decides which tools exist for whoever is
speaking - guests get `DISCORD_SAFE_TOOLS` (search, read, memory, the Discord
tools) and never see `run_cmd`, `write_file`, `delete_file` or `call_api`,
because a blocked tool is left out of the system prompt entirely rather than
refused after the model has already spent a turn on it. **The permission rules**
then decide, per call, what runs without asking - and the bot reads
`.permissions.discord.json` on top of `.permissions.json`, so a rule that should
only ever apply to the bot does not have to be mixed into the shared set. Copy
`.permissions.discord.json.example` to start.

Non-owners never reach a machine-writing tool at all; owners reach them one
button click at a time.

### Sessions, memory and concurrency

One conversation per channel by default (`DISCORD_SESSION_SCOPE = "channel"`,
or `"user"` / `"channel+user"`), saved under `sessions/discord/` and reloaded
when the bot restarts. **Memory is always per person**: the five memory tools
read and write `memory/discord-<user id>.json` for whoever is speaking, so one
user's notes never surface in another's answers even in a shared channel.

Whose memories a turn reads is decided when the turn is *queued*, not when it
finally runs - in a busy channel the speaker changes in between, and a note must
not land in someone else's file. A turn that is stopped half way still saves its
transcript, so the exchange survives a restart.

Channels run concurrently, up to `DISCORD_MAX_CONCURRENT_TURNS`; within one
channel turns queue behind a lock so replies cannot interleave. Tool handlers
are synchronous, so they run in a worker thread and reach back into the bot's
loop for their prompts - which is why a 2-minute `run_cmd` in one channel does
not freeze another. `/stop`, or the **중지** button on a reply, cancels a turn.

### Commands

| Command | Description |
| :--- | :--- |
| `/call <prompt> [file]` | Ask the agent, from anywhere |
| `/new` | Start this conversation over (the old transcript is kept) |
| `/session` | Title, message count, tokens, model, your access level |
| `/export` | Download the conversation as Markdown |
| `/stop` | Cancel the turn in progress |
| `/tools` | The tools available to you here |
| `/memory` | Your own memory entries |
| `/mcp [tools\|reload]` | MCP server status, their tools, reconnect (owner) |
| `/think <on\|off>` | Show or hide a reasoning model's thinking |
| `/model [provider] [model]` | Show the provider; owners can switch it |
| `/perms [allow\|deny] [rule]` | Show or add Discord permission rules (owner) |

### Setting the app up on Discord

1. Developer portal → **Bot** → enable the **Message Content** privileged
   intent (mentions and replies cannot be read without it).
2. **Installation** → enable both *Guild Install* and *User Install*, scopes
   `bot` and `applications.commands`.
3. Invite it with `bot` + `applications.commands`; Send Messages, Embed Links,
   Read Message History and Attach Files are enough.
4. `python bot.py`. Commands are synced globally at startup and can take a few
   minutes to appear the first time.

### Language

The bot speaks English. What the *model* answers in is `config.REPLY_LANGUAGE`:
empty (the default) tells it to reply in whatever language the user wrote in,
and setting it to `"Korean"`, `"English"`, `"日本語"`… pins one for every reply.

Known limits: the provider, the model and the MCP connections are process-wide,
so `/model` changes them for every channel at once; live shell sessions
(`run_cmd`/`send_input`) share one global registry and stay owner-only for that
reason; and a tool already running cannot be interrupted by `/stop`, which
takes effect as soon as it returns.

---

## 10. Logs

Three layers, each one including the layer above it:

| `LOG_LEVEL` | What it adds |
| :--- | :--- |
| `error` | something failed, or was refused |
| `tool` | **(default)** every tool call - who asked, what it did, how it ended - and every approval decision |
| `request` | every message and command that reached the bot |
| `debug` | the harness talking to itself, and discord.py's own logs |

The layers nest on screen as well as in verbosity. Everything one exchange does
carries the same four-character turn id, and its tool lines are indented under
the request that caused them, so a busy channel still reads as separate
conversations instead of one interleaved stream:

```
14:22:01 REQ  a3f2  » alice(77) in #general in srv · "이 서버 정보 알려줘"
14:22:01 TOOL a3f2    ▸ discord_server_info for alice(77)
14:22:02 TOOL a3f2    ✓ discord_server_info · 812 chars · 0.21s
14:22:07 REQ  a3f2  « done · 1 tool(s) · 6.1s
14:23:40 WARN ----    ⛔ stranger(999) is not allowed here (#general in srv)
```

The `tool` layer is the **audit trail**, which is why it sits below errors
rather than beside debug output: a bot that can run shell commands should leave
a record of who asked it to, and of every approval granted, refused or timed
out. Blocked calls - a tool outside someone's policy, or a `deny` rule - are
logged as warnings, so a default install still records them.

```bash
LOG_LEVEL=tool
LOG_FILE=logs/bot.log     # also write there, rotating at 2MB, keeping 3
LOG_PREVIEW=120           # characters of a message to log; 0 logs only its length
```

`LOG_PREVIEW=0` is the privacy setting: requests are still counted and timed,
but no message text is ever written down. Colour is used only when the output
is a terminal, so a log file or `journalctl` stays clean.

---

## 11. Running Commands

`run_cmd` captures the command's output, which means the command can never show
anything to the user while it runs - including a prompt. So it is given no
stdin at all (`DEVNULL`) and a deadline (`config.CMD_TIMEOUT`, 120s).

Without that, an interactive program inherited the terminal and sat waiting for
input, with its prompt captured and invisible: the whole app froze with nothing
on screen and no way to tell what it wanted. A command that simply runs too long
is now stopped along with everything it started, and whatever it printed first
is kept.

### Answering a program while it runs

`run_cmd` does not wait for the command to finish. It stays connected, draining
output as it appears, and when the command is waiting the output so far comes
back with a session id:

```
추측:

[Waiting] 'python3 game.py' is still running and has printed nothing for 0.6s,
so it is most likely waiting for input. Answer it with send_input:
    {"name": "send_input", "arguments": {"session": "s1"}}
```

The model reads the prompt, answers it with `send_input`, and gets whatever the
program prints next - one exchange per turn, until the program exits. That is
how it tests something it has just written: it plays through the program
itself. `end_process` stops one that will not exit. The first few answers can
also be sent up front in a `<stdin>` block on `run_cmd`.

**Telling "waiting" from "busy".** Going quiet proves nothing on its own - a
program that is merely computing looks identical from outside. The best signal
each platform offers is used, strongest first:

| Signal | Where | What it proves |
| :--- | :--- | :--- |
| The output ends without a newline (`추측: `) | everywhere | It is shaped like a prompt |
| `/proc/<pid>/syscall` says a thread is parked in `read()` on fd 0 | Linux | It really is waiting on stdin - and a "no" is trusted too |
| The process tree has burned no CPU | Windows, macOS | It is idle, but sleeping and waiting look the same |
| Silence lasting `CMD_WAIT_TIMEOUT` | everywhere | Nothing better was available |

On Linux the `/proc` answer is exact in both directions, so a `sleep 3` is
simply waited out. Windows and macOS have no equivalent, so idle CPU only
shortens the wait to `CMD_IDLE_GRACE` instead of ending it - a long silent
pause may be offered to the model as a prompt, and an empty `send_input` picks
the output back up when it turns out not to be one.

A command that never stops printing is killed at `CMD_TIMEOUT` along with
everything it started, and at most `CMD_MAX_SESSIONS` live commands are kept.

**Text that is not ASCII.** A command's output is decoded as UTF-8 first,
because UTF-8 is the only candidate that can report that it is wrong - almost
any byte is legal cp949, so guessing a code page first would silently turn good
text into mojibake. If the bytes turn out not to be UTF-8, which is what a
Python older than 3.15 printing to a pipe on Korean Windows produces, the
console code page takes over for the rest of that command - both for what it
prints and for what `send_input` sends back to it.

---

## 12. Tool Permissions

Until now the only gate was the approval prompt, and `/automode on` turned it
off for everything at once - including `run_cmd` and `delete_file`. Rules give
the middle ground.

Rules are read from `./.permissions.json` and `~/.localchat/permissions.json`;
rules from both files apply. Copy `.permissions.json.example` to start.

```json
{
  "allow": ["run_cmd(git status)", "mcp__filesystem__*"],
  "deny":  ["delete_file", "run_cmd(rm *)", "write_file(*/.env)"]
}
```

A rule is a tool name, optionally followed by a pattern in parentheses matched
against the call's main argument - the command for `run_cmd`, the path for a
file tool, the URL for a network tool. Both halves accept `*` and `?`. A pattern
with no wildcard also covers `<pattern> <anything>`, so `run_cmd(git status)`
already allows `git status --short`.

- **deny** never runs and never asks. The tool's handler is never reached, and
  the model is told it is blocked so it stops retrying.
- **allow** runs without a prompt.
- Everything else asks, exactly as before - an empty rule set changes nothing.

The bot adds a **profile** on top: it also reads `.permissions.discord.json`
and `~/.localchat/permissions.discord.json`, and rules saved from an approval
button land in the project one. So the rules that govern a channel full of
people stay separate from the ones you keep for yourself.

At an approval prompt the choices are now `[y/n/a]`, where `a` allows this
exact call from now on and appends the rule to `.permissions.json`. `/perms`
lists the active rules, `/perms allow <rule>` and `/perms deny <rule>` add one
by hand, and `/perms reload` re-reads the files.

---

## 13. Reasoning Models

Reasoning models (qwen3, deepseek-r1, gpt-oss) emit their scratch work before
the answer - either wrapped in `<think>` tags in the content stream, or in
Ollama's separate `thinking` field. It is not the answer, so:

- it is **not printed** (`/think on` shows it dimmed if you want to watch),
- it is **never stored** in the conversation history, which matters most: on a
  local model the context budget is small, and reasoning is often longer than
  the answer it produces.

Set `config.STORE_THINKING = True` to keep it in history anyway, or
`config.SHOW_THINKING = True` to have it shown from startup.

The same stream filter hides the `<tool_call>` tag, so a model that explains
itself and *then* calls a tool shows only the explanation.

---

## 14. Architecture

The codebase is organized cleanly around the following components:

- **`bot.py`**: The entry point - triggers, slash commands, access control, and the per-turn wiring.
- **`ui.py`**: The `Sink` seam every piece of output and every question goes through, plus `ToolPolicy` and a `ConsoleSink` that logs and refuses to answer for anyone.
- **`discord_ui.py`**: `DiscordSink` - streamed message edits, tool embeds, approval / question / plan buttons, and the thread-to-loop bridge.
- **`discord_tools.py`**: The Discord-only tools and the prompt section they are advertised in.
- **`agent.py`**: `AgentSession` - one conversation per place the bot is talking, with its own history, memory namespace, lock and persistence.
- **`logs.py`**: The three log layers, the per-turn id that ties them together, and the rotating file handler.
- **`env.py`**: Reads `.env` into the environment - the one place deployment settings and secrets come from, with no dependency.
- **`config.py`**: Settings. The three that describe a *conversation* rather than the program - loaded skills, session title, token history - are scoped to the running task, so concurrent channels cannot overwrite each other's.
- **`llm_client.py`**: The conversation loop - streaming a reply, parsing the tool calls out of it, running them. Knows nothing about which provider answered, nor about where the answer is being shown.
- **`tools.py`**: Tool implementations and the dispatch table.
- **`skills.py`**: Skill discovery, frontmatter parsing, and on-demand loading.
- **`providers.py`**: The provider abstraction - Ollama, Anthropic, OpenAI, Gemini - and the saved connection.
- **`sse.py`**: Reading server-sent events without waiting for data that has not been sent. Shared by the providers and the MCP client.
- **`permissions.py`**: Permission rule loading, matching, and the allow/deny/ask decision.
- **`shell_session.py`**: Live commands - output draining, waiting-for-input detection, and the session registry.
- **`mcp_client.py`**: MCP transports (stdio / streamable HTTP / SSE), the JSON-RPC session, tool and resource calls, and the prompt section they are advertised in.
- **`websearch.py`**: Multi-source retrieval, page extraction, and BM25 reranking.
- **`context.py`**: Token budgeting, tool-result trimming, and context compression.
- **`session.py`**: Session save/load/list and the persistent memory store, both redirectable per conversation.
- **`systemprompt.py`**: System prompt builder, tool specification schemas (in JSON format), and context summarization prompt definitions.
- **`skills/`**: Project-level skills. Personal skills live in `~/.localchat/skills/`.
- **`.permissions.json`**: Project-level tool permission rules (see `.permissions.json.example`). Personal ones live in `~/.localchat/permissions.json`.
- **`.env`**: Secrets and deployment settings, git-ignored (see `.env.example`). Personal ones can live in `~/.localchat/.env`.
- **`.permissions.discord.json`**: Extra permission rules that apply only to the Discord bot (see `.permissions.discord.json.example`).
- **`workspace/discord/`**: Per-conversation folder for Discord attachments and files the model produces.
- **`memory/discord-<user id>.json`**: One memory store per Discord user.
- **`.mcp.json`**: Project-level MCP server declarations (see `.mcp.json.example`). Personal ones live in `~/.localchat/mcp.json`.
- **`sessions/discord/`**: JSON transcripts, one per conversation, reloaded when the bot restarts.

---

## 15. License

[Nihagosepeungeodachuehasem License](https://github.com/200mill/nihagosepeungeodachuehasem-license)
v1.1 - see [`LICENSE`](LICENSE), or [`LICENSE.ko`](LICENSE.ko) for the Korean
original. In short: do whatever you want with it, and nobody is liable for
anything.
