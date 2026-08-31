# TODO
> For ai harness

## list
* [x] Upgrade system prompt
* [x] Enhance file write (almost file_edit?)
* [x] When use url_get, do not return raw html
* [x] Memory
* [x] Get user's input.(Choose from a list of options - multiple questions per call)
* [x] Skills (SKILL.md instruction packs, loaded on demand)
* [x] Session titles (AI-generated or set by hand with /title)
* [x] MCP servers (.mcp.json, stdio/http/sse, tools + resources + prompts)
* [x] Hide <tool_call> when the model talks before calling a tool
* [x] Reasoning models: keep <think> off the screen and out of the history
* [x] Tool permissions (.permissions.json allow/deny, 'a' at the prompt)
* [x] Several tool calls in one turn
* [x] Repair console input that arrives as surrogate escapes (Korean input on a cp949 console)
* [x] Raw <content> blocks so a file body never has to be escaped into JSON
* [x] Repair malformed tool JSON (unclosed brackets, flattened arguments, broken payload)
* [x] Ask the model to resend a tool call we could not parse, instead of ending the turn
* [x] run_cmd: no stdin + timeout, so an interactive program can no longer freeze the app
* [x] Live shell sessions: run_cmd hands the prompt to the model, send_input answers it
* [x] Waiting-for-input detection per platform (/proc on Linux, CPU elsewhere) + tests/test_platform.py
* [x] Decode command output by what it actually is (UTF-8, else the console code page)
* [x] /connect: Anthropic, OpenAI (+compatible), Gemini alongside Ollama
* [x] Discord bot: mention / reply / `/call`, user-installable so it works anywhere
* [x] A `Sink` seam so the same agent loop drives a terminal or a chat client
* [x] Tool calls as live embeds, approvals and questions as buttons
* [x] Discord tools: read channel history (bot embeds included), server info
* [x] Per-channel sessions, per-user memory, and a Discord-only permission profile
* [x] .env for secrets and deployment settings, so the repo can be published
