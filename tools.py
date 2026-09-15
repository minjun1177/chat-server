import os
import re
import json
import subprocess
import time
import shutil
import hashlib
import requests
import psutil
import config
from config import TREE_SITTER_AVAILABLE, _TS_LANGUAGES, _EXT_TO_LANG
from skills import handle_use_skill
import logs
import mcp_client
import discord_tools
import ui
from ui import approval_prompt as _approval_prompt
import permissions
import shell_session
from websearch import search_web as search_pipeline, strip_html, read_page

if TREE_SITTER_AVAILABLE:
    from config import Parser, Query, QueryCursor


def handle_search_web(query: str) -> str:
    return search_pipeline(query)


def _trim_output(text: str) -> str:
    limit = config.CMD_OUTPUT_CHARS
    text = text.rstrip()
    if len(text) <= limit:
        return text
    return ("...[earlier output trimmed]\n" + text[-limit:])


def _session_result(session, output: str, ended: bool, timed_out: bool) -> str:
    """Turn one read from a live command into the text the model sees."""
    output = _trim_output(output)

    if timed_out:
        session.close()
        return (f"[Error] '{session.command}' produced output for {config.CMD_TIMEOUT}s "
                "without stopping, so it was killed. Do not run it again unless you "
                "narrow it down."
                + (f"\n\n{output}" if output else ""))

    if ended:
        code = session.close()
        if code:
            return f"[Error] Command failed (exit code {code}).\n{output}" if output \
                else f"[Error] Command failed (exit code {code})."
        return output or "(the command produced no output)"

    # Still running and gone quiet: from out here that is what a program sitting
    # at a prompt looks like.
    shell_session.register(session)
    return (f"{output}\n\n"
            f"[Waiting] '{session.command}' is still running and has printed nothing for "
            f"{config.CMD_IDLE_TIMEOUT:g}s, so it is most likely waiting for input. "
            f"Read the output above and answer it with send_input:\n"
            f'    {{"name": "send_input", "arguments": {{"session": "{session.id}"}}}}\n'
            "    <stdin>\n    <your answer>\n    </stdin>\n"
            "Send an empty <stdin> block to just wait for more output, or call "
            f'end_process with session "{session.id}" to stop it.').strip()


def safe_run_cmd(command_string: str, stdin_text: str = "") -> str:
    """Run a shell command, and stay connected to it while it runs.

    The command is handed to the shell as written. It used to be split with
    `shlex` and then passed to `shell=True`, which mangled Windows paths (shlex
    eats backslashes) and left the shell to re-quote a list it had never been
    given - so `run_cmd` and what actually ran could differ.

    The process is not waited on to completion. Its output is drained as it
    appears, and when it goes quiet the output so far is returned along with the
    session id, so the model can read the prompt and answer it with `send_input`.
    That is the only way an interactive program can work here: its prompts are
    captured, so the user never sees them, and letting it hold the terminal just
    froze the app with nothing on screen.
    """
    command = (command_string or "").strip()
    if not command:
        return "[Error] Empty command."

    stdin_text = stdin_text if isinstance(stdin_text, str) else ""

    details = [("command", command)]
    if stdin_text.strip():
        preview = stdin_text if len(stdin_text) <= 200 else stdin_text[:200] + "...[truncated]"
        details.append(("input", preview.replace("\n", " ⏎ ").strip()))
    if not _approval_prompt("Run Command", details, rule=f"run_cmd({command})"):
        return "[System] User denied command execution."

    try:
        session = shell_session.start(command)
    except Exception as e:
        return f"[Error] Failed to execute command: {e}"

    if stdin_text.strip():
        session.send(stdin_text)

    try:
        output, ended, timed_out = session.read_until_idle(
            config.CMD_IDLE_TIMEOUT, time.time() + config.CMD_TIMEOUT)
    except KeyboardInterrupt:
        session.close()
        return "[System] The command was cancelled by the user."

    return _session_result(session, output, ended, timed_out)


def handle_send_input(session_id: str, text: str) -> str:
    """Answer a running command's prompt and read what it says next."""
    shell_session.prune()
    session = shell_session.get(session_id)
    if session is None:
        running = shell_session.active()
        if not running:
            return ("[Error] No command is waiting for input. Start one with run_cmd "
                    "first - a command that has already finished cannot be resumed.")
        listing = ", ".join(f"'{s.id}' ({s.command})" for s in running)
        return f"[Error] No session '{session_id}'. Currently waiting: {listing}"

    text = text if isinstance(text, str) else str(text or "")
    details = [("session", f"{session.id}  ({session.command})"),
               ("input", text.strip().replace("\n", " ⏎ ") or "(nothing - just wait)")]
    if not _approval_prompt("Send Input", details, rule=f"send_input({session.command})"):
        return "[System] User denied sending input."

    if text.strip() and not session.send(text):
        code = session.close()
        return (f"[Error] '{session.command}' had already exited (code {code}), "
                "so the input went nowhere.")

    try:
        output, ended, timed_out = session.read_until_idle(
            config.CMD_IDLE_TIMEOUT, time.time() + config.CMD_TIMEOUT)
    except KeyboardInterrupt:
        session.close()
        return "[System] The command was cancelled by the user."

    return _session_result(session, output, ended, timed_out)


def handle_end_process(session_id: str) -> str:
    session = shell_session.get(session_id)
    if session is None:
        return f"[Error] No running session '{session_id}'."
    command, sid = session.command, session.id
    session.close()
    return f"[Success] Stopped '{command}' (session {sid})."


_HASHLINE_PATTERN = re.compile(r'^\d+:[0-9a-f]{2}\|')

def _line_hash(line: str) -> str:
    return hashlib.md5(line.encode("utf-8")).hexdigest()[:2]

def _encode_hashlines(content: str) -> str:
    lines = content.split("\n")
    result = []
    for i, line in enumerate(lines, 1):
        h = _line_hash(line)
        result.append(f"{i}:{h}|{line}")
    return "\n".join(result)

def _strip_hashlines(content: str) -> str:
    lines = content.split("\n")
    if not lines:
        return content

    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return content
    matched = sum(1 for l in non_empty if _HASHLINE_PATTERN.match(l))
    ratio = matched / len(non_empty)

    if ratio < 0.8:
        return content

    stripped = []
    for line in lines:
        m = _HASHLINE_PATTERN.match(line)
        if m:
            stripped.append(line[m.end():])
        else:
            stripped.append(line)
    return "\n".join(stripped)



def handle_read_file(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        if not config.RETURN_ALL_FILE_CONTENT and len(content) > config.FILE_MAX_DISPLAY_LENGTH:
            content = content[:config.FILE_MAX_DISPLAY_LENGTH] + "\n...[Too long]..."
        return _encode_hashlines(content)
    except Exception as e:
        return f"[Error] Cannot read file: {e}"

def handle_write_file(filepath: str, content: str) -> str:
    if not filepath:
        return ("[Error] No 'filepath' was given. Send it inside \"arguments\" in the "
                "tool call JSON.")

    content = _strip_hashlines(content)

    # A write with no body is a malformed call, not a request for an empty file -
    # the body went missing between the model and here. Saying so lets the model
    # resend it; writing it out would quietly destroy an existing file.
    if not content.strip():
        existing = os.path.isfile(filepath) and os.path.getsize(filepath) > 0
        return ("[Error] The call carried no content, so nothing was written"
                f"{' (the existing ' + filepath + ' was left untouched)' if existing else ''}. "
                "Send write_file again with the file body in a <content> block right "
                "after the JSON object.")

    preview = content if len(content) <= 200 else content[:200] + "...[truncated]"
    approved = _approval_prompt("Write File", [("path", filepath), ("preview", preview)],
                                rule=f"write_file({filepath})")
    if not approved: return "[System] User denied file write."
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[Success] File written: {filepath}"
    except Exception as e:
        return f"[Error] Cannot write file: {e}"

def handle_edit_file(filepath: str, old_content: str, new_content: str) -> str:
    old_content = _strip_hashlines(old_content)
    new_content = _strip_hashlines(new_content)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            file_content = f.read()
    except Exception as e:
        return f"[Error] Cannot read file: {e}"

    if old_content not in file_content:
        return "[Error] The specified old_content was not found in the file."
    count = file_content.count(old_content)
    if count > 1:
        return f"[Error] old_content found {count} times. Please provide a more specific snippet."

    old_preview = old_content[:150] + ('...' if len(old_content) > 150 else '')
    new_preview = new_content[:150] + ('...' if len(new_content) > 150 else '')
    approved = _approval_prompt("Edit File", [("path", filepath), ("from", old_preview), ("to", new_preview)],
                                rule=f"edit_file({filepath})")
    if not approved: return "[System] User denied file edit."

    new_file_content = file_content.replace(old_content, new_content, 1)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_file_content)
        return f"[Success] File edited: {filepath}"
    except Exception as e:
        return f"[Error] Cannot write file: {e}"

def handle_delete_file(filepath: str) -> str:
    approved = _approval_prompt("Delete File", [("path", filepath)], rule=f"delete_file({filepath})")
    if not approved: return "[System] User denied file deletion."
    try:
        os.remove(filepath)
        return f"[Success] File deleted: {filepath}"
    except Exception as e:
        return f"[Error] Cannot delete file: {e}"

def handle_copy_file(src: str, dst: str) -> str:
    approved = _approval_prompt("Copy File", [("from", src), ("to", dst)], rule=f"copy_file({src})")
    if not approved: return "[System] User denied file copy."
    try:
        shutil.copy2(src, dst)
        return f"[Success] File copied to: {dst}"
    except Exception as e:
        return f"[Error] Cannot copy file: {e}"

def handle_create_dir(dirpath: str) -> str:
    approved = _approval_prompt("Create Directory", [("path", dirpath)], rule=f"create_dir({dirpath})")
    if not approved: return "[System] User denied directory creation."
    try:
        os.makedirs(dirpath, exist_ok=True)
        return f"[Success] Directory created: {dirpath}"
    except Exception as e:
        return f"[Error] Cannot create directory: {e}"

def handle_get_url(url: str) -> str:
    try:
        content = read_page(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if not config.RETURN_ALL_FILE_CONTENT and len(content) > config.FILE_MAX_DISPLAY_LENGTH:
            content = content[:config.FILE_MAX_DISPLAY_LENGTH] + "\n...[Too long]..."
        return content
    except Exception as e:
        return f"[Error] Cannot fetch URL: {e}"

def _as_option_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalise_questions(what_do, prompt, questions) -> list[dict]:
    """Accept every shape a model might send and return [{question, options}].

    The preferred shape is a list of questions, each with its own options. The
    old single-question form (`what_do` plus a flat `prompt` list of choices) is
    still understood, because that is what half the models will keep emitting.
    """
    items = []

    if isinstance(questions, dict):
        questions = [questions]
    if isinstance(questions, list):
        for entry in questions:
            if isinstance(entry, dict):
                text = str(entry.get("question") or entry.get("prompt")
                           or entry.get("what_do") or "").strip()
                options = _as_option_list(entry.get("options") or entry.get("choices")
                                          or entry.get("answers"))
            else:
                text, options = str(entry).strip(), []
            if text:
                items.append({"question": text, "options": options})
    elif isinstance(questions, str) and questions.strip():
        items.append({"question": questions.strip(), "options": []})

    if items:
        return items

    # The old form: one question in `what_do`, its choices in `prompt`.
    text = str(what_do or "").strip()
    options = _as_option_list(prompt)
    if not text and options:
        text, options = options[0], options[1:]
    if text:
        items.append({"question": text, "options": options})
    return items


def _ask_one(question: str, options: list[str], index: int, total: int) -> str | None:
    """Ask the user one question. None means they gave up.

    Discord shows a button per choice and a modal for free text; a sink with
    nobody behind it returns None, which the caller reports as "cancelled".
    """
    return ui.current().ask_choice(question, options, index, total)


def handle_get_input(what_do="", prompt=None, questions=None) -> str:
    items = _normalise_questions(what_do, prompt, questions)
    if not items:
        return "[Error] No question was given. Pass 'questions': [{'question': ..., 'options': [...]}]."

    answers = []
    for i, item in enumerate(items, 1):
        answer = _ask_one(item["question"], item["options"], i, len(items))
        if answer is None:
            if not answers:
                return "[System] User cancelled the questions without answering."
            return ("[System] User stopped after answering the first "
                    f"{len(answers)} question(s):\n" + _format_answers(answers))
        answers.append((item["question"], answer))

    if len(answers) == 1:
        return answers[0][1]
    return _format_answers(answers)


def _format_answers(answers: list[tuple[str, str]]) -> str:
    lines = []
    for i, (question, answer) in enumerate(answers, 1):
        lines.append(f"{i}. {question}\n   -> {answer}")
    return "\n".join(lines)

def handle_list_dir(dirpath: str) -> str:
    if not os.path.exists(dirpath):
        return f"[Error] Directory not found: {dirpath}"
    try:
        items = os.listdir(dirpath)
        formatted_items = []
        for item in items:
            if os.path.isdir(os.path.join(dirpath, item)):
                formatted_items.append(f"{item}/ (Dir)")
            else:
                formatted_items.append(f"{item} (File)")
        return "\n".join(formatted_items) if formatted_items else "[Empty Directory]"
    except Exception as e:
        return f"[Error] list_dir failed: {e}"

def handle_git_status() -> str:
    try:
        res = subprocess.run(["git", "status"], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        return res.stdout or res.stderr
    except Exception as e:
        return f"[Error] git status failed: {e}"

def handle_git_diff() -> str:
    try:
        res = subprocess.run(["git", "diff"], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        return res.stdout or res.stderr
    except Exception as e:
        return f"[Error] git diff failed: {e}"

def handle_get_system_info() -> str:
    try:
        cpu_percent = psutil.cpu_percent(interval=0.5)
        cpu_count = psutil.cpu_count(logical=True)

        mem = psutil.virtual_memory()
        mem_total_gb = mem.total / (1024 ** 3)
        mem_used_gb = mem.used / (1024 ** 3)

        disk = psutil.disk_usage(os.getcwd())
        disk_total_gb = disk.total / (1024 ** 3)
        disk_used_gb = disk.used / (1024 ** 3)

        processes = []
        for p in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
            try:
                processes.append(p.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        top_processes = sorted(processes, key=lambda x: x['memory_percent'] or 0, reverse=True)[:5]
        proc_lines = [
            f"  - PID {p['pid']}: {p['name']} (RAM: {p['memory_percent']:.1f}%)"
            for p in top_processes
        ]

        info = (
            f"[System Status]\n"
            f"\u2022 CPU Usage: {cpu_percent}% ({cpu_count} cores)\n"
            f"\u2022 RAM Usage: {mem.percent}% ({mem_used_gb:.1f}GB / {mem_total_gb:.1f}GB)\n"
            f"\u2022 Disk Usage: {disk.percent}% ({disk_used_gb:.1f}GB / {disk_total_gb:.1f}GB)\n\n"
            f"[Top Memory Processes]\n" + "\n".join(proc_lines)
        )
        return info

    except Exception as e:
        return f"[Error] Failed to fetch system info: {e}"


def handle_search_in_file(query: str, is_regex: bool = False) -> str:
    if not query:
        return "[Error] Search query is required."
    try:
        if is_regex:
            try:
                pattern = re.compile(query)
            except re.error as e:
                return f"[Error] Invalid regex pattern: {e}"
        else:
            pattern = re.compile(re.escape(query))

        matches = []
        max_results = 100
        search_root = os.getcwd()

        for root, dirs, files in os.walk(search_root):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('venv', '__pycache__', 'node_modules', '.git')]
            for filename in files:
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, search_root)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        for line_num, line in enumerate(f, 1):
                            if pattern.search(line):
                                line_preview = line.rstrip()
                                if len(line_preview) > 200:
                                    line_preview = line_preview[:200] + "..."
                                matches.append(f"{rel_path}:{line_num}: {line_preview}")
                                if len(matches) >= max_results:
                                    break
                except (PermissionError, OSError):
                    continue
                if len(matches) >= max_results:
                    break
            if len(matches) >= max_results:
                break

        if not matches:
            return f"[Search] No matches found for: {query}"
        header = f"[Search] Found {len(matches)} match(es) for '{query}':\n"
        if len(matches) >= max_results:
            header = f"[Search] Showing first {max_results} matches for '{query}' (more may exist):\n"
        return header + "\n".join(matches)
    except Exception as e:
        return f"[Error] Search failed: {e}"


def handle_call_api(url: str, method: str, headers: str = "", payload: str = "") -> str:
    if not url:
        return "[Error] URL is required."
    if not method:
        return "[Error] HTTP method is required."

    method = method.upper()
    if method not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
        return f"[Error] Unsupported HTTP method: {method}"

    approved = _approval_prompt("API Call", [("URL", url), ("Method", method)], rule=f"call_api({url})")
    if not approved:
        return "[System] User denied API call."

    try:
        req_headers = {"User-Agent": "LocalChat/1.0"}
        if headers:
            if isinstance(headers, str):
                try:
                    parsed = json.loads(headers)
                    if isinstance(parsed, dict):
                        req_headers.update(parsed)
                except json.JSONDecodeError:
                    return "[Error] Invalid headers JSON format. Expected a JSON object like {\"Key\": \"Value\"}."
            elif isinstance(headers, dict):
                req_headers.update(headers)

        req_body = None
        if payload:
            if isinstance(payload, str):
                try:
                    req_body = json.loads(payload)
                except json.JSONDecodeError:
                    req_body = payload
            elif isinstance(payload, dict):
                req_body = payload

        if method == "GET":
            resp = requests.get(url, headers=req_headers, timeout=30)
        elif method == "POST":
            resp = requests.post(url, headers=req_headers, json=req_body if isinstance(req_body, dict) else None, data=req_body if isinstance(req_body, str) else None, timeout=30)
        elif method == "PUT":
            resp = requests.put(url, headers=req_headers, json=req_body if isinstance(req_body, dict) else None, data=req_body if isinstance(req_body, str) else None, timeout=30)
        elif method == "PATCH":
            resp = requests.patch(url, headers=req_headers, json=req_body if isinstance(req_body, dict) else None, data=req_body if isinstance(req_body, str) else None, timeout=30)
        elif method == "DELETE":
            resp = requests.delete(url, headers=req_headers, timeout=30)

        content = resp.text
        if not config.RETURN_ALL_FILE_CONTENT and len(content) > config.FILE_MAX_DISPLAY_LENGTH:
            content = content[:config.FILE_MAX_DISPLAY_LENGTH] + "\n...[Too long]..."

        result = (
            f"[API Response]\n"
            f"\u2022 Status: {resp.status_code} {resp.reason}\n"
            f"\u2022 Content-Type: {resp.headers.get('Content-Type', 'unknown')}\n\n"
            f"{content}"
        )
        return result
    except requests.exceptions.Timeout:
        return "[Error] API request timed out (30s)."
    except requests.exceptions.ConnectionError:
        return f"[Error] Could not connect to: {url}"
    except Exception as e:
        return f"[Error] API call failed: {e}"



def _get_ts_parser(lang_name: str):
    lang = _TS_LANGUAGES.get(lang_name)
    if lang is None:
        return None, None
    parser = Parser(lang)
    return parser, lang


def _detect_language(filepath: str) -> str | None:
    _, ext = os.path.splitext(filepath)
    return _EXT_TO_LANG.get(ext.lower())


def _extract_params_python(node, src: bytes) -> list[dict]:
    params = []
    param_node = None
    for child in node.children:
        if child.type == "parameters":
            param_node = child
            break
    if param_node is None:
        return params
    for child in param_node.children:
        if child.type in ("identifier",):
            params.append({"name": child.text.decode(), "type": None, "default": None})
        elif child.type == "typed_parameter":
            name = default = ptype = None
            for sub in child.children:
                if sub.type == "identifier" and name is None:
                    name = sub.text.decode()
                elif sub.type == "type":
                    ptype = sub.text.decode()
            params.append({"name": name, "type": ptype, "default": None})
        elif child.type == "default_parameter":
            name = default = None
            for sub in child.children:
                if sub.type == "identifier" and name is None:
                    name = sub.text.decode()
                elif sub.type not in ("=",):
                    default = sub.text.decode()
            params.append({"name": name, "type": None, "default": default})
        elif child.type == "typed_default_parameter":
            name = default = ptype = None
            for sub in child.children:
                if sub.type == "identifier" and name is None:
                    name = sub.text.decode()
                elif sub.type == "type":
                    ptype = sub.text.decode()
                elif sub.type not in ("=", ":"):
                    default = sub.text.decode()
            params.append({"name": name, "type": ptype, "default": default})
        elif child.type in ("list_splat_pattern", "dictionary_splat_pattern"):
            prefix = "*" if child.type == "list_splat_pattern" else "**"
            for sub in child.children:
                if sub.type == "identifier":
                    params.append({"name": prefix + sub.text.decode(), "type": None, "default": None})
    return params


def _extract_params_generic(node, src: bytes) -> list[dict]:
    params = []
    for child in node.children:
        if child.type in ("formal_parameters", "parameter_list", "parameters",
                          "formal_parameter_list"):
            for p in child.named_children:
                params.append({"name": p.text.decode(), "type": None, "default": None})
            break
    return params


def _extract_return_type(node, src: bytes) -> str | None:
    for child in node.children:
        if child.type == "type":
            return child.text.decode()
        if child.type == "return_type":
            return child.text.decode()
    return None


def _extract_decorators(node, src: bytes) -> list[str]:
    decorators = []
    sibling = node.prev_named_sibling
    while sibling and sibling.type == "decorator":
        decorators.insert(0, sibling.text.decode())
        sibling = sibling.prev_named_sibling
    for child in node.children:
        if child.type == "decorator":
            decorators.append(child.text.decode())
    return decorators


_FUNC_TYPES = {
    "function_definition", "function_declaration",
    "method_definition", "method_declaration",
    "arrow_function", "generator_function_declaration",
    "function_item",
}
_CLASS_TYPES = {
    "class_definition", "class_declaration",
    "struct_item", "enum_item", "impl_item",
    "interface_declaration", "enum_declaration",
    "type_declaration",
}


def _walk_skeleton(node, src: bytes, lang: str) -> list[dict]:
    results = []
    for child in node.children:
        entry = None
        if child.type in _FUNC_TYPES:
            name_node = child.child_by_field_name("name")
            name = name_node.text.decode() if name_node else "<anonymous>"
            if lang == "python":
                params = _extract_params_python(child, src)
            else:
                params = _extract_params_generic(child, src)
            ret = _extract_return_type(child, src)
            decorators = _extract_decorators(child, src) if lang == "python" else []
            entry = {
                "type": "function",
                "name": name,
                "line": child.start_point[0] + 1,
                "end_line": child.end_point[0] + 1,
                "parameters": params,
                "return_type": ret,
            }
            if decorators:
                entry["decorators"] = decorators
            body = child.child_by_field_name("body")
            if body:
                inner = _walk_skeleton(body, src, lang)
                if inner:
                    entry["children"] = inner

        elif child.type in _CLASS_TYPES:
            name_node = child.child_by_field_name("name")
            name = name_node.text.decode() if name_node else "<anonymous>"
            bases = []
            for sub in child.children:
                if sub.type in ("argument_list", "superclass", "type_list",
                                "class_heritage"):
                    bases.append(sub.text.decode())
            entry = {
                "type": "class",
                "name": name,
                "line": child.start_point[0] + 1,
                "end_line": child.end_point[0] + 1,
                "bases": bases,
            }
            body = child.child_by_field_name("body")
            if body:
                inner = _walk_skeleton(body, src, lang)
                if inner:
                    entry["methods"] = inner
            else:
                inner = _walk_skeleton(child, src, lang)
                if inner:
                    entry["methods"] = inner

        if entry:
            results.append(entry)
        else:
            deeper = _walk_skeleton(child, src, lang)
            results.extend(deeper)
    return results


def handle_get_code_skeleton(file_path: str) -> str:
    if not TREE_SITTER_AVAILABLE:
        return "[Error] tree-sitter is not installed. Run: pip install tree-sitter tree-sitter-python (and other language grammars)"

    if not file_path:
        return "[Error] file_path is required."

    if not os.path.isfile(file_path):
        return f"[Error] File not found: {file_path}"

    lang_name = _detect_language(file_path)
    if lang_name is None:
        return (f"[Error] Unsupported file extension. "
                f"Supported: {', '.join(sorted(set(_EXT_TO_LANG.values())))}")

    parser, lang = _get_ts_parser(lang_name)
    if parser is None:
        return f"[Error] Language grammar not available: {lang_name}"

    try:
        with open(file_path, "rb") as f:
            src = f.read()
    except Exception as e:
        return f"[Error] Failed to read file: {e}"

    tree = parser.parse(src)
    skeleton = _walk_skeleton(tree.root_node, src, lang_name)

    result = {
        "file": file_path,
        "language": lang_name,
        "skeleton": skeleton,
    }
    return json.dumps(result, indent=2, ensure_ascii=False)


def handle_query_ast_node(file_path: str, pattern: str, language: str = "") -> str:
    if not TREE_SITTER_AVAILABLE:
        return "[Error] tree-sitter is not installed."

    if not file_path:
        return "[Error] file_path is required."
    if not pattern:
        return "[Error] pattern is required."

    if not os.path.isfile(file_path):
        return f"[Error] File not found: {file_path}"

    lang_name = language.strip().lower() if language else _detect_language(file_path)
    if not lang_name or lang_name not in _TS_LANGUAGES:
        return (f"[Error] Could not determine language. "
                f"Supported: {', '.join(sorted(_TS_LANGUAGES.keys()))}")

    parser, lang = _get_ts_parser(lang_name)
    if parser is None:
        return f"[Error] Language grammar not available: {lang_name}"

    try:
        with open(file_path, "rb") as f:
            src = f.read()
    except Exception as e:
        return f"[Error] Failed to read file: {e}"

    tree = parser.parse(src)

    try:
        query = Query(lang, pattern)
    except Exception as e:
        return f"[Error] Invalid Tree-sitter query pattern: {e}"

    cursor = QueryCursor(query)
    matches_raw = list(cursor.matches(tree.root_node))

    if not matches_raw:
        return json.dumps({"file": file_path, "pattern": pattern, "matches": []},
                          indent=2, ensure_ascii=False)

    lines = src.split(b"\n")
    results = []
    max_results = 50

    for pattern_idx, captures in matches_raw:
        if len(results) >= max_results:
            break
        for cap_name, nodes in captures.items():
            for node in nodes:
                if len(results) >= max_results:
                    break
                start_line = node.start_point[0]
                end_line = node.end_point[0]
                ctx_start = max(0, start_line - 1)
                ctx_end = min(len(lines) - 1, end_line + 1)
                if ctx_end - ctx_start > 10:
                    ctx_end = ctx_start + 10
                snippet_lines = []
                for i in range(ctx_start, ctx_end + 1):
                    prefix = ">>>" if start_line <= i <= end_line else "   "
                    snippet_lines.append(f"{prefix} {i + 1}: {lines[i].decode(errors='replace')}")
                results.append({
                    "capture": cap_name,
                    "node_type": node.type,
                    "text": node.text.decode(errors="replace"),
                    "start_line": start_line + 1,
                    "end_line": end_line + 1,
                    "start_col": node.start_point[1],
                    "end_col": node.end_point[1],
                    "snippet": "\n".join(snippet_lines),
                })

    output = {
        "file": file_path,
        "language": lang_name,
        "pattern": pattern,
        "total_matches": len(results),
        "matches": results,
    }
    return json.dumps(output, indent=2, ensure_ascii=False)


def handle_submit_plan_for_approval(context_discovered: str, diff_blueprint: str, verification_steps: str) -> str:
    return ui.current().ask_plan(context_discovered, diff_blueprint, verification_steps)


def handle_mcp_tool_call(function_name: str, arguments: dict) -> str:
    """Run a tool that lives on an attached MCP server."""
    resolved = mcp_client.resolve_tool(function_name)
    if resolved is None:
        available = ", ".join(mcp_client.available_tool_names()) or "(none)"
        return f"[Error] No MCP tool named '{function_name}'. Available MCP tools: {available}"

    server, tool = resolved
    tool_name = str(tool.get("name", ""))

    if not mcp_client.auto_approved(server, tool_name):
        details = [("server", server.name), ("tool", tool_name)]
        for key, value in arguments.items():
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            details.append((key, text))
        if not _approval_prompt("MCP Tool Call", details):
            return f"[System] User denied the MCP call '{tool_name}' on server '{server.name}'."

    return mcp_client.call_tool(server, tool_name, arguments)


def handle_list_mcp_resources(server_name: str = "") -> str:
    return mcp_client.list_resources_text(server_name)


def handle_read_mcp_resource(uri: str, server_name: str = "") -> str:
    return mcp_client.read_resource_text(uri, server_name)


def coerce_arguments(arguments) -> dict:
    """Small models sometimes emit "arguments" as a JSON string, not an object."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return arguments


def dispatch_tool(function_name: str, arguments: dict) -> str | None:
    arguments = coerce_arguments(arguments)

    # Which tools exist at all for this conversation. On a terminal that is
    # everything; on Discord a guest never even sees run_cmd, so a blocked call
    # is refused here without reaching permissions or a handler.
    policy = ui.policy()
    if not policy.permits(function_name):
        logs.blocked(f"{function_name} is not in this conversation's tool set "
                     f"({policy.label or 'default'})")
        return policy.refusal(function_name)

    verdict, rule = permissions.decide(function_name, arguments)
    if verdict == "deny":
        logs.blocked(f"{function_name} denied by permission rule: {rule}")
        return (f"[System] '{function_name}' is blocked by your permission rules "
                f"(rule: {rule}). Do not retry it; tell the user it is blocked.")

    token = config.set_policy_auto_allow(verdict == "allow")
    try:
        return _run_tool(function_name, arguments)
    finally:
        config.reset_policy_auto_allow(token)


def _run_tool(function_name: str, arguments: dict) -> str | None:
    from session import (handle_write_memory, handle_get_memory_list,
                         handle_read_memory, handle_delete_memory, handle_edit_memory)
    if function_name == "search_web": return handle_search_web(arguments.get("query", ""))
    if function_name == "run_cmd": return safe_run_cmd(arguments.get("command", ""), arguments.get("stdin", "") or arguments.get("input", ""))
    if function_name == "send_input": return handle_send_input(arguments.get("session", ""), arguments.get("stdin", "") or arguments.get("input", "") or arguments.get("text", ""))
    if function_name == "end_process": return handle_end_process(arguments.get("session", ""))
    if function_name == "read_file": return handle_read_file(arguments.get("filepath", ""))
    if function_name == "get_url": return handle_get_url(arguments.get("url", ""))
    if function_name == "write_file": return handle_write_file(arguments.get("filepath", ""), arguments.get("content", ""))
    if function_name == "edit_file": return handle_edit_file(arguments.get("filepath", ""), arguments.get("old_content", ""), arguments.get("new_content", ""))
    if function_name == "delete_file": return handle_delete_file(arguments.get("filepath", ""))
    if function_name == "copy_file": return handle_copy_file(arguments.get("src", ""), arguments.get("dst", ""))
    if function_name == "list_dir": return handle_list_dir(arguments.get("dirpath", ""))
    if function_name == "create_dir": return handle_create_dir(arguments.get("dirpath", ""))
    if function_name == "git_status": return handle_git_status()
    if function_name == "git_diff": return handle_git_diff()
    if function_name == "write_memory": return handle_write_memory(arguments.get("id", ""), arguments.get("content", ""))
    if function_name == "get_memory_list": return handle_get_memory_list()
    if function_name == "read_memory": return handle_read_memory(arguments.get("id", ""))
    if function_name == "delete_memory": return handle_delete_memory(arguments.get("id", ""))
    if function_name == "edit_memory": return handle_edit_memory(arguments.get("id", ""), arguments.get("new_content", ""))
    if function_name == "get_user_input": return handle_get_input(arguments.get("what_do", ""), arguments.get("prompt", []), arguments.get("questions"))
    if function_name == "search_in_file": return handle_search_in_file(arguments.get("query", ""), arguments.get("is_regex", False))
    if function_name == "get_system_info": return handle_get_system_info()
    if function_name == "call_api": return handle_call_api(arguments.get("url", ""), arguments.get("method", ""), arguments.get("headers",""), arguments.get("payload",""))
    if function_name == "get_code_skeleton": return handle_get_code_skeleton(arguments.get("file_path", ""))
    if function_name == "query_ast_node": return handle_query_ast_node(arguments.get("file_path", ""), arguments.get("pattern", ""), arguments.get("language", ""))
    if function_name == "submit_plan_for_approval": return handle_submit_plan_for_approval(arguments.get("context_discovered", ""), arguments.get("diff_blueprint", ""), arguments.get("verification_steps", ""))
    if function_name in ("use_skill", "skill"): return handle_use_skill(arguments.get("skill_name", "") or arguments.get("name", "") or arguments.get("skill", ""))
    if function_name == "list_mcp_resources": return handle_list_mcp_resources(arguments.get("server", "") or arguments.get("server_name", ""))
    if function_name == "read_mcp_resource": return handle_read_mcp_resource(arguments.get("uri", ""), arguments.get("server", "") or arguments.get("server_name", ""))
    if mcp_client.is_mcp_tool(function_name): return handle_mcp_tool_call(function_name, arguments)
    if discord_tools.is_discord_tool(function_name): return discord_tools.dispatch(function_name, arguments)

    logs.warn(f"unknown tool: {function_name}")
    return None
