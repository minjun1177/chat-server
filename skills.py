"""Agent Skills: folder-based instruction packs with progressive disclosure.

A skill lives in `skills/<name>/SKILL.md` (or, for a one-file skill,
`skills/<name>.md`) and opens with YAML frontmatter:

    ---
    name: git-commit
    description: Use when the user asks to write a commit message.
    allowed-tools: read_file, run_cmd
    ---
    <markdown instructions>

Only `name` + `description` reach the system prompt. The body is loaded on
demand, when the model calls the `use_skill` tool, so a large library of skills
costs almost nothing until one is actually needed.

Stdlib only on purpose: `systemprompt.py` imports this module, so importing
`config` here at module level would create an import cycle. The one place that
needs runtime state imports `config` inside the function.
"""

import os
import re


SKILL_FILENAME = "SKILL.md"
USER_SKILL_DIR = os.path.join(os.path.expanduser("~"), ".localchat", "skills")

DESC_MAX_LENGTH = 400
BODY_MAX_LENGTH = 20000
RESOURCE_MAX_COUNT = 40

_SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv"}

_cache: dict | None = None


def skill_dirs() -> list[tuple[str, str]]:
    """(source label, directory) pairs, highest precedence first."""
    return [
        ("project", os.path.abspath("skills")),
        ("user", USER_SKILL_DIR),
    ]


# ---------------------------------------------------------------------------
# frontmatter
# ---------------------------------------------------------------------------

_FRONTMATTER_PATTERN = re.compile(r"^﻿?\s*---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", re.DOTALL)
_KEY_PATTERN = re.compile(r"^([A-Za-z0-9_\-]+)[ \t]*:[ \t]*(.*)$")
_ITEM_PATTERN = re.compile(r"^[ \t]+-[ \t]*(.+)$")


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _parse_yaml_block(block: str) -> dict:
    """Parse the small YAML subset that skill frontmatter actually uses."""
    data = {}
    lines = block.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _KEY_PATTERN.match(line)
        if not m:
            continue
        key = m.group(1).strip().lower()
        raw = m.group(2).strip()

        if raw in (">", "|", ">-", "|-", ">+", "|+"):
            folded = raw.startswith(">")
            buf = []
            while i < len(lines):
                nxt = lines[i]
                if nxt.strip() and not nxt.startswith((" ", "\t")):
                    break
                buf.append(nxt.strip())
                i += 1
            value = " ".join(b for b in buf if b) if folded else "\n".join(buf).strip()
        elif raw:
            value = _unquote(raw)
        else:
            items = []
            while i < len(lines):
                im = _ITEM_PATTERN.match(lines[i])
                if not im:
                    break
                items.append(_unquote(im.group(1)))
                i += 1
            value = items if items else ""

        data[key] = value
    return data


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    m = _FRONTMATTER_PATTERN.match(text)
    if not m:
        return {}, text
    return _parse_yaml_block(m.group(1)), text[m.end():].lstrip("\n")


def _as_tool_list(value) -> list[str]:
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        raw = re.split(r"[,\s]+", value.strip("[]"))
    else:
        raw = []
    return [t.strip().strip("'\"") for t in raw if t.strip().strip("'\"")]


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def _list_resources(skill_dir: str, skill_file: str) -> list[str]:
    found = []
    for root, dirs, files in os.walk(skill_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        for name in sorted(files):
            path = os.path.join(root, name)
            if os.path.abspath(path) == os.path.abspath(skill_file):
                continue
            if name.startswith("."):
                continue
            found.append(path)
            if len(found) >= RESOURCE_MAX_COUNT:
                return found
    return found


def _load_skill_file(path: str, source: str, fallback_name: str, skill_dir: str, with_resources: bool) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return None

    meta, body = _parse_frontmatter(text)
    name = str(meta.get("name") or fallback_name).strip()
    if not name:
        return None

    description = meta.get("description", "")
    if isinstance(description, list):
        description = " ".join(description)
    description = " ".join(str(description).split())

    return {
        "name": name,
        "description": description or "(no description)",
        "body": body.strip(),
        "path": path,
        "dir": skill_dir,
        "source": source,
        "allowed_tools": _as_tool_list(meta.get("allowed-tools") or meta.get("allowed_tools")),
        "resources": _list_resources(skill_dir, path) if with_resources else [],
    }


def discover_skills(force: bool = False) -> dict:
    """Scan the skill directories. Earlier directories win on a name collision."""
    global _cache
    if _cache is not None and not force:
        return _cache

    skills = {}
    for source, directory in skill_dirs():
        if not os.path.isdir(directory):
            continue
        try:
            entries = sorted(os.listdir(directory))
        except Exception:
            continue

        for entry in entries:
            if entry.startswith(".") or entry in _SKIP_DIRS:
                continue
            full = os.path.join(directory, entry)

            if os.path.isdir(full):
                skill_file = os.path.join(full, SKILL_FILENAME)
                if not os.path.isfile(skill_file):
                    continue
                skill = _load_skill_file(skill_file, source, entry, full, True)
            elif entry.lower().endswith(".md") and entry.lower() != "readme.md":
                skill = _load_skill_file(full, source, entry[:-3], directory, False)
            else:
                continue

            if skill and skill["name"].lower() not in {k.lower() for k in skills}:
                skills[skill["name"]] = skill

    _cache = skills
    return skills


def list_skills() -> list[dict]:
    return sorted(discover_skills().values(), key=lambda s: s["name"].lower())


def get_skill(name: str) -> dict | None:
    if not name:
        return None
    key = os.path.basename(str(name).strip().strip("/\\")).lower()
    if key.endswith(".md"):
        key = key[:-3]
    for skill in discover_skills().values():
        if skill["name"].lower() == key:
            return skill
    return None


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def skills_catalog_prompt() -> str:
    """The always-on part: names and descriptions only (progressive disclosure)."""
    skills = list_skills()
    if not skills:
        return ""

    lines = [
        "\n### AVAILABLE SKILLS:",
        "Skills are expert instruction packs stored on disk. Only their names and",
        "descriptions are listed below; the full instructions are loaded on demand.",
        "When a user's request matches a skill's description, call `use_skill` FIRST,",
        "then carry out the task by following the instructions it returns.",
        "",
    ]
    for skill in skills:
        desc = skill["description"]
        if len(desc) > DESC_MAX_LENGTH:
            desc = desc[:DESC_MAX_LENGTH - 1].rstrip() + "…"
        lines.append(f"- {skill['name']}: {desc}")
    lines.append("")
    return "\n".join(lines)


def render_skill(skill: dict) -> str:
    body = skill["body"]
    if len(body) > BODY_MAX_LENGTH:
        omitted = len(body) - BODY_MAX_LENGTH
        body = body[:BODY_MAX_LENGTH] + f"\n...[skill body truncated - {omitted} chars omitted]"

    parts = [
        f"[Skill: {skill['name']}]",
        f"Source: {skill['source']} ({skill['path']})",
        f"Description: {skill['description']}",
    ]
    if skill["allowed_tools"]:
        parts.append("Tools this skill expects to use: " + ", ".join(skill["allowed_tools"]))
    parts.append("")
    parts.append("--- INSTRUCTIONS ---")
    parts.append(body)
    parts.append("--- END INSTRUCTIONS ---")

    if skill["resources"]:
        parts.append("")
        parts.append("Bundled files (open with read_file, or run with run_cmd, using these absolute paths):")
        for path in skill["resources"]:
            parts.append(f"- {path}")

    parts.append("")
    parts.append(
        "[System] Follow these instructions for the current task. They override your "
        "general defaults, but never the user's explicit request or the safety rules "
        "in your system prompt."
    )
    return "\n".join(parts)


def loaded_skill_names(messages: list[dict]) -> list[str]:
    """Skills whose full instructions are still present in a message list.

    Used to rebuild the loaded-skill state when a saved session is restored.
    """
    blob = "\n".join(m.get("content", "") for m in messages)
    return [s["name"] for s in list_skills() if f"[Skill: {s['name']}]\nSource:" in blob]


def handle_use_skill(name: str) -> str:
    import config

    if not name:
        available = ", ".join(s["name"] for s in list_skills()) or "(none)"
        return f"[Error] Skill name is required. Available skills: {available}"

    skill = get_skill(name)
    if skill is None:
        available = ", ".join(s["name"] for s in list_skills()) or "(none)"
        return f"[Error] Skill '{name}' not found. Available skills: {available}"

    if skill["name"] in config.LOADED_SKILLS:
        return (
            f"[Skill: {skill['name']}] Already loaded earlier in this conversation. "
            "Its instructions are still in effect - re-read them above instead of reloading."
        )

    config.LOADED_SKILLS.append(skill["name"])
    return render_skill(skill)


if __name__ == "__main__":
    print("This file can not run directly.")
