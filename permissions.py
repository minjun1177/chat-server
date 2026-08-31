"""Tool permission rules: what runs without asking, and what never runs.

Until now the only gate was the approval prompt, and `/automode on` turned it
off for everything at once - including `run_cmd` and `delete_file`. Rules give
the middle ground: wave through the calls you make twenty times a day, and put
a hard stop in front of the ones you never want.

Rules live in `.permissions.json` (project) and `~/.localchat/permissions.json`
(personal); both are read and their rules combined.

    {
      "allow": ["read_file", "run_cmd(git status)", "mcp__github__*"],
      "deny":  ["delete_file", "run_cmd(rm *)", "write_file(*/.env)"]
    }

A rule is a tool name, optionally followed by a pattern in parentheses that is
matched against the call's main argument - the command for `run_cmd`, the path
for a file tool, the URL for a network tool. Both halves accept `*` and `?`
wildcards. A pattern with no wildcard also matches anything that starts with it
followed by a space, so `run_cmd(git status)` covers `git status --short`.

`deny` wins over `allow`, and anything unmatched falls through to the approval
prompt exactly as before - an empty rule set changes nothing.
"""

import json
import os
from fnmatch import fnmatch


PROJECT_CONFIG_FILES = (".permissions.json", "permissions.json")
USER_CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".localchat", "permissions.json")

# A profile adds a second pair of files on top of the usual ones, so a bot can
# run under rules that never apply to the terminal. `use_profile("discord")`
# reads .permissions.discord.json and ~/.localchat/permissions.discord.json,
# and rules saved from a Discord approval land in the project one.
_profile = ""


def use_profile(name: str) -> None:
    """Add a named rule profile on top of the default files ("" = none)."""
    global _profile, _loaded
    _profile = (name or "").strip()
    _loaded = False


def profile() -> str:
    return _profile


def _profile_files() -> list[tuple[str, str]]:
    if not _profile:
        return []
    return [
        ("profile", os.path.abspath(f".permissions.{_profile}.json")),
        ("profile-user", os.path.join(os.path.expanduser("~"), ".localchat",
                                      f"permissions.{_profile}.json")),
    ]


def write_target() -> str:
    """The file a newly saved rule goes into."""
    if _profile:
        return os.path.abspath(f".permissions.{_profile}.json")
    return os.path.abspath(PROJECT_CONFIG_FILES[0])

VERDICTS = ("deny", "allow")

# The argument that identifies *what* a call touches, in the order tools use.
_TARGET_KEYS = ("command", "filepath", "dirpath", "src", "url", "uri", "query", "name", "id")

_rules: dict[str, list[tuple[str, str]]] = {}
_sources: list[str] = []
_loaded = False

errors: list[str] = []


def config_paths() -> list[tuple[str, str]]:
    """(source label, path) pairs, highest precedence first."""
    paths = _profile_files()
    paths.extend(("project", os.path.abspath(name)) for name in PROJECT_CONFIG_FILES)
    paths.append(("user", USER_CONFIG_FILE))
    return paths


def load_rules(force: bool = False) -> dict[str, list[tuple[str, str]]]:
    """Read the rule files. Rules from every file apply; deny always wins."""
    global _loaded
    if _loaded and not force:
        return _rules

    _rules.clear()
    _rules.update({verdict: [] for verdict in VERDICTS})
    _sources.clear()
    errors.clear()

    seen_project = False
    for source, path in config_paths():
        if not os.path.isfile(path):
            continue
        if source == "project":
            if seen_project:
                continue        # .permissions.json wins over permissions.json
            seen_project = True
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"{path}: invalid JSON ({e})")
            continue
        except Exception as e:
            errors.append(f"{path}: {e}")
            continue

        _sources.append(path)
        for verdict in VERDICTS:
            entries = data.get(verdict)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, str) and entry.strip():
                    _rules[verdict].append((entry.strip(), path))

    _loaded = True
    return _rules


def rule_sources() -> list[str]:
    load_rules()
    return list(_sources)


def rules_for(verdict: str) -> list[tuple[str, str]]:
    return list(load_rules().get(verdict, []))


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------

def target_for(arguments: dict) -> str:
    """The argument a rule pattern is matched against."""
    if not isinstance(arguments, dict):
        return ""
    for key in _TARGET_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalise(text: str) -> str:
    # Rules are written with forward slashes; Windows paths arrive with both.
    return text.replace("\\", "/").casefold() if os.name == "nt" else text.replace("\\", "/")


def _split_rule(rule: str) -> tuple[str, str]:
    name, sep, pattern = rule.partition("(")
    if not sep:
        return rule.strip(), ""
    return name.strip(), pattern[:-1].strip() if pattern.endswith(")") else pattern.strip()


def matches(rule: str, tool: str, target: str) -> bool:
    name, pattern = _split_rule(rule)
    if not fnmatch(_normalise(tool), _normalise(name)):
        return False
    if not pattern:
        return True
    if not target:
        return False
    subject, wanted = _normalise(target), _normalise(pattern)
    if fnmatch(subject, wanted):
        return True
    # A pattern with no wildcard also covers "<pattern> <anything>", so a rule
    # for `git status` does not have to be rewritten to allow `git status -s`.
    if not any(c in wanted for c in "*?[") and fnmatch(subject, wanted + " *"):
        return True
    return False


def decide(tool: str, arguments: dict) -> tuple[str, str]:
    """Return ("deny"|"allow"|"ask", the rule that decided it)."""
    import config
    if not getattr(config, "PERMISSIONS_ENABLED", True):
        return "ask", ""

    target = target_for(arguments)
    for verdict in VERDICTS:               # deny is checked first and wins
        for rule, _source in rules_for(verdict):
            if matches(rule, tool, target):
                return verdict, rule
    return "ask", ""


def suggest_rule(tool: str, arguments: dict) -> str:
    """The rule that would allow exactly this call - offered at the prompt."""
    target = target_for(arguments)
    return f"{tool}({target})" if target else tool


# ---------------------------------------------------------------------------
# editing
# ---------------------------------------------------------------------------

def add_rule(rule: str, verdict: str = "allow") -> tuple[bool, str]:
    """Append a rule to the project file. Returns (ok, message)."""
    rule = (rule or "").strip()
    if not rule:
        return False, "an empty rule cannot be saved"
    if verdict not in VERDICTS:
        return False, f"unknown verdict '{verdict}'"

    path = write_target()
    data = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception as e:
            return False, f"{path} could not be read ({e})"
    if not isinstance(data, dict):
        return False, f"{path} does not hold a JSON object"

    entries = data.get(verdict)
    if not isinstance(entries, list):
        entries = []
    if rule in entries:
        return True, f"already listed in {path}"
    entries.append(rule)
    data[verdict] = entries

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except Exception as e:
        return False, f"{path} could not be written ({e})"

    load_rules(force=True)
    return True, path


if __name__ == "__main__":
    print("This file can not run directly.")
