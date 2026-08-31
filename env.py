"""Deployment settings, kept out of the source tree.

`config.py` is committed; the things that differ between two people running
this are not. Account ids, API keys, the Discord token, where sessions are
written - those belong in a `.env` file that git never sees, and `config.py`
keeps only the defaults that are safe to publish.

Files are read in this order, and the first one to define a name wins:

    ./.env                      the project, for a machine that runs one bot
    ~/.localchat/.env           personal, outside any repository

A real environment variable always beats both, so a container's own
configuration is never silently overridden by a file someone left behind.
Values are also pushed into `os.environ`, which means `${VAR}` in `.mcp.json`
and the provider key lookups in `providers.py` pick them up for free.

No dependency: a `.env` is a few lines of `KEY=value`, and parsing it here is
smaller than pulling in a package to do it.
"""

import os


PROJECT_FILE = ".env"
USER_FILE = os.path.join(os.path.expanduser("~"), ".localchat", ".env")

warnings: list[str] = []        # things worth telling the user at startup

_values: dict[str, str] = {}
_sources: list[str] = []
_loaded = False


def files() -> list[str]:
    """Every path that is read, highest precedence first."""
    return [os.path.abspath(PROJECT_FILE), USER_FILE]


def sources() -> list[str]:
    """The files that actually existed, highest precedence first."""
    return list(_sources)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        body = value[1:-1]
        if value[0] == '"':
            # Only a double-quoted value takes escapes, as in every other
            # dotenv: a Windows path in single quotes must stay literal.
            body = (body.replace("\\n", "\n").replace("\\t", "\t")
                        .replace('\\"', '"').replace("\\\\", "\\"))
        return body
    comment = value.find(" #")
    if comment != -1:
        value = value[:comment]
    return value.strip()


def _parse(text: str, path: str) -> dict:
    values = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not name:
            warnings.append(f"{path}:{number}: not KEY=value, ignored")
            continue
        if not all(c.isalnum() or c == "_" for c in name):
            warnings.append(f"{path}:{number}: '{name}' is not a variable name, ignored")
            continue
        values[name] = _unquote(value)
    return values


def _check_mode(path: str) -> None:
    """A file holding a bot token should not be readable by everyone."""
    if os.name != "posix":
        return
    try:
        mode = os.stat(path).st_mode
    except OSError:
        return
    if mode & 0o077:
        warnings.append(f"{path} is readable by other users - chmod 600 it")


def load(force: bool = False) -> dict:
    """Read the .env files. Cheap and idempotent; call it whenever."""
    global _loaded
    if _loaded and not force:
        return _values

    _loaded = True
    _values.clear()
    _sources.clear()
    warnings.clear()

    for path in reversed(files()):          # personal first, project overrides
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8-sig") as handle:
                text = handle.read()
        except OSError as error:
            warnings.append(f"{path} could not be read ({error})")
            continue
        _check_mode(path)
        _values.update(_parse(text, path))
        _sources.insert(0, path)

    for name, value in _values.items():
        os.environ.setdefault(name, value)  # a real environment variable wins
    return _values


# ---------------------------------------------------------------------------
# typed lookups
# ---------------------------------------------------------------------------

def raw(name: str):
    load()
    return os.environ.get(name)


def text(name: str, default: str = "") -> str:
    value = raw(name)
    return default if value is None else value.strip()


def flag(name: str, default: bool = False) -> bool:
    value = raw(name)
    if value is None or not value.strip():
        return default
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    warnings.append(f"{name}={value!r} is not true/false, using {default}")
    return default


def integer(name: str, default: int) -> int:
    value = raw(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip(), 0)
    except ValueError:
        warnings.append(f"{name}={value!r} is not a whole number, using {default}")
        return default


def number(name: str, default: float) -> float:
    value = raw(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value.strip())
    except ValueError:
        warnings.append(f"{name}={value!r} is not a number, using {default}")
        return default


def ids(name: str, default=()) -> list[int]:
    """A list of Discord snowflakes: "123, 456" or "123 456"."""
    value = raw(name)
    if value is None or not value.strip():
        return list(default)
    found = []
    for piece in value.replace(",", " ").split():
        try:
            found.append(int(piece))
        except ValueError:
            warnings.append(f"{name}: '{piece}' is not an id, ignored")
    return found


def words(name: str, default=()) -> list[str]:
    """A comma-separated list of names."""
    value = raw(name)
    if value is None or not value.strip():
        return list(default)
    return [piece.strip() for piece in value.split(",") if piece.strip()]


# ---------------------------------------------------------------------------
# a last look before anything is published
# ---------------------------------------------------------------------------

def audit() -> list[str]:
    """Problems worth saying out loud at startup, `.env` being committed first.

    An open-source repository is exactly where a token gets leaked by accident,
    and the moment to catch it is before the bot has ever run with it.
    """
    problems = list(warnings)
    project = os.path.abspath(PROJECT_FILE)
    if not os.path.isfile(project) or not os.path.isdir(".git"):
        return problems

    import subprocess
    try:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", PROJECT_FILE],
                                 capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return problems
    if tracked.returncode == 0:
        problems.append(
            f"{PROJECT_FILE} is tracked by git - it holds your secrets. "
            f"Run: git rm --cached {PROJECT_FILE}  (and add it to .gitignore)")
    return problems


load()
