"""Live shell sessions: run a command, and talk to it while it is still running.

A captured `subprocess.run` can only ever report what a command printed after it
finished, which is no use for anything that asks a question. Here the process is
kept alive instead: its output is drained continuously, and the moment it goes
quiet - which is what a program waiting at a prompt looks like from outside - the
output so far is handed to the model, which replies with the next line.

Deciding *when* it has gone quiet for good is the whole difficulty: silence
alone cannot tell a program waiting at a prompt from one that is busy. What is
available differs by platform, so the best signal each one offers is used - see
`read_until_idle`. None of them is certain, so the model is told the program is
*probably* waiting, and can send an empty input to keep listening instead.

Stdlib only, so this stays importable from anywhere in the harness.
"""

import atexit
import itertools
import os
import queue
import signal
import subprocess
import threading
import time


# Linux answers "is this parked in read() on fd 0" exactly; nowhere else does.
PROC_READABLE = os.path.isdir("/proc") and os.path.exists("/proc/self/syscall")

_counter = itertools.count(1)
_sessions: "dict[str, Session]" = {}


def _cfg(name, default):
    import config
    return getattr(config, name, default)


def _fallback_encodings() -> list:
    """What the console might be speaking when it is not UTF-8, best first."""
    try:
        import config
        return [enc for enc in config._console_encodings()
                if enc.lower().replace("-", "") != "utf8"]
    except Exception:
        try:
            import locale
            preferred = locale.getpreferredencoding(False)
            return [preferred] if preferred else []
        except Exception:
            return []


def _default_child_encoding() -> str:
    """What to encode input as before the child's output has told us better."""
    if os.name == "nt":
        candidates = _fallback_encodings()
        if candidates:
            return candidates[0]
    return "utf-8"


def _decode_ready(buffer: bytes, encoding: str) -> tuple:
    """Decode the complete part of `buffer`. Returns (text, leftover, decoded).

    A multi-byte character split across two reads is kept back rather than
    mangled; anything else that will not decode means the encoding is wrong.
    """
    try:
        return buffer.decode(encoding), b"", True
    except UnicodeDecodeError as error:
        if error.reason == "unexpected end of data" and error.end == len(buffer):
            return buffer[:error.start].decode(encoding, "replace"), buffer[error.start:], True
        return "", buffer, False


def kill_process_tree(process) -> None:
    """Kill the command and everything it started, not just the shell in front."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                           capture_output=True, check=False)
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


class Session:
    """One running command, with its output drained by background threads."""

    def __init__(self, command: str, process):
        self.id = f"s{next(_counter)}"
        self.command = command
        self.process = process
        self.started = time.time()
        self.encoding = _default_child_encoding()
        self._encoding_settled = False
        self._queue: queue.Queue = queue.Queue()
        self._open_pipes = 0

        for stream in (process.stdout, process.stderr):
            if stream is None:
                continue
            self._open_pipes += 1
            threading.Thread(target=self._pump, args=(stream,), daemon=True).start()

    # -- reading ----------------------------------------------------------

    def _pump(self, stream) -> None:
        """Forward bytes the moment they arrive, not line by line.

        A prompt like `guess: ` has no newline, so anything that waits for one
        would never see the question the program is asking.

        UTF-8 is tried first because it is the only one of the candidates that
        can say it is wrong: almost any byte is legal cp949, so guessing that
        first would silently turn good text into mojibake. If the bytes turn out
        not to be UTF-8 - a Python older than 3.15 printing to a pipe on Korean
        Windows, say - the code page takes over from there, for what this
        program sends *and* for what we send back to it.
        """
        fd = stream.fileno()
        buffer = b""
        encoding = "utf-8"
        downgraded = False

        try:
            while True:
                data = os.read(fd, 65536)
                if not data:
                    break
                buffer += data

                text, buffer, decoded = _decode_ready(buffer, encoding)
                if not decoded and not downgraded:
                    encoding = self._choose_encoding()
                    downgraded = True
                    text, buffer, _ = _decode_ready(buffer, encoding)
                elif not decoded:
                    text, buffer = buffer.decode(encoding, "replace"), b""

                if text:
                    if not downgraded and not self._encoding_settled and not text.isascii():
                        # It really is UTF-8, so answer it in UTF-8 too.
                        self.encoding = "utf-8"
                        self._encoding_settled = True
                    self._queue.put(text)
        except Exception:
            pass
        finally:
            if buffer:
                self._queue.put(buffer.decode(encoding, "replace"))
            self._queue.put(None)

    def _choose_encoding(self) -> str:
        """The child is not speaking UTF-8; fall back to the console's code page."""
        candidates = _fallback_encodings()
        chosen = candidates[0] if candidates else "utf-8"
        self.encoding = chosen
        self._encoding_settled = True
        return chosen

    def cpu_seconds(self) -> float | None:
        """CPU burned by the whole tree so far, or None if it cannot be read.

        Where /proc is not available this is the only thing left to go on. It
        separates a program that is computing from one that is idle - but not an
        idle one that is waiting for input from an idle one that is sleeping,
        which is why it only shortens the wait instead of ending it.
        """
        try:
            import psutil
        except ImportError:
            return None
        try:
            parent = psutil.Process(self.process.pid)
            processes = [parent] + parent.children(recursive=True)
        except Exception:
            return None

        total = 0.0
        seen = False
        for proc in processes:
            try:
                times = proc.cpu_times()
                total += times.user + times.system
                seen = True
            except Exception:
                continue
        return total if seen else None

    def blocked_on_stdin(self) -> bool:
        """Is anything in this process tree parked in read() on fd 0?

        Only /proc can answer this outright, and when it can it is trusted both
        ways - a "no" is as informative as a "yes".
        """
        if not PROC_READABLE:
            return False
        for pid in self._descendants():
            try:
                with open(f"/proc/{pid}/syscall", encoding="ascii") as f:
                    fields = f.read().split()
            except Exception:
                continue
            # "<syscall number> <arg0> ..."; read() is 0 on x86-64, 63 on arm64,
            # and its first argument is the file descriptor.
            if len(fields) >= 2 and fields[0] in ("0", "63"):
                try:
                    if int(fields[1], 16) == 0:
                        return True
                except ValueError:
                    continue
        return False

    def _descendants(self) -> list:
        """The shell plus whatever it started - the program itself is a child."""
        pids, seen, i = [self.process.pid], set(), 0
        while i < len(pids) and len(pids) < 32:
            pid = pids[i]
            i += 1
            if pid in seen:
                continue
            seen.add(pid)
            try:
                for task in os.listdir(f"/proc/{pid}/task"):
                    with open(f"/proc/{pid}/task/{task}/children", encoding="ascii") as f:
                        pids.extend(int(child) for child in f.read().split())
            except Exception:
                pass
        return pids

    def read_until_idle(self, idle: float, deadline: float,
                        patience: float = None) -> tuple[str, bool, bool]:
        """Collect output until the command is waiting, ends, or time runs out.

        Going quiet is not enough on its own - a program that is merely busy
        looks the same. Three things separate them, best first:

        * a prompt is normally left unterminated (`추측: ` has no newline);
        * on Linux, /proc says outright whether a thread is parked in read() on
          fd 0, and is trusted in both directions;
        * everywhere else - Windows above all - CPU time is all there is. Idle
          means "not computing", which covers waiting for input but also plain
          sleeping, so it only shortens the wait rather than ending it.

        With none of them, the silence has to last `patience` seconds before it
        counts as a question.

        Returns (text, ended, timed_out).
        """
        if patience is None:
            patience = float(_cfg("CMD_WAIT_TIMEOUT", 8))
        idle_grace = float(_cfg("CMD_IDLE_GRACE", 2.5))
        chunks = []
        closed = 0
        quiet_since = None
        cpu_mark = None

        while True:
            now = time.time()
            if now >= deadline:
                return "".join(chunks), False, True
            try:
                item = self._queue.get(timeout=min(idle, deadline - now))
            except queue.Empty:
                if time.time() >= deadline:
                    # The wait ended on the deadline, not on the command falling
                    # quiet: it is still talking, it just will not stop.
                    return "".join(chunks), False, True

                text = "".join(chunks)
                if quiet_since is None:
                    quiet_since = time.time()
                    cpu_mark = self.cpu_seconds()

                if text and not text.endswith("\n"):
                    return text, False, False           # shaped like a prompt

                limit = patience
                if PROC_READABLE:
                    if self.blocked_on_stdin():
                        return text, False, False       # /proc: parked in read(0)
                    # /proc says it is not, so there is nothing to hurry for.
                else:
                    cpu_now = self.cpu_seconds()
                    if (cpu_mark is not None and cpu_now is not None
                            and cpu_now - cpu_mark < 0.02):
                        limit = min(patience, idle_grace)

                if time.time() - quiet_since >= limit:
                    return text, False, False
                continue

            if item is None:
                closed += 1
                if closed >= self._open_pipes:
                    return "".join(chunks), True, False
                continue
            quiet_since = None
            chunks.append(item)

    # -- writing ----------------------------------------------------------

    def send(self, text: str) -> bool:
        if not text:
            return True
        if not text.endswith("\n"):
            text += "\n"        # a program blocks on a line without its newline
        try:
            payload = text.encode(self.encoding, "replace")
        except LookupError:
            payload = text.encode("utf-8", "replace")
        try:
            self.process.stdin.write(payload)
            self.process.stdin.flush()
            return True
        except Exception:
            return False

    # -- lifecycle --------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    @property
    def age(self) -> float:
        return time.time() - self.started

    def close(self) -> int | None:
        kill_process_tree(self.process)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
        _sessions.pop(self.id, None)
        try:
            return self.process.wait(timeout=3)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def start(command: str):
    """Launch a command with its pipes wired up. Raises OSError on failure."""
    process = subprocess.Popen(
        command, shell=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=(os.name != "nt"),
    )
    return Session(command, process)


def register(session: Session) -> None:
    _sessions[session.id] = session
    prune()
    limit = int(_cfg("CMD_MAX_SESSIONS", 3))
    while len(_sessions) > limit:
        oldest = min(_sessions.values(), key=lambda s: s.started)
        oldest.close()


def prune() -> None:
    """Drop sessions that have exited or outstayed their welcome."""
    lifetime = float(_cfg("CMD_SESSION_LIFETIME", 900))
    for session in list(_sessions.values()):
        if not session.alive or session.age > lifetime:
            session.close()


def get(session_id: str) -> "Session | None":
    if not session_id:
        # One running session is unambiguous; the model often omits the id.
        return next(iter(_sessions.values())) if len(_sessions) == 1 else None
    return _sessions.get(str(session_id).strip())


def active() -> "list[Session]":
    prune()
    return list(_sessions.values())


def shutdown() -> None:
    for session in list(_sessions.values()):
        session.close()


atexit.register(shutdown)


if __name__ == "__main__":
    print("This file can not run directly.")
