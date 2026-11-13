"""Running one app through the installed ``hexperiment`` command (edge).

This is the only benchmark module that imports ``subprocess``. It starts
``hexperiment run <app>``, waits for it, and reads ``verdict.json`` and the exit
code. It imports nothing from the kit and passes no option that changes a check.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

# Three executions at the 60 s of ``[run] timeout_seconds``, plus a minute for
# start-up and for writing the run folder.
HARD_LIMIT_SECONDS = 240
COMMAND_NAME = "hexperiment"

_STDERR_KEEP_CHARS = 4000
_MESSAGE_MAX_CHARS = 600
_EXPECTED_STATUS = {0: "APPROVED", 1: "REJECTED", 3: "UNVERIFIED"}
_ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class Launched:
    exit_code: int | None  # None: stopped by the runner after the time limit
    stderr: str  # last 4,000 characters
    seconds: float


type Launch = Callable[[Sequence[str], Path, float], Launched]  # argv, cwd, limit


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One run of one app (also used by ``summary``)."""

    status: str  # "APPROVED" | "REJECTED" | "UNVERIFIED" | "ERROR"
    score: float | None
    ghosts: tuple[str, ...]  # ghost name, else check name, of each error finding
    warnings: tuple[str, ...]  # check names of warning findings
    seconds: float
    message: str | None  # the reason, for ERROR only


def find_command(which: Callable[..., str | None] = shutil.which) -> str | None:
    """The ``hexperiment`` command: next to this interpreter first, then on PATH."""
    beside = which(COMMAND_NAME, path=str(Path(sys.executable).parent))
    if beside:
        return beside
    return which(COMMAND_NAME)


# --- starting and stopping the process ------------------------------------------------


def launch(argv: Sequence[str], cwd: Path, limit: float) -> Launched:
    """Run ``argv`` without a shell and stop its whole process tree at ``limit``.

    Output goes to temporary files, not pipes, so a grandchild that outlives the
    command cannot block the read. The process starts in a new process group
    (Windows) or session (elsewhere) so the tree can be killed as one.
    """
    started = time.monotonic()
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            list(argv),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=_new_group_flags(),
            start_new_session=os.name != "nt",
        )
        timed_out = False
        try:
            process.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Also reached by KeyboardInterrupt: no training process is left behind.
            if process.poll() is None:
                _kill_process_tree(process)
                process.wait()
        tail = _read_tail(stderr)
    return Launched(
        exit_code=None if timed_out else process.returncode,
        stderr=tail,
        seconds=time.monotonic() - started,
    )


def _new_group_flags() -> int:
    if os.name == "nt":
        return subprocess.CREATE_NEW_PROCESS_GROUP
    return 0


def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Kill ``process`` and every process it started.

    A small copy of the kit's private helper; the two do the same thing.
    """
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        pass
    # taskkill can miss the process itself; kill it directly so wait() returns.
    process.kill()


def _read_tail(stream: IO[bytes]) -> str:
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(max(0, size - _STDERR_KEEP_CHARS * 4))
    text = stream.read().decode("utf-8", errors="replace")
    return text[-_STDERR_KEEP_CHARS:]


# --- one run ----------------------------------------------------------------------


def run_app(command: str, app_dir: Path, *, launcher: Launch = launch) -> RunRecord:
    """Run ``hexperiment run <app_dir>`` (full mode) and read its outcome."""
    argv = [command, "run", str(app_dir)]
    launched = launcher(argv, Path(app_dir).parent.parent, HARD_LIMIT_SECONDS)
    return read_outcome(Path(app_dir), launched)


def read_outcome(app_dir: Path, launched: Launched) -> RunRecord:
    """Map the exit code and ``verdict.json`` of a run to a ``RunRecord``."""
    code = launched.exit_code
    if code is None:
        return _error(
            launched,
            f"no result after {HARD_LIMIT_SECONDS} seconds; the run was stopped",
        )
    if code == 2:
        return _error(launched, _first_lines(launched.stderr) or _NO_MESSAGE)
    if code not in _EXPECTED_STATUS:
        return _error(launched, f"unexpected exit code {code}")
    try:
        return _from_verdict(app_dir, code, launched.seconds)
    except _Disagreement as problem:
        return _error(launched, str(problem))


_NO_MESSAGE = "hexperiment run exited with code 2 and wrote no message"


class _Disagreement(Exception):
    """The exit code and ``verdict.json`` do not describe the same run."""


def _error(launched: Launched, message: str) -> RunRecord:
    return RunRecord(_ERROR, None, (), (), launched.seconds, message)


def _first_lines(stderr: str) -> str:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return "\n".join(lines)[:_MESSAGE_MAX_CHARS]


def _from_verdict(app_dir: Path, code: int, seconds: float) -> RunRecord:
    verdict = _load_verdict(app_dir / "verdict.json", code)
    expected = _EXPECTED_STATUS[code]
    try:
        status = verdict["status"]
        value = verdict["metric"]["value"]
        findings = verdict["findings"]
        ghosts = {_key(f) for f in findings if f["severity"] == "error"}
        warnings = {f["check"] for f in findings if f["severity"] == "warning"}
        score = None if value is None else float(value)
    except (KeyError, TypeError, ValueError) as problem:
        raise _Disagreement(
            f"verdict.json is missing a field or has a wrong one ({problem!r})"
        ) from None
    if status != expected:
        raise _Disagreement(f"exit code {code} but verdict.json says {status}")
    return RunRecord(
        status, score, tuple(sorted(ghosts)), tuple(sorted(warnings)), seconds, None
    )


def _key(finding: dict[str, Any]) -> str:
    ghost = finding["ghost"]
    return str(ghost) if ghost else str(finding["check"])


def _load_verdict(path: Path, code: int) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise _Disagreement(f"exit code {code} but verdict.json is missing") from None
    except (OSError, ValueError) as problem:
        raise _Disagreement(f"verdict.json cannot be read ({problem})") from None
    if not isinstance(data, dict):
        raise _Disagreement("verdict.json is not a JSON object")
    return data
