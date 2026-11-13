"""The text of ``hexperiment hook``: from a hook event to a message for the agent.

Kiro starts the command after the agent saves a file. Its documentation says that
what the command prints is added to the agent's context; Kiro CLI 3.2.0 and Kiro
IDE 1.4.154 do not do this, so the lasting effect is the fast run itself, which
clears the app's verdict and is never APPROVED. This module finds the app that
owns the saved file and runs it in fast mode; the command in ``cli`` prints the
result and always exits 0 (Requirements 1.2 to 1.7). It receives the runner as an
argument and does not import ``cli``.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from hexperiment.console import print_text
from hexperiment.errors import HexInputError
from hexperiment.hook_event import candidate_paths, session_id
from hexperiment.models import RunOutcome

if TYPE_CHECKING:
    from hexperiment.workspace import RunApp

HOOK_TIMEOUT_SECONDS = 120  # the `timeout` of the hook in .kiro/hooks/hexperiment.json
HOOK_RUN_CAP_SECONDS = 100  # per execution; leaves 20 s for start-up and writing

SESSION_FILE = "agent-session"  # inside the app's .hex/ folder
STATE_FOLDER = ".hex"
MAX_SESSION_FILE_BYTES = 256
# The stop check must not load pandas, NumPy or scikit-learn (Requirement 9.6), and
# `workspace` imports `canary`, which does. So this module repeats the three names
# it needs from `config`/`canary`/`workspace`; test_hook.py pins them to the originals.
CONFIG_NAME = "hex.toml"
SCRIPT_NAME = "train.py"
VENV_MARKER = "pyvenv.cfg"
SKIPPED_FOLDERS = frozenset({"node_modules", "__pycache__"})
ENV_WORKSPACE = "HEXPERIMENT_WORKSPACE"  # as in `workspace`; pinned by test_hook.py
APPROVED = "APPROVED"
MAX_EVENT_BYTES = 1024 * 1024  # the most of a hook event that the command reads

FULL_CHECK_NOTICE = (
    "Fast check only: the canary test did not run, so this result can never be "
    "APPROVED. Before you call the work done, run: hexperiment run {app}"
)


def handle_hook(event_text: str, *, workspace: Path, run: RunApp) -> str:
    """Run the app that owns the saved file in fast mode; ``""`` if there is none.

    The candidate paths of the event are tried in order. A path counts when it
    lies inside the workspace (checked on the text of the path, before any file
    system access, so a network path in an event is never contacted), is an
    existing file and sits below an app (Requirements 1.2, 1.4). The first such
    path starts exactly one fast run, capped so that it ends within the hook
    timeout (Requirement 1.6). The run is an ordinary run, so it is recorded in
    the history and counts as a test-set use (Requirement 1.7). Before the run the
    session of the event is written to the app (Requirement 9.2).
    """
    from hexperiment.workspace import find_app_for_file, relative_posix

    for raw in candidate_paths(event_text):
        absolute = _path_inside_workspace(raw, workspace)
        if absolute is None or not absolute.is_file():
            continue
        app = find_app_for_file(absolute, workspace)
        if app is None:
            continue
        session = session_id(event_text)
        if session is not None:
            record_session(app, session)
        outcome = run(app, full=False, timeout_cap=HOOK_RUN_CAP_SECONDS)
        return format_hook_output(outcome, relative_posix(app, workspace))
    return ""


def record_session(app: Path, session: str) -> None:
    """Write ``session`` to ``<app>/.hex/agent-session``; a failure is ignored."""
    with contextlib.suppress(OSError):
        folder = app / STATE_FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        (folder / SESSION_FILE).write_bytes(f"{session}\n".encode())


def handle_stop(event_text: str, *, workspace: Path) -> str:
    """The message for the agent about its unapproved apps, or ``""`` (Requirement 9).

    An app is named when its ``.hex/agent-session`` holds the session of the event
    and its ``verdict.json`` is missing or does not say ``APPROVED``. Nothing is
    run or written.
    """
    session = session_id(event_text)
    if session is None:
        return ""
    pending = []
    for app, relative in _find_apps(workspace):
        if _read_session(app) != session:
            continue
        status = _read_status(app)
        if status != APPROVED:
            pending.append((relative, status))
    return format_stop_message(sorted(pending)) if pending else ""


def read_event_text() -> str:
    """At most ``MAX_EVENT_BYTES`` of standard input; a terminal counts as empty."""
    stream = sys.stdin
    if stream is None or stream.isatty():
        return ""
    data = stream.buffer.read(MAX_EVENT_BYTES)
    return data.decode("utf-8-sig", errors="replace")


def stop_command() -> int:
    """``hexperiment hook --stop``: tell the agent which of its apps are not approved.

    A message goes to standard error with exit code 1, which Kiro gives to the
    agent; no message, or any failure, gives no output and exit code 0
    (Requirements 9.4, 9.5, 9.6).
    """
    try:
        message = handle_stop(read_event_text(), workspace=_resolve_workspace())
    except Exception:
        return 0
    if not message:
        return 0
    with contextlib.suppress(Exception):
        print_text(message, sys.stderr)
    return 1


def _resolve_workspace() -> Path:
    """``workspace.resolve_workspace()`` for the process, without importing it.

    ``workspace`` imports ``canary`` and so pandas; the stop check must not
    (Requirement 9.6). test_hook.py checks that both give the same answer.
    """
    value = os.environ.get(ENV_WORKSPACE, "").strip()
    base = Path.cwd()
    if not value:
        return base.resolve()
    try:
        resolved = (base / value).resolve()
        is_folder = resolved.is_dir()
    except (OSError, ValueError, RuntimeError):
        is_folder = False
    if not is_folder:
        raise HexInputError(
            f"{ENV_WORKSPACE} is set to {value!r}, which is not an existing folder.",
            f"Set {ENV_WORKSPACE} to the project folder, or unset it to use the "
            "current folder.",
        )
    return resolved


def format_stop_message(pending: Sequence[tuple[str, str | None]]) -> str:
    """The text for the agent: one line per app, then what not to do (Requirement 9.4).

    ``pending`` holds ``(app path, status)``; the status is ``None`` when the app
    has no verdict. ``REJECTED`` asks for a fix or a plain report; every other
    status, including an unknown one, asks for the full audit.
    """
    lines = ["Hexperiment: not approved yet."]
    for app, status in pending:
        lines.append(f"- {app}: {_stop_line(app, status)}")
    lines.append(
        "Do not call the work done until each app above is APPROVED or "
        "reported as not approved."
    )
    return "\n".join(lines)


def _stop_line(app: str, status: str | None) -> str:
    audit = (
        "Run the full audit: the hexperiment MCP tool run_audit with app "
        f"{app} and mode full (or: hexperiment run {app})."
    )
    if status is None:
        return f"changed in this session and has no verdict. {audit}"
    if status == "REJECTED":
        return (
            "changed in this session; its last full audit is REJECTED. Fix the "
            f"findings at their cause, or tell the user plainly that {app} is "
            "REJECTED and why."
        )
    return (
        f"changed in this session; its verdict is UNVERIFIED (fast check only). {audit}"
    )


def _find_apps(workspace: Path) -> list[tuple[Path, str]]:
    """The apps below ``workspace`` as ``(folder, relative path)``, without pandas.

    The same walk as ``workspace.discover_apps`` (links are not followed, hidden
    folders, ``node_modules``, ``__pycache__`` and virtual environments are
    skipped, an app is not entered), but without reading ``hex.toml``.
    """
    root = workspace.resolve()
    found: list[tuple[Path, str]] = []
    for folder, names, files in os.walk(root, followlinks=False):
        here = Path(folder)
        if CONFIG_NAME in files and SCRIPT_NAME in files:
            found.append((here, _relative(here, root)))
            names[:] = []
            continue
        names[:] = [n for n in names if not _skipped(here / n)]
    return found


def _skipped(folder: Path) -> bool:
    name = folder.name
    return (
        name.startswith(".")
        or name in SKIPPED_FOLDERS
        or (folder / VENV_MARKER).exists()
    )


def _relative(folder: Path, root: Path) -> str:
    parts = folder.relative_to(root).parts
    return "/".join(parts) if parts else "."


def _read_session(app: Path) -> str | None:
    try:
        with open(app / STATE_FOLDER / SESSION_FILE, "rb") as handle:
            data = handle.read(MAX_SESSION_FILE_BYTES)
        return data.decode("utf-8").strip()
    except (OSError, ValueError):
        return None


def _read_status(app: Path) -> str | None:
    try:
        verdict = json.loads((app / "verdict.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    status = verdict.get("status") if isinstance(verdict, dict) else None
    return status if isinstance(status, str) else None


def _path_inside_workspace(raw: str, workspace: Path) -> Path | None:
    """The normalised path of ``raw``, or ``None`` if it is not inside ``workspace``.

    This compares text only: ``os.path.normpath`` removes ``.`` and ``..`` and
    ``os.path.normcase`` ignores case and separators on Windows. A relative
    ``raw`` is taken relative to the workspace. A UNC path, a ``file://host/...``
    path and a path on another drive fail the comparison without a stat call
    (Requirement 1.2). Links are checked later, on the resolved path.
    """
    root = os.path.normpath(str(workspace))
    candidate = os.path.normpath(os.path.join(root, raw))
    root_key = os.path.normcase(root)
    candidate_key = os.path.normcase(candidate)
    prefix = root_key if root_key.endswith(os.sep) else root_key + os.sep
    if candidate_key == root_key or not candidate_key.startswith(prefix):
        return None
    return Path(candidate)


def format_hook_output(outcome: RunOutcome, app_relative: str) -> str:
    """The summary (or the failure message) followed by the full-check notice.

    The notice follows a failure message too: a failed fast run will also fail in
    full mode, and the work is still not done. An app path with whitespace is
    quoted so that the command in the notice can be pasted (Requirements 1.3, 1.5).
    """
    text = outcome.stderr if outcome.exit_code == 2 else outcome.stdout
    notice = FULL_CHECK_NOTICE.format(app=_shell_argument(app_relative))
    text = text.rstrip("\n")
    return f"{text}\n{notice}" if text else notice


def _shell_argument(path: str) -> str:
    return f'"{path}"' if any(char.isspace() for char in path) else path
