"""The write guard of the ghost hunter: which files an agent may write.

Kiro starts ``hexperiment guard-write`` before every file write of the agent and
sends the tool call as JSON on standard input (``cwd``, ``tool_name`` and
``tool_input.path``). Exit code 0 lets the write happen; exit code 2 refuses it and
the line on standard error reaches the agent (Requirement 3.8).

Only a Python file inside ``<cwd>/apps/<app>/`` is allowed. The path is resolved
first (``..``, links, absolute paths, letter case), so no spelling gets around the
rule. This module is started at every write, so it imports no pandas, NumPy,
scikit-learn or MCP code, and it does not import ``cli``.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from typing import Any

from hexperiment.console import print_text
from hexperiment.hook import read_event_text

REFUSED = 2
ALLOWED = 0
APPS_FOLDER = "apps"
REFUSED_APP_NAMES = frozenset({"tests"})
REFUSED_FOLDERS = frozenset({"haunted", "data", "runs", ".hex", "hexperiment"})
REFUSED_FILES = frozenset({"fetch_data.py", "hexperiment.py"})
SUFFIX = ".py"


def check_event(event_text: str) -> str | None:
    """``None`` when the write is allowed, else a one-line reason to refuse it."""
    try:
        event = json.loads(event_text)
    except (ValueError, RecursionError):
        return "the hook input is not valid JSON, so the write cannot be checked."
    path, cwd = _event_fields(event)
    if path is None or cwd is None:
        return (
            "the hook input has no tool_input.path or cwd, so the write cannot be "
            "checked."
        )
    try:
        reason = _path_problem(path, cwd)
    except (OSError, ValueError):
        return f"{_show(path)} cannot be resolved, so the write cannot be checked."
    if reason is None:
        return None
    return f"{_show(path)} is refused: {reason}"


def main() -> int:
    """Read the event from standard input; return 0 to allow or 2 to refuse."""
    try:
        reason = check_event(read_event_text())
    except Exception:
        reason = "the hook input could not be read, so the write cannot be checked."
    if reason is None:
        return ALLOWED
    with contextlib.suppress(Exception):
        print_text(f"hexperiment guard-write: {reason}", sys.stderr)
    return REFUSED


def _event_fields(event: Any) -> tuple[str | None, str | None]:
    if not isinstance(event, dict):
        return None, None
    tool_input = event.get("tool_input")
    path = tool_input.get("path") if isinstance(tool_input, dict) else None
    cwd = event.get("cwd")
    path = path if isinstance(path, str) and _usable(path) else None
    cwd = cwd if isinstance(cwd, str) and _usable(cwd) else None
    return path, cwd


def _usable(text: str) -> bool:
    return bool(text.strip()) and "\x00" not in text


def _path_problem(path: str, cwd: str) -> str | None:
    """Why ``path`` may not be written, or ``None`` when it may."""
    target = _resolve(os.path.join(cwd, path))
    apps = _resolve(os.path.join(cwd, APPS_FOLDER))
    prefix = apps if apps.endswith(os.sep) else apps + os.sep
    if not target.startswith(prefix):
        return "only Python files inside apps/<app>/ may be written."
    parts = target[len(prefix) :].split(os.sep)
    if len(parts) < 2:
        return "only Python files inside an app folder apps/<app>/ may be written."
    app, folders, name = parts[0], parts[1:-1], parts[-1]
    if app.casefold() in REFUSED_APP_NAMES:
        return f"apps/{app} is not an app folder."
    if not name.casefold().endswith(SUFFIX):
        return "only Python files may be written."
    blocked = next((f for f in folders if f.casefold() in REFUSED_FOLDERS), None)
    if blocked is not None:
        return f"the {blocked} folder of an app is not for the agent to write."
    if name.casefold() in REFUSED_FILES:
        return f"{name} is not for the agent to write."
    return None


def _resolve(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _show(path: str) -> str:
    """The path as one printable line."""
    return repr(path)
