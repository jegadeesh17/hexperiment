"""The text of ``hexperiment hook``: from a hook event to a message for the agent.

Kiro starts the command after the agent saves a file and puts what it prints into
the agent's context. This module finds the app that owns the saved file and runs
it in fast mode; the command in ``cli`` prints the result and always exits 0
(Requirements 1.2 to 1.7). It receives the runner as an argument and does not
import ``cli``.
"""

from __future__ import annotations

import os
from pathlib import Path

from hexperiment.hook_event import candidate_paths
from hexperiment.models import RunOutcome
from hexperiment.workspace import RunApp, find_app_for_file, relative_posix

HOOK_TIMEOUT_SECONDS = 120  # the `timeout` of the hook in .kiro/hooks/hexperiment.json
HOOK_RUN_CAP_SECONDS = 100  # per execution; leaves 20 s for start-up and writing

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
    the history and counts as a test-set use (Requirement 1.7).
    """
    for raw in candidate_paths(event_text):
        absolute = _path_inside_workspace(raw, workspace)
        if absolute is None or not absolute.is_file():
            continue
        app = find_app_for_file(absolute, workspace)
        if app is None:
            continue
        outcome = run(app, full=False, timeout_cap=HOOK_RUN_CAP_SECONDS)
        return format_hook_output(outcome, relative_posix(app, workspace))
    return ""


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
