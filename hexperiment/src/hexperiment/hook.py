"""The text of ``hexperiment hook``: from a hook event to a message for the agent.

Kiro starts the command after the agent saves a file and puts what it prints into
the agent's context. This module finds the app that owns the saved file and runs
it in fast mode; the command in ``cli`` prints the result and always exits 0
(Requirements 1.2 to 1.7). It receives the runner as an argument and does not
import ``cli``.
"""

from __future__ import annotations

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

    The candidate paths of the event are tried in order. A path counts when it is
    an existing file inside the workspace below an app (Requirements 1.2, 1.4).
    The first such path starts exactly one fast run, capped so that it ends
    within the hook timeout (Requirement 1.6). The run is an ordinary run, so it
    is recorded in the history and counts as a test-set use (Requirement 1.7).
    """
    for raw in candidate_paths(event_text):
        path = Path(raw)
        absolute = path if path.is_absolute() else workspace / path
        if not absolute.is_file():
            continue
        app = find_app_for_file(absolute, workspace)
        if app is None:
            continue
        outcome = run(app, full=False, timeout_cap=HOOK_RUN_CAP_SECONDS)
        return format_hook_output(outcome, relative_posix(app, workspace))
    return ""


def format_hook_output(outcome: RunOutcome, app_relative: str) -> str:
    """The summary (or the failure message) followed by the full-check notice.

    The notice follows a failure message too: a failed fast run will also fail in
    full mode, and the work is still not done (Requirements 1.3, 1.5).
    """
    text = outcome.stderr if outcome.exit_code == 2 else outcome.stdout
    notice = FULL_CHECK_NOTICE.format(app=app_relative)
    text = text.rstrip("\n")
    return f"{text}\n{notice}" if text else notice
