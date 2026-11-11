"""The logic behind the MCP tools and resources, without the MCP SDK.

Every function takes the resolved workspace, refuses an app outside it through
``workspace.resolve_app`` and raises ``HexInputError`` for bad input (Requirements
4.2, 4.5). Nothing here prints or writes; the one call that can change an app
folder is the ``run`` argument of ``run_audit`` (Requirement 4.7).
"""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from typing import Any

from hexperiment.errors import HexInputError
from hexperiment.ghosts import find_ghost, to_dict
from hexperiment.runs import CARD_FILENAME, RUNS_DIRNAME, VERDICT_FILENAME, load_history
from hexperiment.workspace import (
    RunApp,
    discover_apps,
    relative_posix,
    resolve_app,
    resolve_inside,
)

MODE_FULL = "full"
MODE_FAST = "fast"
STATUS_NONE = "none"
EXIT_RUN_FAILED = 2
RESOURCE_PREFIX = "hexperiment://model-card/"


def list_apps(workspace: Path) -> dict[str, Any]:
    """Every app below the workspace with its current status (Requirement 4.3).

    ``status`` is the ``status`` in the app's ``verdict.json``, ``"none"`` when
    the file is absent or unreadable. ``problem`` says why a file or ``hex.toml``
    could not be read.
    """
    root = workspace.resolve()
    apps = []
    for app in discover_apps(root):
        status, verdict_problem = _current_status(app.path)
        problems = [p for p in (app.problem, verdict_problem) if p]
        apps.append(
            {
                "name": app.name,
                "path": app.relative,
                "status": status,
                "problem": " ".join(problems) if problems else None,
            }
        )
    return {"workspace": str(root), "apps": apps}


def get_verdict(workspace: Path, app: str) -> dict[str, Any]:
    """The app's current ``verdict.json``, or a message that it has none.

    The file is resolved first and refused when a link leads outside the
    workspace. A file that is not valid UTF-8 or JSON raises ``HexInputError``.
    """
    app_dir = resolve_app(workspace, app)
    path = resolve_inside(app_dir / VERDICT_FILENAME, workspace, "the verdict file")
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {
            "found": False,
            "message": (
                f"The app {app!r} has no current verdict: no run has finished "
                f"since {VERDICT_FILENAME} was last deleted. Call run_audit to "
                "create one."
            ),
        }
    except UnicodeDecodeError:
        raise _not_utf8(path) from None
    except OSError as error:
        raise _unreadable(path, repr(error)) from None
    try:
        verdict = json.loads(text)
    except ValueError as error:
        raise _unreadable(path, repr(error)) from None
    if not isinstance(verdict, dict):
        raise _unreadable(path, "the top level is not a JSON object")
    return {"found": True, "verdict": verdict}


def get_history(workspace: Path, app: str) -> dict[str, Any]:
    """One entry per earlier run of the app, oldest first (Requirement 4.3)."""
    app_dir = resolve_app(workspace, app)
    runs = [
        {
            "run_id": verdict.run_id,
            "status": verdict.status.value,
            "metric_value": verdict.metric.value,
            "ghosts": sorted({f.ghost for f in verdict.findings if f.ghost}),
        }
        for verdict in load_history(app_dir)
    ]
    return {"app": relative_posix(app_dir, workspace), "runs": runs}


def explain_ghost(name: str) -> dict[str, Any]:
    """What a ghost or check means, why it matters and how to fix it."""
    return dict(to_dict(find_ghost(name)))


def check_audit_request(workspace: Path, app: str, mode: str) -> tuple[Path, bool]:
    """Validate an audit request without running any code (Requirement 4.5).

    Returns the app folder and whether the mode is ``full``.
    """
    app_dir = resolve_app(workspace, app)
    if mode not in (MODE_FULL, MODE_FAST):
        raise HexInputError(
            f"the mode {mode!r} is not known.",
            f"Pass mode {MODE_FULL!r} (the default) or {MODE_FAST!r}.",
        )
    return app_dir, mode == MODE_FULL


def run_audit(workspace: Path, app: str, mode: str, run: RunApp) -> dict[str, Any]:
    """Run the app through ``run`` and return the outcome as a normal result.

    A run that fails is not an error of the tool: the agent needs the failure
    message to fix its code.
    """
    app_dir, full = check_audit_request(workspace, app, mode)
    outcome = run(app_dir, full=full, timeout_cap=None)
    failed = outcome.exit_code == EXIT_RUN_FAILED
    status = STATUS_NONE if outcome.status is None or failed else outcome.status.value
    return {
        "app": relative_posix(app_dir, workspace),
        "mode": MODE_FULL if full else MODE_FAST,
        "status": status,
        "exit_code": outcome.exit_code,
        "summary": outcome.stderr if failed else outcome.stdout,
    }


def model_card(workspace: Path, app: str) -> str:
    """The model card of the app's newest run."""
    app_dir = resolve_app(workspace, app)
    history = load_history(app_dir)
    if not history:
        raise HexInputError(
            f"the app {app!r} has no run yet, so it has no model card.",
            "Call run_audit on the app to create one.",
        )
    run_id = history[-1].run_id
    if run_id in ("", ".", "..") or Path(run_id).name != run_id:
        raise HexInputError(
            f"the newest run of {app!r} has the run id {run_id!r}, which is not a "
            "folder name.",
            "Never edit verdict.json by hand; restore the run folder from version "
            "control.",
        )
    card = resolve_inside(
        app_dir / RUNS_DIRNAME / run_id / CARD_FILENAME, workspace, "the model card"
    )
    try:
        return card.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise HexInputError(
            f"the model card {card} is not valid UTF-8 text.",
            "Never edit a model card by hand; run run_audit again to write a new one.",
        ) from None
    except OSError as error:
        raise HexInputError(
            f"the model card {card} cannot be read ({error!r}).",
            "Run run_audit again to write a new run, or restore the folder from "
            "version control.",
        ) from None


def resource_uri(app_relative: str) -> str:
    """The URI of an app's model card (path relative to the workspace, ``/``)."""
    return RESOURCE_PREFIX + urllib.parse.quote(app_relative, safe="")


def app_from_resource_id(resource_id: str) -> str:
    """The inverse of ``resource_uri``: the app path in a resource id."""
    return urllib.parse.unquote(resource_id)


# --- helpers -------------------------------------------------------------------------


def _current_status(app_dir: Path) -> tuple[str, str | None]:
    """The status in ``verdict.json`` and the reason it could not be read, if any."""
    path = app_dir / VERDICT_FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return STATUS_NONE, None
    except (OSError, ValueError) as error:
        return STATUS_NONE, f"{VERDICT_FILENAME} cannot be read ({error!r})."
    status = raw.get("status") if isinstance(raw, dict) else None
    if not isinstance(status, str):
        return STATUS_NONE, f"{VERDICT_FILENAME} has no status."
    return status, None


def _unreadable(path: Path, reason: str) -> HexInputError:
    return HexInputError(
        f"the file {path} cannot be read ({reason}).",
        "Never edit verdict.json by hand; run the app again to write a new one.",
    )


def _not_utf8(path: Path) -> HexInputError:
    return HexInputError(
        f"the file {path} is not valid UTF-8 text.",
        "Never edit verdict.json by hand; run the app again to write a new one.",
    )
