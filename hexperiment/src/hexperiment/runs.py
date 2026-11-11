"""The run store: run ids, ``runs/<run_id>/`` folders and the history of an app.

This module owns everything that creates or reads ``runs/``. It never overwrites
or deletes an earlier run folder (Requirement 15.1).
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version as installed_version
from pathlib import Path

from hexperiment.canary import ExecutionReport
from hexperiment.card import render
from hexperiment.config import HexConfig
from hexperiment.errors import HexInputError
from hexperiment.models import SplitSignature, Verdict
from hexperiment.verdict import assemble, from_dict, to_json

PACKAGE_NAME = "hexperiment"
RUNS_DIRNAME = "runs"
VERDICT_FILENAME = "verdict.json"
CARD_FILENAME = "model_card.md"
PARTIAL_PREFIX = ".partial-"
DELETE_RETRIES = 5
DELETE_PAUSE_SECONDS = 0.2

_RUN_ID_PATTERN = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}")
_TOKEN_PATTERN = re.compile(r"[0-9a-f]{6}")
_CREATED_AT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_INPUT_CHECK = "input"


def format_run_id(now: datetime, token: str) -> str:
    """Return ``YYYYMMDD-HHMMSS-<6 hex chars>`` for ``now`` (UTC) (Requirement 13.2).

    A timezone-aware ``now`` is converted to UTC; a naive one is taken as UTC.
    """
    if not _TOKEN_PATTERN.fullmatch(token):
        raise HexInputError(
            f"the run id token {token!r} is not 6 lowercase hex characters.",
            "Pass a token such as secrets.token_hex(3), or leave it out.",
        )
    if now.tzinfo is not None:
        now = now.astimezone(UTC)
    stamp = (
        f"{now.year:04d}{now.month:02d}{now.day:02d}"
        f"-{now.hour:02d}{now.minute:02d}{now.second:02d}"
    )
    return f"{stamp}-{token}"


def new_run_id(now: datetime | None = None, token: str | None = None) -> str:
    """A fresh run id from the clock and a random token (Requirement 13.2)."""
    return format_run_id(
        datetime.now(UTC) if now is None else now,
        secrets.token_hex(3) if token is None else token,
    )


def count_test_set_uses(history: Sequence[Verdict], signature: SplitSignature) -> int:
    """Number of earlier runs that scored the same test set (Requirement 12.1).

    Fast, full, ``REJECTED`` and ``UNVERIFIED`` runs with an equal split block all
    count. A run that ended on an input error never scored the test set and does
    not.
    """
    return sum(1 for v in history if v.split == signature and not _is_input_error(v))


def load_history(app_dir: str | os.PathLike[str]) -> list[Verdict]:
    """The verdicts in ``<app_dir>/runs/*/verdict.json``, oldest first.

    Ordered by ``(created_at, run_id)`` (Requirement 15.2). A missing ``runs/``
    folder gives an empty list. A file that cannot be read or parsed raises
    ``HexInputError`` naming it.
    """
    runs_dir = Path(app_dir) / RUNS_DIRNAME
    if not runs_dir.is_dir():
        return []
    # Requirement 15.1: a folder whose name starts with "." is an unfinished run.
    paths = sorted(
        path
        for path in runs_dir.glob(f"*/{VERDICT_FILENAME}")
        if not path.parent.name.startswith(".")
    )
    verdicts = [_read_verdict(path) for path in paths]
    return sorted(verdicts, key=lambda v: (v.created_at, v.run_id))


def clear_app_verdict(app_dir: Path) -> None:
    """Delete ``<app_dir>/verdict.json`` if it exists; never touch ``runs/``.

    On Windows another process may hold the file open, so a failed delete is retried
    up to ``DELETE_RETRIES`` times, ``DELETE_PAUSE_SECONDS`` apart. If it still
    fails, an ``OSError`` says that the old file must not be trusted (Requirement
    16.5).
    """
    path = app_dir / VERDICT_FILENAME
    for attempt in range(DELETE_RETRIES + 1):
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            if attempt == DELETE_RETRIES:
                raise OSError(
                    f"the old {path} could not be removed ({error}) and must not "
                    "be trusted. Close the program that holds it open, or delete "
                    "it by hand, then run the command again."
                ) from error
            time.sleep(DELETE_PAUSE_SECONDS)
        else:
            return


def finalize_run(
    app_dir: Path, config: HexConfig, report: ExecutionReport, *, run_id: str
) -> Verdict:
    """Assemble the verdict of a run and write its evidence.

    The run counts as one more use of the test set unless it ended on an input
    error (Requirement 12.1). ``model_card.md`` and then ``verdict.json`` are written
    as UTF-8 into ``runs/.partial-<run_id>/``, which is renamed to ``runs/<run_id>/``
    as the last step, so a run folder is complete or absent. An existing
    ``runs/<run_id>/`` raises ``FileExistsError`` and is left untouched (Requirement
    15.1). ``verdict.json`` is then copied to ``app_dir`` atomically (Requirement
    13.3).
    """
    _require_run_id(run_id)
    history = load_history(app_dir)
    signature = _signature(config)
    uses = count_test_set_uses(history, signature)
    if report.result.input_error is None:
        uses += 1
    verdict = assemble(
        report.result,
        report.canary,
        config,
        mode=report.mode,
        run_id=run_id,
        created_at=datetime.now(UTC).strftime(_CREATED_AT_FORMAT),
        version=installed_version(PACKAGE_NAME),
        test_set_uses=uses,
    )
    verdict_text = to_json(verdict)
    card_text = render(verdict, history, description=config.app.description)
    run_dir = _publish_run_folder(app_dir, run_id, card_text, verdict_text)
    _replace_app_verdict(app_dir, run_dir / VERDICT_FILENAME, run_id)
    return verdict


# --- helpers -------------------------------------------------------------------------


def _signature(config: HexConfig) -> SplitSignature:
    return SplitSignature(
        data_path=config.data.path,
        test_fraction=config.split.test_fraction,
        seed=config.split.seed,
        time_column=config.split.time_column,
    )


def _is_input_error(verdict: Verdict) -> bool:
    return any(f.check == _INPUT_CHECK for f in verdict.findings)


def _require_run_id(run_id: str) -> None:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise HexInputError(
            f"the run id {run_id!r} is not of the form YYYYMMDD-HHMMSS-<6 hex chars>.",
            "Create it with hexperiment.runs.new_run_id and do not set HEX_RUN_ID "
            "by hand.",
        )


def _read_verdict(path: Path) -> Verdict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise _bad_history_file(path, repr(error)) from None
    if not isinstance(raw, dict):
        raise _bad_history_file(path, "the top level is not a JSON object")
    try:
        return from_dict(raw)
    except HexInputError as error:
        raise _bad_history_file(path, error.problem) from None


def _bad_history_file(path: Path, reason: str) -> HexInputError:
    return HexInputError(
        f"the run history file {path} cannot be read ({reason}).",
        "Restore it from version control, or move its run folder out of runs/ "
        "if it is damaged. Never edit verdict.json by hand.",
    )


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _publish_run_folder(
    app_dir: Path, run_id: str, card_text: str, verdict_text: str
) -> Path:
    """Write the run folder under a hidden name, then rename it (Requirement 15.1).

    A leftover ``.partial-<run_id>`` from an interrupted run is removed first. The
    rename is the last step; an error before it leaves only the hidden folder.
    """
    runs_dir = app_dir / RUNS_DIRNAME
    run_dir = runs_dir / run_id
    partial = runs_dir / f"{PARTIAL_PREFIX}{run_id}"
    _require_absent(run_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    _write_text(partial / CARD_FILENAME, card_text)
    _write_text(partial / VERDICT_FILENAME, verdict_text)
    _require_absent(run_dir)
    partial.rename(run_dir)
    return run_dir


def _require_absent(run_dir: Path) -> None:
    if run_dir.exists():
        raise FileExistsError(
            f"the run folder {run_dir} already exists and is never overwritten. "
            "Run the command again to get a new run id."
        )


def _replace_app_verdict(app_dir: Path, source: Path, run_id: str) -> None:
    """Copy ``source`` to ``<app_dir>/verdict.json`` through a temporary file."""
    temporary = app_dir / f".{VERDICT_FILENAME}.{run_id}.tmp"
    try:
        temporary.write_bytes(source.read_bytes())
        os.replace(temporary, app_dir / VERDICT_FILENAME)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
