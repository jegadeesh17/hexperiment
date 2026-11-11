"""The in-script half of a run: ``audit``, called at the end of ``train.py``.

``audit`` validates what the script produced, writes the snapshot that the canary
comparison needs, and in normal mode runs the data checks. It never modifies its
arguments (Requirement 2.5).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TextIO

import pandas as pd

from hexperiment.canary import (
    ExecutionReport,
    build_snapshot,
    skipped_outcome,
    write_result,
    write_snapshot,
)
from hexperiment.checks import run_data_checks
from hexperiment.config import (
    ExecMode,
    HexConfig,
    current_mode,
    load_config,
    run_id_from_env,
)
from hexperiment.data import (
    ArrayLike1D,
    validate_audit_inputs,
    validate_snapshot_inputs,
)
from hexperiment.errors import HexInputError
from hexperiment.models import DataAuditResult, Finding, RunMode, Severity, Verdict
from hexperiment.runs import finalize_run, new_run_id
from hexperiment.verdict import format_summary

_INPUT_CHECK = "input"
_INPUT_LOCATION = "audit()"


def audit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_score: ArrayLike1D,
    features: Sequence[str],
    script: str | os.PathLike[str],
    y_score_train: ArrayLike1D | None = None,
) -> Verdict | None:
    """Audit one training run (Requirement 2.1).

    ``script`` is the path of the calling training script; its folder is the app
    directory, and ``hex.toml`` is read from there (Requirement 2.2). The mode
    comes from ``HEX_MODE``:

    - ``canary`` and ``control``: validate the training side only, write the
      snapshot and return ``None``. No check runs and no verdict is written
      (Requirement 2.4).
    - ``normal``: validate everything, write the snapshot, run the data checks
      and write ``result.json`` for the parent process (Requirement 2.3). An
      input error is recorded the same way, and then raised again (Requirement
      4.7).

    Without ``HEX_RUN_ID`` (a standalone run) a normal audit also finalizes a
    ``fast`` run itself and returns its ``Verdict``; otherwise it returns ``None``
    and the parent process writes the verdict.
    """
    script_path = Path(script).resolve()
    app_dir = script_path.parent
    config = load_config(app_dir)
    mode = current_mode()
    run_id = run_id_from_env()
    if mode is ExecMode.NORMAL:
        return _audit_normal(
            train, test, y_score, y_score_train, features, script_path, config, run_id
        )
    snapshot_inputs = validate_snapshot_inputs(train, y_score_train, features)
    snapshot = build_snapshot(
        mode,
        run_id,
        train,
        test,
        snapshot_inputs.features,
        snapshot_inputs.y_score_train,
    )
    write_snapshot(app_dir, snapshot)
    return None


def _audit_normal(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_score: ArrayLike1D,
    y_score_train: ArrayLike1D | None,
    features: Sequence[str],
    script: Path,
    config: HexConfig,
    run_id: str | None,
) -> Verdict | None:
    app_dir = script.parent
    try:
        validated = validate_audit_inputs(
            train, test, y_score, y_score_train, features, config
        )
        snapshot = build_snapshot(
            ExecMode.NORMAL,
            run_id,
            train,
            test,
            validated.features,
            validated.y_score_train,
        )
        write_snapshot(app_dir, snapshot)
        result = run_data_checks(
            train,
            test,
            validated,
            config=config,
            script_source=_read_source(script),
            script_name=script.name,
        )
    except HexInputError as error:
        # Requirement 4.7: an input error from a check is recorded like one
        # from the validation.
        _record_input_error(app_dir, config, run_id, error)
        raise
    result = replace(result, run_id=run_id)
    write_result(app_dir, result, run_id)
    if run_id is not None:
        return None
    verdict = _finalize_standalone(app_dir, config, result)
    print_text(format_summary(verdict))
    return verdict


def print_text(text: str, file: TextIO | None = None) -> None:
    """Print ``text``; a console that cannot encode it never raises.

    ``file`` defaults to the current ``sys.stdout``. This and ``cli.main`` are the
    only printers of the package (Requirement 16.4).
    """
    stream = sys.stdout if file is None else file
    try:
        print(text, file=stream)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        line_text = (text + "\n").replace("\n", os.linesep)
        data = line_text.encode(encoding, errors="replace")
        buffer = getattr(stream, "buffer", None)
        if buffer is None:
            stream.write(data.decode(encoding, errors="replace"))
        else:
            stream.flush()
            buffer.write(data)
        stream.flush()


def _record_input_error(
    app_dir: Path, config: HexConfig, run_id: str | None, error: HexInputError
) -> None:
    """Write the input error as ``result.json``, and a verdict when standalone.

    Requirement 4.7: loops and hooks must see a failed run, not a crash.
    """
    result = DataAuditResult(
        run_id=run_id,
        metric=None,
        checks=(),
        extra_findings=(),
        data=None,
        input_error=_input_error_finding(error),
    )
    write_result(app_dir, result, run_id)
    if run_id is None:
        _finalize_standalone(app_dir, config, result)


def _finalize_standalone(
    app_dir: Path, config: HexConfig, result: DataAuditResult
) -> Verdict:
    """Finalize a ``fast`` run for a script that was started without the command."""
    report = ExecutionReport(result, skipped_outcome(), RunMode.FAST)
    return finalize_run(app_dir, config, report, run_id=new_run_id())


def _input_error_finding(error: HexInputError) -> Finding:
    return Finding(
        check=_INPUT_CHECK,
        ghost=None,
        severity=Severity.ERROR,
        location=_INPUT_LOCATION,
        message=error.problem,
        evidence={},
        fix=error.fix,
    )


def _read_source(script: Path) -> str | None:
    """The text of the training script, or ``None`` when it cannot be read."""
    try:
        # utf-8-sig: a byte order mark must not break the parse of the hint.
        return script.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return None
