"""Running one app and reading its outcome (no process is started)."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from hexbench.execution import (
    HARD_LIMIT_SECONDS,
    Launched,
    RunRecord,
    find_command,
    read_outcome,
    run_app,
)
from hexperiment.models import (
    CheckSummary,
    Finding,
    MetricResult,
    Outcome,
    RunMode,
    Severity,
    SplitSignature,
    Status,
    Verdict,
)


def _finding(check: str, ghost: str | None, severity: Severity) -> Finding:
    return Finding(
        check=check,
        ghost=ghost,
        severity=severity,
        location="train.py",
        message="m",
        evidence={},
        fix="f",
    )


def _verdict(
    status: Status,
    findings: tuple[Finding, ...] = (),
    value: float | None = 0.7123,
) -> dict[str, object]:
    verdict = Verdict(
        status=status,
        mode=RunMode.FULL,
        app="case",
        run_id="r1",
        created_at="2026-01-01T00:00:00Z",
        hexperiment_version="0.1.0",
        metric=MetricResult("roc_auc", value, 0.6, value is not None and value >= 0.6),
        checks=(CheckSummary("overlap", "The Doppelganger", Outcome.PASSED),),
        findings=findings,
        waivers=(),
        test_set_uses=1,
        split=SplitSignature("data/dataset.csv", 0.2, 0, None),
        data=None,
    )
    return verdict.to_dict()


def _write(app: Path, data: object) -> None:
    app.mkdir(parents=True, exist_ok=True)
    (app / "verdict.json").write_text(json.dumps(data), encoding="utf-8")


def _launched(code: int | None, stderr: str = "", seconds: float = 1.5) -> Launched:
    return Launched(exit_code=code, stderr=stderr, seconds=seconds)


def test_exit_0_is_an_approved_record(tmp_path: Path) -> None:
    _write(tmp_path, _verdict(Status.APPROVED))
    record = read_outcome(tmp_path, _launched(0))
    assert record == RunRecord("APPROVED", 0.7123, (), (), 1.5, None)


def test_exit_1_reads_ghosts_and_sorts_warnings(tmp_path: Path) -> None:
    findings = (
        _finding("canary", "The Peeking Spirit", Severity.ERROR),
        _finding("overlap", "The Doppelganger", Severity.ERROR),
        _finding("overlap", "The Doppelganger", Severity.ERROR),
        _finding("split_order", None, Severity.WARNING),
        _finding("determinism", "The Shapeshifter", Severity.WARNING),
        _finding("too_good", None, Severity.WARNING),
    )
    _write(tmp_path, _verdict(Status.REJECTED, findings))
    record = read_outcome(tmp_path, _launched(1))
    assert record.status == "REJECTED"
    assert record.ghosts == ("The Doppelganger", "The Peeking Spirit")
    assert record.warnings == ("determinism", "split_order", "too_good")


def test_error_finding_without_a_ghost_uses_the_check_name(tmp_path: Path) -> None:
    findings = (_finding("metric", None, Severity.ERROR),)
    _write(tmp_path, _verdict(Status.REJECTED, findings))
    assert read_outcome(tmp_path, _launched(1)).ghosts == ("metric",)


def test_exit_3_is_unverified(tmp_path: Path) -> None:
    _write(tmp_path, _verdict(Status.UNVERIFIED))
    record = read_outcome(tmp_path, _launched(3))
    assert record.status == "UNVERIFIED"
    assert record.message is None


def test_input_error_verdict_is_a_rejection_under_input(tmp_path: Path) -> None:
    findings = (_finding("input", None, Severity.ERROR),)
    _write(tmp_path, _verdict(Status.REJECTED, findings, value=None))
    record = read_outcome(tmp_path, _launched(1))
    assert record.status == "REJECTED"
    assert record.score is None
    assert record.ghosts == ("input",)


def test_exit_2_is_an_error_with_the_first_lines_of_stderr(tmp_path: Path) -> None:
    stderr = "\n  \nhex.toml is wrong\nfix it\n" + "x" * 1000
    record = read_outcome(tmp_path, _launched(2, stderr))
    assert record.status == "ERROR"
    assert record.score is None
    assert record.message is not None
    assert record.message.startswith("hex.toml is wrong\nfix it\n")
    assert len(record.message) == 600


def test_exit_2_without_a_message_still_says_something(tmp_path: Path) -> None:
    record = read_outcome(tmp_path, _launched(2, ""))
    assert record.status == "ERROR"
    assert record.message


def test_timeout_is_an_error(tmp_path: Path) -> None:
    record = read_outcome(tmp_path, _launched(None, seconds=240.2))
    assert record.status == "ERROR"
    assert record.message == "no result after 240 seconds; the run was stopped"
    assert record.seconds == 240.2


def test_unexpected_exit_code_is_an_error(tmp_path: Path) -> None:
    record = read_outcome(tmp_path, _launched(7))
    assert (record.status, record.message) == ("ERROR", "unexpected exit code 7")


def test_missing_file_is_an_error(tmp_path: Path) -> None:
    record = read_outcome(tmp_path, _launched(0))
    assert record.status == "ERROR"
    assert "verdict.json is missing" in (record.message or "")


def test_invalid_json_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "verdict.json").write_text("{not json", encoding="utf-8")
    record = read_outcome(tmp_path, _launched(1))
    assert record.status == "ERROR"
    assert "cannot be read" in (record.message or "")


@pytest.mark.parametrize("field", ["status", "metric", "findings"])
def test_missing_field_is_an_error(tmp_path: Path, field: str) -> None:
    data = _verdict(Status.REJECTED)
    del data[field]
    _write(tmp_path, data)
    record = read_outcome(tmp_path, _launched(1))
    assert record.status == "ERROR"
    assert "missing a field" in (record.message or "")


def test_status_that_disagrees_with_the_code_is_an_error(tmp_path: Path) -> None:
    _write(tmp_path, _verdict(Status.APPROVED))
    record = read_outcome(tmp_path, _launched(1))
    assert record.status == "ERROR"
    assert record.message == "exit code 1 but verdict.json says APPROVED"


def test_json_that_is_not_an_object_is_an_error(tmp_path: Path) -> None:
    _write(tmp_path, [1, 2])
    assert read_outcome(tmp_path, _launched(0)).status == "ERROR"


# --- find_command ---------------------------------------------------------------------


def test_find_command_prefers_the_interpreter_folder() -> None:
    calls: list[dict[str, str]] = []

    def which(name: str, **kwargs: str) -> str | None:
        calls.append(kwargs)
        return "beside" if kwargs.get("path") else "on-path"

    assert find_command(which) == "beside"
    assert calls[0]["path"] == str(Path(sys.executable).parent)


def test_find_command_falls_back_to_path() -> None:
    def which(name: str, **kwargs: str) -> str | None:
        return None if kwargs.get("path") else "on-path"

    assert find_command(which) == "on-path"


def test_find_command_returns_none_when_missing() -> None:
    assert find_command(lambda name, **kwargs: None) is None


# --- run_app --------------------------------------------------------------------------


def test_run_app_passes_exactly_run_and_the_app_and_the_hard_limit(
    tmp_path: Path,
) -> None:
    app = tmp_path / "case" / "leaky"
    _write(app, _verdict(Status.APPROVED))
    seen: list[tuple[list[str], Path, float]] = []

    def launcher(argv: Sequence[str], cwd: Path, limit: float) -> Launched:
        seen.append((list(argv), cwd, limit))
        return _launched(0)

    record = run_app("hexperiment", app, launcher=launcher)
    assert record.status == "APPROVED"
    assert seen == [(["hexperiment", "run", str(app)], tmp_path, HARD_LIMIT_SECONDS)]
    assert HARD_LIMIT_SECONDS == 240
