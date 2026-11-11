"""Tests of the ``hexperiment`` command (Requirements 16.1 to 16.5).

A fake runner stands in for the training script, so these tests start no Python
process; ``test_canary_scripts.py`` runs the command on real scripts.
"""

import io
import json
import sys
import tomllib
from functools import partial
from importlib.metadata import entry_points
from pathlib import Path

import numpy as np
import pytest

from hexperiment import cli
from hexperiment.canary import LoadMarker, Snapshot, run_executions
from hexperiment.cli import (
    EXIT_APPROVED,
    EXIT_REJECTED,
    EXIT_RUN_FAILED,
    EXIT_UNVERIFIED,
    build_parser,
    exit_code_for,
    main,
)
from hexperiment.config import ExecMode
from hexperiment.models import (
    CheckResult,
    DataAuditResult,
    DataSummary,
    Finding,
    MetricResult,
    Outcome,
    Severity,
    Status,
)

HEX_TOML = """\
[app]
name = "demo"

[data]
path = "data.csv"

[target]
column = "target"

[metric]
name = "roc_auc"
threshold = 0.8
"""

STALE_VERDICT = '{"status": "APPROVED"}\n'
HINT_LOCATION = "train.py:7"
RUN_ID = "20251101-093012-a1b2c3"


def make_app(root: Path, *, train: bool = True, toml: str = HEX_TOML) -> Path:
    app = root / "app"
    app.mkdir()
    (app / "hex.toml").write_text(toml, encoding="utf-8")
    if train:
        (app / "train.py").write_text("raise SystemExit\n", encoding="utf-8")
    (app / "verdict.json").write_text(STALE_VERDICT, encoding="utf-8")
    return app


def make_result(*findings: Finding) -> DataAuditResult:
    """A normal-mode result that passed the metric, with ``findings`` on checks."""
    names = dict.fromkeys(["split_order", *(f.check for f in findings)])
    checks = []
    for name in names:
        own = tuple(f for f in findings if f.check == name)
        failed = any(f.severity is Severity.ERROR for f in own)
        ghost = next((f.ghost for f in own), None)
        outcome = Outcome.FAILED if failed else Outcome.PASSED
        checks.append(CheckResult(name, ghost, outcome, own))
    return DataAuditResult(
        run_id=None,
        metric=MetricResult("roc_auc", 0.85, 0.8, True),
        checks=tuple(checks),
        extra_findings=(),
        data=DataSummary(80, 20, 0.5, 0.5, None, None),
        input_error=None,
    )


def hint(location: str = HINT_LOCATION) -> Finding:
    return Finding(
        check="split_order",
        ghost=None,
        severity=Severity.WARNING,
        location=location,
        message="mean() is called before the first split",
        evidence={},
        fix="this runs before the split; make sure it does not use test rows",
    )


def make_snapshot(hash_offset: int = 0, mode: ExecMode = ExecMode.NORMAL) -> Snapshot:
    ids = np.arange(4, dtype=np.int64)
    return Snapshot(
        mode=mode,
        run_id=RUN_ID,
        features=("x",),
        train_ids=ids,
        test_ids=np.array([4, 5], dtype=np.int64),
        train_row_hash=(ids + 1000 + hash_offset).astype(np.uint64),
        y_score_train=np.array([0.1, 0.2, 0.3, 0.4]),
    )


def marker() -> LoadMarker:
    return LoadMarker(ExecMode.CANARY, RUN_ID, np.array([4, 5], dtype=np.int64))


@pytest.fixture
def run_with(monkeypatch, fake_runner):
    """Make ``main`` use a fake runner built from the given plans."""

    def install(*, normal=None, canary=None, control=None):
        runner = fake_runner(normal=normal, canary=canary, control=control)
        run = partial(run_executions, runner=runner)
        monkeypatch.setattr(cli, "run_executions", run)
        return runner

    return install


def passing_plans(fake_runner, *findings: Finding) -> dict:
    plan = fake_runner.Plan
    return {
        "normal": plan(snapshot=make_snapshot(), result=make_result(*findings)),
        "canary": plan(snapshot=make_snapshot(), marker=marker()),
    }


def leaking_plans(fake_runner, *findings: Finding) -> dict:
    plan = fake_runner.Plan
    return {
        "normal": plan(snapshot=make_snapshot(), result=make_result(*findings)),
        "canary": plan(snapshot=make_snapshot(hash_offset=7), marker=marker()),
        "control": plan(snapshot=make_snapshot()),
    }


# --- arguments ---------------------------------------------------------------------


def test_the_default_mode_is_a_full_run():
    args = build_parser().parse_args(["run", "some_app"])
    assert args.command == "run"
    assert args.app_dir == "some_app"
    assert args.full is True


def test_fast_and_full_flags_set_the_mode():
    parser = build_parser()
    assert parser.parse_args(["run", "app", "--fast"]).full is False
    assert parser.parse_args(["run", "app", "--full"]).full is True


@pytest.mark.parametrize(
    "argv",
    [[], ["run"], ["run", "app", "--fast", "--full"], ["run", "app", "--slow"]],
)
def test_bad_arguments_are_rejected_with_exit_code_2(argv, capsys):
    with pytest.raises(SystemExit) as stop:
        main(argv)
    assert stop.value.code == 2
    assert "usage" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (Status.APPROVED, 0),
        (Status.REJECTED, 1),
        (Status.UNVERIFIED, 3),
    ],
)
def test_exit_code_for_each_status(status, code):
    assert exit_code_for(status) == code


def test_the_exit_code_constants_are_the_documented_ones():
    assert (EXIT_APPROVED, EXIT_REJECTED, EXIT_RUN_FAILED, EXIT_UNVERIFIED) == (
        0,
        1,
        2,
        3,
    )


def test_the_console_script_points_at_main():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["scripts"]
    assert scripts == {"hexperiment": "hexperiment.cli:main"}
    installed = {
        e.name: e.value for e in entry_points(group="console_scripts")
    }.get("hexperiment")
    assert installed in (None, "hexperiment.cli:main")
    assert cli.main is main


# --- output ------------------------------------------------------------------------


def test_a_full_run_prints_the_status_the_metric_and_one_line_per_finding(
    tmp_path, run_with, fake_runner, capsys
):
    app = make_app(tmp_path)
    run_with(**leaking_plans(fake_runner, hint()))
    code = main(["run", str(app)])
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert code == 1
    assert captured.err == ""
    assert lines[0].startswith("REJECTED (full)")
    assert "roc_auc 0.8500 >= 0.8000" in lines[0]
    verdict = json.loads((app / "verdict.json").read_text(encoding="utf-8"))
    assert lines[0].endswith(f"run {verdict['run_id']}")
    assert len(lines) == 1 + len(verdict["findings"]) == 3
    warning = next(line for line in lines if line.lstrip().startswith("warning"))
    assert HINT_LOCATION in warning
    assert "make sure it does not use test rows" in warning
    error = next(line for line in lines if line.lstrip().startswith("error"))
    assert "The Peeking Spirit" in error
    assert "train.py" in error
    assert f"Suspect lines: {HINT_LOCATION}." in error


def test_a_failed_metric_is_printed_with_a_less_than_sign(
    tmp_path, run_with, fake_runner, capsys
):
    app = make_app(tmp_path)
    plans = passing_plans(fake_runner)
    failing = DataAuditResult(
        run_id=None,
        metric=MetricResult("roc_auc", 0.7, 0.8, False),
        checks=make_result().checks,
        extra_findings=(),
        data=make_result().data,
        input_error=None,
    )
    plans["normal"] = fake_runner.Plan(snapshot=make_snapshot(), result=failing)
    run_with(**plans)
    assert main(["run", str(app)]) == 1
    assert "roc_auc 0.7000 < 0.8000" in capsys.readouterr().out


def test_a_full_run_that_passes_is_approved_and_leaves_the_verdict(
    tmp_path, run_with, fake_runner, capsys
):
    app = make_app(tmp_path)
    runner = run_with(**passing_plans(fake_runner))
    assert main(["run", str(app)]) == 0
    assert runner.calls == [ExecMode.NORMAL, ExecMode.CANARY]
    assert capsys.readouterr().out.startswith("APPROVED (full)")
    (run_dir,) = (app / "runs").iterdir()
    app_copy = (app / "verdict.json").read_bytes()
    assert app_copy == (run_dir / "verdict.json").read_bytes()
    assert (run_dir / "model_card.md").is_file()
    assert json.loads(app_copy)["status"] == "APPROVED"


def test_a_fast_run_executes_once_and_is_unverified(
    tmp_path, run_with, fake_runner, capsys
):
    app = make_app(tmp_path)
    runner = run_with(**passing_plans(fake_runner))
    assert main(["run", str(app), "--fast"]) == 3
    assert runner.calls == [ExecMode.NORMAL]
    assert capsys.readouterr().out.startswith("UNVERIFIED (fast)")


def test_a_relative_app_folder_is_resolved(
    tmp_path, run_with, fake_runner, monkeypatch, capsys
):
    make_app(tmp_path)
    monkeypatch.chdir(tmp_path)
    run_with(**passing_plans(fake_runner))
    assert main(["run", "app", "--fast"]) == 3
    assert (tmp_path / "app" / "runs").is_dir()


# --- exit code 2: no new verdict, no stale verdict ---------------------------------


def stale_verdict_is_gone(app: Path) -> bool:
    return not (app / "verdict.json").exists()


def test_a_missing_app_folder_returns_2(tmp_path, capsys):
    assert main(["run", str(tmp_path / "nowhere")]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("hexperiment: ")
    assert "nowhere" in captured.err


def test_a_missing_train_script_returns_2_and_removes_the_stale_verdict(
    tmp_path, capsys
):
    app = make_app(tmp_path, train=False)
    assert main(["run", str(app)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "train.py" in captured.err
    assert stale_verdict_is_gone(app)
    assert not (app / "runs").exists()


def test_a_bad_config_returns_2_and_removes_the_stale_verdict(tmp_path, capsys):
    app = make_app(tmp_path, toml=HEX_TOML.replace("threshold = 0.8", "threshold = 7"))
    assert main(["run", str(app)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("hexperiment: ")
    assert "hex.toml" in err
    assert "metric.threshold" in err
    assert stale_verdict_is_gone(app)


def test_a_missing_config_returns_2_and_removes_the_stale_verdict(tmp_path, capsys):
    app = make_app(tmp_path)
    (app / "hex.toml").unlink()
    assert main(["run", str(app)]) == 2
    assert "hex.toml" in capsys.readouterr().err
    assert stale_verdict_is_gone(app)


def test_a_failed_execution_returns_2_with_the_stderr_tail(
    tmp_path, run_with, fake_runner, capsys
):
    app = make_app(tmp_path)
    failure = "train.py exited with code 1\nLast lines of stderr:\nValueError: boom"
    run_with(normal=fake_runner.Plan(fail=failure))
    assert main(["run", str(app)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ValueError: boom" in captured.err
    assert stale_verdict_is_gone(app)
    assert not (app / "runs").exists()


def test_a_timed_out_execution_returns_2(tmp_path, run_with, fake_runner, capsys):
    app = make_app(tmp_path)
    run_with(normal=fake_runner.Plan(timed_out=True))
    assert main(["run", str(app)]) == 2
    assert "longer than the timeout" in capsys.readouterr().err
    assert stale_verdict_is_gone(app)


def test_a_script_that_never_audits_returns_2(tmp_path, run_with, fake_runner, capsys):
    app = make_app(tmp_path)
    run_with(normal=fake_runner.Plan())
    assert main(["run", str(app)]) == 2
    assert "hexperiment.audit" in capsys.readouterr().err
    assert stale_verdict_is_gone(app)


def test_a_malformed_history_file_returns_2_and_writes_nothing_new(
    tmp_path, run_with, fake_runner, capsys
):
    app = make_app(tmp_path)
    broken = app / "runs" / "20250101-000000-abcdef"
    broken.mkdir(parents=True)
    (broken / "verdict.json").write_text("{not json", encoding="utf-8")
    run_with(**passing_plans(fake_runner))
    assert main(["run", str(app)]) == 2
    assert "20250101-000000-abcdef" in capsys.readouterr().err
    assert stale_verdict_is_gone(app)
    assert [p.name for p in (app / "runs").iterdir()] == ["20250101-000000-abcdef"]


def test_an_existing_run_folder_returns_2_and_stays_untouched(
    tmp_path, run_with, fake_runner, monkeypatch, capsys
):
    app = make_app(tmp_path)
    taken = app / "runs" / RUN_ID
    taken.mkdir(parents=True)
    (taken / "note.txt").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(cli, "new_run_id", lambda: RUN_ID)
    run_with(**passing_plans(fake_runner))
    assert main(["run", str(app)]) == 2
    assert RUN_ID in capsys.readouterr().err
    assert (taken / "note.txt").read_text(encoding="utf-8") == "keep"
    assert stale_verdict_is_gone(app)


# --- console encoding --------------------------------------------------------------


def ascii_stream() -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding="ascii", write_through=True)


def test_the_summary_survives_a_console_that_cannot_encode_it(
    tmp_path, run_with, fake_runner, monkeypatch
):
    app = make_app(tmp_path)
    proxy = Finding(
        check="target_proxy",
        ghost="The Possession",
        severity=Severity.ERROR,
        location="größe",
        message="größe alone predicts the target",
        evidence={},
        fix="drop it",
    )
    run_with(**passing_plans(fake_runner, proxy))
    stdout = ascii_stream()
    monkeypatch.setattr(sys, "stdout", stdout)
    assert main(["run", str(app), "--fast"]) == 1
    text = stdout.buffer.getvalue().decode("ascii")
    assert "gr??e" in text
    assert text.startswith("REJECTED (fast)")


def test_an_error_message_survives_a_console_that_cannot_encode_it(
    tmp_path, monkeypatch
):
    stderr = ascii_stream()
    monkeypatch.setattr(sys, "stderr", stderr)
    assert main(["run", str(tmp_path / "größe")]) == 2
    text = stderr.buffer.getvalue().decode("ascii")
    assert text.startswith("hexperiment: ")
    assert "gr??e" in text
