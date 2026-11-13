"""Tests of the ``hexperiment`` command (Requirements 16.1 to 16.5).

A fake runner stands in for the training script, so these tests start no Python
process; ``test_canary_scripts.py`` runs the command on real scripts.
"""

import importlib
import io
import json
import os
import shutil
import sys
import tomllib
from functools import partial
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import numpy as np
import pytest

from hexperiment import cli
from hexperiment import hook as hook_module
from hexperiment.canary import LoadMarker, Snapshot, run_executions
from hexperiment.cli import (
    EXIT_APPROVED,
    EXIT_REJECTED,
    EXIT_RUN_FAILED,
    EXIT_UNVERIFIED,
    build_parser,
    exit_code_for,
    main,
    run_app,
)
from hexperiment.config import ExecMode
from hexperiment.entry import main as entry_main
from hexperiment.models import (
    CheckResult,
    DataAuditResult,
    DataSummary,
    Finding,
    MetricResult,
    Outcome,
    RunOutcome,
    Severity,
    Status,
)
from hexperiment.runs import app_lock

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


def test_pyproject_maps_the_console_script_to_the_light_entry_point():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["scripts"]
    assert scripts == {
        "hexperiment": "hexperiment.entry:main",
        "hexperiment-mcp": "hexperiment.mcp_server:main",
    }
    module, _, attribute = scripts["hexperiment"].partition(":")
    assert getattr(importlib.import_module(module), attribute) is entry_main


def test_the_console_script_is_installed_in_the_environment():
    try:
        dist = distribution("hexperiment")
    except PackageNotFoundError:
        pytest.skip(
            "the hexperiment package is not installed; "
            'run: .venv/Scripts/python -m pip install -e "hexperiment[dev]"'
        )
    installed = {
        entry.name: entry.value
        for entry in dist.entry_points
        if entry.group == "console_scripts"
    }
    assert installed.get("hexperiment") == "hexperiment.entry:main", (
        "the package is installed but has no console script 'hexperiment'; "
        "reinstall it with pip install -e"
    )
    executable = "hexperiment.exe" if os.name == "nt" else "hexperiment"
    script = Path(sys.executable).parent / executable
    assert script.is_file(), f"{script} is missing; reinstall with pip install -e"


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


def test_a_timed_out_execution_returns_2_with_the_message_of_the_runner(
    tmp_path, capsys
):
    toml = HEX_TOML + "\n[run]\ntimeout_seconds = 1\n"
    app = make_app(tmp_path, toml=toml)
    (app / "train.py").write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    assert main(["run", str(app)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(
        "hexperiment: train.py ran longer than 1 seconds in normal mode and was "
        "stopped. Make training faster, or raise [run] timeout_seconds in hex.toml."
    )
    assert stale_verdict_is_gone(app)
    assert not (app / "runs").exists()


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


# --- exit code 2 for failures that are not REJECTED --------------------------------


def test_a_delete_that_fails_twice_is_retried_and_the_run_goes_on(
    tmp_path, run_with, fake_runner, refuse_unlink, capsys
):
    app = make_app(tmp_path)
    attempts, pauses = refuse_unlink(app, failures=2)
    run_with(**passing_plans(fake_runner))
    assert main(["run", str(app)]) == 0
    assert (len(attempts), pauses) == (3, [0.2, 0.2])
    assert json.loads((app / "verdict.json").read_text(encoding="utf-8"))["run_id"]


def test_a_stale_verdict_that_cannot_be_deleted_returns_2_and_runs_nothing(
    tmp_path, run_with, fake_runner, refuse_unlink, capsys
):
    app = make_app(tmp_path)
    attempts, _ = refuse_unlink(app, failures=None)
    runner = run_with(**passing_plans(fake_runner))
    assert main(["run", str(app)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("hexperiment: ")
    assert "could not be removed" in captured.err
    assert "must not be trusted" in captured.err
    assert len(attempts) == 6
    assert runner.calls == []
    assert not (app / "runs").exists()


@pytest.mark.parametrize("error", [RuntimeError("boom"), KeyError("key")])
def test_an_unexpected_exception_returns_2_with_the_type_and_a_traceback(
    error, tmp_path, run_with, fake_runner, monkeypatch, capsys
):
    app = make_app(tmp_path)
    run_with(**passing_plans(fake_runner))

    def explode(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli, "finalize_run", explode)
    assert main(["run", str(app)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    first, *rest = captured.err.splitlines()
    assert first == f"hexperiment: internal error: {type(error).__name__}: {error}"
    assert any(line.startswith("Traceback") for line in rest)
    assert stale_verdict_is_gone(app)
    assert not (app / "runs").exists()


def test_an_unexpected_exception_while_running_returns_2(tmp_path, monkeypatch, capsys):
    app = make_app(tmp_path)

    def explode(*args, **kwargs):
        raise ZeroDivisionError("division by zero")

    monkeypatch.setattr(cli, "run_executions", explode)
    assert main(["run", str(app)]) == 2
    assert "internal error: ZeroDivisionError: division by zero" in (
        capsys.readouterr().err
    )
    assert stale_verdict_is_gone(app)


def test_a_keyboard_interrupt_is_not_swallowed(
    tmp_path, run_with, fake_runner, monkeypatch
):
    app = make_app(tmp_path)
    run_with(**passing_plans(fake_runner))

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "finalize_run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        main(["run", str(app)])


@pytest.mark.parametrize("error", [OSError("closed pipe"), ValueError("I/O on closed")])
def test_a_failure_to_print_the_summary_keeps_the_status_exit_code(
    error, tmp_path, run_with, fake_runner, monkeypatch
):
    app = make_app(tmp_path)
    run_with(**leaking_plans(fake_runner))
    printed = []

    def print_text(text, file=None):
        printed.append(text)
        raise error

    monkeypatch.setattr(cli, "print_text", print_text)
    assert main(["run", str(app)]) == 1
    assert len(printed) == 1
    assert printed[0].startswith("REJECTED (full)")
    stored = json.loads((app / "verdict.json").read_text(encoding="utf-8"))
    assert stored["status"] == "REJECTED"


def test_a_failure_to_print_an_error_still_returns_2(tmp_path, monkeypatch):
    def print_text(text, file=None):
        raise OSError("closed pipe")

    monkeypatch.setattr(cli, "print_text", print_text)
    assert main(["run", str(tmp_path / "nowhere")]) == 2


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


# --- run_app: the text of a run, nothing printed (Requirements 1.3, 1.5, 4.6) -------


def test_run_app_returns_an_approved_outcome_and_prints_nothing(
    tmp_path, run_with, fake_runner, capsys
):
    app = make_app(tmp_path)
    run_with(**passing_plans(fake_runner))
    outcome = run_app(app, full=True)
    assert isinstance(outcome, RunOutcome)
    assert (outcome.exit_code, outcome.status, outcome.stderr) == (
        0,
        Status.APPROVED,
        "",
    )
    assert outcome.stdout.startswith("APPROVED (full)")
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")


def test_run_app_returns_a_rejected_outcome_with_the_findings(
    tmp_path, run_with, fake_runner, capsys
):
    app = make_app(tmp_path)
    run_with(**leaking_plans(fake_runner, hint()))
    outcome = run_app(app, full=True)
    assert (outcome.exit_code, outcome.status, outcome.stderr) == (
        1,
        Status.REJECTED,
        "",
    )
    lines = outcome.stdout.splitlines()
    assert lines[0].startswith("REJECTED (full)")
    assert len(lines) == 3
    assert any("The Peeking Spirit" in line for line in lines)
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")


def test_run_app_returns_an_unverified_outcome_for_a_fast_run(
    tmp_path, run_with, fake_runner, capsys
):
    app = make_app(tmp_path)
    runner = run_with(**passing_plans(fake_runner))
    outcome = run_app(app, full=False)
    assert (outcome.exit_code, outcome.status, outcome.stderr) == (
        3,
        Status.UNVERIFIED,
        "",
    )
    assert outcome.stdout.startswith("UNVERIFIED (fast)")
    assert runner.calls == [ExecMode.NORMAL]
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")


def test_run_app_returns_a_failed_outcome_with_the_message(
    tmp_path, run_with, fake_runner, capsys
):
    app = make_app(tmp_path)
    failure = "train.py exited with code 1\nLast lines of stderr:\nValueError: boom"
    run_with(normal=fake_runner.Plan(fail=failure))
    outcome = run_app(app, full=True)
    assert outcome == RunOutcome(2, None, "", f"hexperiment: {failure}")
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")


def test_run_app_removes_the_stale_verdict_when_the_run_fails(
    tmp_path, run_with, fake_runner
):
    app = make_app(tmp_path)
    run_with(normal=fake_runner.Plan(fail="train.py exited with code 1"))
    assert (app / "verdict.json").is_file()
    assert run_app(app, full=True).exit_code == 2
    assert stale_verdict_is_gone(app)


def test_run_app_reports_a_missing_folder_as_a_failed_outcome(tmp_path, capsys):
    outcome = run_app(tmp_path / "nowhere", full=True)
    assert (outcome.exit_code, outcome.status, outcome.stdout) == (2, None, "")
    assert outcome.stderr.startswith("hexperiment: ")
    assert "nowhere" in outcome.stderr
    assert capsys.readouterr().out == ""


def test_run_app_reports_an_unexpected_exception_with_a_traceback(
    tmp_path, monkeypatch
):
    app = make_app(tmp_path)

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "run_executions", explode)
    outcome = run_app(app, full=True)
    first, *rest = outcome.stderr.splitlines()
    assert first == "hexperiment: internal error: RuntimeError: boom"
    assert any(line.startswith("Traceback") for line in rest)
    assert (outcome.exit_code, outcome.status, outcome.stdout) == (2, None, "")


def test_run_app_does_not_swallow_a_keyboard_interrupt(
    tmp_path, run_with, fake_runner, monkeypatch
):
    app = make_app(tmp_path)
    run_with(**passing_plans(fake_runner))

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "finalize_run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_app(app, full=True)


def test_main_prints_exactly_the_text_of_run_app(
    tmp_path, run_with, fake_runner, monkeypatch, capsys
):
    app = make_app(tmp_path)
    monkeypatch.setattr(cli, "new_run_id", lambda: RUN_ID)
    run_with(**leaking_plans(fake_runner, hint()))
    code = main(["run", str(app)])
    printed = capsys.readouterr()
    shutil.rmtree(app / "runs")
    outcome = run_app(app, full=True)
    assert code == outcome.exit_code == 1
    assert printed.out == outcome.stdout + "\n"
    assert printed.err == ""


def test_main_prints_the_failure_message_of_run_app_on_stderr(tmp_path, capsys):
    missing = tmp_path / "nowhere"
    code = main(["run", str(missing)])
    printed = capsys.readouterr()
    outcome = run_app(missing, full=True)
    assert code == outcome.exit_code == 2
    assert printed.err == outcome.stderr + "\n"
    assert printed.out == ""


# --- timeout_cap (Requirement 1.6) ---------------------------------------------------

TIMEOUT_TOML = HEX_TOML + "\n[run]\ntimeout_seconds = 30\n"


@pytest.mark.parametrize(
    ("toml", "cap", "expected"),
    [
        (HEX_TOML + "\n[run]\ntimeout_seconds = 120\n", 100, 100),
        (TIMEOUT_TOML, 100, 30),
        (HEX_TOML + "\n[run]\ntimeout_seconds = 120\n", None, 120),
        (HEX_TOML, None, 120),
        (HEX_TOML, 100, 100),
    ],
    ids=["cap-below", "configured-below", "no-cap", "default", "default-capped"],
)
def test_timeout_cap_limits_the_timeout_of_every_execution(
    toml, cap, expected, tmp_path, run_with, fake_runner
):
    app = make_app(tmp_path, toml=toml)
    runner = run_with(**passing_plans(fake_runner))
    assert run_app(app, full=True, timeout_cap=cap).exit_code == 0
    assert runner.calls == [ExecMode.NORMAL, ExecMode.CANARY]
    assert runner.timeouts == [expected, expected]


def test_timeout_cap_leaves_hex_toml_untouched(tmp_path, run_with, fake_runner):
    app = make_app(tmp_path, toml=TIMEOUT_TOML)
    before = (app / "hex.toml").read_bytes()
    run_with(**passing_plans(fake_runner))
    run_app(app, full=False, timeout_cap=5)
    assert (app / "hex.toml").read_bytes() == before


# --- hexperiment hook (Requirements 1.2 to 1.7) -------------------------------------

HOOK_NOTICE = (
    "Fast check only: the canary test did not run, so this result can never be "
    "APPROVED. Before you call the work done, run: hexperiment run app"
)


def feed_stdin(monkeypatch, data: str | bytes) -> None:
    """Replace standard input with ``data``, as a pipe from Kiro would."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    stream = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stream)


def hook_event_for(app: Path, name: str = "train.py") -> str:
    """The shape of event that Kiro writes: flat, with an absolute file_path."""
    return json.dumps(
        {
            "session_id": "sess_01HZ",
            "hook_event_name": "PostFileSave",
            "cwd": str(app.parent),
            "file_path": str(app / name),
        }
    )


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    """The workspace of the hook: the working directory, no override variable."""
    monkeypatch.delenv("HEXPERIMENT_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def proxy_finding() -> Finding:
    return Finding(
        check="target_proxy",
        ghost="The Possession",
        severity=Severity.ERROR,
        location="tenure",
        message="tenure alone predicts the target",
        evidence={},
        fix="drop it if it is recorded after the outcome",
    )


def test_the_parser_accepts_hook_without_arguments():
    parser = build_parser()
    assert parser.parse_args(["hook"]).command == "hook"
    with pytest.raises(SystemExit) as stop:
        parser.parse_args(["hook", "extra"])
    assert stop.value.code == 2


def test_a_clean_fast_run_prints_the_summary_and_the_notice_and_exits_0(
    project, run_with, fake_runner, monkeypatch, capsys
):
    app = make_app(project)
    runner = run_with(**passing_plans(fake_runner))
    feed_stdin(monkeypatch, hook_event_for(app))
    assert main(["hook"]) == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0].startswith("UNVERIFIED (fast)")
    assert lines[-1] == HOOK_NOTICE
    assert len(lines) == 2
    assert captured.err == ""
    assert runner.calls == [ExecMode.NORMAL]


def test_the_hook_run_is_capped_below_the_hook_timeout(
    project, run_with, fake_runner, monkeypatch
):
    app = make_app(project, toml=HEX_TOML + "\n[run]\ntimeout_seconds = 120\n")
    runner = run_with(**passing_plans(fake_runner))
    feed_stdin(monkeypatch, hook_event_for(app))
    assert main(["hook"]) == 0
    assert runner.timeouts == [100]


def test_a_short_configured_timeout_is_kept_by_the_hook(
    project, run_with, fake_runner, monkeypatch
):
    app = make_app(project, toml=HEX_TOML + "\n[run]\ntimeout_seconds = 30\n")
    runner = run_with(**passing_plans(fake_runner))
    feed_stdin(monkeypatch, hook_event_for(app))
    assert main(["hook"]) == 0
    assert runner.timeouts == [30]


def test_a_rejected_fast_run_exits_0_and_prints_the_finding(
    project, run_with, fake_runner, monkeypatch, capsys
):
    app = make_app(project)
    run_with(**passing_plans(fake_runner, proxy_finding()))
    feed_stdin(monkeypatch, hook_event_for(app))
    assert main(["hook"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("REJECTED (fast)")
    assert any("The Possession" in line and "tenure" in line for line in lines)
    assert lines[-1] == HOOK_NOTICE


def test_a_failed_run_exits_0_and_prints_the_message(
    project, run_with, fake_runner, monkeypatch, capsys
):
    app = make_app(project)
    failure = "train.py exited with code 1\nLast lines of stderr:\nValueError: boom"
    run_with(normal=fake_runner.Plan(fail=failure))
    feed_stdin(monkeypatch, hook_event_for(app))
    assert main(["hook"]) == 0
    captured = capsys.readouterr()
    assert captured.out == f"hexperiment: {failure}\n{HOOK_NOTICE}\n"
    assert captured.err == ""
    assert stale_verdict_is_gone(app)


def test_a_hook_run_is_recorded_and_counts_as_a_test_set_use(
    project, run_with, fake_runner, monkeypatch, capsys
):
    app = make_app(project)
    run_with(**passing_plans(fake_runner))
    feed_stdin(monkeypatch, hook_event_for(app))
    assert main(["hook"]) == 0
    (run_dir,) = (app / "runs").iterdir()
    first = json.loads((app / "verdict.json").read_text(encoding="utf-8"))
    assert first["run_id"] == run_dir.name
    stored = (app / "verdict.json").read_bytes()
    assert (run_dir / "verdict.json").read_bytes() == stored
    assert (run_dir / "model_card.md").is_file()
    feed_stdin(monkeypatch, hook_event_for(app))
    assert main(["hook"]) == 0
    assert len(list((app / "runs").iterdir())) == 2
    second = json.loads((app / "verdict.json").read_text(encoding="utf-8"))
    assert second["test_set_uses"] == first["test_set_uses"] + 1
    capsys.readouterr()


def test_the_hook_can_find_the_app_from_a_helper_file(
    project, run_with, fake_runner, monkeypatch, capsys
):
    app = make_app(project)
    helper = app / "features" / "encode.py"
    helper.parent.mkdir()
    helper.write_text("x = 1\n", encoding="utf-8")
    runner = run_with(**passing_plans(fake_runner))
    feed_stdin(monkeypatch, hook_event_for(app, "features/encode.py"))
    assert main(["hook"]) == 0
    assert runner.calls == [ExecMode.NORMAL]
    assert capsys.readouterr().out.splitlines()[0].startswith("UNVERIFIED (fast)")


@pytest.mark.parametrize("data", ["", "   \n", "{not json", "null", "[]"])
def test_input_without_a_file_prints_nothing_and_exits_0(
    data, project, run_with, fake_runner, monkeypatch, capsys
):
    make_app(project)
    runner = run_with(**passing_plans(fake_runner))
    feed_stdin(monkeypatch, data)
    assert main(["hook"]) == 0
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")
    assert runner.calls == []


def test_a_non_python_file_prints_nothing_and_exits_0(
    project, run_with, fake_runner, monkeypatch, capsys
):
    app = make_app(project)
    runner = run_with(**passing_plans(fake_runner))
    feed_stdin(monkeypatch, hook_event_for(app, "hex.toml"))
    assert main(["hook"]) == 0
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")
    assert runner.calls == []
    assert (app / "verdict.json").is_file()


def test_a_python_file_outside_the_workspace_prints_nothing(
    tmp_path, run_with, fake_runner, monkeypatch, capsys
):
    workspace = tmp_path / "project"
    workspace.mkdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    app = make_app(foreign)
    monkeypatch.delenv("HEXPERIMENT_WORKSPACE", raising=False)
    monkeypatch.chdir(workspace)
    runner = run_with(**passing_plans(fake_runner))
    feed_stdin(monkeypatch, hook_event_for(app))
    assert main(["hook"]) == 0
    assert capsys.readouterr().out == ""
    assert runner.calls == []


def test_the_workspace_variable_overrides_the_working_directory(
    tmp_path, run_with, fake_runner, monkeypatch, capsys
):
    workspace = tmp_path / "project"
    workspace.mkdir()
    app = make_app(workspace)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("HEXPERIMENT_WORKSPACE", str(workspace))
    runner = run_with(**passing_plans(fake_runner))
    feed_stdin(monkeypatch, hook_event_for(app))
    assert main(["hook"]) == 0
    assert runner.calls == [ExecMode.NORMAL]
    assert capsys.readouterr().out.startswith("UNVERIFIED (fast)")


def test_a_terminal_as_standard_input_counts_as_empty_and_is_never_read(
    project, run_with, fake_runner, monkeypatch, capsys
):
    class Terminal:
        buffer = None  # reading it would raise

        def isatty(self) -> bool:
            return True

    make_app(project)
    runner = run_with(**passing_plans(fake_runner))
    monkeypatch.setattr(sys, "stdin", Terminal())
    assert main(["hook"]) == 0
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")
    assert runner.calls == []


def test_a_missing_standard_input_counts_as_empty(project, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", None)
    assert main(["hook"]) == 0
    assert capsys.readouterr().out == ""


def test_a_byte_order_mark_and_invalid_utf8_do_not_stop_the_hook(
    project, run_with, fake_runner, monkeypatch, capsys
):
    app = make_app(project)
    runner = run_with(**passing_plans(fake_runner))
    event = json.loads(hook_event_for(app))
    event["note"] = "PLACEHOLDER"
    raw = json.dumps(event).encode("utf-8").replace(b"PLACEHOLDER", b"\xff\xfe")
    feed_stdin(monkeypatch, b"\xef\xbb\xbf" + raw)
    assert main(["hook"]) == 0
    assert runner.calls == [ExecMode.NORMAL]
    assert capsys.readouterr().out.startswith("UNVERIFIED (fast)")


def test_only_the_first_mebibyte_of_input_is_read(
    project, run_with, fake_runner, monkeypatch, capsys
):
    app = make_app(project)
    runner = run_with(**passing_plans(fake_runner))
    event = json.loads(hook_event_for(app))
    event["padding"] = "x" * (2 * cli.MAX_EVENT_BYTES)
    stream = io.TextIOWrapper(io.BytesIO(json.dumps(event).encode()), encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stream)
    assert main(["hook"]) == 0
    assert capsys.readouterr().out == ""
    assert runner.calls == []
    assert stream.buffer.tell() == cli.MAX_EVENT_BYTES


def test_an_exception_in_the_handler_prints_one_line_and_exits_0(
    project, monkeypatch, capsys
):
    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "handle_hook", explode)
    feed_stdin(monkeypatch, "{}")
    assert main(["hook"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "hexperiment hook failed: RuntimeError: boom\n"
    assert captured.err == ""


def test_an_invalid_workspace_variable_prints_one_line_and_exits_0(
    project, monkeypatch, capsys
):
    monkeypatch.setenv("HEXPERIMENT_WORKSPACE", str(project / "nowhere"))
    feed_stdin(monkeypatch, "{}")
    assert main(["hook"]) == 0
    out = capsys.readouterr().out
    prefix = "hexperiment hook failed: HexInputError: HEXPERIMENT_WORKSPACE"
    assert out.startswith(prefix)
    assert len(out.splitlines()) == 1


def test_a_failure_to_print_does_not_change_the_hook_exit_code(
    project, run_with, fake_runner, monkeypatch
):
    app = make_app(project)
    run_with(**passing_plans(fake_runner, proxy_finding()))
    feed_stdin(monkeypatch, hook_event_for(app))

    def broken(text, file=None):
        raise OSError("closed pipe")

    monkeypatch.setattr(cli, "print_text", broken)
    assert main(["hook"]) == 0


def test_a_keyboard_interrupt_in_the_hook_is_not_swallowed(project, monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "handle_hook", interrupt)
    feed_stdin(monkeypatch, "{}")
    with pytest.raises(KeyboardInterrupt):
        main(["hook"])


# --- the run lock (Requirements 1.8, 8) -----------------------------------------------


def test_a_held_lock_refuses_the_run_with_exit_2_and_changes_nothing(
    tmp_path, run_with, fake_runner, capsys
):
    app = make_app(tmp_path)
    before = (app / "verdict.json").read_bytes()
    runner = run_with(**passing_plans(fake_runner))
    with app_lock(app):
        assert main(["run", str(app)]) == EXIT_RUN_FAILED
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == (
        f"hexperiment: another hexperiment run is in progress on {app}; "
        "wait for it to finish, then run again"
    )
    assert runner.calls == []
    assert not (app / "runs").exists()
    assert (app / "verdict.json").read_bytes() == before


def test_run_app_returns_the_refusal_in_stderr_when_the_lock_is_held(tmp_path):
    app = make_app(tmp_path)
    with app_lock(app):
        outcome = run_app(app, full=False)
    assert outcome.exit_code == EXIT_RUN_FAILED
    assert outcome.status is None
    assert outcome.stdout == ""
    assert "another hexperiment run is in progress" in outcome.stderr


def test_the_lock_is_released_when_the_run_ends(tmp_path, run_with, fake_runner):
    app = make_app(tmp_path)
    run_with(**passing_plans(fake_runner))
    assert main(["run", str(app)]) != EXIT_RUN_FAILED
    with app_lock(app):
        pass


def test_the_lock_is_released_when_the_run_fails(tmp_path):
    app = make_app(tmp_path, train=False)
    assert main(["run", str(app)]) == EXIT_RUN_FAILED
    with app_lock(app):
        pass


def test_the_hook_passes_on_the_refusal_of_a_held_lock_and_exits_0(
    project, run_with, fake_runner, monkeypatch, capsys
):
    app = make_app(project)
    runner = run_with(**passing_plans(fake_runner))
    feed_stdin(monkeypatch, hook_event_for(app))
    with app_lock(app):
        assert main(["hook"]) == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0] == (
        f"hexperiment: another hexperiment run is in progress on {app.resolve()}; "
        "wait for it to finish, then run again"
    )
    assert lines[-1] == HOOK_NOTICE
    assert runner.calls == []
    assert not (app / "runs").exists()


def test_run_help_describes_the_command_and_lists_the_four_exit_codes(capsys):
    with pytest.raises(SystemExit) as stop:
        main(["run", "--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    assert "Run the app's train.py under the audit" in text
    assert "Full mode also runs the canary test." in text
    codes = {
        EXIT_APPROVED: "APPROVED",
        EXIT_REJECTED: "REJECTED",
        EXIT_RUN_FAILED: "the run could not finish",
        EXIT_UNVERIFIED: "UNVERIFIED",
    }
    for code, meaning in codes.items():
        line = next(ln for ln in text.splitlines() if ln.startswith(f"  {code}  "))
        assert meaning in line


# --- hexperiment hook --stop (Requirement 9) ------------------------------------------

STOP_SESSION = "sess_01HZ"


def stop_event(session: str = STOP_SESSION) -> str:
    return json.dumps(
        {"session_id": session, "hook_event_name": "Stop", "cwd": "C:\\work"}
    )


def test_the_parser_accepts_the_stop_flag_only_on_hook():
    parser = build_parser()
    assert parser.parse_args(["hook", "--stop"]).stop is True
    assert parser.parse_args(["hook"]).stop is False
    with pytest.raises(SystemExit) as stop:
        parser.parse_args(["run", "app", "--stop"])
    assert stop.value.code == 2


def test_stop_names_an_unapproved_app_of_the_session_on_stderr_and_exits_1(
    project, monkeypatch, capsys
):
    app = make_app(project)
    (app / "verdict.json").write_text('{"status": "UNVERIFIED"}', encoding="utf-8")
    (app / ".hex").mkdir()
    (app / ".hex" / "agent-session").write_text(STOP_SESSION + "\n", encoding="utf-8")
    feed_stdin(monkeypatch, stop_event())
    assert main(["hook", "--stop"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert lines[0] == "Hexperiment: not approved yet."
    assert lines[1].startswith(
        "- app: changed in this session; its verdict is UNVERIFIED"
    )


def test_stop_exits_0_without_output_for_another_session(project, monkeypatch, capsys):
    app = make_app(project)
    (app / ".hex").mkdir()
    (app / ".hex" / "agent-session").write_text("sess_other\n", encoding="utf-8")
    feed_stdin(monkeypatch, stop_event())
    assert main(["hook", "--stop"]) == 0
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")


@pytest.mark.parametrize("data", ["", "   ", "{not json", "{}", "[]"])
def test_stop_exits_0_without_output_for_an_empty_event(
    data, project, monkeypatch, capsys
):
    make_app(project)
    feed_stdin(monkeypatch, data)
    assert main(["hook", "--stop"]) == 0
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")


def test_stop_exits_0_without_output_when_the_handler_raises(
    project, monkeypatch, capsys
):
    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(hook_module, "handle_stop", explode)
    feed_stdin(monkeypatch, stop_event())
    assert main(["hook", "--stop"]) == 0
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")


def test_stop_exits_0_without_output_for_an_invalid_workspace_variable(
    project, monkeypatch, capsys
):
    monkeypatch.setenv("HEXPERIMENT_WORKSPACE", str(project / "nowhere"))
    feed_stdin(monkeypatch, stop_event())
    assert main(["hook", "--stop"]) == 0
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")


def test_stop_starts_no_run(project, run_with, fake_runner, monkeypatch, capsys):
    app = make_app(project)
    (app / "verdict.json").unlink()
    (app / ".hex").mkdir()
    (app / ".hex" / "agent-session").write_text(STOP_SESSION + "\n", encoding="utf-8")
    runner = run_with(**passing_plans(fake_runner))
    feed_stdin(monkeypatch, stop_event())
    assert main(["hook", "--stop"]) == 1
    assert runner.calls == []
    assert not (app / "runs").exists()
    capsys.readouterr()


def test_a_save_hook_then_a_stop_hook_name_the_app(
    project, run_with, fake_runner, monkeypatch, capsys
):
    app = make_app(project)
    run_with(**passing_plans(fake_runner))
    feed_stdin(monkeypatch, hook_event_for(app))
    assert main(["hook"]) == 0
    capsys.readouterr()
    feed_stdin(monkeypatch, stop_event("sess_01HZ"))
    assert main(["hook", "--stop"]) == 1
    assert "- app: changed in this session; its verdict is UNVERIFIED" in (
        capsys.readouterr().err
    )
