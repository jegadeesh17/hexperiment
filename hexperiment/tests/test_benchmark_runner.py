"""Running one app, reading its outcome, and the runner (one test starts a process).

The runner is reached through the fake launcher, ``find`` and clock arguments of
``runner.main``: no test runs a case, starts ``hexperiment run`` or a training
script. The one process this file starts is the entry point with an unknown key.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

import update_readme
from hexbench.cases import all_cases
from hexbench.execution import (
    HARD_LIMIT_SECONDS,
    Launched,
    RunRecord,
    find_command,
    read_outcome,
    run_app,
)
from hexbench.mechanisms import MECHANISMS, NOT_MEASURED
from hexbench.report import BEGIN, END
from hexbench.runner import main
from hexbench.summary import CaseRecord, build_results
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


# --- runner.main with fakes ---------------------------------------------------------

type Behave = Callable[[str, str], tuple[int | None, dict[str, object] | None, float]]

PEEKING = "The Peeking Spirit"
BENCHMARK = Path(__file__).resolve().parents[1] / "benchmark"


def _rejected(value: float | None = 0.8) -> dict[str, object]:
    findings = (_finding("canary", PEEKING, Severity.ERROR),)
    return _verdict(Status.REJECTED, findings, value=value)


def _default_behaviour(
    case_id: str, variant: str
) -> tuple[int, dict[str, object], float]:
    if variant == "leaky":
        return 1, _rejected(0.8), 2.0
    return 0, _verdict(Status.APPROVED, value=0.7), 1.0


class _Launches:
    """A fake launcher: checks the app on disk, then writes a verdict.json."""

    def __init__(self, behave: Behave = _default_behaviour) -> None:
        self.behave = behave
        self.apps: list[Path] = []
        self.argv: list[list[str]] = []
        self.limits: list[float] = []

    def __call__(self, argv: Sequence[str], cwd: Path, limit: float) -> Launched:
        app = Path(argv[2])
        assert cwd == app.parent.parent
        for name in ("train.py", "hex.toml", "data/dataset.csv"):
            assert (app / name).is_file(), f"{name} is missing in {app}"
        self.apps.append(app)
        self.argv.append(list(argv))
        self.limits.append(limit)
        code, data, seconds = self.behave(app.parent.name, app.name)
        if data is not None:
            (app / "verdict.json").write_text(json.dumps(data), encoding="utf-8")
        return Launched(exit_code=code, stderr="boom\n", seconds=seconds)

    @property
    def roots(self) -> set[Path]:
        return {app.parent.parent for app in self.apps}


class _Clock:
    def __init__(self, *times: float) -> None:
        self.times = list(times)

    def __call__(self) -> float:
        return self.times.pop(0)


@pytest.fixture(autouse=True)
def _temp_under_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner's temporary folder goes under ``tmp_path``."""
    folder = tmp_path / "system-temp"
    folder.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(folder))


def _run(
    tmp_path: Path,
    *args: str,
    launcher: Callable[[Sequence[str], Path, float], Launched] | None = None,
    find: Callable[[], str | None] = lambda: "hexperiment",
    clock: Callable[[], float] = lambda: 0.0,
) -> tuple[int, Path]:
    out = tmp_path / "out" / "results.json"
    out.parent.mkdir(exist_ok=True)
    fake = launcher if launcher is not None else _Launches()
    code = main(["--out", str(out), *args], launcher=fake, find=find, clock=clock)
    return code, out


def test_a_full_pass_is_39_cases_and_78_launches_in_order(tmp_path: Path) -> None:
    def behave(case_id: str, variant: str) -> tuple[int, dict[str, object], float]:
        if variant == "leaky" and case_id.startswith("weak-proxy"):
            return 0, _verdict(Status.APPROVED, value=0.9), 2.0  # a missed leak
        return _default_behaviour(case_id, variant)

    fake = _Launches(behave)
    code, out = _run(tmp_path, launcher=fake)
    cases = all_cases()
    assert len(cases) == 39
    assert code == 0  # exit 0 although leaks were missed
    expected = [(c.case_id, v) for c in cases for v in ("leaky", "clean")]
    assert [(a.parent.name, a.name) for a in fake.apps] == expected
    assert all(not root.exists() for root in fake.roots)
    results = json.loads(out.read_text(encoding="utf-8"))
    assert results["complete"] is True and results["only"] is None
    assert len(results["mechanisms"]) == 13 and len(results["cases"]) == 39
    assert results["clean"]["scripts"] == 39
    weak = next(m for m in results["mechanisms"] if m["key"] == "weak-proxy")
    assert (weak["rejected"], weak["scripts"]) == (0, 3)


def test_only_runs_six_apps_and_writes_a_partial_file(tmp_path: Path) -> None:
    fake = _Launches()
    clock = _Clock(100.0, 1280.44)
    code, out = _run(tmp_path, "--only", "scaler-all-rows", launcher=fake, clock=clock)
    assert code == 0
    assert len(fake.apps) == 6
    assert {a.parent.name for a in fake.apps} == {
        "scaler-all-rows-lr",
        "scaler-all-rows-rf",
        "scaler-all-rows-hgb",
    }
    assert set(fake.limits) == {HARD_LIMIT_SECONDS}
    assert all(argv[1:2] == ["run"] and len(argv) == 3 for argv in fake.argv)
    results = json.loads(out.read_text(encoding="utf-8"))
    assert results["complete"] is False and results["only"] == "scaler-all-rows"
    assert [m["key"] for m in results["mechanisms"]] == ["scaler-all-rows"]
    environment = results["environment"]
    assert environment["run_seconds"] == 1180.4
    assert set(environment) == {
        "hexperiment",
        "python",
        "numpy",
        "pandas",
        "scikit-learn",
        "os",
        "cpu_count",
        "run_seconds",
    }
    assert results["created_at"].endswith("Z")


def test_one_line_per_case_and_a_summary_on_standard_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = _run(tmp_path, "--only", "no-test-set")
    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    case_lines = [line for line in lines if line.startswith("[")]
    assert len(case_lines) == 3
    assert case_lines[0].startswith("[1/3] no-test-set-lr")
    assert "leaky REJECTED 0.8000 The Peeking Spirit" in case_lines[0]
    assert "clean APPROVED 0.7000" in case_lines[0]
    assert "3.0 s" in case_lines[0]
    assert "Summary" in lines
    assert any(
        line.startswith("no-test-set") and "rejected 3 of 3" in line for line in lines
    )
    assert any("false alarms 0 in 3" in line for line in lines)
    assert lines[-1] == f"Results written to {out}"


def test_equal_scores_print_no_effect(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def behave(case_id: str, variant: str) -> tuple[int, dict[str, object], float]:
        if variant == "leaky":
            return 1, _rejected(0.7000), 1.0
        return 0, _verdict(Status.APPROVED, value=0.70004), 1.0

    code, _ = _run(tmp_path, "--only", "no-test-set", launcher=_Launches(behave))
    output = capsys.readouterr().out
    assert code == 0
    assert output.count("NO EFFECT") == 3 + 1  # three case lines and the list title
    assert "no-test-set-lr" in output.split("NO EFFECT (leaky")[1]


def test_a_missing_score_is_reported_as_effect_not_checked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def behave(
        case_id: str, variant: str
    ) -> tuple[int, dict[str, object] | None, float]:
        if variant == "leaky" and case_id.endswith("-rf"):
            return 2, None, 1.0  # an ERROR run has no score
        return _default_behaviour(case_id, variant)

    code, _ = _run(tmp_path, "--only", "no-test-set", launcher=_Launches(behave))
    output = capsys.readouterr().out
    assert code == 0
    tail = output.split("Effect not checked (a score is missing):")[1]
    assert "no-test-set-rf" in tail.splitlines()[1]
    assert "NO EFFECT (leaky and clean scores are equal): none" in output


def test_runs_over_30_seconds_are_listed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def behave(case_id: str, variant: str) -> tuple[int, dict[str, object], float]:
        code, data, seconds = _default_behaviour(case_id, variant)
        slow = (variant, case_id) == ("clean", "no-test-set-hgb")
        return code, data, 45.26 if slow else seconds

    _run(tmp_path, "--only", "no-test-set", launcher=_Launches(behave))
    output = capsys.readouterr().out
    listed = output.split("Runs over 30 seconds:")[1]
    assert listed.splitlines()[1] == "  no-test-set-hgb clean: 45.3 s"
    assert "no-test-set-lr" not in listed


def test_an_error_run_is_recorded_and_the_loop_goes_on(tmp_path: Path) -> None:
    def behave(
        case_id: str, variant: str
    ) -> tuple[int, dict[str, object] | None, float]:
        if case_id == "no-test-set-lr" and variant == "leaky":
            return 2, None, 1.0
        return _default_behaviour(case_id, variant)

    fake = _Launches(behave)
    code, out = _run(tmp_path, "--only", "no-test-set", launcher=fake)
    assert code == 0 and len(fake.apps) == 6
    results = json.loads(out.read_text(encoding="utf-8"))
    mechanism = results["mechanisms"][0]
    assert mechanism["statuses"]["ERROR"] == 1 and mechanism["rejected"] == 2
    first = results["cases"][0]["leaky"]
    assert first["status"] == "ERROR" and first["message"] == "boom"


def test_ctrl_c_keeps_an_existing_file_and_deletes_the_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = _Launches()

    def interrupting(argv: Sequence[str], cwd: Path, limit: float) -> Launched:
        if len(fake.apps) == 2:
            raise KeyboardInterrupt
        return fake(argv, cwd, limit)

    out = tmp_path / "out" / "results.json"
    out.parent.mkdir()
    out.write_text("old", encoding="utf-8")
    code = main(
        ["--out", str(out), "--only", "no-test-set"],
        launcher=interrupting,
        find=lambda: "hexperiment",
    )
    assert code == 130
    assert out.read_text(encoding="utf-8") == "old"
    assert list(out.parent.iterdir()) == [out]
    assert all(not root.exists() for root in fake.roots)
    assert "interrupted; no results written" in capsys.readouterr().err


def test_an_unexpected_exception_exits_2_and_keeps_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = _Launches()

    def failing(argv: Sequence[str], cwd: Path, limit: float) -> Launched:
        if len(fake.apps) == 1:
            raise RuntimeError("the launcher broke")
        return fake(argv, cwd, limit)

    out = tmp_path / "out" / "results.json"
    out.parent.mkdir()
    out.write_text("old", encoding="utf-8")
    code = main(
        ["--out", str(out), "--only", "no-test-set"],
        launcher=failing,
        find=lambda: "hexperiment",
    )
    assert code == 2
    assert out.read_text(encoding="utf-8") == "old"
    assert list(out.parent.iterdir()) == [out]
    assert all(not root.exists() for root in fake.roots)
    assert "the launcher broke" in capsys.readouterr().err


def test_an_unknown_key_exits_2_before_any_launch_and_lists_the_keys(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = _Launches()
    code, out = _run(tmp_path, "--only", "no-such-key", launcher=fake)
    message = capsys.readouterr().err
    assert code == 2 and not fake.apps and not out.exists()
    assert "no-such-key" in message
    assert all(m.key in message for m in MECHANISMS)


def test_a_missing_command_exits_2_before_any_launch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = _Launches()
    code, out = _run(tmp_path, launcher=fake, find=lambda: None)
    message = capsys.readouterr().err
    assert code == 2 and not fake.apps and not out.exists()
    assert "`hexperiment` command was not found" in message
    assert 'pip install -e "hexperiment[dev]"' in message


def test_a_missing_out_exits_2_before_any_launch() -> None:
    fake = _Launches()
    code = main([], launcher=fake, find=lambda: "hexperiment")
    assert code == 2 and not fake.apps


def test_an_out_in_a_missing_folder_exits_2_before_any_launch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = _Launches()
    out = tmp_path / "no-such-folder" / "results.json"
    code = main(["--out", str(out)], launcher=fake, find=lambda: "hexperiment")
    assert code == 2 and not fake.apps and not out.parent.exists()
    assert "no-such-folder" in capsys.readouterr().err


def test_a_successful_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    code, out = _run(tmp_path, "--only", "no-test-set")
    assert code == 0
    assert list(out.parent.iterdir()) == [out]
    assert out.read_bytes().endswith(b"}\n") and b"\r" not in out.read_bytes()


def test_print_is_not_called_anywhere_in_the_benchmark() -> None:
    files = [*(BENCHMARK / "hexbench").glob("*.py"), *BENCHMARK.glob("*.py")]
    assert len(files) >= 11
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ]
        assert not calls, f"print is called in {path.name}"


def test_the_entry_point_imports_and_rejects_an_unknown_key(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    done = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK / "run_benchmark.py"),
            "--only",
            "no-such-key",
            "--out",
            str(out),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
        check=False,
    )
    assert done.returncode == 2, done.stderr
    assert not out.exists()
    for mechanism in MECHANISMS:
        assert mechanism.key in done.stderr + done.stdout


# --- update_readme.py ---------------------------------------------------------------


def _run_record(status: str, score: float | None) -> RunRecord:
    ghosts = (PEEKING,) if status == "REJECTED" else ()
    return RunRecord(status, score, ghosts, (), 1.0, None)


def _results(only: str | None = None) -> dict[str, object]:
    records = [
        CaseRecord(
            f"no-test-set-{k}",
            "no-test-set",
            k,
            1010,
            _run_record("REJECTED", 0.8),
            _run_record("APPROVED", 0.7),
        )
        for k in ("lr", "rf", "hgb")
    ]
    return build_results(
        records,
        mechanisms=MECHANISMS,
        not_measured=NOT_MEASURED,
        environment={
            "hexperiment": "0.1.0",
            "python": "3.13.0",
            "scikit-learn": "1.7.0",
        },
        created_at="2026-01-02T03:04:05Z",
        only=only,
    )


def _setup(
    tmp_path: Path, results: dict[str, object], readme: str
) -> tuple[Path, Path]:
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")
    readme_path = tmp_path / "README.md"
    readme_path.write_bytes(readme.encode("utf-8"))
    return results_path, readme_path


_README = f"intro\n\n{BEGIN}\nold table\n{END}\n\noutro\n"


def test_update_readme_replaces_the_block_and_prints_only_the_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results_path, readme = _setup(tmp_path, _results(), _README)
    assert update_readme.main([str(results_path)], readme=readme) == 0
    text = readme.read_text(encoding="utf-8")
    assert "old table" not in text and "no-test-set" not in text.split(BEGIN)[0]
    assert text.startswith("intro\n\n" + BEGIN) and text.endswith(END + "\n\noutro\n")
    assert "No test set" in text and "clean twins: 0 false alarms in 3 scripts" in text
    assert capsys.readouterr().out == f"{readme}\n"


def test_update_readme_is_idempotent(tmp_path: Path) -> None:
    results_path, readme = _setup(tmp_path, _results(), _README)
    update_readme.main([str(results_path)], readme=readme)
    first = readme.read_bytes()
    update_readme.main([str(results_path)], readme=readme)
    assert readme.read_bytes() == first


def test_update_readme_refuses_an_incomplete_results_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results_path, readme = _setup(tmp_path, _results(only="no-test-set"), _README)
    assert update_readme.main([str(results_path)], readme=readme) == 2
    assert readme.read_text(encoding="utf-8") == _README
    assert "--only" in capsys.readouterr().err


@pytest.mark.parametrize(
    "text",
    [
        "no markers here\n",
        f"{BEGIN}\nonly a begin\n",
        f"{BEGIN}\na\n{END}\n{BEGIN}\nb\n{END}\n",
    ],
)
def test_update_readme_refuses_a_missing_or_repeated_marker(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], text: str
) -> None:
    results_path, readme = _setup(tmp_path, _results(), text)
    assert update_readme.main([str(results_path)], readme=readme) == 2
    assert readme.read_text(encoding="utf-8") == text
    assert "exactly once" in capsys.readouterr().err


def test_update_readme_refuses_a_missing_results_file(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(_README, encoding="utf-8")
    code = update_readme.main([str(tmp_path / "absent.json")], readme=readme)
    assert code == 2
    assert readme.read_text(encoding="utf-8") == _README
