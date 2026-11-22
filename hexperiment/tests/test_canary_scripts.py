"""End-to-end scenarios with real training scripts (Requirements 4.7, 16.2 to 16.5).

Each scenario is one test class. Its class-scoped fixture builds an app in a
temporary folder and runs ``cli.main`` once, in this process, which starts
``train.py`` as a child process; the assertions of the class share that result.
"""

import contextlib
import io
import json
import runpy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from sklearn.metrics import roc_auc_score

import hexperiment
from app_builder import THRESHOLD, build_app, place_stale_verdict
from hexperiment import load_history
from hexperiment.cli import main
from hexperiment.config import ChecksConfig
from hexperiment.models import Status, Verdict


@dataclass(frozen=True)
class ScenarioRun:
    app: Path
    code: int
    stdout: str
    stderr: str
    seconds: float

    @property
    def verdict(self) -> dict[str, Any] | None:
        """The app-level ``verdict.json``, or ``None`` when there is none."""
        path = self.app / "verdict.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def run_folders(self) -> list[Path]:
        runs = self.app / "runs"
        return sorted(runs.iterdir()) if runs.exists() else []

    def hex_dir(self, mode: str) -> Path:
        return self.app / ".hex" / mode

    @property
    def scrambled_rows(self) -> int:
        """Rows that the canary execution's ``load`` reports as scrambled."""
        marker = self.hex_dir("canary") / "loaded.json"
        return json.loads(marker.read_text(encoding="utf-8"))["n_scrambled"]


def run_scenario(
    tmp_path_factory: pytest.TempPathFactory,
    scenario: str,
    *,
    stale_verdict: bool = False,
    **options: Any,
) -> ScenarioRun:
    """Build the app of ``scenario`` and run ``cli.main`` on it once."""
    app = build_app(tmp_path_factory.mktemp(scenario), scenario, **options)
    if stale_verdict:
        place_stale_verdict(app)
    out, err = io.StringIO(), io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(["run", str(app)])
    return ScenarioRun(
        app=app,
        code=code,
        stdout=out.getvalue(),
        stderr=err.getvalue(),
        seconds=time.perf_counter() - started,
    )


def findings_of(verdict: dict[str, Any], check: str) -> list[dict[str, Any]]:
    return [f for f in verdict["findings"] if f["check"] == check]


def outcome_of(verdict: dict[str, Any], check: str) -> str:
    (entry,) = (c for c in verdict["checks"] if c["name"] == check)
    return entry["outcome"]


def line_of(app: Path, text: str) -> int:
    """The 1-based line of ``train.py`` that holds ``text`` (exactly one line)."""
    lines = (app / "train.py").read_text(encoding="utf-8").splitlines()
    (number,) = (i for i, line in enumerate(lines, 1) if text in line)
    return number


def errors_of(verdict: dict[str, Any]) -> list[dict[str, Any]]:
    return [f for f in verdict["findings"] if f["severity"] == "error"]


class LeakScenario:
    """Assertions shared by the scenarios whose canary must find a leak.

    A subclass defines the class-scoped fixture ``run`` and ``suspect``, the text of
    the line the split-order hint must point at, or ``None`` when the leak is in
    another file and the hint must stay silent (Requirement 10.5).
    """

    suspect: str | None = None

    def test_the_run_is_rejected_with_exit_code_1(self, run):
        assert run.code == 1
        assert run.stderr == ""
        assert run.verdict["status"] == "REJECTED"
        assert run.verdict["mode"] == "full"

    def test_the_metric_passes_so_only_the_leak_rejects(self, run):
        metric = run.verdict["metric"]
        assert metric["passed"] is True
        assert metric["value"] >= metric["threshold"]

    def test_the_peeking_spirit_is_the_only_error(self, run):
        (error,) = errors_of(run.verdict)
        assert error["ghost"] == "The Peeking Spirit"
        assert error["check"] == "canary"
        assert error["location"] == "train.py"
        assert error["evidence"]["rows_changed"] > 0
        assert error["evidence"]["train_rows"] > 100

    def test_the_canary_failed_and_training_is_deterministic(self, run):
        assert outcome_of(run.verdict, "canary") == "failed"
        assert outcome_of(run.verdict, "determinism") == "passed"
        assert findings_of(run.verdict, "determinism") == []

    def test_the_control_execution_ran_once_to_rule_out_noise(self, run):
        assert run.hex_dir("normal").is_dir()
        assert run.hex_dir("canary").is_dir()
        assert run.hex_dir("control").is_dir()

    def test_the_split_order_hint_points_at_the_leaking_line(self, run):
        hints = findings_of(run.verdict, "split_order")
        (error,) = errors_of(run.verdict)
        if self.suspect is None:
            assert hints == []
            assert "Suspect lines" not in error["fix"]
            return
        location = f"train.py:{line_of(run.app, self.suspect)}"
        assert [h["location"] for h in hints] == [location]
        assert {h["severity"] for h in hints} == {"warning"}
        assert f"Suspect lines: {location}." in error["fix"]

    def test_the_summary_names_the_ghost(self, run):
        first, *rest = run.stdout.splitlines()
        assert first.startswith("REJECTED (full)")
        assert any("The Peeking Spirit" in line for line in rest)

    def test_the_run_is_kept_as_evidence(self, run):
        (folder,) = run.run_folders
        assert folder.name == run.verdict["run_id"]
        assert (folder / "verdict.json").is_file()
        assert (folder / "model_card.md").is_file()


class TestMeanImputeGlobal(LeakScenario):
    """Scenario 1: the mean over all rows fills missing values before the split."""

    suspect = ".mean()"

    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, tmp_path_factory) -> ScenarioRun:
        return run_scenario(tmp_path_factory, "mean_impute_global")


class TestScalerGlobal(LeakScenario):
    """Scenario 2: a scaler is fitted on all rows before the split."""

    suspect = "fit_transform("

    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, tmp_path_factory) -> ScenarioRun:
        return run_scenario(tmp_path_factory, "scaler_global")


class TestHelperLeak(LeakScenario):
    """Scenario 3: the leak is in ``helpers.py``, which the hint cannot see."""

    suspect = None

    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, tmp_path_factory) -> ScenarioRun:
        return run_scenario(tmp_path_factory, "helper_leak")


class TestTargetEncodingGlobal(LeakScenario):
    """Scenario 11: a per-category target mean over all rows, in ``train.py``."""

    suspect = '.transform("mean")'

    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, tmp_path_factory) -> ScenarioRun:
        return run_scenario(tmp_path_factory, "target_encoding_global")


class TestTrainOnlyFit:
    """Scenario 4: everything is fitted on training rows after the split."""

    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, tmp_path_factory) -> ScenarioRun:
        return run_scenario(tmp_path_factory, "train_only_fit")

    def test_the_run_is_approved_with_exit_code_0(self, run):
        assert run.code == 0
        assert run.stderr == ""
        assert run.verdict["status"] == "APPROVED"
        assert run.verdict["mode"] == "full"
        assert run.stdout.splitlines()[0].startswith("APPROVED (full)")

    def test_the_canary_passed_and_nothing_was_flagged(self, run):
        assert outcome_of(run.verdict, "canary") == "passed"
        assert outcome_of(run.verdict, "determinism") == "passed"
        assert outcome_of(run.verdict, "split_order") == "passed"
        assert run.verdict["findings"] == []

    def test_no_control_execution_was_needed(self, run):
        assert run.hex_dir("canary").is_dir()
        assert not run.hex_dir("control").exists()

    def test_the_canary_execution_scrambled_test_rows(self, run):
        assert run.scrambled_rows > 0

    def test_the_run_is_kept_as_evidence(self, run):
        (folder,) = run.run_folders
        assert (folder / "verdict.json").read_bytes() == (
            run.app / "verdict.json"
        ).read_bytes()
        assert (folder / "model_card.md").is_file()


class TestUnrelatedFit:
    """Scenario 5: a scaler fitted on a separate synthetic dataset is not a leak."""

    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, tmp_path_factory) -> ScenarioRun:
        return run_scenario(tmp_path_factory, "unrelated_fit")

    def test_the_run_is_approved_with_exit_code_0(self, run):
        assert run.code == 0
        assert run.verdict["status"] == "APPROVED"
        assert outcome_of(run.verdict, "canary") == "passed"
        assert errors_of(run.verdict) == []

    def test_training_is_deterministic(self, run):
        assert outcome_of(run.verdict, "determinism") == "passed"
        assert findings_of(run.verdict, "determinism") == []

    def test_only_the_split_order_hint_warns(self, run):
        (finding,) = run.verdict["findings"]
        assert finding["check"] == "split_order"
        assert finding["severity"] == "warning"
        assert finding["location"] == f"train.py:{line_of(run.app, '.fit(reference)')}"

    def test_no_control_execution_was_needed(self, run):
        assert not run.hex_dir("control").exists()

    def test_the_canary_execution_scrambled_test_rows(self, run):
        assert run.scrambled_rows > 0


class TestFitOnAllRows(LeakScenario):
    """Scenario 12: the model sees the test rows, the training features do not move."""

    suspect = None

    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, tmp_path_factory) -> ScenarioRun:
        return run_scenario(tmp_path_factory, "fit_on_all_rows")

    def test_it_is_detected_through_the_training_scores_alone(self, run):
        (error,) = errors_of(run.verdict)
        assert error["evidence"]["feature_rows_changed"] == 0
        assert error["evidence"]["rows_changed"] > 0
        assert error["evidence"]["largest_change"] > 1e-9


class TestNoTrainScores:
    """Scenario 13: without ``y_score_train`` the canary cannot clear the run."""

    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, tmp_path_factory) -> ScenarioRun:
        return run_scenario(tmp_path_factory, "no_train_scores")

    def test_the_run_is_unverified_with_exit_code_3(self, run):
        assert run.code == 3
        assert run.stderr == ""
        assert run.verdict["status"] == "UNVERIFIED"
        assert run.verdict["metric"]["passed"] is True
        assert run.stdout.splitlines()[0].startswith("UNVERIFIED (full)")

    def test_the_canary_is_inconclusive_and_nothing_is_an_error(self, run):
        assert outcome_of(run.verdict, "canary") == "inconclusive"
        assert outcome_of(run.verdict, "determinism") == "passed"
        assert errors_of(run.verdict) == []

    def test_the_warning_shows_how_to_pass_y_score_train(self, run):
        (warning,) = findings_of(run.verdict, "canary")
        assert warning["severity"] == "warning"
        assert warning["ghost"] is None
        assert "without training predictions" in warning["message"]
        assert "y_score_train=model.predict_proba(train[features])[:, 1]" in (
            warning["fix"]
        )

    def test_no_control_execution_was_needed(self, run):
        assert run.hex_dir("canary").is_dir()
        assert not run.hex_dir("control").exists()


class TestNoSeed:
    """Scenario 6: training without a seed is The Shapeshifter, not a leak."""

    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, tmp_path_factory) -> ScenarioRun:
        return run_scenario(tmp_path_factory, "no_seed")

    def test_the_run_is_unverified_with_exit_code_3(self, run):
        assert run.code == 3
        assert run.verdict["status"] == "UNVERIFIED"
        assert run.verdict["metric"]["passed"] is True
        assert run.stdout.splitlines()[0].startswith("UNVERIFIED (full)")

    def test_the_shapeshifter_is_reported_as_a_warning(self, run):
        (finding,) = findings_of(run.verdict, "determinism")
        assert finding["ghost"] == "The Shapeshifter"
        assert finding["severity"] == "warning"
        assert "random_state" in finding["fix"]
        assert outcome_of(run.verdict, "determinism") == "failed"

    def test_the_peeking_spirit_is_not_reported(self, run):
        assert errors_of(run.verdict) == []
        assert findings_of(run.verdict, "canary") == []
        assert all(f["ghost"] != "The Peeking Spirit" for f in run.verdict["findings"])
        assert outcome_of(run.verdict, "canary") == "inconclusive"

    def test_the_control_execution_ran(self, run):
        assert run.hex_dir("control").is_dir()


class TestDirectRead:
    """Scenario 7: a script that reads the CSV itself cannot be scrambled."""

    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, tmp_path_factory) -> ScenarioRun:
        return run_scenario(tmp_path_factory, "direct_read")

    def test_the_run_is_unverified_with_exit_code_3(self, run):
        assert run.code == 3
        assert run.verdict["status"] == "UNVERIFIED"
        assert errors_of(run.verdict) == []

    def test_the_canary_was_not_applied(self, run):
        assert outcome_of(run.verdict, "canary") == "not applied"
        assert outcome_of(run.verdict, "determinism") == "skipped"
        (warning,) = findings_of(run.verdict, "canary")
        assert warning["severity"] == "warning"
        assert "hexperiment.load" in warning["message"]

    def test_the_data_reader_is_reported_at_its_line(self, run):
        (reader,) = findings_of(run.verdict, "split_order")
        assert reader["severity"] == "warning"
        assert reader["location"] == f"train.py:{line_of(run.app, 'read_csv(')}"
        assert "without hexperiment.load()" in reader["message"]
        assert reader["evidence"] == {"call": "read_csv"}

    def test_the_load_marker_was_never_written(self, run):
        assert not (run.hex_dir("canary") / "loaded.json").exists()
        assert not run.hex_dir("control").exists()


class TestLoadBypass:
    """Scenario 7b: ``load()`` is called and discarded, the CSV is read directly."""

    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, tmp_path_factory) -> ScenarioRun:
        return run_scenario(tmp_path_factory, "load_bypass")

    def test_the_run_is_unverified_and_never_approved(self, run):
        assert run.code == 3
        assert run.verdict["status"] == "UNVERIFIED"
        assert errors_of(run.verdict) == []

    def test_the_load_marker_was_written_so_only_the_test_columns_give_it_away(
        self, run
    ):
        assert run.scrambled_rows > 0

    def test_the_canary_was_not_applied_and_determinism_was_skipped(self, run):
        assert outcome_of(run.verdict, "canary") == "not applied"
        assert outcome_of(run.verdict, "determinism") == "skipped"
        (warning,) = findings_of(run.verdict, "canary")
        assert warning["severity"] == "warning"
        assert warning["ghost"] is None
        assert "scramble did not reach the audited frames" in warning["message"]

    def test_the_data_reader_hint_is_not_raised_for_the_csv_module(self, run):
        assert findings_of(run.verdict, "split_order") == []

    def test_no_control_execution_ran(self, run):
        assert not run.hex_dir("control").exists()


class TestPartialLoadBypass:
    """Scenario 7c: the target comes from ``load()``, the features from a CSV read."""

    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, tmp_path_factory) -> ScenarioRun:
        return run_scenario(tmp_path_factory, "partial_load_bypass")

    def test_the_run_is_unverified_and_never_approved(self, run):
        assert run.code == 3
        assert run.verdict["status"] == "UNVERIFIED"
        assert errors_of(run.verdict) == []

    def test_the_canary_was_not_applied_and_determinism_was_skipped(self, run):
        assert outcome_of(run.verdict, "canary") == "not applied"
        assert outcome_of(run.verdict, "determinism") == "skipped"

    def test_the_warning_names_the_features_the_scramble_missed_but_not_the_target(
        self, run
    ):
        (warning,) = findings_of(run.verdict, "canary")
        assert warning["severity"] == "warning"
        assert warning["ghost"] is None
        assert "scramble did not reach the audited frames" in warning["message"]
        columns = warning["evidence"]["columns"]
        assert "target" not in columns
        assert {"x1", "x2"} <= set(columns)
        assert "x1, x2" in warning["message"]

    def test_no_control_execution_ran(self, run):
        assert not run.hex_dir("control").exists()


class TestConstantColumn:
    """Scenario 4b: a constant feature cannot be scrambled, and is not a miss."""

    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, tmp_path_factory) -> ScenarioRun:
        return run_scenario(tmp_path_factory, "constant_column")

    def test_the_run_is_approved_with_exit_code_0(self, run):
        assert run.code == 0
        assert run.verdict["status"] == "APPROVED"
        assert run.verdict["findings"] == []

    def test_the_canary_was_applied_and_passed(self, run):
        assert outcome_of(run.verdict, "canary") == "passed"
        assert outcome_of(run.verdict, "determinism") == "passed"
        assert run.scrambled_rows > 0


class TestRepeatRun:
    """Requirement 13.5: the same full run twice gives the same verdict content.

    The scenario has a finding (the split-order warning), so the comparison of the
    findings is not vacuous.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def runs(cls, tmp_path_factory) -> tuple[ScenarioRun, list[Verdict]]:
        first = run_scenario(tmp_path_factory, "unrelated_fit")
        second_out, second_err = io.StringIO(), io.StringIO()
        with (
            contextlib.redirect_stdout(second_out),
            contextlib.redirect_stderr(second_err),
        ):
            code = main(["run", str(first.app)])
        assert (first.code, code) == (0, 0)
        return first, load_history(first.app)

    def test_there_are_two_runs_with_different_ids(self, runs):
        _, history = runs
        assert len(history) == 2
        assert history[0].run_id != history[1].run_id

    def test_status_metric_and_findings_are_identical(self, runs):
        _, (first, second) = runs
        assert first.status == second.status == Status.APPROVED
        assert first.metric == second.metric
        assert first.findings != ()
        assert first.findings == second.findings

    def test_the_second_run_counts_as_another_test_set_use(self, runs):
        _, (first, second) = runs
        assert (first.test_set_uses, second.test_set_uses) == (1, 2)

    def test_the_app_verdict_is_the_latest_run(self, runs):
        first_run, (_, second) = runs
        written = json.loads((first_run.app / "verdict.json").read_text("utf-8"))
        assert written["run_id"] == second.run_id


class TestInputError:
    """Scenario 8: a ``y_score`` of the wrong length is a recorded, rejected run."""

    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, tmp_path_factory) -> ScenarioRun:
        return run_scenario(tmp_path_factory, "input_error")

    def test_the_exit_code_is_1(self, run):
        assert run.code == 1
        assert run.stderr == ""

    def test_the_verdict_is_rejected_with_one_input_finding(self, run):
        verdict = run.verdict
        assert verdict["status"] == "REJECTED"
        assert verdict["mode"] == "full"
        (finding,) = verdict["findings"]
        assert finding["check"] == "input"
        assert finding["severity"] == "error"
        assert "y_score" in finding["message"]
        assert verdict["metric"]["value"] is None
        assert verdict["metric"]["passed"] is False

    def test_only_the_normal_execution_ran(self, run):
        assert (run.hex_dir("normal") / "result.json").is_file()
        assert not run.hex_dir("canary").exists()
        assert not run.hex_dir("control").exists()

    def test_the_run_is_kept_as_evidence(self, run):
        (folder,) = run.run_folders
        assert folder.name == run.verdict["run_id"]
        assert (folder / "verdict.json").is_file()
        assert (folder / "model_card.md").is_file()

    def test_the_run_does_not_count_as_a_test_set_use(self, run):
        assert run.verdict["test_set_uses"] == 0

    def test_the_summary_names_the_input_error(self, run):
        first, second = run.stdout.splitlines()
        assert first.startswith("REJECTED (full)")
        assert second.lstrip().startswith("error")
        assert "audit()" in second


class TestTimeout:
    """Scenario 9: a script that outlasts ``timeout_seconds`` fails the run."""

    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, tmp_path_factory) -> ScenarioRun:
        return run_scenario(
            tmp_path_factory, "timeout", stale_verdict=True, timeout_seconds=1
        )

    def test_the_exit_code_is_2(self, run):
        assert run.code == 2
        assert run.stdout == ""

    def test_the_error_names_the_timeout(self, run):
        assert run.stderr.startswith("hexperiment: ")
        assert "longer than 1 seconds" in run.stderr
        assert "timeout_seconds" in run.stderr

    def test_no_run_folder_is_created(self, run):
        assert run.run_folders == []

    def test_the_stale_verdict_no_longer_exists(self, run):
        assert not (run.app / "verdict.json").exists()

    def test_the_script_was_stopped_promptly(self, run):
        assert run.seconds < 30


class TestNoAudit:
    """Scenario 10: a script that never calls ``hexperiment.audit`` fails the run."""

    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, tmp_path_factory) -> ScenarioRun:
        return run_scenario(tmp_path_factory, "no_audit", stale_verdict=True)

    def test_the_exit_code_is_2(self, run):
        assert run.code == 2
        assert run.stdout == ""

    def test_the_error_says_audit_was_not_called(self, run):
        assert run.stderr.startswith("hexperiment: ")
        assert "hexperiment.audit" in run.stderr

    def test_no_run_folder_is_created(self, run):
        assert run.run_folders == []

    def test_the_stale_verdict_no_longer_exists(self, run):
        assert not (run.app / "verdict.json").exists()


def test_the_clean_pipeline_scores_clear_of_the_threshold_and_the_too_good_cutoff(
    tmp_path, monkeypatch
):
    """Scenario 4 must pass the metric gate without raising "too good to be true".

    The template runs in this process with ``audit`` replaced by a capture, so no
    subprocess is started.
    """
    for name in ("HEX_MODE", "HEX_RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    captured: dict[str, Any] = {}

    def capture(train, test, y_score, features, script, y_score_train=None):
        captured["auc"] = roc_auc_score(test["target"], y_score)

    monkeypatch.setattr(hexperiment, "audit", capture)
    app = build_app(tmp_path, "train_only_fit")
    runpy.run_path(str(app / "train.py"), run_name="__main__")
    cutoff = min(THRESHOLD + ChecksConfig().too_good_margin, 0.99)
    assert captured["auc"] >= THRESHOLD + 0.02
    assert captured["auc"] <= cutoff - 0.02
