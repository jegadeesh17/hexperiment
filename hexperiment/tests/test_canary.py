import json
import math
import subprocess
import sys
import textwrap
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from hexperiment import HexInputError
from hexperiment.canary import (
    SCORE_TOLERANCE,
    CanaryOutcome,
    ExecutionFailed,
    ExecutionTimedOut,
    LoadMarker,
    Snapshot,
    build_snapshot,
    clear_mode_dir,
    compare_snapshots,
    decide_canary,
    hex_dir,
    hint_locations,
    read_load_marker,
    read_result,
    read_snapshot,
    run_executions,
    run_train_script,
    skipped_outcome,
    write_load_marker,
    write_result,
    write_snapshot,
)
from hexperiment.config import WAIVABLE_CHECKS, ExecMode, HexConfig, parse_config
from hexperiment.models import (
    CheckResult,
    DataAuditResult,
    DataSummary,
    Finding,
    MetricResult,
    Outcome,
    RunMode,
    Severity,
)

FEATURES = ["age", "plan"]


def make_frame(ids=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    n = len(ids)
    return pd.DataFrame(
        {
            "age": [20.0 + i for i in range(n)],
            "plan": ["basic", "pro", None, "basic", "pro"][:n],
            "target": [0, 1, 0, 1, 1][:n],
        },
        index=list(ids),
    )


def make_snapshot(
    train_ids=(0, 1, 2, 3),
    hashes=None,
    scores=(0.1, 0.2, 0.3, 0.4),
    test_ids=(4, 5),
    mode=ExecMode.NORMAL,
) -> Snapshot:
    ids = np.asarray(train_ids, dtype=np.int64)
    if hashes is None:
        hashes = [1000 + i for i in train_ids]
    return Snapshot(
        mode=mode,
        run_id="20251101-093012-a1b2c3",
        features=tuple(FEATURES),
        train_ids=ids,
        test_ids=np.asarray(test_ids, dtype=np.int64),
        train_row_hash=np.asarray(hashes, dtype=np.uint64),
        y_score_train=None if scores is None else np.asarray(scores, dtype=float),
    )


def with_scores(snapshot: Snapshot, scores) -> Snapshot:
    return make_snapshot(
        train_ids=snapshot.train_ids,
        hashes=snapshot.train_row_hash,
        scores=scores,
        test_ids=snapshot.test_ids,
    )


def with_hashes(snapshot: Snapshot, hashes) -> Snapshot:
    return make_snapshot(
        train_ids=snapshot.train_ids,
        hashes=hashes,
        scores=snapshot.y_score_train,
        test_ids=snapshot.test_ids,
    )


def clean_marker(scrambled=(4, 5)) -> LoadMarker:
    return LoadMarker(
        mode=ExecMode.CANARY,
        run_id="20251101-093012-a1b2c3",
        scrambled_ids=np.asarray(scrambled, dtype=np.int64),
    )


class ControlRunner:
    """A ``run_control`` stand-in that counts its calls."""

    def __init__(self, snapshot: Snapshot) -> None:
        self.snapshot = snapshot
        self.calls = 0

    def __call__(self) -> Snapshot:
        self.calls += 1
        return self.snapshot


def changed_row(snapshot: Snapshot) -> Snapshot:
    hashes = snapshot.train_row_hash.copy()
    hashes[1] += np.uint64(1)
    return with_hashes(snapshot, hashes)


# --- build_snapshot ---------------------------------------------------------------


def test_build_snapshot_sorts_rows_by_id_and_keeps_scores_aligned():
    train = make_frame(ids=(3, 0, 2))
    scores = np.array([0.3, 0.0, 0.2])
    snapshot = build_snapshot(
        ExecMode.NORMAL, "run", train, make_frame(ids=(9, 5)), FEATURES, scores
    )
    assert snapshot.train_ids.tolist() == [0, 2, 3]
    assert snapshot.test_ids.tolist() == [5, 9]
    assert snapshot.y_score_train.tolist() == [0.0, 0.2, 0.3]
    expected = pd.util.hash_pandas_object(train[FEATURES], index=False)
    assert snapshot.train_row_hash.tolist() == expected.loc[[0, 2, 3]].tolist()
    assert snapshot.train_row_hash.dtype == np.uint64
    assert snapshot.features == tuple(FEATURES)
    assert snapshot.mode is ExecMode.NORMAL


def test_build_snapshot_hash_ignores_index_and_non_feature_columns():
    a = make_frame(ids=(0, 1, 2))
    b = make_frame(ids=(7, 8, 9))
    b["target"] = [1, 1, 1]
    snap_a = build_snapshot(ExecMode.NORMAL, None, a, a.iloc[:0], FEATURES, None)
    snap_b = build_snapshot(ExecMode.NORMAL, None, b, b.iloc[:0], FEATURES, None)
    assert snap_a.train_row_hash.tolist() == snap_b.train_row_hash.tolist()
    assert snap_a.y_score_train is None


def test_build_snapshot_rejects_wrong_score_length():
    train = make_frame()
    with pytest.raises(HexInputError, match="y_score_train"):
        build_snapshot(
            ExecMode.NORMAL, None, train, train.iloc[:0], FEATURES, np.zeros(3)
        )


def test_build_snapshot_rejects_non_integer_index():
    train = make_frame().set_index("plan")
    with pytest.raises(HexInputError, match="index of train"):
        build_snapshot(ExecMode.NORMAL, None, train, make_frame(), ["age"], None)


def test_build_snapshot_rejects_repeated_training_row_ids():
    # Oversampling before audit repeats row ids, so rows cannot be aligned.
    train = make_frame(ids=(0, 1, 1, 2, 2))
    with pytest.raises(HexInputError, match="2 row ids appear more than once") as info:
        build_snapshot(
            ExecMode.NORMAL, None, train, make_frame(ids=(8, 9)), FEATURES, None
        )
    assert "row id 1" in str(info.value)
    assert "oversample" in str(info.value)


def test_build_snapshot_accepts_repeated_ids_only_across_train_and_test():
    train = make_frame(ids=(0, 1, 2))
    snapshot = build_snapshot(
        ExecMode.NORMAL, None, train, make_frame(ids=(2, 3)), FEATURES, None
    )
    assert snapshot.train_ids.tolist() == [0, 1, 2]


@given(
    ids=st.lists(st.integers(0, 30), min_size=1, max_size=12, unique=True),
    data=st.data(),
)
def test_oversampled_training_rows_are_always_rejected(ids, data):
    repeats = data.draw(st.lists(st.sampled_from(ids), min_size=1, max_size=5))
    all_ids = ids + repeats
    train = pd.DataFrame(
        {"age": [float(i) for i in all_ids], "plan": ["a"] * len(all_ids)},
        index=all_ids,
    )
    with pytest.raises(HexInputError, match="more than once in train"):
        build_snapshot(ExecMode.NORMAL, None, train, train.iloc[:0], FEATURES, None)


def test_row_reordering_does_not_change_compare_snapshots():
    train = make_frame()
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    reversed_train = train.iloc[::-1]
    a = build_snapshot(ExecMode.NORMAL, None, train, train, FEATURES, scores)
    b = build_snapshot(
        ExecMode.CANARY, None, reversed_train, reversed_train, FEATURES, scores[::-1]
    )
    diff = compare_snapshots(a, b)
    assert diff.ids_equal
    assert not diff.differs
    assert diff.max_abs_change == 0.0


# --- compare_snapshots ------------------------------------------------------------


def test_compare_equal_snapshots():
    diff = compare_snapshots(make_snapshot(), make_snapshot())
    assert diff.ids_equal
    assert (diff.rows_changed, diff.feature_rows_changed, diff.score_rows_changed) == (
        0,
        0,
        0,
    )
    assert diff.max_abs_change == 0.0
    assert not diff.differs


def test_compare_one_changed_feature_row():
    a = make_snapshot()
    diff = compare_snapshots(a, changed_row(a))
    assert diff.differs
    assert diff.rows_changed == 1
    assert diff.feature_rows_changed == 1
    assert diff.score_rows_changed == 0


def test_score_change_of_exactly_the_tolerance_is_not_a_change():
    assert SCORE_TOLERANCE == 1e-9
    a = make_snapshot(scores=(0.0, 0.2, 0.3, 0.4))
    b = with_scores(a, (1e-9, 0.2, 0.3, 0.4))
    assert abs(b.y_score_train[0] - a.y_score_train[0]) == 1e-9
    diff = compare_snapshots(a, b)
    assert diff.score_rows_changed == 0
    assert not diff.differs
    assert diff.max_abs_change == 1e-9


def test_score_change_just_below_the_tolerance_is_not_a_change():
    a = make_snapshot(scores=(0.0, 0.2, 0.3, 0.4))
    b = with_scores(a, (np.nextafter(1e-9, 0.0), 0.2, 0.3, 0.4))
    assert not compare_snapshots(a, b).differs


def test_score_change_just_above_the_tolerance_is_a_change():
    a = make_snapshot(scores=(0.0, 0.2, 0.3, 0.4))
    b = with_scores(a, (np.nextafter(1e-9, 1.0), 0.2, 0.3, 0.4))
    diff = compare_snapshots(a, b)
    assert diff.differs
    assert diff.score_rows_changed == 1
    assert diff.rows_changed == 1
    assert diff.feature_rows_changed == 0


def test_rows_changed_counts_rows_changed_by_hash_or_score_once():
    a = make_snapshot()
    hashes = a.train_row_hash.copy()
    hashes[[0, 1]] += np.uint64(1)
    b = make_snapshot(hashes=hashes, scores=(0.15, 0.2, 0.3, 0.9))
    diff = compare_snapshots(a, b)
    assert diff.feature_rows_changed == 2
    assert diff.score_rows_changed == 2
    assert diff.rows_changed == 3
    assert diff.max_abs_change == pytest.approx(0.5)


def test_compare_different_id_sets():
    diff = compare_snapshots(make_snapshot(), make_snapshot(train_ids=(0, 1, 2, 4)))
    assert not diff.ids_equal
    assert diff.differs
    assert diff.max_abs_change is None


@pytest.mark.parametrize("missing", ["a", "b", "both"])
def test_compare_without_scores_uses_hashes_only(missing):
    with_score = make_snapshot()
    other_scores = with_scores(with_score, (0.9, 0.9, 0.9, 0.9))
    a = make_snapshot(scores=None) if missing in ("a", "both") else with_score
    b = make_snapshot(scores=None) if missing in ("b", "both") else other_scores
    diff = compare_snapshots(a, b)
    assert diff.max_abs_change is None
    assert diff.score_rows_changed == 0
    assert not diff.differs
    assert compare_snapshots(a, changed_row(b)).feature_rows_changed == 1


@pytest.mark.parametrize("nan_in", ["a", "b", "both"])
def test_nan_training_score_counts_as_a_change(nan_in):
    # A NaN prediction can never be shown to be unchanged, even against a NaN.
    first_a = np.nan if nan_in in ("a", "both") else 0.1
    first_b = np.nan if nan_in in ("b", "both") else 0.1
    a = make_snapshot(scores=(first_a, 0.2, 0.3, 0.4))
    diff = compare_snapshots(a, with_scores(a, (first_b, 0.2, 0.3, 0.4)))
    assert diff.differs
    assert (diff.rows_changed, diff.score_rows_changed) == (1, 1)
    assert diff.feature_rows_changed == 0
    assert math.isnan(diff.max_abs_change)


def test_compare_with_a_non_default_tolerance():
    a = make_snapshot(scores=(0.1, 0.2, 0.3, 0.4))
    b = with_scores(a, (0.1 + 1e-4, 0.2, 0.3, 0.4 + 1e-2))
    wide = compare_snapshots(a, b, tolerance=1e-3)
    assert (wide.rows_changed, wide.score_rows_changed) == (1, 1)
    assert wide.max_abs_change == pytest.approx(1e-2)
    assert compare_snapshots(a, b).score_rows_changed == 2
    assert not compare_snapshots(a, b, tolerance=0.5).differs
    # The tolerance applies to scores only; hashes must still match exactly.
    hashed = compare_snapshots(a, changed_row(a), tolerance=1.0)
    assert (hashed.rows_changed, hashed.feature_rows_changed) == (1, 1)


# --- decide_canary: one example per step ------------------------------------------


def assert_unwaivable(outcome: CanaryOutcome) -> None:
    # Requirement 10.7: no canary finding can be named by a waiver.
    assert all(f.check not in WAIVABLE_CHECKS for f in outcome.findings)


def test_step_1_marker_missing():
    normal = make_snapshot()
    control = ControlRunner(normal)
    outcome = decide_canary(normal, changed_row(normal), None, control, ())
    assert (outcome.outcome, outcome.determinism) == (
        Outcome.NOT_APPLIED,
        Outcome.SKIPPED,
    )
    (finding,) = outcome.findings
    assert finding.check == "canary"
    assert finding.severity is Severity.WARNING
    assert "hexperiment.load" in finding.message
    assert control.calls == 0
    assert_unwaivable(outcome)


def test_step_2_scrambled_ids_in_train():
    normal = make_snapshot()
    control = ControlRunner(normal)
    outcome = decide_canary(
        normal, changed_row(normal), clean_marker(scrambled=(2, 3, 4)), control, ()
    )
    assert (outcome.outcome, outcome.determinism) == (
        Outcome.NOT_APPLIED,
        Outcome.SKIPPED,
    )
    (finding,) = outcome.findings
    assert finding.severity is Severity.WARNING
    assert finding.evidence == {"scrambled_train_rows": 2}
    assert "hexperiment.split" in finding.fix
    assert control.calls == 0
    assert_unwaivable(outcome)


def test_step_3_id_sets_differ():
    normal = make_snapshot()
    canary = make_snapshot(train_ids=(0, 1, 2), scores=(0.1, 0.2, 0.3))
    control = ControlRunner(normal)
    outcome = decide_canary(normal, canary, clean_marker(), control, ())
    assert (outcome.outcome, outcome.determinism) == (Outcome.SKIPPED, Outcome.SKIPPED)
    (finding,) = outcome.findings
    assert finding.severity is Severity.WARNING
    assert "inconclusive" in finding.message
    assert "depends on data values" in finding.message
    assert control.calls == 0
    assert_unwaivable(outcome)


def test_step_4_no_difference():
    normal = make_snapshot()
    control = ControlRunner(changed_row(normal))
    outcome = decide_canary(normal, make_snapshot(), clean_marker(), control, ())
    assert outcome == CanaryOutcome(Outcome.PASSED, Outcome.PASSED, ())
    assert control.calls == 0


def test_step_5_difference_with_matching_control():
    normal = make_snapshot()
    canary = with_scores(changed_row(normal), (0.1, 0.25, 0.3, 0.4))
    control = ControlRunner(make_snapshot())
    outcome = decide_canary(normal, canary, clean_marker(), control, ())
    assert (outcome.outcome, outcome.determinism) == (Outcome.FAILED, Outcome.PASSED)
    (finding,) = outcome.findings
    assert finding.check == "canary"
    assert finding.ghost == "The Peeking Spirit"
    assert finding.severity is Severity.ERROR
    assert finding.location == "train.py"
    assert finding.evidence == {
        "train_rows": 4,
        "rows_changed": 1,
        "feature_rows_changed": 1,
        "largest_change": 0.05,
    }
    assert finding.message == "scrambling the test rows changed 1 of 4 training rows"
    assert finding.fix == (
        "test rows influenced training; fit preprocessing on training rows only, "
        "after the split."
    )
    assert control.calls == 1
    assert_unwaivable(outcome)


def test_step_6_difference_with_differing_control():
    normal = make_snapshot()
    canary = changed_row(normal)
    control_snapshot = with_scores(normal, (0.1, 0.2, 0.3, 0.7))
    control = ControlRunner(control_snapshot)
    outcome = decide_canary(normal, canary, clean_marker(), control, ("train.py:3",))
    assert (outcome.outcome, outcome.determinism) == (
        Outcome.INCONCLUSIVE,
        Outcome.FAILED,
    )
    (finding,) = outcome.findings
    assert finding.check == "determinism"
    assert finding.ghost == "The Shapeshifter"
    assert finding.severity is Severity.WARNING
    assert finding.evidence == {"rows_changed": 1, "largest_change": 0.3}
    assert "set random_state everywhere" in finding.fix
    assert control.calls == 1
    assert_unwaivable(outcome)


def test_step_6_control_with_other_train_ids_differs():
    normal = make_snapshot()
    control = ControlRunner(make_snapshot(train_ids=(0, 1, 2), scores=None))
    outcome = decide_canary(normal, changed_row(normal), clean_marker(), control, ())
    assert outcome.outcome is Outcome.INCONCLUSIVE
    assert outcome.determinism is Outcome.FAILED
    assert outcome.findings[0].evidence == {"rows_changed": 0, "largest_change": None}


def test_peeking_spirit_evidence_has_no_nan_when_a_score_is_nan():
    normal = make_snapshot()
    canary = with_scores(normal, (np.nan, 0.2, 0.3, 0.4))
    outcome = decide_canary(normal, canary, clean_marker(), ControlRunner(normal), ())
    assert outcome.outcome is Outcome.FAILED
    (finding,) = outcome.findings
    assert finding.ghost == "The Peeking Spirit"
    assert finding.evidence["largest_change"] is None
    json.dumps(finding.evidence, allow_nan=False)


def test_shapeshifter_evidence_has_no_nan_when_a_score_is_nan():
    normal = make_snapshot()
    control = ControlRunner(with_scores(normal, (np.nan, 0.2, 0.3, 0.4)))
    outcome = decide_canary(normal, changed_row(normal), clean_marker(), control, ())
    (finding,) = outcome.findings
    assert finding.ghost == "The Shapeshifter"
    assert finding.evidence == {"rows_changed": 1, "largest_change": None}
    json.dumps(finding.evidence, allow_nan=False)


def test_step_2_is_checked_before_step_3():
    normal = make_snapshot()
    canary = make_snapshot(train_ids=(0, 1, 4), scores=None)
    outcome = decide_canary(normal, canary, clean_marker(), ControlRunner(normal), ())
    assert outcome.outcome is Outcome.NOT_APPLIED


def test_differing_ids_never_pass_even_when_rows_match_by_position():
    normal = make_snapshot()
    canary = make_snapshot(train_ids=(0, 1, 2, 6), hashes=normal.train_row_hash)
    control = ControlRunner(normal)
    outcome = decide_canary(normal, canary, clean_marker(), control, ())
    assert outcome.outcome is Outcome.SKIPPED
    assert control.calls == 0


# --- decide_canary: malformed markers and other features --------------------------


@pytest.mark.parametrize(
    ("marker", "evidence", "problem"),
    [
        (
            replace(clean_marker(), scrambled_ids=None),
            {"marker_mode": "canary", "scrambled_rows": None},
            "records no scrambled test rows",
        ),
        (
            clean_marker(scrambled=()),
            {"marker_mode": "canary", "scrambled_rows": 0},
            "records no scrambled test rows",
        ),
        (
            replace(clean_marker(), mode=ExecMode.NORMAL),
            {"marker_mode": "normal", "scrambled_rows": 2},
            "written in normal mode",
        ),
        (
            replace(clean_marker(), mode=ExecMode.CONTROL, scrambled_ids=None),
            {"marker_mode": "control", "scrambled_rows": None},
            "written in control mode",
        ),
    ],
)
def test_a_malformed_marker_is_not_applied_and_never_passes(marker, evidence, problem):
    # The canary equals normal, so a marker that is not checked would pass.
    normal = make_snapshot()
    control = ControlRunner(normal)
    outcome = decide_canary(normal, make_snapshot(), marker, control, ())
    assert (outcome.outcome, outcome.determinism) == (
        Outcome.NOT_APPLIED,
        Outcome.SKIPPED,
    )
    (finding,) = outcome.findings
    assert finding.check == "canary"
    assert finding.severity is Severity.WARNING
    assert finding.message.startswith("the canary test was not applied")
    assert problem in finding.message
    assert finding.evidence == evidence
    assert "hexperiment run" in finding.fix
    assert control.calls == 0
    assert_unwaivable(outcome)


@pytest.mark.parametrize(
    ("features", "difference"),
    [
        (("age", "income"), "only in normal: plan; only in canary: income"),
        (("age",), "only in normal: plan"),
        (("age", "plan", "income"), "only in canary: income"),
        (("plan", "age"), "the same names in another order or number"),
    ],
)
def test_canary_with_other_features_and_matching_control_is_the_peeking_spirit(
    features, difference
):
    # Feature selection on all rows: the features move when only test rows do.
    normal = make_snapshot()
    canary = replace(changed_row(normal), features=features)
    control = ControlRunner(normal)
    outcome = decide_canary(normal, canary, clean_marker(), control, ("train.py:7",))
    assert (outcome.outcome, outcome.determinism) == (Outcome.FAILED, Outcome.PASSED)
    (finding,) = outcome.findings
    assert finding.check == "canary"
    assert finding.ghost == "The Peeking Spirit"
    assert finding.severity is Severity.ERROR
    assert finding.message == (
        f"features selected changed when only test rows were altered: {difference}"
    )
    assert finding.evidence == {
        "train_rows": 4,
        "rows_changed": 1,
        "feature_rows_changed": 1,
        "largest_change": 0.0,
        "normal_features": FEATURES,
        "canary_features": list(features),
    }
    assert "Select features on the training rows only." in finding.fix
    assert finding.fix.endswith("Suspect lines: train.py:7.")
    assert control.calls == 1
    assert_unwaivable(outcome)


def test_features_differing_with_identical_rows_still_go_to_the_control():
    # Equal hashes and scores do not make a changed feature list a pass.
    normal = make_snapshot()
    canary = replace(normal, features=("age",))
    control = ControlRunner(normal)
    outcome = decide_canary(normal, canary, clean_marker(), control, ())
    assert (outcome.outcome, outcome.determinism) == (Outcome.FAILED, Outcome.PASSED)
    (finding,) = outcome.findings
    assert finding.ghost == "The Peeking Spirit"
    assert finding.evidence["rows_changed"] == 0
    assert finding.evidence["canary_features"] == ["age"]
    assert control.calls == 1


def test_features_differing_without_control_scores_still_fail():
    # The feature change is a leak signal even when the control has no scores to
    # compare, unlike a change in predictions alone.
    normal = make_snapshot()
    canary = replace(normal, features=("age",))
    control = ControlRunner(with_scores(normal, None))
    outcome = decide_canary(normal, canary, clean_marker(), control, ())
    assert (outcome.outcome, outcome.determinism) == (Outcome.FAILED, Outcome.PASSED)
    assert outcome.findings[0].ghost == "The Peeking Spirit"


def test_canary_other_features_and_control_rows_differ_is_the_shapeshifter():
    normal = make_snapshot()
    canary = replace(changed_row(normal), features=("age",))
    control = ControlRunner(changed_row(normal))
    outcome = decide_canary(normal, canary, clean_marker(), control, ())
    assert (outcome.outcome, outcome.determinism) == (
        Outcome.INCONCLUSIVE,
        Outcome.FAILED,
    )
    (finding,) = outcome.findings
    assert finding.ghost == "The Shapeshifter"
    assert finding.evidence == {"rows_changed": 1, "largest_change": 0.0}
    assert control.calls == 1


def test_canary_and_control_with_other_features_is_the_shapeshifter():
    normal = make_snapshot()
    canary = replace(changed_row(normal), features=("age",))
    control = ControlRunner(replace(normal, features=("plan",)))
    outcome = decide_canary(normal, canary, clean_marker(), control, ())
    assert (outcome.outcome, outcome.determinism) == (
        Outcome.INCONCLUSIVE,
        Outcome.FAILED,
    )
    (finding,) = outcome.findings
    assert finding.ghost == "The Shapeshifter"
    assert finding.evidence == {
        "normal_features": FEATURES,
        "control_features": ["plan"],
    }
    assert control.calls == 1


def test_control_with_other_features_is_the_shapeshifter():
    normal = make_snapshot()
    control = ControlRunner(replace(normal, features=("age", "income")))
    outcome = decide_canary(normal, changed_row(normal), clean_marker(), control, ())
    assert (outcome.outcome, outcome.determinism) == (
        Outcome.INCONCLUSIVE,
        Outcome.FAILED,
    )
    (finding,) = outcome.findings
    assert finding.check == "determinism"
    assert finding.ghost == "The Shapeshifter"
    assert finding.message == (
        "training is not deterministic: an identical execution used other "
        "features (only in normal: plan; only in control: income)"
    )
    assert finding.evidence == {
        "normal_features": FEATURES,
        "control_features": ["age", "income"],
    }
    assert control.calls == 1
    assert_unwaivable(outcome)


# --- decide_canary: training predictions in some executions only -------------------


def mixed_warnings(outcome: CanaryOutcome) -> list[Finding]:
    return [f for f in outcome.findings if "some executions but not" in f.message]


@pytest.mark.parametrize("missing_in", ["normal", "canary"])
def test_scores_in_only_normal_or_canary_never_pass(missing_in):
    # Equal hashes, and the predictions moved where they were supplied: before
    # this rule the comparison fell back to the hashes and passed.
    normal = make_snapshot(scores=None if missing_in == "normal" else (0.1,) * 4)
    canary = make_snapshot(scores=None if missing_in == "canary" else (0.9,) * 4)
    control = ControlRunner(normal)
    outcome = decide_canary(normal, canary, clean_marker(), control, ())
    assert (outcome.outcome, outcome.determinism) == (
        Outcome.INCONCLUSIVE,
        Outcome.INCONCLUSIVE,
    )
    (warning,) = outcome.findings
    supplied = "canary" if missing_in == "normal" else "normal"
    assert warning.check == "canary"
    assert warning.ghost is None
    assert warning.severity is Severity.WARNING
    assert warning.message == (
        "training predictions were supplied in some executions but not others "
        f"(supplied in {supplied}; missing in {missing_in}), so they could not "
        "be compared"
    )
    assert warning.evidence == {
        "executions_with_y_score_train": [supplied],
        "executions_without_y_score_train": [missing_in],
    }
    assert "every execution" in warning.fix
    assert weaker_warnings(outcome) == []
    assert control.calls == 0
    assert_unwaivable(outcome)


def test_scores_in_only_normal_still_fail_on_a_feature_change():
    normal = make_snapshot()
    canary = with_scores(changed_row(normal), None)
    control = ControlRunner(normal)
    outcome = decide_canary(normal, canary, clean_marker(), control, ())
    assert (outcome.outcome, outcome.determinism) == (Outcome.FAILED, Outcome.PASSED)
    assert [f.ghost for f in outcome.findings] == ["The Peeking Spirit", None]
    (warning,) = mixed_warnings(outcome)
    assert warning.evidence == {
        "executions_with_y_score_train": ["normal", "control"],
        "executions_without_y_score_train": ["canary"],
    }
    assert control.calls == 1


def test_control_without_scores_cannot_confirm_a_prediction_change():
    # Only the predictions changed, and the control has none to compare.
    normal = make_snapshot()
    canary = with_scores(normal, (0.1, 0.9, 0.3, 0.4))
    control = ControlRunner(with_scores(normal, None))
    outcome = decide_canary(normal, canary, clean_marker(), control, ())
    assert (outcome.outcome, outcome.determinism) == (
        Outcome.INCONCLUSIVE,
        Outcome.INCONCLUSIVE,
    )
    (warning,) = outcome.findings
    assert warning.evidence == {
        "executions_with_y_score_train": ["normal", "canary"],
        "executions_without_y_score_train": ["control"],
    }
    assert control.calls == 1
    assert_unwaivable(outcome)


@pytest.mark.parametrize(
    ("control_has_scores", "control_changes", "expected"),
    [
        (False, False, (Outcome.FAILED, Outcome.PASSED)),
        (False, True, (Outcome.INCONCLUSIVE, Outcome.FAILED)),
        (True, False, (Outcome.FAILED, Outcome.PASSED)),
        (True, True, (Outcome.INCONCLUSIVE, Outcome.FAILED)),
    ],
)
def test_control_with_other_score_presence_warns_after_a_feature_change(
    control_has_scores, control_changes, expected
):
    # Normal and canary agree on scores; the control does not.
    scores = None if control_has_scores else (0.1, 0.2, 0.3, 0.4)
    normal = make_snapshot(scores=scores)
    control_scores = (0.1, 0.2, 0.3, 0.4) if control_has_scores else None
    control_snapshot = with_scores(normal, control_scores)
    if control_changes:
        control_snapshot = changed_row(control_snapshot)
    control = ControlRunner(control_snapshot)
    outcome = decide_canary(normal, changed_row(normal), clean_marker(), control, ())
    assert (outcome.outcome, outcome.determinism) == expected
    (warning,) = mixed_warnings(outcome)
    assert warning.evidence["executions_without_y_score_train"] == (
        ["normal", "canary"] if control_has_scores else ["control"]
    )
    assert weaker_warnings(outcome) == []
    assert control.calls == 1


# --- decide_canary: missing y_score_train and hint lines --------------------------


def weaker_warnings(outcome: CanaryOutcome) -> list[Finding]:
    return [f for f in outcome.findings if "weaker" in f.message]


def score_warnings(outcome: CanaryOutcome) -> list[Finding]:
    return [f for f in outcome.findings if "without training predictions" in f.message]


@pytest.mark.parametrize(
    ("canary_change", "control_change", "expected"),
    [
        (False, False, Outcome.INCONCLUSIVE),
        (True, False, Outcome.FAILED),
        (True, True, Outcome.INCONCLUSIVE),
    ],
)
def test_missing_training_scores_warns_in_steps_4_to_6(
    canary_change, control_change, expected
):
    normal = make_snapshot(scores=None)
    canary = changed_row(normal) if canary_change else normal
    control = ControlRunner(changed_row(normal) if control_change else normal)
    outcome = decide_canary(normal, canary, clean_marker(), control, ())
    assert outcome.outcome is expected
    (warning,) = score_warnings(outcome)
    assert warning.check == "canary"
    assert warning.ghost is None
    assert warning.severity is Severity.WARNING
    assert warning.fix == (
        "pass y_score_train=model.predict_proba(train[features])[:, 1] to "
        "hexperiment.audit"
    )
    if expected is Outcome.FAILED:
        spirit = next(f for f in outcome.findings if f.ghost == "The Peeking Spirit")
        assert spirit.evidence["largest_change"] is None


def test_step_4_without_training_scores_is_inconclusive_and_needs_no_control():
    # Requirement 2.6: equal hashes do not rule out a model fitted on test rows.
    normal = make_snapshot(scores=None)
    control = ControlRunner(normal)
    outcome = decide_canary(normal, normal, clean_marker(), control, ())
    assert outcome.outcome is Outcome.INCONCLUSIVE
    assert outcome.determinism is Outcome.PASSED
    assert control.calls == 0
    (warning,) = outcome.findings
    assert "compared" in warning.message
    assert "training features" in warning.message
    assert "inconclusive" in warning.message


def test_step_4_with_training_scores_still_passes():
    normal = make_snapshot()
    outcome = decide_canary(normal, normal, clean_marker(), ControlRunner(normal), ())
    assert (outcome.outcome, outcome.determinism) == (Outcome.PASSED, Outcome.PASSED)
    assert outcome.findings == ()


def test_a_difference_without_training_scores_is_still_reported():
    normal = make_snapshot(scores=None)
    control = ControlRunner(normal)
    outcome = decide_canary(normal, changed_row(normal), clean_marker(), control, ())
    assert outcome.outcome is Outcome.FAILED
    assert control.calls == 1


def test_scores_present_gives_no_weaker_warning():
    normal = make_snapshot()
    outcome = decide_canary(normal, normal, clean_marker(), ControlRunner(normal), ())
    assert weaker_warnings(outcome) == []


@pytest.mark.parametrize("marker", [None, clean_marker(scrambled=(0, 4))])
def test_no_weaker_warning_when_the_canary_was_not_applied(marker):
    normal = make_snapshot(scores=None)
    outcome = decide_canary(normal, normal, marker, ControlRunner(normal), ())
    assert weaker_warnings(outcome) == []


def test_peeking_spirit_fix_lists_the_hint_lines():
    normal = make_snapshot()
    outcome = decide_canary(
        normal,
        changed_row(normal),
        clean_marker(),
        ControlRunner(normal),
        ("train.py:14", "train.py:20"),
    )
    (finding,) = outcome.findings
    assert finding.fix == (
        "test rows influenced training; fit preprocessing on training rows only, "
        "after the split. Suspect lines: train.py:14, train.py:20."
    )


def split_order_finding(location: str) -> Finding:
    return Finding(
        check="split_order",
        ghost=None,
        severity=Severity.WARNING,
        location=location,
        message="mean() is called before the first split",
        evidence={"call": "mean", "first_split_line": 30},
        fix="this runs before the split; make sure it does not use test rows",
    )


def make_result(checks) -> DataAuditResult:
    return DataAuditResult(
        run_id="run",
        metric=MetricResult("roc_auc", 0.8, 0.7, True),
        checks=tuple(checks),
        extra_findings=(),
        data=None,
        input_error=None,
    )


def test_hint_locations_reads_split_order_line_locations():
    overlap = CheckResult(
        "row_overlap",
        "The Doppelganger",
        Outcome.FAILED,
        (split_order_finding("train vs test"),),
    )
    split_order = CheckResult(
        "split_order",
        None,
        Outcome.PASSED,
        tuple(
            split_order_finding(loc)
            for loc in ("train.py:14", "train.py:20", "train.py:14")
        ),
    )
    result = make_result([overlap, split_order])
    assert hint_locations(result) == ("train.py:14", "train.py:20")


def test_hint_locations_skips_the_inconclusive_warning_and_empty_results():
    inconclusive = CheckResult(
        "split_order", None, Outcome.INCONCLUSIVE, (split_order_finding("train.py"),)
    )
    assert hint_locations(make_result([inconclusive])) == ()
    assert hint_locations(make_result([])) == ()


def test_hint_locations_ignores_line_like_decoys():
    # Line locations outside a split_order check, or under another check name
    # inside it, are not split-order hints.
    other_check = CheckResult(
        "target_proxy",
        "The Possession",
        Outcome.FAILED,
        (
            split_order_finding("train.py:3"),
            replace(split_order_finding("train.py:4"), check="target_proxy"),
        ),
    )
    split_order = CheckResult(
        "split_order",
        None,
        Outcome.PASSED,
        (
            replace(split_order_finding("train.py:5"), check="row_overlap"),
            replace(split_order_finding("train.py:6"), check="canary"),
            split_order_finding("train.py:"),
            split_order_finding("train.py:7a"),
            split_order_finding("train.py:8 (fit)"),
            split_order_finding("helpers.py:9"),
            split_order_finding("train.py:14"),
        ),
    )
    result = make_result([other_check, split_order])
    assert hint_locations(result) == ("helpers.py:9", "train.py:14")


def test_skipped_outcome():
    assert skipped_outcome() == CanaryOutcome(Outcome.SKIPPED, Outcome.SKIPPED, ())


# --- properties -------------------------------------------------------------------


@st.composite
def snapshots(draw) -> Snapshot:
    n = draw(st.integers(min_value=1, max_value=12))
    ids = sorted(draw(st.sets(st.integers(0, 200), min_size=n, max_size=n)))
    hashes = draw(
        st.lists(st.integers(0, 2**64 - 1), min_size=n, max_size=n)
    )
    has_scores = draw(st.booleans())
    scores = (
        draw(st.lists(st.floats(0.0, 1.0), min_size=n, max_size=n))
        if has_scores
        else None
    )
    return make_snapshot(train_ids=ids, hashes=hashes, scores=scores, test_ids=())


@given(snapshots())
def test_comparing_a_snapshot_with_itself_never_differs(snapshot):
    diff = compare_snapshots(snapshot, snapshot)
    assert diff.ids_equal
    assert not diff.differs
    assert diff.rows_changed == 0
    assert diff.max_abs_change == (None if snapshot.y_score_train is None else 0.0)


@given(
    st.lists(
        st.tuples(
            st.one_of(st.none(), st.floats(-1e6, 1e6)),
            st.sampled_from(["a", "b", "c", None]),
            st.floats(0.0, 1.0),
        ),
        min_size=1,
        max_size=15,
    ),
    st.randoms(use_true_random=False),
)
def test_row_order_never_changes_the_snapshot(rows, rnd):
    frame = pd.DataFrame(
        {"age": [r[0] for r in rows], "plan": [r[1] for r in rows]},
        index=range(len(rows)),
    )
    scores = np.array([r[2] for r in rows])
    order = list(range(len(rows)))
    rnd.shuffle(order)
    a = build_snapshot(ExecMode.NORMAL, None, frame, frame, FEATURES, scores)
    b = build_snapshot(
        ExecMode.NORMAL, None, frame.iloc[order], frame, FEATURES, scores[order]
    )
    assert not compare_snapshots(a, b).differs
    assert a.train_row_hash.tolist() == b.train_row_hash.tolist()


def _variant(base: Snapshot, kind: str) -> Snapshot:
    if kind == "same":
        return base
    if kind == "hash":
        return changed_row(base)
    if kind == "score":
        if base.y_score_train is None:
            return base
        scores = base.y_score_train.copy()
        scores[0] = 1.0 - scores[0] if scores[0] != 0.5 else 0.9
        return with_scores(base, scores)
    # "ids": one training row is replaced by a row that is not a test row.
    ids = base.train_ids.copy()
    ids[-1] = 100
    return make_snapshot(train_ids=ids, hashes=base.train_row_hash, scores=None)


@given(
    has_scores=st.booleans(),
    marker_kind=st.sampled_from(["missing", "clean", "scrambled_in_train"]),
    canary_kind=st.sampled_from(["same", "hash", "score", "ids"]),
    control_kind=st.sampled_from(["same", "hash", "score", "ids"]),
)
def test_control_runs_at_most_once_and_only_when_needed(
    has_scores, marker_kind, canary_kind, control_kind
):
    normal = make_snapshot(scores=(0.1, 0.2, 0.3, 0.4) if has_scores else None)
    canary = _variant(normal, canary_kind)
    control = ControlRunner(_variant(normal, control_kind))
    marker = {
        "missing": None,
        "clean": clean_marker(),
        "scrambled_in_train": clean_marker(scrambled=(3, 4, 5)),
    }[marker_kind]

    outcome = decide_canary(normal, canary, marker, control, ())

    ids_equal = np.array_equal(normal.train_ids, canary.train_ids)
    canary_differs = compare_snapshots(normal, canary).differs
    needed = marker_kind == "clean" and ids_equal and canary_differs
    assert control.calls <= 1
    assert control.calls == (1 if needed else 0)
    if needed:
        control_differs = compare_snapshots(normal, control.snapshot).differs
        expected = Outcome.INCONCLUSIVE if control_differs else Outcome.FAILED
        assert outcome.outcome is expected


@given(
    presence=st.tuples(st.booleans(), st.booleans(), st.booleans()),
    canary_kind=st.sampled_from(["same", "hash", "score"]),
    control_kind=st.sampled_from(["same", "hash", "score"]),
)
def test_training_scores_in_some_executions_only_never_pass(
    presence, canary_kind, control_kind
):
    base = make_snapshot()

    def build(kind: str, has_scores: bool) -> Snapshot:
        snapshot = _variant(base, kind)
        return snapshot if has_scores else with_scores(snapshot, None)

    normal = build("same", presence[0])
    canary = build(canary_kind, presence[1])
    control = ControlRunner(build(control_kind, presence[2]))

    outcome = decide_canary(normal, canary, clean_marker(), control, ())

    compared = presence if control.calls else presence[:2]
    if len(set(compared)) > 1:
        assert outcome.outcome is not Outcome.PASSED
        assert len(mixed_warnings(outcome)) == 1
        assert weaker_warnings(outcome) == []
    elif not any(compared):
        # Either the weaker warning (a difference) or the step 4 warning (none).
        assert len(score_warnings(outcome)) == 1
        assert mixed_warnings(outcome) == []
        assert outcome.outcome is not Outcome.PASSED
    else:
        assert weaker_warnings(outcome) == mixed_warnings(outcome) == []


# --- the .hex/ protocol (Requirements 2.3, 2.4, 10.6) ------------------------------

RUN = "20251101-093012-a1b2c3"
OTHER_RUN = "20251101-093055-ffffff"


def assert_same_snapshot(actual: Snapshot, expected: Snapshot) -> None:
    """Compare snapshots bit for bit, so that -0.0 and NaN payloads count too."""
    assert actual.mode is expected.mode
    assert actual.run_id == expected.run_id
    assert actual.features == expected.features
    for name in ("train_ids", "test_ids", "train_row_hash", "y_score_train"):
        a, e = getattr(actual, name), getattr(expected, name)
        if e is None:
            assert a is None
        else:
            assert a.dtype == e.dtype
            assert a.tobytes() == e.tobytes()


def error_finding() -> Finding:
    return Finding(
        check="input",
        ghost=None,
        severity=Severity.ERROR,
        location="audit()",
        message="y_score has 3 entries, but test has 5 rows",
        evidence={"expected": 5, "found": 3},
        fix="pass one score per test row",
    )


def audit_result(*, input_error: bool = False) -> DataAuditResult:
    if input_error:
        return DataAuditResult(
            run_id=RUN,
            metric=None,
            checks=(),
            extra_findings=(),
            data=None,
            input_error=error_finding(),
        )
    overlap = Finding(
        check="row_overlap",
        ghost="The Doppelganger",
        severity=Severity.ERROR,
        location="train vs test",
        message="3 of 20 test rows also appear in train",
        evidence={"rows": 3, "test_rows": 20, "fraction": 0.15},
        fix=(
            "after the split, drop every test row that also appears in the "
            "training rows"
        ),
    )
    return DataAuditResult(
        run_id=RUN,
        metric=MetricResult("roc_auc", 0.912345, 0.8, True),
        checks=(
            CheckResult("row_overlap", "The Doppelganger", Outcome.FAILED, (overlap,)),
            CheckResult("temporal", "The Prophet", Outcome.NOT_APPLICABLE),
        ),
        extra_findings=(),
        data=DataSummary(80, 20, 0.25, 0.3, ("2024-01-01", "2024-03-01"), None),
        input_error=None,
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rewrite_json(path, **changes) -> None:
    payload = read_json(path)
    payload.update(changes)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_hex_dir_is_one_folder_per_mode_below_dot_hex(tmp_path):
    assert hex_dir(tmp_path, ExecMode.CANARY) == tmp_path / ".hex" / "canary"
    assert hex_dir(tmp_path, ExecMode.NORMAL) == tmp_path / ".hex" / "normal"


# snapshots


def test_snapshot_round_trip_with_training_scores(tmp_path):
    train = make_frame(ids=(3, 0, 2))
    scores = np.array([0.1 + 0.2, 0.0, 1.0 / 3.0])  # 0.1 + 0.2 is not exact
    written = build_snapshot(
        ExecMode.NORMAL, RUN, train, make_frame(ids=(9, 5)), FEATURES, scores
    )
    write_snapshot(tmp_path, written)
    read = read_snapshot(tmp_path, ExecMode.NORMAL, RUN)
    assert_same_snapshot(read, written)
    assert read.train_row_hash.dtype == np.uint64
    assert read.train_row_hash.max() > 2**63  # a real hash needs the full range


def test_snapshot_round_trip_without_training_scores(tmp_path):
    written = make_snapshot(scores=None, mode=ExecMode.CONTROL)
    write_snapshot(tmp_path, written)
    assert_same_snapshot(read_snapshot(tmp_path, ExecMode.CONTROL, RUN), written)
    with np.load(hex_dir(tmp_path, ExecMode.CONTROL) / "snapshot.npz") as npz:
        assert sorted(npz.files) == ["test_ids", "train_ids", "train_row_hash"]
    meta = read_json(hex_dir(tmp_path, ExecMode.CONTROL) / "meta.json")
    assert meta["has_y_score_train"] is False


def test_snapshot_meta_json_describes_the_snapshot(tmp_path):
    write_snapshot(tmp_path, make_snapshot(mode=ExecMode.CANARY))
    path = hex_dir(tmp_path, ExecMode.CANARY) / "meta.json"
    meta = read_json(path)
    assert set(meta) == {
        "features",
        "format_version",
        "has_y_score_train",
        "hexperiment_version",
        "mode",
        "n_test",
        "n_train",
        "run_id",
        "script",
    }
    assert meta["format_version"] == 1
    assert meta["features"] == FEATURES
    assert meta["has_y_score_train"] is True
    assert (meta["mode"], meta["n_train"], meta["n_test"]) == ("canary", 4, 2)
    assert (meta["run_id"], meta["script"]) == (RUN, "train.py")
    assert list(meta) == sorted(meta)  # keys are written sorted


def test_write_snapshot_replaces_an_earlier_snapshot_of_the_same_mode(tmp_path):
    write_snapshot(tmp_path, make_snapshot(scores=(0.1, 0.2, 0.3, 0.4)))
    second = make_snapshot(train_ids=(0, 1), hashes=[5, 6], scores=None)
    write_snapshot(tmp_path, second)
    assert_same_snapshot(read_snapshot(tmp_path, ExecMode.NORMAL, RUN), second)


def test_read_snapshot_rejects_a_snapshot_of_another_run(tmp_path):
    write_snapshot(tmp_path, make_snapshot())
    with pytest.raises(ExecutionFailed, match="another run is using this app") as info:
        read_snapshot(tmp_path, ExecMode.NORMAL, OTHER_RUN)
    assert RUN in str(info.value)
    assert OTHER_RUN in str(info.value)


def test_read_snapshot_rejects_a_standalone_snapshot(tmp_path):
    standalone = build_snapshot(
        ExecMode.NORMAL, None, make_frame(), make_frame(ids=(8, 9)), FEATURES, None
    )
    write_snapshot(tmp_path, standalone)
    with pytest.raises(ExecutionFailed, match="another run is using this app"):
        read_snapshot(tmp_path, ExecMode.NORMAL, RUN)


@pytest.mark.parametrize("version", [2, 0, "1", None, True])
def test_read_snapshot_rejects_another_format_version(tmp_path, version):
    write_snapshot(tmp_path, make_snapshot())
    meta = hex_dir(tmp_path, ExecMode.NORMAL) / "meta.json"
    rewrite_json(meta, format_version=version)
    with pytest.raises(ExecutionFailed, match="format_version.*reinstall hexperiment"):
        read_snapshot(tmp_path, ExecMode.NORMAL, RUN)


def test_read_snapshot_without_files_says_the_script_did_not_audit(tmp_path):
    with pytest.raises(ExecutionFailed, match="did not call hexperiment.audit"):
        read_snapshot(tmp_path, ExecMode.NORMAL, RUN)


def test_read_snapshot_reads_only_the_requested_mode(tmp_path):
    write_snapshot(tmp_path, make_snapshot(mode=ExecMode.NORMAL))
    with pytest.raises(ExecutionFailed, match="canary mode"):
        read_snapshot(tmp_path, ExecMode.CANARY, RUN)


def test_read_snapshot_reports_a_missing_array_file(tmp_path):
    write_snapshot(tmp_path, make_snapshot())
    (hex_dir(tmp_path, ExecMode.NORMAL) / "snapshot.npz").unlink()
    with pytest.raises(ExecutionFailed, match=r"snapshot\.npz does not exist"):
        read_snapshot(tmp_path, ExecMode.NORMAL, RUN)


def test_read_snapshot_refuses_pickled_arrays(tmp_path):
    write_snapshot(tmp_path, make_snapshot())
    npz = hex_dir(tmp_path, ExecMode.NORMAL) / "snapshot.npz"
    np.savez(
        npz,
        train_ids=np.array([0, 1, 2, 3], dtype=np.int64),
        test_ids=np.array([4, 5], dtype=np.int64),
        train_row_hash=np.array([{"a": 1}] * 4, dtype=object),
        y_score_train=np.zeros(4),
    )
    with pytest.raises(ExecutionFailed, match="cannot be used"):
        read_snapshot(tmp_path, ExecMode.NORMAL, RUN)


@pytest.mark.parametrize(
    "damage",
    [
        "not json",
        "not an object",
        "missing features",
        "bad array",
        "wrong dtype",
        "length mismatch",
        "scores disagree with meta",
    ],
)
def test_read_snapshot_rejects_damaged_files(tmp_path, damage):
    write_snapshot(tmp_path, make_snapshot())
    folder = hex_dir(tmp_path, ExecMode.NORMAL)
    arrays = {
        "train_ids": np.array([0, 1, 2, 3], dtype=np.int64),
        "test_ids": np.array([4, 5], dtype=np.int64),
        "train_row_hash": np.array([1, 2, 3, 4], dtype=np.uint64),
        "y_score_train": np.zeros(4),
    }
    if damage == "not json":
        (folder / "meta.json").write_text("{", encoding="utf-8")
    elif damage == "not an object":
        (folder / "meta.json").write_text("[1, 2]", encoding="utf-8")
    elif damage == "missing features":
        meta = read_json(folder / "meta.json")
        del meta["features"]
        (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    elif damage == "bad array":
        (folder / "snapshot.npz").write_bytes(b"this is not an npz file")
    elif damage == "wrong dtype":
        np.savez(folder / "snapshot.npz", **{**arrays, "train_ids": np.arange(4.0)})
    elif damage == "length mismatch":
        np.savez(folder / "snapshot.npz", **{**arrays, "y_score_train": np.zeros(3)})
    else:
        del arrays["y_score_train"]
        np.savez(folder / "snapshot.npz", **arrays)
    with pytest.raises(ExecutionFailed, match="cannot be used"):
        read_snapshot(tmp_path, ExecMode.NORMAL, RUN)


# load markers


def test_normal_marker_round_trip_writes_no_scrambled_ids(tmp_path):
    marker = LoadMarker(ExecMode.NORMAL, RUN, None)
    write_load_marker(tmp_path, marker)
    assert read_load_marker(tmp_path, ExecMode.NORMAL, RUN) == marker
    folder = hex_dir(tmp_path, ExecMode.NORMAL)
    assert not (folder / "scrambled_ids.npy").exists()
    assert read_json(folder / "loaded.json") == {
        "format_version": 1,
        "mode": "normal",
        "n_scrambled": None,
        "run_id": RUN,
    }


def test_canary_marker_round_trip_keeps_the_scrambled_ids(tmp_path):
    write_load_marker(tmp_path, clean_marker(scrambled=(9, 4, 17)))
    read = read_load_marker(tmp_path, ExecMode.CANARY, RUN)
    assert read.mode is ExecMode.CANARY
    assert read.run_id == RUN
    assert read.scrambled_ids.dtype == np.int64
    assert read.scrambled_ids.tolist() == [9, 4, 17]
    meta = read_json(hex_dir(tmp_path, ExecMode.CANARY) / "loaded.json")
    assert meta["n_scrambled"] == 3


def test_canary_marker_with_no_scrambled_rows_is_not_a_missing_marker(tmp_path):
    write_load_marker(tmp_path, clean_marker(scrambled=()))
    read = read_load_marker(tmp_path, ExecMode.CANARY, RUN)
    assert read is not None
    assert read.scrambled_ids.tolist() == []


def test_read_load_marker_returns_none_when_the_script_never_loaded(tmp_path):
    assert read_load_marker(tmp_path, ExecMode.CANARY, RUN) is None
    write_snapshot(tmp_path, make_snapshot(mode=ExecMode.CANARY))  # same folder
    assert read_load_marker(tmp_path, ExecMode.CANARY, RUN) is None


def test_read_load_marker_rejects_a_marker_of_another_run(tmp_path):
    write_load_marker(tmp_path, clean_marker())
    with pytest.raises(ExecutionFailed, match="another run is using this app"):
        read_load_marker(tmp_path, ExecMode.CANARY, OTHER_RUN)


def test_read_load_marker_rejects_another_format_version(tmp_path):
    write_load_marker(tmp_path, clean_marker())
    rewrite_json(hex_dir(tmp_path, ExecMode.CANARY) / "loaded.json", format_version=2)
    with pytest.raises(ExecutionFailed, match="format_version 2"):
        read_load_marker(tmp_path, ExecMode.CANARY, RUN)


@pytest.mark.parametrize("damage", ["ids missing", "ids count", "ids dtype"])
def test_read_load_marker_rejects_damaged_scrambled_ids(tmp_path, damage):
    write_load_marker(tmp_path, clean_marker(scrambled=(4, 5)))
    ids = hex_dir(tmp_path, ExecMode.CANARY) / "scrambled_ids.npy"
    if damage == "ids missing":
        ids.unlink()
    elif damage == "ids count":
        np.save(ids, np.array([4, 5, 6], dtype=np.int64))
    else:
        np.save(ids, np.array([4.0, 5.0]))
    with pytest.raises(ExecutionFailed, match="cannot be used"):
        read_load_marker(tmp_path, ExecMode.CANARY, RUN)


# results


def test_result_round_trip(tmp_path):
    written = audit_result()
    write_result(tmp_path, written, RUN)
    assert read_result(tmp_path, RUN) == written


def test_result_round_trip_with_an_input_error(tmp_path):
    written = audit_result(input_error=True)
    write_result(tmp_path, written, RUN)
    read = read_result(tmp_path, RUN)
    assert read == written
    assert read.input_error == error_finding()
    assert read.metric is None


def test_result_json_is_sorted_utf8_json_with_run_id_and_version(tmp_path):
    write_result(tmp_path, audit_result(), RUN)
    path = hex_dir(tmp_path, ExecMode.NORMAL) / "result.json"
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert payload["format_version"] == 1
    assert payload["run_id"] == RUN
    assert list(payload) == sorted(payload)
    assert text.endswith("}\n")


def test_result_file_carries_the_callers_run_id(tmp_path):
    write_result(tmp_path, audit_result(), OTHER_RUN)
    assert read_result(tmp_path, OTHER_RUN).run_id == OTHER_RUN
    with pytest.raises(ExecutionFailed, match="another run is using this app"):
        read_result(tmp_path, RUN)


def test_read_result_rejects_a_standalone_result(tmp_path):
    write_result(tmp_path, audit_result(), None)
    with pytest.raises(ExecutionFailed, match="another run is using this app"):
        read_result(tmp_path, RUN)


def test_read_result_rejects_another_format_version(tmp_path):
    write_result(tmp_path, audit_result(), RUN)
    rewrite_json(hex_dir(tmp_path, ExecMode.NORMAL) / "result.json", format_version=3)
    with pytest.raises(ExecutionFailed, match="format_version 3"):
        read_result(tmp_path, RUN)


def test_read_result_without_a_file_says_the_script_did_not_audit(tmp_path):
    with pytest.raises(ExecutionFailed, match="did not call hexperiment.audit"):
        read_result(tmp_path, RUN)


def test_read_result_rejects_a_file_that_is_not_an_audit_result(tmp_path):
    write_result(tmp_path, audit_result(), RUN)
    path = hex_dir(tmp_path, ExecMode.NORMAL) / "result.json"
    payload = read_json(path)
    del payload["checks"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExecutionFailed, match="not a valid audit result"):
        read_result(tmp_path, RUN)


# clear_mode_dir


def test_clear_mode_dir_removes_stale_files_of_that_mode_only(tmp_path):
    write_snapshot(tmp_path, make_snapshot(mode=ExecMode.NORMAL))
    write_result(tmp_path, audit_result(), RUN)
    write_snapshot(tmp_path, make_snapshot(mode=ExecMode.CANARY))
    write_load_marker(tmp_path, clean_marker())
    stale = hex_dir(tmp_path, ExecMode.NORMAL) / "left_over.txt"
    stale.write_text("stale", encoding="utf-8")

    clear_mode_dir(tmp_path, ExecMode.NORMAL)

    assert not hex_dir(tmp_path, ExecMode.NORMAL).exists()
    with pytest.raises(ExecutionFailed, match="did not call hexperiment.audit"):
        read_snapshot(tmp_path, ExecMode.NORMAL, RUN)
    with pytest.raises(ExecutionFailed, match="did not call hexperiment.audit"):
        read_result(tmp_path, RUN)
    assert read_snapshot(tmp_path, ExecMode.CANARY, RUN).mode is ExecMode.CANARY
    assert read_load_marker(tmp_path, ExecMode.CANARY, RUN) is not None


def test_clear_mode_dir_accepts_a_missing_folder(tmp_path):
    clear_mode_dir(tmp_path, ExecMode.CONTROL)
    clear_mode_dir(tmp_path, ExecMode.CONTROL)
    assert not (tmp_path / ".hex").exists()


def test_clear_mode_dir_leaves_a_folder_that_can_be_written_again(tmp_path):
    write_snapshot(tmp_path, make_snapshot())
    clear_mode_dir(tmp_path, ExecMode.NORMAL)
    again = make_snapshot(scores=None)
    write_snapshot(tmp_path, again)
    assert_same_snapshot(read_snapshot(tmp_path, ExecMode.NORMAL, RUN), again)


# property


@st.composite
def written_snapshots(draw) -> Snapshot:
    int64 = st.integers(-(2**63), 2**63 - 1)
    train_ids = sorted(draw(st.sets(int64, max_size=15)))
    test_ids = sorted(draw(st.sets(int64, max_size=15)))
    n = len(train_ids)
    hashes = draw(st.lists(st.integers(0, 2**64 - 1), min_size=n, max_size=n))
    has_scores = draw(st.booleans())
    scores = draw(st.lists(st.floats(), min_size=n, max_size=n)) if has_scores else None
    return Snapshot(
        mode=draw(st.sampled_from(list(ExecMode))),
        run_id=draw(st.text(min_size=1, max_size=30)),
        features=tuple(draw(st.lists(st.text(max_size=10), max_size=4))),
        train_ids=np.array(train_ids, dtype=np.int64),
        test_ids=np.array(test_ids, dtype=np.int64),
        train_row_hash=np.array(hashes, dtype=np.uint64),
        y_score_train=None if scores is None else np.array(scores, dtype=np.float64),
    )


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(snapshot=written_snapshots())
def test_write_then_read_snapshot_returns_equal_arrays(tmp_path, snapshot):
    clear_mode_dir(tmp_path, snapshot.mode)
    write_snapshot(tmp_path, snapshot)
    read = read_snapshot(tmp_path, snapshot.mode, snapshot.run_id)
    assert_same_snapshot(read, snapshot)


# --- run_executions with a fake runner (Requirements 10.1 to 10.5, 16.5) -----------


def make_config(app_dir: Path, timeout_seconds: float = 120) -> HexConfig:
    raw = {
        "data": {"path": "data.csv"},
        "target": {"column": "target"},
        "metric": {"name": "roc_auc", "threshold": 0.8},
        "run": {"timeout_seconds": timeout_seconds},
    }
    return parse_config(raw, source=app_dir / "hex.toml", app_dir=app_dir)


def hinted_result() -> DataAuditResult:
    hint = CheckResult(
        "split_order", None, Outcome.PASSED, (split_order_finding("train.py:14"),)
    )
    return replace(audit_result(), checks=(hint,))


def run_full(app_dir: Path, runner, **config):
    return run_executions(
        app_dir, make_config(app_dir, **config), RUN, full=True, runner=runner
    )


def normal_plan(fake_runner, **changes):
    plan = fake_runner.Plan(snapshot=make_snapshot(), result=audit_result())
    return replace(plan, **changes)


def canary_plan(fake_runner, snapshot=None, marker="clean", **changes):
    plan = fake_runner.Plan(
        snapshot=make_snapshot() if snapshot is None else snapshot,
        marker=clean_marker() if marker == "clean" else marker,
    )
    return replace(plan, **changes)


def leak_plans(fake_runner, **changes):
    """Normal, a canary that differs in one row, and a control equal to normal."""
    return {
        "normal": normal_plan(fake_runner),
        "canary": canary_plan(fake_runner, snapshot=changed_row(make_snapshot())),
        "control": fake_runner.Plan(snapshot=make_snapshot()),
    } | changes


def test_a_fast_run_executes_normal_once_and_skips_the_canary(tmp_path, fake_runner):
    runner = fake_runner(normal=normal_plan(fake_runner))
    config = make_config(tmp_path)
    report = run_executions(tmp_path, config, RUN, full=False, runner=runner)
    assert runner.calls == [ExecMode.NORMAL]
    assert report.mode is RunMode.FAST
    assert report.canary == skipped_outcome()
    assert report.result == audit_result()


def test_a_fast_run_still_needs_the_snapshot_of_the_normal_execution(
    tmp_path, fake_runner
):
    runner = fake_runner(normal=fake_runner.Plan(result=audit_result()))
    config = make_config(tmp_path)
    with pytest.raises(ExecutionFailed, match="did not call hexperiment.audit"):
        run_executions(tmp_path, config, RUN, full=False, runner=runner)


def test_a_full_run_with_equal_snapshots_executes_twice(tmp_path, fake_runner):
    runner = fake_runner(
        normal=normal_plan(fake_runner), canary=canary_plan(fake_runner)
    )
    report = run_full(tmp_path, runner)
    assert runner.calls == [ExecMode.NORMAL, ExecMode.CANARY]
    assert report.mode is RunMode.FULL
    assert report.canary == CanaryOutcome(Outcome.PASSED, Outcome.PASSED, ())
    assert report.result == audit_result()


def test_a_differing_canary_triggers_exactly_one_control_run(tmp_path, fake_runner):
    runner = fake_runner(**leak_plans(fake_runner))
    run_full(tmp_path, runner)
    assert runner.calls == [ExecMode.NORMAL, ExecMode.CANARY, ExecMode.CONTROL]


def test_a_control_that_matches_normal_gives_the_peeking_spirit(tmp_path, fake_runner):
    report = run_full(tmp_path, fake_runner(**leak_plans(fake_runner)))
    assert report.canary.outcome is Outcome.FAILED
    assert report.canary.determinism is Outcome.PASSED
    (finding,) = report.canary.findings
    assert finding.ghost == "The Peeking Spirit"
    assert finding.severity is Severity.ERROR
    assert finding.evidence["rows_changed"] == 1


def test_a_control_that_differs_gives_the_shapeshifter(tmp_path, fake_runner):
    different = with_scores(make_snapshot(), (0.1, 0.2, 0.3, 0.5))
    plans = leak_plans(fake_runner, control=fake_runner.Plan(snapshot=different))
    runner = fake_runner(**plans)
    report = run_full(tmp_path, runner)
    assert runner.calls == [ExecMode.NORMAL, ExecMode.CANARY, ExecMode.CONTROL]
    assert report.canary.outcome is Outcome.INCONCLUSIVE
    assert report.canary.determinism is Outcome.FAILED
    (finding,) = report.canary.findings
    assert finding.ghost == "The Shapeshifter"
    assert finding.severity is Severity.WARNING


def test_the_peeking_spirit_lists_the_lines_of_the_split_order_hint(
    tmp_path, fake_runner
):
    normal = normal_plan(fake_runner, result=hinted_result())
    report = run_full(tmp_path, fake_runner(**leak_plans(fake_runner, normal=normal)))
    (finding,) = report.canary.findings
    assert finding.fix.endswith("Suspect lines: train.py:14.")


def test_differing_id_sets_skip_the_canary_without_a_control_run(
    tmp_path, fake_runner
):
    other_ids = make_snapshot(train_ids=(0, 1, 2, 100), scores=None)
    runner = fake_runner(
        normal=normal_plan(fake_runner),
        canary=canary_plan(fake_runner, snapshot=other_ids),
    )
    report = run_full(tmp_path, runner)
    assert runner.calls == [ExecMode.NORMAL, ExecMode.CANARY]
    assert report.canary.outcome is Outcome.SKIPPED
    assert report.canary.determinism is Outcome.SKIPPED
    (warning,) = report.canary.findings
    assert warning.severity is Severity.WARNING


def test_scrambled_rows_in_train_give_not_applied(tmp_path, fake_runner):
    canary = canary_plan(
        fake_runner,
        snapshot=changed_row(make_snapshot()),
        marker=clean_marker(scrambled=(3, 4, 5)),
    )
    runner = fake_runner(normal=normal_plan(fake_runner), canary=canary)
    report = run_full(tmp_path, runner)
    assert runner.calls == [ExecMode.NORMAL, ExecMode.CANARY]
    assert report.canary.outcome is Outcome.NOT_APPLIED
    assert report.canary.determinism is Outcome.SKIPPED
    (warning,) = report.canary.findings
    assert "scrambled test rows ended up in the training set" in warning.message


def test_a_missing_load_marker_gives_not_applied(tmp_path, fake_runner):
    runner = fake_runner(
        normal=normal_plan(fake_runner),
        canary=canary_plan(fake_runner, marker=None),
    )
    report = run_full(tmp_path, runner)
    assert runner.calls == [ExecMode.NORMAL, ExecMode.CANARY]
    assert report.canary.outcome is Outcome.NOT_APPLIED
    (warning,) = report.canary.findings
    assert "did not load its data with hexperiment.load" in warning.message


@pytest.mark.parametrize("exits_with_error", [True, False])
def test_an_input_error_stops_after_the_first_execution(
    tmp_path, fake_runner, exits_with_error
):
    failure = "train.py exited with code 1" if exits_with_error else None
    runner = fake_runner(
        normal=fake_runner.Plan(result=audit_result(input_error=True), fail=failure),
        canary=canary_plan(fake_runner),
    )
    report = run_full(tmp_path, runner)
    assert runner.calls == [ExecMode.NORMAL]
    assert report.mode is RunMode.FULL
    assert report.canary == skipped_outcome()
    assert report.result.input_error == error_finding()
    assert report.result.metric is None


def test_a_fast_run_with_an_input_error_is_a_fast_report(tmp_path, fake_runner):
    plan = fake_runner.Plan(
        result=audit_result(input_error=True), fail="train.py exited with code 1"
    )
    runner = fake_runner(normal=plan)
    config = make_config(tmp_path)
    report = run_executions(tmp_path, config, RUN, full=False, runner=runner)
    assert report.mode is RunMode.FAST
    assert report.result.input_error == error_finding()


def test_a_failed_normal_execution_without_an_input_error_raises(
    tmp_path, fake_runner
):
    normal = normal_plan(fake_runner, fail="train.py exited with code 1: boom")
    runner = fake_runner(**leak_plans(fake_runner, normal=normal))
    with pytest.raises(ExecutionFailed, match="exited with code 1: boom"):
        run_full(tmp_path, runner)
    assert runner.calls == [ExecMode.NORMAL]


@pytest.mark.parametrize("full", [True, False])
def test_a_timeout_is_never_accepted_as_a_recorded_input_error(
    tmp_path, fake_runner, full
):
    plan = fake_runner.Plan(result=audit_result(input_error=True), timed_out=True)
    runner = fake_runner(normal=plan, canary=canary_plan(fake_runner))
    config = make_config(tmp_path)
    with pytest.raises(ExecutionTimedOut):
        run_executions(tmp_path, config, RUN, full=full, runner=runner)
    assert runner.calls == [ExecMode.NORMAL]


def test_a_failed_execution_with_an_input_error_of_another_run_raises(
    tmp_path, fake_runner
):
    plan = fake_runner.Plan(
        result=audit_result(input_error=True),
        written_run_id=OTHER_RUN,
        fail="train.py exited with code 1: boom",
    )
    runner = fake_runner(normal=plan, canary=canary_plan(fake_runner))
    with pytest.raises(ExecutionFailed, match="boom"):
        run_full(tmp_path, runner)
    assert runner.calls == [ExecMode.NORMAL]


def test_a_script_that_never_audits_raises(tmp_path, fake_runner):
    runner = fake_runner()
    with pytest.raises(ExecutionFailed, match="did not call hexperiment.audit"):
        run_full(tmp_path, runner)
    assert runner.calls == [ExecMode.NORMAL]


def test_a_missing_normal_snapshot_raises(tmp_path, fake_runner):
    runner = fake_runner(normal=fake_runner.Plan(result=audit_result()))
    with pytest.raises(ExecutionFailed, match="did not call hexperiment.audit"):
        run_full(tmp_path, runner)
    assert runner.calls == [ExecMode.NORMAL]


@pytest.mark.parametrize("mode", [ExecMode.NORMAL, ExecMode.CANARY, ExecMode.CONTROL])
def test_a_snapshot_of_another_run_raises_in_every_mode(tmp_path, fake_runner, mode):
    plans = leak_plans(fake_runner)
    plans[mode.value] = replace(plans[mode.value], written_run_id=OTHER_RUN)
    with pytest.raises(ExecutionFailed, match="another run is using this app"):
        run_full(tmp_path, fake_runner(**plans))


def test_a_canary_execution_that_fails_raises(tmp_path, fake_runner):
    canary = canary_plan(fake_runner, fail="train.py exited with code 1: canary")
    runner = fake_runner(normal=normal_plan(fake_runner), canary=canary)
    with pytest.raises(ExecutionFailed, match="canary"):
        run_full(tmp_path, runner)
    assert runner.calls == [ExecMode.NORMAL, ExecMode.CANARY]


def test_a_missing_canary_snapshot_raises(tmp_path, fake_runner):
    runner = fake_runner(normal=normal_plan(fake_runner))
    with pytest.raises(ExecutionFailed, match="canary mode"):
        run_full(tmp_path, runner)
    assert runner.calls == [ExecMode.NORMAL, ExecMode.CANARY]


def test_a_missing_control_snapshot_raises(tmp_path, fake_runner):
    plans = leak_plans(fake_runner, control=None)
    runner = fake_runner(**plans)
    with pytest.raises(ExecutionFailed, match="control mode"):
        run_full(tmp_path, runner)
    assert runner.calls == [ExecMode.NORMAL, ExecMode.CANARY, ExecMode.CONTROL]


def test_a_control_execution_that_fails_raises(tmp_path, fake_runner):
    control = fake_runner.Plan(fail="train.py exited with code 1: control")
    runner = fake_runner(**leak_plans(fake_runner, control=control))
    with pytest.raises(ExecutionFailed, match="control"):
        run_full(tmp_path, runner)


@pytest.mark.parametrize("mode", [ExecMode.NORMAL, ExecMode.CANARY, ExecMode.CONTROL])
def test_the_folder_of_a_mode_is_cleared_before_its_execution(
    tmp_path, fake_runner, mode
):
    # A runner that writes nothing must never be read through a stale file.
    plans = leak_plans(fake_runner)
    plans[mode.value] = fake_runner.Plan()
    write_snapshot(tmp_path, make_snapshot(mode=mode))
    write_load_marker(tmp_path, clean_marker())
    if mode is ExecMode.NORMAL:
        write_result(tmp_path, audit_result(), RUN)
    with pytest.raises(ExecutionFailed, match="did not call hexperiment.audit"):
        run_full(tmp_path, fake_runner(**plans))


def test_the_configured_timeout_reaches_every_execution(tmp_path, fake_runner):
    runner = fake_runner(**leak_plans(fake_runner))
    run_full(tmp_path, runner, timeout_seconds=7.5)
    assert runner.timeouts == [7.5, 7.5, 7.5]


@pytest.mark.parametrize("full", [False, True])
def test_run_executions_never_touches_the_app_level_verdict(
    tmp_path, fake_runner, full
):
    runner = fake_runner(
        normal=normal_plan(fake_runner), canary=canary_plan(fake_runner)
    )
    config = make_config(tmp_path)
    run_executions(tmp_path, config, RUN, full=full, runner=runner)
    assert not (tmp_path / "verdict.json").exists()

    (tmp_path / "verdict.json").write_text("sentinel", encoding="utf-8")
    run_executions(tmp_path, config, RUN, full=full, runner=runner)
    assert (tmp_path / "verdict.json").read_text(encoding="utf-8") == "sentinel"


def test_run_executions_leaves_the_app_verdict_alone_when_it_fails(
    tmp_path, fake_runner
):
    (tmp_path / "verdict.json").write_text("sentinel", encoding="utf-8")
    with pytest.raises(ExecutionFailed):
        run_full(tmp_path, fake_runner())
    assert (tmp_path / "verdict.json").read_text(encoding="utf-8") == "sentinel"


# --- run_train_script (Requirements 16.2, 16.3, 16.5) --------------------------------


def write_script(app_dir: Path, body: str) -> None:
    (app_dir / "train.py").write_text(textwrap.dedent(body), encoding="utf-8")


def test_run_train_script_runs_the_script_in_the_app_folder_with_the_run_env(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.chdir(tmp_path.parent)  # the caller's folder is not the app's
    write_script(
        tmp_path,
        """
        import json, os, sys

        seen = {
            "mode": os.environ.get("HEX_MODE"),
            "run_id": os.environ.get("HEX_RUN_ID"),
            "hash_seed": os.environ.get("PYTHONHASHSEED"),
            "cwd": os.getcwd(),
            "executable": sys.executable,
        }
        with open("seen.json", "w", encoding="utf-8") as handle:
            json.dump(seen, handle)
        """,
    )
    run_train_script(tmp_path, ExecMode.CANARY, RUN, 60)
    seen = read_json(tmp_path / "seen.json")
    assert (seen["mode"], seen["run_id"]) == ("canary", RUN)
    assert seen["hash_seed"] is None
    assert Path(seen["cwd"]).samefile(tmp_path)
    assert Path(seen["executable"]).samefile(sys.executable)


def test_run_train_script_raises_with_the_last_40_lines_of_stderr(tmp_path):
    write_script(
        tmp_path,
        """
        import sys

        for i in range(1, 61):
            print(f"err {i:03d}", file=sys.stderr)
        sys.exit(3)
        """,
    )
    with pytest.raises(ExecutionFailed, match="exited with code 3") as info:
        run_train_script(tmp_path, ExecMode.NORMAL, RUN, 60)
    message = str(info.value)
    assert "err 060" in message
    assert "err 021" in message
    assert "err 020" not in message


def test_run_train_script_reports_an_exception_of_the_script(tmp_path):
    write_script(tmp_path, "raise ValueError('boom from train.py')\n")
    with pytest.raises(ExecutionFailed, match="boom from train.py"):
        run_train_script(tmp_path, ExecMode.NORMAL, RUN, 60)


def test_run_train_script_stops_a_script_after_the_timeout(tmp_path):
    write_script(tmp_path, "import time\ntime.sleep(60)\n")
    started = time.monotonic()
    with pytest.raises(ExecutionFailed, match="longer than 1 seconds"):
        run_train_script(tmp_path, ExecMode.NORMAL, RUN, 1)
    assert time.monotonic() - started < 30


def test_run_train_script_starts_python_directly_with_the_documented_arguments(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    calls = []
    waits = []

    class FakeProcess:
        pid = 1
        returncode = 0

        def __init__(self, args, **kwargs):
            calls.append((args, kwargs))

        def wait(self, timeout=None):
            waits.append(timeout)
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    run_train_script(tmp_path, ExecMode.CONTROL, RUN, 12.5)
    ((args, kwargs),) = calls
    assert waits == [12.5]
    assert args == [sys.executable, "train.py"]
    assert kwargs.get("shell", False) is False
    assert kwargs["cwd"] == tmp_path
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["env"]["HEX_MODE"] == "control"
    assert kwargs["env"]["HEX_RUN_ID"] == RUN
    assert kwargs["env"]["PYTHONIOENCODING"] == "utf-8"
    assert "PYTHONHASHSEED" not in kwargs["env"]


def test_run_train_script_does_not_use_pipes_for_output(tmp_path, monkeypatch):
    # Requirement 16.3: a pipe held by a surviving child would keep the call waiting.
    seen = {}
    real_popen = subprocess.Popen

    def spy(args, **kwargs):
        seen.update(kwargs)
        return real_popen(args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", spy)
    write_script(tmp_path, "print('out')\n")
    run_train_script(tmp_path, ExecMode.NORMAL, RUN, 60)
    for stream in ("stdout", "stderr"):
        assert seen[stream] not in (subprocess.PIPE, subprocess.DEVNULL, None)
        assert seen[stream].closed  # the temporary file is gone afterwards


def test_run_train_script_does_not_wait_for_input(tmp_path):
    write_script(
        tmp_path,
        """
        import sys

        with open("stdin.txt", "w", encoding="utf-8") as handle:
            handle.write(repr(sys.stdin.read()))
        """,
    )
    started = time.monotonic()
    run_train_script(tmp_path, ExecMode.NORMAL, RUN, 60)
    assert (tmp_path / "stdin.txt").read_text(encoding="utf-8") == "''"
    assert time.monotonic() - started < 30


def test_run_train_script_gives_the_child_the_new_mode_and_run_id(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HEX_MODE", "control")
    monkeypatch.setenv("HEX_RUN_ID", "20000101-000000-stale0")
    write_script(
        tmp_path,
        """
        import os

        with open("seen.txt", "w", encoding="utf-8") as handle:
            handle.write(os.environ["HEX_MODE"] + " " + os.environ["HEX_RUN_ID"])
        """,
    )
    run_train_script(tmp_path, ExecMode.CANARY, RUN, 60)
    assert (tmp_path / "seen.txt").read_text(encoding="utf-8") == f"canary {RUN}"


def test_run_train_script_reports_non_ascii_stderr_intact(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONIOENCODING", "ascii")  # the child must override this
    write_script(
        tmp_path,
        """
        import sys

        print("naïve café 日本語 ✓", file=sys.stderr)
        sys.exit(3)
        """,
    )
    with pytest.raises(ExecutionFailed, match="exited with code 3") as info:
        run_train_script(tmp_path, ExecMode.NORMAL, RUN, 60)
    assert "naïve café 日本語 ✓" in str(info.value)


def test_run_train_script_replaces_bytes_that_are_not_utf8_in_stderr(tmp_path):
    write_script(
        tmp_path,
        """
        import sys

        sys.stderr.buffer.write(b"before \\xff after\\n")
        sys.stderr.flush()
        sys.exit(1)
        """,
    )
    with pytest.raises(ExecutionFailed, match="before \ufffd after"):
        run_train_script(tmp_path, ExecMode.NORMAL, RUN, 60)


def test_run_train_script_timeout_raises_the_timeout_error_with_stderr(tmp_path):
    write_script(
        tmp_path,
        """
        import sys, time

        print("still working", file=sys.stderr, flush=True)
        time.sleep(60)
        """,
    )
    with pytest.raises(ExecutionTimedOut, match="still working"):
        run_train_script(tmp_path, ExecMode.NORMAL, RUN, 2)
    assert issubclass(ExecutionTimedOut, ExecutionFailed)


def test_run_train_script_timeout_kills_child_processes(tmp_path):
    # Requirement 16.3: a worker the script started must not keep the run alive.
    write_script(
        tmp_path,
        """
        import pathlib, subprocess, sys, time

        worker = (
            "import pathlib, time\\n"
            "for i in range(300):\\n"
            "    pathlib.Path('beat.txt').write_text(str(i))\\n"
            "    time.sleep(0.1)\\n"
        )
        subprocess.Popen([sys.executable, "-c", worker])
        while not pathlib.Path("beat.txt").exists():
            time.sleep(0.05)
        time.sleep(60)
        """,
    )
    started = time.monotonic()
    with pytest.raises(ExecutionTimedOut):
        run_train_script(tmp_path, ExecMode.NORMAL, RUN, 2)
    assert time.monotonic() - started < 10
    beat = tmp_path / "beat.txt"
    assert beat.exists()
    before = beat.read_text()
    time.sleep(0.6)
    assert beat.read_text() == before  # the worker stopped writing: it is dead
