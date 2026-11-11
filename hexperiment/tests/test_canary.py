import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hexperiment import HexInputError
from hexperiment.canary import (
    SCORE_TOLERANCE,
    CanaryOutcome,
    LoadMarker,
    Snapshot,
    build_snapshot,
    compare_snapshots,
    decide_canary,
    hint_locations,
    skipped_outcome,
)
from hexperiment.config import WAIVABLE_CHECKS, ExecMode
from hexperiment.models import (
    CheckResult,
    DataAuditResult,
    Finding,
    MetricResult,
    Outcome,
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


# --- decide_canary: missing y_score_train and hint lines --------------------------


def weaker_warnings(outcome: CanaryOutcome) -> list[Finding]:
    return [f for f in outcome.findings if "weaker" in f.message]


@pytest.mark.parametrize(
    ("canary_change", "control_change", "expected"),
    [
        (False, False, Outcome.PASSED),
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
    (warning,) = weaker_warnings(outcome)
    assert warning.check == "canary"
    assert warning.ghost is None
    assert warning.severity is Severity.WARNING
    assert "y_score_train" in warning.fix
    if expected is Outcome.FAILED:
        spirit = next(f for f in outcome.findings if f.ghost == "The Peeking Spirit")
        assert spirit.evidence["largest_change"] is None


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
