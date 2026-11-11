import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from hexperiment.checks.metric import check_metric, compute_metric
from hexperiment.errors import HexInputError
from hexperiment.models import MetricResult, Severity

# --- compute_metric (Requirement 9.1) ------------------------------------------------


def test_a_perfect_ranking_has_auc_one_and_a_reversed_one_zero() -> None:
    y = np.array([False, False, True, True])
    assert compute_metric(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert compute_metric(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_equal_scores_have_auc_one_half() -> None:
    y = np.array([False, True, False, True])
    assert compute_metric(y, np.full(4, 0.5)) == 0.5


def test_a_known_auc() -> None:
    # Positive scores 0.4 and 0.8; negative scores 0.1 and 0.5: 3 of 4 pairs ordered.
    y = np.array([False, False, True, True])
    assert compute_metric(y, np.array([0.1, 0.5, 0.4, 0.8])) == 0.75


def test_compute_metric_returns_a_python_float() -> None:
    y = np.array([False, True])
    assert type(compute_metric(y, np.array([0.2, 0.7]))) is float


@pytest.mark.parametrize("labels", [[True, True, True], [False, False, False]])
def test_a_single_class_target_raises(labels: list[bool]) -> None:
    with pytest.raises(HexInputError, match="single class") as error:
        compute_metric(np.array(labels), np.array([0.1, 0.5, 0.9]))
    assert error.value.fix


def test_scores_of_the_wrong_length_raise() -> None:
    with pytest.raises(HexInputError, match="3 value") as error:
        compute_metric(np.array([False, True, True]), np.array([0.1, 0.9]))
    assert "2" in str(error.value)


def test_compute_metric_does_not_modify_its_arguments() -> None:
    y, score = np.array([False, True, True, False]), np.array([0.3, 0.6, 0.2, 0.1])
    y_before, score_before = y.copy(), score.copy()
    compute_metric(y, score)
    assert np.array_equal(y, y_before)
    assert np.array_equal(score, score_before)


# --- check_metric: the gate and the warning (Requirements 9.2, 9.3, 17.1) -----------


def test_a_metric_below_the_threshold_fails_without_a_warning() -> None:
    result, findings = check_metric(0.62, "roc_auc", 0.8, 0.15)
    assert result == MetricResult("roc_auc", 0.62, 0.8, False)
    assert findings == ()


def test_a_metric_between_the_threshold_and_the_margin_passes_quietly() -> None:
    result, findings = check_metric(0.85, "roc_auc", 0.8, 0.15)
    assert result == MetricResult("roc_auc", 0.85, 0.8, True)
    assert findings == ()


def test_a_value_equal_to_the_threshold_passes() -> None:
    result, findings = check_metric(0.8, "roc_auc", 0.8, 0.15)
    assert result.passed is True
    assert findings == ()


def test_a_value_just_below_the_threshold_fails() -> None:
    result, _ = check_metric(np.nextafter(0.8, 0.0), "roc_auc", 0.8, 0.15)
    assert result.passed is False
    assert type(result.value) is float  # a numpy scalar is converted


def test_a_value_equal_to_threshold_plus_margin_is_not_too_good() -> None:
    threshold, margin = 0.5, 0.25  # exact in binary floating point
    result, findings = check_metric(threshold + margin, "roc_auc", threshold, margin)
    assert result.passed is True
    assert findings == ()


def test_a_value_above_threshold_plus_margin_is_too_good_to_be_true() -> None:
    result, findings = check_metric(0.99, "roc_auc", 0.8, 0.15)

    assert result == MetricResult("roc_auc", 0.99, 0.8, True)
    (finding,) = findings
    assert finding.check == "metric"
    assert finding.ghost is None
    assert finding.severity is Severity.WARNING
    assert finding.location == "metric"
    assert finding.message.startswith("too good to be true")
    assert finding.evidence == {"value": 0.99, "threshold": 0.8}
    assert finding.fix == (
        "treat this as a leak until proven otherwise: look for a column that "
        "encodes the target"
    )


def test_a_value_just_above_threshold_plus_margin_is_too_good() -> None:
    value = float(np.nextafter(0.75, 1.0))
    _, findings = check_metric(value, "roc_auc", 0.5, 0.25)
    assert len(findings) == 1


def test_a_zero_margin_flags_any_value_above_the_threshold() -> None:
    assert check_metric(0.8, "roc_auc", 0.8, 0.0)[1] == ()
    assert len(check_metric(0.81, "roc_auc", 0.8, 0.0)[1]) == 1


def warns(value: float, threshold: float, margin: float) -> bool:
    return bool(check_metric(value, "roc_auc", threshold, margin)[1])


def test_a_perfect_auc_warns_when_threshold_plus_margin_exceeds_the_cap() -> None:
    # 0.9 + 0.15 = 1.05, so the limit is 0.99 and a perfect AUC is above it.
    result, findings = check_metric(1.0, "roc_auc", 0.9, 0.15)
    assert result.passed is True
    (finding,) = findings
    assert finding.message.startswith("too good to be true")
    assert finding.evidence == {"value": 1.0, "threshold": 0.9}


def test_an_auc_of_exactly_the_cap_does_not_warn() -> None:
    assert not warns(0.99, 0.9, 0.15)
    assert warns(float(np.nextafter(0.99, 1.0)), 0.9, 0.15)


def test_the_cap_does_not_lower_a_limit_that_is_already_below_it() -> None:
    # 0.7 + 0.15 = 0.85 is below 0.99, so 0.85 stays the limit.
    limit = 0.7 + 0.15
    assert not warns(0.84, 0.7, 0.15)
    assert not warns(limit, 0.7, 0.15)
    assert warns(0.86, 0.7, 0.15)
    assert warns(float(np.nextafter(limit, 1.0)), 0.7, 0.15)


def test_evidence_is_rounded_to_six_decimals_and_the_value_is_not() -> None:
    result, findings = check_metric(0.987654321, "roc_auc", 0.8, 0.1)
    assert result.value == 0.987654321
    assert findings[0].evidence["value"] == 0.987654


# --- properties (Requirements 9.2, 9.3, 17.2) ----------------------------------------


@given(
    value=st.one_of(
        st.floats(0.0, 1.0),
        st.sampled_from([0.99, float(np.nextafter(0.99, 1.0)), 0.995, 1.0]),
    ),
    threshold=st.one_of(st.floats(0.01, 0.99), st.sampled_from([0.8, 0.9, 0.95])),
    margin=st.one_of(st.floats(0.0, 0.5), st.sampled_from([0.15, 0.25])),
)
def test_the_gate_and_the_warning_follow_the_comparisons(
    value: float, threshold: float, margin: float
) -> None:
    result, findings = check_metric(value, "roc_auc", threshold, margin)
    assert result.passed == (value >= threshold)
    assert bool(findings) == (value > min(threshold + margin, 0.99))
    assert result.value == value
    assert result.threshold == threshold


@given(
    rows=st.lists(
        st.tuples(st.booleans(), st.floats(0.0, 1.0)), min_size=2, max_size=40
    ),
    order=st.randoms(use_true_random=False),
)
def test_reordering_rows_never_changes_the_metric(rows, order) -> None:
    labels = [label for label, _ in rows]
    assume(any(labels) and not all(labels))
    y = np.array(labels)
    score = np.array([s for _, s in rows])
    permutation = list(range(len(rows)))
    order.shuffle(permutation)

    expected = check_metric(compute_metric(y, score), "roc_auc", 0.6, 0.15)
    actual = check_metric(
        compute_metric(y[permutation], score[permutation]), "roc_auc", 0.6, 0.15
    )
    assert actual == expected
