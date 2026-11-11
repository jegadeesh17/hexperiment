"""The metric gate and the "too good to be true" warning (Requirement 9)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from hexperiment.errors import HexInputError
from hexperiment.models import Finding, MetricResult, Severity

CHECK = "metric"
LOCATION = "metric"
FIX = (
    "treat this as a leak until proven otherwise: look for a column that encodes "
    "the target"
)

_EVIDENCE_DECIMALS = 6

# A metric above this value is too good whatever the threshold (Requirement 9.3).
TOO_GOOD_CAP = 0.99


def compute_metric(y_true: NDArray[np.bool_], y_score: NDArray[np.float64]) -> float:
    """Return the ROC AUC of ``y_score`` against ``y_true`` (Requirement 9.1)."""
    if len(y_true) != len(y_score):
        raise HexInputError(
            f"y_true has {len(y_true)} value(s) but y_score has {len(y_score)}.",
            "Pass one score per test row, in the order of the test frame.",
        )
    if not (y_true.any() and not y_true.all()):
        raise HexInputError(
            "the test target has a single class, so ROC AUC is undefined.",
            "Make sure the test rows hold both classes, and that [target] positive "
            "in hex.toml is a value that occurs in the target column.",
        )
    return float(roc_auc_score(y_true, y_score))


def check_metric(
    value: float, name: str, threshold: float, too_good_margin: float
) -> tuple[MetricResult, tuple[Finding, ...]]:
    """Apply the threshold and report a metric far above it as a warning.

    The metric passes when ``value >= threshold`` (Requirement 9.2). A value above
    the lower of ``threshold + too_good_margin`` and ``TOO_GOOD_CAP`` also yields a
    ``warning`` (Requirement 9.3).
    """
    value = float(value)
    result = MetricResult(
        name=name, value=value, threshold=threshold, passed=value >= threshold
    )
    limit = min(threshold + too_good_margin, TOO_GOOD_CAP)
    if not value > limit:
        return result, ()
    finding = Finding(
        check=CHECK,
        ghost=None,
        severity=Severity.WARNING,
        location=LOCATION,
        message=(
            f"too good to be true: {name} {value:.4f} is above {limit:.4f}, the "
            f"most that the threshold {threshold:g} and the margin allow"
        ),
        evidence={
            "value": round(value, _EVIDENCE_DECIMALS),
            "threshold": round(threshold, _EVIDENCE_DECIMALS),
        },
        fix=FIX,
    )
    return result, (finding,)
