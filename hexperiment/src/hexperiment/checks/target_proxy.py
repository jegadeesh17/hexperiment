"""The Possession: one feature alone predicts the target (Requirement 6)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from hexperiment.models import CheckResult, Finding, Outcome, Severity

CHECK = "target_proxy"
GHOST = "The Possession"
FIX = (
    "check whether this column is recorded after the outcome; if so, drop it"
)

_EVIDENCE_DECIMALS = 6


def standalone_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature: str,
    target: str,
    y_true_train: NDArray[np.bool_],
    y_true_test: NDArray[np.bool_],
) -> float | None:
    """Score how well ``feature`` alone predicts the target on the test rows.

    Numeric and boolean features: ``max(AUC, 1 - AUC)`` of the raw value against
    the test target, with rows that have a missing value left out. Other dtypes:
    the AUC of the training-set positive rate of each category applied to the test
    rows; unseen categories and missing values get the overall training positive
    rate (Requirement 6.1). Returns ``None`` when the feature has a single unique
    value in the test rows, or when fewer than two classes are left to rank
    (Requirement 6.3). ``target`` names the label column; the labels themselves
    arrive as ``y_true_train`` and ``y_true_test``.
    """
    test_values = test[feature]
    if test_values.nunique(dropna=False) < 2:
        return None
    train_values = train[feature]
    if _is_numeric(train_values) and _is_numeric(test_values):
        return _numeric_score(test_values, y_true_test)
    return _category_score(train_values, test_values, y_true_train, y_true_test)


def check_target_proxy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    target: str,
    y_true_train: NDArray[np.bool_],
    y_true_test: NDArray[np.bool_],
    auc_max: float,
) -> CheckResult:
    """Report an ``error`` for each feature whose standalone score is ``>= auc_max``.

    A score equal to ``auc_max`` is flagged (Requirement 6.2). The result does not
    depend on the order of the rows (Requirement 6.4).
    """
    findings = []
    for feature in dict.fromkeys(features):
        score = standalone_score(
            train, test, feature, target, y_true_train, y_true_test
        )
        if score is not None and score >= auc_max:
            findings.append(_finding(feature, score, auc_max))
    outcome = Outcome.FAILED if findings else Outcome.PASSED
    return CheckResult(CHECK, GHOST, outcome, tuple(findings))


def _finding(feature: str, score: float, auc_max: float) -> Finding:
    return Finding(
        check=CHECK,
        ghost=GHOST,
        severity=Severity.ERROR,
        location=str(feature),
        message=f"{feature} alone predicts the target with AUC {score:.4f}",
        evidence={
            "score": round(score, _EVIDENCE_DECIMALS),
            "threshold": round(auc_max, _EVIDENCE_DECIMALS),
        },
        fix=FIX,
    )


def _is_numeric(values: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(values.dtype)) and not (
        pd.api.types.is_complex_dtype(values.dtype)
    )


def _has_both_classes(y_true: NDArray[np.bool_]) -> bool:
    return bool(y_true.any() and not y_true.all())


def _numeric_score(
    values: pd.Series, y_true_test: NDArray[np.bool_]
) -> float | None:
    """``max(AUC, 1 - AUC)`` of the raw values, leaving out missing values.

    AUC depends only on the order of the values, so the values are replaced by
    their ranks first: ties stay ties and infinite values cannot fail.
    """
    x = values.to_numpy(dtype=np.float64, na_value=np.nan)
    present = ~np.isnan(x)
    y = y_true_test[present]
    if not _has_both_classes(y):
        return None
    ranks = pd.Series(x[present]).rank(method="average").to_numpy()
    auc = float(roc_auc_score(y, ranks))
    return max(auc, 1.0 - auc)


def _category_score(
    train_values: pd.Series,
    test_values: pd.Series,
    y_true_train: NDArray[np.bool_],
    y_true_test: NDArray[np.bool_],
) -> float | None:
    """AUC of the training positive rate of each category, applied to test rows.

    Rates are integer ratios (positives over count), so they are exact and a
    reordering of the rows cannot change a tie (Requirement 6.4).
    """
    if not _has_both_classes(y_true_test):
        return None
    training = pd.DataFrame(
        {
            "value": train_values.astype(object).to_numpy(),
            "positive": y_true_train.astype(np.int64),
        }
    )
    totals = training.groupby("value", dropna=True, sort=False)["positive"].agg(
        ["sum", "count"]
    )
    rates = totals["sum"] / totals["count"]
    base_rate = int(y_true_train.sum()) / len(y_true_train)
    mapped = test_values.astype(object).map(rates).fillna(base_rate)
    return float(roc_auc_score(y_true_test, mapped.to_numpy(dtype=np.float64)))
