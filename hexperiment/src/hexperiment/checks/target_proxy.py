"""The Possession: one feature alone predicts the target (Requirement 6)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from hexperiment.errors import HexInputError
from hexperiment.models import CheckResult, Finding, Outcome, Severity

CHECK = "target_proxy"
GHOST = "The Possession"
FIX = (
    "check whether this column is recorded after the outcome; if so, drop it"
)

# A numeric feature with at most this many distinct non-missing test values also
# gets the category score (Requirement 6.1).
MAX_CATEGORY_LEVELS = 50

_EVIDENCE_DECIMALS = 6

# numpy dtype kinds, which pandas extension dtypes also report.
_KIND_BOOL = "b"
_KIND_DATETIME = "M"
_KIND_TIMEDELTA = "m"
_KINDS_NUMERIC = frozenset("iuf")
_VALUE_KINDS = frozenset(
    {_KIND_BOOL, _KIND_DATETIME, _KIND_TIMEDELTA, *_KINDS_NUMERIC}
)


def standalone_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature: str,
    target: str,
    y_true_train: NDArray[np.bool_],
    y_true_test: NDArray[np.bool_],
) -> float | None:
    """Score how well ``feature`` alone predicts the target on the test rows.

    Numeric, boolean, datetime and timedelta features get the value score:
    ``max(AUC, 1 - AUC)`` of the raw value against the test target, computed once
    with missing values ranked lowest and once ranked highest, keeping the higher
    result. Other features get the category score: the AUC of the training-set
    positive rate of each category applied to the test rows, where unseen
    categories and missing values get the training positive rate. A numeric
    feature with at most ``MAX_CATEGORY_LEVELS`` distinct non-missing test values
    gets both scores and keeps the higher (Requirement 6.1).

    Returns ``None`` only when the feature has a single unique value in the test
    rows, counting a missing value as a value (Requirement 6.3). ``target`` names
    the label column; the labels themselves arrive as ``y_true_train`` and
    ``y_true_test``. A test target with one class raises ``HexInputError``,
    because no feature may be skipped for that reason (Requirements 4.3, 6.3).
    """
    test_values = test[feature]
    if test_values.nunique(dropna=False) == 1:
        return None
    _require_two_classes(y_true_test, target)
    kind = _kind(test_values)
    values = test_values
    if kind is None:
        dates = _iso_datetimes(train[feature], test_values)
        if dates is not None:
            kind, values = _KIND_DATETIME, dates
    scores: list[float] = []
    if kind is not None:
        scores.append(_value_score(values, kind, y_true_test))
    if kind is None or _few_levels(values, kind):
        scores.append(
            _category_score(train[feature], test_values, y_true_train, y_true_test)
        )
    return max(scores)


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


def _require_two_classes(y_true_test: NDArray[np.bool_], target: str) -> None:
    if y_true_test.any() and not y_true_test.all():
        return
    raise HexInputError(
        f"the test target {target!r} has a single class, so no feature can be "
        "scored against it.",
        "Make sure the test rows hold both classes, and that [target] positive "
        "in hex.toml is a value that occurs in the target column.",
    )


# --- the value score ----------------------------------------------------------------


def _kind(values: pd.Series) -> str | None:
    """Return the dtype kind when the feature gets the value score, else ``None``.

    Categorical, string, object and complex features have other kinds and get the
    category score only.
    """
    dtype = values.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return None
    kind = getattr(dtype, "kind", None)
    return kind if kind in _VALUE_KINDS else None


def _few_levels(values: pd.Series, kind: str) -> bool:
    """True for a numeric feature with at most ``MAX_CATEGORY_LEVELS`` levels."""
    return kind in _KINDS_NUMERIC and values.nunique() <= MAX_CATEGORY_LEVELS


def _is_text(values: pd.Series) -> bool:
    """True for an object or string column whose non-missing values are all text."""
    if values.dtype != object and not pd.api.types.is_string_dtype(values.dtype):
        return False
    return pd.api.types.infer_dtype(values, skipna=True) in ("string", "empty")


def _iso_datetimes(
    train_values: pd.Series, test_values: pd.Series
) -> pd.Series | None:
    """Parse a text feature as ISO 8601 dates, or return ``None`` if it is not one.

    ``load`` reads CSV without parsing dates, so a date column arrives as text. It
    counts as a datetime feature when every non-missing value in train and test
    parses with ``format="ISO8601"`` and ``utc=True``; one value that does not parse
    keeps it a text feature (Requirement 6.1). All values go through one parse call,
    which raises at the first value that fails, so an ID column costs one attempt.
    The result has the index of ``test_values``, with ``NaT`` where it is missing.
    """
    if not (_is_text(test_values) and _is_text(train_values)):
        return None
    test_present = test_values.dropna()
    if test_present.empty:
        return None
    combined = np.concatenate(
        [
            test_present.to_numpy(dtype=object),
            train_values.dropna().to_numpy(dtype=object),
        ]
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(combined, utc=True, format="ISO8601")
    except (ValueError, TypeError, OverflowError):
        return None
    dates = pd.Series(parsed[: len(test_present)], index=test_present.index)
    return dates.reindex(test_values.index)


def _value_score(
    values: pd.Series, kind: str, y_true_test: NDArray[np.bool_]
) -> float:
    """``max(AUC, 1 - AUC)`` of the raw values, with missing values ranked both ways.

    AUC depends only on the order of the values, so the values are replaced by
    their ranks first: ties stay ties, infinite values cannot fail, and integer
    nanosecond values lose no precision. Missing values form one tied group,
    placed below every value and then above every value; the higher score wins
    (Requirement 6.1).
    """
    present = values.notna().to_numpy(dtype=bool)
    ranks = _sortable(values[present], kind).rank(method="average")
    ranks = ranks.to_numpy(dtype=np.float64)
    best = 0.0
    for missing_rank in (0.0, float(len(ranks) + 1)):
        full = np.full(len(values), missing_rank)
        full[present] = ranks
        auc = float(roc_auc_score(y_true_test, full))
        best = max(best, auc, 1.0 - auc)
    return best


def _sortable(present: pd.Series, kind: str) -> pd.Series:
    """Map non-missing values to a numpy dtype that keeps their order exactly.

    Datetimes go to UTC and then to ``int64`` nanoseconds, timedeltas to ``int64``
    nanoseconds, booleans to 0 and 1 (Requirement 6.1).
    """
    if kind == _KIND_DATETIME:
        if present.dt.tz is not None:
            present = present.dt.tz_convert("UTC").dt.tz_localize(None)
        return present.astype("datetime64[ns]").astype(np.int64)
    if kind == _KIND_TIMEDELTA:
        return present.astype("timedelta64[ns]").astype(np.int64)
    if kind == _KIND_BOOL:
        return present.astype(np.int64)
    if kind == "f":
        return present.astype(np.float64)
    return present.astype(getattr(present.dtype, "numpy_dtype", present.dtype))


# --- the category score -------------------------------------------------------------


def _category_score(
    train_values: pd.Series,
    test_values: pd.Series,
    y_true_train: NDArray[np.bool_],
    y_true_test: NDArray[np.bool_],
) -> float:
    """AUC of the training positive rate of each category, applied to test rows.

    Rates are integer ratios (positives over count), so they are exact and a
    reordering of the rows cannot change a tie (Requirement 6.4).
    """
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
