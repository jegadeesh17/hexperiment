"""The Doppelganger: identical rows in both train and test (Requirement 5)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from hexperiment.models import CheckResult, Finding, Outcome, Severity

CHECK = "row_overlap"
GHOST = "The Doppelganger"
LOCATION = "train vs test"
FIX = "drop exact duplicates before splitting"

_EVIDENCE_DECIMALS = 6
_INDICATOR = "__hexperiment_overlap__"


def check_row_overlap(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    target: str,
    max_fraction: float,
) -> CheckResult:
    """Count test rows identical to a training row over ``features`` plus ``target``.

    A test row counts once however many training rows equal it. Missing values
    compare equal to missing values. The result is an ``error`` finding when the
    overlapping fraction of test rows is greater than ``max_fraction``; a
    fraction equal to ``max_fraction`` passes (Requirements 5.1, 5.2). The count
    depends only on the two sets of rows, never on their order (Requirement 5.3).
    """
    columns = list(dict.fromkeys([*features, target]))
    overlapping = _count_overlapping(train[columns], test[columns], columns)
    n_test = len(test)
    fraction = overlapping / n_test if n_test else 0.0
    if fraction <= max_fraction:
        return CheckResult(CHECK, GHOST, Outcome.PASSED)
    finding = Finding(
        check=CHECK,
        ghost=GHOST,
        severity=Severity.ERROR,
        location=LOCATION,
        message=(
            f"{overlapping} of {n_test} test row(s) are identical, over the "
            "features and the target, to a row in train"
        ),
        evidence={
            "rows": overlapping,
            "test_rows": n_test,
            "fraction": round(fraction, _EVIDENCE_DECIMALS),
        },
        fix=FIX,
    )
    return CheckResult(CHECK, GHOST, Outcome.FAILED, (finding,))


def _count_overlapping(
    train_rows: pd.DataFrame, test_rows: pd.DataFrame, columns: list[str]
) -> int:
    """Left-merge test onto the distinct training rows and count the matches.

    Deduplicating train first keeps one output row per test row, so a test row is
    never counted twice. The merge is exact (no hashing) and pandas matches a
    missing value with a missing value.
    """
    merged = test_rows.merge(
        train_rows.drop_duplicates(), on=columns, how="left", indicator=_INDICATOR
    )
    return int((merged[_INDICATOR] == "both").sum())
