"""The Doppelganger: identical rows in both train and test (Requirement 5)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from hexperiment.errors import HexInputError
from hexperiment.models import CheckResult, Finding, Outcome, Severity

CHECK = "row_overlap"
GHOST = "The Doppelganger"
LOCATION = "train vs test"
FIX = "after the split, drop every test row that also appears in the training rows"

_EVIDENCE_DECIMALS = 6
_INDICATOR = "__hexperiment_overlap__"
_PAD = "__hexperiment_pad__"


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
    missing value with a missing value. A column whose dtype differs between the
    two frames, or that is categorical, is compared as object values, so ``1``
    equals ``1.0`` and a category equals the same text. Values that cannot be
    compared raise ``HexInputError`` (Requirement 5.1).
    """
    left, right = _comparable_frames(train_rows, test_rows, columns)
    try:
        merged = right.merge(
            left.drop_duplicates(), on=columns, how="left", indicator=_INDICATOR
        )
    except (TypeError, ValueError) as error:
        raise _uncomparable(left, right, columns, error) from None
    return int((merged[_INDICATOR] == "both").sum())


def _comparable_frames(
    train_rows: pd.DataFrame, test_rows: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Key frames in which both sides of each column can be merged safely."""
    left: dict[str, pd.Series] = {}
    right: dict[str, pd.Series] = {}
    for column in columns:
        a = train_rows[column].reset_index(drop=True)
        b = test_rows[column].reset_index(drop=True)
        _require_same_timezone_kind(column, a, b)
        if a.dtype != b.dtype or _is_categorical(a) or _is_categorical(b):
            a, b = a.astype(object), b.astype(object)
        left[column], right[column] = a, b
    return pd.DataFrame(left), pd.DataFrame(right)


def _is_categorical(values: pd.Series) -> bool:
    return isinstance(values.dtype, pd.CategoricalDtype)


def _require_same_timezone_kind(column: str, a: pd.Series, b: pd.Series) -> None:
    """A naive datetime never equals an aware one, so the pair is an input error.

    Treating them as different would hide a duplicate row behind a timezone
    mismatch, so the check refuses to guess which timezone the naive side means.
    """
    aware = [isinstance(v.dtype, pd.DatetimeTZDtype) for v in (a, b)]
    datetimes = [getattr(v.dtype, "kind", None) == "M" for v in (a, b)]
    if all(datetimes) and aware[0] != aware[1]:
        raise HexInputError(
            f"column {column!r} is timezone-aware in one of train and test and "
            f"timezone-naive in the other (train dtype {a.dtype}, test dtype "
            f"{b.dtype}), so its values cannot be compared.",
            f"Give {column!r} the same timezone handling in both frames, for "
            "example with pd.to_datetime(..., utc=True) before the split.",
        )


def _uncomparable(
    left: pd.DataFrame, right: pd.DataFrame, columns: list[str], error: Exception
) -> HexInputError:
    """Name the key column whose values cannot be compared, and both dtypes."""
    suspects = [c for c in columns if not _can_compare(left[c], right[c])]
    named = suspects or columns
    described = "; ".join(
        f"{c!r} (train dtype {left[c].dtype}, test dtype {right[c].dtype})"
        for c in named
    )
    return HexInputError(
        f"the values of column(s) {described} cannot be compared between train "
        f"and test ({error}).",
        "Convert the column to one plain type in both frames (numbers, text or "
        "datetimes) before the split, or leave it out of features.",
    )


def _can_compare(left: pd.Series, right: pd.Series) -> bool:
    """Whether one column merges on its own, with a pad column for the row path."""
    if not (_hashable(left) and _hashable(right)):
        return False
    try:
        pad = {_PAD: 0}
        one = pd.DataFrame({"key": left, **pad}).drop_duplicates()
        pd.DataFrame({"key": right, **pad}).merge(one, on=["key", _PAD], how="left")
    except (TypeError, ValueError):
        return False
    return True


def _hashable(values: pd.Series) -> bool:
    if values.dtype != object:
        return True
    try:
        for value in values:
            hash(value)
    except TypeError:
        return False
    return True
