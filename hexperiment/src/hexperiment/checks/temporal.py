"""The Prophet: training rows from after test rows (Requirement 7)."""

from __future__ import annotations

import pandas as pd

from hexperiment.data import parse_times
from hexperiment.errors import HexInputError
from hexperiment.models import CheckResult, Finding, Outcome, Severity

CHECK = "temporal"
GHOST = "The Prophet"
FIX = "split by time with hexperiment.split"


def check_temporal(
    train: pd.DataFrame, test: pd.DataFrame, time_column: str | None
) -> CheckResult:
    """Compare the latest training timestamp with the earliest test timestamp.

    Without a ``time_column`` the check is ``not applicable`` (Requirement 7.3).
    Otherwise it reports an ``error`` when any training timestamp is later than
    the earliest test timestamp; equal timestamps are not a leak (Requirements
    7.1, 7.2). Timestamps are compared as instants in UTC, so mixed timezones
    are comparable.
    """
    if time_column is None:
        return CheckResult(CHECK, GHOST, Outcome.NOT_APPLICABLE)
    train_times = _times(train, time_column, "the train frame")
    test_times = _times(test, time_column, "the test frame")
    latest_train = train_times.max()
    earliest_test = test_times.min()
    if not latest_train > earliest_test:
        return CheckResult(CHECK, GHOST, Outcome.PASSED)
    earlier = int((test_times < latest_train).sum())
    finding = Finding(
        check=CHECK,
        ghost=GHOST,
        severity=Severity.ERROR,
        location=time_column,
        message=(
            f"{earlier} test row(s) are earlier than the latest training row "
            f"({latest_train.isoformat()})"
        ),
        evidence={
            "test_rows_earlier": earlier,
            "latest_train": latest_train.isoformat(),
            "earliest_test": earliest_test.isoformat(),
        },
        fix=FIX,
    )
    return CheckResult(CHECK, GHOST, Outcome.FAILED, (finding,))


def _times(frame: pd.DataFrame, column: str, frame_name: str) -> pd.Series:
    if column not in frame.columns:
        raise HexInputError(
            f"time column {column!r} is missing from {frame_name}.",
            f"Keep {column!r} in the data, or remove split.time_column from hex.toml.",
        )
    return parse_times(frame[column], column, frame_name)
