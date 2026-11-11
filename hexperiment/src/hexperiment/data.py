"""Data loading and splitting: the hash split and the time split.

The split decides which rows are test rows. It never looks at feature values, so
the same rows are test rows in every execution of a training script
(Requirements 1.2 to 1.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hexperiment.config import SplitConfig
from hexperiment.errors import HexInputError

_GOLDEN_GAMMA = 0x9E3779B97F4A7C15
_MIX_1 = 0xBF58476D1CE4E5B9
_MIX_2 = 0x94D049BB133111EB
_UINT64_MODULUS = 1 << 64
_FRAME_NAME = "the frame being split"


@dataclass(frozen=True, slots=True)
class TimeCutoff:
    """First test row of a time split, in ``(time, row id)`` order.

    A row is a test row when its ``(time, row id)`` is at or after the cutoff.
    """

    time: pd.Timestamp  # UTC
    row_id: int


def hash_fraction(row_ids: NDArray[np.int64], seed: int) -> NDArray[np.float64]:
    """Map each row id to a number in ``[0, 1)`` that depends only on id and seed.

    Uses the SplitMix64 finalizer in ``uint64`` arithmetic, so the result is the
    same in every process (Python's string ``hash`` is never involved).
    """
    start = (seed + 1) * _GOLDEN_GAMMA % _UINT64_MODULUS
    ids = np.asarray(row_ids, dtype=np.int64).astype(np.uint64)
    with np.errstate(over="ignore"):
        z = ids + np.uint64(start)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(_MIX_1)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(_MIX_2)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def parse_times(series: pd.Series, column: str, frame_name: str) -> pd.Series:
    """Parse a time column to UTC timestamps.

    Timezone-naive and timezone-aware values become comparable because everything
    is converted with ``utc=True``. Unparseable or missing values raise.
    """
    try:
        parsed = pd.to_datetime(series, utc=True)
    except (ValueError, TypeError) as error:
        raise HexInputError(
            f"time column {column!r} in {frame_name} cannot be parsed as "
            f"datetimes ({error}).",
            f"Make every value of {column!r} a date or timestamp, or remove "
            "split.time_column from hex.toml.",
        ) from None
    missing = int(parsed.isna().sum())
    if missing:
        raise HexInputError(
            f"time column {column!r} in {frame_name} has {missing} missing or "
            "unparseable value(s).",
            "Fill or drop those rows before loading, or remove split.time_column "
            "from hex.toml.",
        )
    return parsed


def compute_cutoff(df: pd.DataFrame, split: SplitConfig) -> TimeCutoff | None:
    """Return the time cutoff for ``df``, or ``None`` without a ``time_column``.

    Rows are ordered by ``(time, row id)``; the last
    ``min(n - 1, max(1, floor(n * fraction + 0.5)))`` rows are the test rows, and
    the first of them is the cutoff. Equal timestamps are ordered by row id, so
    the result is deterministic (Requirement 1.3).
    """
    if split.time_column is None:
        return None
    times = _time_values(df, split.time_column)
    return _cutoff_from_times(times, _row_ids(df), split.test_fraction)


def test_mask(
    df: pd.DataFrame, split: SplitConfig, cutoff: TimeCutoff | None = None
) -> NDArray[np.bool_]:
    """Return a boolean array that is true for the test rows of ``df``.

    Without a ``time_column`` membership depends only on the row id, the seed and
    the fraction (Requirement 1.4). With one, a row is a test row when its
    ``(time, row id)`` is at or after the cutoff. Pass the cutoff stored at load
    time so that rows removed before the split cannot move the boundary; without
    one, the cutoff is computed from ``df``.
    """
    ids = _row_ids(df)
    if split.time_column is None:
        return hash_fraction(ids, split.seed) < split.test_fraction
    times = _time_values(df, split.time_column)
    if cutoff is None:
        cutoff = _cutoff_from_times(times, ids, split.test_fraction)
    after = (times > cutoff.time).to_numpy(dtype=bool)
    at = (times == cutoff.time).to_numpy(dtype=bool)
    return after | (at & (ids >= cutoff.row_id))


test_mask.__test__ = False  # type: ignore[attr-defined]  # not a pytest test


def split_frame(
    df: pd.DataFrame, split: SplitConfig, cutoff: TimeCutoff | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train, test)`` copies of ``df`` that keep the row-id index."""
    mask = test_mask(df, split, cutoff)
    return df.loc[~mask].copy(), df.loc[mask].copy()


# --- helpers -------------------------------------------------------------------------


def _row_ids(df: pd.DataFrame) -> NDArray[np.int64]:
    if not pd.api.types.is_integer_dtype(df.index.dtype):
        raise HexInputError(
            f"the index of {_FRAME_NAME} has dtype {df.index.dtype}, but row ids "
            "must be integers.",
            "Split the frame returned by hexperiment.load without resetting or "
            "replacing its index.",
        )
    return df.index.to_numpy(dtype=np.int64)


def _time_values(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise HexInputError(
            f"time column {column!r} is missing from {_FRAME_NAME}.",
            f"Keep {column!r} in the data, or remove split.time_column from hex.toml.",
        )
    return parse_times(df[column], column, _FRAME_NAME)


def _cutoff_from_times(
    times: pd.Series, ids: NDArray[np.int64], fraction: float
) -> TimeCutoff:
    n = len(times)
    if n < 2:
        raise HexInputError(
            f"cannot make a time split from {n} row(s); a split needs at least 2.",
            "Load more data, or check that no filtering removed the rows.",
        )
    n_test = min(n - 1, max(1, math.floor(n * fraction + 0.5)))
    order = np.lexsort((ids, pd.DatetimeIndex(times).asi8))
    first_test = int(order[n - n_test])
    return TimeCutoff(time=times.iloc[first_test], row_id=int(ids[first_test]))
