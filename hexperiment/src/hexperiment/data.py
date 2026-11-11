"""Data loading, splitting and the canary scramble.

The split decides which rows are test rows. It never looks at feature values, so
the same rows are test rows in every execution of a training script
(Requirements 1.2 to 1.4). In canary mode the values in those rows are replaced
by a tilted resample (Requirement 1.5).
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
_SCRAMBLE_FRAME_NAME = "the frame being scrambled"

# Fixed second seed word of the canary generator, so the scramble draws are
# independent of any other use of ``split.seed``. The bytes spell "HEXC".
CANARY_SALT = 0x48455843


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


def scramble_test_rows(
    df: pd.DataFrame,
    mask: NDArray[np.bool_] | pd.Series,
    time_column: str | None,
    seed: int,
) -> pd.DataFrame:
    """Return a copy of ``df`` with the test rows replaced by a tilted resample.

    Only cells of rows where ``mask`` is true change, in every column except
    ``time_column``. The index, the row count, the dtypes, the training rows and
    the time column are untouched. Columns are tilted one by one, in column
    order, by ``tilt_column`` with one generator
    ``np.random.default_rng([seed, CANARY_SALT])``, and each column's test values
    are passed to the tilt in row-id order, so the result depends only on the
    frame, the mask and the seed (Requirement 1.5).

    ``mask`` is a boolean array with one entry per row, or a boolean Series that
    is aligned to ``df`` by index label. Any other mask raises ``HexInputError``,
    as does a ``time_column`` that ``df`` does not have.
    """
    if time_column is not None:
        _require_time_column(df, time_column, _SCRAMBLE_FRAME_NAME)
    in_test = _mask_values(df, mask)
    ids = _row_ids(df, _SCRAMBLE_FRAME_NAME)
    positions = np.flatnonzero(in_test)
    positions = positions[np.argsort(ids[positions], kind="stable")]
    rng = np.random.default_rng([seed, CANARY_SALT])
    out = df.copy()
    for location, name in enumerate(df.columns):
        if name == time_column:
            continue
        column = df.iloc[:, location]
        tilted = tilt_column(column.iloc[positions], rng)
        out.isetitem(location, _write_at(column, positions, tilted))
    return out


def tilt_column(values: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Return tilted replacements for the test values of one column.

    The result has the index, length and dtype of ``values``, and every written
    value is one that ``values`` already holds (Requirement 1.5):

    - missing values keep their positions and are never drawn;
    - one distinct value: unchanged;
    - two distinct values: every occurrence of one becomes the other;
    - three or more: each non-missing position, in the order given, gets a draw
      with replacement from the upper half of the values (numeric dtypes: by
      value, widened downward to at least two distinct values; other dtypes: the
      most frequent ``ceil(k / 2)`` categories, ties by ascending ``str``). If all
      draws are equal, the first one is replaced by the highest-ranked pool value
      that differs, so the column never becomes constant (Requirement 17.2).

    Values that cannot be hashed (lists, dicts, arrays) raise ``HexInputError``
    naming the series.
    """
    sources = _tilt_sources(values, rng)
    if sources is None:
        return values.copy()
    return values.take(sources).set_axis(values.index)


# --- helpers -------------------------------------------------------------------------


def _tilt_sources(
    values: pd.Series, rng: np.random.Generator
) -> NDArray[np.intp] | None:
    """Return, for each position of ``values``, the position its new value comes from.

    ``None`` means the column is unchanged (at most one distinct value). Taking
    values by position keeps the dtype (including categories) exactly and makes
    an unseen value impossible. Only ``tilt_column`` calls this.
    """
    present = np.flatnonzero(values.notna().to_numpy(dtype=bool))
    kept = values.iloc[present]
    numeric = pd.api.types.is_numeric_dtype(values.dtype)
    try:
        codes, ranks, k = _value_ranks(kept, numeric)
    except TypeError:
        label = "the series" if values.name is None else f"column {values.name!r}"
        raise HexInputError(
            f"{label} holds values that cannot be hashed (for example lists, "
            "dicts or arrays), so the canary scramble cannot tilt it.",
            "Convert those values to strings or tuples, or drop the column "
            "before training.",
        ) from None
    if k <= 1:
        return None
    sources = np.arange(len(values))
    if k == 2:
        first_of_rank = [int(np.argmax(ranks == r)) for r in (0, 1)]
        swapped = np.where(ranks == 0, first_of_rank[1], first_of_rank[0])
        sources[present] = present[swapped]
        return sources
    pool = _numeric_pool(codes, k) if numeric else np.flatnonzero(ranks < -(-k // 2))
    draws = pool[rng.integers(0, len(pool), size=len(present))]
    drawn = ranks[draws]
    if np.all(drawn == drawn[0]):
        differing = pool[ranks[pool] != drawn[0]]
        draws[0] = differing[np.argmin(ranks[differing])]
    sources[present] = present[draws]
    return sources


def _value_ranks(
    kept: pd.Series, numeric: bool
) -> tuple[NDArray[np.intp], NDArray[np.intp], int]:
    """Rank the distinct non-missing values; rank 0 is the highest-ranked.

    Returns ``(codes, ranks, k)``: for numeric dtypes ``codes`` orders the values
    ascending and rank 0 is the largest value; for other dtypes rank 0 is the most
    frequent category, ties broken by ascending ``str(value)``. Both arrays are
    aligned with ``kept``.
    """
    codes, uniques = pd.factorize(kept, sort=numeric)
    k = len(uniques)
    if numeric:
        return codes, (k - 1) - codes, k
    counts = np.bincount(codes, minlength=k)
    order = sorted(range(k), key=lambda i: (-int(counts[i]), str(uniques[i])))
    rank_of_code = np.empty(k, dtype=np.intp)
    rank_of_code[order] = np.arange(k)
    return codes, rank_of_code[codes], k


def _numeric_pool(codes: NDArray[np.intp], k: int) -> NDArray[np.intp]:
    """Positions of the upper half of the sorted values, with repeats.

    The pool starts at ``min(floor(m / 2), index of the first copy of the
    second-largest distinct value)`` in the ascending list, so it always holds at
    least two distinct values.
    """
    ascending = np.argsort(codes, kind="stable")
    second_largest_start = int(np.count_nonzero(codes < k - 2))
    return ascending[min(len(codes) // 2, second_largest_start) :]


def _row_ids(df: pd.DataFrame, frame_name: str = _FRAME_NAME) -> NDArray[np.int64]:
    if not pd.api.types.is_integer_dtype(df.index.dtype):
        raise HexInputError(
            f"the index of {frame_name} has dtype {df.index.dtype}, but row ids "
            "must be integers.",
            "Use the frame returned by hexperiment.load without resetting or "
            "replacing its index.",
        )
    return df.index.to_numpy(dtype=np.int64)


def _require_time_column(df: pd.DataFrame, column: str, frame_name: str) -> None:
    if column not in df.columns:
        raise HexInputError(
            f"time column {column!r} is missing from {frame_name}.",
            f"Keep {column!r} in the data, or remove split.time_column from hex.toml.",
        )


def _time_values(df: pd.DataFrame, column: str) -> pd.Series:
    _require_time_column(df, column, _FRAME_NAME)
    return parse_times(df[column], column, _FRAME_NAME)


def _mask_values(
    df: pd.DataFrame, mask: NDArray[np.bool_] | pd.Series
) -> NDArray[np.bool_]:
    """Check a scramble mask and return it as a boolean array in ``df`` row order."""
    hint = "Compute the mask with test_mask on the same frame that is scrambled."
    if isinstance(mask, pd.Series):
        if not pd.api.types.is_bool_dtype(mask.dtype) or mask.isna().any():
            raise HexInputError(
                f"the test mask is a Series of dtype {mask.dtype}, but it must "
                "hold True or False for every row.",
                hint,
            )
        return _align_mask(df, mask).to_numpy(dtype=bool)
    array = np.asarray(mask)
    if array.dtype != np.bool_:
        raise HexInputError(
            f"the test mask has dtype {array.dtype}, but it must be boolean.",
            hint,
        )
    if array.shape != (len(df),):
        raise HexInputError(
            f"the test mask has shape {array.shape}, but {_SCRAMBLE_FRAME_NAME} "
            f"has {len(df)} rows.",
            hint,
        )
    return array


def _align_mask(df: pd.DataFrame, mask: pd.Series) -> pd.Series:
    """Return ``mask`` in the row order of ``df``, matched by index label."""
    hint = "Compute the mask with test_mask on the same frame that is scrambled."
    if len(mask) != len(df):
        raise HexInputError(
            f"the test mask has {len(mask)} entries, but {_SCRAMBLE_FRAME_NAME} "
            f"has {len(df)} rows.",
            hint,
        )
    if mask.index.equals(df.index):
        return mask
    same_labels = (
        mask.index.is_unique
        and df.index.is_unique
        and bool(mask.index.isin(df.index).all())
    )
    if not same_labels:
        raise HexInputError(
            f"the index of the test mask does not match the index of "
            f"{_SCRAMBLE_FRAME_NAME}; mask labels must be the same unique row ids.",
            hint,
        )
    return mask.reindex(df.index)


def _write_at(
    column: pd.Series, positions: NDArray[np.intp], replacement: pd.Series
) -> pd.Series:
    """Return ``column`` with ``replacement`` written at ``positions``, same dtype."""
    array = column.array.copy()
    array[positions] = replacement.array
    return pd.Series(array, index=column.index, name=column.name)


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
