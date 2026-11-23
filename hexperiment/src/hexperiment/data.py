"""Data loading, splitting and the canary scramble.

``load`` and ``split`` are the public entry points. The split decides which rows
are test rows. It never looks at feature values, so the same rows are test rows in
every execution of a training script (Requirements 1.2 to 1.4). In canary mode the
values in those rows are replaced by a tilted resample (Requirement 1.5).
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hexperiment.canary import LoadMarker, write_load_marker
from hexperiment.child import own_read
from hexperiment.config import (
    ExecMode,
    HexConfig,
    SplitConfig,
    TargetConfig,
    current_mode,
    load_config,
    run_id_from_env,
)
from hexperiment.errors import HexInputError

type ArrayLike1D = Sequence[float] | NDArray[np.floating] | pd.Series

_GOLDEN_GAMMA = 0x9E3779B97F4A7C15
_MIX_1 = 0xBF58476D1CE4E5B9
_MIX_2 = 0x94D049BB133111EB
_UINT64_MODULUS = 1 << 64
_FRAME_NAME = "the frame being split"
_SCRAMBLE_FRAME_NAME = "the frame being scrambled"

# Key under which ``load`` stores the app directory and the time cutoff in
# ``df.attrs``, for ``split`` to read (Requirements 1.2, 1.3).
ATTRS_KEY = "hexperiment"

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


@dataclass(frozen=True, slots=True)
class ValidatedInputs:
    """Inputs of ``audit`` after validation (normal mode).

    Scores are float64 copies; the labels are ``target == target.positive``.
    """

    features: tuple[str, ...]
    y_true_train: NDArray[np.bool_]
    y_true_test: NDArray[np.bool_]
    y_score: NDArray[np.float64]
    y_score_train: NDArray[np.float64] | None


@dataclass(frozen=True, slots=True)
class SnapshotInputs:
    """All that canary and control mode need to write a snapshot."""

    features: tuple[str, ...]
    y_score_train: NDArray[np.float64] | None


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


def load(app_dir: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    """Read the app's CSV and return it with the row id as its index.

    ``app_dir`` defaults to the folder of the calling script. The index is
    ``0..n-1`` in file order. With a ``split.time_column`` the time cutoff is
    computed here, from the whole file, and stored with the app directory in
    ``df.attrs`` for ``split``. With ``HEX_MODE=canary`` the test rows are
    replaced by a tilted resample (Requirement 1.5); in any other mode the file
    values are returned as read. When ``HEX_RUN_ID`` is set, a marker file tells
    the parent process that the script loaded its data here (Requirement 10.6).
    """
    directory = _app_dir(app_dir, sys._getframe(1).f_globals)
    config = load_config(directory)
    mode = current_mode()
    run_id = run_id_from_env()
    df = _read_csv(directory, config.data.path)
    cutoff = compute_cutoff(df, config.split)
    scrambled_ids = None
    if mode is ExecMode.CANARY:
        mask = test_mask(df, config.split, cutoff)
        scrambled_ids = df.index.to_numpy(dtype=np.int64)[mask]
        df = scramble_test_rows(df, mask, config.split.time_column, config.split.seed)
    if run_id is not None:
        columns = tuple(str(name) for name in df.columns)
        write_load_marker(directory, LoadMarker(mode, run_id, scrambled_ids, columns))
    df.attrs[ATTRS_KEY] = {"app_dir": str(directory), "cutoff": cutoff}
    return df


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame returned by ``load`` into ``(train, test)`` copies.

    The app directory and the time cutoff come from ``df.attrs``, so rows dropped
    or filtered after ``load`` cannot move the time boundary: a remaining row is a
    test row exactly when its ``(time, row id)`` is at or after the stored cutoff.
    If ``attrs`` was lost, the app is the calling script's folder and the cutoff is
    recomputed from ``df``, which can move the boundary after rows were removed
    (the canary decision still guards against that). Raises ``HexInputError`` when
    either side is empty.
    """
    stored = df.attrs.get(ATTRS_KEY)
    if stored is None:
        directory = _app_dir(
            None,
            sys._getframe(1).f_globals,
            fix="Pass the frame returned by hexperiment.load to hexperiment.split, "
            "and call split from a script file.",
        )
        cutoff = None
    else:
        directory, cutoff = _stored_state(stored)
    config = load_config(directory)
    train, test = split_frame(df, config.split, cutoff)
    _require_both_sides(train, test, config.split)
    return train, test


def validate_audit_inputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_score: ArrayLike1D,
    y_score_train: ArrayLike1D | None,
    features: Sequence[str],
    config: HexConfig,
) -> ValidatedInputs:
    """Validate the arguments of ``audit`` in normal mode (Requirement 4).

    Raises ``HexInputError`` for the first problem found, checked in this order:
    an empty frame (4.1); a missing target column (4.2); missing values in the
    target of either frame (4.2); a score array of the wrong length, with NaN or
    outside ``[0, 1]`` (4.4); an empty ``features`` list, or a feature missing
    from either frame (4.5); a missing or unparseable time column (4.6); a test
    target with a single class (4.3). Nothing is modified.
    """
    _require_rows(train, "train")
    _require_rows(test, "test")
    target = config.target
    for label, frame in (("train", train), ("test", test)):
        _require_target(frame, label, target.column)
    for label, frame in (("train", train), ("test", test)):
        _require_target_values(frame, label, target.column)
    scores = _as_scores(y_score, "y_score")
    scores_train = (
        None if y_score_train is None else _as_scores(y_score_train, "y_score_train")
    )
    _require_length(scores, "y_score", len(test), "test")
    if scores_train is not None:
        _require_length(scores_train, "y_score_train", len(train), "train")
    _require_probabilities(scores, "y_score")
    if scores_train is not None:
        _require_probabilities(scores_train, "y_score_train")
    names = _feature_names(features)
    _require_some_features(names)
    _require_features(names, (("train", train), ("test", test)))
    time_column = config.split.time_column
    if time_column is not None:
        for label, frame in (("train", train), ("test", test)):
            frame_name = f"the {label} frame"
            _require_time_column(frame, time_column, frame_name)
            parse_times(frame[time_column], time_column, frame_name)
    y_true_test = _is_positive(test, target)
    _require_two_classes(test, target, y_true_test)
    return ValidatedInputs(
        features=names,
        y_true_train=_is_positive(train, target),
        y_true_test=y_true_test,
        y_score=scores,
        y_score_train=scores_train,
    )


def validate_snapshot_inputs(
    train: pd.DataFrame,
    y_score_train: ArrayLike1D | None,
    features: Sequence[str],
) -> SnapshotInputs:
    """Validate what a canary or control execution needs to write its snapshot.

    Only the training-side rules of Requirement 4 apply here: ``train`` is not
    empty (4.1), ``y_score_train`` (when given) has one score per training row,
    none NaN and none outside ``[0, 1]`` (4.4), and every feature is a column of
    ``train`` (4.5). The test target, ``y_score``, the time column and the test
    frame are not looked at, so a scrambled test set cannot raise (Requirement
    2.4). A NaN score is rejected because the canary treats NaN as a change.
    """
    _require_rows(train, "train")
    scores_train = (
        None if y_score_train is None else _as_scores(y_score_train, "y_score_train")
    )
    if scores_train is not None:
        _require_length(scores_train, "y_score_train", len(train), "train")
        _require_probabilities(scores_train, "y_score_train")
    names = _feature_names(features)
    _require_some_features(names)
    _require_features(names, (("train", train),))
    return SnapshotInputs(features=names, y_score_train=scores_train)


# --- helpers -------------------------------------------------------------------------


def _app_dir(
    app_dir: str | os.PathLike[str] | None,
    caller_globals: Mapping[str, Any],
    *,
    fix: str = "Pass the app folder: hexperiment.load(app_dir=...).",
) -> Path:
    """Resolve the app folder: the given one, else the calling script's folder."""
    if app_dir is not None:
        return Path(app_dir).resolve()
    script = caller_globals.get("__file__")
    if not isinstance(script, str | os.PathLike):
        raise HexInputError(
            "hexperiment cannot tell which app this call belongs to, because the "
            "calling code has no __file__ (it is not a script file).",
            fix,
        )
    return Path(script).resolve().parent


def _stored_state(stored: Any) -> tuple[Path, TimeCutoff | None]:
    """Read the app directory and cutoff that ``load`` left in ``df.attrs``."""
    if (
        not isinstance(stored, Mapping)
        or not isinstance(stored.get("app_dir"), str)
        or "cutoff" not in stored
        or not isinstance(stored["cutoff"], TimeCutoff | None)
    ):
        raise HexInputError(
            f'df.attrs["{ATTRS_KEY}"] is {stored!r}, which is not what '
            "hexperiment.load stores.",
            f'Do not change df.attrs["{ATTRS_KEY}"]; pass the frame from '
            "hexperiment.load to hexperiment.split.",
        )
    return Path(stored["app_dir"]), stored["cutoff"]


def _read_csv(directory: Path, relative: str) -> pd.DataFrame:
    """Read the data file (Requirement 1.6); every failure names the resolved path."""
    path = (directory / relative).resolve()
    try:
        with own_read():
            df = pd.read_csv(path)
    except FileNotFoundError:
        raise HexInputError(
            f'data.path "{relative}" does not exist (looked at {path}).',
            "Put the data file at that path, or correct [data] path in hex.toml "
            "(an app made by hexperiment init reads data/dataset.csv).",
        ) from None
    except pd.errors.EmptyDataError:
        raise HexInputError(
            f"the data file {path} is empty.",
            "Run fetch_data.py again, or correct [data] path in hex.toml.",
        ) from None
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise HexInputError(
            f"the data file {path} cannot be read as CSV ({error}).",
            "Check that the file is a UTF-8 CSV with a header row.",
        ) from None
    except OSError as error:
        raise HexInputError(
            f"the data file {path} cannot be opened ({error.strerror or error}).",
            "Check that [data] path in hex.toml points to a readable file.",
        ) from None
    if df.empty:
        raise HexInputError(
            f"the data file {path} has a header but no data rows.",
            "Run fetch_data.py again, or correct [data] path in hex.toml.",
        )
    if not isinstance(df.index, pd.RangeIndex):
        # pandas turns the first column into the index when every row has one more
        # field than the header; the row id must be the position in the file.
        raise HexInputError(
            f"the data file {path} has rows with more fields than its header.",
            "Fix the CSV so that every row has one value per header column.",
        )
    return df


def _require_both_sides(
    train: pd.DataFrame, test: pd.DataFrame, config: SplitConfig
) -> None:
    for name, frame in (("train", train), ("test", test)):
        if len(frame) == 0:
            raise HexInputError(
                f"the {name} set is empty after the split "
                f"(split.test_fraction is {config.test_fraction:g}, "
                f"{len(train) + len(test)} rows were split).",
                f"Do not filter away every {name} row before hexperiment.split, "
                "and check that the data has enough rows.",
            )


def _require_rows(frame: pd.DataFrame, label: str) -> None:
    if len(frame) == 0:
        raise HexInputError(
            f"{label} is empty (it has no rows).",
            f"Pass the {label} frame from hexperiment.split, and do not filter "
            f"away every {label} row before audit.",
        )


def _require_target(frame: pd.DataFrame, label: str, column: str) -> None:
    if column not in frame.columns:
        raise HexInputError(
            f"target column {column!r} is missing from {label}.",
            f"Keep {column!r} in the {label} frame you pass to audit (it is set by "
            "[target] column in hex.toml); list only model inputs in features.",
        )


def _require_target_values(frame: pd.DataFrame, label: str, column: str) -> None:
    missing = int(frame[column].isna().sum())
    if missing:
        raise HexInputError(
            f"target column {column!r} has {missing} missing value(s) in {label} "
            f"({len(frame)} rows).",
            f"Remove the rows with a missing {column!r} before hexperiment.split "
            f"(for example df.dropna(subset=[{column!r}])), or fill in the label.",
        )


def _as_scores(values: ArrayLike1D, name: str) -> NDArray[np.float64]:
    """Return ``values`` as a one-dimensional float64 copy."""
    if values is None:
        raise HexInputError(
            f"{name} is missing (None), so there are no predicted probabilities "
            "to audit.",
            f"Pass {name} as the predicted probability of the positive class, one "
            f"per row, for example model.predict_proba(X)[:, 1] as {name}.",
        )
    try:
        if isinstance(values, pd.Series):
            array = values.to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            array = np.array(values, dtype=np.float64)
    except (TypeError, ValueError):
        raise HexInputError(
            f"{name} cannot be converted to numbers.",
            f"Pass {name} as a list, array or Series of predicted probabilities, "
            "one per row.",
        ) from None
    if array.ndim != 1:
        raise HexInputError(
            f"{name} must be one-dimensional (one probability per row), but it has "
            f"shape {array.shape}.",
            f"Pass the probability of the positive class only, for example "
            f"model.predict_proba(X)[:, 1] as {name}.",
        )
    return array


def _require_length(
    scores: NDArray[np.float64], name: str, rows: int, label: str
) -> None:
    if len(scores) != rows:
        raise HexInputError(
            f"{name} has {len(scores)} value(s) but {label} has {rows} row(s).",
            f"Pass one score per {label} row, in the same order as the {label} frame.",
        )


def _require_probabilities(scores: NDArray[np.float64], name: str) -> None:
    """Reject NaN and values outside ``[0, 1]`` (Requirement 4.4)."""
    valid = (scores >= 0.0) & (scores <= 1.0)
    if valid.all():
        return
    nan_count = int(np.isnan(scores).sum())
    outside = int((~valid).sum()) - nan_count
    parts = []
    if nan_count:
        parts.append(f"{nan_count} NaN value(s)")
    if outside:
        parts.append(f"{outside} value(s) outside [0, 1]")
    first = int(np.flatnonzero(~valid)[0])
    raise HexInputError(
        f"{name} has {' and '.join(parts)} (first at position {first}).",
        "Pass predicted probabilities of the positive class, for example "
        "model.predict_proba(X)[:, 1], and fix rows that give NaN before predicting.",
    )


def _feature_names(features: Sequence[str]) -> tuple[str, ...]:
    if isinstance(features, str):
        raise HexInputError(
            f"features is the string {features!r}, but it must be a list of column "
            "names.",
            f"Pass features=[{features!r}] (a list), not the bare name.",
        )
    return tuple(features)


def _require_some_features(names: tuple[str, ...]) -> None:
    if not names:
        raise HexInputError(
            "features is empty, so the model used no input columns.",
            "Pass the names of the columns the model was trained on, for example "
            "features=['age', 'plan'].",
        )


def _require_features(
    names: tuple[str, ...], frames: Sequence[tuple[str, pd.DataFrame]]
) -> None:
    missing = []
    for name in names:
        absent = [label for label, frame in frames if name not in frame.columns]
        if absent:
            missing.append(f"{name!r} (missing from {' and '.join(absent)})")
    if missing:
        where = " and ".join(label for label, _ in frames)
        raise HexInputError(
            f"features not found as columns of {where}: {', '.join(missing)}.",
            f"Use only column names that exist in {where}; check the spelling "
            "and that the column was not dropped before audit.",
        )


def _is_positive(frame: pd.DataFrame, target: TargetConfig) -> NDArray[np.bool_]:
    """Label rows ``target == positive``; missing values were rejected earlier."""
    matches = frame[target.column].eq(target.positive)
    return matches.fillna(False).to_numpy(dtype=bool)


def _require_two_classes(
    test: pd.DataFrame, target: TargetConfig, y_true: NDArray[np.bool_]
) -> None:
    if y_true.any() and not y_true.all():
        return
    values = list(test[target.column].drop_duplicates().head(10))
    shown = ", ".join(repr(v) for v in values)
    positives = int(y_true.sum())
    raise HexInputError(
        f"the test target {target.column!r} has a single class, so the metric is "
        f"undefined: {positives} of {len(test)} test row(s) equal the positive "
        f"value {target.positive!r} (distinct test target values: {shown}).",
        f"Set [target] positive in hex.toml to a value that occurs in "
        f"{target.column!r}, and make sure the test rows hold both classes "
        "(more rows, or a larger split.test_fraction).",
    )


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
