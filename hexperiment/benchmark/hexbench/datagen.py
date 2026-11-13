"""Pure data generators for the benchmark cases.

Every generator is a pure function ``(seed, **params) -> pd.DataFrame``. It draws from
one ``np.random.default_rng(seed)`` in a fixed order, uses element-wise numpy
operations and no matrix product (so no BLAS), and rounds every float column to
4 decimals. The target is ``y`` and the features are ``f00``, ``f01``, ...
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from hexbench.spec import DataSpec

DECIMALS = 4
_LEVELS = 600
_HIGH_CARDINALITY_ROWS = 3000
_ENTITY_SINGLES = 2000
_ENTITY_GROUPS = 300
_ENTITY_GROUP_SIZE = 4


def _names(count: int) -> list[str]:
    return [f"f{i:02d}" for i in range(count)]


def _round(frame: pd.DataFrame) -> pd.DataFrame:
    floats = frame.select_dtypes(include="float").columns
    frame[floats] = frame[floats].round(DECIMALS)
    return frame


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _tabular(
    rng: np.random.Generator,
    *,
    n_rows: int,
    n_columns: int,
    n_informative: int,
    prevalence: float,
    shift: float,
    heavy_tail: bool,
    missing: float,
    label_flip: float,
    extra_name: str | None,
    extra_kind: str | None,
    extra_sigma: float,
) -> pd.DataFrame:
    if n_informative > n_columns:
        raise ValueError(
            f"n_informative ({n_informative}) is larger than n_columns ({n_columns})"
        )
    if (extra_name is None) != (extra_kind is None):
        raise ValueError("extra_name and extra_kind must be given together")
    if extra_kind not in (None, "proxy", "copy"):
        raise ValueError(f"extra_kind must be 'proxy' or 'copy', not {extra_kind!r}")
    y = (rng.random(n_rows) < prevalence).astype(np.int64)
    flipped = rng.random(n_rows) < label_flip
    y = np.where(flipped, 1 - y, y)
    informative = rng.permutation(n_columns)[:n_informative]
    x = rng.standard_normal((n_rows, n_columns))
    x[:, informative] += (2 * y - 1)[:, None] * shift
    if heavy_tail:
        x = np.sinh(1.5 * x)
    holes = rng.random((n_rows, n_columns)) < missing
    x = np.where(holes, np.nan, x)
    noise = rng.standard_normal(n_rows)
    frame = pd.DataFrame(x, columns=_names(n_columns))
    if extra_name is not None:
        if extra_kind == "proxy":
            frame[extra_name] = (2 * y - 1) + extra_sigma * noise
        else:
            frame[extra_name] = y
    frame["y"] = y
    return _round(frame)


def tabular(
    seed: int,
    *,
    n_rows: int = 3000,
    n_columns: int = 60,
    n_informative: int = 6,
    prevalence: float = 0.5,
    shift: float = 0.20,
    heavy_tail: bool = False,
    missing: float = 0.0,
    label_flip: float = 0.0,
    extra_name: str | None = None,
    extra_kind: str | None = None,
    extra_sigma: float = 1.0,
) -> pd.DataFrame:
    """Gaussian columns; ``shift`` separates the classes on the informative ones."""
    return _tabular(
        np.random.default_rng(seed),
        n_rows=n_rows,
        n_columns=n_columns,
        n_informative=n_informative,
        prevalence=prevalence,
        shift=shift,
        heavy_tail=heavy_tail,
        missing=missing,
        label_flip=label_flip,
        extra_name=extra_name,
        extra_kind=extra_kind,
        extra_sigma=extra_sigma,
    )


def high_cardinality(seed: int) -> pd.DataFrame:
    """3,000 rows, a category of 600 levels with a random effect each, 12 columns."""
    rng = np.random.default_rng(seed)
    n = _HIGH_CARDINALITY_ROWS
    levels = rng.integers(0, _LEVELS, size=n)
    effect = rng.standard_normal(_LEVELS)
    x = rng.standard_normal((n, 12))
    logit = effect[levels] + 0.4 * (x[:, 0] + x[:, 1] + x[:, 2])
    y = (rng.random(n) < _sigmoid(logit)).astype(np.int64)
    frame = pd.DataFrame(x, columns=_names(12))
    frame.insert(0, "cat", [f"c{level:03d}" for level in levels])
    frame["y"] = y
    return _round(frame)


def drifting(seed: int, *, n_rows: int = 3000) -> pd.DataFrame:
    """One row per hour; the sign of the effect of ``f01`` flips after 75% of rows."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_rows, 16))
    after = np.arange(n_rows) >= int(0.75 * n_rows)
    b = np.where(after, -1.0, 1.0)
    y = (rng.random(n_rows) < _sigmoid(1.5 * x[:, 0] + b * x[:, 1])).astype(np.int64)
    frame = pd.DataFrame(x, columns=_names(16))
    stamps = pd.date_range("2021-01-01 00:00:00", periods=n_rows, freq="h")
    frame.insert(0, "event_time", stamps.strftime("%Y-%m-%d %H:%M:%S"))
    frame["t_index"] = np.arange(n_rows) / n_rows
    frame["y"] = y
    return _round(frame)


def entities(seed: int) -> pd.DataFrame:
    """2,000 single-row entities and 300 entities of four noisy rows each."""
    rng = np.random.default_rng(seed)
    counts = np.array(
        [1] * _ENTITY_SINGLES + [_ENTITY_GROUP_SIZE] * _ENTITY_GROUPS, dtype=np.int64
    )
    n_entities = len(counts)
    z = rng.standard_normal((n_entities, 100))
    label = (rng.random(n_entities) < _sigmoid(1.5 * z[:, 0])).astype(np.int64)
    order = rng.permutation(100)
    owner = np.repeat(np.arange(n_entities), counts)
    rows = z[owner][:, order] + 0.3 * rng.standard_normal((len(owner), 100))
    frame = pd.DataFrame(rows, columns=_names(100))
    frame.insert(0, "entity_id", [f"e{i + 1:04d}" for i in owner])
    frame["y"] = label[owner]
    shuffled = frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)
    return _round(shuffled)


def duplicated(
    seed: int,
    *,
    n_rows: int = 2500,
    n_columns: int = 60,
    n_informative: int = 6,
    prevalence: float = 0.5,
    shift: float = 0.20,
    label_flip: float = 0.0,
    n_repeated: int = 1000,
) -> pd.DataFrame:
    """A ``tabular`` frame with ``n_repeated`` of its rows appended a second time."""
    if not 0 <= n_repeated <= n_rows:
        raise ValueError(f"n_repeated ({n_repeated}) must be between 0 and {n_rows}")
    rng = np.random.default_rng(seed)
    base = _tabular(
        rng,
        n_rows=n_rows,
        n_columns=n_columns,
        n_informative=n_informative,
        prevalence=prevalence,
        shift=shift,
        heavy_tail=False,
        missing=0.0,
        label_flip=label_flip,
        extra_name=None,
        extra_kind=None,
        extra_sigma=1.0,
    )
    chosen = rng.choice(n_rows, size=n_repeated, replace=False)
    both = pd.concat([base, base.iloc[chosen]], ignore_index=True)
    return both.iloc[rng.permutation(len(both))].reset_index(drop=True)


GENERATORS: Mapping[str, Callable[..., pd.DataFrame]] = {
    "tabular": tabular,
    "high_cardinality": high_cardinality,
    "drifting": drifting,
    "entities": entities,
    "duplicated": duplicated,
}


def generate(data: DataSpec, seed: int) -> pd.DataFrame:
    """Call the generator named in a ``DataSpec`` with its parameters."""
    try:
        generator = GENERATORS[data.generator]
    except KeyError:
        known = ", ".join(sorted(GENERATORS))
        raise ValueError(
            f"unknown generator {data.generator!r}; known generators: {known}"
        ) from None
    return generator(seed, **data.params)
