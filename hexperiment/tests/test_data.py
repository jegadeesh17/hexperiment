import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

import hexperiment
from hexperiment import HexInputError, data
from hexperiment.canary import LoadMarker, hex_dir, read_load_marker
from hexperiment.config import ExecMode, HexConfig, SplitConfig, parse_config
from hexperiment.data import (
    SnapshotInputs,
    TimeCutoff,
    ValidatedInputs,
    compute_cutoff,
    hash_fraction,
    parse_times,
    scramble_test_rows,
    split_frame,
    tilt_column,
    validate_audit_inputs,
    validate_snapshot_inputs,
)

MASK64 = (1 << 64) - 1


def splitmix_reference(row_id: int, seed: int) -> float:
    """Plain-Python SplitMix64 finalizer, used to check the numpy version."""
    x = (row_id + (seed + 1) * 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    x ^= x >> 31
    return (x >> 11) / 2**53


def hash_split(fraction: float = 0.2, seed: int = 0) -> SplitConfig:
    return SplitConfig(test_fraction=fraction, seed=seed)


def time_split(fraction: float = 0.2, seed: int = 0) -> SplitConfig:
    return SplitConfig(test_fraction=fraction, seed=seed, time_column="t")


def day(n: int) -> str:
    return (pd.Timestamp("2024-01-01") + pd.Timedelta(days=n)).strftime("%Y-%m-%d")


def time_frame(days: list[int], index: list[int] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {"t": [day(d) for d in days], "x": np.arange(len(days))},
        index=index if index is not None else pd.RangeIndex(len(days)),
    )


def n_test_for(n: int, fraction: float) -> int:
    return min(n - 1, max(1, math.floor(n * fraction + 0.5)))


# --- hash_fraction (Requirement 1.4) -------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 12345, 2**40])
def test_hash_fraction_matches_the_splitmix64_reference(seed: int) -> None:
    ids = np.arange(200, dtype=np.int64)
    expected = [splitmix_reference(int(i), seed) for i in ids]
    assert hash_fraction(ids, seed).tolist() == expected


def test_hash_fraction_values_are_in_the_half_open_unit_interval() -> None:
    values = hash_fraction(np.arange(50_000, dtype=np.int64), 3)
    assert values.min() >= 0.0
    assert values.max() < 1.0


@pytest.mark.parametrize("fraction", [0.1, 0.2, 0.5, 0.9])
def test_hash_fraction_is_close_to_uniform(fraction: float) -> None:
    values = hash_fraction(np.arange(20_000, dtype=np.int64), 0)
    assert abs((values < fraction).mean() - fraction) < 0.015


def test_hash_fraction_changes_with_the_seed() -> None:
    ids = np.arange(100, dtype=np.int64)
    assert not np.array_equal(hash_fraction(ids, 0), hash_fraction(ids, 1))


def test_hash_fraction_of_an_empty_array_is_empty() -> None:
    assert hash_fraction(np.array([], dtype=np.int64), 0).shape == (0,)


# --- parse_times ---------------------------------------------------------------------


def test_parse_times_treats_naive_and_aware_values_as_utc() -> None:
    naive = parse_times(pd.Series(["2024-01-01 10:00:00"]), "t", "train")
    aware = parse_times(pd.Series(["2024-01-01 12:00:00+02:00"]), "t", "test")
    assert naive.iloc[0] == aware.iloc[0]
    assert str(naive.dt.tz) == "UTC"
    assert str(aware.dt.tz) == "UTC"


def test_parse_times_orders_mixed_timezones_by_instant() -> None:
    series = pd.Series(
        [
            "2024-01-01T12:00:00+02:00",  # 10:00 UTC
            "2024-01-01T10:30:00Z",
            "2024-01-01T09:00:00-02:00",  # 11:00 UTC
        ]
    )
    parsed = parse_times(series, "t", "train")
    assert list(parsed.argsort()) == [0, 1, 2]


def test_parse_times_names_column_and_frame_for_unparseable_values() -> None:
    with pytest.raises(HexInputError) as caught:
        parse_times(
            pd.Series(["2024-01-01", "not a date"]), "signup", "the train frame"
        )
    assert "signup" in str(caught.value)
    assert "the train frame" in str(caught.value)


def test_parse_times_rejects_missing_values() -> None:
    with pytest.raises(HexInputError, match="signup") as caught:
        parse_times(pd.Series(["2024-01-01", None]), "signup", "the test frame")
    assert "the test frame" in str(caught.value)


# --- compute_cutoff (Requirement 1.3) ------------------------------------------------


def test_compute_cutoff_is_none_without_a_time_column() -> None:
    assert compute_cutoff(time_frame([1, 2, 3]), hash_split()) is None


def test_compute_cutoff_returns_the_first_test_row_in_time_order() -> None:
    frame = time_frame(
        [5, 3, 9, 1, 7, 2, 8, 4, 6, 0]
    )  # ids by time: 9,3,5,1,7,0,4,8,2,6
    cutoff = compute_cutoff(frame, time_split(0.3))
    assert cutoff == TimeCutoff(time=pd.Timestamp(day(7), tz="UTC"), row_id=4)


@pytest.mark.parametrize(
    ("n", "fraction", "expected"),
    [
        (10, 0.25, 3),  # 2.5 rounds half up
        (10, 0.2, 2),
        (10, 0.04, 1),  # at least one test row
        (2, 0.2, 1),
        (3, 0.99, 2),  # at least one training row
        (5, 0.5, 3),
    ],
)
def test_test_size_rounds_half_up_and_keeps_both_sides_non_empty(
    n: int, fraction: float, expected: int
) -> None:
    train, test = split_frame(time_frame(list(range(n))), time_split(fraction))
    assert len(test) == expected
    assert len(train) == n - expected


@pytest.mark.parametrize("n", [0, 1])
def test_fewer_than_two_rows_raises(n: int) -> None:
    with pytest.raises(HexInputError, match="at least 2"):
        compute_cutoff(time_frame(list(range(n))), time_split())


def test_equal_timestamps_at_the_boundary_are_ordered_by_row_id() -> None:
    frame = time_frame([1, 2, 2, 2, 3])  # n_test = 2 -> rows 3 and 4
    cutoff = compute_cutoff(frame, time_split(0.4))
    assert cutoff == TimeCutoff(time=pd.Timestamp(day(2), tz="UTC"), row_id=3)

    train, test = split_frame(frame, time_split(0.4))
    assert train.index.tolist() == [0, 1, 2]
    assert test.index.tolist() == [3, 4]


def test_mixed_timezones_are_compared_as_instants() -> None:
    frame = pd.DataFrame(
        {
            "t": [
                "2024-01-01T12:00:00+02:00",  # 10:00 UTC
                "2024-01-01T10:30:00Z",
                "2024-01-01T09:00:00-02:00",  # 11:00 UTC
                "2024-01-01T08:00:00Z",
            ]
        }
    )
    train, test = split_frame(frame, time_split(0.5))
    assert sorted(train.index) == [0, 3]  # 10:00 and 08:00 UTC
    assert sorted(test.index) == [1, 2]  # 10:30 and 11:00 UTC


def test_unparseable_timestamp_names_the_column() -> None:
    frame = pd.DataFrame({"t": ["2024-01-01", "oops", "2024-01-03"]})
    with pytest.raises(HexInputError, match="'t'"):
        compute_cutoff(frame, time_split())


def test_missing_time_column_raises() -> None:
    with pytest.raises(HexInputError, match="'t'"):
        split_frame(pd.DataFrame({"x": [1, 2, 3]}), time_split())


# --- test_mask and split_frame -------------------------------------------------------


def test_hash_split_is_independent_of_row_values() -> None:
    a = pd.DataFrame({"v": np.arange(500)})
    b = pd.DataFrame({"v": np.arange(500)[::-1] * 7.5, "w": "z"})
    assert np.array_equal(
        data.test_mask(a, hash_split(0.3, 4)), data.test_mask(b, hash_split(0.3, 4))
    )


def test_hash_split_fraction_is_close_to_the_requested_one() -> None:
    train, test = split_frame(pd.DataFrame({"v": np.arange(10_000)}), hash_split(0.2))
    assert abs(len(test) / 10_000 - 0.2) < 0.015
    assert len(train) + len(test) == 10_000


def test_split_frame_keeps_the_row_id_index_and_returns_copies() -> None:
    frame = pd.DataFrame({"v": np.arange(100)})
    train, test = split_frame(frame, hash_split(0.25))
    assert train.index.is_monotonic_increasing and test.index.is_monotonic_increasing
    assert sorted(train.index.tolist() + test.index.tolist()) == list(range(100))
    assert (train["v"] == train.index).all()

    train.loc[train.index[0], "v"] = -1
    test.loc[test.index[0], "v"] = -1
    assert (frame["v"] == np.arange(100)).all()


def test_time_split_with_a_non_integer_index_raises() -> None:
    frame = time_frame([1, 2, 3], index=[0, 1, 2]).rename(index=str)
    with pytest.raises(HexInputError, match="row ids"):
        split_frame(frame, time_split())


def test_frame_with_rows_dropped_keeps_the_boundary_given_a_stored_cutoff() -> None:
    frame = time_frame(list(range(10)))  # n_test = 3 -> rows 7, 8, 9
    split = time_split(0.3)
    cutoff = compute_cutoff(frame, split)
    assert cutoff is not None and cutoff.row_id == 7

    reduced = frame.drop(index=[2, 5, 8, 9])  # dropna()-style filtering
    train, test = split_frame(reduced, split, cutoff)
    assert train.index.tolist() == [0, 1, 3, 4, 6]
    assert test.index.tolist() == [7]

    # Without the stored cutoff the boundary moves and row 6 becomes a test row.
    _, moved_test = split_frame(reduced, split)
    assert 6 in moved_test.index


def test_dropping_every_test_row_leaves_an_empty_test_frame() -> None:
    frame = time_frame(list(range(10)))
    split = time_split(0.3)
    cutoff = compute_cutoff(frame, split)
    train, test = split_frame(frame.iloc[:7], split, cutoff)
    assert len(train) == 7
    assert test.empty


def test_hash_split_is_unchanged_when_rows_are_dropped() -> None:
    frame = pd.DataFrame({"v": np.arange(300)})
    split = hash_split(0.3, 2)
    _, full_test = split_frame(frame, split)
    kept = frame.drop(index=frame.index[::3])
    _, kept_test = split_frame(kept, split)
    assert kept_test.index.tolist() == [i for i in full_test.index if i in kept.index]


# --- properties ----------------------------------------------------------------------

fractions = st.floats(min_value=0.01, max_value=0.99)
seeds = st.integers(min_value=0, max_value=2**32)


@given(
    ids=st.lists(st.integers(0, 10**9), min_size=1, max_size=60, unique=True),
    seed=seeds,
    fraction=fractions,
    values=st.lists(st.floats(allow_nan=False), min_size=60, max_size=60),
)
def test_hash_membership_depends_only_on_id_seed_and_fraction(
    ids: list[int], seed: int, fraction: float, values: list[float]
) -> None:
    split = hash_split(fraction, seed)
    plain = pd.DataFrame({"a": np.zeros(len(ids))}, index=ids)
    other = pd.DataFrame({"a": values[: len(ids)], "b": "text"}, index=ids)
    mask = data.test_mask(plain, split)
    assert np.array_equal(mask, data.test_mask(other, split))
    for position, row_id in enumerate(ids):
        alone = pd.DataFrame({"a": [0.0]}, index=[row_id])
        assert bool(data.test_mask(alone, split)[0]) == bool(mask[position])


@given(
    k=st.integers(1, 80),
    extra=st.integers(0, 80),
    seed=seeds,
    fraction=fractions,
)
def test_first_ids_get_the_same_decision_for_any_total(
    k: int, extra: int, seed: int, fraction: float
) -> None:
    split = hash_split(fraction, seed)
    short = data.test_mask(pd.DataFrame(index=pd.RangeIndex(k)), split)
    long = data.test_mask(pd.DataFrame(index=pd.RangeIndex(k + extra)), split)
    assert np.array_equal(short, long[:k])


@given(
    days=st.lists(st.integers(0, 15), min_size=2, max_size=60),
    fraction=fractions,
)
def test_time_split_puts_every_training_time_before_every_test_time(
    days: list[int], fraction: float
) -> None:
    frame = time_frame(days)
    train, test = split_frame(frame, time_split(fraction))
    assert len(test) == n_test_for(len(days), fraction)
    assert len(train) == len(days) - len(test)
    assert sorted(train.index.tolist() + test.index.tolist()) == list(range(len(days)))
    assert pd.to_datetime(train["t"]).max() <= pd.to_datetime(test["t"]).min()


@given(
    days=st.lists(st.integers(0, 15), min_size=2, max_size=60),
    fraction=fractions,
    keep=st.lists(st.booleans(), min_size=60, max_size=60),
)
def test_dropping_rows_never_moves_a_row_across_a_stored_cutoff(
    days: list[int], fraction: float, keep: list[bool]
) -> None:
    frame = time_frame(days)
    split = time_split(fraction)
    cutoff = compute_cutoff(frame, split)
    full_mask = pd.Series(data.test_mask(frame, split, cutoff), index=frame.index)

    kept = frame[np.array(keep[: len(days)], dtype=bool)]
    kept_mask = pd.Series(data.test_mask(kept, split, cutoff), index=kept.index)
    assert kept_mask.equals(full_mask.loc[kept.index])


# --- tilt_column and scramble_test_rows (Requirement 1.5) ----------------------------

LEVELS = ["lo", "mid", "hi", "unused"]


class ConstantDraws:
    """Stands in for a generator whose draws all pick the first pool entry."""

    def integers(self, low: int, high: int, size: int) -> np.ndarray:
        return np.zeros(size, dtype=np.int64)


class ScriptedDraws:
    """Stands in for a generator that returns fixed pool indices and records calls."""

    def __init__(self, draws: list[int]) -> None:
        self.draws = draws
        self.calls: list[tuple[int, int, int]] = []

    def integers(self, low: int, high: int, size: int) -> np.ndarray:
        self.calls.append((low, high, size))
        assert size == len(self.draws)
        assert max(self.draws) < high, f"the pool has {high} entries, expected more"
        return np.array(self.draws, dtype=np.int64)


class CyclingDraws:
    """Stands in for a generator that visits every pool entry in order, cyclically."""

    def integers(self, low: int, high: int, size: int) -> np.ndarray:
        return np.arange(size, dtype=np.int64) % high


def every_row(n: int) -> np.ndarray:
    return np.ones(n, dtype=bool)


def stamps(codes: list[int | None]) -> pd.DatetimeIndex:
    """Dates ``day(code)``; ``None`` becomes NaT."""
    return pd.to_datetime([None if c is None else day(c) for c in codes])


def mixed_dtype_frame() -> pd.DataFrame:
    """Dtypes that are easy to lose on write-back, with missing values and 3+ levels."""
    codes: list[int | None] = [1, 2, 2, 3, 3, 3, None, 1]
    return pd.DataFrame(
        {
            "when": stamps(codes),
            "when_tz": stamps(codes).tz_localize("Europe/Paris"),
            "gap": pd.to_timedelta(codes, unit="h"),
            "count": pd.array([5, None, 7, 7, 9, 5, 9, 7], dtype="Int64"),
            "maybe": pd.array(
                [True, None, False, True, None, True, False, False], dtype="boolean"
            ),
            "level": pd.Categorical(
                ["lo", "hi", "hi", None, "mid", "hi", "mid", "lo"],
                categories=LEVELS,
                ordered=True,
            ),
        },
        index=np.arange(100, 108),
    )


def random_frame(n: int = 200) -> pd.DataFrame:
    gen = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "x": gen.normal(size=n),
            "plan": gen.choice(list("abcde"), size=n),
            "y": gen.integers(0, 2, size=n),
        }
    )


def test_two_valued_columns_are_swapped() -> None:
    target = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  # 30% positives
    plan = ["basic", "pro", "pro", "basic", "basic", "basic", "pro", "basic", "basic"]
    frame = pd.DataFrame(
        {"target": target, "flag": [True, False] * 5, "plan": plan + ["basic"]}
    )
    out = scramble_test_rows(frame, every_row(10), None, 0)
    assert out["target"].tolist() == [1 - v for v in target]
    assert out["target"].mean() == pytest.approx(0.7)
    assert out["flag"].tolist() == [not v for v in frame["flag"]]
    swap = {"basic": "pro", "pro": "basic"}
    assert out["plan"].tolist() == [swap[v] for v in frame["plan"]]
    assert out.dtypes.equals(frame.dtypes)


def test_a_column_with_one_distinct_test_value_is_unchanged() -> None:
    values = pd.Series([4.0, np.nan, 4.0, 4.0], index=[3, 5, 8, 9])
    out = tilt_column(values, np.random.default_rng(0))
    pd.testing.assert_series_equal(out, values)


@pytest.mark.parametrize("seed", range(5))
def test_ten_distinct_values_draw_from_the_upper_half(seed: int) -> None:
    values = pd.Series([7, 2, 9, 4, 1, 10, 3, 8, 6, 5], dtype="int64")
    out = tilt_column(values, np.random.default_rng(seed))
    assert out.min() >= 6  # the sixth-smallest value, first of the upper half
    assert out.mean() > values.mean()
    assert out.dtype == values.dtype


@pytest.mark.parametrize("seed", range(20))
def test_a_one_valued_upper_half_is_widened_to_the_second_largest_value(
    seed: int,
) -> None:
    values = pd.Series([3, 1, 3, 2, 3, 3, 3])  # sorted: 1, 2, 3, 3, 3, 3, 3
    out = tilt_column(values, np.random.default_rng(seed))
    assert set(out) == {2, 3}  # pool [2, 3, 3, 3, 3, 3]; 1 is never drawn


@pytest.mark.parametrize("seed", range(10))
def test_other_dtypes_draw_from_the_more_frequent_half_of_categories(
    seed: int,
) -> None:
    values = pd.Series(list("eeeeedddab"))  # counts 5, 3, 1, 1
    out = tilt_column(values, np.random.default_rng(seed))
    assert set(out) == {"d", "e"}


@pytest.mark.parametrize("seed", range(10))
def test_a_tie_at_the_category_cutoff_is_broken_by_ascending_str(seed: int) -> None:
    values = pd.Series(["p", "b", "p", "a", "p", "b", "p", "a"])  # 4, 2, 2; keep 2
    out = tilt_column(values, np.random.default_rng(seed))
    assert set(out) == {"a", "p"}


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([1.5, np.nan, 2.5, 3.5, np.nan, 4.5]),
        pd.Series(["a", None, "b", "c", None, "a"], dtype=object),
        pd.Series(["a", None, "b", "c", None, "a"], dtype="string"),
        pd.Series([1, None, 2, 3, None, 1], dtype="Int64"),
    ],
    ids=["float", "object", "string", "nullable-int"],
)
def test_missing_values_keep_their_positions(values: pd.Series) -> None:
    out = tilt_column(values, np.random.default_rng(0))
    assert out.isna().tolist() == values.isna().tolist()
    assert out.dtype == values.dtype
    assert set(out.dropna()) <= set(values.dropna())


def test_constant_numeric_draws_are_repaired_with_the_largest_different_value() -> (
    None
):
    values = pd.Series([3.0, 1.0, 2.0])  # pool [2, 3]; every draw picks 2
    out = tilt_column(values, ConstantDraws())  # type: ignore[arg-type]
    assert out.tolist() == [3.0, 2.0, 2.0]


def test_constant_category_draws_are_repaired_with_the_most_frequent_category() -> (
    None
):
    # counts a=3, b=2, c=1, so the pool is the positions of a and b. The first pool
    # entry is position 1 ("a"), so every draw is "a" and the repair writes the
    # highest-ranked different pool value ("b", first at position 3) at position 0.
    # If the pool were the whole column, the first entry would be "c" (position 0),
    # every draw would be "c", and the result would be a, c, c, c, c, c instead.
    values = pd.Series(["c", "a", "a", "b", "b", "a"])
    out = tilt_column(values, ConstantDraws())  # type: ignore[arg-type]
    assert out.tolist() == ["b", "a", "a", "a", "a", "a"]
    assert "c" not in set(out)


def test_category_draws_never_leave_the_more_frequent_half() -> None:
    values = pd.Series(["c", "a", "a", "b", "b", "a"])
    out = tilt_column(values, CyclingDraws())  # type: ignore[arg-type]
    # Pool is the positions of a and b in column order: a, a, b, b, a.
    assert out.tolist() == ["a", "a", "b", "b", "a", "a"]


def test_nine_distinct_values_draw_from_the_exact_upper_half() -> None:
    # m = 9 non-missing values, all distinct, so there is no widening. The pool
    # starts at floor(9 / 2) = 4, giving sorted[4:] = 50..90 (five values). A start
    # of (9 + 1) // 2 = 5 would give four values and exclude 50.
    values = pd.Series([70, 20, 90, 40, 10, 60, 30, 80, 50])
    rng = ScriptedDraws([0, 1, 2, 3, 4, 0, 1, 2, 3])
    out = tilt_column(values, rng)  # type: ignore[arg-type]
    assert rng.calls == [(0, 5, 9)]
    assert out.tolist() == [50, 60, 70, 80, 90, 50, 60, 70, 80]
    assert out.dtype == values.dtype


def test_nine_distinct_values_with_missing_ones_keep_the_pool_exact() -> None:
    values = pd.Series([70, None, 20, 90, 40, 10, None, 60, 30, 80, 50], dtype="Int64")
    rng = ScriptedDraws([4, 3, 2, 1, 0, 4, 3, 2, 1])
    out = tilt_column(values, rng)  # type: ignore[arg-type]
    assert rng.calls == [(0, 5, 9)]
    assert out.isna().tolist() == values.isna().tolist()
    assert out.dropna().tolist() == [90, 80, 70, 60, 50, 90, 80, 70, 60]


def test_the_time_column_is_never_scrambled() -> None:
    frame = pd.DataFrame(
        {"t": [day(d) for d in [3, 1, 4, 1, 5, 9, 2, 6]], "x": np.arange(1, 9)}
    )
    out = scramble_test_rows(frame, every_row(8), "t", 0)
    pd.testing.assert_series_equal(out["t"], frame["t"])
    assert out["x"].min() >= 5  # the other column was tilted


def test_only_test_rows_change_and_the_input_is_left_alone() -> None:
    frame = pd.DataFrame({"x": np.arange(20, dtype=float)}, index=np.arange(100, 120))
    mask = np.arange(20) % 2 == 0
    before = frame.copy()
    out = scramble_test_rows(frame, mask, None, 1)
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(out[~mask], frame[~mask])
    assert out.index.equals(frame.index)
    assert not out[mask].equals(frame[mask])
    assert set(out.loc[mask, "x"]) <= set(frame.loc[mask, "x"])


def test_the_same_seed_gives_the_same_frame_and_another_seed_a_different_one() -> None:
    frame = random_frame()
    mask = hash_fraction(np.arange(200, dtype=np.int64), 0) < 0.3
    first = scramble_test_rows(frame, mask, None, 5)
    pd.testing.assert_frame_equal(first, scramble_test_rows(frame, mask, None, 5))
    assert not first.equals(scramble_test_rows(frame, mask, None, 6))


def test_test_values_are_tilted_in_row_id_order_whatever_the_frame_order() -> None:
    frame = random_frame()
    mask = hash_fraction(np.arange(200, dtype=np.int64), 0) < 0.3
    shuffled = frame.sample(frac=1.0, random_state=3)
    shuffled_mask = mask[shuffled.index.to_numpy()]
    out = scramble_test_rows(shuffled, shuffled_mask, None, 5).sort_index()
    pd.testing.assert_frame_equal(out, scramble_test_rows(frame, mask, None, 5))


def test_a_categorical_column_keeps_its_categories() -> None:
    level = pd.Categorical(
        ["lo", "mid", "hi", "hi", "mid", "hi", None], categories=LEVELS, ordered=True
    )
    frame = pd.DataFrame({"level": level})
    out = scramble_test_rows(frame, every_row(7), None, 0)
    assert out["level"].dtype == frame["level"].dtype
    assert list(out["level"].cat.categories) == LEVELS
    assert pd.isna(out["level"].iloc[6])
    assert set(out["level"].dropna()) <= {"hi", "mid"}  # counts 3, 2, 1; keep 2


def test_a_mask_of_the_wrong_length_raises() -> None:
    with pytest.raises(HexInputError, match="mask"):
        scramble_test_rows(pd.DataFrame({"x": [1, 2, 3]}), every_row(2), None, 0)


def test_an_empty_mask_returns_an_equal_copy() -> None:
    frame = random_frame(20)
    out = scramble_test_rows(frame, np.zeros(20, dtype=bool), None, 0)
    pd.testing.assert_frame_equal(out, frame)
    assert out is not frame


def test_an_empty_mask_keeps_int64_category_and_datetime_columns() -> None:
    frame = mixed_dtype_frame()
    out = scramble_test_rows(frame, np.zeros(len(frame), dtype=bool), None, 0)
    pd.testing.assert_frame_equal(out, frame)
    assert out.dtypes.equals(frame.dtypes)
    assert out is not frame


# --- generator pin and dtype preservation --------------------------------------------


def test_canary_salt_is_the_bytes_hexc() -> None:
    assert data.CANARY_SALT == int.from_bytes(b"HEXC", "big")


def test_scramble_reproduces_from_the_documented_generator() -> None:
    """Pins the stream: ``default_rng([seed, CANARY_SALT])``, columns in order."""
    gen = np.random.default_rng(11)
    index = gen.permutation(np.arange(40, 90))  # frame order differs from id order
    frame = pd.DataFrame(
        {
            "a": gen.normal(size=50).round(3),
            "t": [day(d) for d in gen.integers(0, 30, size=50)],
            "b": gen.choice(list("pqrst"), size=50),
            "c": gen.integers(0, 12, size=50),
        },
        index=index,
    )
    mask = gen.random(50) < 0.4
    seed = 21
    out = scramble_test_rows(frame, mask, "t", seed)

    rng = np.random.default_rng([seed, data.CANARY_SALT])
    test_rows = frame[mask].sort_index()
    for name in ["a", "b", "c"]:  # "t" is skipped and draws nothing
        expected = tilt_column(test_rows[name], rng)
        pd.testing.assert_series_equal(out.loc[expected.index, name], expected)
    pd.testing.assert_series_equal(out["t"], frame["t"])
    assert not out.equals(frame)


AWKWARD_COLUMNS = ["when", "when_tz", "gap", "count", "maybe", "level"]


@pytest.mark.parametrize("column", AWKWARD_COLUMNS)
@pytest.mark.parametrize("seed", range(3))
def test_scramble_keeps_dtype_and_missing_cells_of_awkward_dtypes(
    column: str, seed: int
) -> None:
    frame = mixed_dtype_frame()
    mask = np.array([True, True, False, True, True, True, True, True])
    out = scramble_test_rows(frame, mask, None, seed)
    assert out[column].dtype == frame[column].dtype
    assert out[column].isna().tolist() == frame[column].isna().tolist()
    pd.testing.assert_series_equal(out.loc[~mask, column], frame.loc[~mask, column])
    original = frame.loc[mask, column].dropna()
    assert set(out.loc[mask, column].dropna()) <= set(original)
    assert out.loc[mask, column].nunique() >= 2


def test_a_datetime_column_that_is_not_the_time_column_is_tilted() -> None:
    frame = mixed_dtype_frame()
    out = scramble_test_rows(frame, every_row(len(frame)), "when_tz", 0)
    pd.testing.assert_series_equal(out["when_tz"], frame["when_tz"])  # the time column
    assert out["when"].dtype == frame["when"].dtype
    # day 1: 2 rows, day 2: 2, day 3: 3. The top two categories are day 3 and, by
    # the tie rule (ascending str), day 1.
    assert set(out["when"].dropna()) <= {pd.Timestamp(day(3)), pd.Timestamp(day(1))}
    assert out["when"].isna().tolist() == frame["when"].isna().tolist()


def test_a_timezone_aware_column_keeps_its_timezone() -> None:
    frame = mixed_dtype_frame()
    out = scramble_test_rows(frame, every_row(len(frame)), None, 0)
    assert str(out["when_tz"].dt.tz) == "Europe/Paris"
    assert set(out["when_tz"].dropna()) <= set(frame["when_tz"].dropna())


def test_a_timedelta_column_stays_a_timedelta() -> None:
    frame = mixed_dtype_frame()
    out = scramble_test_rows(frame, every_row(len(frame)), None, 0)
    assert out["gap"].dtype == frame["gap"].dtype
    # 1h: 2 rows, 2h: 2, 3h: 3. The top two are 3h and, by ascending str, 1h.
    assert set(out["gap"].dropna()) <= {pd.Timedelta(hours=3), pd.Timedelta(hours=1)}


def test_nullable_int_and_boolean_columns_with_missing_values() -> None:
    frame = mixed_dtype_frame()
    out = scramble_test_rows(frame, every_row(len(frame)), None, 0)
    assert str(out["count"].dtype) == "Int64"
    assert out["count"].isna().tolist() == [False, True] + [False] * 6
    assert min(out["count"].dropna()) >= 7  # values 5, 7, 9: pool is the upper half
    assert str(out["maybe"].dtype) == "boolean"
    assert out["maybe"].isna().tolist() == frame["maybe"].isna().tolist()
    # two distinct non-missing values (True x3, False x3) are swapped in place
    swapped = [v if v is pd.NA else not v for v in frame["maybe"]]
    assert out["maybe"].tolist() == swapped


# --- input validation ----------------------------------------------------------------


@pytest.mark.parametrize(
    "mask",
    [
        np.array([1, 0, 1, 0]),
        np.array([1.0, 0.0, 1.0, 0.0]),
        np.array([True, False, True, None], dtype=object),
        np.array([[True, False], [True, False]]),
        pd.Series([1, 0, 1, 0]),
        pd.Series([1.0, 0.0, 1.0, 0.0]),
        pd.Series([True, False, True, None], dtype="boolean"),
    ],
    ids=["int", "float", "object", "2d", "int-series", "float-series", "boolean-na"],
)
def test_a_mask_that_is_not_boolean_raises(mask: object) -> None:
    frame = pd.DataFrame({"x": [1, 2, 3, 4]})
    with pytest.raises(HexInputError, match="mask"):
        scramble_test_rows(frame, mask, None, 0)  # type: ignore[arg-type]


def test_a_series_mask_is_aligned_with_the_frame_by_label() -> None:
    frame = random_frame(40).set_index(np.arange(500, 540))
    array = hash_fraction(np.arange(40, dtype=np.int64), 3) < 0.4
    expected = scramble_test_rows(frame, array, None, 2)

    def scrambled_with(mask: pd.Series) -> pd.DataFrame:
        return scramble_test_rows(frame, mask, None, 2)

    labelled = pd.Series(array, index=frame.index)
    pd.testing.assert_frame_equal(scrambled_with(labelled), expected)
    reordered = labelled.iloc[::-1]  # same labels, other order: not positional
    pd.testing.assert_frame_equal(scrambled_with(reordered), expected)
    pd.testing.assert_frame_equal(scrambled_with(labelled.astype("boolean")), expected)


@pytest.mark.parametrize(
    "labels",
    [[0, 1, 2, 9], [0, 1, 1, 3], [0, 1, 2]],
    ids=["other-label", "duplicate-label", "short"],
)
def test_a_series_mask_with_other_labels_raises(labels: list[int]) -> None:
    frame = pd.DataFrame({"x": [1, 2, 3, 4]})
    mask = pd.Series([True, False, True, False][: len(labels)], index=labels)
    with pytest.raises(HexInputError, match="mask"):
        scramble_test_rows(frame, mask, None, 0)


def test_a_time_column_missing_from_the_frame_raises() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(HexInputError, match="'when'") as caught:
        scramble_test_rows(frame, every_row(3), "when", 0)
    assert "missing" in str(caught.value)


def test_unhashable_object_values_raise_with_the_column_name() -> None:
    frame = pd.DataFrame({"ok": [1, 2, 3, 4], "tags": [[1], [2], [3], [1]]})
    with pytest.raises(HexInputError, match="'tags'") as caught:
        scramble_test_rows(frame, every_row(4), None, 0)
    assert not isinstance(caught.value, TypeError)


@pytest.mark.parametrize("bad", [[1], {"a": 1}, {1, 2}, np.array([1, 2])])
def test_tilt_column_rejects_unhashable_values(bad: object) -> None:
    values = pd.Series([bad, bad, bad], name="payload", dtype=object)
    with pytest.raises(HexInputError, match="'payload'"):
        tilt_column(values, np.random.default_rng(0))
    with pytest.raises(HexInputError, match="cannot be hashed"):
        tilt_column(values.rename(None), np.random.default_rng(0))


# --- scramble properties (Requirement 17.2) ------------------------------------------


@st.composite
def scramble_cases(draw: st.DrawFn) -> tuple[pd.DataFrame, np.ndarray, int]:
    """A frame with numeric, nullable, text, boolean, categorical and time columns.

    It also holds a datetime column that is not the time column, a timezone-aware
    one and a timedelta one, all with missing values.
    """
    n = draw(st.integers(1, 16))
    index = draw(st.lists(st.integers(0, 10**6), min_size=n, max_size=n, unique=True))

    def column(elements: st.SearchStrategy) -> list:
        return draw(st.lists(elements, min_size=n, max_size=n))

    small = st.integers(-3, 3)
    floats = st.one_of(st.none(), small.map(float), st.floats(-1e6, 1e6))

    def maybe(*values: object) -> st.SearchStrategy:
        return st.sampled_from([None, *values])

    when = stamps(column(maybe(*range(4))))
    frame = pd.DataFrame(
        {
            "num": np.array([np.nan if v is None else v for v in column(floats)]),
            "count": pd.array(column(maybe(*range(-3, 4))), dtype="Int64"),
            "text": np.array(column(maybe(*"abcde")), dtype=object),
            "label": pd.array(column(maybe(*"xyz")), dtype="string"),
            "flag": np.array(column(st.booleans()), dtype=bool),
            "maybe": pd.array(column(maybe(True, False)), dtype="boolean"),
            "level": pd.Categorical(column(maybe(*LEVELS[:3])), categories=LEVELS),
            "when": when,
            "when_tz": stamps(column(maybe(*range(4)))).tz_localize("Asia/Tokyo"),
            "gap": pd.to_timedelta(column(maybe(*range(4))), unit="h"),
            "target": np.array(column(st.sampled_from([0, 1])), dtype=np.int64),
            "t": [day(d) for d in column(st.integers(0, 30))],
        },
        index=index,
    )
    mask = np.array(column(st.booleans()), dtype=bool)
    return frame, mask, draw(seeds)


@given(case=scramble_cases())
def test_scramble_guarantees_hold_on_arbitrary_frames(
    case: tuple[pd.DataFrame, np.ndarray, int],
) -> None:
    """One frame draw checks every scramble guarantee (building it is the slow part).

    Covers Requirement 17.2 (no unseen value, no constant column, both classes of
    a binary target kept) plus the frame invariants and determinism of 1.5.
    """
    frame, mask, seed = case
    before = frame.copy(deep=True)
    out = scramble_test_rows(frame, mask, "t", seed)

    # The input is left alone; shape, index, dtypes, training rows, time column.
    pd.testing.assert_frame_equal(frame, before)
    assert out.index.equals(frame.index)
    assert out.shape == frame.shape
    assert out.dtypes.equals(frame.dtypes)
    pd.testing.assert_frame_equal(out[~mask], frame[~mask])
    pd.testing.assert_series_equal(out["t"], frame["t"])

    # Same frame, mask and seed give the same result.
    pd.testing.assert_frame_equal(out, scramble_test_rows(frame, mask, "t", seed))

    for name in frame.columns:
        original, scrambled = frame.loc[mask, name], out.loc[mask, name]
        # Missing values stay in place and no unseen value appears.
        assert scrambled.isna().tolist() == original.isna().tolist(), name
        assert set(scrambled.dropna().tolist()) <= set(original.dropna().tolist()), name
        # A column with two or more distinct test values never becomes constant.
        if original.nunique() >= 2:
            assert scrambled.nunique() >= 2, name

    # A target with exactly two distinct test values keeps both, with counts swapped.
    counts = frame.loc[mask, "target"].value_counts()
    if len(counts) == 2:
        swapped = out.loc[mask, "target"].value_counts()
        assert swapped[0] == counts[1]
        assert swapped[1] == counts[0]


@given(
    base=st.permutations([1.0, 2.0, 3.0]),
    extra=st.lists(st.sampled_from([1.0, 2.0, 3.0]), max_size=3),
    as_text=st.booleans(),
    seed=seeds,
)
def test_three_valued_columns_with_few_test_rows_never_become_constant(
    base: list[float], extra: list[float], as_text: bool, seed: int
) -> None:
    values = base + extra
    frame = pd.DataFrame({"x": [str(v) for v in values] if as_text else values})
    out = scramble_test_rows(frame, every_row(len(values)), None, seed)
    assert out["x"].nunique() >= 2


@given(
    values=st.lists(st.one_of(st.none(), st.integers(-20, 20)), min_size=3),
    dtype=st.sampled_from(["Int64", "float64"]),
    seed=seeds,
)
def test_tilt_writes_numeric_values_at_or_above_the_pool_minimum(
    values: list[int | None], dtype: str, seed: int
) -> None:
    present = sorted(v for v in values if v is not None)
    distinct = sorted(set(present))
    assume(len(distinct) >= 3)
    pool_min = min(present[len(present) // 2], distinct[-2])
    out = tilt_column(pd.Series(values, dtype=dtype), np.random.default_rng(seed))
    assert all(v >= pool_min for v in out.dropna())


@given(
    values=st.lists(st.one_of(st.none(), st.integers(-20, 20)), min_size=3),
    dtype=st.sampled_from(["Int64", "float64"]),
    seed=seeds,
)
def test_tilt_raises_the_mean_when_it_starts_at_or_below_the_pool_minimum(
    values: list[int | None], dtype: str, seed: int
) -> None:
    """Under the design the mean rises only when the old mean is not above the pool.

    Every written value is at or above the pool minimum, and the repair step makes
    at least two different values appear, so at least one is strictly above it. A
    mean that starts above the pool minimum can fall; that case is not claimed.
    """
    present = sorted(v for v in values if v is not None)
    distinct = sorted(set(present))
    assume(len(distinct) >= 3)
    pool_min = min(present[len(present) // 2], distinct[-2])
    series = pd.Series(values, dtype=dtype)
    assume(series.mean() <= pool_min)
    out = tilt_column(series, np.random.default_rng(seed))
    assert out.mean() > series.mean()


@given(
    pair=st.lists(st.integers(-50, 50), min_size=2, max_size=2, unique=True),
    picks=st.lists(st.sampled_from([0, 1, None]), min_size=2, max_size=30),
    as_text=st.booleans(),
    seed=seeds,
)
def test_tilt_on_a_two_valued_column_is_its_own_inverse(
    pair: list[int], picks: list[int | None], as_text: bool, seed: int
) -> None:
    values = [None if p is None else pair[p] for p in picks]
    if as_text:
        text = [None if v is None else str(v) for v in values]
        series = pd.Series(text, dtype=object)
    else:
        series = pd.Series(values, dtype="float64")
    assume(series.nunique() == 2)
    rng = np.random.default_rng(seed)
    once = tilt_column(series, rng)
    assert not once.equals(series)
    pd.testing.assert_series_equal(tilt_column(once, rng), series)


# --- load and split (Requirements 1.1 to 1.3, 1.5, 1.6, 10.6) ------------------------

RUN = "20251101-093012-a1b2c3"
N_ROWS = 60  # coprime with 7, so the time column below is a permutation


@pytest.fixture(autouse=True)
def clean_hex_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEX_MODE", raising=False)
    monkeypatch.delenv("HEX_RUN_ID", raising=False)


def app_frame(n: int = N_ROWS) -> pd.DataFrame:
    """Rows whose time order differs from the file order."""
    return pd.DataFrame(
        {
            "id": [f"r{(i * 11) % n}" for i in range(n)],
            "x": [round(0.37 * i + (i % 5) ** 2, 3) for i in range(n)],
            "c": [("red", "green", "blue")[i % 3] for i in range(n)],
            "y": [(i * 3) % 7 % 2 for i in range(n)],
            "t": [day((i * 7) % n) for i in range(n)],
        }
    )


def make_app(
    tmp_path: Path,
    *,
    time_column: str | None = None,
    fraction: float = 0.25,
    seed: int = 3,
    frame: pd.DataFrame | None = None,
) -> Path:
    app = tmp_path / "app"
    app.mkdir(parents=True)
    lines = [
        "[data]",
        'path = "data.csv"',
        "[target]",
        'column = "y"',
        "[metric]",
        'name = "roc_auc"',
        "threshold = 0.7",
        "[split]",
        f"test_fraction = {fraction}",
        f"seed = {seed}",
    ]
    if time_column is not None:
        lines.append(f'time_column = "{time_column}"')
    (app / "hex.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (frame if frame is not None else app_frame()).to_csv(
        app / "data.csv", index=False
    )
    return app


def file_frame(app: Path) -> pd.DataFrame:
    return pd.read_csv(app / "data.csv")


def config_for(time_column: str | None, fraction: float = 0.25, seed: int = 3):
    return SplitConfig(test_fraction=fraction, seed=seed, time_column=time_column)


def run_in_script(app: Path, code: str, **names) -> dict:
    """Run ``code`` as if it were module-level code of ``<app>/train.py``."""
    namespace = {"__file__": str(app / "train.py"), "hexperiment": hexperiment}
    namespace.update(names)
    exec(code, namespace)
    return namespace


# load: reading the file (Requirements 1.1, 1.6)


def test_load_missing_csv_names_the_resolved_path(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    (app / "data.csv").unlink()
    with pytest.raises(HexInputError) as error:
        data.load(app)
    assert str((app / "data.csv").resolve()) in str(error.value)
    assert "fetch_data.py" in str(error.value)


def test_load_empty_csv_names_the_resolved_path(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    (app / "data.csv").write_bytes(b"")
    with pytest.raises(HexInputError, match="is empty") as error:
        data.load(app)
    assert str((app / "data.csv").resolve()) in str(error.value)


def test_load_csv_with_a_header_and_no_rows_names_the_resolved_path(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path, frame=app_frame().iloc[:0])
    with pytest.raises(HexInputError, match="no data rows") as error:
        data.load(app)
    assert str((app / "data.csv").resolve()) in str(error.value)


def test_load_data_path_that_is_a_folder_names_the_resolved_path(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path)
    (app / "data.csv").unlink()
    (app / "data.csv").mkdir()
    with pytest.raises(HexInputError) as error:
        data.load(app)
    assert str((app / "data.csv").resolve()) in str(error.value)


def test_load_resolves_data_path_relative_to_the_app_folder(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    (app / "sub").mkdir()
    (app / "data.csv").rename(app / "sub" / "rows.csv")
    toml = (app / "hex.toml").read_text(encoding="utf-8")
    (app / "hex.toml").write_text(
        toml.replace('"data.csv"', '"sub/rows.csv"'), encoding="utf-8"
    )
    assert len(data.load(app)) == N_ROWS


def test_load_index_is_the_row_position_in_file_order(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    df = data.load(app)
    assert isinstance(df.index, pd.RangeIndex)
    assert df.index.tolist() == list(range(N_ROWS))
    assert df["id"].tolist() == file_frame(app)["id"].tolist()  # a column


def test_load_rejects_rows_with_more_fields_than_the_header(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    text = (app / "data.csv").read_text(encoding="utf-8").splitlines()
    # With one field too many on every row, pandas would use the first column as
    # the index and the row ids would no longer be the file positions.
    padded = [text[0]] + [line + ",extra" for line in text[1:]]
    (app / "data.csv").write_text("\n".join(padded) + "\n", encoding="utf-8")
    with pytest.raises(HexInputError, match="more fields than its header") as error:
        data.load(app)
    assert str((app / "data.csv").resolve()) in str(error.value)


def test_load_rejects_a_ragged_csv_and_names_the_path(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with (app / "data.csv").open("a", encoding="utf-8") as handle:
        handle.write("r1,2.0,red,1,2024-01-01,extra,extra\n")
    with pytest.raises(HexInputError, match="cannot be read as CSV") as error:
        data.load(app)
    assert str((app / "data.csv").resolve()) in str(error.value)


def test_load_accepts_a_string_app_dir(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    assert len(data.load(str(app))) == N_ROWS


def test_load_defaults_to_the_folder_of_the_calling_script(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    namespace = run_in_script(app, "df = hexperiment.load()")
    assert namespace["df"].attrs["hexperiment"]["app_dir"] == str(app.resolve())
    assert len(namespace["df"]) == N_ROWS


def test_load_without_app_dir_and_without_a_script_file_asks_for_app_dir() -> None:
    with pytest.raises(HexInputError, match="app_dir"):
        exec("hexperiment.load()", {"hexperiment": hexperiment})


def test_load_reports_an_invalid_execution_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = make_app(tmp_path)
    monkeypatch.setenv("HEX_MODE", "haunted")
    with pytest.raises(HexInputError, match="HEX_MODE"):
        data.load(app)


# load: execution modes (Requirement 1.5)


@pytest.mark.parametrize("mode", [None, "normal", "control"])
@pytest.mark.parametrize("time_column", [None, "t"])
def test_load_returns_the_file_values_outside_canary_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str | None,
    time_column: str | None,
) -> None:
    app = make_app(tmp_path, time_column=time_column)
    if mode is not None:
        monkeypatch.setenv("HEX_MODE", mode)
    pd.testing.assert_frame_equal(data.load(app), file_frame(app))


@pytest.mark.parametrize("time_column", [None, "t"])
def test_load_in_canary_mode_scrambles_exactly_the_test_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, time_column: str | None
) -> None:
    app = make_app(tmp_path, time_column=time_column)
    monkeypatch.setenv("HEX_MODE", "canary")
    df = data.load(app)

    original = file_frame(app)
    split_config = config_for(time_column)
    mask = data.test_mask(original, split_config)
    expected = scramble_test_rows(original, mask, time_column, split_config.seed)
    pd.testing.assert_frame_equal(df, expected)

    assert 0 < mask.sum() < N_ROWS
    assert not df[mask].equals(original[mask])  # the test rows changed
    pd.testing.assert_frame_equal(df[~mask], original[~mask])  # training rows did not
    assert df.index.tolist() == original.index.tolist()
    if time_column is not None:
        assert df["t"].tolist() == original["t"].tolist()


def test_load_in_canary_mode_uses_the_seed_from_hex_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HEX_MODE", "canary")
    one = data.load(make_app(tmp_path / "a", seed=1, time_column="t"))
    two = data.load(make_app(tmp_path / "b", seed=2, time_column="t"))
    # A time split ignores the seed, so the same rows are scrambled; the draws differ.
    assert one["t"].tolist() == two["t"].tolist()
    assert not one.equals(two)


# load: what split needs later (Requirements 1.2, 1.3)


def test_load_stores_the_app_dir_and_no_cutoff_for_a_hash_split(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path)
    assert data.load(app).attrs["hexperiment"] == {
        "app_dir": str(app.resolve()),
        "cutoff": None,
    }


def test_load_stores_the_cutoff_computed_from_the_whole_file(tmp_path: Path) -> None:
    app = make_app(tmp_path, time_column="t")
    stored = data.load(app).attrs["hexperiment"]
    expected = compute_cutoff(file_frame(app), config_for("t"))
    assert isinstance(expected, TimeCutoff)
    assert stored == {"app_dir": str(app.resolve()), "cutoff": expected}


def test_cutoff_is_the_same_in_canary_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = make_app(tmp_path, time_column="t")
    normal = data.load(app).attrs["hexperiment"]["cutoff"]
    monkeypatch.setenv("HEX_MODE", "canary")
    assert data.load(app).attrs["hexperiment"]["cutoff"] == normal


# load: the marker (Requirement 10.6)


@pytest.mark.parametrize("mode", [None, "normal", "canary", "control"])
def test_load_writes_no_marker_without_a_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str | None
) -> None:
    app = make_app(tmp_path)
    if mode is not None:
        monkeypatch.setenv("HEX_MODE", mode)
    data.load(app)
    assert not (app / ".hex").exists()


@pytest.mark.parametrize("mode", [ExecMode.NORMAL, ExecMode.CONTROL])
def test_load_writes_a_marker_without_ids_outside_canary_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: ExecMode
) -> None:
    app = make_app(tmp_path)
    monkeypatch.setenv("HEX_MODE", mode.value)
    monkeypatch.setenv("HEX_RUN_ID", RUN)
    data.load(app)
    assert read_load_marker(app, mode, RUN) == LoadMarker(mode, RUN, None)
    assert not (hex_dir(app, mode) / "scrambled_ids.npy").exists()


@pytest.mark.parametrize("time_column", [None, "t"])
def test_load_writes_the_scrambled_ids_in_canary_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, time_column: str | None
) -> None:
    app = make_app(tmp_path, time_column=time_column)
    monkeypatch.setenv("HEX_MODE", "canary")
    monkeypatch.setenv("HEX_RUN_ID", RUN)
    data.load(app)

    marker = read_load_marker(app, ExecMode.CANARY, RUN)
    mask = data.test_mask(file_frame(app), config_for(time_column))
    assert marker is not None
    assert marker.scrambled_ids.tolist() == np.flatnonzero(mask).tolist()
    assert len(marker.scrambled_ids) == mask.sum() > 0


def test_load_marker_carries_the_run_id_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = make_app(tmp_path)
    monkeypatch.setenv("HEX_RUN_ID", RUN)
    data.load(app)
    assert read_load_marker(app, ExecMode.NORMAL, RUN).run_id == RUN


# split (Requirements 1.2, 1.3)


def test_split_without_a_time_column_matches_split_frame(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    df = data.load(app)
    train, test = data.split(df)
    expected_train, expected_test = split_frame(file_frame(app), config_for(None))
    pd.testing.assert_frame_equal(train, expected_train)
    pd.testing.assert_frame_equal(test, expected_test)
    assert len(train) + len(test) == N_ROWS
    assert set(train.index).isdisjoint(test.index)


def test_split_with_a_time_column_puts_the_latest_rows_in_test(tmp_path: Path) -> None:
    app = make_app(tmp_path, time_column="t")
    train, test = data.split(data.load(app))
    assert len(test) == 15
    assert pd.to_datetime(train["t"]).max() < pd.to_datetime(test["t"]).min()


def test_split_returns_copies_that_keep_the_row_id_index(tmp_path: Path) -> None:
    df = data.load(make_app(tmp_path))
    train, test = data.split(df)
    assert train.index.is_monotonic_increasing
    assert train.index.isin(df.index).all()
    train.loc[train.index[0], "x"] = -1.0
    test.loc[test.index[0], "x"] = -1.0
    assert (df["x"] != -1.0).all()


def test_split_in_canary_mode_gives_the_same_row_ids_as_in_normal_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = make_app(tmp_path, time_column="t")
    normal_train, normal_test = data.split(data.load(app))
    monkeypatch.setenv("HEX_MODE", "canary")
    train, test = data.split(data.load(app))
    assert train.index.tolist() == normal_train.index.tolist()
    assert test.index.tolist() == normal_test.index.tolist()
    pd.testing.assert_frame_equal(train, normal_train)  # training rows unchanged


def late_rows_filtered(app: Path) -> tuple[pd.DataFrame, list[int]]:
    """Load, set x to NaN in 10 of the 15 test rows, and drop them with dropna()."""
    df = data.load(app)
    _, test = data.split(df)
    dropped = test.index[:10].tolist()
    df.loc[dropped, "x"] = np.nan
    return df.dropna(), dropped


def test_split_uses_the_stored_cutoff_after_filtering(tmp_path: Path) -> None:
    app = make_app(tmp_path, time_column="t")
    filtered, dropped = late_rows_filtered(app)
    full_train, full_test = data.split(data.load(app))
    assert "hexperiment" in filtered.attrs  # dropna keeps attrs

    train, test = data.split(filtered)

    assert len(filtered) == N_ROWS - 10
    assert train.index.tolist() == full_train.index.tolist()
    assert test.index.tolist() == sorted(set(full_test.index) - set(dropped))


def test_split_without_attrs_recomputes_the_cutoff_from_the_received_frame(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path, time_column="t")
    filtered, dropped = late_rows_filtered(app)
    full_train, _ = data.split(data.load(app))
    filtered.attrs.clear()

    namespace = run_in_script(app, "result = hexperiment.split(df)", df=filtered)
    train, test = namespace["result"]

    expected_train, expected_test = split_frame(filtered, config_for("t"))
    assert train.index.tolist() == expected_train.index.tolist()
    assert test.index.tolist() == expected_test.index.tolist()
    # The boundary moved: 8 rows that were training rows are now test rows.
    assert len(test) == 13
    assert set(full_train.index) - set(train.index) != set()
    assert not set(dropped) & set(filtered.index)


def test_split_without_attrs_gives_the_same_hash_split(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    df = data.load(app)
    expected_train, expected_test = data.split(df)
    cleared = df.copy()
    cleared.attrs.clear()
    namespace = run_in_script(app, "result = hexperiment.split(df)", df=cleared)
    train, test = namespace["result"]
    pd.testing.assert_frame_equal(train, expected_train)
    pd.testing.assert_frame_equal(test, expected_test)


def test_split_without_attrs_and_without_a_script_file_cannot_find_the_app(
    tmp_path: Path,
) -> None:
    df = data.load(make_app(tmp_path))
    df.attrs.clear()
    with pytest.raises(HexInputError, match="no __file__"):
        exec("hexperiment.split(df)", {"hexperiment": hexperiment, "df": df})


def test_split_rejects_attrs_that_load_did_not_write(tmp_path: Path) -> None:
    df = data.load(make_app(tmp_path))
    df.attrs["hexperiment"] = {"app_dir": 5}
    with pytest.raises(HexInputError, match="not what hexperiment.load stores"):
        data.split(df)


@pytest.mark.parametrize("time_column", [None, "t"])
def test_split_raises_when_filtering_removed_every_test_row(
    tmp_path: Path, time_column: str | None
) -> None:
    app = make_app(tmp_path, time_column=time_column)
    df = data.load(app)
    _, test = data.split(df)
    only_train = df.drop(index=test.index)
    with pytest.raises(HexInputError, match="test set is empty") as error:
        data.split(only_train)
    assert "before hexperiment.split" in str(error.value)


@pytest.mark.parametrize("time_column", [None, "t"])
def test_split_raises_when_filtering_removed_every_training_row(
    tmp_path: Path, time_column: str | None
) -> None:
    app = make_app(tmp_path, time_column=time_column)
    df = data.load(app)
    train, _ = data.split(df)
    only_test = df.drop(index=train.index)
    with pytest.raises(HexInputError, match="train set is empty"):
        data.split(only_test)


def test_split_raises_when_the_hash_split_leaves_no_test_rows(tmp_path: Path) -> None:
    app = make_app(tmp_path, fraction=0.01, frame=app_frame(5))
    df = data.load(app)
    assert not data.test_mask(df, config_for(None, fraction=0.01)).any()
    with pytest.raises(HexInputError, match="test set is empty"):
        data.split(df)


# --- validate_audit_inputs (Requirements 4.1 to 4.6) ---------------------------------

SCORES_TRAIN = np.linspace(0.1, 0.9, 8)
SCORES_TEST = np.array([0.2, 0.8, 0.3, 0.7])


def audit_config(
    *, positive: int | str | bool = 1, time_column: str | None = None
) -> HexConfig:
    raw: dict = {
        "data": {"path": "data.csv"},
        "target": {"column": "y", "positive": positive},
        "metric": {"name": "roc_auc", "threshold": 0.7},
    }
    if time_column is not None:
        raw["split"] = {"time_column": time_column}
    return parse_config(raw, source=Path("hex.toml"), app_dir=Path("."))


def audit_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "x": np.arange(8, dtype=float),
            "c": list("abababab"),
            "y": [0, 1] * 4,
            "t": [day(i) for i in range(8)],
        }
    )
    test = pd.DataFrame(
        {
            "x": [8.0, 9.0, 10.0, 11.0],
            "c": list("abab"),
            "y": [0, 1, 0, 1],
            "t": [day(i) for i in range(8, 12)],
        },
        index=np.arange(8, 12),
    )
    return train, test


def audit_args(**changes: object) -> dict:
    train, test = audit_frames()
    args = {
        "train": train,
        "test": test,
        "y_score": SCORES_TEST.copy(),
        "y_score_train": SCORES_TRAIN.copy(),
        "features": ["x", "c"],
        "config": audit_config(),
    }
    args.update(changes)
    return args


def run_validate(**changes: object) -> ValidatedInputs:
    return validate_audit_inputs(**audit_args(**changes))


def with_test_column(name: str, values: list) -> pd.DataFrame:
    test = audit_frames()[1]
    test[name] = values
    return test


def with_train_column(name: str, values: list) -> pd.DataFrame:
    train = audit_frames()[0]
    train[name] = values
    return train


def scores_with(index: int, value: float) -> np.ndarray:
    scores = SCORES_TEST.copy()
    scores[index] = value
    return scores


def train_scores_with(index: int, value: float) -> np.ndarray:
    scores = SCORES_TRAIN.copy()
    scores[index] = value
    return scores


REJECTED_AUDIT_INPUTS = {
    # Requirement 4.1
    "empty train": (lambda: {"train": audit_frames()[0].iloc[:0]}, ["train is empty"]),
    "empty test": (lambda: {"test": audit_frames()[1].iloc[:0]}, ["test is empty"]),
    # Requirement 4.2
    "target missing from train": (
        lambda: {"train": audit_frames()[0].drop(columns="y")},
        ["'y'", "missing from train"],
    ),
    "target missing from test": (
        lambda: {"test": audit_frames()[1].drop(columns="y")},
        ["'y'", "missing from test"],
    ),
    # Requirement 4.2: missing target values
    "target has missing values in train": (
        lambda: {"train": with_train_column("y", [0, 1, None, 1, 0, None, 0, 1])},
        ["'y'", "2 missing value(s) in train", "8 rows", "dropna(subset=['y'])"],
    ),
    "target has missing values in test": (
        lambda: {"test": with_test_column("y", [0, 1, np.nan, 1])},
        ["'y'", "1 missing value(s) in test", "4 rows"],
    ),
    "string target has a missing value in test": (
        lambda: {"test": with_test_column("y", ["no", "yes", None, "yes"])},
        ["'y'", "1 missing value(s) in test"],
    ),
    "nullable target has a missing value in train": (
        lambda: {
            "train": with_train_column(
                "y", pd.array([0, 1, 0, 1, 0, 1, 0, pd.NA], dtype="Int64")
            )
        },
        ["'y'", "1 missing value(s) in train"],
    ),
    # Requirement 4.5: no features
    "features is an empty list": (
        lambda: {"features": []},
        ["features is empty", "features=['age', 'plan']"],
    ),
    "features is an empty tuple": (lambda: {"features": ()}, ["features is empty"]),
    # Requirement 4.3
    "single-class test target": (
        lambda: {"test": with_test_column("y", [0, 0, 0, 0])},
        ["single class", "'y'", "positive value 1", "0 of 4", "distinct", "0"],
    ),
    "positive value matches no test row": (
        lambda: {"test": with_test_column("y", ["no", "yes", "no", "yes"])},
        ["single class", "positive value 1", "'no'", "'yes'"],
    ),
    # Requirement 4.4
    "y_score too short": (
        lambda: {"y_score": SCORES_TEST[:3]},
        ["y_score has 3 value(s) but test has 4 row(s)"],
    ),
    "y_score too long": (
        lambda: {"y_score": np.append(SCORES_TEST, 0.5)},
        ["y_score has 5 value(s) but test has 4 row(s)"],
    ),
    "y_score_train too short": (
        lambda: {"y_score_train": SCORES_TRAIN[:7]},
        ["y_score_train has 7 value(s) but train has 8 row(s)"],
    ),
    "y_score NaN": (
        lambda: {"y_score": scores_with(2, np.nan)},
        ["y_score has 1 NaN value(s)", "position 2"],
    ),
    "y_score_train NaN": (
        lambda: {"y_score_train": train_scores_with(5, np.nan)},
        ["y_score_train has 1 NaN value(s)", "position 5"],
    ),
    "y_score above 1": (
        lambda: {"y_score": scores_with(1, 1.5)},
        ["y_score has 1 value(s) outside [0, 1]", "position 1"],
    ),
    "y_score below 0": (
        lambda: {"y_score": scores_with(0, -0.01)},
        ["y_score has 1 value(s) outside [0, 1]", "position 0"],
    ),
    "y_score infinite": (
        lambda: {"y_score": scores_with(3, np.inf)},
        ["y_score has 1 value(s) outside [0, 1]"],
    ),
    "y_score_train out of range": (
        lambda: {"y_score_train": train_scores_with(0, 2.0)},
        ["y_score_train has 1 value(s) outside [0, 1]", "position 0"],
    ),
    "NaN and out of range together": (
        lambda: {"y_score": np.array([np.nan, 2.0, 0.5, 0.5])},
        ["1 NaN value(s) and 1 value(s) outside [0, 1]", "position 0"],
    ),
    "y_score missing NA in a nullable Series": (
        lambda: {"y_score": pd.Series([0.2, pd.NA, 0.3, 0.7], dtype="Float64")},
        ["1 NaN value(s)", "position 1"],
    ),
    "y_score is two-dimensional": (
        lambda: {"y_score": np.column_stack([1 - SCORES_TEST, SCORES_TEST])},
        ["y_score must be one-dimensional", "predict_proba(X)[:, 1]"],
    ),
    "y_score is not numeric": (
        lambda: {"y_score": ["a", "b", "c", "d"]},
        ["y_score cannot be converted to numbers"],
    ),
    "y_score is None": (
        lambda: {"y_score": None},
        ["y_score must be one-dimensional"],
    ),
    # Requirement 4.5
    "feature not in either frame": (
        lambda: {"features": ["x", "zzz"]},
        ["'zzz'", "missing from train and test"],
    ),
    "feature missing from test only": (
        lambda: {"test": audit_frames()[1].drop(columns="c")},
        ["'c'", "missing from test"],
    ),
    "feature missing from train only": (
        lambda: {"train": audit_frames()[0].drop(columns="x")},
        ["'x'", "missing from train"],
    ),
    "features is a bare string": (
        lambda: {"features": "x"},
        ["features is the string 'x'", "list"],
    ),
    # Requirement 4.6
    "time column missing from train": (
        lambda: {
            "config": audit_config(time_column="t"),
            "train": audit_frames()[0].drop(columns="t"),
        },
        ["time column 't'", "missing from the train frame"],
    ),
    "time column missing from test": (
        lambda: {
            "config": audit_config(time_column="t"),
            "test": audit_frames()[1].drop(columns="t"),
        },
        ["time column 't'", "missing from the test frame"],
    ),
    "time column unparseable in test": (
        lambda: {
            "config": audit_config(time_column="t"),
            "test": with_test_column("t", ["soon", "later", "never", "maybe"]),
        },
        ["time column 't'", "the test frame", "cannot be parsed"],
    ),
    "time column unparseable in train": (
        lambda: {
            "config": audit_config(time_column="t"),
            "train": with_train_column("t", ["not a date"] * 8),
        },
        ["time column 't'", "the train frame", "cannot be parsed"],
    ),
    "time column has missing values": (
        lambda: {
            "config": audit_config(time_column="t"),
            "train": with_train_column(
                "t", [day(0), None] + [day(i) for i in range(6)]
            ),
        },
        ["time column 't'", "the train frame", "missing or unparseable"],
    ),
}


@pytest.mark.filterwarnings("ignore:Could not infer format:UserWarning")
@pytest.mark.parametrize(
    ("make_changes", "needles"),
    REJECTED_AUDIT_INPUTS.values(),
    ids=REJECTED_AUDIT_INPUTS.keys(),
)
def test_validate_audit_inputs_rejects_and_names_the_offender(
    make_changes, needles: list[str]
) -> None:
    with pytest.raises(HexInputError) as error:
        run_validate(**make_changes())
    for needle in needles:
        assert needle in str(error.value), (needle, str(error.value))
    assert error.value.problem
    assert error.value.fix
    assert str(error.value).endswith(error.value.fix)


def test_validate_audit_inputs_reports_the_first_problem_in_the_documented_order() -> (
    None
):
    train, test = audit_frames()
    # Each pair has two problems; the earlier rule must be the one reported.
    pairs = [
        (  # 4.1 before 4.2
            {"train": train.iloc[:0], "test": test.drop(columns="y")},
            "train is empty",
        ),
        (  # 4.2 before 4.4
            {"test": test.drop(columns="y"), "y_score": SCORES_TEST[:2]},
            "missing from test",
        ),
        (  # 4.2 (missing column) before 4.2 (missing values)
            {"test": test.drop(columns="y"), "train": train.assign(y=np.nan)},
            "missing from test",
        ),
        (  # 4.2 (missing values) before 4.4
            {"test": test.assign(y=[0, 1, np.nan, 1]), "y_score": SCORES_TEST[:2]},
            "missing value(s) in test",
        ),
        (  # 4.2 (train) before 4.2 (test)
            {
                "train": train.assign(y=np.nan),
                "test": test.assign(y=[0, 1, np.nan, 1]),
            },
            "in train",
        ),
        (  # 4.4 before 4.5 (empty features)
            {"y_score": scores_with(0, np.nan), "features": []},
            "NaN",
        ),
        (  # 4.5 (empty features) before 4.6
            {"features": [], "config": audit_config(time_column="gone")},
            "features is empty",
        ),
        (  # 4.4 before 4.5
            {"y_score": scores_with(0, np.nan), "features": ["nope"]},
            "NaN",
        ),
        (  # 4.5 before 4.6
            {"features": ["nope"], "config": audit_config(time_column="gone")},
            "'nope'",
        ),
        (  # 4.6 before 4.3
            {
                "config": audit_config(time_column="gone"),
                "test": with_test_column("y", [0, 0, 0, 0]),
            },
            "time column 'gone'",
        ),
    ]
    for changes, expected in pairs:
        with pytest.raises(HexInputError) as error:
            run_validate(**changes)
        assert expected in str(error.value), (expected, str(error.value))


def test_validate_audit_inputs_returns_features_labels_and_scores() -> None:
    result = run_validate()
    assert isinstance(result, ValidatedInputs)
    assert result.features == ("x", "c")
    assert result.y_true_train.tolist() == [False, True] * 4
    assert result.y_true_test.tolist() == [False, True, False, True]
    assert result.y_true_train.dtype == np.bool_
    assert result.y_score.dtype == np.float64
    assert np.array_equal(result.y_score, SCORES_TEST)
    assert result.y_score_train is not None
    assert np.array_equal(result.y_score_train, SCORES_TRAIN)


def test_validate_audit_inputs_accepts_a_missing_y_score_train() -> None:
    assert run_validate(y_score_train=None).y_score_train is None


@pytest.mark.parametrize(
    ("column", "positive"),
    [
        (["no", "yes", "no", "yes"], "yes"),
        ([False, True, False, True], True),
        ([0.0, 1.0, 0.0, 1.0], 1),
        ([1, 2, 1, 3], 2),
        (pd.array([False, True, False, True], dtype="boolean"), True),
        (pd.Categorical(["a", "b", "a", "b"]), "b"),
    ],
    ids=["string", "bool", "float", "multiclass", "nullable-bool", "categorical"],
)
def test_validate_audit_inputs_labels_rows_equal_to_the_positive_value(
    column: object, positive: object
) -> None:
    test = with_test_column("y", column)  # type: ignore[arg-type]
    result = run_validate(test=test, config=audit_config(positive=positive))  # type: ignore[arg-type]
    expected = test["y"].eq(positive).to_numpy(dtype=bool)
    assert result.y_true_test.tolist() == expected.tolist()
    assert result.y_true_test.any()
    assert not result.y_true_test.all()


@pytest.mark.parametrize(
    "make_scores",
    [
        list,
        tuple,
        lambda a: pd.Series(a, index=[40, 30, 20, 10][: len(a)]),
        lambda a: pd.Series(a, dtype="Float64"),
        lambda a: a.astype(np.float32).astype(np.float64),
    ],
    ids=["list", "tuple", "series-by-position", "nullable-series", "float32"],
)
def test_validate_audit_inputs_matches_scores_to_rows_by_position(make_scores) -> None:
    scores = np.array([0.25, 0.5, 0.75, 1.0])
    result = run_validate(y_score=make_scores(scores))
    assert np.array_equal(result.y_score, scores)
    assert result.y_score.dtype == np.float64


def test_validate_audit_inputs_accepts_scores_at_the_ends_of_the_range() -> None:
    result = run_validate(y_score=np.array([0.0, 1.0, 0.0, 1.0]))
    assert result.y_score.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_validate_audit_inputs_does_not_need_a_time_column_when_none_is_set() -> None:
    train, test = audit_frames()
    result = run_validate(train=train.drop(columns="t"), test=test.drop(columns="t"))
    assert result.features == ("x", "c")


def test_validate_audit_inputs_accepts_a_parseable_time_column() -> None:
    result = run_validate(config=audit_config(time_column="t"))
    assert result.features == ("x", "c")


def test_validate_audit_inputs_accepts_a_time_column_that_is_not_a_feature() -> None:
    result = run_validate(features=["x"], config=audit_config(time_column="t"))
    assert result.features == ("x",)


def test_validate_audit_inputs_returns_copies_and_modifies_nothing() -> None:
    train, test = audit_frames()
    train_before, test_before = train.copy(deep=True), test.copy(deep=True)
    scores, scores_train = SCORES_TEST.copy(), SCORES_TRAIN.copy()
    features = ["x", "c"]

    result = validate_audit_inputs(
        train, test, scores, scores_train, features, audit_config()
    )

    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(test, test_before)
    assert np.array_equal(scores, SCORES_TEST)
    assert np.array_equal(scores_train, SCORES_TRAIN)
    assert features == ["x", "c"]
    assert not np.shares_memory(result.y_score, scores)
    assert result.y_score_train is not None
    assert not np.shares_memory(result.y_score_train, scores_train)
    scores[0] = 0.99  # a later change by the caller does not reach the result
    assert result.y_score[0] == SCORES_TEST[0]


@given(
    scores=st.lists(
        st.one_of(st.floats(0, 1), st.floats(allow_nan=True, allow_infinity=True)),
        min_size=4,
        max_size=4,
    ),
    train_scores=st.booleans(),
)
def test_a_score_is_accepted_exactly_when_it_is_a_probability(
    scores: list[float], train_scores: bool
) -> None:
    valid = all(0.0 <= s <= 1.0 for s in scores)
    name = "y_score_train" if train_scores else "y_score"
    values = (scores + scores)[:8] if train_scores else scores
    if valid:
        result = run_validate(**{name: values})
        assert np.array_equal(getattr(result, name), values)
    else:
        with pytest.raises(HexInputError, match=name):
            run_validate(**{name: values})


# --- validate_snapshot_inputs (Requirements 2.4, 4.1, 4.4, 4.5) ----------------------

REJECTED_SNAPSHOT_INPUTS = {
    "empty train": (lambda: {"train": audit_frames()[0].iloc[:0]}, ["train is empty"]),
    "feature missing from train": (
        lambda: {"features": ["x", "zzz"]},
        ["'zzz'", "missing from train"],
    ),
    "features is a bare string": (
        lambda: {"features": "x"},
        ["features is the string 'x'"],
    ),
    "y_score_train too short": (
        lambda: {"y_score_train": SCORES_TRAIN[:7]},
        ["y_score_train has 7 value(s) but train has 8 row(s)"],
    ),
    "y_score_train NaN": (
        lambda: {"y_score_train": train_scores_with(3, np.nan)},
        ["y_score_train has 1 NaN value(s)", "position 3"],
    ),
    "y_score_train out of range": (
        lambda: {"y_score_train": train_scores_with(7, -0.5)},
        ["y_score_train has 1 value(s) outside [0, 1]", "position 7"],
    ),
    "y_score_train two-dimensional": (
        lambda: {"y_score_train": np.column_stack([SCORES_TRAIN, SCORES_TRAIN])},
        ["y_score_train must be one-dimensional"],
    ),
}


def snapshot_args(**changes: object) -> dict:
    args = {
        "train": audit_frames()[0],
        "y_score_train": SCORES_TRAIN.copy(),
        "features": ["x", "c"],
    }
    args.update(changes)
    return args


@pytest.mark.parametrize(
    ("make_changes", "needles"),
    REJECTED_SNAPSHOT_INPUTS.values(),
    ids=REJECTED_SNAPSHOT_INPUTS.keys(),
)
def test_validate_snapshot_inputs_rejects_training_side_problems(
    make_changes, needles: list[str]
) -> None:
    with pytest.raises(HexInputError) as error:
        validate_snapshot_inputs(**snapshot_args(**make_changes()))
    for needle in needles:
        assert needle in str(error.value), (needle, str(error.value))
    assert error.value.fix


def test_validate_snapshot_inputs_returns_features_and_a_score_copy() -> None:
    scores = SCORES_TRAIN.copy()
    result = validate_snapshot_inputs(
        audit_frames()[0], scores, ["x", "c"]
    )
    assert isinstance(result, SnapshotInputs)
    assert result.features == ("x", "c")
    assert result.y_score_train is not None
    assert result.y_score_train.dtype == np.float64
    assert np.array_equal(result.y_score_train, SCORES_TRAIN)
    assert not np.shares_memory(result.y_score_train, scores)


def test_validate_snapshot_inputs_accepts_a_missing_y_score_train() -> None:
    result = validate_snapshot_inputs(audit_frames()[0], None, ["x"])
    assert result == SnapshotInputs(features=("x",), y_score_train=None)


def test_validate_snapshot_inputs_does_not_modify_its_arguments() -> None:
    train = audit_frames()[0]
    before = train.copy(deep=True)
    scores = SCORES_TRAIN.copy()
    validate_snapshot_inputs(train, scores, ["x", "c"])
    pd.testing.assert_frame_equal(train, before)
    assert np.array_equal(scores, SCORES_TRAIN)


@pytest.mark.parametrize(
    ("make_changes", "needle"),
    [
        (lambda: {"test": with_test_column("y", [1, 1, 1, 1])}, "single class"),
        (lambda: {"y_score": SCORES_TEST[:2]}, "y_score has 2 value(s)"),
        (lambda: {"y_score": scores_with(0, np.nan)}, "y_score has 1 NaN"),
        (
            lambda: {
                "config": audit_config(time_column="t"),
                "train": audit_frames()[0].drop(columns="t"),
                "test": audit_frames()[1].drop(columns="t"),
            },
            "time column 't'",
        ),
        (lambda: {"test": audit_frames()[1].drop(columns=["y", "c"])}, "missing from"),
        (lambda: {"test": audit_frames()[1].iloc[:0]}, "test is empty"),
        (
            lambda: {"test": with_test_column("y", [0, 1, np.nan, 1])},
            "missing value(s) in test",
        ),
    ],
    ids=[
        "single-class test target",
        "wrong y_score length",
        "NaN y_score",
        "missing time column",
        "test frame without columns",
        "empty test",
        "missing test target values",
    ],
)
def test_snapshot_validation_accepts_what_only_scoring_needs(
    make_changes, needle: str
) -> None:
    changes = make_changes()
    with pytest.raises(HexInputError, match=re.escape(needle)):
        run_validate(**changes)
    # The snapshot needs only the training frame, the features, y_score_train.
    train = changes.get("train", audit_frames()[0])
    result = validate_snapshot_inputs(train, SCORES_TRAIN, ["x"])
    assert result.features == ("x",)


@given(
    scores=st.lists(
        st.one_of(st.floats(0, 1), st.floats(allow_nan=True, allow_infinity=True)),
        min_size=8,
        max_size=8,
    )
)
def test_snapshot_inputs_accept_y_score_train_exactly_when_all_are_probabilities(
    scores: list[float],
) -> None:
    valid = all(0.0 <= s <= 1.0 for s in scores)
    if valid:
        result = validate_snapshot_inputs(audit_frames()[0], scores, ["x"])
        assert result.y_score_train is not None
        assert result.y_score_train.tolist() == scores
    else:
        with pytest.raises(HexInputError, match="y_score_train"):
            validate_snapshot_inputs(audit_frames()[0], scores, ["x"])
