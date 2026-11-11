import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from hexperiment import HexInputError, data
from hexperiment.config import SplitConfig
from hexperiment.data import (
    TimeCutoff,
    compute_cutoff,
    hash_fraction,
    parse_times,
    scramble_test_rows,
    split_frame,
    tilt_column,
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
