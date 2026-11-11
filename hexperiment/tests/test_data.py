import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hexperiment import HexInputError, data
from hexperiment.config import SplitConfig
from hexperiment.data import (
    TimeCutoff,
    compute_cutoff,
    hash_fraction,
    parse_times,
    split_frame,
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
