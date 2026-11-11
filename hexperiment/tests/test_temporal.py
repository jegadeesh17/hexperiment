import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hexperiment.checks.temporal import check_temporal
from hexperiment.config import SplitConfig
from hexperiment.data import split_frame
from hexperiment.errors import HexInputError
from hexperiment.models import CheckResult, Outcome, Severity


def day(n: int) -> str:
    return (pd.Timestamp("2024-01-01") + pd.Timedelta(days=n)).strftime("%Y-%m-%d")


def frame(times: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"t": times, "x": np.arange(len(times))})


def days(values: list[int]) -> pd.DataFrame:
    return frame([day(v) for v in values])


# --- leaking and clean cases (Requirements 7.1, 7.2, 17.1) ---------------------------


def test_training_rows_from_the_future_are_the_prophet() -> None:
    train, test = days([1, 2, 3, 4, 5]), days([3, 5, 6, 7])
    result = check_temporal(train, test, "t")

    assert result.name == "temporal"
    assert result.ghost == "The Prophet"
    assert result.outcome is Outcome.FAILED
    (finding,) = result.findings
    assert finding.check == "temporal"
    assert finding.ghost == "The Prophet"
    assert finding.severity is Severity.ERROR
    assert finding.location == "t"
    # Only the test row on day 3 is earlier than the latest training row (day 5).
    assert finding.evidence == {
        "test_rows_earlier": 1,
        "latest_train": "2024-01-06T00:00:00+00:00",
        "earliest_test": "2024-01-04T00:00:00+00:00",
    }
    assert finding.fix == "split by time with hexperiment.split"
    assert "1 test row(s)" in finding.message


def test_a_random_split_of_time_ordered_rows_is_flagged() -> None:
    df = days(list(range(40)))
    train, test = split_frame(df, SplitConfig(test_fraction=0.3, seed=1))
    result = check_temporal(train, test, "t")
    assert result.outcome is Outcome.FAILED
    assert result.findings[0].evidence["test_rows_earlier"] > 0


def test_a_split_by_time_passes() -> None:
    df = days([5, 3, 9, 1, 7, 2, 8, 4, 6, 0])
    train, test = split_frame(df, SplitConfig(test_fraction=0.3, time_column="t"))
    assert check_temporal(train, test, "t") == CheckResult(
        "temporal", "The Prophet", Outcome.PASSED, ()
    )


# --- boundaries ----------------------------------------------------------------------


def test_equal_timestamps_at_the_boundary_are_not_a_leak() -> None:
    result = check_temporal(days([1, 2, 5]), days([5, 5, 8]), "t")
    assert result.outcome is Outcome.PASSED
    assert result.findings == ()


def test_one_second_of_overlap_is_a_leak() -> None:
    train = frame(["2024-01-05 12:00:01"])
    test = frame(["2024-01-05 12:00:00", "2024-01-06 00:00:00"])
    result = check_temporal(train, test, "t")
    assert result.outcome is Outcome.FAILED
    assert result.findings[0].evidence["test_rows_earlier"] == 1


def test_the_count_includes_only_test_rows_strictly_earlier_than_the_latest_train() -> (
    None
):
    train, test = days([4, 5]), days([2, 3, 5, 6])
    (finding,) = check_temporal(train, test, "t").findings
    assert finding.evidence["test_rows_earlier"] == 2  # days 2 and 3, not day 5


def test_no_time_column_is_not_applicable() -> None:
    train, test = days([9]), days([1])
    result = check_temporal(train, test, None)
    assert result == CheckResult(
        "temporal", "The Prophet", Outcome.NOT_APPLICABLE, ()
    )
    # Not even the columns are read.
    no_t = pd.DataFrame({"x": [1, 2]})
    assert check_temporal(no_t, no_t, None).outcome is Outcome.NOT_APPLICABLE


def test_timestamps_with_different_timezones_are_compared_as_instants() -> None:
    # The latest training instant is 10:00 UTC and the earliest test instant is
    # 09:30 UTC, although "12:00" sorts after "09:30" as text.
    train = frame(["2024-01-05T08:00:00Z", "2024-01-05T12:00:00+02:00"])
    test = frame(["2024-01-05T09:30:00Z", "2024-01-06T00:00:00Z"])
    result = check_temporal(train, test, "t")
    assert result.outcome is Outcome.FAILED
    assert result.findings[0].evidence["latest_train"] == "2024-01-05T10:00:00+00:00"


def test_a_later_wall_clock_time_in_an_earlier_instant_is_not_a_leak() -> None:
    # 10:00+02:00 is 08:00 UTC, before the test instant 09:00 UTC.
    train = frame(["2024-01-05T10:00:00+02:00"])
    test = frame(["2024-01-05T09:00:00Z"])
    assert check_temporal(train, test, "t").outcome is Outcome.PASSED


def test_naive_timestamps_are_read_as_utc() -> None:
    train = frame(["2024-01-05 10:00:00"])
    test = frame(["2024-01-05T10:00:00Z"])
    assert check_temporal(train, test, "t").outcome is Outcome.PASSED
    later = frame(["2024-01-05T09:59:59Z"])
    assert check_temporal(train, later, "t").outcome is Outcome.FAILED


@pytest.mark.parametrize("which", ["train", "test"])
def test_an_unparseable_time_raises_naming_the_frame(which: str) -> None:
    good, bad = days([1, 2]), frame(["2024-01-03", "someday"])
    train, test = (bad, good) if which == "train" else (good, bad)
    with pytest.raises(HexInputError, match=f"the {which} frame") as error:
        check_temporal(train, test, "t")
    assert "'t'" in str(error.value)


@pytest.mark.parametrize("which", ["train", "test"])
def test_a_missing_time_column_raises_naming_the_frame(which: str) -> None:
    good, bad = days([1, 2]), pd.DataFrame({"x": [1, 2]})
    train, test = (bad, good) if which == "train" else (good, bad)
    with pytest.raises(HexInputError, match=f"missing from the {which} frame"):
        check_temporal(train, test, "t")


def test_the_arguments_are_not_modified() -> None:
    train, test = days([1, 2, 5]), days([3, 6])
    before = (train.copy(deep=True), test.copy(deep=True))
    check_temporal(train, test, "t")
    pd.testing.assert_frame_equal(train, before[0])
    pd.testing.assert_frame_equal(test, before[1])


# --- a clean synthetic dataset (Requirement 17.4) ------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_time_split_of_random_timestamps_has_no_error(seed: int) -> None:
    gen = np.random.default_rng(seed)
    stamps = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        gen.integers(0, 365 * 24 * 3600, size=2000), unit="s"
    )
    df = pd.DataFrame(
        {"t": stamps.strftime("%Y-%m-%d %H:%M:%S"), "x": gen.normal(size=2000)}
    )
    train, test = split_frame(df, SplitConfig(time_column="t"))
    result = check_temporal(train, test, "t")
    assert not [f for f in result.findings if f.severity is Severity.ERROR]
    assert result.outcome is Outcome.PASSED


# --- properties (Requirements 17.2, 7.2) ---------------------------------------------


@given(
    train=st.lists(st.integers(0, 12), min_size=1, max_size=20),
    test=st.lists(st.integers(0, 12), min_size=1, max_size=20),
    order=st.randoms(use_true_random=False),
)
def test_the_result_follows_the_times_and_ignores_row_order(
    train: list[int], test: list[int], order
) -> None:
    result = check_temporal(days(train), days(test), "t")

    leaks = max(train) > min(test)
    assert (result.outcome is Outcome.FAILED) == leaks
    if leaks:
        (finding,) = result.findings
        assert finding.evidence["test_rows_earlier"] == sum(
            t < max(train) for t in test
        )
        assert finding.evidence["test_rows_earlier"] >= 1
    else:
        assert result.findings == ()

    shuffled_train, shuffled_test = train[:], test[:]
    order.shuffle(shuffled_train)
    order.shuffle(shuffled_test)
    assert check_temporal(days(shuffled_train), days(shuffled_test), "t") == result
