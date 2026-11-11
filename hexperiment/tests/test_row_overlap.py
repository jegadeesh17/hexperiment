import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hexperiment.checks.row_overlap import check_row_overlap
from hexperiment.config import SplitConfig
from hexperiment.data import split_frame
from hexperiment.errors import HexInputError
from hexperiment.models import CheckResult, Outcome, Severity

FEATURES = ["a", "b"]


def frame(rows: list[tuple], index: list[int] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["a", "b", "y"],
        index=index if index is not None else pd.RangeIndex(len(rows)),
    )


def train_rows(n: int = 20) -> list[tuple]:
    return [(float(i), f"c{i % 4}", i % 2) for i in range(n)]


def disjoint_rows(n: int = 20) -> list[tuple]:
    return [(float(i) + 0.5, f"c{i % 4}", i % 2) for i in range(n)]


def check(train: pd.DataFrame, test: pd.DataFrame, max_fraction: float = 0.0):
    return check_row_overlap(train, test, FEATURES, "y", max_fraction)


# --- leaking and clean cases (Requirements 5.1, 5.2, 17.1) ---------------------------


def test_copied_test_rows_are_reported_as_the_doppelganger() -> None:
    train = frame(train_rows())
    test_rows = disjoint_rows()
    test_rows[2], test_rows[9], test_rows[15] = train_rows()[3:6]
    result = check(train, frame(test_rows))

    assert result.name == "row_overlap"
    assert result.ghost == "The Doppelganger"
    assert result.outcome is Outcome.FAILED
    (finding,) = result.findings
    assert finding.check == "row_overlap"
    assert finding.ghost == "The Doppelganger"
    assert finding.severity is Severity.ERROR
    assert finding.location == "train vs test"
    assert finding.evidence == {"rows": 3, "test_rows": 20, "fraction": 0.15}
    assert finding.fix == (
        "after the split, drop every test row that also appears in the training rows"
    )
    assert "3 of 20" in finding.message


def test_disjoint_rows_pass() -> None:
    result = check(frame(train_rows()), frame(disjoint_rows()))
    assert result == CheckResult(
        "row_overlap", "The Doppelganger", Outcome.PASSED, ()
    )


def test_a_single_overlapping_row_is_an_error_at_the_default_threshold() -> None:
    test_rows = disjoint_rows()
    test_rows[0] = train_rows()[0]
    result = check(frame(train_rows()), frame(test_rows), max_fraction=0.0)
    assert result.outcome is Outcome.FAILED
    assert result.findings[0].evidence["rows"] == 1


# --- boundaries ----------------------------------------------------------------------


def two_of_ten() -> tuple[pd.DataFrame, pd.DataFrame]:
    test_rows = disjoint_rows(10)
    test_rows[1], test_rows[7] = train_rows()[0], train_rows()[1]
    return frame(train_rows()), frame(test_rows)


def test_a_fraction_equal_to_max_fraction_is_not_flagged() -> None:
    train, test = two_of_ten()
    result = check(train, test, max_fraction=0.2)
    assert result.outcome is Outcome.PASSED
    assert result.findings == ()


def test_a_fraction_above_max_fraction_is_flagged() -> None:
    train, test = two_of_ten()
    result = check(train, test, max_fraction=0.19)
    assert result.outcome is Outcome.FAILED
    assert result.findings[0].evidence["fraction"] == 0.2


def test_identical_missing_values_make_rows_equal() -> None:
    train = frame([(np.nan, "x", 1), (1.0, None, 0), (2.0, "y", 1)])
    test = frame([(np.nan, "x", 1), (1.0, None, 0), (9.0, "z", 0)])
    result = check(train, test)
    assert result.findings[0].evidence["rows"] == 2


def test_a_missing_value_does_not_equal_a_present_one() -> None:
    train = frame([(np.nan, "x", 1), (1.0, "x", 1)])
    test = frame([(2.0, "x", 1), (np.nan, "y", 1)])
    assert check(train, test).outcome is Outcome.PASSED


def test_rows_are_compared_over_features_and_target_only() -> None:
    train = frame(train_rows())
    train["noise"] = range(20)
    test = frame([train_rows()[4]])
    test["noise"] = [99]
    assert check(train, test).findings[0].evidence["rows"] == 1


def test_the_same_features_with_another_target_are_not_a_doppelganger() -> None:
    a, b, y = train_rows()[4]
    train = frame([(a, b, y)])
    test = frame([(a, b, 1 - y)])
    assert check(train, test).outcome is Outcome.PASSED


def test_a_row_that_appears_twice_in_train_counts_once() -> None:
    train = frame(train_rows() + [train_rows()[3], train_rows()[3]])
    test_rows = disjoint_rows(9) + [train_rows()[3]]
    finding = check(train, frame(test_rows)).findings[0]
    assert finding.evidence == {"rows": 1, "test_rows": 10, "fraction": 0.1}


def test_a_row_that_appears_twice_in_test_counts_twice() -> None:
    test_rows = disjoint_rows(8) + [train_rows()[3]] * 2
    finding = check(frame(train_rows()), frame(test_rows)).findings[0]
    assert finding.evidence == {"rows": 2, "test_rows": 10, "fraction": 0.2}


def test_the_target_listed_among_the_features_is_handled() -> None:
    train = frame(train_rows())
    test = frame([train_rows()[2]])
    result = check_row_overlap(train, test, ["a", "y"], "y", 0.0)
    assert result.findings[0].evidence["rows"] == 1


def test_an_empty_test_frame_has_nothing_to_overlap() -> None:
    result = check(frame(train_rows()), frame([]))
    assert result.outcome is Outcome.PASSED


def test_frames_with_different_indexes_are_compared_by_values() -> None:
    train = frame(train_rows(), index=list(range(100, 120)))
    test = frame([train_rows()[0]], index=[7])
    assert check(train, test).findings[0].evidence["rows"] == 1


def test_reordering_either_frame_gives_the_same_result() -> None:
    train = frame(train_rows())
    test_rows = disjoint_rows()
    test_rows[5] = train_rows()[8]
    test = frame(test_rows)
    expected = check(train, test)
    shuffled_train = train.sample(frac=1.0, random_state=1)
    shuffled_test = test.sample(frac=1.0, random_state=2)
    assert check(shuffled_train, shuffled_test) == expected


def test_the_arguments_are_not_modified() -> None:
    train, test = frame(train_rows()), frame(train_rows()[:5] + disjoint_rows(5))
    before = (train.copy(deep=True), test.copy(deep=True))
    check(train, test)
    pd.testing.assert_frame_equal(train, before[0])
    pd.testing.assert_frame_equal(test, before[1])


# --- columns with different or unusual dtypes (Requirement 5.1) ------------------


def keyed(train_a: object, test_a: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frames that differ only in column ``a``; ``b`` and ``y`` are constant."""

    def build(values: object) -> pd.DataFrame:
        return pd.DataFrame({"a": values, "b": "p", "y": 1})

    return build(train_a), build(test_a)


def overlap_count(train: pd.DataFrame, test: pd.DataFrame) -> int:
    result = check(train, test)
    return result.findings[0].evidence["rows"] if result.findings else 0


def stamps(*text: str, tz: str | None = None, unit: str = "ns") -> pd.DatetimeIndex:
    index = pd.to_datetime(list(text)).as_unit(unit)
    return index if tz is None else index.tz_localize(tz)


MIXED_DTYPE_CASES = {
    "bool": ([True, True], [True, False], 1),
    "nullable boolean": (
        pd.array([True, None], dtype="boolean"),
        pd.array([True, None, False], dtype="boolean"),
        2,
    ),
    "categorical with other categories": (
        pd.Categorical(["x", "y"]),
        pd.Categorical(["x", "z"]),
        1,
    ),
    "categorical vs text": (pd.Categorical(["x", "y"]), ["x", "z", "y"], 2),
    "text vs categorical": (["x", "y"], pd.Categorical(["y", "z"]), 1),
    "categorical with missing": (
        pd.Categorical(["x", None]),
        pd.Categorical(["x", None, "y"]),
        2,
    ),
    "nullable string": (
        pd.array(["x", None], dtype="string"),
        pd.array(["x", None, "q"], dtype="string"),
        2,
    ),
    "nullable string vs object": (
        pd.array(["x", None], dtype="string"),
        pd.Series(["x", None, "q"], dtype=object),
        2,
    ),
    "tz-aware, same zone": (
        stamps("2024-01-01", "2024-01-02", tz="Europe/Paris"),
        stamps("2024-01-01", "2024-01-09", tz="Europe/Paris"),
        1,
    ),
    "tz-aware, other zone, same instant": (
        stamps("2024-01-01 00:00", tz="Europe/Paris"),
        stamps("2023-12-31 23:00", "2024-01-01 00:00", tz="UTC"),
        1,
    ),
    "tz-aware with NaT": (
        stamps("2024-01-01", None, tz="UTC"),
        stamps("2024-01-01", None, "2024-02-02", tz="UTC"),
        2,
    ),
    "naive datetimes, other units": (
        stamps("2024-01-01", "2024-01-02", unit="us"),
        stamps("2024-01-02", "2024-03-03", unit="ns"),
        1,
    ),
    "int64 vs float64": ([1, 2, 3], [1.0, 3.0, 4.5], 2),
    "float64 vs int64": ([1.0, 2.5], [1, 2], 1),
    "nullable Int64 vs float64": (
        pd.array([1, None], dtype="Int64"),
        [1.0, np.nan, 7.0],
        2,
    ),
    "object numbers vs int64": (pd.Series([1, 2], dtype=object), [1, 3], 1),
    "object text vs int64": (pd.Series(["1", "2"], dtype=object), [1, 2], 0),
    "object text vs datetimes": (["2024-01-01"], stamps("2024-01-01"), 0),
    "timedeltas": (
        pd.to_timedelta([1, 2], unit="h"),
        pd.to_timedelta([2, 3], unit="h"),
        1,
    ),
}


@pytest.mark.parametrize(
    ("train_a", "test_a", "expected"),
    MIXED_DTYPE_CASES.values(),
    ids=MIXED_DTYPE_CASES.keys(),
)
def test_columns_with_unusual_or_different_dtypes_are_compared_by_value(
    train_a: object, test_a: object, expected: int
) -> None:
    train, test = keyed(train_a, test_a)
    assert overlap_count(train, test) == expected


def test_an_int64_value_and_an_equal_float64_value_count_as_overlap() -> None:
    train, test = keyed([1, 2, 3], [1.0, 3.0, 4.5])
    assert train["a"].dtype != test["a"].dtype
    assert check(train, test).findings[0].evidence == {
        "rows": 2,
        "test_rows": 3,
        "fraction": round(2 / 3, 6),
    }


@pytest.mark.parametrize("naive_in_train", [True, False], ids=["train", "test"])
def test_naive_and_tz_aware_datetimes_raise_an_input_error(
    naive_in_train: bool,
) -> None:
    naive, aware = stamps("2024-01-01"), stamps("2024-01-01", tz="UTC")
    train, test = keyed(*((naive, aware) if naive_in_train else (aware, naive)))
    with pytest.raises(HexInputError) as caught:
        check(train, test)
    message = str(caught.value)
    assert "'a'" in message
    assert "timezone-naive" in message
    assert str(train["a"].dtype) in message
    assert str(test["a"].dtype) in message
    assert caught.value.fix


def test_values_that_cannot_be_hashed_raise_an_input_error_naming_the_column() -> None:
    train, test = keyed([[1], [2]], [[1], [3]])
    with pytest.raises(HexInputError) as caught:
        check(train, test)
    message = str(caught.value)
    assert "'a'" in message
    assert "'b'" not in message
    assert "train dtype object, test dtype object" in message
    assert caught.value.fix


def test_unhashable_values_in_one_frame_name_both_dtypes() -> None:
    train, test = keyed([1, 2], [[1], [3]])
    with pytest.raises(HexInputError, match="train dtype object, test dtype object"):
        check(train, test)


def test_the_key_frames_are_not_modified_by_the_dtype_handling() -> None:
    train, test = keyed(pd.Categorical(["x", "y"]), [1.5, 2.5])
    before = (train.copy(deep=True), test.copy(deep=True))
    check(train, test)
    pd.testing.assert_frame_equal(train, before[0])
    pd.testing.assert_frame_equal(test, before[1])


# --- a clean synthetic dataset (Requirement 17.4) ------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_random_dataset_split_without_duplicates_has_no_error(seed: int) -> None:
    gen = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "a": gen.normal(size=2000),
            "b": gen.choice(["p", "q", "r"], size=2000),
            "y": gen.integers(0, 2, size=2000),
        }
    )
    train, test = split_frame(df, SplitConfig(seed=seed))
    result = check(train, test)
    assert not [f for f in result.findings if f.severity is Severity.ERROR]
    assert result.outcome is Outcome.PASSED


# --- properties (Requirements 17.2, 5.3) ---------------------------------------------

row = st.tuples(st.integers(0, 3), st.integers(0, 3), st.integers(0, 1))


@given(
    train=st.lists(row, min_size=1, max_size=12),
    others=st.lists(row, max_size=12),
    pick=st.data(),
)
def test_any_test_row_copied_from_train_is_detected(
    train: list[tuple], others: list[tuple], pick: st.DataObject
) -> None:
    copied = pick.draw(st.sampled_from(train))
    position = pick.draw(st.integers(0, len(others)))
    test_rows = others[:position] + [copied] + others[position:]

    result = check(frame(train), frame(test_rows), max_fraction=0.0)

    expected = sum(r in set(train) for r in test_rows)
    assert expected >= 1
    assert result.outcome is Outcome.FAILED
    (finding,) = result.findings
    assert finding.severity is Severity.ERROR
    assert finding.evidence["rows"] == expected
    assert finding.evidence["test_rows"] == len(test_rows)


@given(
    train=st.lists(row, min_size=1, max_size=12),
    test=st.lists(row, min_size=1, max_size=12),
    max_fraction=st.sampled_from([0.0, 0.1, 0.5]),
    order=st.randoms(use_true_random=False),
)
def test_reordering_rows_never_changes_the_result(
    train: list[tuple], test: list[tuple], max_fraction: float, order
) -> None:
    expected = check(frame(train), frame(test), max_fraction)
    shuffled_train, shuffled_test = train[:], test[:]
    order.shuffle(shuffled_train)
    order.shuffle(shuffled_test)
    assert check(frame(shuffled_train), frame(shuffled_test), max_fraction) == expected
    overlapping = sum(r in set(train) for r in test)
    assert (expected.outcome is Outcome.FAILED) == (
        overlapping / len(test) > max_fraction
    )


@given(
    train=st.lists(row, min_size=1, max_size=12),
    test=st.lists(row, min_size=1, max_size=12),
)
def test_the_dtype_of_a_column_never_changes_the_count(
    train: list[tuple], test: list[tuple]
) -> None:
    same = check(frame(train), frame(test))
    as_float = check(frame(train), frame(test).astype({"a": "float64", "b": "float64"}))
    as_category = check(frame(train).astype({"a": "category"}), frame(test))
    as_nullable = check(frame(train), frame(test).astype({"a": "Int64"}))
    assert as_float == same
    assert as_category == same
    assert as_nullable == same
