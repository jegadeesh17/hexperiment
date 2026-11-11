"""Behaviour of ``drop_seen_test_rows`` in the honest scripts (Requirements 2.4, 3.4).

The function removes the test rows whose key columns also occur in the training
rows. Each test loads the honest script of an app as a module and calls it.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pandas as pd
import pytest
from conftest import APP_NAMES, script_module
from hypothesis import given, settings
from hypothesis import strategies as st

KEY = ["a", "b", "t"]
DropSeen = Callable[[pd.DataFrame, pd.DataFrame, list[str]], pd.DataFrame]


@pytest.fixture(params=APP_NAMES)
def drop_seen(request: pytest.FixtureRequest) -> DropSeen:
    """``drop_seen_test_rows`` of the honest script of each app."""
    return script_module(request.param, haunted=False).drop_seen_test_rows


def frame(rows: list[tuple[float, str, int]], index: list[int]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=KEY, index=index)


def test_a_test_row_equal_to_a_training_row_is_dropped(drop_seen: DropSeen) -> None:
    train = frame([(1.0, "x", 0), (2.0, "y", 1)], [0, 1])
    test = frame([(2.0, "y", 1), (3.0, "z", 0)], [10, 11])
    assert drop_seen(train, test, KEY).index.tolist() == [11]


def test_only_the_key_columns_are_compared(drop_seen: DropSeen) -> None:
    train = frame([(1.0, "x", 0)], [0]).assign(other=1)
    test = frame([(1.0, "x", 0)], [10]).assign(other=2)
    assert drop_seen(train, test, KEY).empty


def test_a_row_that_differs_only_in_the_target_stays(drop_seen: DropSeen) -> None:
    train = frame([(1.0, "x", 0)], [0])
    test = frame([(1.0, "x", 1)], [10])
    assert drop_seen(train, test, KEY).index.tolist() == [10]


def test_nan_equals_nan(drop_seen: DropSeen) -> None:
    train = frame([(math.nan, "x", 0)], [0])
    test = frame([(math.nan, "x", 0), (1.0, "x", 0)], [10, 11])
    assert drop_seen(train, test, KEY).index.tolist() == [11]


def test_duplicates_within_the_test_rows_stay(drop_seen: DropSeen) -> None:
    train = frame([(1.0, "x", 0)], [0])
    test = frame([(5.0, "q", 1), (5.0, "q", 1), (1.0, "x", 0)], [10, 11, 12])
    assert drop_seen(train, test, KEY).index.tolist() == [10, 11]


def test_duplicates_within_the_training_rows_do_not_matter(drop_seen: DropSeen) -> None:
    train = frame([(1.0, "x", 0), (1.0, "x", 0)], [0, 1])
    test = frame([(1.0, "x", 0), (2.0, "x", 0)], [10, 11])
    assert drop_seen(train, test, KEY).index.tolist() == [11]


def test_the_index_labels_and_values_are_kept(drop_seen: DropSeen) -> None:
    train = frame([(1.0, "x", 0)], [7])
    test = frame([(2.0, "a", 1), (1.0, "x", 0), (3.0, "b", 0)], [40, 20, 30])
    result = drop_seen(train, test, KEY)
    assert result.index.tolist() == [40, 30]
    pd.testing.assert_frame_equal(result, test.loc[[40, 30]])


def test_the_inputs_are_not_changed(drop_seen: DropSeen) -> None:
    train = frame([(1.0, "x", 0)], [0])
    test = frame([(1.0, "x", 0), (2.0, "y", 1)], [10, 11])
    train_before, test_before = train.copy(), test.copy()
    drop_seen(train, test, KEY)
    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(test, test_before)


def test_an_empty_training_set_drops_nothing(drop_seen: DropSeen) -> None:
    train = frame([], [])
    test = frame([(1.0, "x", 0), (1.0, "x", 0)], [10, 11])
    assert drop_seen(train, test, KEY).index.tolist() == [10, 11]


ROWS = st.lists(
    st.tuples(
        st.sampled_from([math.nan, 0.0, 1.0]),
        st.sampled_from(["x", "y"]),
        st.integers(0, 1),
    ),
    max_size=12,
)


def normalised(row: tuple[float, str, int]) -> tuple[object, ...]:
    return tuple(None if isinstance(v, float) and math.isnan(v) else v for v in row)


@settings(max_examples=50, deadline=None)
@given(train_rows=ROWS, test_rows=ROWS)
def test_result_is_the_test_rows_not_seen_in_training(
    train_rows: list[tuple[float, str, int]], test_rows: list[tuple[float, str, int]]
) -> None:
    drop_seen = script_module(APP_NAMES[0], haunted=False).drop_seen_test_rows
    train = frame(train_rows, list(range(len(train_rows))))
    test = frame(test_rows, list(range(100, 100 + len(test_rows))))
    seen = {normalised(row) for row in train_rows}
    expected = [
        100 + i for i, row in enumerate(test_rows) if normalised(row) not in seen
    ]
    assert drop_seen(train, test, KEY).index.tolist() == expected
