"""The data generators of the benchmark. Each dataset is generated once per module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import roc_auc_score

from hexbench.cases import all_cases
from hexbench.datagen import generate, tabular
from hexbench.mechanisms import MECHANISMS

BY_KEY = {m.key: m for m in MECHANISMS}
SEEDS = {
    case.mechanism.key: case.data_seed for case in all_cases() if case.model.key == "lr"
}
KEYS = [m.key for m in MECHANISMS]


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return {key: generate(BY_KEY[key].data, SEEDS[key]) for key in KEYS}


def _features(key: str) -> list[str]:
    return eval(BY_KEY[key].data.feature_expr, {})  # builds a list of names only


def test_same_seed_gives_the_same_frame(frames: dict[str, pd.DataFrame]) -> None:
    for key in (
        "noisy-proxy",
        "duplicated-rows",
        "target-encoding-all-rows",
        "random-split-in-script",
        "entity-leak",
        "imputer-all-rows",
    ):
        again = generate(BY_KEY[key].data, SEEDS[key])
        pd.testing.assert_frame_equal(frames[key], again)


@pytest.mark.parametrize("key", KEYS)
def test_at_least_2000_rows(frames: dict[str, pd.DataFrame], key: str) -> None:
    assert len(frames[key]) >= 2000


@pytest.mark.parametrize("key", KEYS)
def test_target_is_binary_with_both_classes(
    frames: dict[str, pd.DataFrame], key: str
) -> None:
    assert set(frames[key]["y"].unique()) == {0, 1}


def test_float_values_are_rounded_to_4_decimals(
    frames: dict[str, pd.DataFrame],
) -> None:
    for key, frame in frames.items():
        for column in frame.select_dtypes(include="float").columns:
            values = frame[column].dropna()
            assert (values.round(4) == values).all(), (key, column)


def test_columns_cover_the_base_features(frames: dict[str, pd.DataFrame]) -> None:
    for key, frame in frames.items():
        assert set(_features(key)) <= set(frame.columns), key


def test_stand_alone_auc_of_the_two_proxies(frames: dict[str, pd.DataFrame]) -> None:
    for key in ("noisy-proxy", "weak-proxy"):
        assert BY_KEY[key].data.params["extra_name"] == "risk_score"
    noisy = roc_auc_score(
        frames["noisy-proxy"]["y"], frames["noisy-proxy"]["risk_score"]
    )
    weak = roc_auc_score(frames["weak-proxy"]["y"], frames["weak-proxy"]["risk_score"])
    assert noisy >= 0.97
    assert 0.86 <= weak <= 0.92


def test_refund_flag_equals_the_target(frames: dict[str, pd.DataFrame]) -> None:
    frame = frames["post-outcome-copy"]
    assert (frame["refund_flag"] == frame["y"]).all()


def test_entities_has_singletons_groups_and_a_text_id(
    frames: dict[str, pd.DataFrame],
) -> None:
    frame = frames["entity-leak"]
    sizes = frame["entity_id"].value_counts()
    assert (sizes == 1).sum() == 2000
    assert (sizes == 4).sum() == 300
    assert len(sizes) == 2300
    assert pd.api.types.is_string_dtype(frame["entity_id"])
    assert "entity_id" not in _features("entity-leak")
    assert frame.groupby("entity_id")["y"].nunique().max() == 1


def test_duplicated_has_exactly_1000_repeated_rows(
    frames: dict[str, pd.DataFrame],
) -> None:
    frame = frames["duplicated-rows"]
    assert len(frame) == 3500
    assert int(frame.duplicated().sum()) == 1000


def test_drifting_time_is_increasing_and_parseable(
    frames: dict[str, pd.DataFrame],
) -> None:
    for key in ("random-split-in-script", "time-column-undeclared"):
        times = pd.to_datetime(frames[key]["event_time"], format="%Y-%m-%d %H:%M:%S")
        assert times.is_monotonic_increasing
        assert times.is_unique


def test_missing_share_of_the_imputer_mechanism(
    frames: dict[str, pd.DataFrame],
) -> None:
    frame = frames["imputer-all-rows"][_features("imputer-all-rows")]
    share = float(frame.isna().to_numpy().mean())
    assert abs(share - 0.25) <= 0.02


def test_no_missing_value_outside_the_imputer_mechanism(
    frames: dict[str, pd.DataFrame],
) -> None:
    for key, frame in frames.items():
        if key != "imputer-all-rows":
            assert not frame.isna().any().any(), key


def test_generate_rejects_an_unknown_generator() -> None:
    spec = BY_KEY["noisy-proxy"].data
    broken = type(spec)("no_such_generator", {}, spec.feature_expr)
    with pytest.raises(ValueError, match="no_such_generator"):
        generate(broken, 1)


@settings(max_examples=50, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_any_seed_gives_the_same_frame_twice(seed: int) -> None:
    first = tabular(seed, n_rows=40, n_columns=4, n_informative=2)
    second = tabular(seed, n_rows=40, n_columns=4, n_informative=2)
    pd.testing.assert_frame_equal(first, second)
    assert np.isfinite(first.to_numpy(dtype=float)).all()
