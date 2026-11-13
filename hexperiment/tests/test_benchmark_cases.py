"""The mechanisms of the benchmark and the cases built from them (no case is run)."""

from __future__ import annotations

import re

from hexbench.mechanisms import LEAK_TYPE_NAMES, MECHANISMS, NOT_MEASURED
from hexperiment.ghosts import CATALOG

_KEY = re.compile(r"^[a-z]+(-[a-z]+)*$")


def test_keys_are_unique_and_well_formed() -> None:
    keys = [m.key for m in MECHANISMS]
    assert len(keys) == len(set(keys))
    assert all(_KEY.match(key) for key in keys)


def test_leak_types_are_known() -> None:
    assert all(m.leak_type in LEAK_TYPE_NAMES for m in MECHANISMS)


def test_leak_type_names_are_the_papers_names() -> None:
    assert dict(LEAK_TYPE_NAMES) == {
        "L1.1": "No test set",
        "L1.2": "Pre-processing on training and test set",
        "L1.3": "Feature selection on training and test set",
        "L1.4": "Duplicates in datasets",
        "L2": "Model uses features that are not legitimate",
        "L3.1": "Temporal leakage",
        "L3.2": "Nonindependence between training and test samples",
        "L3.3": "Sampling bias in test distribution",
    }


def test_designed_to_catch_names_are_ghosts_of_the_kit() -> None:
    ghosts = {entry.ghost for entry in CATALOG if entry.ghost is not None}
    for mechanism in MECHANISMS:
        assert set(mechanism.designed_to_catch) <= ghosts, mechanism.key


def test_thresholds_and_default_checks() -> None:
    for mechanism in MECHANISMS:
        expected = {
            "univariate-selection-all-rows": 0.52,
            "random-split-in-script": 0.55,
            "time-column-undeclared": 0.55,
        }.get(mechanism.key, 0.60)
        assert mechanism.threshold == expected, mechanism.key
        for lines in mechanism.toml_slots.values():
            for line in lines:
                assert "[checks" not in line.text, mechanism.key
                assert "waiver" not in line.text.lower(), mechanism.key


def test_there_are_13_mechanisms_in_the_order_of_the_design() -> None:
    assert [m.key for m in MECHANISMS] == [
        "no-test-set",
        "scaler-all-rows",
        "imputer-all-rows",
        "oversample-before-split",
        "target-encoding-all-rows",
        "univariate-selection-all-rows",
        "duplicated-rows",
        "post-outcome-copy",
        "noisy-proxy",
        "weak-proxy",
        "random-split-in-script",
        "time-column-undeclared",
        "entity-leak",
    ]


def test_the_leakage_types_of_requirement_1_1_are_covered() -> None:
    types = [m.leak_type for m in MECHANISMS]
    assert types.count("L1.1") == 1
    assert types.count("L1.2") == 4
    assert types.count("L1.3") == 1
    assert types.count("L1.4") == 1
    assert types.count("L2") == 3
    assert types.count("L3.1") == 2
    assert types.count("L3.2") == 1


def test_l3_3_is_not_measured_and_in_no_mechanism() -> None:
    assert [n.leak_type for n in NOT_MEASURED] == ["L3.3"]
    entry = NOT_MEASURED[0]
    assert entry.leak_type_name == LEAK_TYPE_NAMES["L3.3"]
    assert entry.reason == (
        "It depends on the population a claim is about, "
        "which neither the code nor the data shows."
    )
    assert all(m.leak_type != "L3.3" for m in MECHANISMS)


def test_designed_to_catch_and_data_columns_of_the_late_mechanisms() -> None:
    by_key = {m.key: m for m in MECHANISMS}
    empty = sorted(m.key for m in MECHANISMS if not m.designed_to_catch)
    assert empty == ["entity-leak", "weak-proxy"]
    assert "The Prophet" in by_key["time-column-undeclared"].designed_to_catch
    for key in ("random-split-in-script", "time-column-undeclared"):
        assert by_key[key].data.time_column == "event_time"
    assert by_key["entity-leak"].data.entity_column == "entity_id"
    others = [m for m in MECHANISMS if m.key != "entity-leak"]
    assert all(m.data.entity_column is None for m in others)
    assert by_key["time-column-undeclared"].slots == {}
    assert by_key["time-column-undeclared"].toml_slots["split_extra"][0].side == "leaky"
