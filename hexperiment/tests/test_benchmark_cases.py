"""The mechanisms of the benchmark and the cases built from them (no case is run)."""

from __future__ import annotations

import ast
import difflib
import re
import tomllib
from collections import Counter
from pathlib import Path

import pytest

from hexbench.cases import all_cases, render_app
from hexbench.mechanisms import LEAK_TYPE_NAMES, MECHANISMS, NOT_MEASURED
from hexbench.spec import MARK
from hexperiment.config import parse_config
from hexperiment.ghosts import CATALOG

_KEY = re.compile(r"^[a-z]+(-[a-z]+)*$")
_FIT = re.compile(r"\bfit(_transform|_resample)?\(")
CASES = all_cases()
VARIANTS = ("leaky", "clean")
_PAIRS = [(case, variant) for case in CASES for variant in VARIANTS]
_IDS = [f"{case.case_id}-{variant}" for case, variant in _PAIRS]


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


# --- the 39 cases -----------------------------------------------------------------


def test_there_are_39_cases_with_unique_ids() -> None:
    assert len(CASES) == 39
    assert len({case.case_id for case in CASES}) == 39
    assert [case.case_id for case in CASES[:3]] == [
        "no-test-set-lr",
        "no-test-set-rf",
        "no-test-set-hgb",
    ]


def test_data_seeds_follow_the_formula() -> None:
    for m, mechanism in enumerate(MECHANISMS, start=1):
        cases = [c for c in CASES if c.mechanism is mechanism]
        assert [c.model.key for c in cases] == ["lr", "rf", "hgb"]
        assert [c.data_seed for c in cases] == [1000 + 10 * m + k for k in range(3)]
        assert all(c.case_id == f"{mechanism.key}-{c.model.key}" for c in cases)


def test_every_model_has_a_fixed_seed() -> None:
    assert all("random_state=0" in c.model.expression for c in CASES)


def _toml(case, variant) -> dict:
    return tomllib.loads(render_app(case, variant).hex_toml)


def _leak_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if MARK.strip() in line]


def _diff(a: str, b: str) -> tuple[list[str], list[str]]:
    lines = difflib.unified_diff(a.splitlines(), b.splitlines(), n=0, lineterm="")
    removed = [x[1:] for x in lines if x.startswith("-") and not x.startswith("---")]
    lines = difflib.unified_diff(a.splitlines(), b.splitlines(), n=0, lineterm="")
    added = [x[1:] for x in lines if x.startswith("+") and not x.startswith("+++")]
    return removed, added


def _main_calls(source: str) -> tuple[bool, bool]:
    main = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = [
        ast.unparse(node.func) for node in ast.walk(main) if isinstance(node, ast.Call)
    ]
    last = main.body[-1]
    ends_with_audit = (
        isinstance(last, ast.Expr)
        and isinstance(last.value, ast.Call)
        and ast.unparse(last.value.func) == "hexperiment.audit"
    )
    return "hexperiment.load" in calls, ends_with_audit


@pytest.mark.parametrize(("case", "variant"), _PAIRS, ids=_IDS)
def test_the_app_compiles_and_its_config_is_valid(case, variant) -> None:
    files = render_app(case, variant)
    compile(files.train_py, f"{case.case_id}/{variant}/train.py", "exec")
    raw = tomllib.loads(files.hex_toml)
    assert "checks" not in raw
    assert "waivers" not in raw
    config = parse_config(raw, source=Path("hex.toml"), app_dir=Path(case.case_id))
    assert config.app.name == case.case_id
    assert config.metric.threshold == case.mechanism.threshold
    assert config.waivers == ()


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.case_id)
def test_the_two_apps_differ_only_in_the_leak_lines(case) -> None:
    leaky = render_app(case, "leaky")
    clean = render_app(case, "clean")
    for a, b in ((leaky.train_py, clean.train_py), (leaky.hex_toml, clean.hex_toml)):
        removed, added = _diff(a, b)
        assert all(MARK.strip() in line for line in removed)
        assert not any(MARK.strip() in line for line in added)
        assert Counter(removed) == Counter(a.splitlines()) - Counter(b.splitlines())
        assert Counter(added) == Counter(b.splitlines()) - Counter(a.splitlines())
    assert _leak_lines(leaky.train_py) + _leak_lines(leaky.hex_toml)
    assert not _leak_lines(clean.train_py) and not _leak_lines(clean.hex_toml)


@pytest.mark.parametrize(("case", "variant"), _PAIRS, ids=_IDS)
def test_the_script_loads_and_ends_with_an_audit(case, variant) -> None:
    loads, ends_with_audit = _main_calls(render_app(case, variant).train_py)
    assert loads
    assert ends_with_audit


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.case_id)
def test_the_clean_twin_splits_before_any_fit(case) -> None:
    source = render_app(case, "clean").train_py
    split_at = source.index("hexperiment.split(")
    assert not _FIT.search(source[:split_at])


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.case_id)
def test_time_column_is_declared_where_it_should_be(case) -> None:
    declared = {
        variant: _toml(case, variant)["split"].get("time_column")
        for variant in VARIANTS
    }
    if case.mechanism.leak_type != "L3.1":
        assert declared == {"leaky": None, "clean": None}
    elif case.mechanism.key == "time-column-undeclared":
        assert declared == {"leaky": None, "clean": "event_time"}
    else:
        assert declared == {"leaky": "event_time", "clean": "event_time"}


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.case_id)
def test_the_entity_column_is_not_a_feature(case) -> None:
    entity = case.mechanism.data.entity_column
    if entity is not None:
        features = eval(case.mechanism.data.feature_expr, {})  # a list of names
        assert entity not in features
        assert entity in render_app(case, "clean").train_py


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.case_id)
def test_the_two_configs_differ_only_in_split_extra(case) -> None:
    leaky = render_app(case, "leaky").hex_toml
    clean = render_app(case, "clean").hex_toml
    removed, added = _diff(leaky, clean)
    extra = case.mechanism.toml_slots.get("split_extra", ())
    allowed_leaky = {line.text for line in extra} | {
        f"{line.text}{MARK}{line.note}"
        if not line.text.strip()
        else f"{line.text}  {MARK}{line.note}"
        for line in extra
    }
    allowed_clean = {line.text for line in extra}
    assert set(removed) <= allowed_leaky
    assert set(added) <= allowed_clean
    assert f'name = "{case.case_id}"' in leaky
    assert f'name = "{case.case_id}"' in clean
