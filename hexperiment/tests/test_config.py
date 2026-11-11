import copy
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hexperiment import HexInputError
from hexperiment.config import (
    ChecksConfig,
    ExecMode,
    HexConfig,
    RunConfig,
    SplitConfig,
    Waiver,
    current_mode,
    load_config,
    parse_config,
    run_id_from_env,
)

SOURCE = Path("apps") / "demo" / "hex.toml"
APP_DIR = Path("apps") / "demo"


def minimal() -> dict[str, Any]:
    return {
        "data": {"path": "data/demo.csv"},
        "target": {"column": "churned"},
        "metric": {"name": "roc_auc", "threshold": 0.8},
    }


def parse(raw: dict[str, Any]) -> HexConfig:
    return parse_config(raw, source=SOURCE, app_dir=APP_DIR)


def with_value(table: str, key: str, value: Any) -> dict[str, Any]:
    raw = minimal()
    raw.setdefault(table, {})[key] = value
    return raw


def with_waiver(**waiver: Any) -> dict[str, Any]:
    raw = minimal()
    raw["waivers"] = [waiver]
    return raw


def rejects(raw: dict[str, Any], *fragments: str) -> HexInputError:
    """Parse ``raw``, expect HexInputError, and check the message names each fragment
    and the file."""
    with pytest.raises(HexInputError) as caught:
        parse(raw)
    message = str(caught.value)
    assert str(SOURCE) in message
    for fragment in fragments:
        assert fragment in message
    return caught.value


# --- keys and defaults (Requirement 3.1) ----------------------------------------


def test_minimal_config_gets_every_default() -> None:
    config = parse(minimal())
    assert config.app.name == "demo"
    assert config.app.description == ""
    assert config.data.path == "data/demo.csv"
    assert config.target.column == "churned"
    assert config.target.positive == 1
    assert config.metric.name == "roc_auc"
    assert config.metric.threshold == 0.8
    assert config.split == SplitConfig(test_fraction=0.2, seed=0, time_column=None)
    assert config.checks == ChecksConfig(
        proxy_auc_max=0.95,
        overlap_max_fraction=0.0,
        too_good_margin=0.15,
        max_test_uses=5,
    )
    assert config.run == RunConfig(timeout_seconds=120)
    assert config.waivers == ()
    assert config.app_dir == APP_DIR


def test_every_key_can_be_set() -> None:
    raw = {
        "app": {"name": "churn", "description": "Predict cancellations."},
        "data": {"path": "data/churn.csv"},
        "target": {"column": "cancelled", "positive": "yes"},
        "metric": {"name": "roc_auc", "threshold": 0.7},
        "split": {"test_fraction": 0.3, "seed": 7, "time_column": "signup"},
        "checks": {
            "proxy_auc_max": 0.9,
            "overlap_max_fraction": 0.05,
            "too_good_margin": 0.1,
            "max_test_uses": 3,
        },
        "run": {"timeout_seconds": 30},
        "waivers": [
            {
                "check": "target_proxy",
                "column": "tenure",
                "reason": "Known at scoring.",
            },
            {"check": "temporal", "reason": "Dates are shifted by design."},
        ],
    }
    config = parse(raw)
    assert config.app.name == "churn"
    assert config.app.description == "Predict cancellations."
    assert config.target.positive == "yes"
    assert config.metric.threshold == 0.7
    assert config.split == SplitConfig(test_fraction=0.3, seed=7, time_column="signup")
    assert config.checks == ChecksConfig(0.9, 0.05, 0.1, 3)
    assert config.run.timeout_seconds == 30
    assert config.waivers == (
        Waiver("target_proxy", "Known at scoring.", "tenure"),
        Waiver("temporal", "Dates are shifted by design.", None),
    )


def test_app_name_defaults_to_the_app_folder_name(tmp_path: Path) -> None:
    app_dir = tmp_path / "churn_app"
    config = parse_config(minimal(), source=app_dir / "hex.toml", app_dir=app_dir)
    assert config.app.name == "churn_app"


@pytest.mark.parametrize("positive", [1, 0, "yes", True])
def test_target_positive_accepts_int_string_and_bool(positive: Any) -> None:
    assert parse(with_value("target", "positive", positive)).target.positive == positive


@pytest.mark.parametrize("positive", [1.5, ["a"], None])
def test_target_positive_rejects_other_types(positive: Any) -> None:
    rejects(with_value("target", "positive", positive), "target.positive")


def test_integer_thresholds_are_stored_as_floats() -> None:
    config = parse(with_value("checks", "proxy_auc_max", 1))
    assert config.checks.proxy_auc_max == 1.0
    assert isinstance(config.checks.proxy_auc_max, float)


# --- required keys (Requirement 3.3) --------------------------------------------


@pytest.mark.parametrize(
    ("table", "key"),
    [
        ("data", "path"),
        ("target", "column"),
        ("metric", "name"),
        ("metric", "threshold"),
    ],
)
def test_missing_required_key_names_key_and_file(table: str, key: str) -> None:
    raw = minimal()
    del raw[table][key]
    rejects(raw, f"{table}.{key}")


@pytest.mark.parametrize("table", ["data", "target", "metric"])
def test_missing_required_table_is_rejected(table: str) -> None:
    raw = minimal()
    del raw[table]
    with pytest.raises(HexInputError) as caught:
        parse(raw)
    assert f"{table}." in str(caught.value)
    assert str(SOURCE) in str(caught.value)


def test_missing_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(HexInputError, match="hex.toml") as caught:
        load_config(tmp_path)
    assert str(tmp_path / "hex.toml") in str(caught.value)


def test_unreadable_file_names_the_path(tmp_path: Path) -> None:
    (tmp_path / "hex.toml").mkdir()  # a directory cannot be read as a file
    with pytest.raises(HexInputError) as caught:
        load_config(tmp_path)
    assert str(tmp_path / "hex.toml") in str(caught.value)


def test_invalid_toml_names_the_path(tmp_path: Path) -> None:
    (tmp_path / "hex.toml").write_text("[data\npath = 1", encoding="utf-8")
    with pytest.raises(HexInputError, match="not valid TOML") as caught:
        load_config(tmp_path)
    assert str(tmp_path / "hex.toml") in str(caught.value)


def test_load_config_reads_the_file(tmp_path: Path) -> None:
    (tmp_path / "hex.toml").write_text(
        '[app]\nname = "x"\n[data]\npath = "d.csv"\n[target]\ncolumn = "y"\n'
        '[metric]\nname = "roc_auc"\nthreshold = 0.75\n'
        '[[waivers]]\ncheck = "row_overlap"\nreason = "Duplicates are real orders."\n',
        encoding="utf-8",
    )
    config = load_config(tmp_path)
    assert config.app.name == "x"
    assert config.metric.threshold == 0.75
    assert config.waivers == (Waiver("row_overlap", "Duplicates are real orders."),)
    assert config.app_dir == tmp_path


# --- value validation (Requirements 3.2, 3.4) -----------------------------------


def test_only_roc_auc_is_supported() -> None:
    rejects(with_value("metric", "name", "accuracy"), "metric.name", "roc_auc")


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0, -0.1, 1.5, True, False, "0.5", None])
@pytest.mark.parametrize(
    ("table", "key"), [("metric", "threshold"), ("split", "test_fraction")]
)
def test_open_unit_interval_numbers_reject_everything_else(
    table: str, key: str, value: Any
) -> None:
    rejects(with_value(table, key, value), f"{table}.{key}")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_rejected(value: float) -> None:
    rejects(with_value("metric", "threshold", value), "metric.threshold")


@pytest.mark.parametrize(
    ("table", "key", "value"),
    [
        ("checks", "proxy_auc_max", 0),
        ("checks", "proxy_auc_max", 1.01),
        ("checks", "overlap_max_fraction", -0.01),
        ("checks", "overlap_max_fraction", 1.01),
        ("checks", "too_good_margin", -0.01),
        ("checks", "max_test_uses", 0),
        ("checks", "max_test_uses", 2.5),
        ("checks", "max_test_uses", True),
        ("run", "timeout_seconds", 0),
        ("run", "timeout_seconds", -5),
        ("run", "timeout_seconds", "60"),
        ("split", "seed", -1),
        ("split", "seed", 1.5),
        ("split", "seed", True),
        ("split", "time_column", ""),
        ("split", "time_column", 3),
        ("data", "path", ""),
        ("target", "column", 7),
        ("app", "name", 7),
        ("app", "description", 7),
    ],
)
def test_out_of_range_values_name_key_and_file(
    table: str, key: str, value: Any
) -> None:
    rejects(with_value(table, key, value), f"{table}.{key}")


@pytest.mark.parametrize(
    ("table", "key", "value"),
    [
        ("checks", "proxy_auc_max", 1),
        ("checks", "overlap_max_fraction", 0),
        ("checks", "overlap_max_fraction", 1),
        ("checks", "too_good_margin", 0),
        ("checks", "max_test_uses", 1),
        ("run", "timeout_seconds", 0.5),
        ("split", "seed", 0),
        ("app", "description", ""),
    ],
)
def test_range_boundaries_that_are_allowed(table: str, key: str, value: Any) -> None:
    parse(with_value(table, key, value))


# --- unknown tables and keys (strict validation: unknown keys fail loudly) ---


def test_unknown_table_is_rejected() -> None:
    raw = minimal()
    raw["metrics"] = {"name": "roc_auc"}
    rejects(raw, "metrics")


def test_unknown_key_is_rejected_with_its_table() -> None:
    rejects(with_value("checks", "proxy_auc_mx", 0.9), "checks.proxy_auc_mx")


def test_non_table_value_is_rejected() -> None:
    raw = minimal()
    raw["split"] = 3
    rejects(raw, "[split]")


# --- waivers (Requirement 11.1, 11.3, 10.7) -------------------------------------


@pytest.mark.parametrize(
    "check", ["canary", "determinism", "split_order", "input", "x", ""]
)
def test_only_three_checks_can_be_waived(check: str) -> None:
    rejects(
        with_waiver(check=check, reason="A long enough reason."), "waivers[1].check"
    )


@pytest.mark.parametrize("check", ["row_overlap", "target_proxy", "temporal"])
def test_waivable_checks_are_accepted(check: str) -> None:
    config = parse(with_waiver(check=check, reason="A long enough reason."))
    assert config.waivers == (Waiver(check, "A long enough reason."),)


@pytest.mark.parametrize("reason", ["", "short", "   padded   ", "123456789", 12345])
def test_waiver_reason_needs_ten_characters_after_stripping(reason: Any) -> None:
    rejects(with_waiver(check="temporal", reason=reason), "waivers[1].reason")


def test_waiver_reason_of_exactly_ten_characters_is_kept_stripped() -> None:
    config = parse(with_waiver(check="temporal", reason="  1234567890  "))
    assert config.waivers[0].reason == "1234567890"


@pytest.mark.parametrize("check", ["row_overlap", "temporal"])
def test_waiver_column_is_only_for_target_proxy(check: str) -> None:
    rejects(
        with_waiver(check=check, column="age", reason="A long enough reason."),
        "waivers[1].column",
    )


def test_waiver_column_must_be_a_non_empty_string() -> None:
    rejects(
        with_waiver(check="target_proxy", column="", reason="A long enough reason."),
        "waivers[1].column",
    )


@pytest.mark.parametrize("missing", ["check", "reason"])
def test_waiver_missing_required_key(missing: str) -> None:
    waiver = {"check": "temporal", "reason": "A long enough reason."}
    del waiver[missing]
    rejects(with_waiver(**waiver), f"waivers[1].{missing}")


def test_waiver_unknown_key_is_rejected() -> None:
    rejects(
        with_waiver(check="temporal", reason="A long enough reason.", why="x"),
        "waivers[1].why",
    )


def test_second_waiver_is_numbered_two() -> None:
    raw = minimal()
    raw["waivers"] = [
        {"check": "temporal", "reason": "A long enough reason."},
        {"check": "canary", "reason": "A long enough reason."},
    ]
    rejects(raw, "waivers[2].check")


def test_waivers_must_be_an_array_of_tables() -> None:
    raw = minimal()
    raw["waivers"] = {"check": "temporal"}
    rejects(raw, "waivers")


# --- execution mode -------------------------------------------------------------


def test_mode_defaults_to_normal_when_unset() -> None:
    assert current_mode({}) is ExecMode.NORMAL


@pytest.mark.parametrize("mode", list(ExecMode))
def test_mode_reads_each_valid_value(mode: ExecMode) -> None:
    assert current_mode({"HEX_MODE": mode.value}) is mode


@pytest.mark.parametrize("value", ["", "Normal", "CANARY", "fast", " canary"])
def test_invalid_mode_raises_and_names_the_variable(value: str) -> None:
    with pytest.raises(HexInputError) as caught:
        current_mode({"HEX_MODE": value})
    assert "HEX_MODE" in str(caught.value)
    assert "canary" in str(caught.value)


def test_mode_reads_the_process_environment_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HEX_MODE", raising=False)
    assert current_mode() is ExecMode.NORMAL
    monkeypatch.setenv("HEX_MODE", "control")
    assert current_mode() is ExecMode.CONTROL


def test_run_id_is_none_when_unset_and_the_value_when_set() -> None:
    assert run_id_from_env({}) is None
    assert run_id_from_env({"HEX_RUN_ID": "20250101-000000-abcdef"}) == (
        "20250101-000000-abcdef"
    )


def test_empty_run_id_is_rejected() -> None:
    with pytest.raises(HexInputError, match="HEX_RUN_ID"):
        run_id_from_env({"HEX_RUN_ID": "  "})


def test_run_id_reads_the_process_environment_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEX_RUN_ID", "abc")
    assert run_id_from_env() == "abc"


# --- properties -----------------------------------------------------------------

outside_open_unit_interval = st.one_of(
    st.floats(allow_nan=True, allow_infinity=True).filter(lambda x: not 0 < x < 1),
    st.integers().filter(lambda x: not 0 < x < 1),
)


@given(outside_open_unit_interval)
def test_any_threshold_outside_the_open_interval_is_rejected(value: float) -> None:
    raw = copy.deepcopy(minimal())
    raw["metric"]["threshold"] = value
    with pytest.raises(HexInputError, match=r"metric\.threshold"):
        parse(raw)


@given(outside_open_unit_interval)
def test_any_test_fraction_outside_the_open_interval_is_rejected(value: float) -> None:
    with pytest.raises(HexInputError, match=r"split\.test_fraction"):
        parse(with_value("split", "test_fraction", value))


@given(st.floats(min_value=0, max_value=1, exclude_min=True, exclude_max=True))
def test_any_value_inside_the_open_interval_is_accepted(value: float) -> None:
    raw = with_value("split", "test_fraction", value)
    raw["metric"]["threshold"] = value
    config = parse(raw)
    assert config.split.test_fraction == value
    assert config.metric.threshold == value
