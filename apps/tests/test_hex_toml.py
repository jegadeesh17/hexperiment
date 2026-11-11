"""Each app's hex.toml loads with the core loader and keeps every default check.

Requirements 2.1, 3.1 and 5.1. Steering rules 6 and 7: no weakened check, no waiver.
"""

from __future__ import annotations

import tomllib

import pytest
from conftest import FLIGHTS, HOTELS, App
from hexperiment.config import ChecksConfig, HexConfig, load_config

EXPECTED = {
    HOTELS: {"target": "is_canceled", "threshold": 0.85, "time_column": None},
    FLIGHTS: {"target": "late", "threshold": 0.60, "time_column": "time_hour"},
}


@pytest.fixture
def config(app: App) -> HexConfig:
    return load_config(app.folder)


def test_name_matches_the_folder(app: App, config: HexConfig) -> None:
    assert config.app.name == app.name


def test_target(app: App, config: HexConfig) -> None:
    assert config.target.column == EXPECTED[app.name]["target"]
    assert config.target.positive == 1


def test_metric_and_threshold(app: App, config: HexConfig) -> None:
    assert config.metric.name == "roc_auc"
    assert config.metric.threshold == EXPECTED[app.name]["threshold"]


def test_split(app: App, config: HexConfig) -> None:
    assert config.split.test_fraction == 0.2
    assert isinstance(config.split.seed, int)
    assert config.split.time_column == EXPECTED[app.name]["time_column"]


def test_timeout_fits_a_full_run(config: HexConfig) -> None:
    assert 60 <= config.run.timeout_seconds <= 300


def test_data_path_is_the_prepared_file(app: App, config: HexConfig) -> None:
    assert config.data.path == "data/" + app.fetch.PREPARED_NAME


def test_no_check_is_weakened_and_no_waiver(config: HexConfig) -> None:
    assert config.checks == ChecksConfig()
    assert config.waivers == ()


def test_the_file_has_no_checks_table_or_waivers(app: App) -> None:
    with (app.folder / "hex.toml").open("rb") as handle:
        raw = tomllib.load(handle)
    assert "checks" not in raw
    assert "waivers" not in raw
