"""Prepared synthetic data works with ``hexperiment.load`` and ``hexperiment.split``.

This shows that the files ``fetch_data.py`` writes fit the core without real data
(Requirements 1.6, 1.7 and 5.1). The real ``hex.toml`` is copied next to a synthetic
``data/prepared.csv`` in a temporary app folder.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest
from conftest import APPS_DIR, FLIGHTS, HOTELS
from hexperiment.config import ENV_MODE, ENV_RUN_ID

import hexperiment


def make_app_folder(
    app_name: str, prepared: pd.DataFrame, fetch: ModuleType, folder: Path
) -> Path:
    """Copy the app's ``hex.toml`` and write ``prepared`` where ``data.path`` points."""
    shutil.copy(APPS_DIR / app_name / "hex.toml", folder / "hex.toml")
    (folder / "data").mkdir()
    prepared.to_csv(
        folder / "data" / fetch.PREPARED_NAME, index=False, lineterminator="\n"
    )
    return folder


@pytest.fixture(autouse=True)
def normal_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_MODE, raising=False)
    monkeypatch.delenv(ENV_RUN_ID, raising=False)


def test_flights_prepared_file_loads_and_splits_by_time(
    flights_fetch: ModuleType, raw_flights: Any, tmp_path: Path
) -> None:
    delays = [(-10 if i % 4 else 40) for i in range(300)]
    delays[7] = math.nan
    prepared = flights_fetch.prepare(raw_flights(delays))
    folder = make_app_folder(FLIGHTS, prepared, flights_fetch, tmp_path)

    df = hexperiment.load(folder)
    train, test = hexperiment.split(df)

    assert len(train) > 0
    assert len(test) > 0
    assert len(train) + len(test) == 299
    train_times = pd.to_datetime(train["time_hour"], utc=True)
    test_times = pd.to_datetime(test["time_hour"], utc=True)
    assert train_times.max() <= test_times.min()
    for name in ("dep_time", "dep_delay", "arr_time", "arr_delay", "air_time"):
        assert name not in df.columns
    assert set(df["late"]) == {0, 1}


def test_hotels_prepared_file_loads_and_splits_at_random(
    hotels_fetch: ModuleType, raw_hotels: Any, tmp_path: Path
) -> None:
    prepared = hotels_fetch.prepare(raw_hotels(300))
    folder = make_app_folder(HOTELS, prepared, hotels_fetch, tmp_path)

    df = hexperiment.load(folder)
    train, test = hexperiment.split(df)

    assert len(train) > 0
    assert len(test) > 0
    assert len(train) + len(test) == 300
    assert abs(len(test) / 300 - 0.2) < 0.1
    assert "reservation_status" in df.columns
    assert "reservation_status_date" in df.columns
    assert set(train.index).isdisjoint(test.index)
