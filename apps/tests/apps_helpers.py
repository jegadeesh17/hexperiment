"""Shared test helpers: load scripts by path, the two apps, synthetic raw frames.

The test files import these with ``from apps_helpers import ...``. They live here
and not in ``conftest.py`` because pytest can load more than one ``conftest`` module
when both test folders are run, and ``from conftest import ...`` then depends on the
order of the folders. Fixtures stay in ``conftest.py``.

Nothing here uses the network or needs the downloaded data. The fetch modules and
``verify.py`` are loaded on first use, so a test that does not need them runs even
before those files exist.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType

import pandas as pd

TESTS_DIR = Path(__file__).resolve().parent
APPS_DIR = TESTS_DIR.parent
HOTELS = "hotel-cancellations"
FLIGHTS = "flight-delays"
APP_NAMES = (HOTELS, FLIGHTS)

RawHotels = Callable[..., pd.DataFrame]
RawFlights = Callable[..., pd.DataFrame]


def load_module(path: Path | str, name: str) -> ModuleType:
    """Import the Python file at ``path`` as module ``name`` (folders have hyphens)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@cache
def fetch_module(app_name: str) -> ModuleType:
    """Return the ``fetch_data`` module of an app (loaded once per process)."""
    path = APPS_DIR / app_name / "fetch_data.py"
    return load_module(path, "fetch_data_" + app_name.replace("-", "_"))


@cache
def verify_module_loaded() -> ModuleType:
    """Return ``apps/verify.py`` (loaded once per process)."""
    return load_module(APPS_DIR / "verify.py", "apps_verify")


@cache
def script_module(app_name: str, *, haunted: bool) -> ModuleType:
    """Return the honest or haunted ``train.py`` of an app as a module.

    Importing runs only the constants and definitions: ``main()`` is guarded by
    ``__name__``, so no data is read and nothing is trained.
    """
    folder = APPS_DIR / app_name
    path = folder / "haunted" / "train.py" if haunted else folder / "train.py"
    kind = "haunted" if haunted else "honest"
    return load_module(path, f"script_{app_name.replace('-', '_')}_{kind}")


@dataclass(frozen=True, slots=True)
class App:
    """One example app. ``fetch`` loads ``fetch_data.py`` when first used."""

    name: str
    folder: Path

    @property
    def fetch(self) -> ModuleType:
        return fetch_module(self.name)

    @property
    def is_flights(self) -> bool:
        return self.name == FLIGHTS


# --- synthetic raw frames -----------------------------------------------------

_HOTEL_TEXT = {
    "hotel": ("Resort Hotel", "City Hotel"),
    "arrival_date_month": ("July", "August", "May"),
    "meal": ("BB", "HB", "SC"),
    "country": ("PRT", "GBR", "ESP", "FRA"),
    "market_segment": ("Online TA", "Direct", "Corporate"),
    "distribution_channel": ("TA/TO", "Direct", "Corporate"),
    "reserved_room_type": ("A", "D", "E"),
    "assigned_room_type": ("A", "D", "K"),
    "deposit_type": ("No Deposit", "Non Refund"),
    "customer_type": ("Transient", "Contract", "Group"),
    "reservation_status": ("Check-Out", "Canceled", "No-Show"),
}


def _hotel_column(name: str, n: int) -> list[object]:
    if name in _HOTEL_TEXT:
        choices = _HOTEL_TEXT[name]
        return [choices[(i * 7 + len(name)) % len(choices)] for i in range(n)]
    if name == "is_canceled":
        return [int(i % 3 == 0) for i in range(n)]
    if name == "lead_time":
        return [(i * 13) % 300 for i in range(n)]
    if name == "adr":
        return [50.0 + (i % 40) * 2.5 for i in range(n)]
    if name in ("children", "agent", "company"):
        return [float(i % 5) for i in range(n)]
    if name == "reservation_status_date":
        return [f"2015-07-{1 + i % 28:02d}" for i in range(n)]
    if name == "arrival_date_year":
        return [2015 + i % 3 for i in range(n)]
    return [i % 4 for i in range(n)]


def make_raw_hotels(required: Sequence[str], n: int = 300) -> pd.DataFrame:
    """Return ``n`` synthetic hotel rows with every column in ``required``."""
    return pd.DataFrame({name: _hotel_column(name, n) for name in required})


def make_raw_flights(
    required: Sequence[str],
    arr_delays: Sequence[float],
    *,
    time_hours: Sequence[str] | None = None,
    row_number_column: str | None = "rownames",
) -> pd.DataFrame:
    """Return one synthetic flight per entry of ``arr_delays`` (NaN means cancelled).

    The frame has every column in ``required``, and a leading row-number column
    named ``row_number_column`` (the Rdatasets export has one) unless that is None.
    ``time_hours`` gives the ``time_hour`` text; by default it is hourly from
    2013-01-01T10:00:00Z with the ``Z`` offset of the real file.
    """
    n = len(arr_delays)
    if time_hours is None:
        stamps = pd.date_range("2013-01-01 10:00", periods=n, freq="h")
        time_hours = [stamp.strftime("%Y-%m-%dT%H:%M:%SZ") for stamp in stamps]
    if len(time_hours) != n:
        raise ValueError("time_hours must have one entry per arr_delays entry")
    parsed = pd.to_datetime(pd.Series(list(time_hours)), utc=True)
    delays = [float(value) for value in arr_delays]
    columns: dict[str, list[object]] = {
        "year": [2013] * n,
        "month": parsed.dt.month.tolist(),
        "day": parsed.dt.day.tolist(),
        "dep_time": [float(500 + i) for i in range(n)],
        "sched_dep_time": [500 + (i % 1000) for i in range(n)],
        "dep_delay": [float(i % 30) for i in range(n)],
        "arr_time": [float(800 + i) for i in range(n)],
        "sched_arr_time": [800 + (i % 1000) for i in range(n)],
        "arr_delay": delays,
        "carrier": [("UA", "AA", "B6", "DL")[i % 4] for i in range(n)],
        "flight": [100 + i for i in range(n)],
        "tailnum": [f"N{i % 50:03d}" for i in range(n)],
        "origin": [("EWR", "JFK", "LGA")[i % 3] for i in range(n)],
        "dest": [("IAH", "MIA", "ORD", "ATL")[i % 4] for i in range(n)],
        "air_time": [float(100 + i % 200) for i in range(n)],
        "distance": [500 + (i % 7) * 100 for i in range(n)],
        "hour": parsed.dt.hour.tolist(),
        "minute": [0] * n,
        "time_hour": list(time_hours),
    }
    data: dict[str, list[object]] = {}
    if row_number_column is not None:
        data[row_number_column] = list(range(1, n + 1))
    for name in required:
        data[name] = columns.get(name, [0] * n)
    return pd.DataFrame(data)
