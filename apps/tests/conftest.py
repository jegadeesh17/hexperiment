"""Fixtures of the apps tests: the network guard, the two apps, synthetic raw frames.

The helpers these fixtures use live in ``apps_helpers.py``.
"""

from __future__ import annotations

import socket
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
from apps_helpers import (
    APP_NAMES,
    APPS_DIR,
    FLIGHTS,
    HOTELS,
    TESTS_DIR,
    App,
    RawFlights,
    RawHotels,
    fetch_module,
    make_raw_flights,
    make_raw_hotels,
    verify_module_loaded,
)


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any apps test that tries to use the network (Requirements 5.1, 5.6).

    Connecting, sending a datagram and looking up a host name are refused. Only
    tests under this folder are guarded. Pytest's root folder can lie outside
    ``apps/`` (``hexperiment/`` when both test folders are run), and then this
    autouse fixture would also reach tests that are not ours.
    """
    if not Path(request.path).resolve().is_relative_to(TESTS_DIR):
        return

    def refuse(*args: object, **kwargs: object) -> None:
        raise RuntimeError("a test tried to use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket.socket, "sendto", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture(scope="session")
def hotels_fetch() -> ModuleType:
    return fetch_module(HOTELS)


@pytest.fixture(scope="session")
def flights_fetch() -> ModuleType:
    return fetch_module(FLIGHTS)


@pytest.fixture(scope="session")
def verify_module() -> ModuleType:
    return verify_module_loaded()


@pytest.fixture(params=APP_NAMES)
def app(request: pytest.FixtureRequest) -> App:
    """Each test that uses it runs once for each example app."""
    return App(name=request.param, folder=APPS_DIR / request.param)


@pytest.fixture
def raw_hotels(hotels_fetch: ModuleType) -> RawHotels:
    """Factory ``raw_hotels(n=300)`` for synthetic hotel rows."""

    def build(n: int = 300) -> pd.DataFrame:
        return make_raw_hotels(hotels_fetch.REQUIRED_COLUMNS, n)

    return build


@pytest.fixture
def raw_flights(flights_fetch: ModuleType) -> RawFlights:
    """Factory ``raw_flights(arr_delays, time_hours=None, row_number_column=...)``."""

    def build(
        arr_delays: Sequence[float],
        *,
        time_hours: Sequence[str] | None = None,
        row_number_column: str | None = "rownames",
    ) -> pd.DataFrame:
        return make_raw_flights(
            flights_fetch.REQUIRED_COLUMNS,
            arr_delays,
            time_hours=time_hours,
            row_number_column=row_number_column,
        )

    return build
