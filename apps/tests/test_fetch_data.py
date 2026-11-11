"""Offline tests of both fetch_data.py files (Requirements 1.1 to 1.7 and 5.1).

Fake openers return ``io.BytesIO``; nothing here touches the network or ``data/``.
"""

from __future__ import annotations

import hashlib
import io
import math
import urllib.error
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest
from conftest import FLIGHTS, HOTELS, App, make_raw_flights
from hypothesis import given, settings
from hypothesis import strategies as st

from hexperiment import HexInputError

# Requirement 1.2.
PINNED = {
    HOTELS: {
        "url": "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv",
        "sha256": "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06",
        "size": 16_855_599,
        "filename": "hotels.csv",
    },
    FLIGHTS: {
        "url": "https://vincentarelbundock.github.io/Rdatasets/csv/nycflights13/flights.csv",
        "sha256": "b791a2590c9167aa0bd3cb45af3f64101180878cf088c183eae4011e3a654f42",
        "size": 33_206_996,
        "filename": "flights.csv",
    },
}
FAKE_URL = "https://example.invalid/data/source.csv"


# --- fakes --------------------------------------------------------------------


class FakeOpener:
    """An ``urlopen`` stand-in that serves fixed bytes and counts its calls."""

    def __init__(self, content: bytes) -> None:
        self.content = content
        self.calls = 0

    def __call__(self, request: Any, timeout: float | None = None) -> io.BytesIO:
        self.calls += 1
        return io.BytesIO(self.content)


def failing_opener(request: Any, timeout: float | None = None) -> io.BytesIO:
    raise urllib.error.URLError("name resolution failed")


class BrokenHandle(io.BytesIO):
    """Delivers one chunk, then fails like a dropped connection."""

    def __init__(self) -> None:
        super().__init__(b"first chunk")
        self.reads = 0

    def read(self, size: int | None = -1) -> bytes:
        self.reads += 1
        if self.reads > 1:
            raise ConnectionResetError("connection dropped")
        return super().read(size)


def broken_opener(request: Any, timeout: float | None = None) -> io.BytesIO:
    return BrokenHandle()


def names_in(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


# --- fixtures -----------------------------------------------------------------


@pytest.fixture
def raw_csv(app: App, request: pytest.FixtureRequest) -> bytes:
    """A small valid source file for the app, as CSV bytes."""
    if app.is_flights:
        frame = request.getfixturevalue("raw_flights")([5, 20, math.nan, 15, 16])
    else:
        frame = request.getfixturevalue("raw_hotels")(30)
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


@pytest.fixture
def tiny(app: App, raw_csv: bytes, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Replace ``SOURCE`` by a pinned description of the small synthetic file."""
    source = app.fetch.Source(
        url=FAKE_URL,
        sha256=hashlib.sha256(raw_csv).hexdigest(),
        size=len(raw_csv),
        filename=app.fetch.SOURCE.filename,
    )
    monkeypatch.setattr(app.fetch, "SOURCE", source)
    return source


# --- pinned constants and the digest check ------------------------------------


def test_pinned_source_matches_the_requirement(app: App) -> None:
    expected = PINNED[app.name]
    source = app.fetch.SOURCE
    assert source.url == expected["url"]
    assert source.sha256 == expected["sha256"]
    assert source.size == expected["size"]
    assert source.filename == expected["filename"]
    assert app.fetch.PREPARED_NAME == "prepared.csv"


def test_check_digest_accepts_equal_values(app: App) -> None:
    app.fetch.check_digest("abc123", "abc123", url=FAKE_URL)


def test_check_digest_rejects_different_values_and_names_the_url(app: App) -> None:
    with pytest.raises(HexInputError) as caught:
        app.fetch.check_digest("abc123", "def456", url=FAKE_URL)
    assert FAKE_URL in str(caught.value)
    assert "abc123" in str(caught.value)


# --- download -----------------------------------------------------------------


def test_download_writes_verified_data(
    app: App, tiny: Any, raw_csv: bytes, tmp_path: Path
) -> None:
    dest = tmp_path / tiny.filename
    app.fetch.download(tiny, dest, opener=FakeOpener(raw_csv))
    assert dest.read_bytes() == raw_csv
    assert names_in(tmp_path) == [tiny.filename]


def test_download_streams_in_several_chunks(
    app: App, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = bytes(range(256)) * 100
    monkeypatch.setattr(app.fetch, "CHUNK_BYTES", 1000)
    source = app.fetch.Source(
        url=FAKE_URL,
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
        filename="x.csv",
    )
    app.fetch.download(source, tmp_path / "x.csv", opener=FakeOpener(content))
    assert (tmp_path / "x.csv").read_bytes() == content


def bad_downloads(good: bytes) -> dict[str, Any]:
    flipped = bytes([good[0] ^ 1]) + good[1:]
    return {
        "wrong checksum": FakeOpener(flipped),
        "too short": FakeOpener(good[:-1]),
        "larger than pinned": FakeOpener(good + b"extra"),
        "network error": failing_opener,
        "connection dropped": broken_opener,
    }


@pytest.mark.parametrize(
    "case",
    [
        "wrong checksum",
        "too short",
        "larger than pinned",
        "network error",
        "connection dropped",
    ],
)
def test_failed_download_names_the_url_and_leaves_nothing(
    app: App, tiny: Any, raw_csv: bytes, tmp_path: Path, case: str
) -> None:
    opener = bad_downloads(raw_csv)[case]
    with pytest.raises(HexInputError) as caught:
        app.fetch.download(tiny, tmp_path / tiny.filename, opener=opener)
    assert FAKE_URL in str(caught.value)
    assert caught.value.fix
    assert names_in(tmp_path) == []


@pytest.mark.parametrize("case", ["wrong checksum", "too short", "network error"])
def test_failed_download_keeps_an_existing_file(
    app: App, tiny: Any, raw_csv: bytes, tmp_path: Path, case: str
) -> None:
    dest = tmp_path / tiny.filename
    dest.write_bytes(b"older content")
    with pytest.raises(HexInputError):
        app.fetch.download(tiny, dest, opener=bad_downloads(raw_csv)[case])
    assert dest.read_bytes() == b"older content"
    assert names_in(tmp_path) == [tiny.filename]


def test_download_stops_reading_when_the_pinned_size_is_exceeded(
    app: App, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: list[int] = []

    class Endless(io.RawIOBase):
        def read(self, size: int = -1) -> bytes:
            reads.append(size)
            return b"x" * 10

    monkeypatch.setattr(app.fetch, "CHUNK_BYTES", 10)
    source = app.fetch.Source(url=FAKE_URL, sha256="0" * 64, size=25, filename="x.csv")
    with pytest.raises(HexInputError, match="larger"):
        app.fetch.download(source, tmp_path / "x.csv", opener=lambda *a, **k: Endless())
    assert len(reads) == 3
    assert names_in(tmp_path) == []


# --- fetch --------------------------------------------------------------------


def test_fetch_with_a_matching_source_does_not_download(
    app: App, tiny: Any, raw_csv: bytes, tmp_path: Path
) -> None:
    (tmp_path / tiny.filename).write_bytes(raw_csv)
    opener = FakeOpener(b"never served")
    prepared = app.fetch.fetch(tmp_path, opener=opener)
    assert opener.calls == 0
    assert prepared == tmp_path / app.fetch.PREPARED_NAME
    assert prepared.is_file()
    assert (tmp_path / tiny.filename).read_bytes() == raw_csv
    assert names_in(tmp_path) == sorted([tiny.filename, app.fetch.PREPARED_NAME])


def test_fetch_replaces_a_source_with_the_wrong_checksum(
    app: App, tiny: Any, raw_csv: bytes, tmp_path: Path
) -> None:
    (tmp_path / tiny.filename).write_bytes(b"corrupted")
    opener = FakeOpener(raw_csv)
    app.fetch.fetch(tmp_path, opener=opener)
    assert opener.calls == 1
    assert (tmp_path / tiny.filename).read_bytes() == raw_csv


def test_fetch_downloads_a_missing_source(
    app: App, tiny: Any, raw_csv: bytes, tmp_path: Path
) -> None:
    data_dir = tmp_path / "data"
    opener = FakeOpener(raw_csv)
    prepared = app.fetch.fetch(data_dir, opener=opener)
    assert opener.calls == 1
    assert prepared.is_file()
    assert (data_dir / tiny.filename).read_bytes() == raw_csv


def test_fetch_rewrites_the_prepared_file_every_time(
    app: App, tiny: Any, raw_csv: bytes, tmp_path: Path
) -> None:
    (tmp_path / tiny.filename).write_bytes(raw_csv)
    prepared = app.fetch.fetch(tmp_path, opener=FakeOpener(b""))
    first = prepared.read_bytes()
    prepared.write_bytes(b"stale")
    app.fetch.fetch(tmp_path, opener=FakeOpener(b""))
    assert prepared.read_bytes() == first


def test_fetch_failure_writes_no_prepared_file(
    app: App, tiny: Any, tmp_path: Path
) -> None:
    with pytest.raises(HexInputError):
        app.fetch.fetch(tmp_path, opener=failing_opener)
    assert names_in(tmp_path) == []


def test_fetch_reports_an_unreadable_source(
    app: App, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b""
    source = app.fetch.Source(
        url=FAKE_URL,
        sha256=hashlib.sha256(content).hexdigest(),
        size=0,
        filename=app.fetch.SOURCE.filename,
    )
    monkeypatch.setattr(app.fetch, "SOURCE", source)
    (tmp_path / source.filename).write_bytes(content)
    with pytest.raises(HexInputError, match="cannot be read"):
        app.fetch.fetch(tmp_path, opener=FakeOpener(b""))
    assert app.fetch.PREPARED_NAME not in names_in(tmp_path)


# --- write_atomic -------------------------------------------------------------


def test_write_atomic_writes_a_complete_file(app: App, tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    app.fetch.write_atomic(frame, tmp_path / "prepared.csv")
    assert (tmp_path / "prepared.csv").read_text(encoding="utf-8") == "a,b\n1,x\n2,y\n"
    assert names_in(tmp_path) == ["prepared.csv"]


def test_write_atomic_failure_leaves_no_file(
    app: App, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def half_write(self: pd.DataFrame, path: Any, **kwargs: Any) -> None:
        Path(path).write_text("a,b\n1,", encoding="utf-8")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", half_write)
    with pytest.raises(OSError, match="disk full"):
        app.fetch.write_atomic(pd.DataFrame({"a": [1]}), tmp_path / "prepared.csv")
    assert names_in(tmp_path) == []


def test_write_atomic_failure_keeps_an_existing_file(
    app: App, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "prepared.csv").write_text("old\n", encoding="utf-8")

    def fail(self: pd.DataFrame, path: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    with pytest.raises(OSError):
        app.fetch.write_atomic(pd.DataFrame({"a": [1]}), tmp_path / "prepared.csv")
    assert (tmp_path / "prepared.csv").read_text(encoding="utf-8") == "old\n"
    assert names_in(tmp_path) == ["prepared.csv"]


# --- main ---------------------------------------------------------------------


def test_main_returns_1_names_the_url_and_writes_nothing(
    app: App, tiny: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = app.fetch.main([], data_dir=tmp_path, opener=failing_opener)
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("fetch_data: ")
    assert FAKE_URL in captured.err
    assert app.fetch.PREPARED_NAME not in names_in(tmp_path)


def test_main_returns_1_for_a_wrong_checksum(
    app: App,
    tiny: Any,
    raw_csv: bytes,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    opener = FakeOpener(b"x" + raw_csv[1:])
    code = app.fetch.main([], data_dir=tmp_path, opener=opener)
    assert code == 1
    assert "SHA-256" in capsys.readouterr().err
    assert names_in(tmp_path) == []


def test_main_returns_1_when_the_folder_cannot_be_written(
    app: App,
    tiny: Any,
    raw_csv: bytes,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    blocker = tmp_path / "data"
    blocker.write_text("this is a file, not a folder", encoding="utf-8")
    code = app.fetch.main([], data_dir=blocker, opener=FakeOpener(raw_csv))
    assert code == 1
    assert "fetch_data: " in capsys.readouterr().err


def test_main_succeeds_and_prints_ascii_progress(
    app: App,
    tiny: Any,
    raw_csv: bytes,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = app.fetch.main([], data_dir=tmp_path, opener=FakeOpener(raw_csv))
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert "wrote" in captured.out
    assert captured.out.isascii()
    assert (tmp_path / app.fetch.PREPARED_NAME).is_file()


# --- prepare: hotels ----------------------------------------------------------


def test_hotels_prepare_keeps_every_column_and_row(
    hotels_fetch: ModuleType, raw_hotels: Any
) -> None:
    source = raw_hotels(120)
    before = source.copy()
    prepared = hotels_fetch.prepare(source)
    pd.testing.assert_frame_equal(prepared, before)
    assert "reservation_status" in prepared.columns
    assert "reservation_status_date" in prepared.columns
    assert prepared is not source
    pd.testing.assert_frame_equal(source, before)


def test_hotels_prepare_names_a_missing_column(
    hotels_fetch: ModuleType, raw_hotels: Any
) -> None:
    source = raw_hotels(10).drop(columns=["lead_time", "adr"])
    with pytest.raises(HexInputError) as caught:
        hotels_fetch.prepare(source)
    assert "lead_time" in str(caught.value)
    assert "adr" in str(caught.value)


# --- prepare: flights ---------------------------------------------------------


def test_flights_prepare_drops_flights_without_an_arrival_delay(
    flights_fetch: ModuleType, raw_flights: Any
) -> None:
    prepared = flights_fetch.prepare(raw_flights([10, math.nan, 20, math.nan]))
    assert prepared["flight"].tolist() == [100, 102]


def test_flights_prepare_late_threshold_is_more_than_15_minutes(
    flights_fetch: ModuleType, raw_flights: Any
) -> None:
    prepared = flights_fetch.prepare(raw_flights([15, 16, -20, 0, 15.5, 200]))
    assert prepared["late"].tolist() == [0, 1, 0, 0, 1, 1]
    assert str(prepared["late"].dtype) == "int64"


def test_flights_prepare_drops_the_row_number_and_post_departure_columns(
    flights_fetch: ModuleType, raw_flights: Any
) -> None:
    prepared = flights_fetch.prepare(raw_flights([1, 2, 3]))
    for name in (
        "rownames",
        "dep_time",
        "dep_delay",
        "arr_time",
        "arr_delay",
        "air_time",
    ):
        assert name not in prepared.columns
    assert prepared.columns[-1] == "late"
    assert prepared.shape[1] == 15
    assert list(prepared.index) == [0, 1, 2]


@pytest.mark.parametrize("row_number_column", ["Unnamed: 0", None])
def test_flights_prepare_handles_other_row_number_columns(
    flights_fetch: ModuleType, raw_flights: Any, row_number_column: str | None
) -> None:
    prepared = flights_fetch.prepare(
        raw_flights([1, 2], row_number_column=row_number_column)
    )
    assert "Unnamed: 0" not in prepared.columns
    assert prepared.shape[1] == 15


@pytest.mark.parametrize(
    "text",
    [
        "2013-01-01 10:00:00",
        "2013-01-01T10:00:00",
        "2013-01-01T10:00:00Z",
        "2013-01-01T05:00:00-05:00",
    ],
)
def test_flights_prepare_writes_iso_utc_for_naive_and_offset_text(
    flights_fetch: ModuleType, raw_flights: Any, text: str
) -> None:
    prepared = flights_fetch.prepare(raw_flights([1], time_hours=[text]))
    assert prepared["time_hour"].tolist() == ["2013-01-01T10:00:00Z"]


def test_flights_prepare_keeps_row_order(
    flights_fetch: ModuleType, raw_flights: Any
) -> None:
    hours = ["2013-03-01T09:00:00Z", "2013-01-01T09:00:00Z", "2013-02-01T09:00:00Z"]
    prepared = flights_fetch.prepare(raw_flights([1, math.nan, 3], time_hours=hours))
    assert prepared["time_hour"].tolist() == [
        "2013-03-01T09:00:00Z",
        "2013-02-01T09:00:00Z",
    ]


def test_flights_prepare_rejects_an_unparseable_time_hour(
    flights_fetch: ModuleType, raw_flights: Any
) -> None:
    source = raw_flights([1, 2])
    source.loc[1, "time_hour"] = "not a time"
    with pytest.raises(HexInputError) as caught:
        flights_fetch.prepare(source)
    assert "time_hour" in str(caught.value)


def test_flights_prepare_names_a_missing_column(
    flights_fetch: ModuleType, raw_flights: Any
) -> None:
    source = raw_flights([1, 2]).drop(columns=["carrier"])
    with pytest.raises(HexInputError) as caught:
        flights_fetch.prepare(source)
    assert "carrier" in str(caught.value)


def test_flights_prepare_does_not_change_its_input(
    flights_fetch: ModuleType, raw_flights: Any
) -> None:
    source = raw_flights([1, math.nan, 30])
    before = source.copy()
    flights_fetch.prepare(source)
    pd.testing.assert_frame_equal(source, before)


@settings(max_examples=50, deadline=None)
@given(
    delays=st.lists(
        st.one_of(
            st.just(math.nan),
            st.floats(
                min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False
            ),
            st.integers(min_value=14, max_value=17).map(float),
        ),
        max_size=40,
    )
)
def test_flights_prepare_properties(
    flights_fetch: ModuleType, delays: list[float]
) -> None:
    source = make_raw_flights(flights_fetch.REQUIRED_COLUMNS, delays)
    prepared = flights_fetch.prepare(source)
    kept = [i for i, delay in enumerate(delays) if not math.isnan(delay)]
    assert prepared["flight"].tolist() == [100 + i for i in kept]
    assert prepared["late"].tolist() == [int(delays[i] > 15) for i in kept]
    assert prepared["time_hour"].tolist() == [source["time_hour"].iloc[i] for i in kept]
    for name in (
        "rownames",
        "dep_time",
        "dep_delay",
        "arr_time",
        "arr_delay",
        "air_time",
    ):
        assert name not in prepared.columns
