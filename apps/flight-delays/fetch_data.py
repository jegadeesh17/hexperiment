"""Download, verify and prepare the 2013 New York flights data (Requirement 1).

Run from the repository root:

    .venv/Scripts/python apps/flight-delays/fetch_data.py

The source file ``data/flights.csv`` is downloaded from a pinned URL only when it is
missing or does not match the pinned SHA-256. ``data/prepared.csv`` is rewritten from
the verified source on every run. Neither file is committed.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import os
import sys
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import IO
from urllib.request import Request, urlopen

import pandas as pd

from hexperiment import HexInputError

Opener = Callable[..., AbstractContextManager[IO[bytes]]]  # urlopen-compatible
Progress = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class Source:
    """A pinned download: where it comes from and what it must look like."""

    url: str
    sha256: str
    size: int
    filename: str


# Requirement 1.2.
SOURCE = Source(
    url="https://vincentarelbundock.github.io/Rdatasets/csv/nycflights13/flights.csv",
    sha256="b791a2590c9167aa0bd3cb45af3f64101180878cf088c183eae4011e3a654f42",
    size=33_206_996,
    filename="flights.csv",
)
PREPARED_NAME = "prepared.csv"
CHUNK_BYTES = 1 << 20
TIMEOUT_SECONDS = 60  # per network read

# Source columns that ``prepare`` and the training scripts need (Requirement 1.6).
REQUIRED_COLUMNS: tuple[str, ...] = (
    "year",
    "month",
    "day",
    "dep_time",
    "sched_dep_time",
    "dep_delay",
    "arr_time",
    "sched_arr_time",
    "arr_delay",
    "carrier",
    "flight",
    "tailnum",
    "origin",
    "dest",
    "air_time",
    "distance",
    "hour",
    "minute",
    "time_hour",
)
# The Rdatasets export adds a row-number column; the real file names it "rownames".
ROW_NUMBER_COLUMNS: tuple[str, ...] = ("rownames", "Unnamed: 0")
# Known only after departure (Requirement 1.6).
AFTER_DEPARTURE: tuple[str, ...] = (
    "dep_time",
    "dep_delay",
    "arr_time",
    "arr_delay",
    "air_time",
)
LATE_MINUTES = 15
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


# --- pure functions -----------------------------------------------------------


def prepare(source: pd.DataFrame) -> pd.DataFrame:
    """Return the prepared frame (Requirement 1.6). Pure: no file, network or clock.

    Drops the row-number column and the flights without an arrival delay, adds
    ``late`` (1 when ``arr_delay`` is more than 15 minutes), rewrites ``time_hour``
    as ISO 8601 UTC (text without an offset is taken as UTC), and drops the columns
    that are known only after departure. Row order is kept.
    """
    missing = [name for name in REQUIRED_COLUMNS if name not in source.columns]
    if missing:
        raise HexInputError(
            f"the source is missing required column(s): {', '.join(missing)}.",
            f"The data is not the pinned {SOURCE.filename}. Delete it and run "
            "fetch_data.py again.",
        )
    frame = source.drop(columns=[c for c in ROW_NUMBER_COLUMNS if c in source.columns])
    delay = _numeric(frame["arr_delay"], "arr_delay")
    frame = frame.loc[delay.notna()].copy()
    frame["late"] = (delay.loc[frame.index] > LATE_MINUTES).astype("int64")
    frame["time_hour"] = _iso_utc(frame["time_hour"], "time_hour")
    return frame.drop(columns=list(AFTER_DEPARTURE)).reset_index(drop=True)


def check_digest(actual: str, expected: str, *, url: str) -> None:
    """Raise ``HexInputError`` unless both SHA-256 values match (Requirement 1.4)."""
    if actual.lower() != expected.lower():
        raise HexInputError(
            f"the file downloaded from {url} has SHA-256 {actual}, "
            f"expected {expected}.",
            "The source changed or the download was corrupted; do not use it. Run "
            "fetch_data.py again, and if it repeats, report the new checksum.",
        )


# --- edges: files and network -------------------------------------------------


def sha256_of(path: Path) -> str:
    """Return the SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(source: Source, dest: Path, *, opener: Opener = urlopen) -> None:
    """Download ``source`` to ``dest`` through ``<dest>.part`` (Requirement 1.4).

    The bytes are hashed while they are read, and reading stops as soon as more than
    ``source.size`` bytes arrive. ``dest`` is replaced only after the size and the
    SHA-256 match; on any failure the ``.part`` file is removed and ``dest`` is left
    as it was.
    """
    part = dest.with_name(dest.name + ".part")
    request = Request(source.url, headers={"User-Agent": "hexperiment-fetch-data"})
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with opener(request, timeout=TIMEOUT_SECONDS) as handle:
            digest, total = _stream_to(handle, part, source)
        _check_size(total, source)
        check_digest(digest, source.sha256, url=source.url)
        os.replace(part, dest)
    except (OSError, http.client.HTTPException) as error:
        raise HexInputError(
            f"could not download {source.url} ({error}).",
            "Check the network connection and run fetch_data.py again.",
        ) from None
    finally:
        part.unlink(missing_ok=True)


def write_atomic(frame: pd.DataFrame, dest: Path) -> None:
    """Write ``frame`` as CSV; ``dest`` ends up complete or absent (Requirement 1.4)."""
    part = dest.with_name(dest.name + ".part")
    try:
        frame.to_csv(part, index=False, encoding="utf-8", lineterminator="\n")
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)


def fetch(
    data_dir: Path, *, opener: Opener = urlopen, progress: Progress | None = None
) -> Path:
    """Verify the source, rewrite ``prepared.csv`` from it, return the prepared path."""
    say = progress or _ignore
    data_dir.mkdir(parents=True, exist_ok=True)
    source_path = data_dir / SOURCE.filename
    if _matches_pinned_digest(source_path):
        say(f"{source_path} matches the pinned SHA-256, no download needed.")
    else:
        say(f"downloading {SOURCE.url}")
        download(SOURCE, source_path, opener=opener)
        say(f"downloaded and verified {source_path}")
    prepared = prepare(_read_source(source_path))
    target = data_dir / PREPARED_NAME
    write_atomic(prepared, target)
    say(f"wrote {target} ({len(prepared)} rows, {prepared.shape[1]} columns)")
    return target


def main(
    argv: Sequence[str] | None = None,
    *,
    data_dir: Path | None = None,
    opener: Opener = urlopen,
) -> int:
    """Command line entry point: return 0, or 1 after printing the problem to stderr."""
    parser = argparse.ArgumentParser(
        description="Download, verify and prepare the 2013 New York flights data."
    )
    parser.parse_args(argv)
    directory = (
        data_dir if data_dir is not None else Path(__file__).resolve().parent / "data"
    )
    try:
        fetch(directory, opener=opener, progress=_print_progress)
    except HexInputError as error:
        print(f"fetch_data: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(
            f"fetch_data: could not read or write files in {directory} ({error}). "
            "Check that the folder is writable and the disk has free space.",
            file=sys.stderr,
        )
        return 1
    return 0


# --- helpers ------------------------------------------------------------------


def _numeric(series: pd.Series, column: str) -> pd.Series:
    try:
        return pd.to_numeric(series)
    except (ValueError, TypeError) as error:
        raise HexInputError(
            f"column {column!r} has values that are not numbers ({error}).",
            "The data is not the pinned flights.csv. Delete it and run fetch_data.py "
            "again.",
        ) from None


def _iso_utc(series: pd.Series, column: str) -> pd.Series:
    """Rewrite timestamps as ``YYYY-MM-DDTHH:MM:SSZ``; naive text is taken as UTC."""
    fix = (
        "The data is not the pinned flights.csv. Delete it and run fetch_data.py again."
    )
    try:
        parsed = pd.to_datetime(series, utc=True)
    except (ValueError, TypeError) as error:
        raise HexInputError(
            f"column {column!r} cannot be parsed as timestamps ({error}).", fix
        ) from None
    unparsed = int(parsed.isna().sum())
    if unparsed:
        raise HexInputError(
            f"column {column!r} has {unparsed} missing or unparseable value(s).", fix
        )
    return parsed.dt.strftime(TIME_FORMAT)


def _stream_to(handle: IO[bytes], part: Path, source: Source) -> tuple[str, int]:
    """Copy ``handle`` to ``part``; return the SHA-256 and the byte count."""
    digest = hashlib.sha256()
    total = 0
    with part.open("wb") as out:
        while chunk := handle.read(CHUNK_BYTES):
            total += len(chunk)
            if total > source.size:
                raise HexInputError(
                    f"the download from {source.url} is larger than the expected "
                    f"{source.size} bytes.",
                    "The source changed; do not use it. Run fetch_data.py again, and "
                    "if it repeats, report the problem.",
                )
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest(), total


def _check_size(total: int, source: Source) -> None:
    if total != source.size:
        raise HexInputError(
            f"the file downloaded from {source.url} has {total} bytes, expected "
            f"{source.size}.",
            "The download was cut short or the source changed. Run fetch_data.py "
            "again.",
        )


def _matches_pinned_digest(path: Path) -> bool:
    """True when ``path`` exists and has the pinned SHA-256 (Requirement 1.3)."""
    return path.is_file() and sha256_of(path) == SOURCE.sha256


def _read_source(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
    ) as error:
        raise HexInputError(
            f"{path} cannot be read as CSV ({error}).",
            "Delete the file and run fetch_data.py again.",
        ) from None


def _ignore(message: str) -> None:
    del message


def _print_progress(message: str) -> None:
    print(f"fetch_data: {message}")


if __name__ == "__main__":
    sys.exit(main())
