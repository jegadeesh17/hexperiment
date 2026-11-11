"""Download, verify and prepare the hotel booking data (Requirement 1).

Run from the repository root:

    .venv/Scripts/python apps/hotel-cancellations/fetch_data.py

The source file ``data/hotels.csv`` is downloaded from a pinned URL only when it is
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
    url="https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv",
    sha256="7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06",
    size=16_855_599,
    filename="hotels.csv",
)
PREPARED_NAME = "prepared.csv"
CHUNK_BYTES = 1 << 20
TIMEOUT_SECONDS = 60  # per network read

# Source columns that ``prepare`` and the training scripts need (Requirement 1.7).
REQUIRED_COLUMNS: tuple[str, ...] = (
    "hotel",
    "is_canceled",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "assigned_room_type",
    "booking_changes",
    "deposit_type",
    "agent",
    "company",
    "days_in_waiting_list",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "reservation_status",
    "reservation_status_date",
)


# --- pure functions -----------------------------------------------------------


def prepare(source: pd.DataFrame) -> pd.DataFrame:
    """Return the prepared frame: every source column and row, unchanged.

    ``reservation_status`` and ``reservation_status_date`` stay, because the haunted
    script uses them (Requirement 1.7). Pure: no file, network or clock.
    """
    missing = [name for name in REQUIRED_COLUMNS if name not in source.columns]
    if missing:
        raise HexInputError(
            f"the source is missing required column(s): {', '.join(missing)}.",
            f"The data is not the pinned {SOURCE.filename}. Delete it and run "
            "fetch_data.py again.",
        )
    return source.copy()


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
        description="Download, verify and prepare the hotel booking data."
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
