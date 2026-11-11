"""Each README has the content a reader needs (Requirements 4.1, 4.2, 5.2).

The numbers themselves are checked by hand against the recorded measurements; the
tests check that the text is complete and has no placeholder.
"""

from __future__ import annotations

import re

import pytest
from conftest import APPS_DIR, FLIGHTS, HOTELS, App

ATTRIBUTION = {
    HOTELS: (
        'N. Antonio, A. de Almeida, L. Nunes, "Hotel booking demand datasets", '
        "Data in Brief 22 (2019), CC BY 4.0"
    ),
    FLIGHTS: (
        "nycflights13, CC0, original data from the US Bureau of Transportation "
        "Statistics"
    ),
}
GHOSTS = {
    HOTELS: ("The Possession", "The Peeking Spirit"),
    FLIGHTS: ("The Prophet", "The Peeking Spirit"),
}
PLACEHOLDERS = ("TODO", "TBD", "XXX", "<measured")
MEASUREMENT_DATE = "2026-11-11"


@pytest.fixture
def readme(app: App) -> str:
    path = app.folder / "README.md"
    assert path.is_file(), f"{path} is missing"
    return path.read_text(encoding="utf-8")


def test_attribution_text_is_exact(app: App, readme: str) -> None:
    assert ATTRIBUTION[app.name] in readme


def test_source_url_and_checksum_match_the_fetch_script(app: App, readme: str) -> None:
    assert app.fetch.SOURCE.url in readme
    assert app.fetch.SOURCE.sha256 in readme


def test_fetch_and_run_commands(app: App, readme: str) -> None:
    assert f".venv/Scripts/python apps/{app.name}/fetch_data.py" in readme
    assert f".venv/Scripts/hexperiment run apps/{app.name}" in readme


def test_replay_instructions(app: App, readme: str) -> None:
    assert "haunted/train.py" in readme
    assert f"/workflow run exorcise --app apps/{app.name}" in readme
    assert f"cp apps/{app.name}/haunted/train.py apps/{app.name}/train.py" in readme
    assert (
        f"Copy-Item apps/{app.name}/haunted/train.py apps/{app.name}/train.py"
        in readme
    )


def test_ghosts_are_named(app: App, readme: str) -> None:
    for ghost in GHOSTS[app.name]:
        assert ghost in readme


def test_run_time_is_stated_with_its_date(readme: str) -> None:
    assert re.search(r"took \d+\.\d s", readme)
    assert MEASUREMENT_DATE in readme


def test_results_table_has_the_haunted_and_honest_rows(readme: str) -> None:
    assert "`REJECTED`" in readme
    assert "`APPROVED`" in readme
    assert re.search(r"\| \d\.\d{4} \|", readme)


def test_no_placeholder_text(readme: str) -> None:
    for marker in PLACEHOLDERS:
        assert marker not in readme


def test_hotels_readme_explains_the_dropped_test_rows() -> None:
    text = (APPS_DIR / HOTELS / "README.md").read_text(encoding="utf-8")
    for number in ("7,910", "23,933", "16,023"):
        assert number in text
    assert "also occur in the training rows" in text
    assert "Repeated rows inside the test set stay" in text
    assert "status_day" not in text


def test_no_train_py_line_numbers(readme: str) -> None:
    """Line numbers change; the README refers to the marked LEAK lines instead."""
    assert not re.search(r"train\.py:\d+", readme)
    assert not re.search(r"\bline \d+\b", readme)


def test_flights_readme_says_where_the_canary_catches_the_same_day_leak() -> None:
    text = (APPS_DIR / FLIGHTS / "README.md").read_text(encoding="utf-8")
    assert "cutoff day" in text
    assert "the canary cannot see that" in text
    assert "measures the size" not in text
