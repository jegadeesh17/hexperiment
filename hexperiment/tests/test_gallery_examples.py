"""The gallery of the two committed example apps (Requirements 8.2 and 1.6).

Expected values are read straight from ``runs/*/verdict.json`` with ``json``, not
with the code under test.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

from hexperiment.gallery.records import GalleryInputs
from hexperiment.gallery.site import build_site
from hexperiment.gallery.sources import read_app, read_fonts, read_stylesheet
from hexperiment.gallery.writer import write_site

ROOT = Path(__file__).resolve().parents[2]
APP_DIRS = (ROOT / "apps" / "hotel-cancellations", ROOT / "apps" / "flight-delays")


@dataclass(frozen=True)
class Expected:
    name: str
    slug: str
    runs: list[dict]
    app_dir: Path

    @property
    def first(self) -> dict:
        return self.runs[0]

    @property
    def last(self) -> dict:
        return self.runs[-1]

    @property
    def first_rejected(self) -> dict | None:
        rejected = [r for r in self.runs if r["status"] == "REJECTED"]
        return rejected[0] if rejected else None

    @property
    def last_approved(self) -> dict | None:
        approved = [r for r in self.runs if r["status"] == "APPROVED"]
        return approved[-1] if approved else None

    @property
    def error_ghosts(self) -> set[str]:
        return {
            f["ghost"]
            for run in self.runs
            for f in run["findings"]
            if f["severity"] == "error" and f["ghost"]
        }


def score(run: dict) -> str:
    return f"{run['metric']['value']:.4f}"


def _expected(app_dir: Path) -> Expected:
    runs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (app_dir / "runs").glob("*/verdict.json")
    ]
    runs.sort(key=lambda run: run["created_at"])
    with (app_dir / "hex.toml").open("rb") as handle:
        name = tomllib.load(handle)["app"]["name"]
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return Expected(name, slug, runs, app_dir)


def _tree_hash(folders: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for folder in folders:
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            digest.update(path.relative_to(folder).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class Built:
    out: Path
    hash_before: str
    hash_after: str


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    out = tmp_path_factory.mktemp("gallery") / "site"
    before = _tree_hash(APP_DIRS)
    inputs = GalleryInputs(
        apps=tuple(read_app(app_dir) for app_dir in APP_DIRS),
        benchmark=None,
        repo_url=None,
        stylesheet=read_stylesheet(),
        fonts=read_fonts(),
    )
    write_site(build_site(inputs), out)
    return Built(out, before, _tree_hash(APP_DIRS))


@pytest.fixture(scope="module")
def expected() -> list[Expected]:
    return [_expected(app_dir) for app_dir in APP_DIRS]


@pytest.fixture(scope="module")
def landing(built: Built) -> str:
    return (built.out / "index.html").read_text(encoding="utf-8")


def _case(built: Built, app: Expected) -> str:
    return (built.out / app.slug / "index.html").read_text(encoding="utf-8")


def _block(html: str, opening: str, closing: str) -> str:
    start = html.index(opening)
    return html[start : html.index(closing, start)]


def _case_items(landing: str) -> list[str]:
    section = _block(landing, '<ul class="cases">', "</ul>")
    return section.split('<li class="case">')[1:]


def test_expected_values_are_what_the_apps_record(expected: list[Expected]) -> None:
    for app in expected:
        assert len(app.runs) == 4, app.name
        assert app.first["status"] == "REJECTED", app.name
        assert app.last["status"] == "APPROVED", app.name
        assert app.error_ghosts, app.name


def test_landing_states_each_app(landing: str, expected: list[Expected]) -> None:
    items = _case_items(landing)
    assert len(items) == len(expected)
    for item, app in zip(items, expected, strict=True):
        assert f"<h3>{app.name}</h3>" in item
        assert score(app.first) in item
        assert score(app.last) in item
        assert f"first run, {app.first['status'].lower()}" in item
        assert f"last run, {app.last['status'].lower()}" in item
        assert f'href="{app.slug}/index.html"' in item
        for ghost in app.error_ghosts:
            assert f"<em>{ghost}</em>" in item


@pytest.mark.parametrize("index", [0, 1])
def test_case_page_states_the_record(
    built: Built, expected: list[Expected], index: int
) -> None:
    app = expected[index]
    html = _case(built, app)
    assert "<h1" in html and app.name in _block(html, "<h1", "</h1>")
    scores = _block(html, 'class="scores"', 'class="facts"')
    assert score(app.first) in scores
    assert score(app.last) in scores
    assert f"{app.first['status']}" in scores
    assert f"{app.last['status']}" in scores
    for ghost in app.error_ghosts:
        assert f"<em>{ghost}</em>" in html


def test_hero_is_the_highest_first_rejected_score(
    landing: str, expected: list[Expected]
) -> None:
    hero = max(
        expected, key=lambda app: round(app.first_rejected["metric"]["value"], 4)
    )
    others = [app for app in expected if app is not hero]
    trail = _block(landing, '<ol class="trail">', "</ol>")
    assert score(hero.first_rejected) in _block(trail, 'class="n o"', "</li>")
    assert score(hero.last_approved) in _block(trail, 'class="n g"', "</li>")
    assert f"the score {hero.name} claimed on" in trail
    cta = re.search(r'<a [^>]*class="cta"[^>]*>', landing)
    assert cta is not None
    assert f'href="{hero.slug}/index.html"' in cta.group(0)
    for other in others:
        assert f"the score {other.name} claimed" not in trail


@pytest.mark.parametrize("index", [0, 1])
def test_timeline_has_four_entries_in_order(
    built: Built, expected: list[Expected], index: int
) -> None:
    app = expected[index]
    record = _block(_case(built, app), '<ol class="record">', "</ol>")
    heads = re.findall(r'<p class="run-head">(.*?)</p>', record, re.DOTALL)
    assert len(heads) == 4
    assert record.count('<li class="run">') == 4
    for head, run in zip(heads, app.runs, strict=True):
        assert run["status"] in head
        assert f"{run['mode']} check" in head
        if run["metric"]["value"] is not None:
            assert score(run) in head
    times = re.findall(r"<time[^>]*>([^<]*)</time>", record)
    assert [t.replace(" UTC", "").replace(" ", "T") + "Z" for t in times] == [
        run["created_at"] for run in app.runs
    ]


@pytest.mark.parametrize("index", [0, 1])
def test_diff_appears_on_the_case_page(
    built: Built, expected: list[Expected], index: int
) -> None:
    html = _case(built, expected[index])
    assert 'id="fix-title"' in html
    assert 'class="diff"' in html
    assert "haunted/train.py" in html


@pytest.mark.parametrize("index", [0, 1])
def test_copied_verdict_equals_the_last_run_file(
    built: Built, expected: list[Expected], index: int
) -> None:
    app = expected[index]
    last_id = app.last["run_id"]
    source = (app.app_dir / "runs" / last_id / "verdict.json").read_bytes()
    copied = (built.out / app.slug / "verdict.json").read_bytes()
    assert copied == source.replace(b"\r\n", b"\n")


@pytest.mark.parametrize("index", [0, 1])
def test_model_card_of_last_approved_run_is_shown(
    built: Built, expected: list[Expected], index: int
) -> None:
    app = expected[index]
    approved = app.last_approved
    assert approved is not None
    card_path = app.app_dir / "runs" / approved["run_id"] / "model_card.md"
    card = card_path.read_bytes().replace(b"\r\n", b"\n")
    assert (built.out / app.slug / "model_card.md").read_bytes() == card
    html = _case(built, app)
    assert "Model card of the last approved run" in html
    assert '<div class="card">' in html
    assert score(approved) in _block(html, '<div class="card">', "</section>")
    assert 'href="model_card.md"' in html


def test_app_folders_are_unchanged(built: Built) -> None:
    assert built.hash_before == built.hash_after
