import re
import tomllib
from pathlib import Path, PurePath

import pytest

from gallery_builder import approved, make_source, rejected
from hexperiment.gallery import sources
from hexperiment.gallery.records import GalleryInputs
from hexperiment.gallery.site import build_site
from hexperiment.gallery.sources import read_stylesheet

COLOURS = {
    "prussian": "#17406D",
    "deep-print": "#123359",
    "paper": "#F1F6F9",
    "faded-cyan": "#A9C4D8",
    "lantern": "#F4B04F",
    "verdigris": "#8FE0C4",
}
# Classes the markup carries that need no rule of their own.
STRUCTURAL = {"context", "finding-error", "finding-waived"}


@pytest.fixture(scope="module")
def css() -> str:
    return read_stylesheet().decode("utf-8")


def rule_body(css: str, selector: str) -> str:
    found = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert found, selector
    return found.group(1)


def luminance(hex_colour: str) -> float:
    channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [
        c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_the_stylesheet_ships_as_package_data():
    pyproject = Path(sources.__file__).resolve().parents[3] / "pyproject.toml"
    table = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    patterns = table["tool"]["setuptools"]["package-data"]["hexperiment.gallery"]
    assert any(PurePath("assets/gallery.css").full_match(p) for p in patterns)


def test_the_hex_colours_are_exactly_the_six_of_the_steering_file(css):
    found = {c.upper() for c in re.findall(r"#[0-9A-Fa-f]{3,8}\b", css)}
    assert found == set(COLOURS.values())
    for name, value in COLOURS.items():
        assert f"--{name}: {value};" in css


@pytest.mark.parametrize(
    "word",
    [
        "gradient",
        "box-shadow",
        "text-shadow",
        "transition",
        "@import",
        "text-transform",
        "url(",
    ],
)
def test_forbidden_features_are_absent(css, word):
    assert word not in css


def test_focus_rule(css):
    body = rule_body(css, ":focus-visible")
    assert "outline: 2px solid var(--paper)" in body
    assert "outline-offset: 2px" in body


def test_reduced_motion_wraps_the_only_keyframes(css):
    assert css.count("@keyframes") == 1
    media = css.index("@media (prefers-reduced-motion: no-preference)")
    assert media < css.index("@keyframes") < css.index("@media (min-width: 768px)")
    base = rule_body(css, ".haunted::after")
    assert "translate(0.05em, 0.03em)" in base
    assert "blur(2px)" in base
    assert "opacity: 0.3" in base
    assert "content: attr(data-value)" in base


def test_scroll_boxes_scroll_inside_themselves(css):
    assert "overflow-x: auto" in rule_body(css, ".diff")
    assert "overflow-x: auto" in rule_body(css, ".scroll")


def test_layout_queries_and_widths(css):
    assert "@media (min-width: 768px)" in css
    assert "@media (min-width: 1024px)" in css
    assert "max-width: 72rem" in css
    assert "clamp(5rem, 18vw, 12rem)" in css


def test_font_stacks_and_figures(css):
    assert 'Spectral, Georgia, "Times New Roman", serif' in css
    assert '"IBM Plex Mono", ui-monospace, Consolas, monospace' in css
    assert "font-variant-numeric: tabular-nums lining-nums" in css


def test_status_mark_rules(css):
    assert "border-radius: 4px" in rule_body(css, ".status-rejected .mark")
    assert "var(--lantern)" in rule_body(css, ".status-rejected .mark")
    approved_mark = rule_body(css, ".status-approved .mark")
    assert "var(--verdigris)" in approved_mark
    assert "border-radius: 50%" in approved_mark
    unverified = rule_body(css, ".status-unverified .mark")
    assert "1px dashed var(--faded-cyan)" in unverified


def test_diff_marks_removed_and_added_lines(css):
    assert "var(--lantern)" in rule_body(css, ".diff .line-removed")
    assert "var(--verdigris)" in rule_body(css, ".diff .line-added")


@pytest.mark.parametrize("background", ["prussian", "deep-print"])
@pytest.mark.parametrize("foreground", ["paper", "faded-cyan", "lantern", "verdigris"])
def test_colour_pairs_meet_wcag_aa(foreground, background):
    assert contrast(COLOURS[foreground], COLOURS[background]) >= 4.5


def test_every_class_used_by_the_pages_has_a_rule_or_is_structural(css):
    one, two = (
        make_source("one", runs=[rejected(1.0), approved(0.9)]),
        make_source(
            "two", runs=[rejected(0.8), approved(0.7)], haunted="a\n", train="b\n"
        ),
    )
    files = build_site(
        GalleryInputs(apps=(one, two), benchmark=None, repo_url=None, stylesheet=b"")
    )
    used: set[str] = set()
    for name, data in files.items():
        if name.endswith(".html"):
            for value in re.findall(r'class="([^"]*)"', data.decode("utf-8")):
                used.update(value.split())
    assert {"explain", "haunted", "honest", "diff", "record"} <= used
    missing = {c for c in used - STRUCTURAL if f".{c}" not in css}
    assert not missing
