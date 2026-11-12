import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from hexperiment.gallery import art
from hexperiment.gallery.markup import ART_TAGS, TAGS, VOID_TAGS, Markup

CONSTANTS = {
    name: value
    for name, value in vars(art).items()
    if name.isupper() and isinstance(value, Markup)
}
EXPECTED = {
    "SPRITE",
    "LANDSCAPE",
    "GHOST_RISE",
    "LEAK_FIGURE",
    "ROWS_TRAIN",
    "ROWS_SCRAMBLED",
    "ROWS_CHANGED",
    "PEEK",
    "GRAVEYARD_LANDING",
    "CORNER_MOON",
    "GRAVEYARD_CASE",
}
SYMBOLS = {"bat", "ghost", "tree", "stone"}
MOCKUPS = (
    Path(__file__).resolve().parents[2] / ".kiro/specs/hexperiment-gallery/mockups"
)
# Where each constant was copied from.
SOURCE = {name: "landing.html" for name in EXPECTED} | {
    "CORNER_MOON": "case.html",
    "GRAVEYARD_CASE": "case.html",
}


class ArtParser(HTMLParser):
    """Collects every start tag with its attributes and the hidden state."""

    def __init__(self, html: str) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool]] = []
        self.tags: list[tuple[str, dict[str, str | None], bool]] = []
        self.errors: list[str] = []
        self.feed(html)
        self.close()

    def _open(self, tag, attrs) -> tuple[dict[str, str | None], bool]:
        attributes = dict(attrs)
        hidden = attributes.get("aria-hidden") == "true" or any(
            h for _, h in self.stack
        )
        self.tags.append((tag, attributes, hidden))
        return attributes, hidden

    def handle_starttag(self, tag, attrs):
        _, hidden = self._open(tag, attrs)
        if tag not in VOID_TAGS:
            self.stack.append((tag, hidden))

    def handle_startendtag(self, tag, attrs):
        self._open(tag, attrs)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1][0] != tag:
            self.errors.append(f"unexpected </{tag}>")
            return
        self.stack.pop()


def parsed(name: str) -> ArtParser:
    return ArtParser(CONSTANTS[name].html)


def test_art_defines_exactly_the_constants_of_the_design():
    assert set(CONSTANTS) == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_constant_is_balanced_and_uses_only_allowed_tags(name):
    result = parsed(name)
    assert not result.errors
    assert not result.stack
    assert result.tags
    for tag, _, _ in result.tags:
        assert tag in ART_TAGS | TAGS, tag


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_no_handler_style_or_script(name):
    assert "<script" not in CONSTANTS[name].html.lower()
    for _, attributes, _ in parsed(name).tags:
        for attribute in attributes:
            assert not attribute.startswith("on"), attribute
            assert attribute != "style"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_links_point_only_at_the_sprite_and_urls_only_at_local_paints(name):
    for _, attributes, _ in parsed(name).tags:
        if "href" in attributes:
            assert attributes["href"] in {f"#{s}" for s in SYMBOLS}
    for found in re.findall(r"url\([^)]*\)", CONSTANTS[name].html):
        assert found in {"url(#sky)", "url(#fence)"}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_drawing_is_hidden_except_the_leak_figure(name):
    for tag, attributes, hidden in parsed(name).tags:
        if tag != "svg":
            continue
        if name == "LEAK_FIGURE":
            assert attributes.get("role") == "img"
            assert (attributes.get("aria-label") or "").strip()
            assert not hidden
        else:
            assert hidden, name


def test_the_sprite_defines_exactly_the_four_symbols():
    ids = {a.get("id") for tag, a, _ in parsed("SPRITE").tags if tag == "symbol"}
    assert ids == SYMBOLS


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_constant_is_copied_from_its_reference_page(name):
    reference = MOCKUPS / SOURCE[name]
    if not reference.is_file():
        pytest.skip("the reference pages are not in this checkout")
    assert CONSTANTS[name].html in reference.read_text(encoding="utf-8")


def test_art_imports_only_literal():
    text = Path(art.__file__).read_text(encoding="utf-8")
    imports = [
        line for line in text.splitlines() if line.startswith(("import", "from"))
    ]
    assert imports == [
        "from __future__ import annotations",
        "from hexperiment.gallery.markup import literal",
    ]
