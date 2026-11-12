from html.parser import HTMLParser

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hexperiment.gallery.markup import (
    ART_TAGS,
    TAGS,
    Markup,
    code,
    document,
    el,
    esc,
    literal,
    prose,
    spaced,
)

PROPERTY = settings(max_examples=50, deadline=None)


class TextCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []
        self.tags: list[str] = []

    def handle_data(self, data: str) -> None:
        self.text.append(data)

    def handle_starttag(self, tag, attrs) -> None:
        self.tags.append(tag)


def parsed(html: str) -> TextCollector:
    collector = TextCollector()
    collector.feed(html)
    collector.close()
    return collector


# esc


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<script>alert(1)</script>", "&lt;script&gt;alert(1)&lt;/script&gt;"),
        (
            '"><img src=x onerror=alert(1)>',
            "&quot;&gt;&lt;img src=x onerror=alert(1)&gt;",
        ),
        ("&amp;", "&amp;amp;"),
        ("it's", "it&#x27;s"),
        ("a\x00b", "a\ufffdb"),
        ("a\x1fb\x7fc", "a\ufffdb\ufffdc"),
        ("a\rb", "a\ufffdb"),
        ("lone \ud800 surrogate", "lone \ufffd surrogate"),
        ("tab\tand\nline", "tab\tand\nline"),
        ("größe", "größe"),
    ],
)
def test_esc_by_character_class(text, expected):
    assert esc(text) == Markup(expected)


def test_esc_refuses_a_non_string():
    with pytest.raises(TypeError):
        esc(1)  # type: ignore[arg-type]


def test_the_escaped_lone_surrogate_encodes_as_utf_8():
    document(el("p", "\ud800")).decode("utf-8")


# el


def test_strings_are_escaped_and_markup_is_not():
    assert el("p", "<b>", literal("<br>")).html == "<p>&lt;b&gt;<br></p>\n"


@pytest.mark.parametrize("tag", ["svg", "i", "use", "path"])
def test_art_tags_are_refused_by_el(tag):
    with pytest.raises(ValueError, match="not allowed"):
        el(tag)


def test_strong_is_an_inline_tag():
    assert el("strong", "a").html == "<strong>a</strong>"


def test_art_tags_and_tags_do_not_overlap():
    assert not ART_TAGS & TAGS


def test_children_are_flattened_and_none_is_skipped():
    assert el("ul", [el("li", "a"), None, (el("li", "b"),)]).html == (
        "<ul><li>a</li>\n<li>b</li>\n</ul>\n"
    )


@pytest.mark.parametrize(
    "tag", ["script", "style", "iframe", "object", "embed", "form", "img", "br"]
)
def test_tags_outside_the_allowlist_are_refused(tag):
    with pytest.raises(ValueError, match="not allowed"):
        el(tag)


@pytest.mark.parametrize("value", [1, 1.5, True, b"x", {"a": 1}])
def test_a_non_text_child_is_refused(value):
    with pytest.raises(TypeError):
        el("p", value)


@pytest.mark.parametrize(
    "attrs",
    [
        {"onclick": "x()"},
        {"onload": "x()"},
        {"style": "color: red"},
        {"srcdoc": "<p>"},
        {"Bad": "x"},
        {"1a": "x"},
        {"a b": "x"},
    ],
)
def test_bad_attributes_are_refused(attrs):
    with pytest.raises(ValueError):
        el("p", **attrs)


@pytest.mark.parametrize(
    "href",
    [
        "javascript:alert(1)",
        "JavaScript:alert(1)",
        "data:text/html,x",
        "//evil.example/x",
        "ftp://example.com",
        " javascript:alert(1)",
        "java\tscript:alert(1)",
        "a\\b",
        "mailto:a@b.c",
    ],
)
def test_unsafe_addresses_are_refused(href):
    with pytest.raises(ValueError):
        el("a", "x", href=href)


@pytest.mark.parametrize(
    "href",
    [
        "../index.html",
        "assets/gallery.css",
        "#main",
        "https://github.com/o/r",
        "http://x.example/",
    ],
)
def test_relative_and_http_addresses_are_accepted(href):
    assert f'href="{href}"' in el("a", "x", href=href).html


def test_an_attribute_value_is_escaped_with_quotes():
    html = el("a", "x", href="a.html", aria_label='say "hi" <b>').html
    assert 'aria-label="say &quot;hi&quot; &lt;b&gt;"' in html


def test_attribute_names_and_order():
    html = el("p", class_="a", data_value="1", aria_label="b", id="c").html
    assert html == '<p class="a" data-value="1" aria-label="b" id="c"></p>\n'


def test_true_writes_a_bare_attribute_and_false_and_none_omit_it():
    assert el("p", hidden=True, a=False, b=None).html == "<p hidden></p>\n"


def test_an_integer_attribute_value_is_refused():
    with pytest.raises(TypeError):
        el("td", rowspan=2)  # type: ignore[arg-type]


def test_void_tags_have_no_end_tag_and_no_children():
    assert el("meta", charset="utf-8").html == '<meta charset="utf-8">\n'
    assert el("link", rel="stylesheet", href="a.css").html.count("<") == 1
    with pytest.raises(ValueError):
        el("meta", "x")


def test_block_tags_end_with_a_line_feed_and_inline_tags_do_not():
    assert el("p", "a").html.endswith("</p>\n")
    assert el("div").html == "<div></div>\n"
    assert el("em", "a").html == "<em>a</em>"
    assert el("span", "a").html == "<span>a</span>"


def test_inline_text_is_exact():
    html = el("p", el("code", " a  b "), "x").html
    assert html == "<p><code> a  b </code>x</p>\n"
    assert parsed(html).text == [" a  b ", "x", "\n"]


# prose, code, spaced


def test_prose_wraps_backtick_spans_in_code():
    assert prose("use `x < y` now").html == "use <code>x &lt; y</code> now"


def test_prose_leaves_an_unpaired_backtick():
    assert prose("a ` b").html == "a ` b"


def test_prose_does_not_span_lines():
    assert prose("a `b\nc` d").html == "a `b\nc` d"


def test_prose_escapes_a_script_inside_a_code_span():
    assert prose("`<script>`").html == "<code>&lt;script&gt;</code>"


def test_code_escapes():
    assert code("<a>").html == "<code>&lt;a&gt;</code>"


def test_spaced_separates_parts_with_one_space():
    assert spaced("a", code("b"), None, "c").html == "a <code>b</code> c"


# document


def test_document_has_a_doctype_no_bom_and_no_carriage_return():
    data = document(el("p", "a\r\nb"))
    assert data.startswith(b"<!doctype html>\n<p>")
    assert not data.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in data


# properties


@PROPERTY
@given(st.text())
def test_any_text_round_trips_through_esc_and_a_parser(text):
    def replaced(c: str) -> bool:
        return (
            (ord(c) < 32 and c not in "\t\n")
            or ord(c) == 127
            or 0xD800 <= ord(c) <= 0xDFFF
        )

    expected = "".join("\ufffd" if replaced(c) else c for c in text)
    got = "".join(parsed(el("span", text).html).text)
    # HTML parsers turn a carriage return into a line feed; esc already replaced it.
    assert got == expected


@PROPERTY
@given(st.text())
def test_prose_output_has_no_unescaped_angle_bracket_besides_code_tags(text):
    html = prose(text).html
    stripped = html.replace("<code>", "").replace("</code>", "")
    assert "<" not in stripped
    assert ">" not in stripped
    assert set(parsed(html).tags) <= {"code"}
