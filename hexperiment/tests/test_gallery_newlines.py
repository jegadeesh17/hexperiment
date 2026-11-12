import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hexperiment.gallery.newlines import normalize_bytes, normalize_text

PROPERTY = settings(max_examples=50, deadline=None)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a\r\nb", "a\nb"),
        ("a\rb", "a\nb"),
        ("a\n\rb", "a\n\nb"),
        ("a\r\n\r\nb\rc\nd", "a\n\nb\nc\nd"),
        ("no line ending", "no line ending"),
        ("", ""),
        ("\r\n", "\n"),
        ("\r\r", "\n\n"),
    ],
)
def test_normalize_text_cases(text, expected):
    assert normalize_text(text) == expected


def test_other_separators_are_left_alone():
    text = "a\x0bb\x0cc\x85d\u2028e\u2029f"
    assert normalize_text(text) == text


def test_bytes_follow_the_same_rule():
    assert normalize_bytes(b"a\r\nb\rc\n") == b"a\nb\nc\n"
    assert normalize_bytes(b"") == b""


LINE_TEXT = st.text(alphabet="ab \r\n", max_size=40)


@PROPERTY
@given(LINE_TEXT)
def test_both_functions_are_idempotent(text):
    once = normalize_text(text)
    assert normalize_text(once) == once
    data = text.encode()
    assert normalize_bytes(normalize_bytes(data)) == normalize_bytes(data)


@PROPERTY
@given(LINE_TEXT)
def test_text_and_bytes_versions_agree(text):
    assert normalize_bytes(text.encode("utf-8")).decode("utf-8") == normalize_text(text)
    assert "\r" not in normalize_text(text)
