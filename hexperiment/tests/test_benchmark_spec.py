"""The line model and the model specs of the benchmark (no case is run)."""

from __future__ import annotations

import difflib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hexbench.modelspecs import HGB, LR, MODELS, RF
from hexbench.spec import MARK, Line, both, clean, leak, render


def test_leaky_line_needs_a_note() -> None:
    with pytest.raises(ValueError, match="note"):
        Line("x = 1", "leaky")
    with pytest.raises(ValueError, match="note"):
        leak("x = 1", "   ")


@pytest.mark.parametrize("bad", ["a\nb", "a\rb", "a\u2028b"])
def test_no_line_breaks_in_text_or_note(bad: str) -> None:
    with pytest.raises(ValueError, match="line break"):
        both(bad)
    with pytest.raises(ValueError, match="line break"):
        leak("x = 1", bad)


def test_marker_is_only_for_leaky_lines() -> None:
    with pytest.raises(ValueError, match="LEAK"):
        both("x = 1  # LEAK: sneaky")
    with pytest.raises(ValueError, match="LEAK"):
        clean("x = 1  # LEAK: sneaky")


def test_comment_only_leaky_line() -> None:
    text = render([leak("    ", "copies stay")], "leaky")
    assert text == "    # LEAK: copies stay\n"
    assert render([leak("    ", "copies stay")], "clean") == ""


def test_render_both_variants() -> None:
    lines = [
        both("a = 1"),
        leak("b = fit(all_rows)", "fitted on every row"),
        clean("b = fit(train)"),
        both("c = 3"),
    ]
    assert (
        render(lines, "leaky")
        == f"a = 1\nb = fit(all_rows)  {MARK}fitted on every row\nc = 3\n"
    )
    assert render(lines, "clean") == "a = 1\nb = fit(train)\nc = 3\n"


def test_model_specs_are_fixed_and_deterministic() -> None:
    assert [m.key for m in MODELS] == ["lr", "rf", "hgb"]
    for model in MODELS:
        assert "random_state=0" in model.expression
        assert model.expression.startswith(model.import_line.rsplit(" ", 1)[1] + "(")
    assert "n_jobs=1" in RF.expression
    assert "early_stopping=False" in HGB.expression
    assert "C=10.0" in LR.expression


_BODY = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=20
).filter(lambda s: "# LEAK:" not in s)


@st.composite
def _lines(draw: st.DrawFn) -> list[Line]:
    sides = draw(st.lists(st.sampled_from(["both", "leaky", "clean"]), max_size=25))
    out: list[Line] = []
    for index, side in enumerate(sides):
        text = f"{index}: {draw(_BODY)}"  # the index keeps every line unique
        if side == "leaky":
            note = draw(
                st.text(
                    alphabet=st.characters(min_codepoint=33, max_codepoint=126),
                    min_size=1,
                    max_size=15,
                )
            )
            out.append(leak(text, note))
        else:
            out.append(Line(text, side))  # type: ignore[arg-type]
    return out


@settings(max_examples=50)
@given(_lines())
def test_property_only_leaky_lines_are_marked(lines: list[Line]) -> None:
    leaky = render(lines, "leaky").splitlines()
    clean_ = render(lines, "clean").splitlines()
    assert sum("# LEAK:" in row for row in leaky) == sum(
        line.side == "leaky" for line in lines
    )
    assert not any("# LEAK:" in row for row in clean_)


@settings(max_examples=50)
@given(_lines())
def test_property_diff_is_exactly_leaky_and_clean_lines(lines: list[Line]) -> None:
    leaky = render(lines, "leaky").splitlines()
    clean_ = render(lines, "clean").splitlines()
    diff = list(difflib.ndiff(leaky, clean_))
    removed = [row[2:] for row in diff if row.startswith("- ")]
    added = [row[2:] for row in diff if row.startswith("+ ")]
    expected_removed = [row for row in leaky if "# LEAK:" in row]
    expected_added = [line.text for line in lines if line.side == "clean"]
    assert removed == expected_removed
    assert added == expected_added
