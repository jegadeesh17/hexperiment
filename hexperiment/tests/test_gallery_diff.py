from html.parser import HTMLParser

from hypothesis import given, settings
from hypothesis import strategies as st

from hexperiment.gallery.difftext import (
    Change,
    Context,
    DiffLine,
    Gap,
    diff_markup,
    diff_scripts,
)

PROPERTY = settings(max_examples=50, deadline=None)


def texts(lines):
    return [line.text for line in lines]


# structure


def test_equal_texts_give_no_blocks():
    result = diff_scripts("a\nb\n", "a\nb\n")
    assert result.blocks == ()
    assert (result.removed, result.added) == (0, 0)


def test_both_empty_give_no_blocks():
    assert diff_scripts("", "").blocks == ()


def test_a_pure_insert():
    result = diff_scripts("a\nb\n", "a\nx\nb\n")
    change = [b for b in result.blocks if isinstance(b, Change)]
    assert len(change) == 1
    assert change[0].removed == ()
    assert change[0].added == (DiffLine(None, 2, "x"),)
    assert (result.removed, result.added) == (0, 1)


def test_a_pure_delete():
    result = diff_scripts("a\nx\nb\n", "a\nb\n")
    (change,) = [b for b in result.blocks if isinstance(b, Change)]
    assert change.removed == (DiffLine(2, None, "x"),)
    assert change.added == ()
    assert (result.removed, result.added) == (1, 0)


def test_a_replace_with_unequal_lengths_keeps_the_pair_together():
    result = diff_scripts("a\nx\ny\nz\nb\n", "a\nw\nb\n")
    (change,) = [b for b in result.blocks if isinstance(b, Change)]
    assert texts(change.removed) == ["x", "y", "z"]
    assert texts(change.added) == ["w"]
    assert (result.removed, result.added) == (3, 1)


def test_line_numbers_on_both_sides():
    old = "a\nb\nc\nd\n"
    new = "a\nX\nY\nc\nd\n"
    result = diff_scripts(old, new)
    context = [
        line for b in result.blocks if isinstance(b, Context) for line in b.lines
    ]
    assert DiffLine(1, 1, "a") in context
    assert DiffLine(3, 4, "c") in context
    assert DiffLine(4, 5, "d") in context
    (change,) = [b for b in result.blocks if isinstance(b, Change)]
    assert change.removed == (DiffLine(2, None, "b"),)
    assert change.added == (DiffLine(None, 2, "X"), DiffLine(None, 3, "Y"))


def test_changes_far_apart_have_gaps_with_counts():
    old = "\n".join(f"l{i}" for i in range(1, 31)) + "\n"
    lines = old.splitlines()
    lines[1] = "changed 2"
    lines[27] = "changed 28"
    result = diff_scripts(old, "\n".join(lines) + "\n")
    kinds = [type(b).__name__ for b in result.blocks]
    assert kinds == [
        "Context",
        "Change",
        "Context",
        "Gap",
        "Context",
        "Change",
        "Context",
    ]
    gap = next(b for b in result.blocks if isinstance(b, Gap))
    # 30 lines; first group covers lines 1-5, second 25-30 -> lines 6-24 hidden.
    assert gap.hidden == 19


def test_leading_and_trailing_gaps():
    old = "\n".join(f"l{i}" for i in range(1, 21)) + "\n"
    lines = old.splitlines()
    lines[9] = "changed"
    result = diff_scripts(old, "\n".join(lines) + "\n")
    assert isinstance(result.blocks[0], Gap)
    assert isinstance(result.blocks[-1], Gap)
    assert result.blocks[0].hidden == 6
    assert result.blocks[-1].hidden == 7


def test_the_context_argument_changes_the_context_size():
    old = "\n".join(f"l{i}" for i in range(1, 11)) + "\n"
    new = old.replace("l5", "X")
    result = diff_scripts(old, new, context=1)
    context_lines = sum(len(b.lines) for b in result.blocks if isinstance(b, Context))
    assert context_lines == 2


def test_a_script_of_more_than_200_lines_is_compared_without_autojunk():
    old = "\n".join(["same"] * 250 + ["end"]) + "\n"
    new = "\n".join(["same"] * 250 + ["other"]) + "\n"
    result = diff_scripts(old, new)
    assert (result.removed, result.added) == (1, 1)


def test_adding_to_an_empty_text():
    result = diff_scripts("", "a\nb\n")
    assert (result.removed, result.added) == (0, 2)


def test_deleting_everything():
    result = diff_scripts("a\nb\n", "")
    assert (result.removed, result.added) == (2, 0)


def test_one_trailing_newline_adds_no_empty_line():
    assert diff_scripts("a\n", "a").blocks == ()
    assert diff_scripts("a\n\n", "a\n").removed == 1


# line endings and separators


def test_crlf_and_lone_cr_input_give_the_same_diff_as_lf():
    old = "a\nb\nc\n"
    new = "a\nX\nc\nd\n"
    expected = diff_scripts(old, new)
    crlf = diff_scripts(old.replace("\n", "\r\n"), new.replace("\n", "\r\n"))
    assert crlf == expected
    assert diff_scripts(old.replace("\n", "\r"), new.replace("\n", "\r")) == expected


def test_no_line_ends_with_a_carriage_return():
    result = diff_scripts("a\r\nb\r\n", "a\r\nc\r\n")
    for block in result.blocks:
        lines = (
            block.lines
            if isinstance(block, Context)
            else (block.removed + block.added if isinstance(block, Change) else ())
        )
        assert all("\r" not in line.text for line in lines)


def test_form_feed_and_line_separator_do_not_split_a_line():
    result = diff_scripts("a\x0cb\u2028c\n", "a\x0cb\u2028d\n")
    (change,) = [b for b in result.blocks if isinstance(b, Change)]
    assert texts(change.removed) == ["a\x0cb\u2028c"]
    assert texts(change.added) == ["a\x0cb\u2028d"]


# markup


class DiffPage(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.start_tags: list[tuple[str, dict[str, str | None]]] = []
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.start_tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


def page(html: str) -> DiffPage:
    parser = DiffPage()
    parser.feed(html)
    parser.close()
    return parser


def markup(old: str, new: str) -> str:
    result = diff_scripts(old, new)
    return diff_markup(result, old_name="haunted/train.py", new_name="train.py").html


def class_of(tag_attrs) -> str:
    return tag_attrs[1].get("class") or ""


def test_the_box_is_a_labelled_focusable_region():
    parsed = page(markup("a\n", "b\n"))
    tag, attrs = parsed.start_tags[0]
    assert tag == "div"
    assert attrs == {
        "class": "diff",
        "role": "region",
        "aria-label": "Code diff: haunted/train.py against train.py",
        "tabindex": "0",
    }


def test_removed_lines_have_a_minus_and_added_lines_a_plus():
    html = markup("keep\nold\n", "keep\nnew\n")
    assert '<div class="removed"><div class="line line-removed">' in html
    assert '<span class="sign">-</span><code>old</code>' in html
    assert '<span class="sign">+</span><code>new</code>' in html
    assert html.index('class="removed"') < html.index('class="added"')


def test_each_line_text_occurs_once():
    html = markup("one\nold\ntwo\n", "one\nnew\ntwo\n")
    for text in ("<code>one</code>", "<code>old</code>", "<code>new</code>"):
        assert html.count(text) == 1


def test_removed_lines_show_one_number_and_context_lines_two():
    html = markup("a\nb\n", "a\nc\n")
    assert '<span class="no">1</span><span class="no">1</span>' in html
    assert '<span class="no">2</span><span class="sign">-</span>' in html
    assert '<span class="no">2</span><span class="sign">+</span>' in html


def test_gap_text_uses_the_singular_for_one_line():
    old = "\n".join(f"l{i}" for i in range(1, 9)) + "\n"
    html = markup(old, old.replace("l5", "X"))
    assert '<div class="gap">1 unchanged line</div>' in html
    old = "\n".join(f"l{i}" for i in range(1, 11)) + "\n"
    html = markup(old, old.replace("l10", "X"))
    assert '<div class="gap">6 unchanged lines</div>' in html


def test_a_hostile_script_line_is_shown_as_text():
    hostile = "<script>alert(1)</script>"
    html = markup("a\n", f"{hostile}\n")
    assert hostile not in html
    parsed = page(html)
    assert hostile in parsed.text
    assert "script" not in [tag for tag, _ in parsed.start_tags]


def test_empty_diff_still_writes_the_box():
    html = markup("a\n", "a\n")
    assert html.startswith('<div class="diff"')


# properties

LINES = st.lists(st.sampled_from(["a", "b", "c", "", "x y"]), max_size=12)


def join(lines):
    return "".join(f"{line}\n" for line in lines)


@PROPERTY
@given(LINES, LINES)
def test_replaying_the_blocks_rebuilds_both_texts(old_lines, new_lines):
    context = len(old_lines) + len(new_lines) + 1
    result = diff_scripts(join(old_lines), join(new_lines), context=context)
    old_rebuilt, new_rebuilt = [], []
    for block in result.blocks:
        if isinstance(block, Context):
            old_rebuilt += texts(block.lines)
            new_rebuilt += texts(block.lines)
        elif isinstance(block, Change):
            old_rebuilt += texts(block.removed)
            new_rebuilt += texts(block.added)
    if old_lines == new_lines:
        assert result.blocks == ()
    else:
        assert old_rebuilt == old_lines
        assert new_rebuilt == new_lines


@PROPERTY
@given(st.text(alphabet="ab \n", max_size=40))
def test_the_diff_of_any_text_against_its_crlf_version_is_empty(text):
    assert diff_scripts(text, text.replace("\n", "\r\n")).blocks == ()
