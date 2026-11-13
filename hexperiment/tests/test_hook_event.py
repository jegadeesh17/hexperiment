"""Tests of ``candidate_paths`` (Requirements 1.2, 1.4, 7.1).

The first example is the event shape that Kiro really writes to a command hook.
"""

import json
import os

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hexperiment.hook_event import (
    MAX_CANDIDATES,
    MAX_EVENT_DEPTH,
    MAX_EVENT_NODES,
    MAX_PATH_LENGTH,
    PATH_KEYS,
    candidate_paths,
    session_id,
)

# The real event: a flat object; the path is absolute with backslashes and a
# lower-case drive letter, while cwd has an upper-case one.
REAL_CWD = "C:\\work\\project"
REAL_PATH = REAL_CWD.replace("C:", "c:", 1) + "\\apps\\churn\\train.py"
REAL_EVENT = {
    "session_id": "sess_01HZ",
    "hook_event_name": "PostFileSave",
    "cwd": REAL_CWD,
    "file_path": REAL_PATH,
}


def paths(event) -> tuple[str, ...]:
    return candidate_paths(json.dumps(event))


def nest(inner, levels: int):
    """``inner`` wrapped in ``levels`` objects, so ``levels`` containers deep."""
    for _ in range(levels):
        inner = {"a": inner}
    return inner


# --- event shapes ------------------------------------------------------------------


def test_the_real_kiro_event_gives_its_absolute_path():
    assert paths(REAL_EVENT) == (REAL_PATH,)


def test_the_real_event_text_with_escaped_backslashes_round_trips():
    text = (
        '{"session_id": "sess_01HZ", "hook_event_name": "PostFileSave", '
        '"cwd": "C:\\\\Users\\\\user\\\\p", '
        '"file_path": "c:\\\\Users\\\\user\\\\p\\\\apps\\\\x\\\\train.py"}'
    )
    assert candidate_paths(text) == ("c:\\Users\\user\\p\\apps\\x\\train.py",)


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ({"file_path": "apps/x/train.py"}, ("apps/x/train.py",)),
        ({"filePath": "apps/x/train.py"}, ("apps/x/train.py",)),
        ({"File-Path": "apps/x/train.py"}, ("apps/x/train.py",)),
        ({"PATH": "apps/x/train.py"}, ("apps/x/train.py",)),
        ({"tool_input": {"path": "apps/x/train.py"}}, ("apps/x/train.py",)),
        ({"event": {"file": {"path": "apps/x/train.py"}}}, ("apps/x/train.py",)),
        (
            {"files": ["apps/x/train.py", "apps/y/train.py"]},
            ("apps/x/train.py",) + ("apps/y/train.py",),
        ),
        ({"files": [["apps/x/train.py"]]}, ("apps/x/train.py",)),
        ({"file_path": "C:\\p\\apps\\x\\train.py"}, ("C:\\p\\apps\\x\\train.py",)),
        ({"file_path": "/home/me/p/apps/x/train.py"}, ("/home/me/p/apps/x/train.py",)),
        ({"file_path": "apps/x/helpers/utils.py"}, ("apps/x/helpers/utils.py",)),
        ({"file_path": "train.py"}, ("train.py",)),
        (
            {"file_path": "C:\\My Project\\apps\\x y\\train.py"},
            ("C:\\My Project\\apps\\x y\\train.py",),
        ),
        ({"file_path": "  apps/x/train.py  "}, ("apps/x/train.py",)),
        ({"file_path": '"apps/x/train.py"'}, ("apps/x/train.py",)),
        ({"file_path": "'apps/x/train.py'"}, ("apps/x/train.py",)),
    ],
    ids=lambda value: None if isinstance(value, dict | tuple) else str(value),
)
def test_paths_are_found_under_path_keys(event, expected):
    assert paths(event) == expected


def test_a_top_level_array_is_searched():
    assert paths([{"path": "apps/x/train.py"}, {"path": "apps/y/train.py"}]) == (
        "apps/x/train.py",
        "apps/y/train.py",
    )


def test_a_file_uri_becomes_a_local_path():
    uri = "file:///c:/Users/me/apps/x/train.py"
    expected = (
        "c:\\Users\\me\\apps\\x\\train.py"
        if os.name == "nt"
        else "/c:/Users/me/apps/x/train.py"
    )
    (found,) = paths({"uri": uri})
    # url2pathname on Windows upper-cases the drive letter.
    assert found.lower() == expected.lower()


def test_a_file_uri_with_encoded_characters_is_decoded():
    (found,) = paths({"uri": "file:///home/me/my%20app/train.py"})
    assert found.replace("\\", "/").endswith("/home/me/my app/train.py")


def test_a_uri_that_is_not_a_file_uri_is_dropped():
    assert paths({"uri": "https://example.com/train.py"}) == ()
    assert paths({"other": "vscode-file://vscode-app/train.py"}) == ()


def test_a_path_at_an_unknown_key_is_found_by_the_second_tier():
    assert paths({"saved": {"where": "apps/x/train.py"}}) == ("apps/x/train.py",)


def test_the_first_tier_wins_over_the_second():
    event = {"note": "apps/other/train.py", "file_path": "apps/x/train.py"}
    assert paths(event) == ("apps/x/train.py",)


def test_the_second_tier_is_used_when_the_event_has_no_value_under_a_path_key():
    assert paths({"saved": "apps/x/train.py", "cwd": "C:\\p"}) == ("apps/x/train.py",)
    assert paths({"file_path": {}, "saved": "apps/x/train.py"}) == ("apps/x/train.py",)
    assert paths({"files": [], "saved": "apps/x/train.py"}) == ("apps/x/train.py",)


def test_a_path_key_that_names_a_non_python_file_stops_the_second_tier():
    event = {"file_path": "apps/x/hex.toml", "note": "apps/y/train.py"}
    assert paths(event) == ()


@pytest.mark.parametrize(
    "value", [None, 5, True, 3.5, "", "apps/x/notes.md", ["apps/x/hex.toml"], [None]]
)
def test_any_value_under_a_path_key_stops_the_second_tier(value):
    event = {"file_path": value, "note": "apps/y/train.py"}
    assert paths(event) == ()


def test_a_value_under_a_nested_path_key_stops_the_second_tier():
    event = {"tool_input": {"path": "apps/x/hex.toml"}, "note": "apps/y/train.py"}
    assert paths(event) == ()


def test_a_python_path_under_a_path_key_still_wins_over_other_strings():
    event = {"note": "apps/y/train.py", "file_path": "apps/x/train.py"}
    assert paths(event) == ("apps/x/train.py",)


def test_file_contents_next_to_the_real_path_are_not_candidates():
    contents = "import helpers\n\nfrom features import encode  # see helpers.py"
    event = {"file_path": "apps/x/train.py", "content": contents}
    assert paths(event) == ("apps/x/train.py",)


def test_file_contents_alone_give_nothing():
    event = {"content": "import helpers\n# see helpers.py", "other": "a\tb.py"}
    assert paths(event) == ()


def test_a_non_python_path_gives_nothing():
    assert paths({"file_path": "apps/x/hex.toml"}) == ()
    assert paths({"file_path": "apps/x/train.py.bak"}) == ()
    assert paths({"file_path": "apps/x/notes.md"}) == ()


def test_an_upper_case_extension_is_python():
    assert paths({"file_path": "apps/x/TRAIN.PY"}) == ("apps/x/TRAIN.PY",)


def test_two_python_paths_keep_their_order():
    event = {"b": "apps/b/train.py", "a": "apps/a/train.py"}
    assert paths(event) == ("apps/b/train.py", "apps/a/train.py")


def test_duplicates_are_removed():
    event = {"file_path": "apps/x/train.py", "path": "apps/x/train.py"}
    assert paths(event) == ("apps/x/train.py",)


def test_at_most_max_candidates_are_returned():
    event = {"files": [f"apps/a{i}/train.py" for i in range(MAX_CANDIDATES + 10)]}
    found = paths(event)
    assert len(found) == MAX_CANDIDATES
    assert found[0] == "apps/a0/train.py"


# --- input that is not an event ----------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["null", "42", "3.5", "true", '"apps/x/train.py"', "{}", "[]", "[1, 2]"],
)
def test_json_that_names_no_path_gives_nothing(text):
    assert candidate_paths(text) == ()


@pytest.mark.parametrize(
    "text", ["", "   ", "\n", "{not json", "apps/x/train.py", "{'a': 1}"]
)
def test_text_that_is_not_json_gives_nothing(text):
    assert candidate_paths(text) == ()


def test_none_gives_nothing_instead_of_raising():
    assert candidate_paths(None) == ()  # type: ignore[arg-type]


def test_non_string_values_under_path_keys_are_ignored():
    event = {
        "file_path": 5,
        "path": None,
        "files": [1, None, {"a": 2}, True],
        "file": True,
        "uri": 3.5,
    }
    assert paths(event) == ()


def test_a_string_with_a_control_character_is_dropped():
    assert paths({"file_path": "apps/x/\ttrain.py"}) == ()
    assert paths({"file_path": "apps/x/train.py\n"}) == ()
    assert paths({"file_path": "apps/x\n/train.py"}) == ()
    assert paths({"file_path": "apps/x/\x00train.py"}) == ()


# --- limits ------------------------------------------------------------------------


def test_a_string_at_the_depth_limit_is_found_and_one_below_it_is_not():
    inside = nest({"path": "apps/x/train.py"}, MAX_EVENT_DEPTH - 1)
    outside = nest({"path": "apps/x/train.py"}, MAX_EVENT_DEPTH)
    assert paths(inside) == ("apps/x/train.py",)
    assert paths(outside) == ()


def test_json_nested_past_the_recursion_limit_gives_nothing():
    assert candidate_paths("[" * 200_000 + "]" * 200_000) == ()
    assert candidate_paths('{"a":' * 50_000) == ()


def test_a_path_after_the_node_limit_is_ignored_and_one_before_it_is_found():
    filler = [0] * (MAX_EVENT_NODES + 5)
    assert paths({"a": filler, "file_path": "apps/x/train.py"}) == ()
    assert paths({"file_path": "apps/x/train.py", "a": filler}) == ("apps/x/train.py",)


def test_a_string_longer_than_the_limit_is_dropped():
    suffix = "/train.py"
    long_path = "a" * (MAX_PATH_LENGTH - len(suffix) + 1) + suffix
    exact = "a" * (MAX_PATH_LENGTH - len(suffix)) + suffix
    assert len(long_path) == MAX_PATH_LENGTH + 1
    assert paths({"file_path": long_path}) == ()
    assert paths({"file_path": exact}) == (exact,)


# --- properties --------------------------------------------------------------------

json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=40)
    | st.sampled_from(["apps/x/train.py", "C:\\p\\a.PY", "file:///a/b.py", "x.py\n"]),
    lambda children: (
        st.lists(children, max_size=5)
        | st.dictionaries(
            st.text(max_size=12) | st.sampled_from(PATH_KEYS), children, max_size=5
        )
    ),
    max_leaves=40,
)


@given(json_values)
def test_candidate_paths_never_raises_and_returns_clean_python_paths(value):
    found = candidate_paths(json.dumps(value))
    assert all(isinstance(path, str) for path in found)
    assert all(path.lower().endswith(".py") for path in found)
    assert len(set(found)) == len(found)
    assert len(found) <= MAX_CANDIDATES


@given(st.text(max_size=200))
def test_candidate_paths_never_raises_on_any_text(text):
    found = candidate_paths(text)
    assert all(path.lower().endswith(".py") for path in found)


planted_paths = st.from_regex(
    r"[A-Za-z0-9_][A-Za-z0-9_./\\-]{0,30}\.py", fullmatch=True
)
noise = st.dictionaries(
    st.sampled_from(["id", "name", "cwd", "kind", "x y"]),
    st.none() | st.booleans() | st.integers(),
    max_size=3,
)
key_spellings = st.sampled_from(
    [*PATH_KEYS, "file_path", "filePath", "File-Path", "FILE_PATH"]
)


@st.composite
def planted_events(draw):
    path = draw(planted_paths)
    value = [path] if draw(st.booleans()) else path
    event = {draw(key_spellings): value, **draw(noise)}
    for _ in range(draw(st.integers(min_value=0, max_value=8))):
        wrapper = draw(st.sampled_from(["object", "list"]))
        event = (
            {"a": event, **draw(noise)} if wrapper == "object" else [event, draw(noise)]
        )
    return path, event


@given(planted_events())
def test_a_planted_path_is_found_at_any_depth_up_to_eight(planted):
    path, event = planted
    assert path in candidate_paths(json.dumps(event))


# --- session_id (Requirement 9) -------------------------------------------------------

STOP_EVENT = {"session_id": "sess_01HZ", "hook_event_name": "Stop", "cwd": REAL_CWD}


def test_the_real_stop_event_gives_its_session_id():
    assert session_id(json.dumps(STOP_EVENT)) == "sess_01HZ"


def test_the_save_event_gives_its_session_id_too():
    assert session_id(json.dumps(REAL_EVENT)) == "sess_01HZ"


@pytest.mark.parametrize("value", ["a", "A-b_9", "x" * 128])
def test_valid_session_ids_are_returned(value):
    assert session_id(json.dumps({"session_id": value})) == value


@pytest.mark.parametrize(
    "event",
    [
        {"hook_event_name": "Stop"},
        {"session_id": None},
        {"session_id": 7},
        {"session_id": ["sess_1"]},
        {"session_id": ""},
        {"session_id": "x" * 129},
        {"session_id": "has space"},
        {"session_id": "a/b"},
        {"session_id": "..\\x"},
        {"session_id": "sess_1\n"},
        {"session_id": "sess_\u00e9"},
        {"event": {"session_id": "sess_1"}},
        ["sess_1"],
        [{"session_id": "sess_1"}],
        "sess_1",
        None,
        7,
    ],
)
def test_other_session_ids_give_none(event):
    assert session_id(json.dumps(event)) is None


@pytest.mark.parametrize("text", ["", "   ", "{not json", "null", "[]", "\x00"])
def test_text_that_is_not_an_event_gives_no_session_id(text):
    assert session_id(text) is None


def test_a_deeply_nested_text_gives_none_without_raising():
    assert session_id("[" * 100_000) is None


@given(st.text(max_size=300))
def test_session_id_never_raises_and_returns_none_or_a_valid_id(text):
    result = session_id(text)
    assert result is None or (0 < len(result) <= 128 and result.isascii())


@given(
    st.recursive(
        st.none() | st.text(max_size=20),
        lambda c: (
            st.lists(c, max_size=3)
            | st.dictionaries(st.text(max_size=8), c, max_size=3)
        ),
        max_leaves=20,
    )
)
def test_session_id_never_raises_on_any_json_value(value):
    result = session_id(json.dumps(value))
    assert result is None or isinstance(result, str)
