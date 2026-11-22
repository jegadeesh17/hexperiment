"""Tests of the write guard (Requirement 3.8).

One test starts a Python process, to see what the command really loads and which
exit code it gives. None starts a training run.
"""

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hexperiment import entry, hook, write_guard

HEAVY = ("pandas", "numpy", "sklearn", "mcp")


def event(cwd: Path, path: object, tool: str = "fs_write") -> str:
    return json.dumps(
        {
            "session_id": "s",
            "hook_event_name": "PreToolUse",
            "cwd": str(cwd),
            "tool_name": tool,
            "tool_input": {"path": path, "text": "x = 1\n"},
        }
    )


REFUSED_PATHS = [
    "helper.py",
    "pyproject.toml",
    ".gitignore",
    ".github/x.yml",
    "power/x.txt",
    "docs/x.md",
    "hexperiment/benchmark/x.py",
    "Hexperiment/src/hexperiment/canary.py",
    "apps/tests/x.py",
    "apps/Tests/x.py",
    "apps/zz-guard/../x.py",
    "apps/zz-guard/VERDICT.json",
    "apps/zz-guard/notes.md",
    "apps/zz-guard/haunted/train.py",
    "apps/zz-guard/Haunted/train.py",
    "apps/zz-guard/data/x.py",
    "apps/zz-guard/runs/r/x.py",
    "apps/zz-guard/.hex/x.py",
    "apps/zz-guard/hexperiment/x.py",
    "apps/zz-guard/lib/data/x.py",
    "apps/zz-guard/fetch_data.py",
    "apps/zz-guard/hexperiment.py",
    "apps/x.py",
    "apps",
    ".kiro/agents/ghost-hunter.json",
]

ALLOWED_PATHS = [
    "apps/zz-guard/train.py",
    "apps/zz-guard/model_utils.py",
    "apps/zz-guard/lib/features.py",
    "apps/zz-guard/./train.py",
]


@pytest.mark.parametrize("path", REFUSED_PATHS)
def test_a_path_outside_the_python_files_of_an_app_is_refused(tmp_path, path):
    reason = write_guard.check_event(event(tmp_path, path))
    assert reason is not None
    assert "\n" not in reason
    assert repr(path) in reason


@pytest.mark.parametrize("path", ALLOWED_PATHS)
def test_a_python_file_inside_an_app_is_allowed(tmp_path, path):
    assert write_guard.check_event(event(tmp_path, path)) is None


def test_an_absolute_path_to_a_python_file_of_an_app_inside_cwd_is_allowed(tmp_path):
    target = tmp_path / "apps" / "zz-guard" / "train.py"
    assert write_guard.check_event(event(tmp_path, str(target))) is None


def test_an_absolute_path_outside_cwd_is_refused(tmp_path):
    cwd = tmp_path / "work"
    other = tmp_path / "other" / "apps" / "zz-guard" / "train.py"
    assert write_guard.check_event(event(cwd, str(other))) is not None


def test_a_sibling_folder_that_starts_with_apps_is_refused(tmp_path):
    assert write_guard.check_event(event(tmp_path, "apps2/zz-guard/train.py"))


def test_a_path_that_leaves_an_app_and_comes_back_is_judged_by_where_it_ends(tmp_path):
    assert write_guard.check_event(event(tmp_path, "apps/a/../b/train.py")) is None
    assert write_guard.check_event(event(tmp_path, "apps/a/data/../train.py")) is None
    assert write_guard.check_event(event(tmp_path, "apps/a/x/../data/train.py"))


@pytest.mark.parametrize("tool", ["fs_write", "str_replace", "fs_append"])
def test_the_tool_name_does_not_change_the_answer(tmp_path, tool):
    assert write_guard.check_event(event(tmp_path, "apps/a/train.py", tool)) is None
    assert write_guard.check_event(event(tmp_path, "helper.py", tool))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "not json",
        "[]",
        "null",
        '"apps/a/train.py"',
        "{}",
        '{"cwd": "C:/p"}',
        '{"cwd": "C:/p", "tool_input": {}}',
        '{"cwd": "C:/p", "tool_input": {"path": ""}}',
        '{"cwd": "C:/p", "tool_input": {"path": 5}}',
        '{"cwd": "C:/p", "tool_input": "apps/a/train.py"}',
        '{"tool_input": {"path": "apps/a/train.py"}}',
        '{"cwd": 7, "tool_input": {"path": "apps/a/train.py"}}',
        '{"cwd": "C:/p", "tool_input": {"path": "apps/a/\\u0000train.py"}}',
        "[" * 100000,
    ],
)
def test_input_that_cannot_be_read_is_refused(text):
    reason = write_guard.check_event(text)
    assert reason is not None
    assert "\n" not in reason


def test_a_path_with_a_line_break_gives_a_one_line_reason(tmp_path):
    reason = write_guard.check_event(event(tmp_path, "x\ny.md"))
    assert reason is not None
    assert "\n" not in reason


def feed(monkeypatch, text: str) -> None:
    stream = io.TextIOWrapper(io.BytesIO(text.encode("utf-8")), encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stream)


def test_main_exits_0_and_prints_nothing_for_an_allowed_write(
    tmp_path, monkeypatch, capsys
):
    feed(monkeypatch, event(tmp_path, "apps/a/train.py"))
    assert write_guard.main() == 0
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_main_exits_2_with_one_line_on_standard_error_for_a_refused_write(
    tmp_path, monkeypatch, capsys
):
    feed(monkeypatch, event(tmp_path, "docs/x.md"))
    assert write_guard.main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(captured.err.strip().splitlines()) == 1
    assert "docs/x.md" in captured.err


def test_main_exits_2_on_empty_and_on_malformed_input(monkeypatch, capsys):
    for text in ("", "{", "\xff"):
        feed(monkeypatch, text)
        assert write_guard.main() == 2
        assert capsys.readouterr().err.strip()


def test_main_reads_at_most_the_event_limit(tmp_path, monkeypatch):
    padded = event(tmp_path, "apps/a/train.py") + " " * hook.MAX_EVENT_BYTES
    feed(monkeypatch, padded)
    assert write_guard.main() == 0
    feed(monkeypatch, " " * hook.MAX_EVENT_BYTES + event(tmp_path, "docs/x.md"))
    assert write_guard.main() == 2


def test_main_exits_2_when_reading_fails(monkeypatch):
    def broken() -> str:
        raise OSError("no input")

    monkeypatch.setattr(write_guard, "read_event_text", broken)
    assert write_guard.main() == 2


def test_the_entry_point_routes_guard_write_to_the_guard(monkeypatch):
    monkeypatch.setattr(write_guard, "main", lambda: 6)
    assert entry.main(["guard-write"]) == 6


@given(st.text(max_size=200))
def test_any_text_gives_none_or_one_line_and_never_raises(text):
    reason = write_guard.check_event(text)
    assert reason is None or "\n" not in reason


@given(
    st.lists(
        st.sampled_from(["a", "b", "..", ".", "data", "haunted", "x.py", "apps"]),
        max_size=6,
    )
)
def test_an_allowed_path_never_resolves_into_a_refused_place(parts):
    cwd = Path(os.path.realpath(os.getcwd()))
    path = "/".join(parts) or "x"
    if write_guard.check_event(event(cwd, path)) is not None:
        return
    resolved = Path(os.path.realpath(cwd / path)).relative_to(
        os.path.realpath(cwd / "apps")
    )
    assert resolved.suffix == ".py"
    assert len(resolved.parts) >= 2
    assert not {"haunted", "data", "runs", ".hex", "hexperiment"} & set(
        resolved.parts[1:-1]
    )


RUN_GUARD = (
    "import sys\n"
    "from hexperiment.entry import main\n"
    "code = main(['guard-write'])\n"
    f"loaded = [m for m in {HEAVY!r} if m in sys.modules]\n"
    "cli = 'hexperiment.cli' in sys.modules\n"
    "sys.stderr.write(f'\\nLOADED={loaded} CLI={cli}')\n"
    "sys.exit(code)\n"
)


def run_guard(cwd: Path, text: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", RUN_GUARD],
        input=text,
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )


def test_the_command_refuses_with_exit_2_and_loads_nothing_heavy(tmp_path):
    result = run_guard(tmp_path, event(tmp_path, "helper.py"))
    assert result.returncode == 2
    assert result.stdout == ""
    message, _, marker = result.stderr.rpartition("\nLOADED=")
    assert "helper.py" in message
    assert marker == "[] CLI=False"


def test_the_command_allows_with_exit_0_and_loads_nothing_heavy(tmp_path):
    result = run_guard(tmp_path, event(tmp_path, "apps/a/train.py"))
    assert result.returncode == 0
    assert result.stderr.strip() == "LOADED=[] CLI=False"
