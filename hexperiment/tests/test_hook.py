"""Tests of ``handle_hook`` and ``format_hook_output`` (Requirements 1.2 to 1.6, 7.1).

A stub runner stands in for ``cli.run_app``, so no test here starts a training run.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hexperiment import canary, config
from hexperiment import hook as hook_module
from hexperiment import workspace as workspace_module
from hexperiment.errors import HexInputError
from hexperiment.hook import (
    FULL_CHECK_NOTICE,
    HOOK_RUN_CAP_SECONDS,
    HOOK_TIMEOUT_SECONDS,
    SESSION_FILE,
    format_hook_output,
    format_stop_message,
    handle_hook,
    handle_stop,
    record_session,
)
from hexperiment.models import RunOutcome, Status

HEX_TOML = """\
[app]
name = "demo"

[data]
path = "data.csv"

[target]
column = "target"

[metric]
name = "roc_auc"
threshold = 0.8
"""

UNVERIFIED_TEXT = (
    "UNVERIFIED (fast) roc_auc 0.8500 >= 0.8000  run 20260101-000000-abcdef"
)
REJECTED_TEXT = (
    "REJECTED (fast) roc_auc 0.8500 >= 0.8000  run 20260101-000000-abcdef\n"
    "  error    The Possession  tenure_months  tenure_months alone predicts the target"
)
APPROVED_TEXT = "APPROVED (full) roc_auc 0.8500 >= 0.8000  run 20260101-000000-abcdef"
FAILURE_TEXT = "hexperiment: train.py exited with code 1\nValueError: boom"

UNVERIFIED = RunOutcome(3, Status.UNVERIFIED, UNVERIFIED_TEXT, "")
REJECTED = RunOutcome(1, Status.REJECTED, REJECTED_TEXT, "")
APPROVED = RunOutcome(0, Status.APPROVED, APPROVED_TEXT, "")
FAILED = RunOutcome(2, None, "", FAILURE_TEXT)


class StubRun:
    """A ``RunApp`` that records its calls and returns a prepared outcome."""

    def __init__(self, outcome: RunOutcome = UNVERIFIED) -> None:
        self.outcome = outcome
        self.calls: list[tuple[Path, bool, float | None]] = []

    def __call__(
        self, app_dir: Path, *, full: bool, timeout_cap: float | None = None
    ) -> RunOutcome:
        self.calls.append((app_dir, full, timeout_cap))
        return self.outcome


def make_app(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "hex.toml").write_text(HEX_TOML, encoding="utf-8")
    (folder / "train.py").write_text("print('train')\n", encoding="utf-8")
    return folder


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x = 1\n", encoding="utf-8")
    return path


def event(**fields) -> str:
    return json.dumps(fields)


def notice(app_relative: str) -> str:
    return FULL_CHECK_NOTICE.format(app=app_relative)


@pytest.fixture
def workspace(tmp_path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


# --- finding the app ---------------------------------------------------------------


def test_a_saved_file_of_an_app_runs_the_app_in_fast_mode(workspace):
    app = make_app(workspace / "apps" / "churn")
    run = StubRun()
    text = event(file_path=str(app / "train.py"), cwd=str(workspace))
    output = handle_hook(text, workspace=workspace, run=run)
    assert run.calls == [(app.resolve(), False, HOOK_RUN_CAP_SECONDS)]
    assert output == f"{UNVERIFIED_TEXT}\n{notice('apps/churn')}"


def test_a_helper_file_inside_an_app_runs_that_app(workspace):
    app = make_app(workspace / "apps" / "churn")
    helper = touch(app / "features" / "encode.py")
    run = StubRun()
    handle_hook(event(file_path=str(helper)), workspace=workspace, run=run)
    assert [call[0] for call in run.calls] == [app.resolve()]


def test_a_relative_path_is_taken_relative_to_the_workspace(workspace):
    app = make_app(workspace / "apps" / "churn")
    run = StubRun()
    handle_hook(event(file_path="apps/churn/train.py"), workspace=workspace, run=run)
    assert [call[0] for call in run.calls] == [app.resolve()]


@pytest.mark.skipif(os.name != "nt", reason="drive letters exist only on Windows")
def test_the_real_event_spelling_with_a_lower_case_drive_runs_the_app(workspace):
    app = make_app(workspace / "apps" / "churn")
    saved = str(app / "train.py")
    real_path = saved[0].lower() + saved[1:]
    cwd = str(workspace)
    text = event(
        session_id="sess_1",
        hook_event_name="PostFileSave",
        cwd=cwd[0].upper() + cwd[1:],
        file_path=real_path,
    )
    run = StubRun()
    output = handle_hook(text, workspace=workspace, run=run)
    assert [call[0] for call in run.calls] == [app.resolve()]
    assert output.endswith(notice("apps/churn"))


def test_the_app_name_in_the_notice_has_forward_slashes(workspace):
    app = make_app(workspace / "apps" / "churn")
    output = handle_hook(
        event(file_path=str(app / "train.py")), workspace=workspace, run=StubRun()
    )
    assert output.endswith("run: hexperiment run apps/churn")
    assert "\\" not in output.splitlines()[-1]


def test_an_app_path_with_a_space_is_quoted_in_the_notice(workspace):
    app = make_app(workspace / "apps" / "my model")
    output = handle_hook(
        event(file_path=str(app / "train.py")), workspace=workspace, run=StubRun()
    )
    assert output.endswith('run: hexperiment run "apps/my model"')


@pytest.mark.parametrize("name", ["a\tb", "a b", "a\u00a0b"])
def test_any_whitespace_in_the_app_path_gets_quotes(name):
    output = format_hook_output(UNVERIFIED, f"apps/{name}")
    assert output.endswith(f'run: hexperiment run "apps/{name}"')


def test_an_app_path_without_whitespace_is_not_quoted():
    output = format_hook_output(UNVERIFIED, "apps/my-model_2")
    assert output.endswith("run: hexperiment run apps/my-model_2")


def test_an_app_at_the_workspace_root_is_named_with_a_dot(workspace):
    make_app(workspace)
    output = handle_hook(
        event(file_path=str(workspace / "train.py")),
        workspace=workspace,
        run=StubRun(),
    )
    assert output.endswith("run: hexperiment run .")


def test_two_files_of_the_same_app_start_one_run(workspace):
    app = make_app(workspace / "apps" / "churn")
    other = touch(app / "helpers.py")
    run = StubRun()
    text = event(files=[str(app / "train.py"), str(other)])
    handle_hook(text, workspace=workspace, run=run)
    assert len(run.calls) == 1


# --- when nothing is run -----------------------------------------------------------


def test_a_python_file_outside_every_app_gives_nothing(workspace):
    make_app(workspace / "apps" / "churn")
    tool = touch(workspace / "scripts" / "tool.py")
    run = StubRun()
    assert handle_hook(event(file_path=str(tool)), workspace=workspace, run=run) == ""
    assert run.calls == []


@pytest.mark.parametrize(
    "text",
    ["", "   ", "{not json", "null", "42", '"apps/x/train.py"', "[]", "{}"],
)
def test_input_that_names_no_file_gives_nothing(text, workspace):
    make_app(workspace / "apps" / "churn")
    run = StubRun()
    assert handle_hook(text, workspace=workspace, run=run) == ""
    assert run.calls == []


def test_a_non_python_file_of_an_app_gives_nothing(workspace):
    app = make_app(workspace / "apps" / "churn")
    run = StubRun()
    for name in ("hex.toml", "notes.md", "train.py.bak"):
        saved = app / name
        if not saved.exists():
            touch(saved)
        text = event(file_path=str(saved))
        assert handle_hook(text, workspace=workspace, run=run) == ""
    assert run.calls == []


def test_a_file_that_does_not_exist_is_skipped(workspace):
    app = make_app(workspace / "apps" / "churn")
    run = StubRun()
    text = event(file_path=str(app / "deleted.py"))
    assert handle_hook(text, workspace=workspace, run=run) == ""
    assert run.calls == []


def test_a_folder_named_like_a_python_file_is_skipped(workspace):
    app = make_app(workspace / "apps" / "churn")
    (app / "package.py").mkdir()
    run = StubRun()
    text = event(file_path=str(app / "package.py"))
    assert handle_hook(text, workspace=workspace, run=run) == ""
    assert run.calls == []


def test_a_file_outside_the_workspace_is_skipped_even_inside_an_app(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    foreign = make_app(tmp_path / "foreign")
    run = StubRun()
    text = event(file_path=str(foreign / "train.py"))
    assert handle_hook(text, workspace=workspace, run=run) == ""
    assert run.calls == []


def test_a_path_with_dot_dot_that_leaves_the_workspace_is_skipped(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    make_app(tmp_path / "foreign")
    run = StubRun()
    text = event(file_path="../foreign/train.py")
    assert handle_hook(text, workspace=workspace, run=run) == ""
    assert run.calls == []


def fail_on_file_system_access(monkeypatch, only_if=None):
    """Make the calls that reach the file system fail.

    With ``only_if``, a call fails only for a path that the function accepts, so
    a test can let a later, local candidate through.
    """

    def guard(original):
        def checked(path, *args, **kwargs):
            if only_if is None or only_if(str(path)):
                raise AssertionError(f"file system access to {str(path)!r}")
            return original(path, *args, **kwargs)

        return checked

    def guard_method(original):
        def checked(self, *args, **kwargs):
            if only_if is None or only_if(str(self)):
                raise AssertionError(f"file system access to {str(self)!r}")
            return original(self, *args, **kwargs)

        return checked

    for name in ("is_file", "exists", "is_dir", "resolve", "stat", "lstat"):
        monkeypatch.setattr(Path, name, guard_method(getattr(Path, name)))
    for name in ("stat", "lstat"):
        monkeypatch.setattr(os, name, guard(getattr(os, name)))
    for name in ("exists", "isfile", "isdir", "realpath"):
        monkeypatch.setattr(os.path, name, guard(getattr(os.path, name)))


UNC_EVENT = {"file_path": "\\\\netserver\\share\\apps\\x\\train.py"}
UNC_URI_EVENT = {"uri": "file://netserver/share/apps/x/train.py"}


@pytest.mark.skipif(os.name != "nt", reason="UNC paths exist only on Windows")
def test_a_unc_path_is_skipped_without_touching_the_file_system(workspace, monkeypatch):
    make_app(workspace / "apps" / "x")
    run = StubRun()
    text = event(**UNC_EVENT)
    fail_on_file_system_access(monkeypatch)
    assert handle_hook(text, workspace=workspace, run=run) == ""
    assert run.calls == []


def test_a_file_uri_with_a_host_is_skipped_without_touching_the_file_system(
    workspace, monkeypatch
):
    make_app(workspace / "apps" / "x")
    run = StubRun()
    text = event(**UNC_URI_EVENT)
    fail_on_file_system_access(monkeypatch)
    assert handle_hook(text, workspace=workspace, run=run) == ""
    assert run.calls == []


@pytest.mark.parametrize(
    "path",
    [
        "../foreign/train.py",
        "apps/../../foreign/train.py",
        "apps/x/../../../foreign/train.py",
        "/elsewhere/apps/x/train.py",
    ],
)
def test_a_path_outside_the_workspace_is_skipped_by_its_text_alone(
    workspace, monkeypatch, path
):
    make_app(workspace / "apps" / "x")
    run = StubRun()
    text = event(file_path=path)
    fail_on_file_system_access(monkeypatch)
    assert handle_hook(text, workspace=workspace, run=run) == ""
    assert run.calls == []


def test_a_network_candidate_before_a_local_one_does_not_stop_the_local_run(
    workspace, monkeypatch
):
    app = make_app(workspace / "apps" / "x")
    local = str(app / "train.py")
    candidates = ["file://netserver/share/apps/x/train.py"]
    if os.name == "nt":
        candidates.insert(0, "\\\\netserver\\share\\apps\\x\\train.py")
    text = event(files=[*candidates, local])
    fail_on_file_system_access(monkeypatch, only_if=lambda p: "netserver" in p)
    run = StubRun()
    output = handle_hook(text, workspace=workspace, run=run)
    assert [call[0] for call in run.calls] == [app.resolve()]
    assert output.endswith(notice("apps/x"))


def test_a_path_with_dot_dot_segments_inside_the_workspace_is_normalised(
    workspace,
):
    app = make_app(workspace / "apps" / "x")
    run = StubRun()
    dotted = str(workspace / "apps" / "y" / ".." / "x" / "train.py")
    handle_hook(event(file_path=dotted), workspace=workspace, run=run)
    assert [call[0] for call in run.calls] == [app.resolve()]


def test_the_first_usable_candidate_wins(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    foreign = make_app(tmp_path / "foreign")
    stray = touch(workspace / "scripts" / "tool.py")
    first = make_app(workspace / "apps" / "first")
    second = make_app(workspace / "apps" / "second")
    run = StubRun()
    text = event(
        files=[
            str(workspace / "apps" / "ghost" / "train.py"),  # does not exist
            str(foreign / "train.py"),  # outside the workspace
            str(stray),  # in no app
            str(first / "train.py"),
            str(second / "train.py"),
        ]
    )
    output = handle_hook(text, workspace=workspace, run=run)
    assert run.calls == [(first.resolve(), False, HOOK_RUN_CAP_SECONDS)]
    assert output.endswith(notice("apps/first"))


# --- the call and its limits -------------------------------------------------------


def test_the_run_is_fast_and_capped_below_the_hook_timeout(workspace):
    app = make_app(workspace / "apps" / "churn")
    run = StubRun()
    handle_hook(event(file_path=str(app / "train.py")), workspace=workspace, run=run)
    ((_, full, cap),) = run.calls
    assert full is False
    assert cap == HOOK_RUN_CAP_SECONDS == 100
    assert HOOK_TIMEOUT_SECONDS == 120
    assert cap < HOOK_TIMEOUT_SECONDS


# --- the output --------------------------------------------------------------------


def test_an_unverified_run_prints_the_summary_and_the_notice():
    assert format_hook_output(UNVERIFIED, "apps/churn") == (
        f"{UNVERIFIED_TEXT}\n"
        "Fast check only: the canary test did not run, so this result can never be "
        "APPROVED. Before you call the work done, run: hexperiment run apps/churn"
    )


def test_a_rejected_run_prints_every_finding_line_and_the_notice():
    output = format_hook_output(REJECTED, "apps/churn")
    lines = output.splitlines()
    assert lines[:2] == REJECTED_TEXT.splitlines()
    assert "The Possession" in lines[1]
    assert lines[-1] == notice("apps/churn")
    assert len(lines) == 3


def test_an_approved_outcome_is_formatted_like_any_other():
    output = format_hook_output(APPROVED, "apps/churn")
    assert output == f"{APPROVED_TEXT}\n{notice('apps/churn')}"


def test_a_failed_run_prints_the_failure_message_and_the_notice():
    output = format_hook_output(FAILED, "apps/churn")
    assert output == f"{FAILURE_TEXT}\n{notice('apps/churn')}"
    assert "ValueError: boom" in output


def test_a_failed_run_never_prints_stdout_text():
    mixed = RunOutcome(2, None, "stale summary", "hexperiment: failed")
    output = format_hook_output(mixed, "apps/x")
    assert "stale summary" not in output
    assert output.startswith("hexperiment: failed\n")


def test_an_outcome_without_text_gives_only_the_notice():
    empty = RunOutcome(3, Status.UNVERIFIED, "", "")
    assert format_hook_output(empty, "apps/x") == notice("apps/x")


def test_trailing_newlines_do_not_leave_a_blank_line():
    padded = RunOutcome(3, Status.UNVERIFIED, UNVERIFIED_TEXT + "\n\n", "")
    assert (
        format_hook_output(padded, "apps/x") == f"{UNVERIFIED_TEXT}\n{notice('apps/x')}"
    )


def test_braces_in_the_app_name_are_kept():
    output = format_hook_output(UNVERIFIED, "apps/{weird}")
    assert output.endswith("run: hexperiment run apps/{weird}")


def test_handle_hook_formats_a_failed_run_for_the_agent(workspace):
    app = make_app(workspace / "apps" / "churn")
    output = handle_hook(
        event(file_path=str(app / "train.py")),
        workspace=workspace,
        run=StubRun(FAILED),
    )
    assert output == f"{FAILURE_TEXT}\n{notice('apps/churn')}"


def test_handle_hook_formats_a_rejected_run_for_the_agent(workspace):
    app = make_app(workspace / "apps" / "churn")
    output = handle_hook(
        event(file_path=str(app / "train.py")),
        workspace=workspace,
        run=StubRun(REJECTED),
    )
    assert output == f"{REJECTED_TEXT}\n{notice('apps/churn')}"


# --- the stop check (Requirement 9) ---------------------------------------------------

SESSION = "sess_01HZ"
OTHER_SESSION = "sess_other"
STOP_EVENT = json.dumps(
    {"session_id": SESSION, "hook_event_name": "Stop", "cwd": "C:\\work"}
)


def session_app(folder: Path, *, session: str | None = SESSION, status=None) -> Path:
    """An app with a session file and, when ``status`` is given, a verdict."""
    make_app(folder)
    if session is not None:
        record_session(folder, session)
    if status is not None:
        (folder / "verdict.json").write_text(
            json.dumps({"status": status}), encoding="utf-8"
        )
    return folder


def tree(root: Path) -> dict[str, bytes | None]:
    return {
        str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
        for p in sorted(root.rglob("*"))
    }


def test_the_light_workspace_resolution_matches_the_original(tmp_path, monkeypatch):
    (tmp_path / "chosen").mkdir()
    monkeypatch.chdir(tmp_path)
    for value in (None, "", "  ", "chosen", str(tmp_path / "chosen")):
        if value is None:
            monkeypatch.delenv(workspace_module.ENV_WORKSPACE, raising=False)
        else:
            monkeypatch.setenv(workspace_module.ENV_WORKSPACE, value)
        assert hook_module._resolve_workspace() == workspace_module.resolve_workspace()
    monkeypatch.setenv(workspace_module.ENV_WORKSPACE, "nowhere")
    with pytest.raises(HexInputError, match="HEXPERIMENT_WORKSPACE"):
        hook_module._resolve_workspace()


def test_the_names_copied_for_the_light_import_equal_the_originals():
    assert hook_module.ENV_WORKSPACE == workspace_module.ENV_WORKSPACE
    assert hook_module.MAX_EVENT_BYTES == 1024 * 1024
    assert hook_module.CONFIG_NAME == config.CONFIG_FILENAME
    assert hook_module.SCRIPT_NAME == canary.SCRIPT_NAME
    assert hook_module.VENV_MARKER == workspace_module.VENV_MARKER
    assert hook_module.SKIPPED_FOLDERS == workspace_module.SKIPPED_FOLDERS


def test_record_session_writes_the_id_with_a_final_newline(tmp_path):
    app = make_app(tmp_path / "app")
    record_session(app, SESSION)
    assert (app / ".hex" / SESSION_FILE).read_bytes() == SESSION.encode() + b"\n"
    assert SESSION_FILE == "agent-session"


def test_record_session_replaces_an_earlier_id(tmp_path):
    app = make_app(tmp_path / "app")
    record_session(app, SESSION)
    record_session(app, OTHER_SESSION)
    assert (app / ".hex" / SESSION_FILE).read_text() == OTHER_SESSION + "\n"


def test_record_session_ignores_a_failed_write(tmp_path):
    app = make_app(tmp_path / "app")
    (app / ".hex").write_text("not a folder", encoding="utf-8")
    record_session(app, SESSION)  # must not raise


def test_the_hook_records_the_session_before_the_run(workspace):
    app = make_app(workspace / "apps" / "churn")
    seen: list[str] = []

    def run(app_dir, *, full, timeout_cap=None):
        seen.append((app_dir / ".hex" / SESSION_FILE).read_text(encoding="utf-8"))
        return UNVERIFIED

    text = event(file_path=str(app / "train.py"), session_id=SESSION)
    handle_hook(text, workspace=workspace, run=run)
    assert seen == [SESSION + "\n"]


def test_an_event_without_a_session_id_records_nothing(workspace):
    app = make_app(workspace / "apps" / "churn")
    handle_hook(
        event(file_path=str(app / "train.py")), workspace=workspace, run=StubRun()
    )
    assert not (app / ".hex" / SESSION_FILE).exists()


def test_a_failed_session_write_does_not_change_the_hook_output(workspace):
    app = make_app(workspace / "apps" / "churn")
    text = event(file_path=str(app / "train.py"), session_id=SESSION)
    clean = handle_hook(text, workspace=workspace, run=StubRun())
    (app / ".hex" / SESSION_FILE).unlink()
    (app / ".hex").rmdir()
    (app / ".hex").write_text("not a folder", encoding="utf-8")
    run = StubRun()
    assert handle_hook(text, workspace=workspace, run=run) == clean
    assert len(run.calls) == 1


def test_no_session_file_gives_no_message(workspace):
    session_app(workspace / "apps" / "churn", session=None)
    assert handle_stop(STOP_EVENT, workspace=workspace) == ""


def test_another_session_gives_no_message(workspace):
    session_app(workspace / "apps" / "churn", session=OTHER_SESSION)
    assert handle_stop(STOP_EVENT, workspace=workspace) == ""


def test_an_approved_app_of_this_session_gives_no_message(workspace):
    session_app(workspace / "apps" / "churn", status="APPROVED")
    assert handle_stop(STOP_EVENT, workspace=workspace) == ""


def test_an_unverified_app_of_this_session_is_named_with_the_audit_instruction(
    workspace,
):
    session_app(workspace / "apps" / "churn", status="UNVERIFIED")
    message = handle_stop(STOP_EVENT, workspace=workspace)
    assert message == format_stop_message([("apps/churn", "UNVERIFIED")])
    assert "run_audit with app apps/churn and mode full" in message


def test_a_rejected_app_of_this_session_is_named_with_the_fix_instruction(workspace):
    session_app(workspace / "apps" / "churn", status="REJECTED")
    message = handle_stop(STOP_EVENT, workspace=workspace)
    assert message == format_stop_message([("apps/churn", "REJECTED")])
    assert "tell the user plainly that apps/churn is REJECTED" in message


def test_an_app_without_a_verdict_is_named(workspace):
    session_app(workspace / "apps" / "churn")
    message = handle_stop(STOP_EVENT, workspace=workspace)
    assert message == format_stop_message([("apps/churn", None)])
    assert "has no verdict" in message


@pytest.mark.parametrize("content", ["{not json", "[]", "null", '{"status": 3}', ""])
def test_an_unusable_verdict_counts_as_no_verdict(workspace, content):
    app = session_app(workspace / "apps" / "churn")
    (app / "verdict.json").write_text(content, encoding="utf-8")
    assert handle_stop(STOP_EVENT, workspace=workspace) == format_stop_message(
        [("apps/churn", None)]
    )


def test_an_unknown_status_is_treated_like_unverified(workspace):
    session_app(workspace / "apps" / "churn", status="MAYBE")
    message = handle_stop(STOP_EVENT, workspace=workspace)
    assert "its verdict is UNVERIFIED (fast check only)" in message


def test_two_apps_are_listed_sorted_by_path(workspace):
    session_app(workspace / "apps" / "zeta", status="REJECTED")
    session_app(workspace / "apps" / "alpha")
    session_app(workspace / "apps" / "mid", status="APPROVED")
    lines = handle_stop(STOP_EVENT, workspace=workspace).splitlines()
    assert [line.split(":")[0] for line in lines[1:-1]] == [
        "- apps/alpha",
        "- apps/zeta",
    ]


def test_an_event_without_a_valid_session_id_gives_no_message(workspace):
    session_app(workspace / "apps" / "churn")
    for text in ("", "{}", "{not json", json.dumps({"session_id": "a b"})):
        assert handle_stop(text, workspace=workspace) == ""


def test_apps_in_hidden_and_environment_folders_are_not_looked_at(workspace):
    session_app(workspace / ".hidden" / "churn")
    session_app(workspace / "node_modules" / "churn")
    venv = workspace / "venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("", encoding="utf-8")
    session_app(venv / "churn")
    assert handle_stop(STOP_EVENT, workspace=workspace) == ""


def test_the_stop_check_names_an_app_at_the_workspace_root_with_a_dot(workspace):
    session_app(workspace)
    assert "- .: " in handle_stop(STOP_EVENT, workspace=workspace)


def test_the_stop_check_runs_nothing_and_writes_nothing(workspace):
    session_app(workspace / "apps" / "churn", status="UNVERIFIED")
    before = tree(workspace)
    handle_stop(STOP_EVENT, workspace=workspace)
    assert tree(workspace) == before


def test_the_message_lines_are_exactly_as_designed():
    message = format_stop_message(
        [("apps/churn", "UNVERIFIED"), ("apps/fraud", "REJECTED"), ("apps/new", None)]
    )
    assert message.splitlines() == [
        "Hexperiment: not approved yet.",
        "- apps/churn: changed in this session; its verdict is UNVERIFIED (fast "
        "check only). Run the full audit: the hexperiment MCP tool run_audit with "
        "app apps/churn and mode full (or: hexperiment run apps/churn).",
        "- apps/fraud: changed in this session; its last full audit is REJECTED. "
        "Fix the findings at their cause, or tell the user plainly that "
        "apps/fraud is REJECTED and why.",
        "- apps/new: changed in this session and has no verdict. Run the full "
        "audit: the hexperiment MCP tool run_audit with app apps/new and mode "
        "full (or: hexperiment run apps/new).",
        "Do not call the work done until each app above is APPROVED or reported "
        "as not approved.",
    ]


def test_importing_the_hook_module_loads_no_data_science_library():
    code = (
        "import sys, hexperiment.hook; "
        "bad = [m for m in ('pandas', 'numpy', 'sklearn') if m in sys.modules]; "
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_stop_check_loads_no_data_science_library(workspace):
    session_app(workspace / "apps" / "churn", status="UNVERIFIED")
    code = (
        "import sys; from pathlib import Path; from hexperiment.hook import "
        f"handle_stop; handle_stop({STOP_EVENT!r}, "
        f"workspace=Path({str(workspace)!r})); "
        "bad = [m for m in ('pandas', 'numpy', 'sklearn') if m in sys.modules]; "
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
