"""Tests of ``handle_hook`` and ``format_hook_output`` (Requirements 1.2 to 1.6, 7.1).

A stub runner stands in for ``cli.run_app``, so no test here starts a training run.
"""

import json
import os
from pathlib import Path

import pytest

from hexperiment.hook import (
    FULL_CHECK_NOTICE,
    HOOK_RUN_CAP_SECONDS,
    HOOK_TIMEOUT_SECONDS,
    format_hook_output,
    handle_hook,
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


def test_the_app_name_in_the_notice_has_forward_slashes_and_keeps_spaces(workspace):
    app = make_app(workspace / "apps" / "my model")
    output = handle_hook(
        event(file_path=str(app / "train.py")), workspace=workspace, run=StubRun()
    )
    assert output.endswith("run: hexperiment run apps/my model")


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
