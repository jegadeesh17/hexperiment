"""The ``hexperiment`` command: ``hexperiment run <app_dir> [--fast | --full]``.

``main`` and a standalone ``audit`` are the only functions of the package that
print (Requirement 16.4). ``run_app`` returns the text of a run instead.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import traceback
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from hexperiment.canary import SCRIPT_NAME, ExecutionFailed, run_executions
from hexperiment.config import load_config
from hexperiment.console import print_text
from hexperiment.errors import HexInputError
from hexperiment.hook import handle_hook
from hexperiment.init import InitReport, format_report, init_app, init_kiro
from hexperiment.models import RunOutcome, Status, Verdict
from hexperiment.runs import app_lock, clear_app_verdict, finalize_run, new_run_id
from hexperiment.verdict import format_summary
from hexperiment.workspace import resolve_workspace

EXIT_APPROVED = 0
EXIT_REJECTED = 1
EXIT_RUN_FAILED = 2
EXIT_UNVERIFIED = 3
MAX_EVENT_BYTES = 1024 * 1024  # the most of a hook event that the command reads


def build_parser() -> argparse.ArgumentParser:
    """The argument parser; ``--fast`` and ``--full`` exclude each other (16.1)."""
    parser = argparse.ArgumentParser(
        prog="hexperiment",
        description="Check an ML app for data leakage before its result counts.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="run an app's train.py and write its verdict and model card"
    )
    run.add_argument("app_dir", help="folder with hex.toml and train.py")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument(
        "--fast",
        dest="full",
        action="store_false",
        help="run train.py once and skip the canary test",
    )
    mode.add_argument(
        "--full",
        dest="full",
        action="store_true",
        help="also run the canary test (the default)",
    )
    run.set_defaults(full=True)
    commands.add_parser(
        "hook",
        help="check the app that owns a file saved by an agent (reads a Kiro hook "
        "event from standard input; always exits 0)",
    )
    init = commands.add_parser(
        "init",
        help="create an app from the templates and/or add the Kiro files to this "
        "project",
    )
    init.add_argument(
        "app_dir", nargs="?", help="new app folder to create, for example apps/churn"
    )
    init.add_argument(
        "--kiro",
        action="store_true",
        help="write the hook, workflow, ghost hunter agent and steering file into "
        "./.kiro (existing files are kept)",
    )
    return parser


def exit_code_for(status: Status) -> int:
    """The exit code of a finished run: 0, 1 or 3 (Requirement 16.5)."""
    if status is Status.APPROVED:
        return EXIT_APPROVED
    if status is Status.REJECTED:
        return EXIT_REJECTED
    return EXIT_UNVERIFIED


def main(argv: Sequence[str] | None = None) -> int:
    """Run a command, print its text and return the exit code (16.1 to 16.5)."""
    args = build_parser().parse_args(argv)
    if args.command == "hook":
        return _hook_command()
    if args.command == "init":
        return _init_command(args.app_dir, kiro=args.kiro)
    outcome = run_app(Path(args.app_dir), full=args.full)
    # Requirement 16.5: the verdict is written, so a failure to print it must not
    # change the exit code.
    if outcome.stdout:
        with contextlib.suppress(Exception):
            print_text(outcome.stdout)
    if outcome.stderr:
        with contextlib.suppress(Exception):
            print_text(outcome.stderr, sys.stderr)
    return outcome.exit_code


def run_app(
    app_dir: Path, *, full: bool, timeout_cap: float | None = None
) -> RunOutcome:
    """Run an app and return its outcome without printing (Requirements 16.1 to 16.5).

    The app-level ``verdict.json`` is deleted before anything else can fail, so
    every run that ends with exit code 2 leaves none behind. Exit code 1 means
    ``REJECTED`` and nothing else: every failure, expected or not, returns 2
    (Requirement 16.5). ``KeyboardInterrupt`` and ``SystemExit`` are not caught.
    ``timeout_cap`` lowers ``[run] timeout_seconds`` for this call (Requirement 1.6).
    """
    try:
        verdict = _run(app_dir, full=full, timeout_cap=timeout_cap)
    except (HexInputError, ExecutionFailed, OSError) as error:
        return _failed(f"hexperiment: {error}")
    except Exception as error:
        return _failed(
            f"hexperiment: internal error: {type(error).__name__}: {error}\n"
            f"{traceback.format_exc().rstrip()}"
        )
    summary = ""
    with contextlib.suppress(Exception):
        summary = format_summary(verdict)
    return RunOutcome(exit_code_for(verdict.status), verdict.status, summary, "")


def _hook_command() -> int:
    """``hexperiment hook``: print the check for the saved file; always return 0.

    A hook that exits non-zero blocks the agent's turn, so every outcome is text:
    the result of the check, nothing for an event that names no app, or one line
    for a crash of the hook itself (Requirements 1.4, 1.5).
    """
    try:
        output = handle_hook(
            _read_event_text(), workspace=resolve_workspace(), run=run_app
        )
    except Exception as error:
        output = f"hexperiment hook failed: {type(error).__name__}: {error}"
    if output:
        with contextlib.suppress(Exception):
            print_text(output)
    return 0


def _init_command(app_dir: str | None, *, kiro: bool) -> int:
    """``hexperiment init [app_dir] [--kiro]``: print what was created and skipped.

    Exit 0 after a run, even if every file was skipped; exit 2 with
    ``hexperiment: <message>`` for invalid use or a file system error
    (Requirements 5.1 to 5.3).
    """
    try:
        if app_dir is None and not kiro:
            raise HexInputError(
                "init needs an app folder, --kiro, or both.",
                "Run `hexperiment init apps/<name>` to create an app, "
                "`hexperiment init --kiro` to add the Kiro files, "
                "or pass both.",
            )
        reports = []
        if app_dir is not None:
            reports.append(init_app(Path(app_dir)))
        if kiro:
            reports.append(init_kiro(Path.cwd()))
        report = InitReport(
            created=tuple(path for r in reports for path in r.created),
            skipped=tuple(path for r in reports for path in r.skipped),
            notes=tuple(note for r in reports for note in r.notes),
        )
        text = format_report(report, Path.cwd())
    except (HexInputError, OSError) as error:
        print_text(f"hexperiment: {error}", sys.stderr)
        return EXIT_RUN_FAILED
    print_text(text)
    return 0


def _read_event_text() -> str:
    """At most ``MAX_EVENT_BYTES`` of standard input; a terminal counts as empty."""
    stream = sys.stdin
    if stream is None or stream.isatty():
        return ""
    data = stream.buffer.read(MAX_EVENT_BYTES)
    return data.decode("utf-8-sig", errors="replace")


def _failed(message: str) -> RunOutcome:
    return RunOutcome(EXIT_RUN_FAILED, None, "", message)


def _run(app_dir: Path, *, full: bool, timeout_cap: float | None = None) -> Verdict:
    if not app_dir.is_dir():
        raise HexInputError(
            f"the app folder {app_dir} does not exist or is not a folder.",
            "Pass the folder that holds hex.toml and train.py.",
        )
    # The lock comes before anything is deleted or written, so a refused run leaves
    # the verdict and the run history as they were (Requirement 8.1).
    with app_lock(app_dir):
        return _run_locked(app_dir.resolve(), full=full, timeout_cap=timeout_cap)


def _run_locked(
    app_dir: Path, *, full: bool, timeout_cap: float | None = None
) -> Verdict:
    clear_app_verdict(app_dir)
    config = load_config(app_dir)
    if timeout_cap is not None:
        capped = min(config.run.timeout_seconds, timeout_cap)
        config = replace(config, run=replace(config.run, timeout_seconds=capped))
    if not (app_dir / SCRIPT_NAME).is_file():
        raise HexInputError(
            f"{app_dir / SCRIPT_NAME} does not exist.",
            f"Create {SCRIPT_NAME} in the app folder; it must end with a call "
            "to hexperiment.audit.",
        )
    run_id = new_run_id()
    report = run_executions(app_dir, config, run_id, full=full)
    return finalize_run(app_dir, config, report, run_id=run_id)
