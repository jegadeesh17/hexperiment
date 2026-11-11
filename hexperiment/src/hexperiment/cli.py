"""The ``hexperiment`` command: ``hexperiment run <app_dir> [--fast | --full]``.

``main`` and a standalone ``audit`` are the only functions of the package that
print (Requirement 16.4).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from hexperiment.auditor import print_text
from hexperiment.canary import ExecutionFailed, run_executions
from hexperiment.config import load_config
from hexperiment.errors import HexInputError
from hexperiment.models import Status, Verdict
from hexperiment.runs import clear_app_verdict, finalize_run, new_run_id
from hexperiment.verdict import format_summary

EXIT_APPROVED = 0
EXIT_REJECTED = 1
EXIT_RUN_FAILED = 2
EXIT_UNVERIFIED = 3

SCRIPT_FILENAME = "train.py"


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
    return parser


def exit_code_for(status: Status) -> int:
    """The exit code of a finished run: 0, 1 or 3 (Requirement 16.5)."""
    if status is Status.APPROVED:
        return EXIT_APPROVED
    if status is Status.REJECTED:
        return EXIT_REJECTED
    return EXIT_UNVERIFIED


def main(argv: Sequence[str] | None = None) -> int:
    """Run an app and return the exit code (Requirements 16.1 to 16.5).

    The app-level ``verdict.json`` is deleted before anything else can fail, so
    every run that ends with exit code 2 leaves none behind (Requirement 16.5).
    """
    args = build_parser().parse_args(argv)
    app_dir = Path(args.app_dir)
    if not app_dir.is_dir():
        return _fail(
            HexInputError(
                f"the app folder {app_dir} does not exist or is not a folder.",
                "Pass the folder that holds hex.toml and train.py.",
            )
        )
    app_dir = app_dir.resolve()
    try:
        clear_app_verdict(app_dir)
        verdict = _run(app_dir, full=args.full)
    except (HexInputError, ExecutionFailed, OSError) as error:
        return _fail(error)
    print_text(format_summary(verdict))
    return exit_code_for(verdict.status)


def _run(app_dir: Path, *, full: bool) -> Verdict:
    config = load_config(app_dir)
    if not (app_dir / SCRIPT_FILENAME).is_file():
        raise HexInputError(
            f"{app_dir / SCRIPT_FILENAME} does not exist.",
            f"Create {SCRIPT_FILENAME} in the app folder; it must end with a call "
            "to hexperiment.audit.",
        )
    run_id = new_run_id()
    report = run_executions(app_dir, config, run_id, full=full)
    return finalize_run(app_dir, config, report, run_id=run_id)


def _fail(error: Exception) -> int:
    print_text(f"hexperiment: {error}", sys.stderr)
    return EXIT_RUN_FAILED
