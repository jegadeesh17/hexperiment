"""The benchmark run: write each case, run both apps, summarise (edge).

``main`` is the whole program of ``run_benchmark.py``. The launcher, the command
finder and the clock are arguments so the tests never start a case.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from hexbench.cases import all_cases, write_case
from hexbench.execution import (
    COMMAND_NAME,
    Launch,
    RunRecord,
    find_command,
    launch,
    run_app,
)
from hexbench.mechanisms import MECHANISMS, NOT_MEASURED
from hexbench.spec import Case
from hexbench.summary import CaseRecord, build_results

EXIT_OK = 0
EXIT_FAILED = 2
EXIT_INTERRUPTED = 130

NO_EFFECT_TOLERANCE = 1e-4
SLOW_RUN_SECONDS = 30.0
_VERSIONED = ("numpy", "pandas", "scikit-learn")

_INSTALL_HINT = (
    f"the `{COMMAND_NAME}` command was not found; install it with "
    '`.venv/Scripts/python -m pip install -e "hexperiment[dev]"` and run the '
    "benchmark with that environment's Python"
)


def emit(text: str, *, error: bool = False) -> None:
    """Write one line to standard output (or standard error). ``print`` is not used."""
    stream = sys.stderr if error else sys.stdout
    stream.write(text + "\n")
    stream.flush()


# --- the command line ---------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_benchmark.py",
        description="Run the leakage benchmark through the hexperiment command.",
    )
    parser.add_argument("--out", required=True, help="where to write the results file")
    parser.add_argument("--only", help="run only the cases of this mechanism key")
    return parser


def _valid_keys() -> list[str]:
    return [m.key for m in MECHANISMS]


def _check_only(only: str | None) -> str | None:
    """A message when ``only`` is not a mechanism key."""
    if only is None or only in _valid_keys():
        return None
    return f"unknown mechanism {only!r}; valid keys are: {', '.join(_valid_keys())}"


def _check_out(out: Path) -> str | None:
    """A message when the results file cannot be written next to ``out``."""
    folder = out.parent
    if not folder.is_dir():
        return (
            f"the folder {str(folder)!r} does not exist; "
            "create it or choose another --out"
        )
    if out.is_dir():
        return f"--out {str(out)!r} is a folder; give a file name"
    try:
        probe = tempfile.NamedTemporaryFile(
            dir=folder, prefix=".hexbench-probe-", delete=False
        )
    except OSError as problem:
        return (
            f"cannot write to the folder {str(folder)!r} ({problem}); "
            "choose another --out"
        )
    probe.close()
    Path(probe.name).unlink(missing_ok=True)
    return None


# --- one case -------------------------------------------------------------------------


def _score_text(run: RunRecord) -> str:
    return "-" if run.score is None else f"{run.score:.4f}"


def _run_text(run: RunRecord) -> str:
    parts = [run.status, _score_text(run)]
    if run.ghosts:
        parts.append(", ".join(run.ghosts))
    return " ".join(parts)


def _no_effect(leaky: RunRecord, clean: RunRecord) -> bool:
    if leaky.score is None or clean.score is None:
        return False
    return abs(leaky.score - clean.score) < NO_EFFECT_TOLERANCE


def _case_line(index: int, total: int, record: CaseRecord) -> str:
    width = len(str(total))
    seconds = record.leaky.seconds + record.clean.seconds
    line = (
        f"[{index:>{width}}/{total}] {record.case_id:<34} "
        f"leaky {_run_text(record.leaky)}   clean {_run_text(record.clean)}   "
        f"{seconds:.1f} s"
    )
    if _no_effect(record.leaky, record.clean):
        line += "  NO EFFECT"
    return line


def _run_case(case: Case, root: Path, command: str, launcher: Launch) -> CaseRecord:
    leaky_dir, clean_dir = write_case(case, root)
    try:
        leaky = run_app(command, leaky_dir, launcher=launcher)
        clean = run_app(command, clean_dir, launcher=launcher)
    finally:
        shutil.rmtree(root / case.case_id, ignore_errors=True)
    return CaseRecord(
        case_id=case.case_id,
        mechanism=case.mechanism.key,
        model=case.model.key,
        data_seed=case.data_seed,
        leaky=leaky,
        clean=clean,
    )


def _select(only: str | None) -> tuple[Case, ...]:
    cases = all_cases()
    if only is None:
        return cases
    return tuple(c for c in cases if c.mechanism.key == only)


# --- the summary printed at the end ---------------------------------------------


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _environment(run_seconds: float) -> dict[str, object]:
    environment: dict[str, object] = {
        "hexperiment": _version("hexperiment"),
        "python": platform.python_version(),
    }
    for name in _VERSIONED:
        environment[name] = _version(name)
    environment["os"] = platform.platform()
    environment["cpu_count"] = os.cpu_count()
    environment["run_seconds"] = round(run_seconds, 1)
    return environment


def _ghost_counts(by_ghost: Mapping[str, int]) -> str:
    if not by_ghost:
        return "no ghost"
    return ", ".join(f"{name} {count}" for name, count in by_ghost.items())


def _mechanism_line(item: Mapping[str, object]) -> str:
    by_ghost: Mapping[str, int] = item["by_ghost"]  # type: ignore[assignment]
    return (
        f"{item['key']:<28} {item['leak_type']:<5} "
        f"rejected {item['rejected']} of {item['scripts']}  {_ghost_counts(by_ghost)}"
    )


def _problem_lines(records: Sequence[CaseRecord]) -> list[str]:
    errors: list[str] = []
    unchecked: list[str] = []
    no_effect: list[str] = []
    slow: list[str] = []
    for r in records:
        for name, run in (("leaky", r.leaky), ("clean", r.clean)):
            if run.status == "ERROR":
                errors.append(f"  {r.case_id} {name}: {run.message}")
            if run.seconds > SLOW_RUN_SECONDS:
                slow.append(f"  {r.case_id} {name}: {run.seconds:.1f} s")
        if _no_effect(r.leaky, r.clean):
            no_effect.append(f"  {r.case_id}")
        elif r.leaky.score is None or r.clean.score is None:
            unchecked.append(f"  {r.case_id}")
    return [
        *_section("ERROR runs", errors),
        *_section("NO EFFECT (leaky and clean scores are equal)", no_effect),
        *_section("Effect not checked (a score is missing)", unchecked),
        *_section(f"Runs over {SLOW_RUN_SECONDS:.0f} seconds", slow),
    ]


def _section(title: str, lines: Sequence[str]) -> list[str]:
    return [f"{title}: {'none' if not lines else ''}".rstrip(), *lines]


def _summary_lines(
    results: Mapping[str, object], records: Sequence[CaseRecord]
) -> list[str]:
    items: Sequence[Mapping[str, object]] = results["mechanisms"]  # type: ignore[assignment]
    clean: Mapping[str, object] = results["clean"]  # type: ignore[assignment]
    return [
        "",
        "Summary",
        *[_mechanism_line(item) for item in items],
        f"{'clean twins':<28} false alarms "
        f"{clean['false_alarms']} in {clean['scripts']}",
        *_problem_lines(records),
    ]


# --- writing the results --------------------------------------------------------


def _write_atomically(out: Path, results: Mapping[str, object]) -> None:
    """Write ``results`` to ``out`` through a temporary file in the same folder."""
    text = json.dumps(results, indent=2, ensure_ascii=False) + "\n"
    handle = tempfile.NamedTemporaryFile(
        dir=out.parent, prefix=".hexbench-", suffix=".tmp", delete=False
    )
    try:
        with handle:
            handle.write(text.encode("utf-8"))
        os.replace(handle.name, out)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


# --- the program ----------------------------------------------------------------


def _benchmark(
    cases: Sequence[Case], root: Path, command: str, launcher: Launch
) -> list[CaseRecord]:
    """Run every case, one at a time and in order, under ``root``."""
    records: list[CaseRecord] = []
    for index, case in enumerate(cases, start=1):
        record = _run_case(case, root, command, launcher)
        records.append(record)
        emit(_case_line(index, len(cases), record))
    return records


def _finish(
    records: Sequence[CaseRecord], only: str | None, out: Path, run_seconds: float
) -> None:
    results = build_results(
        records,
        mechanisms=MECHANISMS,
        not_measured=NOT_MEASURED,
        environment=_environment(run_seconds),
        created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        only=only,
    )
    for line in _summary_lines(results, records):
        emit(line)
    _write_atomically(out, results)
    emit(f"Results written to {out}")


def main(
    argv: Sequence[str] | None = None,
    *,
    launcher: Launch = launch,
    find: Callable[[], str | None] = find_command,
    clock: Callable[[], float] = time.perf_counter,
) -> int:
    """Run the benchmark; return the exit code (0, 2 or 130)."""
    try:
        args = _parser().parse_args(argv)
    except SystemExit as stop:
        return int(stop.code) if isinstance(stop.code, int) else EXIT_FAILED
    out = Path(args.out)
    for problem in (_check_only(args.only), _check_out(out)):
        if problem:
            emit(problem, error=True)
            return EXIT_FAILED
    command = find()
    if command is None:
        emit(_INSTALL_HINT, error=True)
        return EXIT_FAILED
    return _run_and_write(_select(args.only), args.only, command, out, launcher, clock)


def _run_and_write(
    cases: Sequence[Case],
    only: str | None,
    command: str,
    out: Path,
    launcher: Launch,
    clock: Callable[[], float],
) -> int:
    started = clock()
    folder = ""
    try:
        with tempfile.TemporaryDirectory(
            prefix="hexbench-", ignore_cleanup_errors=True
        ) as name:
            folder = name
            records = _benchmark(cases, Path(name), command, launcher)
        _finish(records, only, out, clock() - started)
    except KeyboardInterrupt:
        emit("interrupted; no results written", error=True)
        return EXIT_INTERRUPTED
    except Exception:
        emit(traceback.format_exc(), error=True)
        emit("the benchmark stopped on an error; no results written", error=True)
        return EXIT_FAILED
    finally:
        _report_leftover(folder)
    return EXIT_OK


def _report_leftover(folder: str) -> None:
    if folder and Path(folder).exists():
        emit(f"could not delete the temporary folder {folder}; delete it by hand")
