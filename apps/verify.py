"""Run a script variant of an example app through ``hexperiment run`` in a scratch copy.

Dev tool, not part of the product. Running ``hexperiment run apps/<app>`` writes
``runs/`` and ``verdict.json`` into the app (Requirement 5.3), so verification copies
the app to a temporary folder first and leaves the real folder untouched.

From the repository root:

    .venv/Scripts/python apps/verify.py <hotel-cancellations|flight-delays> \
        [--variant NAME ...] [--fast]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

APPS_DIR = Path(__file__).resolve().parent
HOTELS = "hotel-cancellations"
FLIGHTS = "flight-delays"
VARIANTS: dict[str, tuple[str, ...]] = {
    HOTELS: ("haunted", "possession-fixed", "honest"),
    FLIGHTS: ("haunted", "split-fixed", "honest"),
}
COPY_IGNORE = (
    "runs",
    ".hex",
    "verdict.json",
    "haunted",
    "__pycache__",
    "hotels.csv",
    "flights.csv",
)
LEAK_SPLIT_MARKER = "# LEAK: random split"
FIXED_SPLIT = "train, test = hexperiment.split(df)"
POSSESSION_COLUMN = '"reservation_status"'
POSSESSION_MARKER = "# LEAK: The Possession"
RUN_TIMEOUT_SECONDS = 900  # a safety net around the whole command


@dataclass(frozen=True, slots=True)
class VariantResult:
    """What one scratch run produced."""

    variant: str
    exit_code: int
    status: str
    mode: str
    metric: float | None
    seconds: float
    ghosts: tuple[str, ...]
    warnings: tuple[str, ...]
    summary: str
    stderr: str


def split_fixed_source(source: str) -> str:
    """Return the haunted flights script with the random split replaced by ``split``.

    Exactly one line must end in ``# LEAK: random split``; any other count raises
    ``ValueError`` so a later edit of the haunted script cannot break the variant
    without notice.
    """
    lines = source.splitlines(keepends=True)
    marked = [
        i for i, line in enumerate(lines) if line.rstrip().endswith(LEAK_SPLIT_MARKER)
    ]
    if len(marked) != 1:
        raise ValueError(
            f"expected exactly one line ending in {LEAK_SPLIT_MARKER!r}, "
            f"found {len(marked)}."
        )
    index = marked[0]
    line = lines[index]
    indent = line[: len(line) - len(line.lstrip())]
    ending = line[len(line.rstrip("\r\n")) :]
    lines[index] = f"{indent}{FIXED_SPLIT}{ending}"
    return "".join(lines)


def possession_fixed_source(source: str) -> str:
    """Return the haunted hotels script without the ``reservation_status`` list item.

    Exactly one line must contain both ``"reservation_status"`` and
    ``# LEAK: The Possession``, and it must be a list item on its own line
    (the quoted name, a comma, the comment). That line is removed. Any other
    count, or a line with other code on it, raises ``ValueError`` so a later edit
    of the haunted script cannot break the variant without notice.
    """
    lines = source.splitlines(keepends=True)
    marked = [
        i
        for i, line in enumerate(lines)
        if POSSESSION_COLUMN in line and POSSESSION_MARKER in line
    ]
    if len(marked) != 1:
        raise ValueError(
            f"expected exactly one line with {POSSESSION_COLUMN} and "
            f"{POSSESSION_MARKER!r}, found {len(marked)}."
        )
    index = marked[0]
    if not lines[index].strip().startswith(f"{POSSESSION_COLUMN},"):
        raise ValueError(
            f"the line with {POSSESSION_MARKER!r} must be a list item on its own "
            f"line ({POSSESSION_COLUMN}, followed by the comment); found "
            f"{lines[index].strip()!r}."
        )
    del lines[index]
    return "".join(lines)


def variant_source(app_dir: Path, variant: str) -> str:
    """Return the text to write as ``train.py`` for a variant."""
    if variant == "honest":
        return (app_dir / "train.py").read_text(encoding="utf-8")
    haunted = (app_dir / "haunted" / "train.py").read_text(encoding="utf-8")
    if variant == "haunted":
        return haunted
    if variant == "split-fixed":
        return split_fixed_source(haunted)
    if variant == "possession-fixed":
        return possession_fixed_source(haunted)
    raise ValueError(f"unknown variant {variant!r}")


def console_script() -> str:
    """The ``hexperiment`` command of the virtual environment that runs this tool."""
    found = shutil.which("hexperiment", path=str(Path(sys.executable).parent))
    if found is None:
        raise SystemExit(
            "verify: the hexperiment command is not in the virtual environment of "
            f"{sys.executable}. Install it with: "
            '.venv/Scripts/python -m pip install -e "hexperiment[dev]"'
        )
    return found


def make_scratch(app_dir: Path, scratch_root: Path, variant: str) -> Path:
    """Copy the app without its history and source CSVs, and write the variant."""
    source = variant_source(app_dir, variant)
    scratch = scratch_root / app_dir.name
    shutil.copytree(app_dir, scratch, ignore=shutil.ignore_patterns(*COPY_IGNORE))
    (scratch / "train.py").write_text(source, encoding="utf-8", newline="\n")
    return scratch


def clean_environment() -> dict[str, str]:
    """The environment without any HEX_ variable of an outer run."""
    return {k: v for k, v in os.environ.items() if not k.upper().startswith("HEX_")}


def read_verdict(scratch: Path) -> dict[str, object] | None:
    """Return the scratch ``verdict.json``; None if absent or not a JSON object."""
    path = scratch / "verdict.json"
    if not path.is_file():
        return None
    try:
        verdict = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:  # json.JSONDecodeError and UnicodeDecodeError
        return None
    return verdict if isinstance(verdict, dict) else None


def describe_findings(
    verdict: dict[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (ghosts of error findings, "check: message" of warning findings)."""
    ghosts: list[str] = []
    warnings: list[str] = []
    for finding in verdict.get("findings", []):  # type: ignore[union-attr]
        severity = finding["severity"]
        label = finding["ghost"] or finding["check"]
        if severity == "error":
            ghosts.append(f"{label} @ {finding['location']}")
        elif severity == "warning":
            warnings.append(f"{finding['check']} ({label}) @ {finding['location']}")
    return tuple(ghosts), tuple(warnings)


def remove_scratch(folder: Path) -> None:
    """Delete the scratch folder; warn on stderr, naming it, if that fails."""
    try:
        shutil.rmtree(folder)
    except OSError as error:
        print(
            f"verify: warning: could not remove the scratch folder {folder} ({error}). "
            "Delete it by hand.",
            file=sys.stderr,
        )


def run_variant(app_dir: Path, variant: str, *, full: bool) -> VariantResult:
    """Run one variant in a scratch copy and return its result."""
    tmp = Path(tempfile.mkdtemp(prefix="hexverify-"))
    try:
        return run_in_scratch(app_dir, variant, tmp, full=full)
    finally:
        remove_scratch(tmp)


def run_in_scratch(
    app_dir: Path, variant: str, tmp: Path, *, full: bool
) -> VariantResult:
    """Copy the app into ``tmp``, run ``hexperiment run`` on it, collect the result."""
    command = [console_script(), "run", "", "--full" if full else "--fast"]
    scratch = make_scratch(app_dir, tmp, variant)
    command[2] = str(scratch)
    started = time.perf_counter()
    try:
        done = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=clean_environment(),
            timeout=RUN_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        seconds = time.perf_counter() - started
        return VariantResult(
            variant, -1, "TIMEOUT", "-", None, seconds, (), (), "",
            f"no result after {RUN_TIMEOUT_SECONDS} s",
        )
    seconds = time.perf_counter() - started
    verdict = read_verdict(scratch)
    if verdict is None:
        return VariantResult(
            variant, done.returncode, "NO VERDICT", "-", None, seconds, (), (),
            done.stdout.strip(), done.stderr.strip()[-1500:],
        )
    ghosts, warnings = describe_findings(verdict)
    metric = verdict["metric"]["value"]  # type: ignore[index]
    return VariantResult(
        variant, done.returncode, str(verdict["status"]), str(verdict["mode"]),
        None if metric is None else float(metric), seconds, ghosts, warnings,
        done.stdout.strip(), done.stderr.strip()[-1500:],
    )


def format_result(result: VariantResult) -> str:
    """One table row, then the ghosts, the warnings and the run summary."""
    metric = "-" if result.metric is None else f"{result.metric:.4f}"
    lines = [
        (
            f"variant={result.variant} exit={result.exit_code} status={result.status} "
            f"mode={result.mode} metric={metric} seconds={result.seconds:.1f}"
        ),
        "  ghosts:   " + ("; ".join(result.ghosts) or "none"),
        "  warnings: " + ("; ".join(result.warnings) or "none"),
    ]
    if result.summary:
        lines.append("  summary:")
        lines.extend("    " + line for line in result.summary.splitlines())
    if result.stderr:
        lines.append("  stderr tail:")
        lines.extend("    " + line for line in result.stderr.splitlines())
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verify.py",
        description="Run script variants of an example app in a scratch copy.",
    )
    parser.add_argument("app", choices=sorted(VARIANTS), help="app folder name")
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        help="variant to run (repeatable); default: all variants of the app",
    )
    parser.add_argument("--fast", action="store_true", help="skip the canary test")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the chosen variants; return 0 when every run produced a verdict."""
    args = build_parser().parse_args(argv)
    chosen = tuple(args.variants) if args.variants else VARIANTS[args.app]
    unknown = [v for v in chosen if v not in VARIANTS[args.app]]
    if unknown:
        print(
            f"verify: unknown variant {unknown} for {args.app}; "
            f"choose from {list(VARIANTS[args.app])}.",
            file=sys.stderr,
        )
        return 2
    app_dir = APPS_DIR / args.app
    failed = False
    for variant in chosen:
        result = run_variant(app_dir, variant, full=not args.fast)
        print(format_result(result), flush=True)
        if result.exit_code == 2 or result.status in ("NO VERDICT", "TIMEOUT"):
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
