"""Put the table of ``results.json`` into README.md.

Usage: ``update_readme.py [results.json]``. Run by hand after a complete run.
It prints only the path it wrote.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from hexbench.report import render_block, splice_block
from hexbench.runner import emit

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results.json"
DEFAULT_README = HERE / "README.md"
EXIT_FAILED = 2


class _Refusal(Exception):
    """README.md is left unchanged; the message says why and what to do."""


def _load_results(path: Path) -> dict[str, object]:
    try:
        results = json.loads(path.read_text(encoding="utf-8"))
    except OSError as problem:
        raise _Refusal(
            f"cannot read {path} ({problem}); run the benchmark first"
        ) from None
    except ValueError as problem:
        raise _Refusal(f"{path} is not valid JSON ({problem})") from None
    if not isinstance(results, dict):
        raise _Refusal(f"{path} is not a hexperiment-benchmark results object")
    if results.get("complete") is not True:
        raise _Refusal(
            f"{path} was written with --only (complete is not true); "
            "run the whole benchmark without --only and try again"
        )
    return results


def _update(results_path: Path, readme: Path) -> None:
    results = _load_results(results_path)
    try:
        block = render_block(results)
    except (KeyError, TypeError, AttributeError) as problem:
        raise _Refusal(
            f"{results_path} lacks a field the table needs ({problem!r})"
        ) from None
    try:
        text = readme.read_bytes().decode("utf-8")
    except OSError as problem:
        raise _Refusal(f"cannot read {readme} ({problem})") from None
    try:
        updated = splice_block(text, block)
    except ValueError as problem:
        raise _Refusal(str(problem)) from None
    readme.write_bytes(updated.encode("utf-8"))


def main(argv: Sequence[str] | None = None, *, readme: Path | None = None) -> int:
    """Splice the table into ``readme`` (default: README.md next to this script)."""
    parser = argparse.ArgumentParser(prog="update_readme.py", description=__doc__)
    parser.add_argument("results", nargs="?", type=Path, default=DEFAULT_RESULTS)
    try:
        args = parser.parse_args(argv)
    except SystemExit as stop:
        return int(stop.code) if isinstance(stop.code, int) else EXIT_FAILED
    target = DEFAULT_README if readme is None else readme
    try:
        _update(args.results, target)
    except _Refusal as refusal:
        emit(f"README.md not changed: {refusal}", error=True)
        return EXIT_FAILED
    emit(str(target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
