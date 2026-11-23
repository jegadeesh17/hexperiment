"""Read the apps, the stylesheet and the benchmark file (edge: only reads)."""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

from hexperiment.config import load_config
from hexperiment.errors import HexInputError
from hexperiment.gallery.benchmark import parse_benchmark
from hexperiment.gallery.newlines import normalize_bytes, normalize_text
from hexperiment.gallery.records import AppSource, Benchmark
from hexperiment.models import Status, Verdict
from hexperiment.runs import CARD_FILENAME, RUNS_DIRNAME, VERDICT_FILENAME, load_history
from hexperiment.workspace import is_app

SCRIPT_NAME = "train.py"
HAUNTED_DIRNAME = "haunted"
NOTE_FILENAME = "note.txt"
NOTE_MAX_CHARS = 300
STYLESHEET_PACKAGE = "hexperiment.gallery"

# The pattern of runs._RUN_ID_PATTERN (a test checks that the two agree). It is
# applied before a run id becomes part of a path.
RUN_ID_PATTERN = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}")


def read_app(app_dir: Path) -> AppSource:
    """Everything the gallery shows of one app. Opens nothing for writing."""
    if not is_app(app_dir):
        raise HexInputError(
            f"{app_dir} is not an app folder: it needs hex.toml and train.py.",
            "Pass the folders that hold them, for example apps/churn.",
        )
    config = load_config(app_dir)
    history = tuple(load_history(app_dir))
    if not history:
        raise HexInputError(
            f"{app_dir} has no recorded run (runs/ holds no verdict.json).",
            f"Run `hexperiment run {app_dir}` first, or leave this app out.",
        )
    last_json = _read_evidence(app_dir, history[-1], VERDICT_FILENAME)
    approved = [v for v in history if v.status is Status.APPROVED]
    card = _read_evidence(app_dir, approved[-1], CARD_FILENAME) if approved else None
    notes = _read_notes(app_dir, history)
    train_py, haunted_py = _read_scripts(app_dir)
    return AppSource(
        config=config,
        history=history,
        notes=notes,
        last_verdict_json=last_json,
        approved_card=card,
        train_py=train_py,
        haunted_py=haunted_py,
    )


def read_stylesheet() -> bytes:
    """The packaged stylesheet with LF line endings."""
    resource = resources.files(STYLESHEET_PACKAGE) / "assets" / "gallery.css"
    try:
        return normalize_bytes(resource.read_bytes())
    except (OSError, ModuleNotFoundError) as error:
        raise HexInputError(
            f"the stylesheet gallery.css is missing from the install ({error}).",
            "Reinstall the package: pip install --force-reinstall "
            '"hexperiment[mcp] @ git+https://github.com/jegadeesh17/hexperiment'
            '#subdirectory=hexperiment".',
        ) from None


def read_fonts() -> tuple[tuple[str, bytes], ...]:
    """The packaged font files and licences, sorted by name.

    ``.woff2`` bytes are returned as they are; text files get LF line endings.
    """
    try:
        folder = resources.files(STYLESHEET_PACKAGE) / "assets" / "fonts"
        entries = sorted(
            (item for item in folder.iterdir() if item.is_file()),
            key=lambda item: item.name,
        )
        fonts = tuple(
            (
                item.name,
                item.read_bytes()
                if item.name.endswith(".woff2")
                else normalize_bytes(item.read_bytes()),
            )
            for item in entries
        )
        if not any(name.endswith(".woff2") for name, _ in fonts):
            raise FileNotFoundError("no .woff2 file in assets/fonts")
    except (OSError, ModuleNotFoundError) as error:
        raise HexInputError(
            f"the fonts are missing from the install ({error}).",
            "Reinstall the package: pip install --force-reinstall "
            '"hexperiment[mcp] @ git+https://github.com/jegadeesh17/hexperiment'
            '#subdirectory=hexperiment".',
        ) from None
    return fonts


def read_benchmark(path: Path) -> Benchmark:
    """Read and validate a benchmark results file (a BOM is accepted)."""
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except OSError as error:
        raise HexInputError(
            f"the benchmark file {path} cannot be read ({error}).",
            "Pass the path of an existing results file.",
        ) from None
    except UnicodeDecodeError:
        raise HexInputError(
            f"the benchmark file {path} is not UTF-8 text.",
            "Pass the results.json written by the benchmark.",
        ) from None
    return parse_benchmark(text, source=str(path))


def _run_path(app_dir: Path, verdict: Verdict, filename: str) -> Path:
    """The path of a file in a run folder; the run id is checked first."""
    run_id = verdict.run_id
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise HexInputError(
            f"the run id {run_id!r} in {app_dir / RUNS_DIRNAME} is not of the form "
            "YYYYMMDD-HHMMSS-<6 hex chars>.",
            "Restore the run folder from version control; never edit verdict.json.",
        )
    return app_dir / RUNS_DIRNAME / run_id / filename


def _read_notes(app_dir: Path, history: tuple[Verdict, ...]) -> tuple[str | None, ...]:
    """The optional ``note.txt`` of every run, aligned with ``history``."""
    return tuple(
        _read_note(_run_path(app_dir, verdict, NOTE_FILENAME)) for verdict in history
    )


def _read_note(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise HexInputError(
            f"{path} cannot be read ({error}).", "Check the file and its permissions."
        ) from None
    fix = (
        f"Write one line of at most {NOTE_MAX_CHARS} characters as UTF-8, or delete "
        "the file."
    )
    try:
        text = data.decode("utf-8").strip()
    except UnicodeDecodeError:
        raise HexInputError(f"{path} is not valid UTF-8.", fix) from None
    if not text:
        raise HexInputError(f"{path} is empty.", fix)
    if "\n" in text or "\r" in text:
        raise HexInputError(f"{path} holds more than one line.", fix)
    if len(text) > NOTE_MAX_CHARS:
        raise HexInputError(
            f"{path} is {len(text)} characters long; the limit is {NOTE_MAX_CHARS}.",
            fix,
        )
    return text


def _read_evidence(app_dir: Path, verdict: Verdict, filename: str) -> bytes:
    path = _run_path(app_dir, verdict, filename)
    try:
        return normalize_bytes(path.read_bytes())
    except OSError:
        raise HexInputError(
            f"{path} is missing or cannot be read.",
            "Restore it from version control; the gallery does not write evidence.",
        ) from None


def _read_scripts(app_dir: Path) -> tuple[str | None, str | None]:
    haunted_path = app_dir / HAUNTED_DIRNAME / SCRIPT_NAME
    if not haunted_path.is_file():
        return None, None
    return (
        _read_script(app_dir / SCRIPT_NAME),
        _read_script(haunted_path),
    )


def _read_script(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise HexInputError(
            f"{path} cannot be read ({error}).", "Check the file and its permissions."
        ) from None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise HexInputError(
            f"{path} is not valid UTF-8.", "Save the script as UTF-8."
        ) from None
    return normalize_text(text)
