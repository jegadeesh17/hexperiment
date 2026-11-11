"""``hexperiment init``: write the app templates and the Kiro files into a project.

Templates are package data below ``templates/``. They are copied as bytes, so line
endings are not translated, and a file that already exists is never opened for
writing (Requirements 5.1 to 5.3).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from hexperiment.errors import HexInputError
from hexperiment.workspace import relative_posix

TEMPLATE_PACKAGE = "hexperiment"
TEMPLATE_DIR = "templates"
APPS_FOLDER = "apps/"


@dataclass(frozen=True, slots=True)
class TemplateFile:
    source: str  # path below templates/, with "/" separators
    target: str  # path below the destination folder, with "/" separators


@dataclass(frozen=True, slots=True)
class InitReport:
    created: tuple[Path, ...]
    skipped: tuple[Path, ...]


APP_FILES: tuple[TemplateFile, ...] = (
    TemplateFile("app/hex.toml.tmpl", "hex.toml"),
    TemplateFile("app/train.py.tmpl", "train.py"),
)

KIRO_FILES: tuple[TemplateFile, ...] = (
    TemplateFile("kiro/hooks/hexperiment.json", "hooks/hexperiment.json"),
    TemplateFile(
        "kiro/workflows/exorcise.workflow.json", "workflows/exorcise.workflow.json"
    ),
    TemplateFile("kiro/agents/ghost-hunter.json", "agents/ghost-hunter.json"),
    TemplateFile("kiro/steering/ml-integrity.md", "steering/ml-integrity.md"),
)

KIRO_FOLDER = ".kiro"
_HOOK_TARGET = f"{KIRO_FOLDER}/hooks/hexperiment.json"
_APP_MARKER = APP_FILES[0].target


def init_app(app_dir: Path) -> InitReport:
    """Create ``app_dir`` if needed and add the ``APP_FILES`` that are absent."""
    if app_dir.exists() and not app_dir.is_dir():
        raise HexInputError(
            f"{app_dir} exists and is not a folder.",
            "Pass a folder name for the new app, for example apps/churn.",
        )
    return _write_all(APP_FILES, app_dir)


def init_kiro(project_dir: Path) -> InitReport:
    """Add the files of ``KIRO_FILES`` that are absent below ``project_dir/.kiro``."""
    return _write_all(KIRO_FILES, project_dir / KIRO_FOLDER)


def format_report(report: InitReport, base: Path) -> str:
    """One line per file, paths relative to ``base``, then the notes that apply.

    The notes follow from the files: a ``hex.toml`` in the report means an app was
    initialised in its folder, and the Kiro hook file means ``.kiro`` was written.
    """
    lines = [f"created  {_shown(path, base)}" for path in report.created]
    lines += [
        f"skipped  {_shown(path, base)} (already exists)" for path in report.skipped
    ]
    touched = (*report.created, *report.skipped)
    for path in touched:
        if path.name == _APP_MARKER:
            lines += _app_notes(_shown(path.parent, base))
    if any(_shown(path, base).endswith(_HOOK_TARGET) for path in touched):
        lines += _kiro_notes()
    return "\n".join(lines)


def _write_all(files: tuple[TemplateFile, ...], destination: Path) -> InitReport:
    contents = [(_read_template(item), destination / item.target) for item in files]
    for _, target in contents:
        if target.is_dir():
            raise HexInputError(
                f"{target} is a folder, so the file cannot be created there.",
                "Move or rename the folder, then run the command again.",
            )
    created: list[Path] = []
    skipped: list[Path] = []
    for data, target in contents:
        if _create(target, data):
            created.append(target)
        else:
            skipped.append(target)
    return InitReport(created=tuple(created), skipped=tuple(skipped))


def _read_template(item: TemplateFile) -> bytes:
    template = resources.files(TEMPLATE_PACKAGE).joinpath(
        TEMPLATE_DIR, *item.source.split("/")
    )
    try:
        return template.read_bytes()
    except OSError:
        raise HexInputError(
            f"the template {item.source} is missing from the installed hexperiment "
            "package.",
            'Reinstall the package: pip install --force-reinstall "hexperiment".',
        ) from None


def _create(target: Path, data: bytes) -> bool:
    """Write ``data`` to a new file; ``False`` when the file already exists.

    Exclusive creation fails on an existing file without opening it for writing,
    also when the file appears between the check and the write (Requirement 5.3).
    """
    if target.exists() or target.is_symlink():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        return False
    return True


def _shown(path: Path, base: Path) -> str:
    try:
        return relative_posix(path, base)
    except HexInputError:
        return path.resolve().as_posix()


def _app_notes(app: str) -> list[str]:
    notes = [
        "",
        "Next steps for the app:",
        f"  1. Put the CSV at the data.path in {app}/hex.toml and edit its CHANGE "
        "lines.",
        f"  2. Run: hexperiment run {app}",
        "  3. Add .hex/ to your .gitignore: it holds the canary test snapshots.",
    ]
    if not f"{app}/".startswith(APPS_FOLDER):
        notes.append(
            "  The ghost hunter agent can only edit Python files under apps/; "
            f"{app} is outside it."
        )
    return notes


def _kiro_notes() -> list[str]:
    if shutil.which("hexperiment") is not None:
        return []
    return [
        "",
        "Warning: `hexperiment` is not on your PATH. Kiro hooks start it from their "
        "own PATH, so start Kiro from a shell where the virtual environment is "
        "active, or install hexperiment with a tool such as pipx.",
    ]
