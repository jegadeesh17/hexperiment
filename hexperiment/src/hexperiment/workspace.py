"""The workspace: where apps are, which app owns a file, and what may be run.

Every function that accepts a path from outside (a hook event, a tool argument)
resolves it first and refuses what lies outside the workspace (Requirements 1.2,
4.2, 4.5). Containment compares resolved paths case-insensitively on Windows,
because a hook event carries ``c:\\...`` while the working directory is ``C:\\...``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Protocol

from hexperiment.canary import SCRIPT_NAME
from hexperiment.config import CONFIG_FILENAME, load_config
from hexperiment.errors import HexInputError
from hexperiment.models import RunOutcome

ENV_WORKSPACE = "HEXPERIMENT_WORKSPACE"
VENV_MARKER = "pyvenv.cfg"
SKIPPED_FOLDERS = frozenset({"node_modules", "__pycache__"})


@dataclass(frozen=True, slots=True)
class AppRef:
    name: str  # [app] name from hex.toml, else the folder name
    path: Path  # resolved folder
    relative: str  # path relative to the workspace, "/" separators, "." for itself
    problem: str | None  # why hex.toml could not be read, else None


class RunApp(Protocol):
    """The shape of ``cli.run_app``; callers receive it as an argument."""

    def __call__(
        self, app_dir: Path, *, full: bool, timeout_cap: float | None = None
    ) -> RunOutcome: ...


def resolve_workspace(
    environ: Mapping[str, str] | None = None, cwd: Path | None = None
) -> Path:
    """The workspace folder: ``HEXPERIMENT_WORKSPACE`` when set, else ``cwd``.

    An empty value counts as not set. A value that is not an existing folder
    raises ``HexInputError`` naming the variable (Requirement 4.2).
    """
    env = os.environ if environ is None else environ
    base = Path.cwd() if cwd is None else Path(cwd)
    value = env.get(ENV_WORKSPACE, "").strip()
    if not value:
        return base.resolve()
    folder = base / value
    try:
        resolved = folder.resolve()
        is_folder = resolved.is_dir()
    except (OSError, ValueError, RuntimeError):
        is_folder = False
    if not is_folder:
        raise HexInputError(
            f"{ENV_WORKSPACE} is set to {value!r}, which is not an existing folder.",
            f"Set {ENV_WORKSPACE} to the project folder, or unset it to use the "
            "current folder.",
        )
    return resolved


def is_app(folder: Path) -> bool:
    """A folder is an app when it holds both ``hex.toml`` and ``train.py``."""
    return (folder / CONFIG_FILENAME).is_file() and (folder / SCRIPT_NAME).is_file()


def relative_posix(path: Path, workspace: Path) -> str:
    """``path`` relative to ``workspace`` with ``/`` separators; ``.`` for itself."""
    root = workspace.resolve()
    target = path.resolve()
    if not _is_inside(target, root):
        raise HexInputError(
            f"{path} is not inside the workspace {root}.",
            "Use a path below the workspace folder.",
        )
    return target.relative_to(root).as_posix()


def find_app_for_file(file: Path, workspace: Path) -> Path | None:
    """The nearest app folder above ``file`` that lies inside ``workspace``.

    A relative ``file`` is taken relative to the workspace and links are
    followed. ``None`` when the resolved file is outside the workspace (or is the
    workspace itself) or no folder from the file's own up to the workspace is an
    app (Requirement 1.2). The walk never goes above the workspace.
    """
    root = workspace.resolve()
    try:
        resolved = (file if file.is_absolute() else root / file).resolve()
    except (OSError, ValueError, RuntimeError):
        return None
    if not _is_inside(resolved, root) or _same_path(resolved, root):
        return None
    for folder in resolved.parents:
        if is_app(folder):
            return folder
        if _same_path(folder, root):
            break
    return None


def resolve_inside(path: Path, workspace: Path, what: str) -> Path:
    """``path`` with links followed; ``HexInputError`` if it leaves the workspace.

    The path need not exist. ``what`` names the file in the message. A link inside
    the workspace that points outside it is refused (Requirements 4.2, 4.5).
    """
    root = workspace.resolve()
    try:
        resolved = path.resolve()
    except (OSError, ValueError, RuntimeError) as error:
        raise HexInputError(
            f"{what} {path} cannot be resolved ({error!r}).",
            "Check that the path and any link in it are valid.",
        ) from None
    if not _is_inside(resolved, root):
        raise HexInputError(
            f"{what} {path} resolves to {resolved}, which is outside the workspace "
            f"{root}.",
            "Replace the link with a regular file inside the workspace, or run the "
            "app again to write a new one.",
        )
    return resolved


def resolve_app(workspace: Path, app: str) -> Path:
    """The resolved folder of ``app``, or ``HexInputError`` saying why not.

    ``app`` is relative to the workspace (an absolute path is accepted). The path
    is resolved before it is compared with the workspace, so ``..`` segments and
    links that leave the workspace are refused (Requirements 4.2, 4.5).
    """
    if not isinstance(app, str) or not app.strip() or "\x00" in app:
        raise HexInputError(
            f"the app {app!r} is not a usable folder name.",
            "Pass the app folder relative to the workspace, for example apps/churn.",
        )
    root = workspace.resolve()
    try:
        folder = (root / app).resolve()
    except (OSError, ValueError, RuntimeError) as error:
        raise HexInputError(
            f"the app {app!r} cannot be resolved ({error}).",
            "Pass the app folder relative to the workspace, for example apps/churn.",
        ) from None
    if not _is_inside(folder, root):
        raise HexInputError(
            f"the app {app!r} is outside the workspace {root}.",
            "Pass an app folder below the workspace; apps elsewhere are refused.",
        )
    if not folder.is_dir():
        raise HexInputError(
            f"the app {app!r} is not a folder (no folder {folder}).",
            "Check the name with list_apps and pass an existing app folder.",
        )
    for required in (CONFIG_FILENAME, SCRIPT_NAME):
        if not (folder / required).is_file():
            raise HexInputError(
                f"the app {app!r} has no {required} in {folder}.",
                f"An app folder holds {CONFIG_FILENAME} and {SCRIPT_NAME}; "
                "create the missing file, for example with hexperiment init.",
            )
    return folder


def discover_apps(workspace: Path) -> tuple[AppRef, ...]:
    """Every app below ``workspace``, sorted by relative path (Requirement 4.3).

    Hidden folders, ``node_modules``, ``__pycache__``, virtual environments and
    links are not entered, and the walk does not descend into an app. A broken
    ``hex.toml`` gives the app a ``problem`` instead of hiding the other apps.
    """
    root = workspace.resolve()
    found: list[AppRef] = []
    for current, dirnames, _ in os.walk(root, followlinks=False):
        folder = Path(current)
        if is_app(folder):
            found.append(_app_ref(folder, root))
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if not _is_skipped(folder / d)]
    return tuple(sorted(found, key=lambda app: app.relative))


def _app_ref(folder: Path, root: Path) -> AppRef:
    relative = relative_posix(folder, root)
    try:
        name, problem = load_config(folder).app.name, None
    except HexInputError as error:
        name, problem = folder.name, str(error)
    return AppRef(name=name, path=folder, relative=relative, problem=problem)


def _is_skipped(folder: Path) -> bool:
    name = folder.name
    if name.startswith(".") or name in SKIPPED_FOLDERS:
        return True
    if folder.is_symlink() or os.path.isjunction(folder):
        return True
    return (folder / VENV_MARKER).is_file()


def _is_inside(path: Path, root: Path) -> bool:
    """``path`` equals ``root`` or lies below it; case-insensitive on Windows."""
    path_key = os.path.normcase(str(path))
    root_key = os.path.normcase(str(root))
    prefix = root_key if root_key.endswith(os.sep) else root_key + os.sep
    return path_key == root_key or path_key.startswith(prefix)


def _same_path(first: PurePath, second: PurePath) -> bool:
    return os.path.normcase(str(first)) == os.path.normcase(str(second))
