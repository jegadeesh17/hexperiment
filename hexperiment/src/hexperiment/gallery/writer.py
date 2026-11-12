"""Write the site: build beside the target, then swap it in (edge).

The previous gallery stays in place until the new one is complete (Requirement 1.5).
Steps W1 to W4 are small functions so that tests can stop between them.
"""

from __future__ import annotations

import os
import secrets
import shutil
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from hexperiment.errors import HexInputError
from hexperiment.gallery.summary import MARKER_FILENAME

RETRIES = 5
RETRY_PAUSE_SECONDS = 0.2


def write_site(files: Mapping[str, bytes], out_dir: Path) -> str:
    """Write ``files`` into ``out_dir``; return a warning text, or ``""``.

    A bad key is a bug and raises ``ValueError`` before anything is written.
    """
    check_keys(files)
    out_dir = out_dir.absolute()
    new, old = temporary_names(out_dir)
    try:
        stage_new(files, new)  # W1
        moved_aside = set_old_aside(out_dir, old)  # W2
    except BaseException:
        shutil.rmtree(new, ignore_errors=True)
        raise
    try:
        install_new(new, out_dir)  # W3
    except BaseException as error:
        _recover(error, new, old, out_dir, moved_aside)
        raise
    return remove_old(old) if moved_aside else ""  # W4


def check_keys(files: Mapping[str, bytes]) -> None:
    """Every key must be a relative POSIX path without ``..``, a drive or a ``/``."""
    for key in files:
        parts = key.split("/")
        bad = (
            not key
            or key.startswith("/")
            or "\\" in key
            or ":" in parts[0]
            or any(part in ("", ".", "..") for part in parts)
        )
        if bad:
            raise ValueError(f"the site has a file path that is not allowed: {key!r}")


def temporary_names(out_dir: Path) -> tuple[Path, Path]:
    """``NEW`` and ``OLD``: hidden siblings of ``out_dir`` with a random token."""
    while True:
        token = secrets.token_hex(3)
        new = out_dir.parent / f".{out_dir.name}.new-{token}"
        old = out_dir.parent / f".{out_dir.name}.old-{token}"
        if not new.exists() and not old.exists():
            return new, old


def stage_new(files: Mapping[str, bytes], new: Path) -> None:
    """W1: create ``NEW`` and write every file, sorted, as the bytes given."""
    new.mkdir()
    for key in sorted(files):
        target = new / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[key])


def set_old_aside(out_dir: Path, old: Path) -> bool:
    """W2: leave no ``out_dir``. Returns True when the old gallery moved to ``old``."""
    if not out_dir.exists():
        return False
    if not any(out_dir.iterdir()):
        _retry(lambda: os.rmdir(out_dir), out_dir)
        return False
    if not (out_dir / MARKER_FILENAME).is_file():
        raise HexInputError(
            f"{out_dir} exists, is not empty and was not built by hexperiment gallery "
            f"(it has no {MARKER_FILENAME} file).",
            "Choose an empty or new folder, or delete it yourself if you are sure.",
        )
    _retry(lambda: os.rename(out_dir, old), out_dir)
    return True


def install_new(new: Path, out_dir: Path) -> None:
    """W3: rename ``NEW`` to ``out_dir``, which does not exist now."""
    _retry(lambda: os.rename(new, out_dir), out_dir)


def remove_old(old: Path) -> str:
    """W4: delete ``OLD``. A failure is a warning, not an error."""
    try:
        shutil.rmtree(old)
    except OSError:
        pass
    if old.exists():
        return (
            f"hexperiment: warning: could not remove the old gallery folder {old}; "
            "delete it by hand."
        )
    return ""


def _retry(action: Callable[[], None], target: Path) -> None:
    """Retry a ``PermissionError`` (a Windows handle) before giving up."""
    for attempt in range(RETRIES + 1):
        try:
            action()
        except PermissionError as error:
            if attempt == RETRIES:
                raise HexInputError(
                    f"{target} could not be replaced ({error}).",
                    "Close the program that uses it and run again.",
                ) from error
            time.sleep(RETRY_PAUSE_SECONDS)
        else:
            return


def _recover(
    error: BaseException, new: Path, old: Path, out_dir: Path, moved_aside: bool
) -> None:
    """After a failed W3, put the previous state back (Requirement 1.5)."""
    if moved_aside:
        try:
            os.rename(old, out_dir)
        except OSError as restore_error:
            raise HexInputError(
                f"the new gallery could not be installed ({error}) and the previous "
                f"one could not be put back ({restore_error}).",
                f"Rename {old} (the previous gallery) or {new} (the new one) to "
                f"{out_dir} by hand.",
            ) from error
    shutil.rmtree(new, ignore_errors=True)
