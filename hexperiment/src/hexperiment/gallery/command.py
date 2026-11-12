"""The ``hexperiment gallery`` command: read, build in memory, then write (edge)."""

from __future__ import annotations

import os
import traceback
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlsplit

from hexperiment.errors import HexInputError
from hexperiment.gallery.records import GalleryInputs
from hexperiment.gallery.site import build_site
from hexperiment.gallery.sources import (
    read_app,
    read_benchmark,
    read_stylesheet,
)
from hexperiment.gallery.summary import MARKER_FILENAME, assign_slugs
from hexperiment.gallery.writer import write_site
from hexperiment.models import RunOutcome

EXIT_REFUSED = 2
FORBIDDEN_URL_CHARACTERS = frozenset('"<>')


def build_gallery(
    app_dirs: Sequence[Path],
    out_dir: Path,
    *,
    repo_url: str | None,
    benchmark: Path | None,
) -> RunOutcome:
    """Build the gallery and return its outcome without printing.

    The first problem ends the build with exit code 2 and writes nothing. Only
    ``KeyboardInterrupt`` and ``SystemExit`` are not caught.
    """
    try:
        warning = _build(app_dirs, out_dir, repo_url, benchmark)
    except (HexInputError, OSError) as error:
        return _failed(f"hexperiment: {error}")
    except Exception as error:
        return _failed(
            f"hexperiment: internal error: {type(error).__name__}: {error}\n"
            f"{traceback.format_exc().rstrip()}"
        )
    index = str(out_dir.resolve() / "index.html")
    return RunOutcome(0, None, index, warning)


def validate_repo_url(url: str) -> str:
    """Return ``url`` unchanged if it is an http(s) address with a host (1.8)."""
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        host = None
        parts = None
    clean = not any(
        c.isspace() or ord(c) < 32 or ord(c) == 127 or c in FORBIDDEN_URL_CHARACTERS
        for c in url
    )
    if parts is None or parts.scheme not in ("http", "https") or not host or not clean:
        raise HexInputError(
            f"--repo-url {url!r} is not an http or https URL.",
            "Pass a full address such as https://github.com/owner/repo.",
        )
    return url


def check_out_dir(out_dir: Path, app_dirs: Sequence[Path]) -> None:
    """Refuse an ``OUT_DIR`` the build must not write into (Requirement 1.4)."""
    _check_kind(out_dir)
    resolved = out_dir.resolve()
    for app_dir in app_dirs:
        app = app_dir.resolve()
        if resolved == app or resolved.is_relative_to(app):
            raise HexInputError(
                f"--out {out_dir} is inside the app folder {app_dir}.",
                "Choose a folder outside the apps, for example site.",
            )
        if app.is_relative_to(resolved):
            raise HexInputError(
                f"--out {out_dir} holds the app folder {app_dir}.",
                "Choose a folder outside the apps, for example site.",
            )
    if out_dir.exists() and any(out_dir.iterdir()):
        if not (out_dir / MARKER_FILENAME).is_file():
            raise HexInputError(
                f"{out_dir} exists, is not empty and was not built by hexperiment "
                f"gallery (it has no {MARKER_FILENAME} file).",
                "Choose an empty or new folder, or delete it yourself if you are sure.",
            )


def _check_kind(out_dir: Path) -> None:
    if out_dir.is_symlink() or os.path.isjunction(out_dir):
        raise HexInputError(
            f"--out {out_dir} is a link.",
            "Choose a real folder, for example site.",
        )
    if out_dir.exists() and not out_dir.is_dir():
        raise HexInputError(
            f"--out {out_dir} exists and is a file, not a folder.",
            "Choose a folder that is empty or does not exist yet.",
        )
    if not out_dir.absolute().parent.is_dir():
        raise HexInputError(
            f"the parent folder of --out {out_dir} does not exist.",
            "Create the parent folder first, or choose another folder.",
        )


def _build(
    app_dirs: Sequence[Path],
    out_dir: Path,
    repo_url: str | None,
    benchmark_path: Path | None,
) -> str:
    url = None if repo_url is None else validate_repo_url(repo_url)
    sources = tuple(read_app(app_dir) for app_dir in app_dirs)
    assign_slugs(
        [
            (str(app_dir), source.config.app.name)
            for app_dir, source in zip(app_dirs, sources, strict=True)
        ]
    )
    benchmark = None if benchmark_path is None else read_benchmark(benchmark_path)
    check_out_dir(out_dir, app_dirs)
    stylesheet = read_stylesheet()
    files = build_site(
        GalleryInputs(
            apps=sources, benchmark=benchmark, repo_url=url, stylesheet=stylesheet
        )
    )
    return write_site(files, out_dir)


def _failed(message: str) -> RunOutcome:
    return RunOutcome(EXIT_REFUSED, None, "", message)
