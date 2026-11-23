"""The bootstrap that starts ``train.py`` and watches its reads of the data file.

``canary.run_train_script`` starts ``train.py`` through ``main``. Before the
script's first line, ``main`` installs an audit hook that counts the opens,
copies, links and renames of the app's data file. ``data.load`` marks its own
read, so what is left is a read made
outside ``hexperiment.load``: statistics computed from such a copy are not
scrambled in the canary execution, so the canary cannot check them
(Requirement 10.9). ``audit`` writes the count into the snapshot.

This module imports only the standard library, so the bootstrap does not load
pandas or scikit-learn before the script does. The hook is installed once per
process and never raises. Readers that open files without Python's ``open`` (native
readers such as pyarrow) are not seen.
"""

from __future__ import annotations

import os
import runpy
import sys
import threading
import tomllib
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

_CONFIG_FILENAME = "hex.toml"
_OPEN_EVENT = "open"
# Events whose first argument is a source path that is read, copied, linked or
# moved. On this Python ``os.replace`` raises ``os.rename``, and ``shutil.copy2``
# raises ``_winapi.CopyFile2`` on Windows (``shutil.copyfile`` elsewhere).
_SOURCE_EVENTS = frozenset(
    {
        _OPEN_EVENT,
        "shutil.copyfile",
        "_winapi.CopyFile2",
        "os.link",
        "os.symlink",
        "os.rename",
        "os.replace",
    }
)


class _ThreadState(threading.local):
    """The hook state of one thread; every thread starts from zero."""

    own_reads: int = 0  # nesting depth of ``own_read``
    inside_hook: bool = False


@dataclass(slots=True)
class _Watch:
    """What the audit hook counts. One instance lives for the whole process."""

    data_file: str  # resolved and case-normalised
    outside_reads: int = 0
    threads: _ThreadState = field(default_factory=_ThreadState)


_watch: _Watch | None = None


def normalize_path(path: str | os.PathLike[str] | bytes) -> str:
    """Resolve ``path`` against the working directory, as the hook compares it."""
    return os.path.normcase(os.path.realpath(os.fsdecode(path)))


def opens_data_file(path: object, data_file: str) -> bool:
    """Whether the source path of an open, copy, link or move is ``data_file``.

    ``data_file`` is already normalised (``normalize_path``). A ``str``, ``bytes``
    or path-like ``path`` is resolved against the working directory; an integer
    file descriptor, or anything else, is never the data file.
    """
    if isinstance(path, int) or not isinstance(path, str | bytes | os.PathLike):
        return False
    return normalize_path(path) == data_file


def data_reads_outside_load() -> int:
    """How many times this process opened the data file outside ``load``.

    Zero when no watch is installed (a script that was not started by the command).
    """
    return 0 if _watch is None else _watch.outside_reads


@contextmanager
def own_read() -> Iterator[None]:
    """Mark the opens inside the block as ``hexperiment.load`` reading its file."""
    watch = _watch
    if watch is None:
        yield
        return
    state = watch.threads
    state.own_reads += 1
    try:
        yield
    finally:
        state.own_reads -= 1


def install(data_file: str) -> None:
    """Start counting opens of ``data_file``; a second call changes nothing."""
    global _watch
    if _watch is not None:
        return
    watch = _Watch(normalize_path(data_file))
    _watch = watch
    sys.addaudithook(_make_hook(watch))


def main() -> None:
    """Run ``train.py`` as ``python train.py`` would, with the watch installed.

    The script is the first argument and is looked up in the working directory.
    It runs under the name it is given, so ``sys.argv`` is ``[name]`` and
    ``__file__`` is ``name`` (``runpy`` sets ``sys.argv[0]`` to the path it is
    given). ``sys.path[0]`` is the app folder. An exception is printed with the
    traceback of the script alone, then the exit code is 1, as it is for a plain
    run.

    Every standard-library module the watch needs is imported at the top of this
    module, and the bootstrap command removes the working directory from
    ``sys.path`` before it imports this module, so a file of the app that is named
    like a standard-library module (``tomllib.py``) cannot stop the watch from
    being installed.
    """
    app_dir = os.getcwd()
    name = sys.argv[1]
    data_file = _data_file(app_dir)
    if data_file is not None:
        install(data_file)
    sys.argv = [name]
    sys.path[:] = [app_dir, *(entry for entry in sys.path if entry != "")]
    try:
        runpy.run_path(name, run_name="__main__")
    except Exception as error:
        _print_script_traceback(error, name)
        sys.exit(1)


def _make_hook(watch: _Watch) -> Any:
    def hook(event: str, args: tuple[Any, ...]) -> None:
        if event not in _SOURCE_EVENTS:
            return
        state = watch.threads
        if state.inside_hook or state.own_reads:
            return
        state.inside_hook = True
        try:
            if args and opens_data_file(args[0], watch.data_file):
                watch.outside_reads += 1
        except Exception:
            pass  # a hook must never break the script
        finally:
            state.inside_hook = False

    return hook


def _data_file(app_dir: str) -> str | None:
    """The data file named by ``[data] path``, or ``None`` when it cannot be read.

    An unreadable ``hex.toml`` is not reported here; ``audit`` raises the error
    with its fix when the script reaches it.
    """
    try:
        with open(Path(app_dir) / _CONFIG_FILENAME, "rb") as handle:
            relative = tomllib.load(handle)["data"]["path"]
        if not isinstance(relative, str):
            return None
        return str(Path(app_dir) / relative)
    except Exception:
        return None


def _print_script_traceback(error: BaseException, name: str) -> None:
    """Print ``error`` starting at the first frame of the script ``name``."""
    tb: TracebackType | None = error.__traceback__
    start = tb
    while start is not None and start.tb_frame.f_code.co_filename != name:
        start = start.tb_next
    traceback.print_exception(type(error), error, start or tb)
