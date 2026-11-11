"""The one place that prints, shared by ``auditor`` and ``cli``.

Keeping it apart from both lets the command line print without importing the
in-script half of a run.
"""

from __future__ import annotations

import os
import sys
from typing import TextIO


def print_text(text: str, file: TextIO | None = None) -> None:
    """Print ``text``; a console that cannot encode it never raises.

    ``file`` defaults to the current ``sys.stdout``. This is the only caller of
    ``print`` in the package; ``cli.main`` and a standalone ``audit`` print through
    it (Requirement 16.4).
    """
    stream = sys.stdout if file is None else file
    try:
        print(text, file=stream)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        line_text = (text + "\n").replace("\n", os.linesep)
        data = line_text.encode(encoding, errors="replace")
        buffer = getattr(stream, "buffer", None)
        if buffer is None:
            stream.write(data.decode(encoding, errors="replace"))
        else:
            stream.flush()
            buffer.write(data)
        stream.flush()
