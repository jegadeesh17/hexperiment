"""One rule for line endings: ``\\r\\n`` and ``\\r`` become ``\\n``.

Every text or file read from an app goes through here first, so a checkout with CRLF
line endings gives the same pages as one with LF (Requirement 4.4). Nothing else is
changed: no BOM handling, no stripping of trailing spaces, and ``\\x0b``, ``\\x0c``,
``\\x85``, ``\\u2028`` and ``\\u2029`` stay as they are.
"""

from __future__ import annotations


def normalize_text(text: str) -> str:
    """``\\r\\n`` first, then every remaining ``\\r``, become ``\\n``.

    The order matters: the other order would turn one CRLF into two line breaks.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_bytes(data: bytes) -> bytes:
    """The same rule on bytes."""
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
