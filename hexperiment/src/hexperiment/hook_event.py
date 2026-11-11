"""Find the Python file paths named in a Kiro hook event (Requirements 1.2, 1.4).

The event is the JSON document that Kiro writes to a command hook's standard
input. Its schema is not fixed, so the parser looks for paths in two tiers and
never raises. It decides nothing about files: whether a path exists, lies in the
workspace or belongs to an app is the caller's job.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

# Key names (lower case, "_" and "-" removed) whose string values are paths.
PATH_KEYS: tuple[str, ...] = (
    "filepath",
    "path",
    "paths",
    "file",
    "files",
    "filename",
    "uri",
    "document",
    "target",
)
MAX_EVENT_NODES = 10_000
MAX_EVENT_DEPTH = 12  # the event itself is depth 0; deeper values are ignored
MAX_CANDIDATES = 20
MAX_PATH_LENGTH = 1024
PYTHON_SUFFIX = ".py"


def candidate_paths(event_text: str) -> tuple[str, ...]:
    """The Python paths named in ``event_text``, best candidate first.

    Tier 1 holds strings under a key from ``PATH_KEYS`` (or in a list under one).
    Tier 2 holds every other string. A string is dropped when it is longer than
    ``MAX_PATH_LENGTH`` or holds a control character, which rules out file
    contents and prompts. Quotes are stripped, a ``file://`` URI becomes a path,
    any other URI is dropped, and only names ending in ``.py`` are kept. Tier 2
    is used only when tier 1 gave no such name. Text that is not a JSON object
    or array gives ``()``.
    """
    try:
        event = json.loads(event_text)
    except (ValueError, TypeError, RecursionError):
        return ()
    if not isinstance(event, dict | list):
        return ()
    primary, secondary = _collect(event)
    return tuple(dict.fromkeys(primary or secondary))[:MAX_CANDIDATES]


def _collect(event: dict[str, Any] | list[Any]) -> tuple[list[str], list[str]]:
    """Walk ``event`` in document order without recursion, within the limits."""
    primary: list[str] = []
    secondary: list[str] = []
    stack: list[tuple[Any, int, bool]] = [(event, 0, False)]
    visited = 0
    while stack:
        value, depth, keyed = stack.pop()
        visited += 1
        if visited > MAX_EVENT_NODES:
            break
        if isinstance(value, str):
            path = _python_path(value)
            if path is not None:
                (primary if keyed else secondary).append(path)
        elif depth < MAX_EVENT_DEPTH and isinstance(value, dict):
            children = [(v, depth + 1, _is_path_key(k)) for k, v in value.items()]
            stack.extend(reversed(children))
        elif depth < MAX_EVENT_DEPTH and isinstance(value, list):
            stack.extend((item, depth + 1, keyed) for item in reversed(value))
    return primary, secondary


def _is_path_key(key: str) -> bool:
    return key.lower().replace("_", "").replace("-", "") in PATH_KEYS


def _has_control_character(text: str) -> bool:
    return any(ord(char) < 32 or 0x7F <= ord(char) <= 0x9F for char in text)


def _python_path(raw: str) -> str | None:
    """``raw`` as a path ending in ``.py``, or ``None`` when it is not one."""
    if len(raw) > MAX_PATH_LENGTH or _has_control_character(raw):
        return None
    text = _strip_quotes(raw.strip())
    if text[:5].lower() == "file:":
        text = _file_uri_to_path(text)
    if not text or "://" in text:
        return None
    return text if text.lower().endswith(PYTHON_SUFFIX) else None


def _strip_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1].strip()
    return text


def _file_uri_to_path(uri: str) -> str:
    """The local path of a ``file:`` URI, or ``""`` when it has none."""
    try:
        parsed = urlparse(uri)
        path = parsed.path
        if parsed.netloc and parsed.netloc.lower() != "localhost":
            path = f"//{parsed.netloc}{path}"
        return url2pathname(path)
    except (ValueError, OSError):
        return ""
