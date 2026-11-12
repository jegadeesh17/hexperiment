"""Link checking for a built site (pure).

``broken_links`` is the build's own safety net: a broken link is a bug in the
gallery, so ``build_site`` raises when it finds one (Requirement 4.3).
"""

from __future__ import annotations

import posixpath
from collections.abc import Mapping
from html.parser import HTMLParser

from hexperiment.gallery.pages import FONTS_URL

_URL_ATTRIBUTES = frozenset({"href", "src"})


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value is None:
                continue
            if name in _URL_ATTRIBUTES:
                self.links.append((name, value))
            elif name == "id":
                self.ids.add(value)


def _collect(html_text: str) -> _Collector:
    collector = _Collector()
    collector.feed(html_text)
    collector.close()
    return collector


def page_links(html_text: str) -> tuple[tuple[str, str], ...]:
    """``(attribute, value)`` of every ``href`` and ``src``, in document order."""
    return tuple(_collect(html_text).links)


def _is_external(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _problem(
    page: str,
    value: str,
    ids_by_page: Mapping[str, set[str]],
    files: Mapping[str, bytes],
    repo_url: str | None,
) -> str | None:
    """Why the link ``value`` on ``page`` is broken, or ``None`` when it is fine."""
    if _is_external(value):
        if value == FONTS_URL or (repo_url and value.startswith(repo_url)):
            return None
        return (
            "is an external address that is neither the fonts stylesheet nor --repo-url"
        )
    path, _, fragment = value.partition("#")
    if not path:
        if fragment and fragment in ids_by_page[page]:
            return None
        return "names an id that the page does not have"
    target = posixpath.normpath(posixpath.join(posixpath.dirname(page), path))
    if target == ".." or target.startswith("../") or target not in files:
        return "points to a file the build did not write"
    if fragment and fragment not in ids_by_page.get(target, set()):
        return f"names an id that {target} does not have"
    return None


def broken_links(files: Mapping[str, bytes], repo_url: str | None) -> tuple[str, ...]:
    """One line per broken link of the ``.html`` files, in path order."""
    pages = sorted(path for path in files if path.endswith(".html"))
    collected = {page: _collect(files[page].decode("utf-8")) for page in pages}
    ids_by_page = {page: found.ids for page, found in collected.items()}
    problems = []
    for page in pages:
        for attribute, value in collected[page].links:
            why = _problem(page, value, ids_by_page, files, repo_url)
            if why is not None:
                problems.append(f"{page}: {attribute}={value!r} {why}")
    return tuple(problems)
