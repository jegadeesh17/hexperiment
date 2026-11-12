"""HTML that cannot forget to escape.

Every ``str`` given to :func:`el` is escaped. The only way to turn a ``str`` into
markup is :func:`literal`, which the architecture test allows only with a string
literal. The tag allowlist has no script, style, form or image tag, so those cannot
be produced at all (Requirements 4.1, 4.2).
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable
from dataclasses import dataclass

TAGS = frozenset(
    "html head meta title link body header nav main footer section div h1 h2 h3 "
    "p a ol ul li dl dt dd table caption thead tbody tfoot tr th td code em span "
    "time".split()
)
VOID_TAGS = frozenset({"meta", "link", "br"})
BLOCK_TAGS = frozenset(
    "html head body header nav main footer section div h1 h2 h3 p ol ul li dl dt "
    "dd table caption thead tbody tfoot tr th td title meta link".split()
)
REFUSED_ATTRIBUTES = frozenset({"style", "srcdoc"})
URL_ATTRIBUTES = frozenset({"href", "src"})

_ATTRIBUTE_NAME = re.compile(r"[a-z][a-z0-9-]*")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_UNSAFE_URL_CHARACTER = re.compile(r"[\x00-\x20\x7f\\]")
_UNSAFE_TEXT = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\ud800-\udfff]")
_CODE_SPAN = re.compile(r"`([^`\n]+)`")
_REPLACEMENT = "\ufffd"


@dataclass(frozen=True, slots=True)
class Markup:
    html: str  # already safe


type Child = str | Markup | Iterable[Child] | None
type Attr = str | bool | None


def esc(text: str) -> Markup:
    """Escape ``text`` for HTML text and attribute values.

    Characters that cannot occur in HTML text (C0 controls other than tab and line
    feed, DEL, lone surrogates) become U+FFFD first.
    """
    if not isinstance(text, str):
        raise TypeError(f"esc needs a str, not {type(text).__name__}")
    return Markup(html.escape(_UNSAFE_TEXT.sub(_REPLACEMENT, text), quote=True))


def literal(markup: str) -> Markup:
    """Trusted constant markup. Only call it with a string literal."""
    return Markup(markup)


def code(text: str) -> Markup:
    return el("code", text)


def prose(text: str) -> Markup:
    """Escaped text in which `backtick` spans become ``<code>`` elements."""
    escaped = esc(text).html
    return Markup(_CODE_SPAN.sub(r"<code>\1</code>", escaped))


def spaced(*parts: Child) -> Markup:
    """The parts rendered and separated by one space."""
    rendered = [_render(part) for part in parts if part is not None]
    return Markup(" ".join(rendered))


def el(tag: str, *children: Child, **attrs: Attr) -> Markup:
    """One element. Raises ``ValueError`` or ``TypeError`` for anything refused."""
    if tag not in TAGS:
        raise ValueError(f"the tag {tag!r} is not allowed in the gallery")
    opening = f"<{tag}{_attributes(attrs)}>"
    if tag in VOID_TAGS:
        if children:
            raise ValueError(f"the void tag {tag!r} cannot have children")
        return Markup(opening + "\n")
    inner = "".join(_render(child) for child in children)
    ending = "\n" if tag in BLOCK_TAGS else ""
    return Markup(f"{opening}{inner}</{tag}>{ending}")


def document(root: Markup) -> bytes:
    """The page as UTF-8 bytes: no byte order mark, ``\\n`` line endings."""
    text = "<!doctype html>\n" + root.html
    return text.replace("\r", "\n").encode("utf-8")


def _render(child: Child) -> str:
    if child is None:
        return ""
    if isinstance(child, Markup):
        return child.html
    if isinstance(child, str):
        return esc(child).html
    if isinstance(child, bytes | bool | int | float | dict):
        raise TypeError(
            f"a {type(child).__name__} cannot be a child; format numbers as text first"
        )
    if isinstance(child, Iterable):
        return "".join(_render(item) for item in child)
    raise TypeError(f"a {type(child).__name__} cannot be a child")


def _attributes(attrs: dict[str, Attr]) -> str:
    parts = []
    for key, value in attrs.items():
        name = key.removesuffix("_").replace("_", "-")
        _check_attribute_name(name)
        if value is None or value is False:
            continue
        if value is True:
            parts.append(f" {name}")
            continue
        if not isinstance(value, str):
            raise TypeError(
                f"attribute {name!r} needs text, not {type(value).__name__}"
            )
        if name in URL_ATTRIBUTES:
            _check_url(name, value)
        parts.append(f' {name}="{esc(value).html}"')
    return "".join(parts)


def _check_attribute_name(name: str) -> None:
    if not _ATTRIBUTE_NAME.fullmatch(name):
        raise ValueError(f"{name!r} is not a valid attribute name")
    if name.startswith("on") or name in REFUSED_ATTRIBUTES:
        raise ValueError(f"the attribute {name!r} is not allowed in the gallery")


def _check_url(name: str, value: str) -> None:
    if _UNSAFE_URL_CHARACTER.search(value) or value.startswith("//"):
        raise ValueError(f"{name}={value!r} is not a safe address")
    if _SCHEME.match(value) and not value.startswith(("http://", "https://")):
        raise ValueError(f"{name}={value!r} must be relative or start with http(s)://")
