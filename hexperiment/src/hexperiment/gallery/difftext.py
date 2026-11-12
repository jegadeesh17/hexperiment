"""A line diff of two scripts and its markup (pure)."""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from hexperiment.gallery.markup import Child, Markup, el
from hexperiment.gallery.newlines import normalize_text


@dataclass(frozen=True, slots=True)
class DiffLine:
    old_no: int | None
    new_no: int | None
    text: str


@dataclass(frozen=True, slots=True)
class Context:
    lines: tuple[DiffLine, ...]


@dataclass(frozen=True, slots=True)
class Change:
    removed: tuple[DiffLine, ...]
    added: tuple[DiffLine, ...]


@dataclass(frozen=True, slots=True)
class Gap:
    hidden: int


@dataclass(frozen=True, slots=True)
class ScriptDiff:
    blocks: tuple[Context | Change | Gap, ...]
    removed: int
    added: int


def _split_lines(text: str) -> list[str]:
    """Split on ``\\n`` only; one trailing newline adds no empty line."""
    lines = normalize_text(text).split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def diff_scripts(old: str, new: str, *, context: int = 3) -> ScriptDiff:
    old_lines = _split_lines(old)
    new_lines = _split_lines(new)
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    blocks: list[Context | Change | Gap] = []
    removed_total = added_total = 0
    position = 0  # old lines accounted for so far
    for group in matcher.get_grouped_opcodes(context):
        if group[0][1] > position:
            blocks.append(Gap(group[0][1] - position))
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                blocks.append(
                    Context(
                        tuple(
                            DiffLine(i + 1, j + 1, old_lines[i])
                            for i, j in zip(range(i1, i2), range(j1, j2), strict=True)
                        )
                    )
                )
                continue
            removed = tuple(DiffLine(i + 1, None, old_lines[i]) for i in range(i1, i2))
            added = tuple(DiffLine(None, j + 1, new_lines[j]) for j in range(j1, j2))
            removed_total += len(removed)
            added_total += len(added)
            blocks.append(Change(removed, added))
        position = group[-1][2]
    if blocks and len(old_lines) > position:
        blocks.append(Gap(len(old_lines) - position))
    return ScriptDiff(tuple(blocks), removed_total, added_total)


def _line(kind: str, sign: str, numbers: tuple[int | None, ...], text: str) -> Markup:
    classes = "line" if kind == "context" else f"line line-{kind}"
    cells: list[Child] = [
        el("span", str(number), class_="no") for number in numbers if number is not None
    ]
    cells.append(el("span", sign, class_="sign"))
    cells.append(el("code", text))
    return el("div", cells, class_=classes)


def _gap_text(hidden: int) -> str:
    return f"{hidden} unchanged line" + ("" if hidden == 1 else "s")


def diff_markup(diff: ScriptDiff, *, old_name: str, new_name: str) -> Markup:
    blocks: list[Child] = []
    for block in diff.blocks:
        if isinstance(block, Gap):
            blocks.append(el("div", _gap_text(block.hidden), class_="gap"))
        elif isinstance(block, Context):
            lines = [
                _line("context", " ", (x.old_no, x.new_no), x.text)
                for x in block.lines
            ]
            blocks.append(el("div", lines, class_="context"))
        else:
            removed = [
                _line("removed", "-", (x.old_no,), x.text) for x in block.removed
            ]
            added = [_line("added", "+", (x.new_no,), x.text) for x in block.added]
            blocks.append(
                el(
                    "div",
                    el("div", removed, class_="removed"),
                    el("div", added, class_="added"),
                    class_="change",
                )
            )
    return el(
        "div",
        blocks,
        class_="diff",
        role="region",
        aria_label=f"Code diff: {old_name} against {new_name}",
        tabindex="0",
    )
