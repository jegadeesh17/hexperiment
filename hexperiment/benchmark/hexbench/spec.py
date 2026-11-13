"""One description per case, two renderings (pure).

A script is a list of lines, each marked ``both``, ``leaky`` or ``clean``. The leaky
app and its clean twin are two renderings of the same list, so a line diff of the two
holds only the ``# LEAK:`` lines and their replacements.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

type Side = Literal["both", "leaky", "clean"]
type Variant = Literal["leaky", "clean"]

MARK = "# LEAK: "

# The slots of the train.py skeleton and of the hex.toml text.
SLOTS = (
    "imports",
    "features",
    "helpers",
    "after_load",
    "split",
    "after_split",
    "audit_test",
    "audit_score",
)
TOML_SLOTS = ("split_extra",)

_LINE_BREAK = re.compile("[\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]")


@dataclass(frozen=True, slots=True)
class Line:
    text: str  # one physical line with its indentation; no newline
    side: Side = "both"
    note: str = ""  # leaky lines only: what is wrong, the text after MARK

    def __post_init__(self) -> None:
        if self.side not in ("both", "leaky", "clean"):
            raise ValueError(f"side must be both, leaky or clean, not {self.side!r}")
        if _LINE_BREAK.search(self.text) or _LINE_BREAK.search(self.note):
            raise ValueError("a line and its note must not contain a line break")
        if self.side == "leaky" and not self.note.strip():
            raise ValueError(
                f"a leaky line needs a note that says what is wrong: {self.text!r}"
            )
        if self.side != "leaky" and MARK.strip() in self.text:
            raise ValueError(
                f"only a leaky line may contain {MARK.strip()!r}: {self.text!r}"
            )
        if self.side != "leaky" and self.note:
            raise ValueError(f"only a leaky line has a note: {self.text!r}")


def both(text: str) -> Line:
    return Line(text, "both")


def leak(text: str, note: str) -> Line:
    """A leaky line; ``text`` may be only spaces, which makes a comment line."""
    return Line(text, "leaky", note)


def clean(text: str) -> Line:
    return Line(text, "clean")


def _leaky_text(line: Line) -> str:
    if not line.text.strip():
        return f"{line.text}{MARK}{line.note}"
    return f"{line.text}  {MARK}{line.note}"


def render(lines: Sequence[Line], variant: Variant) -> str:
    """The text of one variant: every line ends with a newline; no lines gives ''."""
    if variant not in ("leaky", "clean"):
        raise ValueError(f"variant must be leaky or clean, not {variant!r}")
    out: list[str] = []
    for line in lines:
        if line.side == "both":
            out.append(line.text)
        elif line.side == variant == "leaky":
            out.append(_leaky_text(line))
        elif line.side == variant == "clean":
            out.append(line.text)
    return "".join(f"{text}\n" for text in out)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str  # "lr", "rf", "hgb"
    import_line: str
    expression: str  # the constructor call, with random_state=0


@dataclass(frozen=True, slots=True)
class DataSpec:
    generator: str  # name of a function in datagen
    params: Mapping[str, float | int | str | bool]
    feature_expr: str  # Python expression for BASE_FEATURES in the script
    entity_column: str | None = None  # L3.2 only; never part of the features
    time_column: str | None = None  # L3.1 only


@dataclass(frozen=True, slots=True)
class Mechanism:
    key: str  # "scaler-all-rows", used by --only and in file names
    leak_type: str  # "L1.2"
    mechanism: str  # one sentence, copied to results.json
    designed_to_catch: tuple[
        str, ...
    ]  # ghost names; () when no check is designed for it
    data: DataSpec
    threshold: float
    slots: Mapping[str, tuple[Line, ...]]  # train.py slots
    toml_slots: Mapping[str, tuple[Line, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = sorted(set(self.slots) - set(SLOTS))
        if unknown:
            raise ValueError(
                f"{self.key}: unknown train.py slot(s) {unknown}; known: {list(SLOTS)}"
            )
        unknown = sorted(set(self.toml_slots) - set(TOML_SLOTS))
        if unknown:
            raise ValueError(
                f"{self.key}: unknown hex.toml slot(s) {unknown}; "
                f"known: {list(TOML_SLOTS)}"
            )


@dataclass(frozen=True, slots=True)
class Case:
    case_id: str  # f"{mechanism.key}-{model.key}"
    mechanism: Mechanism
    model: ModelSpec
    data_seed: int  # 1000 + 10 * mechanism number + model number (0, 1, 2)
