"""The 39 cases and the two renderings of each (pure; ``write_case`` writes files)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hexbench.datagen import generate
from hexbench.mechanisms import MECHANISMS
from hexbench.modelspecs import MODELS
from hexbench.spec import Case, Line, Mechanism, both, render

_SP4 = "    "
_SP8 = "        "

# Slot defaults of the train.py skeleton; a mechanism's slot replaces its default.
_DEFAULTS: dict[str, tuple[Line, ...]] = {
    "features": (both("FEATURES = list(BASE_FEATURES)"),),
    "split": (both(f"{_SP4}train, test = hexperiment.split(df)"),),
    "audit_test": (both(f"{_SP8}test=test,"),),
    "audit_score": (both(f"{_SP8}y_score=model.predict_proba(test[features])[:, 1],"),),
}


@dataclass(frozen=True, slots=True)
class AppFiles:
    train_py: str
    hex_toml: str


def all_cases() -> tuple[Case, ...]:
    """13 mechanisms times the three models, in file order."""
    return tuple(
        Case(
            case_id=f"{mechanism.key}-{model.key}",
            mechanism=mechanism,
            model=model,
            data_seed=1000 + 10 * number + k,
        )
        for number, mechanism in enumerate(MECHANISMS, start=1)
        for k, model in enumerate(MODELS)
    )


def _slot(mechanism: Mechanism, name: str) -> tuple[Line, ...]:
    return mechanism.slots.get(name, _DEFAULTS.get(name, ()))


def _train_lines(case: Case) -> list[Line]:
    m = case.mechanism

    def slot(name: str) -> list[Line]:
        return list(_slot(m, name))

    lines: list[Line] = [
        both("from __future__ import annotations"),
        both(""),
        *slot("imports"),
        both(case.model.import_line),
        both("import hexperiment"),
        both(""),
        both('TARGET = "y"'),
        both(f"BASE_FEATURES = {m.data.feature_expr}"),
        *slot("features"),
        *slot("helpers"),
        both(""),
        both(""),
        both("def main() -> None:"),
        both(f"{_SP4}df = hexperiment.load()"),
        *slot("after_load"),
        *slot("split"),
        both(f"{_SP4}features = list(FEATURES)"),
        *slot("after_split"),
        both(f"{_SP4}model = {case.model.expression}"),
        both(f"{_SP4}model.fit(train[features], train[TARGET])"),
        both(f"{_SP4}hexperiment.audit("),
        both(f"{_SP8}train=train,"),
        *slot("audit_test"),
        *slot("audit_score"),
        both(f"{_SP8}y_score_train=model.predict_proba(train[features])[:, 1],"),
        both(f"{_SP8}features=features,"),
        both(f"{_SP8}script=__file__,"),
        both(f"{_SP4})"),
        both(""),
        both(""),
        both('if __name__ == "__main__":'),
        both(f"{_SP4}main()"),
    ]
    return lines


def _toml_lines(case: Case) -> list[Line]:
    m = case.mechanism
    return [
        both("[app]"),
        both(f'name = "{case.case_id}"'),
        both('description = "Benchmark case. Generated; do not edit."'),
        both("[data]"),
        both('path = "data/dataset.csv"'),
        both("[target]"),
        both('column = "y"'),
        both("positive = 1"),
        both("[metric]"),
        both('name = "roc_auc"'),
        both(f"threshold = {m.threshold!r}"),
        both("[split]"),
        both("test_fraction = 0.2"),
        both("seed = 0"),
        *m.toml_slots.get("split_extra", ()),
        both("[run]"),
        both("timeout_seconds = 60"),
    ]


def render_app(case: Case, variant: Literal["leaky", "clean"]) -> AppFiles:
    """The ``train.py`` and ``hex.toml`` of one app of a case."""
    return AppFiles(
        train_py=render(_train_lines(case), variant),
        hex_toml=render(_toml_lines(case), variant),
    )


def _write_bytes(path: Path, text: str) -> None:
    """Write UTF-8 text with ``\\n`` line endings, whatever the platform."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def write_case(case: Case, root: Path) -> tuple[Path, Path]:
    """Write the leaky app and its clean twin under ``root/<case_id>/``.

    The dataset is generated once and each app gets its own copy. Returns the
    leaky and the clean app folders.
    """
    frame = generate(case.mechanism.data, case.data_seed)
    dataset = frame.to_csv(index=False, lineterminator="\n")
    folders: list[Path] = []
    for variant in ("leaky", "clean"):
        app_dir = Path(root) / case.case_id / variant
        files = render_app(case, variant)
        _write_bytes(app_dir / "train.py", files.train_py)
        _write_bytes(app_dir / "hex.toml", files.hex_toml)
        _write_bytes(app_dir / "data" / "dataset.csv", dataset)
        folders.append(app_dir)
    return folders[0], folders[1]
