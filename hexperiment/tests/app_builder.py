"""Builds the small apps that the end-to-end scenarios run (Requirement 17.3).

An app is a ``hex.toml``, the shared 600-row CSV, and the scripts rendered from
``tests/fixtures/scripts/<scenario>/*.py.tmpl``. The templates carry a ``.tmpl``
suffix so that ruff and pytest do not collect them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent / "fixtures" / "scripts"
TEMPLATE_SUFFIX = ".tmpl"

N_ROWS = 600
DATA_SEED = 20251101
MISSING_FRACTION = 0.1
CATEGORY_EFFECT = {"a": -0.36, "b": 0.0, "c": 0.42}
COEFFICIENT_X1 = 0.5
COEFFICIENT_X2 = 0.3
THRESHOLD = 0.62

HEX_TOML = """\
[app]
name = "{scenario}"
description = "Scenario {scenario} of the canary tests."

[data]
path = "data.csv"

[target]
column = "target"

[metric]
name = "roc_auc"
threshold = {threshold}

[run]
timeout_seconds = {timeout_seconds}
"""


def shared_frame(n_rows: int = N_ROWS, seed: int = DATA_SEED) -> pd.DataFrame:
    """The CSV every scenario reads.

    ``x1`` is continuous, ``x2`` is continuous with about 10% missing values,
    ``category`` has three values with different target rates, and ``target`` is
    0 or 1. The two numeric columns have many distinct test values, which the
    tilted resample of Requirement 1.5 needs. The signal is moderate, so a model
    scores a ROC AUC between ``THRESHOLD`` and the "too good" limit
    ``THRESHOLD`` plus the default margin of 0.15.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n_rows)
    x2 = rng.normal(size=n_rows)
    category = rng.choice(list(CATEGORY_EFFECT), size=n_rows)
    effect = np.array([CATEGORY_EFFECT[c] for c in category])
    logit = COEFFICIENT_X1 * x1 + COEFFICIENT_X2 * x2 + effect
    target = (rng.random(n_rows) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    x2[rng.random(n_rows) < MISSING_FRACTION] = np.nan
    return pd.DataFrame({"x1": x1, "x2": x2, "category": category, "target": target})


def build_app(
    root: Path,
    scenario: str,
    *,
    threshold: float = THRESHOLD,
    timeout_seconds: float = 120,
) -> Path:
    """Write the app of ``scenario`` into ``root/<scenario>`` and return its folder."""
    templates = SCRIPTS_DIR / scenario
    sources = sorted(templates.glob(f"*.py{TEMPLATE_SUFFIX}"))
    if not sources:
        raise FileNotFoundError(f"no script templates for scenario {scenario!r}")
    app = root / scenario
    app.mkdir(parents=True)
    toml = HEX_TOML.format(
        scenario=scenario, threshold=threshold, timeout_seconds=timeout_seconds
    )
    (app / "hex.toml").write_text(toml, encoding="utf-8")
    shared_frame().to_csv(app / "data.csv", index=False)
    for source in sources:
        text = source.read_text(encoding="utf-8")
        (app / source.name.removesuffix(TEMPLATE_SUFFIX)).write_text(
            text, encoding="utf-8", newline="\n"
        )
    return app


def place_stale_verdict(app: Path) -> Path:
    """Put an ``APPROVED`` ``verdict.json`` in the app, as an earlier run would."""
    path = app / "verdict.json"
    path.write_text('{"status": "APPROVED"}\n', encoding="utf-8")
    return path
