"""The three models of every mechanism (pure). Every one has a fixed random_state."""

from __future__ import annotations

from hexbench.spec import ModelSpec

LR = ModelSpec(
    key="lr",
    import_line="from sklearn.linear_model import LogisticRegression",
    expression="LogisticRegression(C=10.0, max_iter=2000, random_state=0)",
)
RF = ModelSpec(
    key="rf",
    import_line="from sklearn.ensemble import RandomForestClassifier",
    expression="RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=0)",
)
HGB = ModelSpec(
    key="hgb",
    import_line="from sklearn.ensemble import HistGradientBoostingClassifier",
    expression=(
        "HistGradientBoostingClassifier("
        "max_iter=100, early_stopping=False, random_state=0)"
    ),
)

# Model number k in the data seed 1000 + 10 * m + k is the position in this tuple.
MODELS: tuple[ModelSpec, ...] = (LR, RF, HGB)
