"""The leak mechanisms of the benchmark (pure data).

Each mechanism is one ``Mechanism`` value: the leaky and clean lines are copied from
the design, one physical line each. The ghost names are text; a test compares them with
the catalogue of the kit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from hexbench.spec import DataSpec, Mechanism, both, clean, leak

# The eight leakage types, word for word as on PMC10499856 (checked 2026-11-12).
LEAK_TYPE_NAMES: Mapping[str, str] = MappingProxyType(
    {
        "L1.1": "No test set",
        "L1.2": "Pre-processing on training and test set",
        "L1.3": "Feature selection on training and test set",
        "L1.4": "Duplicates in datasets",
        "L2": "Model uses features that are not legitimate",
        "L3.1": "Temporal leakage",
        "L3.2": "Nonindependence between training and test samples",
        "L3.3": "Sampling bias in test distribution",
    }
)

DOPPELGANGER = "The Doppelganger"
POSSESSION = "The Possession"
PEEKING_SPIRIT = "The Peeking Spirit"
PROPHET = "The Prophet"

_SP4 = "    "
_SP8 = "        "


def _columns(count: int) -> str:
    return f'[f"f{{i:02d}}" for i in range({count})]'


# 1 ------------------------------------------------------------------------------
NO_TEST_SET = Mechanism(
    key="no-test-set",
    leak_type="L1.1",
    mechanism="The model is scored on the rows it was trained on.",
    designed_to_catch=(DOPPELGANGER,),
    data=DataSpec("tabular", {"n_rows": 3000, "n_columns": 60}, _columns(60)),
    threshold=0.60,
    slots={
        "audit_test": (
            leak(
                f"{_SP8}test=train,",
                "the model is scored on its own training rows",
            ),
            clean(f"{_SP8}test=test,"),
        ),
        "audit_score": (
            leak(
                f"{_SP8}y_score=model.predict_proba(train[features])[:, 1],",
                "same rows",
            ),
            clean(f"{_SP8}y_score=model.predict_proba(test[features])[:, 1],"),
        ),
    },
)

# 2 ------------------------------------------------------------------------------
SCALER_ALL_ROWS = Mechanism(
    key="scaler-all-rows",
    leak_type="L1.2",
    mechanism="A standard scaler is fitted on all rows before the split.",
    designed_to_catch=(PEEKING_SPIRIT,),
    data=DataSpec(
        "tabular",
        {"n_rows": 3000, "n_columns": 20, "n_informative": 6, "heavy_tail": True},
        _columns(20),
    ),
    threshold=0.60,
    slots={
        "imports": (both("from sklearn.preprocessing import StandardScaler"),),
        "helpers": (
            both(""),
            both(""),
            both("def winsorise(frame, scaler):"),
            both(f"{_SP4}out = frame.copy()"),
            both(
                f"{_SP4}out[FEATURES] = "
                "scaler.transform(frame[FEATURES]).clip(-2.0, 2.0)"
            ),
            both(f"{_SP4}return out"),
        ),
        "after_load": (
            leak(
                f"{_SP4}scaler = StandardScaler().fit(df[FEATURES])",
                "fitted on every row, test rows included",
            ),
        ),
        "after_split": (
            clean(f"{_SP4}scaler = StandardScaler().fit(train[FEATURES])"),
            both(
                f"{_SP4}train, test = winsorise(train, scaler), winsorise(test, scaler)"
            ),
        ),
    },
)

# 3 ------------------------------------------------------------------------------
IMPUTER_ALL_ROWS = Mechanism(
    key="imputer-all-rows",
    leak_type="L1.2",
    mechanism="A mean imputer is fitted on all rows before the split.",
    designed_to_catch=(PEEKING_SPIRIT,),
    data=DataSpec(
        "tabular",
        {
            "n_rows": 3000,
            "n_columns": 20,
            "n_informative": 6,
            "heavy_tail": True,
            "missing": 0.25,
        },
        _columns(20),
    ),
    threshold=0.60,
    slots={
        "imports": (both("from sklearn.impute import SimpleImputer"),),
        "after_load": (
            leak(
                f"{_SP4}df[FEATURES] = "
                'SimpleImputer(strategy="mean").fit_transform(df[FEATURES])',
                "mean over every row",
            ),
        ),
        "after_split": (
            clean(
                f'{_SP4}imputer = SimpleImputer(strategy="mean").fit(train[FEATURES])'
            ),
            clean(f"{_SP4}train[FEATURES] = imputer.transform(train[FEATURES])"),
            clean(f"{_SP4}test[FEATURES] = imputer.transform(test[FEATURES])"),
        ),
    },
)

# 4 ------------------------------------------------------------------------------
OVERSAMPLE_BEFORE_SPLIT = Mechanism(
    key="oversample-before-split",
    leak_type="L1.2",
    mechanism="The rare class is randomly over-sampled before the split.",
    designed_to_catch=(DOPPELGANGER,),
    data=DataSpec(
        "tabular",
        {"n_rows": 4000, "n_columns": 30, "prevalence": 0.10, "shift": 0.25},
        _columns(30),
    ),
    threshold=0.60,
    slots={
        "imports": (both("import pandas as pd"),),
        "after_load": (
            leak(
                f"{_SP4}minority = df[df[TARGET] == 1]",
                "the rare class is copied before the split",
            ),
            leak(
                f"{_SP4}extra = minority.sample("
                "n=len(df) - 2 * len(minority), replace=True, random_state=0)",
                "the copies are drawn from the rare class of every row",
            ),
            leak(
                f"{_SP4}df = pd.concat([df, extra], ignore_index=True)",
                "copies get new row ids, so they split independently",
            ),
        ),
    },
)

# 5 ------------------------------------------------------------------------------
TARGET_ENCODING_ALL_ROWS = Mechanism(
    key="target-encoding-all-rows",
    leak_type="L1.2",
    mechanism=(
        "A category's mean label is computed from all rows and used as a feature."
    ),
    designed_to_catch=(PEEKING_SPIRIT,),
    data=DataSpec("high_cardinality", {}, _columns(12)),
    threshold=0.60,
    slots={
        "features": (both('FEATURES = [*BASE_FEATURES, "cat_te"]'),),
        "after_load": (
            leak(
                f'{_SP4}df["cat_te"] = df.groupby("cat")[TARGET].transform("mean")',
                "level mean of every row's label, test labels included",
            ),
        ),
        "after_split": (
            clean(f'{_SP4}sums = train.groupby("cat")[TARGET].agg(["sum", "count"])'),
            clean(f"{_SP4}prior = train[TARGET].mean()"),
            clean(
                f"{_SP4}own = "
                '(train["cat"].map(sums["sum"]) - train[TARGET]) / '
                '(train["cat"].map(sums["count"]) - 1)'
            ),
            clean(
                f'{_SP4}train["cat_te"] = own.fillna(prior)'
                "  # leave-one-out on training rows"
            ),
            clean(
                f'{_SP4}test["cat_te"] = '
                'test["cat"].map(sums["sum"] / sums["count"]).fillna(prior)'
            ),
        ),
    },
)

# 6 ------------------------------------------------------------------------------
UNIVARIATE_SELECTION_ALL_ROWS = Mechanism(
    key="univariate-selection-all-rows",
    leak_type="L1.3",
    mechanism=(
        "The 15 best columns are chosen with the labels of all rows before the split."
    ),
    designed_to_catch=(PEEKING_SPIRIT,),
    data=DataSpec(
        "tabular",
        {"n_rows": 2400, "n_columns": 500, "n_informative": 5, "shift": 0.08},
        _columns(500),
    ),
    threshold=0.52,
    slots={
        "imports": (
            both("import numpy as np"),
            both("from sklearn.feature_selection import SelectKBest, f_classif"),
        ),
        "after_load": (
            leak(
                f"{_SP4}selector = "
                "SelectKBest(f_classif, k=15).fit(df[FEATURES], df[TARGET])",
                "scores every column with every row's label",
            ),
        ),
        "after_split": (
            clean(
                f"{_SP4}selector = "
                "SelectKBest(f_classif, k=15).fit(train[FEATURES], train[TARGET])"
            ),
            both(f"{_SP4}features = list(np.array(FEATURES)[selector.get_support()])"),
        ),
    },
)

# 7 ------------------------------------------------------------------------------
DUPLICATED_ROWS = Mechanism(
    key="duplicated-rows",
    leak_type="L1.4",
    mechanism=(
        "Part of the rows appear twice in the source file, and the split is random."
    ),
    designed_to_catch=(DOPPELGANGER,),
    data=DataSpec(
        "duplicated",
        {"n_rows": 2500, "n_columns": 40, "label_flip": 0.10, "n_repeated": 1000},
        _columns(40),
    ),
    threshold=0.60,
    slots={
        "imports": (both("import pandas as pd"),),
        "after_split": (
            leak(
                _SP4,
                "copies of one row are in train and in test; nothing removes them",
            ),
            clean(f"{_SP4}key = [*features, TARGET]"),
            clean(
                f"{_SP4}seen = set(pd.util.hash_pandas_object(train[key], index=False))"
            ),
            clean(
                f"{_SP4}seen_in_test = "
                "pd.util.hash_pandas_object(test[key], index=False).isin(seen)"
            ),
            clean(f"{_SP4}test = test.loc[~seen_in_test.to_numpy()]"),
        ),
    },
)

# 8 ------------------------------------------------------------------------------
POST_OUTCOME_COPY = Mechanism(
    key="post-outcome-copy",
    leak_type="L2",
    mechanism="A column recorded after the outcome copies the label.",
    designed_to_catch=(POSSESSION,),
    data=DataSpec(
        "tabular",
        {
            "n_rows": 3000,
            "n_columns": 30,
            "extra_name": "refund_flag",
            "extra_kind": "copy",
        },
        _columns(30),
    ),
    threshold=0.60,
    slots={
        "features": (
            leak(
                'FEATURES = [*BASE_FEATURES, "refund_flag"]',
                "recorded after the outcome, equal to it",
            ),
            clean("FEATURES = list(BASE_FEATURES)"),
        ),
    },
)

_RISK_SCORE_FEATURES = (
    leak(
        'FEATURES = [*BASE_FEATURES, "risk_score"]',
        "a noisy copy of the label",
    ),
    clean("FEATURES = list(BASE_FEATURES)"),
)

# 9 ------------------------------------------------------------------------------
NOISY_PROXY = Mechanism(
    key="noisy-proxy",
    leak_type="L2",
    mechanism="A noisy measurement of the label is used as a feature.",
    designed_to_catch=(POSSESSION,),
    data=DataSpec(
        "tabular",
        {
            "n_rows": 3000,
            "n_columns": 30,
            "extra_name": "risk_score",
            "extra_kind": "proxy",
            "extra_sigma": 0.65,
        },
        _columns(30),
    ),
    threshold=0.60,
    slots={"features": _RISK_SCORE_FEATURES},
)

# 10 -----------------------------------------------------------------------------
WEAK_PROXY = Mechanism(
    key="weak-proxy",
    leak_type="L2",
    mechanism=(
        "A weaker proxy of the label (stand-alone AUC about 0.89) is used as a feature."
    ),
    designed_to_catch=(),
    data=DataSpec(
        "tabular",
        {
            "n_rows": 3000,
            "n_columns": 30,
            "extra_name": "risk_score",
            "extra_kind": "proxy",
            "extra_sigma": 1.15,
        },
        _columns(30),
    ),
    threshold=0.60,
    slots={"features": _RISK_SCORE_FEATURES},
)

_DRIFTING_FEATURES = '[*[f"f{i:02d}" for i in range(16)], "t_index"]'

# 11 -----------------------------------------------------------------------------
RANDOM_SPLIT_IN_SCRIPT = Mechanism(
    key="random-split-in-script",
    leak_type="L3.1",
    mechanism=(
        "Time-ordered data is split at random by the script "
        "instead of with `hexperiment.split`."
    ),
    designed_to_catch=(PROPHET,),
    data=DataSpec(
        "drifting", {"n_rows": 3000}, _DRIFTING_FEATURES, time_column="event_time"
    ),
    threshold=0.55,
    slots={
        "imports": (
            leak(
                "from sklearn.model_selection import train_test_split",
                "a random split of time-ordered data",
            ),
        ),
        "split": (
            leak(
                f"{_SP4}train, test = train_test_split("
                "df, test_size=0.2, random_state=0)",
                "later rows train the model, earlier rows test it",
            ),
            clean(f"{_SP4}train, test = hexperiment.split(df)"),
        ),
    },
    toml_slots={"split_extra": (both('time_column = "event_time"'),)},
)

# 12 -----------------------------------------------------------------------------
TIME_COLUMN_UNDECLARED = Mechanism(
    key="time-column-undeclared",
    leak_type="L3.1",
    mechanism=(
        "The time column is not declared in `hex.toml`, "
        "so `hexperiment.split` splits at random."
    ),
    designed_to_catch=(PROPHET,),
    data=DataSpec(
        "drifting", {"n_rows": 3000}, _DRIFTING_FEATURES, time_column="event_time"
    ),
    threshold=0.55,
    slots={},
    toml_slots={
        "split_extra": (
            leak(
                "",
                "time_column is not declared, so hexperiment.split splits at random",
            ),
            clean('time_column = "event_time"'),
        ),
    },
)

# 13 -----------------------------------------------------------------------------
ENTITY_LEAK = Mechanism(
    key="entity-leak",
    leak_type="L3.2",
    mechanism=(
        "Several rows of one entity, each differing by small noise, "
        "are split at random."
    ),
    designed_to_catch=(),
    data=DataSpec("entities", {}, _columns(100), entity_column="entity_id"),
    threshold=0.60,
    slots={
        "after_split": (
            leak(
                _SP4,
                "rows of one entity are split at random, "
                "so entities are in train and in test",
            ),
            clean(f'{_SP4}train_entities = set(train["entity_id"])'),
            clean(f'{_SP4}test = test.loc[~test["entity_id"].isin(train_entities)]'),
        ),
    },
)


@dataclass(frozen=True, slots=True)
class NotMeasured:
    leak_type: str
    leak_type_name: str
    reason: str


NOT_MEASURED: tuple[NotMeasured, ...] = (
    NotMeasured(
        leak_type="L3.3",
        leak_type_name=LEAK_TYPE_NAMES["L3.3"],
        reason=(
            "It depends on the population a claim is about, "
            "which neither the code nor the data shows."
        ),
    ),
)

MECHANISMS: tuple[Mechanism, ...] = (
    NO_TEST_SET,
    SCALER_ALL_ROWS,
    IMPUTER_ALL_ROWS,
    OVERSAMPLE_BEFORE_SPLIT,
    TARGET_ENCODING_ALL_ROWS,
    UNIVARIATE_SELECTION_ALL_ROWS,
    DUPLICATED_ROWS,
    POST_OUTCOME_COPY,
    NOISY_PROXY,
    WEAK_PROXY,
    RANDOM_SPLIT_IN_SCRIPT,
    TIME_COLUMN_UNDECLARED,
    ENTITY_LEAK,
)
