from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

import hexperiment

TARGET = "late"
CATEGORICAL = ["carrier", "origin", "dest"]
NUMERIC = [
    "month",
    "day",
    "sched_dep_time",
    "sched_arr_time",
    "distance",
]
# Removed carrier_day_late_rate: it was the carrier's same-day late rate computed over
# all rows, so it was built from the labels of test flights, which are not known when
# a flight is scheduled.
FEATURES = CATEGORICAL + NUMERIC


def drop_seen_test_rows(
    train: pd.DataFrame, test: pd.DataFrame, key: list[str]
) -> pd.DataFrame:
    """Return test without the rows whose key columns also occur in train.

    Only the overlap with the training rows is removed. Duplicates within the test
    rows stay, like duplicates within the training rows. Rows are
    compared by a hash of their key values, in which NaN equals NaN. The index labels
    are kept.
    """
    seen = set(pd.util.hash_pandas_object(train[key], index=False))
    keep = ~pd.util.hash_pandas_object(test[key], index=False).isin(seen).to_numpy()
    return test.loc[keep]


def build_model(categorical: list[str], numeric: list[str]) -> Pipeline:
    """Encode the categorical columns, then fit a gradient boosting classifier."""
    encode = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan,
                    min_frequency=50,
                ),
                categorical,
            )
        ],
        remainder="passthrough",
    )
    model = HistGradientBoostingClassifier(
        categorical_features=[True] * len(categorical) + [False] * len(numeric),
        max_iter=150,
        learning_rate=0.1,
        early_stopping=False,
        random_state=0,
    )
    return Pipeline([("encode", encode), ("model", model)])


def main() -> None:
    df = hexperiment.load()
    # Split by time with the kit: every training flight is earlier than every test flight.
    train, test = hexperiment.split(df)
    test = drop_seen_test_rows(train, test, FEATURES + [TARGET])
    model = build_model(CATEGORICAL, NUMERIC).fit(train[FEATURES], train[TARGET])
    hexperiment.audit(
        train=train,
        test=test,
        y_score=model.predict_proba(test[FEATURES])[:, 1],
        y_score_train=model.predict_proba(train[FEATURES])[:, 1],
        features=FEATURES,
        script=__file__,
    )


if __name__ == "__main__":
    main()
