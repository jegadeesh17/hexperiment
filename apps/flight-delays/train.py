# Honest training script: will a 2013 New York flight arrive more than 15 minutes late,
# judged from its schedule alone? (Requirements 3.1, 3.4)
#
# hexperiment.split puts the latest 20% of flights in the test set. Every learned step
# sits inside one Pipeline fitted on training rows only.
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
# month and day stay on purpose: a scheduled flight repeats every day with the same
# columns, so without a date identical rows would sit in both train and test.
NUMERIC = ["month", "day", "sched_dep_time", "sched_arr_time", "distance"]
FEATURES = CATEGORICAL + NUMERIC
# Not used: flight and tailnum (identifiers), hour and minute (already inside
# sched_dep_time). fetch_data.py removed dep_time, dep_delay, arr_time and air_time:
# they exist only after departure.


def drop_seen_test_rows(
    train: pd.DataFrame, test: pd.DataFrame, key: list[str]
) -> pd.DataFrame:
    """Return test without the rows whose key columns already occur in train."""
    seen = pd.concat([train[key], test[key]]).duplicated().to_numpy()
    return test.loc[~seen[len(train) :]]


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
    train, test = hexperiment.split(df)  # the latest 20% by time_hour is the test set
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
