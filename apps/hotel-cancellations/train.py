# Honest training script: will a hotel booking be cancelled? (Requirements 2.1, 2.4)
#
# Every learned step sits inside one Pipeline that is fitted on training rows only.
# Overlap with the training rows is removed from the test rows after the split.
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

import hexperiment

TARGET = "is_canceled"
CATEGORICAL = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "agent",
    "customer_type",
]
NUMERIC = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]
FEATURES = CATEGORICAL + NUMERIC
# Not used. reservation_status and reservation_status_date are filled in when the
# booking ends (cancelled, checked out, no-show), so they contain the answer.
# assigned_room_type is assigned at check-in or after a change, so it is not known
# when the cancellation happens. company is 94% missing.


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
    df["agent"] = df["agent"].fillna(-1).astype(int).astype(str)  # an id, not a quantity
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
