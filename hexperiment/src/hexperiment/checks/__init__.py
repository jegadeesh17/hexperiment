"""The data checks: one module per check, each a pure function.

``run_data_checks`` is the only place that lists the checks and their order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hexperiment.checks.metric import check_metric, compute_metric
from hexperiment.checks.row_overlap import check_row_overlap
from hexperiment.checks.split_order import check_split_order
from hexperiment.checks.target_proxy import check_target_proxy
from hexperiment.checks.temporal import check_temporal
from hexperiment.config import HexConfig
from hexperiment.data import ValidatedInputs, parse_times
from hexperiment.models import DataAuditResult, DataSummary

_RATE_DECIMALS = 6


def run_data_checks(
    train: pd.DataFrame,
    test: pd.DataFrame,
    v: ValidatedInputs,
    *,
    config: HexConfig,
    script_source: str | None,
    script_name: str,
) -> DataAuditResult:
    """Run the data checks and the metric gate on validated inputs.

    The checks run in a fixed order: ``row_overlap``, ``target_proxy``,
    ``temporal``, ``split_order``, then the metric (Requirement 2.3). The metric
    result and its "too good" warning travel next to the checks. Nothing is
    modified. ``script_source`` is the text of the training script, or ``None``
    when it could not be read.
    """
    target = config.target.column
    checks = (
        check_row_overlap(
            train, test, v.features, target, config.checks.overlap_max_fraction
        ),
        check_target_proxy(
            train,
            test,
            v.features,
            target,
            v.y_true_train,
            v.y_true_test,
            config.checks.proxy_auc_max,
        ),
        check_temporal(train, test, config.split.time_column),
        check_split_order(script_source, script_name),
    )
    metric, warnings = check_metric(
        compute_metric(v.y_true_test, v.y_score),
        config.metric.name,
        config.metric.threshold,
        config.checks.too_good_margin,
    )
    return DataAuditResult(
        run_id=None,
        metric=metric,
        checks=checks,
        extra_findings=warnings,
        data=_summarize(train, test, v, config.split.time_column),
        input_error=None,
    )


def _summarize(
    train: pd.DataFrame,
    test: pd.DataFrame,
    v: ValidatedInputs,
    time_column: str | None,
) -> DataSummary:
    return DataSummary(
        n_train=len(train),
        n_test=len(test),
        positive_rate_train=_rate(v.y_true_train),
        positive_rate_test=_rate(v.y_true_test),
        train_time_range=_time_range(train, time_column, "the train frame"),
        test_time_range=_time_range(test, time_column, "the test frame"),
    )


def _rate(y_true: NDArray[np.bool_]) -> float:
    return round(float(y_true.mean()), _RATE_DECIMALS)


def _time_range(
    frame: pd.DataFrame, time_column: str | None, frame_name: str
) -> tuple[str, str] | None:
    """The earliest and latest time of a split, as ``Timestamp.isoformat()`` (UTC)."""
    if time_column is None:
        return None
    times = parse_times(frame[time_column], time_column, frame_name)
    return (times.min().isoformat(), times.max().isoformat())
