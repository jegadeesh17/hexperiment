import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hexperiment.checks import run_data_checks
from hexperiment.config import HexConfig, parse_config
from hexperiment.data import ValidatedInputs, split_frame, validate_audit_inputs
from hexperiment.models import DataAuditResult, Outcome, Severity

CLEAN_SCRIPT = """\
import hexperiment

df = hexperiment.load()
train, test = hexperiment.split(df)
"""


def make_config(
    *, time_column: str | None = None, threshold: float = 0.7, **checks: float
) -> HexConfig:
    raw: dict = {
        "data": {"path": "data.csv"},
        "target": {"column": "y"},
        "metric": {"name": "roc_auc", "threshold": threshold},
        "split": {"test_fraction": 0.2, "seed": 0},
        "checks": checks,
    }
    if time_column is not None:
        raw["split"]["time_column"] = time_column
    return parse_config(raw, source=Path("hex.toml"), app_dir=Path("."))


def random_frame(seed: int, n: int = 2000) -> pd.DataFrame:
    """Independent features and an independent binary target."""
    gen = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "num_a": gen.normal(size=n),
            "num_b": gen.uniform(size=n),
            "cat": gen.choice(["red", "green", "blue", "grey"], size=n),
            "y": gen.integers(0, 2, size=n),
        }
    )


def run_checks(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_score: np.ndarray,
    *,
    config: HexConfig,
    features: list[str],
    source: str | None = CLEAN_SCRIPT,
) -> tuple[DataAuditResult, ValidatedInputs]:
    v = validate_audit_inputs(train, test, y_score, None, features, config)
    result = run_data_checks(
        train, test, v, config=config, script_source=source, script_name="train.py"
    )
    return result, v


def errors(result: DataAuditResult) -> list:
    findings = [f for c in result.checks for f in c.findings] + list(
        result.extra_findings
    )
    return [f for f in findings if f.severity is Severity.ERROR]


# --- the result and its order (Requirement 2.3) --------------------------------------


def leaking_case() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Test rows copied from train, a feature equal to the target, a perfect score."""
    gen = np.random.default_rng(3)
    train = pd.DataFrame(
        {"x": gen.normal(size=60), "leak": gen.integers(0, 2, size=60)}
    )
    train["y"] = train["leak"]
    test = train.iloc[:20].copy()
    test.index = np.arange(60, 80)
    return train, test, test["y"].to_numpy(dtype=float)


def test_the_checks_run_in_the_fixed_order() -> None:
    train, test, score = leaking_case()
    result, _ = run_checks(
        train, test, score, config=make_config(), features=["x", "leak"]
    )

    assert [c.name for c in result.checks] == [
        "row_overlap",
        "target_proxy",
        "temporal",
        "split_order",
    ]
    assert [c.outcome for c in result.checks] == [
        Outcome.FAILED,
        Outcome.FAILED,
        Outcome.NOT_APPLICABLE,
        Outcome.PASSED,
    ]
    assert result.metric is not None
    assert result.metric.name == "roc_auc"
    assert result.metric.value == 1.0
    assert result.metric.passed is True
    assert [f.check for f in result.extra_findings] == ["metric"]
    assert result.run_id is None
    assert result.input_error is None


def test_a_failing_metric_is_recorded_without_a_finding() -> None:
    train, test, score = leaking_case()
    result, _ = run_checks(
        train, test, 1.0 - score, config=make_config(), features=["x", "leak"]
    )
    assert result.metric is not None
    assert result.metric.value == 0.0
    assert result.metric.passed is False
    assert result.extra_findings == ()


def test_the_split_order_hint_reads_the_script_source() -> None:
    train, test, score = leaking_case()
    config = make_config()
    source = "m = df.x.mean()\ntrain, test = hexperiment.split(df)\n"
    result, _ = run_checks(
        train, test, score, config=config, features=["x"], source=source
    )
    hint = result.checks[3]
    assert [f.location for f in hint.findings] == ["train.py:1"]

    missing, _ = run_checks(
        train, test, score, config=config, features=["x"], source=None
    )
    assert missing.checks[3].outcome is Outcome.INCONCLUSIVE


def test_config_values_reach_the_checks() -> None:
    train, test, score = leaking_case()
    lenient = make_config(
        overlap_max_fraction=1.0, proxy_auc_max=1.0, too_good_margin=0.5
    )
    result, _ = run_checks(
        train, test, score, config=lenient, features=["x", "leak"]
    )
    assert result.checks[0].outcome is Outcome.PASSED  # 20 of 20 is not above 1.0
    assert result.checks[1].outcome is Outcome.FAILED  # a score of 1.0 equals 1.0
    assert result.extra_findings == ()  # 1.0 is not above 0.7 + 0.5


def test_the_result_holds_only_json_values() -> None:
    train, test, score = leaking_case()
    result, _ = run_checks(
        train, test, score, config=make_config(), features=["x", "leak"]
    )
    text = json.dumps(result.to_dict(), allow_nan=False)
    assert DataAuditResult.from_dict(json.loads(text)) == result


def test_the_input_arrays_and_frames_are_not_modified() -> None:
    train, test, score = leaking_case()
    config = make_config()
    v = validate_audit_inputs(train, test, score, None, ["x", "leak"], config)
    train_before, test_before = train.copy(deep=True), test.copy(deep=True)
    arrays_before = (
        v.y_true_train.copy(),
        v.y_true_test.copy(),
        v.y_score.copy(),
    )

    run_data_checks(
        train, test, v, config=config, script_source=CLEAN_SCRIPT, script_name="t.py"
    )

    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(test, test_before)
    assert np.array_equal(v.y_true_train, arrays_before[0])
    assert np.array_equal(v.y_true_test, arrays_before[1])
    assert np.array_equal(v.y_score, arrays_before[2])
    assert v.features == ("x", "leak")


# --- the data summary ----------------------------------------------------------------


def test_the_summary_counts_rows_and_positive_rates() -> None:
    train = pd.DataFrame({"x": np.arange(8.0), "y": [0, 0, 0, 1, 0, 0, 0, 1]})
    test = pd.DataFrame({"x": [9.0, 10.0, 11.0], "y": [1, 0, 1]}, index=[8, 9, 10])
    result, _ = run_checks(
        train,
        test,
        np.array([0.9, 0.2, 0.7]),
        config=make_config(),
        features=["x"],
    )
    assert result.data is not None
    assert result.data.n_train == 8
    assert result.data.n_test == 3
    assert result.data.positive_rate_train == 0.25
    assert result.data.positive_rate_test == pytest.approx(0.666667)
    assert result.data.positive_rate_test == round(2 / 3, 6)
    assert result.data.train_time_range is None
    assert result.data.test_time_range is None


def test_the_summary_has_time_ranges_in_isoformat_utc_when_a_time_column_is_set() -> (
    None
):
    train = pd.DataFrame(
        {
            "t": [
                "2024-03-05T08:00:00+00:00",
                "2024-01-02T00:00:00+00:00",
                "2024-02-10T12:30:00+02:00",
            ],
            "x": [1.0, 2.0, 3.0],
            "y": [0, 1, 0],
        }
    )
    test = pd.DataFrame(
        {
            "t": ["2024-05-01T00:00:00+00:00", "2024-04-01T03:00:00+03:00"],
            "x": [4.0, 5.0],
            "y": [1, 0],
        },
        index=[3, 4],
    )
    result, _ = run_checks(
        train,
        test,
        np.array([0.8, 0.3]),
        config=make_config(time_column="t"),
        features=["x"],
    )
    assert result.data is not None
    assert result.data.train_time_range == (
        pd.Timestamp("2024-01-02", tz="UTC").isoformat(),
        pd.Timestamp("2024-03-05 08:00:00", tz="UTC").isoformat(),
    )
    assert result.data.train_time_range == (
        "2024-01-02T00:00:00+00:00",
        "2024-03-05T08:00:00+00:00",
    )
    assert result.data.test_time_range == (
        "2024-04-01T00:00:00+00:00",
        "2024-05-01T00:00:00+00:00",
    )
    assert result.checks[2].outcome is Outcome.PASSED


# --- a clean synthetic dataset (Requirement 17.4) ------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_independent_random_data_gives_no_error_finding(seed: int) -> None:
    config = make_config()
    train, test = split_frame(random_frame(seed), config.split)
    assert len(train) + len(test) == 2000
    score = np.random.default_rng(seed + 100).uniform(size=len(test))

    result, _ = run_checks(
        train, test, score, config=config, features=["num_a", "num_b", "cat"]
    )

    assert errors(result) == []
    assert [c.outcome for c in result.checks] == [
        Outcome.PASSED,
        Outcome.PASSED,
        Outcome.NOT_APPLICABLE,
        Outcome.PASSED,
    ]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_independent_random_data_with_a_time_split_gives_no_error_finding(
    seed: int,
) -> None:
    config = make_config(time_column="t")
    frame = random_frame(seed)
    frame["t"] = (
        pd.Timestamp("2024-01-01")
        + pd.to_timedelta(
            np.random.default_rng(seed + 50).integers(0, 365 * 86400, size=2000),
            unit="s",
        )
    ).strftime("%Y-%m-%d %H:%M:%S")
    train, test = split_frame(frame, config.split)
    score = np.random.default_rng(seed + 100).uniform(size=len(test))

    result, _ = run_checks(
        train, test, score, config=config, features=["num_a", "num_b", "cat"]
    )

    assert errors(result) == []
    assert result.checks[2].outcome is Outcome.PASSED
    assert result.data is not None
    assert result.data.train_time_range is not None
    assert result.data.test_time_range is not None
    assert result.data.train_time_range[1] <= result.data.test_time_range[0]
