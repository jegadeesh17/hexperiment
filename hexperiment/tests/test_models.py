import json
from dataclasses import replace

from hypothesis import given
from hypothesis import strategies as st

from hexperiment.models import (
    CheckResult,
    CheckSummary,
    DataAuditResult,
    DataSummary,
    Finding,
    JsonValue,
    MetricResult,
    Outcome,
    RunMode,
    Severity,
    SplitSignature,
    Status,
    Verdict,
    WaiverRecord,
)


def make_finding(
    severity: Severity = Severity.ERROR, waiver_reason: str | None = None
) -> Finding:
    return Finding(
        check="target_proxy",
        ghost="The Possession",
        severity=severity,
        location="tenure_months",
        message="tenure_months alone predicts the target with AUC 0.9612",
        evidence={"score": 0.961203, "threshold": 0.95},
        fix="drop the column if it is recorded after the outcome",
        waiver_reason=waiver_reason,
    )


def make_summary(with_time: bool = False) -> DataSummary:
    train_range = ("2024-01-01T00:00:00Z", "2024-09-30T00:00:00Z")
    test_range = ("2024-10-01T00:00:00Z", "2024-12-31T00:00:00Z")
    return DataSummary(
        n_train=5613,
        n_test=1387,
        positive_rate_train=0.265454,
        positive_rate_test=0.270368,
        train_time_range=train_range if with_time else None,
        test_time_range=test_range if with_time else None,
    )


def make_metric() -> MetricResult:
    return MetricResult(name="roc_auc", value=0.873412, threshold=0.8, passed=True)


def make_verdict() -> Verdict:
    return Verdict(
        status=Status.REJECTED,
        mode=RunMode.FULL,
        app="churn",
        run_id="20251101-093012-a1b2c3",
        created_at="2025-11-01T09:30:12Z",
        hexperiment_version="0.1.0",
        metric=make_metric(),
        checks=(
            CheckSummary("row_overlap", "The Doppelganger", Outcome.PASSED),
            CheckSummary("split_order", None, Outcome.PASSED),
            CheckSummary("canary", "The Peeking Spirit", Outcome.FAILED),
        ),
        findings=(
            make_finding(Severity.WAIVED, "Known at prediction time."),
            make_finding(Severity.WARNING),
        ),
        waivers=(WaiverRecord("target_proxy", "tenure_months", "Known.", 1),),
        test_set_uses=3,
        split=SplitSignature("data/churn.csv", 0.2, 0, None),
        data=make_summary(with_time=True),
    )


def through_json(data: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return json.loads(json.dumps(data, sort_keys=True))


def test_finding_round_trip() -> None:
    finding = make_finding()
    assert Finding.from_dict(finding.to_dict()) == finding
    assert Finding.from_dict(through_json(finding.to_dict())) == finding


def test_finding_waiver_reason_only_when_waived() -> None:
    assert "waiver_reason" not in make_finding(Severity.ERROR).to_dict()
    assert "waiver_reason" not in make_finding(Severity.WARNING).to_dict()
    waived = make_finding(Severity.WAIVED, "Known at prediction time.")
    assert waived.to_dict()["waiver_reason"] == "Known at prediction time."
    assert Finding.from_dict(waived.to_dict()) == waived


def test_check_result_round_trip() -> None:
    result = CheckResult(
        "target_proxy", "The Possession", Outcome.FAILED, (make_finding(),)
    )
    assert CheckResult.from_dict(through_json(result.to_dict())) == result
    empty = CheckResult("temporal", "The Prophet", Outcome.NOT_APPLICABLE)
    assert CheckResult.from_dict(empty.to_dict()) == empty


def test_check_summary_round_trip() -> None:
    summary = CheckSummary("canary", "The Peeking Spirit", Outcome.NOT_APPLIED)
    assert CheckSummary.from_dict(through_json(summary.to_dict())) == summary
    assert summary.to_dict()["outcome"] == "not applied"


def test_metric_result_round_trip() -> None:
    metric = make_metric()
    assert MetricResult.from_dict(through_json(metric.to_dict())) == metric
    undefined = MetricResult("roc_auc", None, 0.8, False)
    assert MetricResult.from_dict(through_json(undefined.to_dict())) == undefined


def test_waiver_record_round_trip() -> None:
    reason = "Known at prediction time."
    record = WaiverRecord("target_proxy", "tenure_months", reason, 1)
    assert WaiverRecord.from_dict(through_json(record.to_dict())) == record
    no_column = WaiverRecord("row_overlap", None, "Duplicates are expected here.", 2)
    assert WaiverRecord.from_dict(through_json(no_column.to_dict())) == no_column


def test_split_signature_round_trip() -> None:
    signature = SplitSignature("data/churn.csv", 0.2, 7, "signup_date")
    assert SplitSignature.from_dict(through_json(signature.to_dict())) == signature


def test_data_summary_round_trip_with_and_without_time_ranges() -> None:
    for with_time in (False, True):
        summary = make_summary(with_time)
        assert DataSummary.from_dict(through_json(summary.to_dict())) == summary


def test_data_audit_result_round_trip() -> None:
    result = DataAuditResult(
        run_id="20251101-093012-a1b2c3",
        metric=make_metric(),
        checks=(CheckResult("row_overlap", "The Doppelganger", Outcome.PASSED),),
        extra_findings=(make_finding(Severity.WARNING),),
        data=make_summary(),
        input_error=None,
    )
    assert DataAuditResult.from_dict(through_json(result.to_dict())) == result


def test_data_audit_result_round_trip_with_input_error() -> None:
    error = Finding(
        check="input",
        ghost=None,
        severity=Severity.ERROR,
        location="audit()",
        message="y_score has 3 entries but test has 5 rows",
        evidence={},
        fix="pass one score per test row",
    )
    result = DataAuditResult(None, None, (), (), None, error)
    assert DataAuditResult.from_dict(through_json(result.to_dict())) == result


def test_verdict_round_trip() -> None:
    verdict = make_verdict()
    assert Verdict.from_dict(through_json(verdict.to_dict())) == verdict


def test_verdict_to_dict_uses_plain_strings() -> None:
    data = make_verdict().to_dict()
    assert data["status"] == "REJECTED"
    assert data["mode"] == "full"
    assert sorted(data) == [
        "app", "checks", "checks_overrides", "created_at", "data", "findings",
        "hexperiment_version", "metric", "mode", "run_id", "split", "status",
        "test_set_uses", "waivers",
    ]  # fmt: skip


json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**53), max_value=2**53),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(),
)
json_values = st.recursive(
    json_scalars,
    lambda inner: st.one_of(
        st.lists(inner, max_size=4),
        st.dictionaries(st.text(), inner, max_size=4),
    ),
    max_leaves=10,
)


@given(
    check=st.text(),
    ghost=st.none() | st.text(),
    severity=st.sampled_from(Severity),
    location=st.text(),
    message=st.text(),
    evidence=st.dictionaries(st.text(), json_values, max_size=5),
    fix=st.text(),
    reason=st.text(),
)
def test_finding_round_trip_property(
    check: str,
    ghost: str | None,
    severity: Severity,
    location: str,
    message: str,
    evidence: dict[str, JsonValue],
    fix: str,
    reason: str,
) -> None:
    finding = Finding(
        check=check,
        ghost=ghost,
        severity=severity,
        location=location,
        message=message,
        evidence=evidence,
        fix=fix,
        waiver_reason=reason if severity is Severity.WAIVED else None,
    )
    assert Finding.from_dict(finding.to_dict()) == finding
    assert Finding.from_dict(through_json(finding.to_dict())) == finding


def test_verdict_checks_overrides_defaults_to_empty() -> None:
    verdict = make_verdict()
    assert verdict.checks_overrides == {}
    assert verdict.to_dict()["checks_overrides"] == {}


def test_verdict_checks_overrides_round_trip() -> None:
    verdict = replace(
        make_verdict(), checks_overrides={"proxy_auc_max": 0.99, "max_test_uses": 9}
    )
    assert Verdict.from_dict(through_json(verdict.to_dict())) == verdict


def test_verdict_from_dict_without_checks_overrides_gives_empty() -> None:
    data = make_verdict().to_dict()
    del data["checks_overrides"]
    assert Verdict.from_dict(data).checks_overrides == {}
