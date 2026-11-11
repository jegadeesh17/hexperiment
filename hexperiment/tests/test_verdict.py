import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hexperiment.canary import (
    CanaryOutcome,
    LoadMarker,
    Snapshot,
    decide_canary,
    skipped_outcome,
)
from hexperiment.config import ExecMode, HexConfig, Waiver, parse_config
from hexperiment.errors import HexInputError
from hexperiment.models import (
    CheckResult,
    CheckSummary,
    DataAuditResult,
    DataSummary,
    Finding,
    MetricResult,
    Outcome,
    RunMode,
    Severity,
    SplitSignature,
    Status,
    Verdict,
    WaiverRecord,
)
from hexperiment.verdict import (
    FINDING_ORDER,
    apply_waivers,
    assemble,
    decide_status,
    format_summary,
    from_dict,
    to_json,
)

RUN_ID = "20251101-093012-a1b2c3"
CREATED_AT = "2025-11-01T09:30:12Z"
REASON = "Known at prediction time and legitimately strong."
FIX_PROXY = "check whether this column is recorded after the outcome; if so, drop it"
FIX_HINT = "this runs before the split; make sure it does not use test rows"
FIX_SPIRIT = (
    "test rows influenced training; fit preprocessing on training rows only, "
    "after the split. Suspect lines: train.py:14."
)


def finding(
    check: str,
    severity: Severity = Severity.ERROR,
    location: str = "x",
    ghost: str | None = None,
    message: str | None = None,
    fix: str = "fix it",
    evidence: dict | None = None,
) -> Finding:
    return Finding(
        check=check,
        ghost=ghost,
        severity=severity,
        location=location,
        message=message or f"{check} at {location}",
        evidence=evidence or {},
        fix=fix,
        waiver_reason=REASON if severity is Severity.WAIVED else None,
    )


def proxy(column: str = "x", severity: Severity = Severity.ERROR) -> Finding:
    return finding("target_proxy", severity, column, "The Possession", fix=FIX_PROXY)


def overlap() -> Finding:
    return finding("row_overlap", location="train vs test", ghost="The Doppelganger")


def prophet() -> Finding:
    return finding("temporal", location="t", ghost="The Prophet")


def waiver(check: str, column: str | None = None, reason: str = REASON) -> Waiver:
    return Waiver(check=check, reason=reason, column=column)


def make_config(
    waivers: tuple[Waiver, ...] = (), max_test_uses: int = 5
) -> HexConfig:
    raw: dict = {
        "app": {"name": "churn", "description": "Cancellations."},
        "data": {"path": "data/churn.csv"},
        "target": {"column": "y"},
        "metric": {"name": "roc_auc", "threshold": 0.8},
        "split": {"test_fraction": 0.25, "seed": 4, "time_column": "t"},
        "checks": {"max_test_uses": max_test_uses},
        "waivers": [
            {"check": w.check, "reason": w.reason}
            | ({"column": w.column} if w.column else {})
            for w in waivers
        ],
    }
    return parse_config(raw, source=Path("hex.toml"), app_dir=Path("churn"))


GOOD_METRIC = MetricResult("roc_auc", 0.85, 0.8, True)
BAD_METRIC = MetricResult("roc_auc", 0.7, 0.8, False)
SUMMARY = DataSummary(10, 5, 0.5, 0.4, None, None)


def passed(name: str, ghost: str | None, *findings: Finding) -> CheckResult:
    return CheckResult(name, ghost, Outcome.PASSED, tuple(findings))


def failed(name: str, ghost: str | None, *findings: Finding) -> CheckResult:
    return CheckResult(name, ghost, Outcome.FAILED, tuple(findings))


def clean_checks() -> tuple[CheckResult, ...]:
    return (
        passed("row_overlap", "The Doppelganger"),
        passed("target_proxy", "The Possession"),
        CheckResult("temporal", "The Prophet", Outcome.NOT_APPLICABLE),
        passed("split_order", None),
    )


def audit_result(
    checks: tuple[CheckResult, ...] | None = None,
    extra: tuple[Finding, ...] = (),
    metric: MetricResult | None = GOOD_METRIC,
    input_error: Finding | None = None,
) -> DataAuditResult:
    return DataAuditResult(
        run_id=None,
        metric=metric,
        checks=clean_checks() if checks is None else checks,
        extra_findings=extra,
        data=SUMMARY,
        input_error=input_error,
    )


PASSED_CANARY = CanaryOutcome(Outcome.PASSED, Outcome.PASSED, ())
SPIRIT = finding(
    "canary", location="train.py", ghost="The Peeking Spirit", fix=FIX_SPIRIT
)
FAILED_CANARY = CanaryOutcome(Outcome.FAILED, Outcome.PASSED, (SPIRIT,))
SHAPESHIFTER = finding(
    "determinism", Severity.WARNING, "train.py", "The Shapeshifter"
)
NOISY_CANARY = CanaryOutcome(Outcome.INCONCLUSIVE, Outcome.FAILED, (SHAPESHIFTER,))


def build(
    result: DataAuditResult | None = None,
    canary: CanaryOutcome = PASSED_CANARY,
    config: HexConfig | None = None,
    *,
    mode: RunMode = RunMode.FULL,
    test_set_uses: int = 1,
    run_id: str = RUN_ID,
    created_at: str = CREATED_AT,
) -> Verdict:
    return assemble(
        audit_result() if result is None else result,
        canary,
        make_config() if config is None else config,
        mode=mode,
        run_id=run_id,
        created_at=created_at,
        version="0.1.0",
        test_set_uses=test_set_uses,
    )


# --- decide_status: the truth table (Requirement 13.1) -------------------------------

ERROR_STATES = {
    "none": [],
    "warning": [finding("split_order", Severity.WARNING)],
    "waived": [proxy(severity=Severity.WAIVED)],
    "error": [proxy()],
    "error and waived": [proxy("a"), proxy("b", Severity.WAIVED)],
}


def expected_status(
    metric_passed: bool, has_error: bool, canary: Outcome, mode: RunMode
) -> Status:
    if not metric_passed:
        return Status.REJECTED
    if has_error:
        return Status.REJECTED
    if mode is RunMode.FULL and canary is Outcome.PASSED:
        return Status.APPROVED
    return Status.UNVERIFIED


@pytest.mark.parametrize(
    ("metric_passed", "state", "canary", "mode"),
    list(itertools.product([True, False], ERROR_STATES, Outcome, RunMode)),
)
def test_decide_status_follows_the_truth_table(
    metric_passed: bool, state: str, canary: Outcome, mode: RunMode
) -> None:
    has_error = state in ("error", "error and waived")
    actual = decide_status(metric_passed, ERROR_STATES[state], canary, mode)
    assert actual is expected_status(metric_passed, has_error, canary, mode)


@pytest.mark.parametrize(
    ("metric_passed", "findings", "canary", "mode", "status"),
    [
        (True, [], Outcome.PASSED, RunMode.FULL, Status.APPROVED),
        (False, [], Outcome.PASSED, RunMode.FULL, Status.REJECTED),
        (True, [proxy()], Outcome.PASSED, RunMode.FULL, Status.REJECTED),
        (True, [SPIRIT], Outcome.FAILED, RunMode.FULL, Status.REJECTED),
        (True, [], Outcome.PASSED, RunMode.FAST, Status.UNVERIFIED),
        (True, [], Outcome.SKIPPED, RunMode.FAST, Status.UNVERIFIED),
        (True, [], Outcome.INCONCLUSIVE, RunMode.FULL, Status.UNVERIFIED),
        (True, [], Outcome.NOT_APPLIED, RunMode.FULL, Status.UNVERIFIED),
        (True, [], Outcome.SKIPPED, RunMode.FULL, Status.UNVERIFIED),
        (False, [proxy()], Outcome.FAILED, RunMode.FAST, Status.REJECTED),
        (True, [proxy(severity=Severity.WAIVED)], Outcome.PASSED, RunMode.FULL,
         Status.APPROVED),
    ],
)
def test_decide_status_examples(metric_passed, findings, canary, mode, status) -> None:
    assert decide_status(metric_passed, findings, canary, mode) is status


# --- apply_waivers (Requirements 10.7, 11.2) -----------------------------------------


def test_a_matching_waiver_marks_the_finding_waived_with_the_reason() -> None:
    original = proxy("tenure")
    findings, records = apply_waivers([original], [waiver("target_proxy", "tenure")])

    (waived,) = findings
    assert waived.severity is Severity.WAIVED
    assert waived.waiver_reason == REASON
    assert (waived.check, waived.ghost, waived.location) == (
        original.check,
        original.ghost,
        original.location,
    )
    assert (waived.message, waived.evidence, waived.fix) == (
        original.message,
        original.evidence,
        original.fix,
    )
    assert records == (WaiverRecord("target_proxy", "tenure", REASON, 1),)
    assert original.severity is Severity.ERROR  # the input is not modified


def test_a_waiver_with_a_column_waives_only_that_column() -> None:
    findings, records = apply_waivers(
        [proxy("a"), proxy("b")], [waiver("target_proxy", "b")]
    )
    assert [f.severity for f in findings] == [Severity.ERROR, Severity.WAIVED]
    assert records[0].findings_waived == 1


def test_a_waiver_without_a_column_waives_every_finding_of_its_check() -> None:
    # Only row_overlap and temporal waivers can lack a column (Requirement 11.1).
    second = finding("row_overlap", location="elsewhere", ghost="The Doppelganger")
    findings, records = apply_waivers(
        [overlap(), second, proxy("a")], [waiver("row_overlap")]
    )
    assert [f.severity for f in findings] == [
        Severity.WAIVED,
        Severity.WAIVED,
        Severity.ERROR,
    ]
    assert records == (WaiverRecord("row_overlap", None, REASON, 2),)


@pytest.mark.parametrize("check", ["row_overlap", "temporal"])
def test_overlap_and_temporal_findings_can_be_waived(check: str) -> None:
    target = overlap() if check == "row_overlap" else prophet()
    findings, records = apply_waivers([target], [waiver(check)])
    assert findings[0].severity is Severity.WAIVED
    assert len(records) == 1


@pytest.mark.parametrize(
    "other",
    [SPIRIT, SHAPESHIFTER, finding("split_order", Severity.WARNING, "train.py:3")]
    + [finding("metric", Severity.WARNING), finding("input")],
    ids=["canary", "determinism", "split_order", "metric", "input"],
)
def test_other_findings_are_never_waived(other: Finding) -> None:
    findings, records = apply_waivers([other], [waiver(other.check)])
    assert findings[0] == other
    assert records == ()
    stale = findings[1]
    assert stale.check == "waiver"  # the waiver matched nothing
    assert stale.severity is Severity.WARNING


def test_a_waiver_that_matches_nothing_becomes_a_warning() -> None:
    findings, records = apply_waivers([proxy("a")], [waiver("target_proxy", "gone")])

    assert records == ()
    original, stale = findings
    assert original.severity is Severity.ERROR  # the other column is not waived
    assert stale == Finding(
        check="waiver",
        ghost=None,
        severity=Severity.WARNING,
        location="target_proxy:gone",
        message="the waiver for target_proxy:gone matched no finding",
        evidence={"check": "target_proxy", "column": "gone"},
        fix="remove this waiver from hex.toml if the finding no longer occurs",
    )


def test_a_stale_waiver_without_a_column_is_located_at_its_check() -> None:
    findings, _ = apply_waivers([], [waiver("temporal")])
    (stale,) = findings
    assert stale.location == "temporal"
    assert stale.evidence == {"check": "temporal", "column": None}
    assert stale.severity is Severity.WARNING


def test_stale_warnings_follow_the_findings_in_waiver_order() -> None:
    findings, records = apply_waivers(
        [proxy("a"), overlap()],
        [waiver("temporal"), waiver("target_proxy", "a"), waiver("row_overlap", None)]
        + [waiver("target_proxy", "z")],
    )
    assert [f.check for f in findings] == [
        "target_proxy",
        "row_overlap",
        "waiver",
        "waiver",
    ]
    assert [f.location for f in findings[2:]] == ["temporal", "target_proxy:z"]
    assert [r.check for r in records] == ["target_proxy", "row_overlap"]


def test_two_waivers_for_one_finding_both_count_and_the_first_reason_is_used() -> None:
    first, second = "The first reason given.", "The second reason given."
    findings, records = apply_waivers(
        [proxy("a")],
        [waiver("target_proxy", None, first), waiver("target_proxy", "a", second)],
    )
    assert findings[0].waiver_reason == first
    assert [r.reason for r in records] == [first, second]
    assert [r.findings_waived for r in records] == [1, 1]
    assert len(findings) == 1  # neither waiver is stale


def test_no_waivers_returns_the_findings_unchanged() -> None:
    original = [proxy("a"), overlap()]
    findings, records = apply_waivers(original, [])
    assert findings == tuple(original)
    assert records == ()


# --- assemble (Requirements 4.7, 12.2, 13.1, 13.2) ----------------------------------


def test_an_input_error_gives_a_rejected_verdict_with_one_finding() -> None:
    error = finding("input", location="audit()", fix="pass one score per row")
    verdict = build(
        DataAuditResult(None, None, (), (), None, error),
        skipped_outcome(),
        make_config((waiver("target_proxy", "x"),), max_test_uses=1),
        test_set_uses=9,
    )

    assert verdict.status is Status.REJECTED
    assert verdict.findings == (error,)  # no test-set or stale-waiver warning
    assert verdict.checks == ()
    assert verdict.waivers == ()
    assert verdict.metric == MetricResult("roc_auc", None, 0.8, False)
    assert verdict.test_set_uses == 9
    assert verdict.data is None


def test_a_clean_full_run_is_approved_and_lists_six_checks() -> None:
    verdict = build()

    assert verdict.status is Status.APPROVED
    assert verdict.mode is RunMode.FULL
    assert verdict.findings == ()
    assert verdict.checks == (
        CheckSummary("row_overlap", "The Doppelganger", Outcome.PASSED),
        CheckSummary("target_proxy", "The Possession", Outcome.PASSED),
        CheckSummary("temporal", "The Prophet", Outcome.NOT_APPLICABLE),
        CheckSummary("split_order", None, Outcome.PASSED),
        CheckSummary("canary", "The Peeking Spirit", Outcome.PASSED),
        CheckSummary("determinism", "The Shapeshifter", Outcome.PASSED),
    )


def test_the_verdict_carries_the_run_and_config_fields() -> None:
    verdict = build(test_set_uses=3)
    assert verdict.app == "churn"
    assert verdict.run_id == RUN_ID
    assert verdict.created_at == CREATED_AT
    assert verdict.hexperiment_version == "0.1.0"
    assert verdict.metric == GOOD_METRIC
    assert verdict.test_set_uses == 3
    assert verdict.split == SplitSignature("data/churn.csv", 0.25, 4, "t")
    assert verdict.data == SUMMARY


def test_a_fast_run_is_unverified() -> None:
    verdict = build(canary=skipped_outcome(), mode=RunMode.FAST)
    assert verdict.status is Status.UNVERIFIED
    assert verdict.mode is RunMode.FAST
    assert [c.outcome for c in verdict.checks[-2:]] == [Outcome.SKIPPED] * 2


def test_a_failed_metric_is_rejected_without_a_finding() -> None:
    verdict = build(audit_result(metric=BAD_METRIC))
    assert verdict.status is Status.REJECTED
    assert verdict.findings == ()


def test_a_data_check_error_is_rejected() -> None:
    checks = clean_checks()[:1] + (failed("target_proxy", "The Possession", proxy()),)
    verdict = build(audit_result(checks + clean_checks()[2:]))
    assert verdict.status is Status.REJECTED
    assert verdict.checks[1].outcome is Outcome.FAILED
    assert [f.check for f in verdict.findings] == ["target_proxy"]


def test_the_peeking_spirit_is_rejected_and_listed() -> None:
    verdict = build(canary=FAILED_CANARY)
    assert verdict.status is Status.REJECTED
    assert verdict.findings == (SPIRIT,)
    assert verdict.checks[-2:] == (
        CheckSummary("canary", "The Peeking Spirit", Outcome.FAILED),
        CheckSummary("determinism", "The Shapeshifter", Outcome.PASSED),
    )


def test_the_shapeshifter_is_a_warning_that_leaves_the_run_unverified() -> None:
    verdict = build(canary=NOISY_CANARY)
    assert verdict.status is Status.UNVERIFIED
    assert verdict.findings == (SHAPESHIFTER,)
    assert verdict.checks[-2:] == (
        CheckSummary("canary", "The Peeking Spirit", Outcome.INCONCLUSIVE),
        CheckSummary("determinism", "The Shapeshifter", Outcome.FAILED),
    )


def with_waivers(config: HexConfig, *extra: Waiver) -> HexConfig:
    """Add waivers that ``parse_config`` would reject (a canary waiver)."""
    return replace(config, waivers=(*config.waivers, *extra))


def test_a_canary_finding_is_never_waived() -> None:
    config = with_waivers(make_config(), waiver("canary"), waiver("determinism"))
    verdict = build(canary=FAILED_CANARY, config=config)

    assert verdict.status is Status.REJECTED
    assert SPIRIT in verdict.findings  # still an error
    assert verdict.waivers == ()
    assert [f.location for f in verdict.findings if f.check == "waiver"] == [
        "canary",
        "determinism",
    ]


def test_waived_findings_do_not_reject_and_the_check_passes() -> None:
    checks = (
        passed("row_overlap", "The Doppelganger"),
        failed("target_proxy", "The Possession", proxy("a"), proxy("b")),
        *clean_checks()[2:],
    )
    config = make_config((waiver("target_proxy", "a"), waiver("target_proxy", "b")))
    verdict = build(audit_result(checks), config=config)

    assert verdict.status is Status.APPROVED
    assert [f.severity for f in verdict.findings] == [Severity.WAIVED] * 2
    assert verdict.checks[1] == CheckSummary(
        "target_proxy", "The Possession", Outcome.PASSED
    )
    assert verdict.waivers == (
        WaiverRecord("target_proxy", "a", REASON, 1),
        WaiverRecord("target_proxy", "b", REASON, 1),
    )


def test_a_check_with_one_waived_and_one_open_error_stays_failed() -> None:
    checks = (
        passed("row_overlap", "The Doppelganger"),
        failed("target_proxy", "The Possession", proxy("a"), proxy("b")),
        *clean_checks()[2:],
    )
    config = make_config((waiver("target_proxy", "a"),))
    verdict = build(audit_result(checks), config=config)
    assert verdict.status is Status.REJECTED
    assert verdict.checks[1].outcome is Outcome.FAILED
    assert [f.severity for f in verdict.findings] == [Severity.WAIVED, Severity.ERROR]


def test_a_stale_waiver_is_a_warning_that_does_not_change_the_status() -> None:
    config = make_config((waiver("temporal"), waiver("target_proxy", "gone")))
    verdict = build(config=config)

    assert verdict.status is Status.APPROVED
    assert verdict.waivers == ()
    assert [(f.check, f.severity, f.location) for f in verdict.findings] == [
        ("waiver", Severity.WARNING, "target_proxy:gone"),
        ("waiver", Severity.WARNING, "temporal"),
    ]


def test_the_test_set_uses_warning_appears_above_the_limit_only() -> None:
    at_limit = build(test_set_uses=5)
    assert at_limit.findings == ()

    over = build(test_set_uses=6)
    (warning,) = over.findings
    assert warning.check == "test_set_uses"
    assert warning.ghost is None
    assert warning.severity is Severity.WARNING
    assert warning.location == "runs/"
    assert warning.message == (
        "the metric may be optimistic because the test set has been "
        "evaluated 6 times"
    )
    assert warning.evidence == {"uses": 6, "max_test_uses": 5}
    assert warning.fix
    assert over.status is Status.APPROVED  # a warning never changes the status
    assert over.test_set_uses == 6


def test_the_too_good_warning_is_kept_and_cannot_be_waived() -> None:
    too_good = finding("metric", Severity.WARNING, "metric", message="too good")
    config = with_waivers(make_config(), waiver("metric"))
    verdict = build(audit_result(extra=(too_good,)), config=config)

    assert too_good in verdict.findings
    assert verdict.waivers == ()
    assert verdict.status is Status.APPROVED


def test_findings_are_sorted_by_check_order_then_location_then_message() -> None:
    stale_config = make_config((waiver("temporal"),))
    hint_b = finding("split_order", Severity.WARNING, "train.py:20", message="b")
    hint_a = finding("split_order", Severity.WARNING, "train.py:3", message="z")
    hint_c = finding("split_order", Severity.WARNING, "train.py:3", message="a")
    too_good = finding("metric", Severity.WARNING, "metric")
    checks = (
        passed("row_overlap", "The Doppelganger", overlap()),
        failed("target_proxy", "The Possession", proxy("b"), proxy("a")),
        CheckResult("temporal", "The Prophet", Outcome.NOT_APPLICABLE),
        passed("split_order", None, hint_b, hint_a, hint_c),
    )
    verdict = build(
        audit_result(checks, extra=(too_good,)),
        NOISY_CANARY,
        stale_config,
        test_set_uses=9,
    )

    assert [(f.check, f.location, f.message) for f in verdict.findings] == [
        ("row_overlap", "train vs test", overlap().message),
        ("target_proxy", "a", "target_proxy at a"),
        ("target_proxy", "b", "target_proxy at b"),
        ("split_order", "train.py:20", "b"),
        ("split_order", "train.py:3", "a"),
        ("split_order", "train.py:3", "z"),
        ("determinism", "train.py", "determinism at train.py"),
        ("metric", "metric", "metric at metric"),
        ("test_set_uses", "runs/", verdict.findings[8].message),
        ("waiver", "temporal", "the waiver for temporal matched no finding"),
    ]
    assert [f.check for f in verdict.findings] == sorted(
        (f.check for f in verdict.findings), key=FINDING_ORDER.index
    )


def test_the_order_of_the_input_findings_does_not_change_the_output() -> None:
    first = (proxy("a"), proxy("b"), proxy("c"))
    config = make_config()

    def run(order: tuple[Finding, ...]) -> Verdict:
        return build(audit_result((failed("target_proxy", "The Possession", *order),)),
                     config=config)

    assert run(first).findings == run(first[::-1]).findings


def test_a_result_without_a_metric_or_an_input_error_is_rejected_as_malformed() -> None:
    with pytest.raises(HexInputError, match="no metric") as error:
        build(audit_result(metric=None))
    assert error.value.fix


def test_assemble_does_not_modify_the_result() -> None:
    result = audit_result(
        (failed("target_proxy", "The Possession", proxy("a")), *clean_checks()[2:])
    )
    before = result.to_dict()
    build(result, config=make_config((waiver("target_proxy", "a"),)))
    assert result.to_dict() == before


# --- to_json and from_dict (Requirement 13.4) ----------------------------------------


def test_to_json_has_sorted_keys_two_space_indent_and_a_final_newline() -> None:
    text = to_json(build(canary=FAILED_CANARY))

    assert text.endswith("}\n")
    assert not text.endswith("\n\n")
    assert text.startswith('{\n  "app": "churn",\n  "checks": [\n    {\n')
    parsed = json.loads(text, object_pairs_hook=lambda pairs: [k for k, _ in pairs])
    assert parsed == sorted(parsed)  # top-level keys


def test_to_json_keeps_non_ascii_text() -> None:
    verdict = build(audit_result(extra=(finding("metric", Severity.WARNING,
                                                message="café ✓"),)))
    text = to_json(verdict)
    assert "café ✓" in text
    assert "\\u" not in text
    assert from_dict(json.loads(text)) == verdict


def test_non_finite_numbers_become_null() -> None:
    bad = finding(
        "metric",
        Severity.WARNING,
        evidence={
            "a": float("nan"),
            "b": [float("inf"), 1.5],
            "c": {"d": -float("inf")},
        },
    )
    verdict = build(audit_result(extra=(bad,)))
    text = to_json(verdict)
    for token in ("NaN", "Infinity"):
        assert token not in text
    (loaded,) = json.loads(text)["findings"]
    assert loaded["evidence"] == {"a": None, "b": [None, 1.5], "c": {"d": None}}


def test_a_nan_metric_value_becomes_null() -> None:
    result = audit_result(metric=MetricResult("roc_auc", float("nan"), 0.8, False))
    assert json.loads(to_json(build(result)))["metric"]["value"] is None


def test_the_json_has_every_field_of_requirement_13_2() -> None:
    checks = (
        clean_checks()[0],
        failed("target_proxy", "The Possession", proxy("a")),
        *clean_checks()[2:],
    )
    verdict = build(
        audit_result(checks),
        FAILED_CANARY,
        make_config((waiver("target_proxy", "a"),)),
    )
    raw = json.loads(to_json(verdict))

    assert set(raw) >= {
        "status",
        "mode",
        "app",
        "run_id",
        "created_at",
        "hexperiment_version",
        "metric",
        "checks",
        "findings",
        "waivers",
        "test_set_uses",
    }
    assert raw["metric"] == {
        "name": "roc_auc",
        "passed": True,
        "threshold": 0.8,
        "value": 0.85,
    }
    assert set(raw["checks"][0]) == {"name", "ghost", "outcome"}
    waived_json = next(f for f in raw["findings"] if f["severity"] == "waived")
    assert waived_json["waiver_reason"] == REASON
    other = next(f for f in raw["findings"] if f["severity"] == "error")
    assert "waiver_reason" not in other
    assert set(other) == {
        "check",
        "evidence",
        "fix",
        "ghost",
        "location",
        "message",
        "severity",
    }


def test_from_dict_rejects_malformed_data_with_a_hex_input_error() -> None:
    raw = json.loads(to_json(build()))
    for broken in (
        {k: v for k, v in raw.items() if k != "status"},
        {**raw, "status": "MAYBE"},
        {**raw, "checks": [{"name": "x"}]},
        {**raw, "metric": None},
    ):
        with pytest.raises(HexInputError) as error:
            from_dict(broken)
        assert error.value.fix
    with pytest.raises(HexInputError):
        from_dict([])  # type: ignore[arg-type]


# --- format_summary (Requirement 16.4) -----------------------------------------------


def summary_verdict(
    findings: tuple[Finding, ...], metric: MetricResult, status: Status
) -> Verdict:
    return Verdict(
        status=status,
        mode=RunMode.FULL,
        app="churn",
        run_id=RUN_ID,
        created_at=CREATED_AT,
        hexperiment_version="0.1.0",
        metric=metric,
        checks=(),
        findings=findings,
        waivers=(),
        test_set_uses=1,
        split=SplitSignature("d.csv", 0.2, 0, None),
        data=None,
    )


EXAMPLE_SUMMARY = (
    "REJECTED (full)  roc_auc 0.8734 >= 0.8000  run 20251101-093012-a1b2c3\n"
    "  waived   The Possession      tenure_months  "
    "check whether this column is recorded after the outcome; if so, drop it\n"
    "  warning  split_order         train.py:14    "
    "this runs before the split; make sure it does not use test rows\n"
    "  error    The Peeking Spirit  train.py       "
    "test rows influenced training; fit preprocessing on training rows only, "
    "after the split. Suspect lines: train.py:14."
)


def test_format_summary_matches_the_example_of_the_design() -> None:
    verdict = summary_verdict(
        (
            proxy("tenure_months", Severity.WAIVED),
            finding(
                "split_order", Severity.WARNING, "train.py:14", fix=FIX_HINT
            ),
            SPIRIT,
        ),
        MetricResult("roc_auc", 0.8734, 0.8, True),
        Status.REJECTED,
    )
    assert format_summary(verdict) == EXAMPLE_SUMMARY


def test_format_summary_uses_less_than_when_the_metric_fails() -> None:
    verdict = summary_verdict((), MetricResult("roc_auc", 0.71234, 0.8, False),
                              Status.REJECTED)
    assert format_summary(verdict) == (
        "REJECTED (full)  roc_auc 0.7123 < 0.8000  run 20251101-093012-a1b2c3"
    )


def test_format_summary_without_findings_is_one_line_without_trailing_text() -> None:
    verdict = summary_verdict((), MetricResult("roc_auc", 0.9, 0.8, True),
                              Status.APPROVED)
    text = format_summary(verdict)
    assert text == "APPROVED (full)  roc_auc 0.9000 >= 0.8000  run " + RUN_ID
    assert "\n" not in text


def test_format_summary_of_an_input_error_shows_the_check_name() -> None:
    error = finding("input", location="audit()", fix="pass one score per row")
    verdict = build(DataAuditResult(None, None, (), (), None, error),
                    skipped_outcome())
    assert format_summary(verdict).splitlines() == [
        "REJECTED (full)  roc_auc n/a < 0.8000  run " + RUN_ID,
        "  error    input  audit()  pass one score per row",
    ]


def test_format_summary_keeps_each_finding_on_one_line() -> None:
    odd = finding("metric", Severity.WARNING, "a\nb", fix="line one\nline two  end")
    verdict = summary_verdict((odd,), GOOD_METRIC, Status.UNVERIFIED)
    lines = format_summary(verdict).splitlines()
    assert lines[1] == "  warning  metric  a b  line one line two end"


def test_format_summary_of_an_assembled_verdict() -> None:
    checks = (
        passed("row_overlap", "The Doppelganger"),
        failed("target_proxy", "The Possession", proxy("tenure_months")),
        *clean_checks()[2:3],
        passed("split_order", None,
               finding("split_order", Severity.WARNING, "train.py:14", fix=FIX_HINT)),
    )
    verdict = build(
        audit_result(checks, metric=MetricResult("roc_auc", 0.8734, 0.8, True)),
        FAILED_CANARY,
        make_config((waiver("target_proxy", "tenure_months"),)),
    )
    assert format_summary(verdict) == EXAMPLE_SUMMARY


# --- properties ----------------------------------------------------------------------

texts = st.text(max_size=12)
scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-(2**60), 2**60),
    st.floats(allow_nan=False, allow_infinity=False),
    texts,
)
evidence_values = st.one_of(scalars, st.lists(scalars, max_size=3))


@st.composite
def findings_st(draw: st.DrawFn) -> Finding:
    severity = draw(st.sampled_from(list(Severity)))
    return Finding(
        check=draw(st.sampled_from(FINDING_ORDER)),
        ghost=draw(st.one_of(st.none(), texts)),
        severity=severity,
        location=draw(texts),
        message=draw(texts),
        evidence=draw(st.dictionaries(texts, evidence_values, max_size=3)),
        fix=draw(texts),
        waiver_reason=draw(texts) if severity is Severity.WAIVED else None,
    )


finite = st.floats(allow_nan=False, allow_infinity=False)
time_ranges = st.one_of(st.none(), st.tuples(texts, texts))
metrics_st = st.builds(
    MetricResult,
    name=texts,
    value=st.one_of(st.none(), finite),
    threshold=finite,
    passed=st.booleans(),
)
summaries_st = st.builds(
    DataSummary,
    n_train=st.integers(0, 10**6),
    n_test=st.integers(0, 10**6),
    positive_rate_train=finite,
    positive_rate_test=finite,
    train_time_range=time_ranges,
    test_time_range=time_ranges,
)
verdicts_st = st.builds(
    Verdict,
    status=st.sampled_from(list(Status)),
    mode=st.sampled_from(list(RunMode)),
    app=texts,
    run_id=texts,
    created_at=texts,
    hexperiment_version=texts,
    metric=metrics_st,
    checks=st.lists(
        st.builds(
            CheckSummary,
            name=texts,
            ghost=st.one_of(st.none(), texts),
            outcome=st.sampled_from(list(Outcome)),
        ),
        max_size=6,
    ).map(tuple),
    findings=st.lists(findings_st(), max_size=5).map(tuple),
    waivers=st.lists(
        st.builds(
            WaiverRecord,
            check=texts,
            column=st.one_of(st.none(), texts),
            reason=texts,
            findings_waived=st.integers(0, 100),
        ),
        max_size=3,
    ).map(tuple),
    test_set_uses=st.integers(0, 10**6),
    split=st.builds(
        SplitSignature,
        data_path=texts,
        test_fraction=finite,
        seed=st.integers(0, 2**32),
        time_column=st.one_of(st.none(), texts),
    ),
    data=st.one_of(st.none(), summaries_st),
)


def reject_constant(name: str) -> None:
    raise AssertionError(f"the JSON contains {name}")


@given(verdict=verdicts_st)
def test_json_round_trips_with_sorted_keys_and_no_nan(verdict: Verdict) -> None:
    text = to_json(verdict)

    seen: list[list[str]] = []

    def record(pairs: list[tuple[str, object]]) -> dict:
        seen.append([k for k, _ in pairs])
        return dict(pairs)

    parsed = json.loads(text, object_pairs_hook=record, parse_constant=reject_constant)
    assert all(keys == sorted(keys) for keys in seen)
    assert from_dict(parsed) == verdict
    assert text.endswith("\n")


@given(
    metric_passed=st.booleans(),
    findings=st.lists(findings_st(), max_size=6),
    extra=st.lists(findings_st(), max_size=4),
    canary=st.sampled_from(list(Outcome)),
    mode=st.sampled_from(list(RunMode)),
)
def test_waived_findings_never_change_the_status(
    metric_passed: bool,
    findings: list[Finding],
    extra: list[Finding],
    canary: Outcome,
    mode: RunMode,
) -> None:
    waived = [
        replace(f, severity=Severity.WAIVED, waiver_reason="a reason") for f in extra
    ]
    assert decide_status(metric_passed, findings + waived, canary, mode) is (
        decide_status(metric_passed, findings, canary, mode)
    )


canary_outcomes_st = st.sampled_from(
    [PASSED_CANARY, FAILED_CANARY, NOISY_CANARY, skipped_outcome()]
)


@given(
    canary=canary_outcomes_st,
    mode=st.sampled_from(list(RunMode)),
    metric=st.sampled_from([GOOD_METRIC, BAD_METRIC]),
    uses=st.integers(1, 9),
)
def test_a_waived_error_gives_the_same_status_as_no_error(
    canary: CanaryOutcome, mode: RunMode, metric: MetricResult, uses: int
) -> None:
    with_error = (
        passed("row_overlap", "The Doppelganger"),
        failed("target_proxy", "The Possession", proxy("a")),
        *clean_checks()[2:],
    )
    waived = build(
        audit_result(with_error, metric=metric),
        canary,
        make_config((waiver("target_proxy", "a"),)),
        mode=mode,
        test_set_uses=uses,
    )
    clean = build(
        audit_result(metric=metric),
        canary,
        make_config(),
        mode=mode,
        test_set_uses=uses,
    )
    assert waived.status is clean.status


tokens = st.text("0123456789abcdef", min_size=6, max_size=6)


@given(
    errors=st.lists(st.sampled_from(["a", "b", "c", "d"]), max_size=4, unique=True),
    waived_columns=st.lists(st.sampled_from(["a", "b", "z"]), max_size=3, unique=True),
    canary=canary_outcomes_st,
    mode=st.sampled_from(list(RunMode)),
    metric=st.sampled_from([GOOD_METRIC, BAD_METRIC]),
    uses=st.integers(1, 9),
    tokens=st.lists(tokens, min_size=2, max_size=2, unique=True),
)
def test_equal_inputs_with_another_run_id_and_date_give_equal_results(
    errors: list[str],
    waived_columns: list[str],
    canary: CanaryOutcome,
    mode: RunMode,
    metric: MetricResult,
    uses: int,
    tokens: list[str],
) -> None:
    proxies = tuple(proxy(c) for c in errors)
    checks = (
        passed("row_overlap", "The Doppelganger"),
        (failed if proxies else passed)("target_proxy", "The Possession", *proxies),
        *clean_checks()[2:],
    )
    config = make_config(tuple(waiver("target_proxy", c) for c in waived_columns))

    def run(token: str, stamp: str) -> Verdict:
        return build(
            audit_result(checks, metric=metric),
            canary,
            config,
            mode=mode,
            test_set_uses=uses,
            run_id=f"20250101-000000-{token}",
            created_at=stamp,
        )

    one = run(tokens[0], "2025-01-01T00:00:00Z")
    two = run(tokens[1], "2031-12-31T23:59:59Z")
    assert one.run_id != two.run_id
    assert one.status is two.status
    assert one.metric == two.metric
    assert one.findings == two.findings
    assert one.checks == two.checks
    assert one.waivers == two.waivers


# --- a full run without training predictions (Requirement 2.6) -----------------------


def snapshot_without_scores(mode: ExecMode) -> Snapshot:
    return Snapshot(
        mode=mode,
        run_id=RUN_ID,
        features=("x",),
        train_ids=np.arange(4, dtype=np.int64),
        test_ids=np.arange(4, 6, dtype=np.int64),
        train_row_hash=np.arange(4, dtype=np.uint64),
        y_score_train=None,
    )


def test_a_full_run_without_training_scores_and_no_difference_ends_unverified():
    normal = snapshot_without_scores(ExecMode.NORMAL)
    canary = snapshot_without_scores(ExecMode.CANARY)
    marker = LoadMarker(ExecMode.CANARY, RUN_ID, np.array([4, 5], dtype=np.int64))

    def no_control() -> Snapshot:
        raise AssertionError("no difference, so no control execution")

    outcome = decide_canary(normal, canary, marker, no_control, ())
    verdict = build(canary=outcome, mode=RunMode.FULL)
    assert verdict.status is Status.UNVERIFIED
    assert verdict.metric.passed
    canary_check = next(c for c in verdict.checks if c.name == "canary")
    assert canary_check.outcome is Outcome.INCONCLUSIVE
    (warning,) = [f for f in verdict.findings if f.check == "canary"]
    assert warning.severity is Severity.WARNING
    assert "y_score_train" in warning.fix
