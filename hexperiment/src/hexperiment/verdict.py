"""Verdict assembly: findings, waivers, status and the text forms of a verdict.

Every function here is pure. The clock, the run id token and the file system belong
to ``runs.py`` and the command line.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from hexperiment.canary import CanaryOutcome
from hexperiment.config import HexConfig, Waiver
from hexperiment.errors import HexInputError
from hexperiment.models import (
    CheckSummary,
    DataAuditResult,
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

# Sort order of findings by check name (Requirement 13.5).
FINDING_ORDER: tuple[str, ...] = (
    "input",
    "row_overlap",
    "target_proxy",
    "temporal",
    "split_order",
    "canary",
    "determinism",
    "metric",
    "test_set_uses",
    "waiver",
)

# Only these checks can be waived (Requirements 10.7, 11.1).
_WAIVABLE_CHECKS = frozenset({"row_overlap", "target_proxy", "temporal"})

_CANARY_CHECK = "canary"
_DETERMINISM_CHECK = "determinism"
_CANARY_GHOST = "The Peeking Spirit"
_DETERMINISM_GHOST = "The Shapeshifter"

_WAIVER_CHECK = "waiver"
_WAIVER_FIX = "remove this waiver from hex.toml if the finding no longer occurs"
_TEST_SET_USES_CHECK = "test_set_uses"
_TEST_SET_USES_LOCATION = "runs/"
_TEST_SET_USES_FIX = (
    "stop tuning against this test set: for a final number, use test rows that "
    "were not used to choose the model"
)

_SEVERITY_WIDTH = len(Severity.WARNING.value)


def assemble(
    result: DataAuditResult,
    canary: CanaryOutcome,
    config: HexConfig,
    *,
    mode: RunMode,
    run_id: str,
    created_at: str,
    version: str,
    test_set_uses: int,
) -> Verdict:
    """Combine the data checks, the canary outcome and the waivers into a verdict.

    An input error gives a ``REJECTED`` verdict with that single finding and no
    check summaries (Requirement 4.7). Otherwise the findings of the checks, the
    metric warning and the canary are gathered, the ``test_set_uses`` warning is
    added when the count exceeds ``checks.max_test_uses`` (Requirement 12.2),
    waivers are applied to the data-check findings only (Requirements 10.7, 11.2),
    and the findings are sorted so that equal inputs give equal findings
    (Requirement 13.5).
    """
    split = SplitSignature(
        data_path=config.data.path,
        test_fraction=config.split.test_fraction,
        seed=config.split.seed,
        time_column=config.split.time_column,
    )
    if result.input_error is not None:
        metric = MetricResult(config.metric.name, None, config.metric.threshold, False)
        return Verdict(
            status=Status.REJECTED,
            mode=mode,
            app=config.app.name,
            run_id=run_id,
            created_at=created_at,
            hexperiment_version=version,
            metric=metric,
            checks=(),
            findings=(result.input_error,),
            waivers=(),
            test_set_uses=test_set_uses,
            split=split,
            data=result.data,
        )
    if result.metric is None:
        raise HexInputError(
            "the data audit result has no metric and no input error.",
            "Do not edit .hex/normal/result.json; run hexperiment again.",
        )
    data_findings = [f for check in result.checks for f in check.findings]
    data_findings.extend(result.extra_findings)
    waived, records = apply_waivers(data_findings, config.waivers)
    findings = [*waived, *canary.findings]
    if test_set_uses > config.checks.max_test_uses:
        findings.append(_test_set_uses_warning(test_set_uses, config))
    ordered = tuple(sorted(findings, key=_finding_key))
    return Verdict(
        status=decide_status(result.metric.passed, ordered, canary.outcome, mode),
        mode=mode,
        app=config.app.name,
        run_id=run_id,
        created_at=created_at,
        hexperiment_version=version,
        metric=result.metric,
        checks=_summaries(result, ordered, canary),
        findings=ordered,
        waivers=records,
        test_set_uses=test_set_uses,
        split=split,
        data=result.data,
    )


def apply_waivers(
    findings: Sequence[Finding], waivers: Sequence[Waiver]
) -> tuple[tuple[Finding, ...], tuple[WaiverRecord, ...]]:
    """Mark the findings that a waiver covers as ``waived`` (Requirement 11.2).

    A waiver matches a finding of its ``check`` and, when it names a ``column``,
    whose location is that column. Only ``row_overlap``, ``target_proxy`` and
    ``temporal`` findings are candidates, so a canary finding is never touched
    (Requirement 10.7). A matched finding takes the reason of the first waiver
    that matches it. Each waiver that matches yields a ``WaiverRecord``; a waiver
    that matches nothing yields a ``waiver`` warning instead, located at
    ``<check>`` or ``<check>:<column>``, appended after the findings.
    """
    result = list(findings)
    records: list[WaiverRecord] = []
    stale: list[Finding] = []
    for waiver in waivers:
        matched = [
            i
            for i, f in enumerate(findings)
            if _is_candidate(f) and _matches(f, waiver)
        ]
        if not matched:
            stale.append(_stale_waiver_warning(waiver))
            continue
        records.append(
            WaiverRecord(waiver.check, waiver.column, waiver.reason, len(matched))
        )
        for i in matched:
            result[i] = _waived(result[i], waiver)
    return tuple(result) + tuple(stale), tuple(records)


def decide_status(
    metric_passed: bool,
    findings: Sequence[Finding],
    canary: Outcome,
    mode: RunMode,
) -> Status:
    """``REJECTED``, ``APPROVED`` or ``UNVERIFIED`` (Requirement 13.1).

    Rejected when the metric fails or any finding is an ``error`` (waived findings
    do not count). Otherwise approved when a full run's canary passed. Otherwise
    unverified: a fast run, or a canary that was skipped, inconclusive or not
    applied.
    """
    if not metric_passed or any(f.severity is Severity.ERROR for f in findings):
        return Status.REJECTED
    if mode is RunMode.FULL and canary is Outcome.PASSED:
        return Status.APPROVED
    return Status.UNVERIFIED


def to_json(verdict: Verdict) -> str:
    """Return ``verdict.json``: sorted keys, two-space indent, no NaN (Req. 13.4)."""
    return (
        json.dumps(
            _without_non_finite(verdict.to_dict()),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def from_dict(raw: Mapping[str, Any]) -> Verdict:
    """Build a ``Verdict`` from parsed ``verdict.json``; the inverse of ``to_json``."""
    try:
        return Verdict.from_dict(raw)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise HexInputError(
            f"the verdict data is missing a field or has an invalid one ({error!r}).",
            "Do not edit verdict.json by hand; run hexperiment again to write it.",
        ) from None


def format_summary(verdict: Verdict) -> str:
    """The text the command prints: a status line, then one line per finding.

    Each finding line shows its severity, its ghost (the check name when it has
    none), its location and its fix (Requirement 16.4).
    """
    lines = [_status_line(verdict)]
    ghosts = [f.ghost or f.check for f in verdict.findings]
    locations = [_one_line(f.location) for f in verdict.findings]
    ghost_width = max(map(len, ghosts), default=0)
    location_width = max(map(len, locations), default=0)
    for finding, ghost, location in zip(
        verdict.findings, ghosts, locations, strict=True
    ):
        lines.append(
            f"  {finding.severity.value:<{_SEVERITY_WIDTH}}  "
            f"{ghost:<{ghost_width}}  {location:<{location_width}}  "
            f"{_one_line(finding.fix)}".rstrip()
        )
    return "\n".join(lines)


# --- helpers -------------------------------------------------------------------------


def _finding_key(finding: Finding) -> tuple[int, str, str]:
    try:
        rank = FINDING_ORDER.index(finding.check)
    except ValueError:
        rank = len(FINDING_ORDER)
    return (rank, finding.location, finding.message)


def _is_candidate(finding: Finding) -> bool:
    return finding.check in _WAIVABLE_CHECKS and finding.severity is not Severity.WAIVED


def _matches(finding: Finding, waiver: Waiver) -> bool:
    if finding.check != waiver.check:
        return False
    return waiver.column is None or finding.location == waiver.column


def _waived(finding: Finding, waiver: Waiver) -> Finding:
    if finding.severity is Severity.WAIVED:
        return finding
    return Finding(
        check=finding.check,
        ghost=finding.ghost,
        severity=Severity.WAIVED,
        location=finding.location,
        message=finding.message,
        evidence=finding.evidence,
        fix=finding.fix,
        waiver_reason=waiver.reason,
    )


def _stale_waiver_warning(waiver: Waiver) -> Finding:
    location = waiver.check
    if waiver.column is not None:
        location = f"{waiver.check}:{waiver.column}"
    return Finding(
        check=_WAIVER_CHECK,
        ghost=None,
        severity=Severity.WARNING,
        location=location,
        message=f"the waiver for {location} matched no finding",
        evidence={"check": waiver.check, "column": waiver.column},
        fix=_WAIVER_FIX,
    )


def _test_set_uses_warning(uses: int, config: HexConfig) -> Finding:
    return Finding(
        check=_TEST_SET_USES_CHECK,
        ghost=None,
        severity=Severity.WARNING,
        location=_TEST_SET_USES_LOCATION,
        message=(
            "the metric may be optimistic because the test set has been "
            f"evaluated {uses} times"
        ),
        evidence={"uses": uses, "max_test_uses": config.checks.max_test_uses},
        fix=_TEST_SET_USES_FIX,
    )


def _summaries(
    result: DataAuditResult, findings: Sequence[Finding], canary: CanaryOutcome
) -> tuple[CheckSummary, ...]:
    """One summary per check, with the outcome that remains after waivers."""
    summaries = [
        CheckSummary(
            c.name, c.ghost, _outcome_after_waivers(c.name, c.outcome, findings)
        )
        for c in result.checks
    ]
    summaries.append(CheckSummary(_CANARY_CHECK, _CANARY_GHOST, canary.outcome))
    summaries.append(
        CheckSummary(_DETERMINISM_CHECK, _DETERMINISM_GHOST, canary.determinism)
    )
    return tuple(summaries)


def _outcome_after_waivers(
    name: str, outcome: Outcome, findings: Sequence[Finding]
) -> Outcome:
    """A failed check is ``passed`` once every one of its errors is waived."""
    if outcome is not Outcome.FAILED:
        return outcome
    unwaived = any(
        f.check == name and f.severity is Severity.ERROR for f in findings
    )
    return Outcome.FAILED if unwaived else Outcome.PASSED


def _status_line(verdict: Verdict) -> str:
    metric = verdict.metric
    value = "n/a" if metric.value is None else f"{metric.value:.4f}"
    relation = ">=" if metric.passed else "<"
    return (
        f"{verdict.status.value} ({verdict.mode.value})  "
        f"{metric.name} {value} {relation} {metric.threshold:.4f}  "
        f"run {verdict.run_id}"
    )


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _without_non_finite(value: JsonValue) -> JsonValue:
    """Replace NaN and infinite floats by ``None`` so the JSON has no NaN."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _without_non_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_without_non_finite(item) for item in value]
    return value
