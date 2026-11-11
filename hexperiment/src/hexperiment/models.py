"""Shared enums and frozen dataclasses.

No logic beyond ``to_dict`` / ``from_dict``. ``to_dict`` returns JSON-ready values
(tuples become lists, enums become strings). ``from_dict`` is its inverse and raises
``KeyError``, ``TypeError`` or ``ValueError`` on malformed input; callers that read
files turn those into ``HexInputError`` with the file name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)


class Status(StrEnum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    UNVERIFIED = "UNVERIFIED"


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    WAIVED = "waived"


class RunMode(StrEnum):
    FAST = "fast"
    FULL = "full"


class Outcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    INCONCLUSIVE = "inconclusive"
    NOT_APPLIED = "not applied"
    NOT_APPLICABLE = "not applicable"


def _time_range_to_list(value: tuple[str, str] | None) -> list[str] | None:
    return None if value is None else [value[0], value[1]]


def _time_range_from_list(value: Any) -> tuple[str, str] | None:
    if value is None:
        return None
    start, end = value
    return (str(start), str(end))


@dataclass(frozen=True, slots=True)
class Finding:
    check: str
    ghost: str | None
    severity: Severity
    location: str
    message: str
    evidence: dict[str, JsonValue]
    fix: str
    waiver_reason: str | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        data: dict[str, JsonValue] = {
            "check": self.check,
            "evidence": dict(self.evidence),
            "fix": self.fix,
            "ghost": self.ghost,
            "location": self.location,
            "message": self.message,
            "severity": self.severity.value,
        }
        # waiver_reason appears in the JSON only for waived findings.
        if self.severity is Severity.WAIVED:
            data["waiver_reason"] = self.waiver_reason
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Finding:
        severity = Severity(data["severity"])
        waived = severity is Severity.WAIVED
        return cls(
            check=data["check"],
            ghost=data["ghost"],
            severity=severity,
            location=data["location"],
            message=data["message"],
            evidence=dict(data["evidence"]),
            fix=data["fix"],
            waiver_reason=data.get("waiver_reason") if waived else None,
        )


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    ghost: str | None
    outcome: Outcome
    findings: tuple[Finding, ...] = ()

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "ghost": self.ghost,
            "name": self.name,
            "outcome": self.outcome.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CheckResult:
        return cls(
            name=data["name"],
            ghost=data["ghost"],
            outcome=Outcome(data["outcome"]),
            findings=tuple(Finding.from_dict(f) for f in data["findings"]),
        )


@dataclass(frozen=True, slots=True)
class CheckSummary:
    name: str
    ghost: str | None
    outcome: Outcome

    def to_dict(self) -> dict[str, JsonValue]:
        return {"ghost": self.ghost, "name": self.name, "outcome": self.outcome.value}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CheckSummary:
        return cls(
            name=data["name"], ghost=data["ghost"], outcome=Outcome(data["outcome"])
        )


@dataclass(frozen=True, slots=True)
class MetricResult:
    name: str
    value: float | None
    threshold: float
    passed: bool

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "passed": self.passed,
            "threshold": self.threshold,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MetricResult:
        value = data["value"]
        return cls(
            name=data["name"],
            value=None if value is None else float(value),
            threshold=float(data["threshold"]),
            passed=bool(data["passed"]),
        )


@dataclass(frozen=True, slots=True)
class WaiverRecord:
    check: str
    column: str | None
    reason: str
    findings_waived: int

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "check": self.check,
            "column": self.column,
            "findings_waived": self.findings_waived,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WaiverRecord:
        return cls(
            check=data["check"],
            column=data["column"],
            reason=data["reason"],
            findings_waived=int(data["findings_waived"]),
        )


@dataclass(frozen=True, slots=True)
class SplitSignature:
    data_path: str
    test_fraction: float
    seed: int
    time_column: str | None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "data_path": self.data_path,
            "seed": self.seed,
            "test_fraction": self.test_fraction,
            "time_column": self.time_column,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SplitSignature:
        return cls(
            data_path=data["data_path"],
            test_fraction=float(data["test_fraction"]),
            seed=int(data["seed"]),
            time_column=data["time_column"],
        )


@dataclass(frozen=True, slots=True)
class DataSummary:
    n_train: int
    n_test: int
    positive_rate_train: float
    positive_rate_test: float
    train_time_range: tuple[str, str] | None
    test_time_range: tuple[str, str] | None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "n_test": self.n_test,
            "n_train": self.n_train,
            "positive_rate_test": self.positive_rate_test,
            "positive_rate_train": self.positive_rate_train,
            "test_time_range": _time_range_to_list(self.test_time_range),
            "train_time_range": _time_range_to_list(self.train_time_range),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DataSummary:
        return cls(
            n_train=int(data["n_train"]),
            n_test=int(data["n_test"]),
            positive_rate_train=float(data["positive_rate_train"]),
            positive_rate_test=float(data["positive_rate_test"]),
            train_time_range=_time_range_from_list(data["train_time_range"]),
            test_time_range=_time_range_from_list(data["test_time_range"]),
        )


@dataclass(frozen=True, slots=True)
class DataAuditResult:
    """What the normal execution writes to ``result.json``."""

    run_id: str | None
    metric: MetricResult | None  # None only when input_error is set
    checks: tuple[CheckResult, ...]
    extra_findings: tuple[Finding, ...]  # metric warning
    data: DataSummary | None
    input_error: Finding | None

    def to_dict(self) -> dict[str, JsonValue]:
        error = self.input_error
        return {
            "checks": [c.to_dict() for c in self.checks],
            "data": None if self.data is None else self.data.to_dict(),
            "extra_findings": [f.to_dict() for f in self.extra_findings],
            "input_error": None if error is None else error.to_dict(),
            "metric": None if self.metric is None else self.metric.to_dict(),
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DataAuditResult:
        metric = data["metric"]
        summary = data["data"]
        input_error = data["input_error"]
        return cls(
            run_id=data["run_id"],
            metric=None if metric is None else MetricResult.from_dict(metric),
            checks=tuple(CheckResult.from_dict(c) for c in data["checks"]),
            extra_findings=tuple(Finding.from_dict(f) for f in data["extra_findings"]),
            data=None if summary is None else DataSummary.from_dict(summary),
            input_error=None if input_error is None else Finding.from_dict(input_error),
        )


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """What one ``hexperiment run`` produced, as text instead of printed output.

    ``exit_code`` is 0, 1, 2 or 3 as in the command. ``status`` is ``None`` when
    ``exit_code`` is 2 (no verdict was written). ``stdout`` is the summary of the
    verdict, ``stderr`` the failure message of a run that failed.
    """

    exit_code: int
    status: Status | None
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class Verdict:
    status: Status
    mode: RunMode
    app: str
    run_id: str
    created_at: str
    hexperiment_version: str
    metric: MetricResult
    checks: tuple[CheckSummary, ...]
    findings: tuple[Finding, ...]
    waivers: tuple[WaiverRecord, ...]
    test_set_uses: int
    split: SplitSignature
    data: DataSummary | None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "app": self.app,
            "checks": [c.to_dict() for c in self.checks],
            "created_at": self.created_at,
            "data": None if self.data is None else self.data.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "hexperiment_version": self.hexperiment_version,
            "metric": self.metric.to_dict(),
            "mode": self.mode.value,
            "run_id": self.run_id,
            "split": self.split.to_dict(),
            "status": self.status.value,
            "test_set_uses": self.test_set_uses,
            "waivers": [w.to_dict() for w in self.waivers],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Verdict:
        summary = data["data"]
        return cls(
            status=Status(data["status"]),
            mode=RunMode(data["mode"]),
            app=data["app"],
            run_id=data["run_id"],
            created_at=data["created_at"],
            hexperiment_version=data["hexperiment_version"],
            metric=MetricResult.from_dict(data["metric"]),
            checks=tuple(CheckSummary.from_dict(c) for c in data["checks"]),
            findings=tuple(Finding.from_dict(f) for f in data["findings"]),
            waivers=tuple(WaiverRecord.from_dict(w) for w in data["waivers"]),
            test_set_uses=int(data["test_set_uses"]),
            split=SplitSignature.from_dict(data["split"]),
            data=None if summary is None else DataSummary.from_dict(summary),
        )
