"""Turning run records into the results object (pure)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from hexbench.execution import RunRecord
from hexbench.mechanisms import LEAK_TYPE_NAMES, NotMeasured
from hexbench.spec import Mechanism

FORMAT = "hexperiment-benchmark/1"
SOURCE = (
    "Kapoor and Narayanan, Leakage and the reproducibility crisis in "
    "machine-learning-based science, Patterns 4(9), 100804, 2023, "
    "doi:10.1016/j.patter.2023.100804"
)
STATUSES = ("APPROVED", "REJECTED", "UNVERIFIED", "ERROR")
SCORE_DECIMALS = 4
SECONDS_DECIMALS = 1


@dataclass(frozen=True, slots=True)
class CaseRecord:
    case_id: str
    mechanism: str  # mechanism key
    model: str
    data_seed: int
    leaky: RunRecord
    clean: RunRecord


def build_results(
    records: Sequence[CaseRecord],
    *,
    mechanisms: Sequence[Mechanism],
    not_measured: Sequence[NotMeasured],
    environment: Mapping[str, object],
    created_at: str,
    only: str | None,
) -> dict[str, object]:
    """The results object of ``hexperiment-benchmark/1``.

    Mechanisms without any record are left out, so a file written with ``only``
    has one item.
    """
    items = [
        _mechanism_item(m, [r for r in records if r.mechanism == m.key])
        for m in mechanisms
        if any(r.mechanism == m.key for r in records)
    ]
    return {
        "format": FORMAT,
        "created_at": created_at,
        "complete": only is None,
        "only": only,
        "environment": dict(environment),
        "source": SOURCE,
        "mechanisms": items,
        "clean": _clean_item([r.clean for r in records]),
        "not_measured": [
            {
                "leak_type": n.leak_type,
                "leak_type_name": n.leak_type_name,
                "reason": n.reason,
            }
            for n in not_measured
        ],
        "cases": [_case_item(r) for r in records],
    }


def _mechanism_item(
    mechanism: Mechanism, records: Sequence[CaseRecord]
) -> dict[str, object]:
    leaky = [r.leaky for r in records]
    statuses = _statuses(leaky)
    return {
        "key": mechanism.key,
        "leak_type": mechanism.leak_type,
        "leak_type_name": LEAK_TYPE_NAMES[mechanism.leak_type],
        "mechanism": mechanism.mechanism,
        "designed_to_catch": list(mechanism.designed_to_catch),
        "scripts": len(records),
        "rejected": statuses["REJECTED"],
        "by_ghost": _by_ghost(leaky),
        "statuses": statuses,
        "mean_score_leaky": _mean_score(leaky),
        "mean_score_clean": _mean_score([r.clean for r in records]),
    }


def _clean_item(runs: Sequence[RunRecord]) -> dict[str, object]:
    statuses = _statuses(runs)
    return {
        "scripts": len(runs),
        "false_alarms": statuses["REJECTED"],
        "by_ghost": _by_ghost(runs),
        "statuses": statuses,
    }


def _statuses(runs: Sequence[RunRecord]) -> dict[str, int]:
    counts = Counter(run.status for run in runs)
    unknown = set(counts) - set(STATUSES)
    if unknown:
        raise ValueError(f"unknown run status {sorted(unknown)!r}")
    return {status: counts.get(status, 0) for status in STATUSES}


def _by_ghost(runs: Sequence[RunRecord]) -> dict[str, int]:
    """Rejected scripts per ghost or check name; a script counts once per key."""
    counts: Counter[str] = Counter()
    for run in runs:
        if run.status == "REJECTED":
            counts.update(set(run.ghosts))
    return dict(sorted(counts.items()))


def _mean_score(runs: Sequence[RunRecord]) -> float | None:
    scores = [run.score for run in runs if run.score is not None]
    if not scores:
        return None
    return round(sum(scores) / len(scores), SCORE_DECIMALS)


def _run_item(run: RunRecord) -> dict[str, object]:
    return {
        "status": run.status,
        "score": run.score,
        "ghosts": list(run.ghosts),
        "warnings": list(run.warnings),
        "seconds": round(run.seconds, SECONDS_DECIMALS),
        "message": run.message,
    }


def _case_item(record: CaseRecord) -> dict[str, object]:
    return {
        "id": record.case_id,
        "mechanism": record.mechanism,
        "model": record.model,
        "data_seed": record.data_seed,
        "leaky": _run_item(record.leaky),
        "clean": _run_item(record.clean),
    }
