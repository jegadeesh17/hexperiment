"""The committed run history shows the haunted, flagged and honest steps in order.

Requirement 5.3. The history comes from real runs of the ``exorcise`` workflow, so
until it exists the checks on the real apps skip. ``problems`` is the checker; the
other tests run it on small synthetic histories in a temporary folder, so a bug in
the checker shows up before the recording, not during it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from conftest import FLIGHTS, HOTELS, App
from hexperiment.models import (
    CheckSummary,
    Finding,
    MetricResult,
    Outcome,
    RunMode,
    Severity,
    SplitSignature,
    Status,
    Verdict,
)
from hexperiment.runs import CARD_FILENAME, RUNS_DIRNAME, VERDICT_FILENAME
from hexperiment.verdict import to_json

import hexperiment

PEEKING = "The Peeking Spirit"


@dataclass(frozen=True, slots=True)
class Arc:
    """What the history of an app must show."""

    first_run_ghosts: tuple[str, ...]  # in the first (haunted) run
    some_run_ghost: str | None  # in at least one run (the step after a partial fix)


ARCS = {
    HOTELS: Arc(("The Possession", PEEKING), None),
    FLIGHTS: Arc(("The Prophet",), PEEKING),
}


def ghosts_of(verdict: Verdict) -> set[str]:
    return {f.ghost for f in verdict.findings if f.ghost is not None}


def problems(app_dir: Path, arc: Arc) -> list[str]:
    """Return what is wrong with the history of ``app_dir``; empty when it is right."""
    history = hexperiment.load_history(app_dir)
    if len(history) < 2:
        return [f"the history has {len(history)} runs; it needs at least 2"]
    found: list[str] = []
    first, last = history[0], history[-1]
    if first.status is not Status.REJECTED:
        found.append(f"the first run is {first.status.value}, not REJECTED")
    missing = set(arc.first_run_ghosts) - ghosts_of(first)
    if missing:
        found.append(f"the first run lacks the ghosts {sorted(missing)}")
    if last.status is not Status.APPROVED:
        found.append(f"the last run is {last.status.value}, not APPROVED")
    if arc.some_run_ghost and not any(
        arc.some_run_ghost in ghosts_of(v) for v in history
    ):
        found.append(f"no run shows {arc.some_run_ghost}")
    statuses = [v.status for v in history]
    if Status.APPROVED in statuses:
        after = statuses[statuses.index(Status.APPROVED) :]
        if Status.REJECTED in after:
            found.append("a REJECTED run comes after an APPROVED run")
    runs_dir = app_dir / RUNS_DIRNAME
    for verdict in history:
        if not (runs_dir / verdict.run_id / CARD_FILENAME).is_file():
            found.append(f"run {verdict.run_id} has no {CARD_FILENAME}")
    latest = app_dir / VERDICT_FILENAME
    last_file = runs_dir / last.run_id / VERDICT_FILENAME
    if not latest.is_file():
        found.append(f"{VERDICT_FILENAME} is missing in the app folder")
    elif latest.read_bytes() != last_file.read_bytes():
        found.append(f"{VERDICT_FILENAME} differs from the last run's verdict")
    return found


# --- the real apps ---------------------------------------------------------------


def test_the_committed_history_shows_the_arc(app: App) -> None:
    if not (app.folder / RUNS_DIRNAME).is_dir():
        pytest.skip(
            f"{app.name} has no runs/ folder yet; the exorcise workflow records it"
        )
    assert problems(app.folder, ARCS[app.name]) == []


# --- the checker on synthetic histories --------------------------------------------


def finding(ghost: str) -> Finding:
    return Finding(
        check="synthetic",
        ghost=ghost,
        severity=Severity.ERROR,
        location="train.py",
        message="synthetic",
        evidence={},
        fix="synthetic",
    )


def make_verdict(index: int, status: Status, ghosts: tuple[str, ...]) -> Verdict:
    return Verdict(
        status=status,
        mode=RunMode.FULL,
        app="synthetic",
        run_id=f"20261111-1200{index:02d}-{index:06x}",
        created_at=f"2026-11-11T12:00:{index:02d}Z",
        hexperiment_version="0",
        metric=MetricResult("roc_auc", 0.7, 0.6, True),
        checks=(CheckSummary("synthetic", None, Outcome.PASSED),),
        findings=tuple(finding(ghost) for ghost in ghosts),
        waivers=(),
        test_set_uses=index + 1,
        split=SplitSignature("data/prepared.csv", 0.2, 0, None),
        data=None,
    )


def write_history(
    app_dir: Path,
    steps: list[tuple[Status, tuple[str, ...]]],
    *,
    cards: bool = True,
    copy_last: bool = True,
) -> None:
    """Write one run folder per step, as ``hexperiment run`` would."""
    text = ""
    for index, (status, ghosts) in enumerate(steps):
        verdict = make_verdict(index, status, ghosts)
        run_dir = app_dir / RUNS_DIRNAME / verdict.run_id
        run_dir.mkdir(parents=True)
        text = to_json(verdict)
        (run_dir / VERDICT_FILENAME).write_text(text, encoding="utf-8", newline="\n")
        if cards:
            (run_dir / CARD_FILENAME).write_text("card", encoding="utf-8")
    if copy_last:
        (app_dir / VERDICT_FILENAME).write_text(text, encoding="utf-8", newline="\n")


REJECTED, APPROVED = Status.REJECTED, Status.APPROVED
GOOD_FLIGHTS = [
    (REJECTED, ("The Prophet",)),
    (REJECTED, (PEEKING,)),
    (APPROVED, ()),
]


def test_a_good_flights_history_has_no_problems(tmp_path: Path) -> None:
    write_history(tmp_path, GOOD_FLIGHTS)
    assert problems(tmp_path, ARCS[FLIGHTS]) == []


def test_a_good_hotels_history_has_no_problems(tmp_path: Path) -> None:
    write_history(
        tmp_path,
        [
            (REJECTED, ("The Possession", PEEKING)),
            (REJECTED, (PEEKING,)),
            (APPROVED, ()),
        ],
    )
    assert problems(tmp_path, ARCS[HOTELS]) == []


def test_a_history_with_one_run_is_a_problem(tmp_path: Path) -> None:
    write_history(tmp_path, [(APPROVED, ())])
    assert any("at least 2" in p for p in problems(tmp_path, ARCS[FLIGHTS]))


def test_no_history_at_all_is_a_problem(tmp_path: Path) -> None:
    assert any("at least 2" in p for p in problems(tmp_path, ARCS[FLIGHTS]))


def test_a_first_run_that_is_not_the_haunted_one_is_a_problem(tmp_path: Path) -> None:
    write_history(tmp_path, [(REJECTED, (PEEKING,)), (APPROVED, ())])
    found = problems(tmp_path, ARCS[FLIGHTS])
    assert any("lacks the ghosts" in p for p in found)


def test_a_last_run_that_is_not_approved_is_a_problem(tmp_path: Path) -> None:
    write_history(tmp_path, GOOD_FLIGHTS[:2])
    assert any("not APPROVED" in p for p in problems(tmp_path, ARCS[FLIGHTS]))


def test_flights_needs_a_run_with_the_peeking_spirit(tmp_path: Path) -> None:
    write_history(tmp_path, [GOOD_FLIGHTS[0], GOOD_FLIGHTS[2]])
    assert any(PEEKING in p for p in problems(tmp_path, ARCS[FLIGHTS]))


def test_a_rejected_run_after_an_approved_run_is_a_problem(tmp_path: Path) -> None:
    steps = [*GOOD_FLIGHTS, (REJECTED, ("The Prophet",)), (APPROVED, ())]
    write_history(tmp_path, steps)
    assert any("after an APPROVED" in p for p in problems(tmp_path, ARCS[FLIGHTS]))


def test_a_missing_model_card_is_a_problem(tmp_path: Path) -> None:
    write_history(tmp_path, GOOD_FLIGHTS, cards=False)
    assert any(CARD_FILENAME in p for p in problems(tmp_path, ARCS[FLIGHTS]))


def test_an_app_verdict_that_is_not_the_last_run_is_a_problem(tmp_path: Path) -> None:
    write_history(tmp_path, GOOD_FLIGHTS)
    (tmp_path / VERDICT_FILENAME).write_text("{}", encoding="utf-8")
    assert any("differs" in p for p in problems(tmp_path, ARCS[FLIGHTS]))


def test_a_missing_app_verdict_is_a_problem(tmp_path: Path) -> None:
    write_history(tmp_path, GOOD_FLIGHTS, copy_last=False)
    assert any("missing" in p for p in problems(tmp_path, ARCS[FLIGHTS]))
