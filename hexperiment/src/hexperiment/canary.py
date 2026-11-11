"""The canary test: snapshots, their comparison and the canary decision.

A snapshot records what training saw in one execution of ``train.py``: the train
and test row ids, a hash of ``train[features]`` per training row and, when given,
the training predictions. The canary execution scrambles the test rows at load
time; in a leak-free pipeline its snapshot equals the normal one (Requirement 10).

Layer 1 (snapshots, comparison, the canary decision) is pure: it has no file,
process or clock access.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hexperiment.config import ExecMode
from hexperiment.errors import HexInputError
from hexperiment.models import (
    CheckResult,
    DataAuditResult,
    Finding,
    JsonValue,
    Outcome,
    RunMode,
    Severity,
)

# Absolute tolerance on y_score_train; row hashes must match exactly
# (Requirement 10.3).
SCORE_TOLERANCE = 1e-9

# The parent always executes this script, so canary findings point at it.
SCRIPT_NAME = "train.py"

# Check names of the findings made here. Neither can be named by a waiver, so
# these findings keep the severity they are created with (Requirement 10.7).
CANARY_CHECK = "canary"
DETERMINISM_CHECK = "determinism"
PEEKING_SPIRIT = "The Peeking Spirit"
SHAPESHIFTER = "The Shapeshifter"

_SPLIT_ORDER_CHECK = "split_order"
_LINE_LOCATION = re.compile(r".+:\d+")
_EVIDENCE_DECIMALS = 6


@dataclass(frozen=True, slots=True)
class Snapshot:
    """What training saw in one execution. Arrays are aligned with ``train_ids``."""

    mode: ExecMode
    run_id: str | None
    features: tuple[str, ...]
    train_ids: NDArray[np.int64]  # sorted ascending
    test_ids: NDArray[np.int64]  # sorted ascending
    train_row_hash: NDArray[np.uint64]
    y_score_train: NDArray[np.float64] | None


@dataclass(frozen=True, slots=True)
class LoadMarker:
    """Proof that an execution loaded its data with ``hexperiment.load``."""

    mode: ExecMode
    run_id: str
    scrambled_ids: NDArray[np.int64] | None  # ids only in canary mode


@dataclass(frozen=True, slots=True)
class SnapshotDiff:
    ids_equal: bool
    rows_changed: int  # hash differs OR score differs by more than the tolerance
    feature_rows_changed: int
    score_rows_changed: int
    max_abs_change: float | None  # None when either snapshot has no y_score_train

    @property
    def differs(self) -> bool:
        return not self.ids_equal or self.rows_changed > 0


@dataclass(frozen=True, slots=True)
class CanaryOutcome:
    outcome: Outcome  # canary check outcome
    determinism: Outcome  # determinism check outcome
    findings: tuple[Finding, ...]


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    result: DataAuditResult
    canary: CanaryOutcome
    mode: RunMode


# --- Layer 1: pure ------------------------------------------------------------------


def build_snapshot(
    mode: ExecMode,
    run_id: str | None,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    y_score_train: NDArray[np.float64] | None,
) -> Snapshot:
    """Record what training saw, with rows sorted by id (Requirement 2.3).

    The hash of ``train[features]`` ignores the index, so it depends on values
    only. ``y_score_train`` is matched to ``train`` by position and reordered with
    it, so reordering the training rows never changes the snapshot.
    """
    train_ids = _row_ids(train, "train")
    test_ids = _row_ids(test, "test")
    order = np.argsort(train_ids, kind="stable")
    hashes = pd.util.hash_pandas_object(train[list(features)], index=False)
    scores = None
    if y_score_train is not None:
        scores = np.asarray(y_score_train, dtype=np.float64)
        if scores.shape != (len(train),):
            raise HexInputError(
                f"y_score_train has shape {scores.shape}, but train has "
                f"{len(train)} rows.",
                "Pass one predicted probability per training row, in the order "
                "of train.",
            )
        scores = scores[order]
    return Snapshot(
        mode=mode,
        run_id=run_id,
        features=tuple(features),
        train_ids=train_ids[order],
        test_ids=np.sort(test_ids, kind="stable"),
        train_row_hash=hashes.to_numpy(dtype=np.uint64)[order],
        y_score_train=scores,
    )


def compare_snapshots(
    a: Snapshot, b: Snapshot, tolerance: float = SCORE_TOLERANCE
) -> SnapshotDiff:
    """Count the training rows that differ between two snapshots.

    Row ids are compared first; rows can only be compared when the id sets are
    equal. A row changed when its hash differs, or when both snapshots have
    scores and its score differs by more than ``tolerance`` (Requirement 10.3).
    """
    if not np.array_equal(a.train_ids, b.train_ids):
        return SnapshotDiff(
            ids_equal=False,
            rows_changed=0,
            feature_rows_changed=0,
            score_rows_changed=0,
            max_abs_change=None,
        )
    feature_changed = a.train_row_hash != b.train_row_hash
    score_changed = np.zeros_like(feature_changed)
    max_abs_change = None
    if a.y_score_train is not None and b.y_score_train is not None:
        delta = np.abs(a.y_score_train - b.y_score_train)
        # Written as "not within tolerance" so that a NaN counts as a change.
        score_changed = ~(delta <= tolerance)
        max_abs_change = float(np.max(delta)) if delta.size else 0.0
    return SnapshotDiff(
        ids_equal=True,
        rows_changed=int(np.count_nonzero(feature_changed | score_changed)),
        feature_rows_changed=int(np.count_nonzero(feature_changed)),
        score_rows_changed=int(np.count_nonzero(score_changed)),
        max_abs_change=max_abs_change,
    )


def decide_canary(
    normal: Snapshot,
    canary: Snapshot,
    marker: LoadMarker | None,
    run_control: Callable[[], Snapshot],
    hint_locations: Sequence[str],
) -> CanaryOutcome:
    """Decide the canary and determinism outcomes from the snapshots.

    The steps are evaluated in order and the first that applies decides.
    ``run_control`` is called at most once, and only at step 5.
    """
    # Step 1: the canary execution never called hexperiment.load (Requirement 10.6).
    if marker is None:
        return CanaryOutcome(
            Outcome.NOT_APPLIED, Outcome.SKIPPED, (_not_loaded_warning(),)
        )
    # Step 2: scrambled rows ended up in train (the script did not split with
    # hexperiment.split), so a difference would not show that test rows leaked.
    scrambled_train_rows = _scrambled_in_train(marker, canary)
    if scrambled_train_rows:
        warning = _scrambled_in_train_warning(scrambled_train_rows)
        return CanaryOutcome(Outcome.NOT_APPLIED, Outcome.SKIPPED, (warning,))
    # Step 3: the training rows changed, so the split depends on data values
    # (Requirement 10.2).
    if not np.array_equal(normal.train_ids, canary.train_ids):
        return CanaryOutcome(
            Outcome.SKIPPED, Outcome.SKIPPED, (_ids_differ_warning(normal, canary),)
        )
    # Requirement 2.6: without training predictions only the hashes are compared.
    weaker = () if normal.y_score_train is not None else (_weaker_warning(),)
    # Step 4: scrambling the test rows changed nothing in training.
    canary_diff = compare_snapshots(normal, canary)
    if not canary_diff.differs:
        return CanaryOutcome(Outcome.PASSED, Outcome.PASSED, weaker)
    # Steps 5 and 6: a difference; a control execution tells a leak apart from
    # non-deterministic training (Requirement 10.3).
    control_diff = compare_snapshots(normal, run_control())
    if not control_diff.differs:
        # Step 5: the control matches, so the test rows influenced training
        # (Requirement 10.5).
        spirit = _peeking_spirit(len(normal.train_ids), canary_diff, hint_locations)
        return CanaryOutcome(Outcome.FAILED, Outcome.PASSED, (spirit, *weaker))
    # Step 6: the control differs too, so training is not deterministic
    # (Requirement 10.4).
    shapeshifter = _shapeshifter(len(normal.train_ids), control_diff)
    return CanaryOutcome(
        Outcome.INCONCLUSIVE, Outcome.FAILED, (shapeshifter, *weaker)
    )


def hint_locations(result: DataAuditResult) -> tuple[str, ...]:
    """Return the ``<script>:<line>`` locations of the split-order hint, in order.

    These are the suspect lines listed in the fix of The Peeking Spirit
    (Requirement 10.5). Findings without a line number are left out.
    """
    locations: list[str] = []
    for check in result.checks:
        if check.name == _SPLIT_ORDER_CHECK:
            locations.extend(_line_locations(check))
    return tuple(dict.fromkeys(locations))


def skipped_outcome() -> CanaryOutcome:
    """The outcome of a run without a canary execution (a fast run)."""
    return CanaryOutcome(Outcome.SKIPPED, Outcome.SKIPPED, ())


# --- helpers ------------------------------------------------------------------------


def _row_ids(frame: pd.DataFrame, name: str) -> NDArray[np.int64]:
    if not pd.api.types.is_integer_dtype(frame.index.dtype):
        raise HexInputError(
            f"the index of {name} has dtype {frame.index.dtype}, but row ids must "
            "be integers.",
            "Keep the index set by hexperiment.load; do not reset or replace it "
            "before calling hexperiment.audit.",
        )
    return frame.index.to_numpy(dtype=np.int64)


def _line_locations(check: CheckResult) -> list[str]:
    return [
        f.location
        for f in check.findings
        if f.check == _SPLIT_ORDER_CHECK and _LINE_LOCATION.fullmatch(f.location)
    ]


def _scrambled_in_train(marker: LoadMarker, canary: Snapshot) -> int:
    # A canary marker always carries the scrambled ids; None means none.
    if marker.scrambled_ids is None:
        return 0
    return int(np.count_nonzero(np.isin(canary.train_ids, marker.scrambled_ids)))


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, _EVIDENCE_DECIMALS)


def _warning(message: str, fix: str, evidence: dict[str, JsonValue]) -> Finding:
    return Finding(
        check=CANARY_CHECK,
        ghost=None,
        severity=Severity.WARNING,
        location=SCRIPT_NAME,
        message=message,
        evidence=evidence,
        fix=fix,
    )


def _not_loaded_warning() -> Finding:
    return _warning(
        "the canary test was not applied: the script did not load its data with "
        "hexperiment.load, so the test rows could not be scrambled",
        "load the data with hexperiment.load() and split it with "
        "hexperiment.split(df)",
        {},
    )


def _scrambled_in_train_warning(rows: int) -> Finding:
    return _warning(
        f"the canary test was not applied: {rows} scrambled test rows ended up in "
        "the training set",
        "split with hexperiment.split(df), not with your own split",
        {"scrambled_train_rows": rows},
    )


def _ids_differ_warning(normal: Snapshot, canary: Snapshot) -> Finding:
    return _warning(
        "the canary test is inconclusive because the split depends on data "
        "values: scrambling the test rows changed which rows are in train",
        "split with hexperiment.split(df) and do not filter rows by value "
        "before the split",
        {
            "normal_train_rows": len(normal.train_ids),
            "canary_train_rows": len(canary.train_ids),
        },
    )


def _weaker_warning() -> Finding:
    return _warning(
        "the canary test is weaker without training predictions: only the "
        "training features were compared",
        "pass y_score_train (the predicted probabilities for the training rows) "
        "to hexperiment.audit",
        {},
    )


def _peeking_spirit(
    train_rows: int, diff: SnapshotDiff, locations: Sequence[str]
) -> Finding:
    fix = (
        "test rows influenced training; fit preprocessing on training rows only, "
        "after the split."
    )
    if locations:
        fix += f" Suspect lines: {', '.join(locations)}."
    return Finding(
        check=CANARY_CHECK,
        ghost=PEEKING_SPIRIT,
        severity=Severity.ERROR,
        location=SCRIPT_NAME,
        message=(
            f"scrambling the test rows changed {diff.rows_changed} of "
            f"{train_rows} training rows"
        ),
        evidence={
            "train_rows": train_rows,
            "rows_changed": diff.rows_changed,
            "feature_rows_changed": diff.feature_rows_changed,
            "largest_change": _round(diff.max_abs_change),
        },
        fix=fix,
    )


def _shapeshifter(train_rows: int, diff: SnapshotDiff) -> Finding:
    if diff.ids_equal:
        changed = f"{diff.rows_changed} of {train_rows} training rows"
    else:
        changed = "which rows are in train"
    return Finding(
        check=DETERMINISM_CHECK,
        ghost=SHAPESHIFTER,
        severity=Severity.WARNING,
        location=SCRIPT_NAME,
        message=(
            f"training is not deterministic: an identical execution changed "
            f"{changed}"
        ),
        evidence={
            "rows_changed": diff.rows_changed,
            "largest_change": _round(diff.max_abs_change),
        },
        fix="training is not deterministic; set random_state everywhere",
    )
