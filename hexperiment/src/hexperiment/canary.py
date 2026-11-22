"""The canary test: snapshots, their comparison and the canary decision.

A snapshot records what training saw in one execution of ``train.py``: the train
and test row ids, a hash of ``train[features]`` per training row and, when given,
the training predictions. The canary execution scrambles the test rows at load
time; in a leak-free pipeline its snapshot equals the normal one (Requirement 10).

Layer 1 (snapshots, comparison, the canary decision) is pure: it has no file,
process or clock access. Layer 2 is the ``.hex/`` exchange protocol: the files
that a training execution leaves for the parent process, and the readers that
refuse files from another run or another format. Layer 3 starts ``train.py`` and
orchestrates the executions of one run (Requirements 10.1, 16.2, 16.3).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hexperiment.config import ENV_MODE, ENV_RUN_ID, ExecMode, HexConfig
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
_PASS_SCORES_FIX = (
    "pass y_score_train=model.predict_proba(train[features])[:, 1] to "
    "hexperiment.audit"
)
_LINE_LOCATION = re.compile(r".+:\d+")
_EVIDENCE_DECIMALS = 6

# The .hex/ exchange protocol (Requirement 2.3). Every file carries a run id and
# this format version; a reader rejects any other version.
FORMAT_VERSION = 1
_HEX_DIRNAME = ".hex"
_META_FILE = "meta.json"
_SNAPSHOT_FILE = "snapshot.npz"
_MARKER_FILE = "loaded.json"
_SCRAMBLED_FILE = "scrambled_ids.npy"
_RESULT_FILE = "result.json"


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
    # Digest of ``test[target]`` in row-id order; ``None`` when the test frame has
    # no target column or the caller gave no target name (Requirement 10.7).
    test_target_hash: str | None = None


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
    target: str | None = None,
) -> Snapshot:
    """Record what training saw, with rows sorted by id (Requirement 2.3).

    The hash of ``train[features]`` ignores the index, so it depends on values
    only. ``y_score_train`` is matched to ``train`` by position and reordered with
    it, so reordering the training rows never changes the snapshot. With a
    ``target`` name, the snapshot also holds a digest of ``test[target]``, which
    shows whether the scramble reached the audited test frame (Requirement 10.7).
    """
    train_ids = _row_ids(train, "train")
    test_ids = _row_ids(test, "test")
    _require_unique_ids(train_ids)
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
        test_target_hash=_test_target_hash(test, test_ids, target),
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
    # Requirement 10.6: a marker that is not a canary marker, or that records no
    # scrambled rows, does not show that any test row was scrambled.
    scrambled_ids = marker.scrambled_ids
    if (
        marker.mode != ExecMode.CANARY
        or scrambled_ids is None
        or len(scrambled_ids) == 0
    ):
        return CanaryOutcome(
            Outcome.NOT_APPLIED, Outcome.SKIPPED, (_bad_marker_warning(marker),)
        )
    # Step 2: scrambled rows ended up in train (the script did not split with
    # hexperiment.split), so a difference would not show that test rows leaked.
    scrambled_train_rows = int(
        np.count_nonzero(np.isin(canary.train_ids, scrambled_ids))
    )
    if scrambled_train_rows:
        warning = _scrambled_in_train_warning(scrambled_train_rows)
        return CanaryOutcome(Outcome.NOT_APPLIED, Outcome.SKIPPED, (warning,))
    # Requirement 10.7: the scramble changes the test target of every audited
    # test frame, so an equal digest shows that the audited frames did not come
    # from the scrambled load, and the comparison below would prove nothing.
    if _scramble_missed(normal, canary):
        return CanaryOutcome(
            Outcome.NOT_APPLIED, Outcome.SKIPPED, (_scramble_missed_warning(),)
        )
    # Step 3: the training rows changed, so the split depends on data values
    # (Requirement 10.2).
    if not np.array_equal(normal.train_ids, canary.train_ids):
        return CanaryOutcome(
            Outcome.SKIPPED, Outcome.SKIPPED, (_ids_differ_warning(normal, canary),)
        )
    return _decide_rows(normal, canary, run_control, hint_locations)


def _decide_rows(
    normal: Snapshot,
    canary: Snapshot,
    run_control: Callable[[], Snapshot],
    hint_locations: Sequence[str],
) -> CanaryOutcome:
    """Steps 4 to 6: compare the training rows of equal id sets."""
    # Step 4: scrambling the test rows changed nothing in training. A changed
    # feature list is a difference: features chosen on all rows (a leak) move when
    # only test rows are altered, so it goes to the control like any other.
    canary_diff = compare_snapshots(normal, canary)
    features_changed = normal.features != canary.features
    if not canary_diff.differs and not features_changed:
        warnings = _score_warnings({"normal": normal, "canary": canary})
        if _mixed_scores(normal, canary):
            # Requirements 2.6, 10.3: the predictions of one execution were not
            # compared, so equal hashes do not show that nothing changed.
            return CanaryOutcome(Outcome.INCONCLUSIVE, Outcome.INCONCLUSIVE, warnings)
        if normal.y_score_train is None:
            # Requirement 2.6: with no predictions in either execution only the
            # training features were compared, so a model fitted on test rows
            # is not ruled out.
            return CanaryOutcome(
                Outcome.INCONCLUSIVE, Outcome.PASSED, (_no_scores_warning(),)
            )
        return CanaryOutcome(Outcome.PASSED, Outcome.PASSED, warnings)
    # Steps 5 and 6: a difference; a control execution tells a leak apart from
    # non-deterministic training (Requirement 10.3).
    control = run_control()
    scores = _score_warnings({"normal": normal, "canary": canary, "control": control})
    train_rows = len(normal.train_ids)
    if normal.features != control.features:
        # Requirement 10.4: an identical execution used other features.
        shapeshifter = _features_shapeshifter(normal.features, control.features)
        return CanaryOutcome(
            Outcome.INCONCLUSIVE, Outcome.FAILED, (shapeshifter, *scores)
        )
    control_diff = compare_snapshots(normal, control)
    if not control_diff.differs:
        only_predictions = (
            canary_diff.feature_rows_changed == 0 and not features_changed
        )
        if only_predictions and _mixed_scores(normal, control):
            # Requirements 2.6, 10.3: only predictions changed, and the control
            # has no predictions to compare, so a leak cannot be told apart
            # from non-deterministic training.
            return CanaryOutcome(Outcome.INCONCLUSIVE, Outcome.INCONCLUSIVE, scores)
        # Step 5: the control matches, so the test rows influenced training
        # (Requirement 10.5).
        spirit = _peeking_spirit(
            train_rows,
            canary_diff,
            hint_locations,
            (normal.features, canary.features) if features_changed else None,
        )
        return CanaryOutcome(Outcome.FAILED, Outcome.PASSED, (spirit, *scores))
    # Step 6: the control differs too, so training is not deterministic
    # (Requirement 10.4).
    shapeshifter = _shapeshifter(train_rows, control_diff)
    return CanaryOutcome(
        Outcome.INCONCLUSIVE, Outcome.FAILED, (shapeshifter, *scores)
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


def _test_target_hash(
    test: pd.DataFrame, test_ids: NDArray[np.int64], target: str | None
) -> str | None:
    """Digest of ``test[target]`` in row-id order, or ``None`` without a target.

    The canary execution does not validate the test frame (Requirement 2.4), so a
    missing target column gives ``None`` instead of an error.
    """
    if target is None or target not in test.columns:
        return None
    order = np.argsort(test_ids, kind="stable")
    row_hashes = pd.util.hash_pandas_object(test[target], index=False)
    ordered = row_hashes.to_numpy(dtype=np.uint64)[order]
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def _scramble_missed(normal: Snapshot, canary: Snapshot) -> bool:
    """True when both executions audited a test target with the same digest."""
    return (
        normal.test_target_hash is not None
        and normal.test_target_hash == canary.test_target_hash
    )


def _require_unique_ids(train_ids: NDArray[np.int64]) -> None:
    # Requirement 2.3: one hash per training row, so every row id is unique.
    unique, counts = np.unique(train_ids, return_counts=True)
    repeated = unique[counts > 1]
    if repeated.size:
        raise HexInputError(
            f"{repeated.size} row ids appear more than once in train (for example "
            f"row id {int(repeated[0])}), but every training row must have its own "
            "row id.",
            "Pass the training rows returned by hexperiment.split to "
            "hexperiment.audit; oversample or resample only inside model fitting, "
            "not in the frame passed to audit.",
        )


def _line_locations(check: CheckResult) -> list[str]:
    return [
        f.location
        for f in check.findings
        if f.check == _SPLIT_ORDER_CHECK and _LINE_LOCATION.fullmatch(f.location)
    ]


def _mixed_scores(*snapshots: Snapshot) -> bool:
    return len({s.y_score_train is None for s in snapshots}) > 1


def _score_warnings(roles: Mapping[str, Snapshot]) -> tuple[Finding, ...]:
    """The warnings about training predictions (Requirement 2.6)."""
    supplied = [name for name, s in roles.items() if s.y_score_train is not None]
    if not supplied:
        return (_weaker_warning(),)
    if len(supplied) < len(roles):
        missing = [name for name in roles if name not in supplied]
        return (_mixed_scores_warning(supplied, missing),)
    return ()


def _feature_difference(a: Sequence[str], b: Sequence[str], b_name: str) -> str:
    only_a = [f for f in a if f not in b]
    only_b = [f for f in b if f not in a]
    parts = []
    if only_a:
        parts.append(f"only in normal: {', '.join(only_a)}")
    if only_b:
        parts.append(f"only in {b_name}: {', '.join(only_b)}")
    if not parts:
        return "the same names in another order or number"
    return "; ".join(parts)


def _round(value: float | None) -> float | None:
    # Evidence is JSON written without NaN or infinity, so those become None.
    if value is None or not math.isfinite(value):
        return None
    return round(value, _EVIDENCE_DECIMALS)


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


def _bad_marker_warning(marker: LoadMarker) -> Finding:
    ids = marker.scrambled_ids
    if marker.mode != ExecMode.CANARY:
        problem = f"its load marker was written in {marker.mode.value} mode"
    else:
        problem = "its load marker records no scrambled test rows"
    return _warning(
        f"the canary test was not applied: {problem}, so it does not show that "
        "the test rows were scrambled",
        "run the app with hexperiment run and load the data with "
        "hexperiment.load(), so that the canary execution scrambles the test rows",
        {
            "marker_mode": marker.mode.value,
            "scrambled_rows": None if ids is None else len(ids),
        },
    )


def _scrambled_in_train_warning(rows: int) -> Finding:
    return _warning(
        f"the canary test was not applied: {rows} scrambled test rows ended up in "
        "the training set",
        "split with hexperiment.split(df), not with your own split",
        {"scrambled_train_rows": rows},
    )


def _scramble_missed_warning() -> Finding:
    return _warning(
        "the canary test was not applied: the scramble did not reach the audited "
        "frames, because the test target is the same in the normal and the canary "
        "execution",
        "build train and test from the frame returned by hexperiment.load() and "
        "split it with hexperiment.split(df); do not read the data file yourself",
        {},
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
        _PASS_SCORES_FIX,
        {},
    )


def _no_scores_warning() -> Finding:
    return _warning(
        "the canary test is inconclusive without training predictions: it "
        "compared only the training features, so it cannot rule out a model "
        "fitted on test rows",
        _PASS_SCORES_FIX,
        {},
    )


def _mixed_scores_warning(supplied: list[str], missing: list[str]) -> Finding:
    return _warning(
        "training predictions were supplied in some executions but not others "
        f"(supplied in {', '.join(supplied)}; missing in {', '.join(missing)}), "
        "so they could not be compared",
        "pass y_score_train to hexperiment.audit in every execution",
        {
            "executions_with_y_score_train": list(supplied),
            "executions_without_y_score_train": list(missing),
        },
    )


def _peeking_spirit(
    train_rows: int,
    diff: SnapshotDiff,
    locations: Sequence[str],
    selection: tuple[Sequence[str], Sequence[str]] | None = None,
) -> Finding:
    fix = (
        "test rows influenced training; fit preprocessing on training rows only, "
        "after the split."
    )
    message = (
        f"scrambling the test rows changed {diff.rows_changed} of "
        f"{train_rows} training rows"
    )
    evidence: dict[str, JsonValue] = {
        "train_rows": train_rows,
        "rows_changed": diff.rows_changed,
        "feature_rows_changed": diff.feature_rows_changed,
        "largest_change": _round(diff.max_abs_change),
    }
    if selection is not None:
        normal_features, canary_features = selection
        difference = _feature_difference(normal_features, canary_features, "canary")
        message = (
            "features selected changed when only test rows were altered: "
            f"{difference}"
        )
        evidence["normal_features"] = list(normal_features)
        evidence["canary_features"] = list(canary_features)
        fix += " Select features on the training rows only."
    if locations:
        fix += f" Suspect lines: {', '.join(locations)}."
    return Finding(
        check=CANARY_CHECK,
        ghost=PEEKING_SPIRIT,
        severity=Severity.ERROR,
        location=SCRIPT_NAME,
        message=message,
        evidence=evidence,
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


def _features_shapeshifter(normal: Sequence[str], control: Sequence[str]) -> Finding:
    difference = _feature_difference(normal, control, "control")
    return Finding(
        check=DETERMINISM_CHECK,
        ghost=SHAPESHIFTER,
        severity=Severity.WARNING,
        location=SCRIPT_NAME,
        message=(
            "training is not deterministic: an identical execution used other "
            f"features ({difference})"
        ),
        evidence={"normal_features": list(normal), "control_features": list(control)},
        fix="training is not deterministic; set random_state everywhere",
    )


# --- Layer 2: the .hex/ exchange protocol (file I/O) ---------------------------------


class ExecutionFailed(RuntimeError):
    """A training execution, or a file it left in ``.hex/``, cannot be used.

    Not exported, and not a ``HexInputError``, which is reserved for invalid input.
    The command line maps it to exit code 2 (Requirement 16.5).
    """


class ExecutionTimedOut(ExecutionFailed):
    """A training execution ran longer than ``run.timeout_seconds`` (Requirement 16.3).

    A timeout never counts as the recorded input error of Requirement 4.7: the
    script did not finish, so what it left in ``.hex/`` is not a result.
    """


def hex_dir(app_dir: Path, mode: ExecMode) -> Path:
    """Return the folder ``<app_dir>/.hex/<mode>/`` that one execution writes to."""
    return Path(app_dir) / _HEX_DIRNAME / mode.value


def clear_mode_dir(app_dir: Path, mode: ExecMode) -> None:
    """Delete the folder of one mode, so a stale file can never be read.

    The folders of the other modes are left alone. A missing folder is fine.
    """
    try:
        shutil.rmtree(hex_dir(app_dir, mode))
    except FileNotFoundError:
        pass


def write_snapshot(app_dir: Path, snapshot: Snapshot) -> None:
    """Write ``snapshot.npz`` and then ``meta.json`` (Requirement 2.3).

    ``meta.json`` comes last: a reader that finds it can rely on the arrays being
    complete. ``y_score_train`` is stored only when the snapshot has it.
    """
    directory = _prepare(app_dir, snapshot.mode)
    arrays = {
        "train_ids": np.asarray(snapshot.train_ids, dtype=np.int64),
        "test_ids": np.asarray(snapshot.test_ids, dtype=np.int64),
        "train_row_hash": np.asarray(snapshot.train_row_hash, dtype=np.uint64),
    }
    if snapshot.y_score_train is not None:
        arrays["y_score_train"] = np.asarray(snapshot.y_score_train, dtype=np.float64)
    np.savez(directory / _SNAPSHOT_FILE, **arrays)
    _write_json(
        directory / _META_FILE,
        {
            "features": list(snapshot.features),
            "format_version": FORMAT_VERSION,
            "has_y_score_train": snapshot.y_score_train is not None,
            "hexperiment_version": package_version("hexperiment"),
            "mode": snapshot.mode.value,
            "n_test": len(snapshot.test_ids),
            "n_train": len(snapshot.train_ids),
            "run_id": snapshot.run_id,
            "script": SCRIPT_NAME,
            "test_target_hash": snapshot.test_target_hash,
        },
    )


def read_snapshot(app_dir: Path, mode: ExecMode, run_id: str) -> Snapshot:
    """Read the snapshot an execution left, for the run ``run_id``.

    Raises ``ExecutionFailed`` when the files are absent, unreadable, written in
    another format version, or written for another run (Requirement 2.3).
    """
    directory = hex_dir(app_dir, mode)
    meta_path = directory / _META_FILE
    meta = _read_envelope(
        meta_path, run_id, f"train.py did not call hexperiment.audit in {mode} mode"
    )
    try:
        features = tuple(_string_list(meta["features"]))
        has_scores = _boolean(meta["has_y_score_train"])
        test_target_hash = _optional_string(meta["test_target_hash"])
    except (KeyError, TypeError) as error:
        raise _corrupt(
            meta_path, f"key {error} is missing or has the wrong type"
        ) from None
    arrays_path = directory / _SNAPSHOT_FILE
    try:
        with np.load(arrays_path, allow_pickle=False) as npz:
            train_ids = _array(npz, "train_ids", np.int64, arrays_path)
            test_ids = _array(npz, "test_ids", np.int64, arrays_path)
            hashes = _array(npz, "train_row_hash", np.uint64, arrays_path)
            has_array = "y_score_train" in npz.files
            scores = (
                _array(npz, "y_score_train", np.float64, arrays_path)
                if has_array
                else None
            )
    except FileNotFoundError:
        raise ExecutionFailed(
            f"{arrays_path} does not exist, but {meta_path} does. Another run may "
            "be using this app; run only one hexperiment run per app at a time."
        ) from None
    except _UNREADABLE_ARRAYS as error:
        raise _corrupt(arrays_path, str(error)) from None
    if has_array != has_scores:
        raise _corrupt(arrays_path, "y_score_train does not match meta.json")
    if len(hashes) != len(train_ids) or (
        scores is not None and len(scores) != len(train_ids)
    ):
        raise _corrupt(arrays_path, "the training arrays have different lengths")
    return Snapshot(
        mode=mode,
        run_id=run_id,
        features=features,
        train_ids=train_ids,
        test_ids=test_ids,
        train_row_hash=hashes,
        y_score_train=scores,
        test_target_hash=test_target_hash,
    )


def write_load_marker(app_dir: Path, marker: LoadMarker) -> None:
    """Record that an execution loaded its data with ``hexperiment.load``.

    ``scrambled_ids.npy`` is written only when the marker carries ids (canary
    mode); ``loaded.json`` comes last and records how many ids to expect.
    """
    directory = _prepare(app_dir, marker.mode)
    ids = marker.scrambled_ids
    if ids is not None:
        np.save(
            directory / _SCRAMBLED_FILE,
            np.asarray(ids, dtype=np.int64),
            allow_pickle=False,
        )
    _write_json(
        directory / _MARKER_FILE,
        {
            "format_version": FORMAT_VERSION,
            "mode": marker.mode.value,
            "n_scrambled": None if ids is None else len(ids),
            "run_id": marker.run_id,
        },
    )


def read_load_marker(app_dir: Path, mode: ExecMode, run_id: str) -> LoadMarker | None:
    """Read the load marker of one execution; ``None`` when there is none.

    An absent marker means the script never called ``hexperiment.load``
    (Requirement 10.6). A marker of another run or format raises ``ExecutionFailed``.
    """
    directory = hex_dir(app_dir, mode)
    path = directory / _MARKER_FILE
    if not path.exists():
        return None
    meta = _read_envelope(path, run_id, "")
    try:
        expected = meta["n_scrambled"]
    except KeyError as error:
        raise _corrupt(path, f"key {error} is missing") from None
    if expected is None:
        return LoadMarker(mode=mode, run_id=run_id, scrambled_ids=None)
    ids_path = directory / _SCRAMBLED_FILE
    try:
        ids = np.load(ids_path, allow_pickle=False)
    except FileNotFoundError:
        raise _corrupt(path, f"{ids_path.name} is missing") from None
    except _UNREADABLE_ARRAYS as error:
        raise _corrupt(ids_path, str(error)) from None
    if ids.dtype != np.int64 or ids.shape != (expected,):
        raise _corrupt(
            ids_path, f"it does not hold the {expected} ids that loaded.json announced"
        )
    return LoadMarker(mode=mode, run_id=run_id, scrambled_ids=ids)


def write_result(app_dir: Path, result: DataAuditResult, run_id: str | None) -> None:
    """Write ``result.json`` of the normal execution.

    The file carries ``run_id`` (the caller's, which wins over ``result.run_id``)
    and the format version next to the fields of ``DataAuditResult.to_dict``.
    """
    directory = _prepare(app_dir, ExecMode.NORMAL)
    payload = {
        **result.to_dict(),
        "format_version": FORMAT_VERSION,
        "run_id": run_id,
    }
    _write_json(directory / _RESULT_FILE, payload)


def read_result(app_dir: Path, run_id: str) -> DataAuditResult:
    """Read the ``result.json`` of the normal execution for the run ``run_id``."""
    path = hex_dir(app_dir, ExecMode.NORMAL) / _RESULT_FILE
    payload = _read_envelope(
        path, run_id, "train.py did not call hexperiment.audit in normal mode"
    )
    payload.pop("format_version")
    try:
        return DataAuditResult.from_dict(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise _corrupt(path, f"it is not a valid audit result ({error!r})") from None


# --- helpers for Layer 2 -------------------------------------------------------------

_UNREADABLE_ARRAYS = (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile)


def _prepare(app_dir: Path, mode: ExecMode) -> Path:
    directory = hex_dir(app_dir, mode)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    )
    path.write_text(text + "\n", encoding="utf-8")


def _corrupt(path: Path, why: str) -> ExecutionFailed:
    return ExecutionFailed(
        f"{path} cannot be used: {why}. Another run may be using this app, or the "
        "training script wrote the file badly; run it again."
    )


def _read_envelope(path: Path, run_id: str, missing: str) -> dict[str, Any]:
    """Read a JSON file and check its format version and run id.

    ``missing`` is the explanation used when the file does not exist.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ExecutionFailed(f"{path} does not exist: {missing}.") from None
    except (OSError, ValueError) as error:
        raise _corrupt(path, f"it is not readable JSON ({error})") from None
    if not isinstance(raw, dict):
        raise _corrupt(path, "it does not hold a JSON object")
    found_version = raw.get("format_version")
    if found_version != FORMAT_VERSION or isinstance(found_version, bool):
        raise ExecutionFailed(
            f"{path} has format_version {found_version!r}, but this hexperiment "
            f"reads format_version {FORMAT_VERSION}. The command and the training "
            "script use different hexperiment versions; reinstall hexperiment in "
            "this environment so that both match."
        )
    found_run = raw.get("run_id")
    if found_run != run_id:
        raise ExecutionFailed(
            f"{path} belongs to run {found_run!r}, but this run is {run_id!r}: "
            "another run is using this app. Run only one hexperiment run per app "
            "at a time, then run it again."
        )
    return raw


def _array(npz: Any, name: str, dtype: type, path: Path) -> NDArray[Any]:
    array = npz[name]
    if array.dtype != dtype or array.ndim != 1:
        wanted = f"a one-dimensional {np.dtype(dtype)} array"
        raise _corrupt(path, f"{name} is not {wanted}")
    return array


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise TypeError("expected a list of strings")
    return value


def _boolean(value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError("expected a boolean")
    return value


def _optional_string(value: Any) -> str | None:
    if value is not None and not isinstance(value, str):
        raise TypeError("expected a string or null")
    return value


# --- Layer 3: running train.py and orchestrating a run -------------------------------

type Runner = Callable[[Path, ExecMode, str, float], None]

_STDERR_TAIL_LINES = 40
_STDERR_READ_BYTES = 1 << 20


def run_train_script(
    app_dir: Path, mode: ExecMode, run_id: str, timeout_seconds: float
) -> None:
    """Run ``<app_dir>/train.py`` once in ``mode`` (Requirement 16.2).

    The script runs with the current interpreter, with ``app_dir`` as its working
    directory, without a shell, and with ``HEX_MODE`` and ``HEX_RUN_ID`` set.
    ``PYTHONHASHSEED`` is left as the caller has it: a script whose result depends
    on hash order is not deterministic, and that should surface as The Shapeshifter
    (Requirement 10.4). A non-zero exit raises ``ExecutionFailed`` and a timeout
    raises its subclass ``ExecutionTimedOut``, both with the last lines of stderr
    (Requirements 16.3, 16.5).

    Output goes to temporary files, not pipes, and stdin is closed, so a child
    process that outlives the script (joblib or multiprocessing workers) can
    neither keep this call waiting for a pipe nor wait for input. On a timeout the
    whole process tree is killed (Requirement 16.3).
    """
    environment = {
        **os.environ,
        ENV_MODE: mode.value,
        ENV_RUN_ID: run_id,
        "PYTHONIOENCODING": "utf-8",
    }
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            [sys.executable, SCRIPT_NAME],
            cwd=app_dir,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=os.name != "nt",
        )
        timed_out = False
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            if process.poll() is None:
                _kill_process_tree(process)
                process.wait()
        tail = _read_stderr_tail(stderr)
    if timed_out:
        raise ExecutionTimedOut(
            _failure_message(
                f"{SCRIPT_NAME} ran longer than {timeout_seconds:g} seconds in "
                f"{mode.value} mode and was stopped. Make training faster, or "
                "raise [run] timeout_seconds in hex.toml.",
                tail,
            )
        )
    if process.returncode != 0:
        raise ExecutionFailed(
            _failure_message(
                f"{SCRIPT_NAME} exited with code {process.returncode} in "
                f"{mode.value} mode. Fix the error and run it again.",
                tail,
            )
        )


def run_executions(
    app_dir: Path,
    config: HexConfig,
    run_id: str,
    *,
    full: bool,
    runner: Runner = run_train_script,
) -> ExecutionReport:
    """Run the executions of one run and decide the canary (Requirement 10.1).

    A fast run executes ``normal`` only and reports the canary as skipped. A full
    run also executes ``canary``, and ``control`` only when ``decide_canary`` asks
    for it. The folder of a mode is cleared before its execution, and the snapshot
    must exist and carry ``run_id`` afterwards, otherwise ``ExecutionFailed`` is
    raised (Requirements 2.3, 16.5). A failed normal execution is accepted only when
    it left an input error in ``result.json``; the other executions are then skipped
    (Requirement 4.7). ``<app_dir>/verdict.json`` is never read or written.
    """
    timeout = config.run.timeout_seconds
    result, normal = _run_normal(app_dir, run_id, timeout, runner)
    if not full:
        return ExecutionReport(result, skipped_outcome(), RunMode.FAST)
    if normal is None:
        return ExecutionReport(result, skipped_outcome(), RunMode.FULL)
    canary = _run_snapshot(app_dir, ExecMode.CANARY, run_id, timeout, runner)
    marker = read_load_marker(app_dir, ExecMode.CANARY, run_id)

    def run_control() -> Snapshot:
        return _run_snapshot(app_dir, ExecMode.CONTROL, run_id, timeout, runner)

    outcome = decide_canary(normal, canary, marker, run_control, hint_locations(result))
    return ExecutionReport(result, outcome, RunMode.FULL)


# --- helpers for Layer 3 -------------------------------------------------------------


def _run_normal(
    app_dir: Path, run_id: str, timeout: float, runner: Runner
) -> tuple[DataAuditResult, Snapshot | None]:
    """Run the normal execution; the snapshot is ``None`` for an input error."""
    clear_mode_dir(app_dir, ExecMode.NORMAL)
    try:
        runner(app_dir, ExecMode.NORMAL, run_id, timeout)
    except ExecutionTimedOut:
        # Requirement 16.3: a stopped script never counts as a recorded error.
        raise
    except ExecutionFailed:
        # Requirement 4.7: audit raised an input error after recording it.
        recorded = _recorded_input_error(app_dir, run_id)
        if recorded is None:
            raise
        return recorded, None
    result = read_result(app_dir, run_id)
    if result.input_error is not None:
        return result, None
    return result, read_snapshot(app_dir, ExecMode.NORMAL, run_id)


def _run_snapshot(
    app_dir: Path, mode: ExecMode, run_id: str, timeout: float, runner: Runner
) -> Snapshot:
    clear_mode_dir(app_dir, mode)
    runner(app_dir, mode, run_id, timeout)
    return read_snapshot(app_dir, mode, run_id)


def _recorded_input_error(app_dir: Path, run_id: str) -> DataAuditResult | None:
    """The normal ``result.json`` of this run when it carries an input error."""
    try:
        result = read_result(app_dir, run_id)
    except ExecutionFailed:
        return None
    return result if result.input_error is not None else None


def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Kill ``process`` and every process it started (Requirement 16.3)."""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        pass
    # The tree kill can miss the script itself (for example when taskkill is not
    # on the path); kill it directly so the wait that follows always returns.
    process.kill()


def _read_stderr_tail(stream: IO[bytes]) -> str:
    """Decode the end of the stderr file written by the child."""
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(max(0, size - _STDERR_READ_BYTES))
    return stream.read().decode("utf-8", errors="replace")


def _failure_message(headline: str, stderr: str | bytes | None) -> str:
    tail = _stderr_tail(stderr)
    if not tail:
        return headline
    return f"{headline}\nLast lines of stderr:\n{tail}"


def _stderr_tail(stderr: str | bytes | None) -> str:
    if stderr is None:
        return ""
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    return "\n".join(stderr.rstrip().splitlines()[-_STDERR_TAIL_LINES:])
