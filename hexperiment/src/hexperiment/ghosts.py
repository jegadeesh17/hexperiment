"""The catalogue of ghosts and checks: what each finding means and how to fix it.

Pure data and one lookup. The ghost and check names come from the constants of the
core modules, so they cannot drift from the names in a verdict (Requirement 4.3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from hexperiment.canary import (
    CANARY_CHECK,
    DETERMINISM_CHECK,
    PEEKING_SPIRIT,
    SHAPESHIFTER,
)
from hexperiment.checks import metric, row_overlap, split_order, target_proxy, temporal
from hexperiment.errors import HexInputError

INPUT_CHECK = "input"
TEST_SET_USES_CHECK = "test_set_uses"
WAIVER_CHECK = "waiver"

_LEADING_THE = re.compile(r"^the[\s_-]+")
_NOT_ALPHANUMERIC = re.compile(r"[^a-z0-9]")


@dataclass(frozen=True, slots=True)
class Ghost:
    check: str  # check name as in verdict findings
    ghost: str | None  # ghost name, None for checks without a ghost
    meaning: str
    why_it_matters: str
    how_to_fix: str


CATALOG: tuple[Ghost, ...] = (
    Ghost(
        check=row_overlap.CHECK,
        ghost=row_overlap.GHOST,
        meaning="Identical rows exist in both train and test.",
        why_it_matters=(
            "The model can memorise a training row and be scored on its copy, so "
            "the metric measures memory."
        ),
        how_to_fix=(
            "Drop exact duplicate rows before splitting, then call `hexperiment.split`."
        ),
    ),
    Ghost(
        check=target_proxy.CHECK,
        ghost=target_proxy.GHOST,
        meaning=(
            "One feature alone predicts the target almost perfectly (stand-alone "
            "AUC at or above `checks.proxy_auc_max`, default 0.95)."
        ),
        why_it_matters=(
            "Such a column is usually recorded after the outcome or derived from "
            "it, and will not exist at prediction time."
        ),
        how_to_fix=(
            "Find out when the column is recorded. If after the outcome, drop it "
            "and say why in a comment. A genuinely strong feature can only be "
            "accepted by a human through a waiver with a reason in `hex.toml`."
        ),
    ),
    Ghost(
        check=CANARY_CHECK,
        ghost=PEEKING_SPIRIT,
        meaning=(
            "Scrambling the test rows at load time changed what training saw, so "
            "test information reached training."
        ),
        why_it_matters=(
            "Statistics, scalers, encoders or selectors fitted on all rows carry "
            "test information into the model. The finding cannot be waived."
        ),
        how_to_fix=(
            "Fit every imputer, scaler, encoder and selector on training rows "
            "only, after `hexperiment.split`, preferably in a `Pipeline`. The "
            "finding lists suspect lines."
        ),
    ),
    Ghost(
        check=temporal.CHECK,
        ghost=temporal.GHOST,
        meaning="Some training rows are later than some test rows.",
        why_it_matters=(
            "The model is trained on the future of the period it is tested on."
        ),
        how_to_fix=(
            "Set `split.time_column` in `hex.toml` and split with "
            "`hexperiment.split`. Never shuffle across time."
        ),
    ),
    Ghost(
        check=DETERMINISM_CHECK,
        ghost=SHAPESHIFTER,
        meaning="Identical executions gave different results (a warning).",
        why_it_matters=(
            "A result that changes between identical runs cannot be verified, so "
            "the verdict cannot be `APPROVED`."
        ),
        how_to_fix=(
            "Set `random_state` or a seed everywhere (models, samplers, NumPy) "
            "and avoid code that depends on hash order or thread timing."
        ),
    ),
    Ghost(
        check=split_order.CHECK,
        ghost=None,
        meaning=(
            "A call that fits or computes statistics appears before the first "
            "split, or data is read without `hexperiment.load` (warnings only)."
        ),
        why_it_matters=(
            "It is a hint to the likely cause of The Peeking Spirit. The canary "
            "is the authority."
        ),
        how_to_fix=(
            "Split first, fit on training rows, read data only with `hexperiment.load`."
        ),
    ),
    Ghost(
        check=metric.CHECK,
        ghost=None,
        meaning=(
            'The metric is below its threshold, or far above it ("too good to be '
            'true" warning).'
        ),
        why_it_matters=(
            "Below: the model is not good enough. Far above: a leak is more likely "
            "than a breakthrough."
        ),
        how_to_fix=(
            "Below: improve the model, never the threshold. Far above: look for a "
            "column that encodes the target."
        ),
    ),
    Ghost(
        check=TEST_SET_USES_CHECK,
        ghost=None,
        meaning=(
            "The same test set has been scored more than `checks.max_test_uses` times."
        ),
        why_it_matters=(
            "Repeated use turns the test set into a tuning set and the metric "
            "becomes optimistic."
        ),
        how_to_fix=(
            "Stop tuning against this test set. Use rows that were not used to "
            "choose the model for a final number."
        ),
    ),
    Ghost(
        check=WAIVER_CHECK,
        ghost=None,
        meaning="A waiver in `hex.toml` matched no finding (a warning).",
        why_it_matters=(
            "A stale waiver could silently cover a problem that appears later."
        ),
        how_to_fix="Remove the waiver from `hex.toml` if the finding no longer occurs.",
    ),
    Ghost(
        check=INPUT_CHECK,
        ghost=None,
        meaning=(
            "`audit` received invalid input, so no check ran and the verdict is "
            "`REJECTED`."
        ),
        why_it_matters="The metric cannot be trusted when its inputs are wrong.",
        how_to_fix=("Read the message: it says what is wrong and how to fix it."),
    ),
)


def find_ghost(name: str) -> Ghost:
    """The catalogue entry for a ghost name or a check name.

    Case, spaces, ``_``, ``-`` and a leading "the" do not matter: ``The Peeking
    Spirit``, ``peeking-spirit`` and ``canary`` give the same entry. An unknown
    name raises ``HexInputError`` that lists the known names (Requirement 4.5).
    """
    wanted = _normalise(name) if isinstance(name, str) else ""
    if wanted:
        for entry in CATALOG:
            if wanted == _normalise(entry.check):
                return entry
            if entry.ghost is not None and wanted == _normalise(entry.ghost):
                return entry
    raise HexInputError(
        f"{name!r} is not a known ghost or check. Known ghosts: {_known_ghosts()}. "
        f"Known checks: {_known_checks()}.",
        "Pass one of these names, as it appears in a finding of verdict.json.",
    )


def to_dict(entry: Ghost) -> dict[str, str | None]:
    """The entry as a JSON-ready dictionary."""
    return {
        "check": entry.check,
        "ghost": entry.ghost,
        "meaning": entry.meaning,
        "why_it_matters": entry.why_it_matters,
        "how_to_fix": entry.how_to_fix,
    }


def _normalise(name: str) -> str:
    lowered = _LEADING_THE.sub("", name.strip().lower())
    return _NOT_ALPHANUMERIC.sub("", lowered)


def _known_ghosts() -> str:
    return ", ".join(entry.ghost for entry in CATALOG if entry.ghost is not None)


def _known_checks() -> str:
    return ", ".join(entry.check for entry in CATALOG)
