"""Case folder names, per-app summaries and the choice of the hero (pure)."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from hexperiment.errors import HexInputError
from hexperiment.gallery.records import AppSource
from hexperiment.ghosts import CATALOG
from hexperiment.models import Finding, Status, Verdict

MARKER_FILENAME = ".hexperiment-gallery"
RESERVED_SLUGS = frozenset(
    {"assets", "con", "prn", "aux", "nul"}
    | {f"com{n}" for n in range(1, 10)}
    | {f"lpt{n}" for n in range(1, 10)}
)

_NOT_SLUG = re.compile(r"[^a-z0-9]+")
_SUSPECT_MARKER = "Suspect lines: "


def slugify(name: str) -> str:
    """Lower case, runs of other characters become one ``-``, edges trimmed."""
    return _NOT_SLUG.sub("-", name.lower()).strip("-")


def assign_slugs(labelled_names: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    """Case folder names for ``(app folder as given, app name)`` pairs.

    Raises ``HexInputError`` for an empty, reserved or repeated slug (Requirement 1.7).
    """
    slugs: list[str] = []
    owners: dict[str, str] = {}
    for label, name in labelled_names:
        slug = slugify(name)
        if not slug:
            raise HexInputError(
                f"the name {name!r} of the app in {label} gives an empty case "
                "folder name.",
                "Use letters or digits in `[app] name` in hex.toml.",
            )
        if slug in RESERVED_SLUGS:
            reason = (
                "which the site keeps for its stylesheet"
                if slug == "assets"
                else "which Windows reserves as a device name"
            )
            raise HexInputError(
                f"the name of the app in {label} gives the case folder {slug!r}, "
                f"{reason}.",
                "Rename the app in hex.toml.",
            )
        if slug in owners:
            raise HexInputError(
                f"the apps in {owners[slug]} and {label} both give the case folder "
                f"{slug!r}.",
                "Give each app a different `[app] name` in hex.toml.",
            )
        owners[slug] = label
        slugs.append(slug)
    return tuple(slugs)


def score_text(value: float | None) -> str:
    """A score with four decimals, or ``no score`` for a run that has none."""
    return "no score" if value is None else f"{value:.4f}"


def ghosts_found(history: Sequence[Verdict]) -> tuple[str, ...]:
    """Distinct ghost names on any finding of any run, in catalogue order.

    Warnings and waived findings count. Names not in the catalogue go last, sorted.
    """
    names = {
        finding.ghost
        for verdict in history
        for finding in verdict.findings
        if finding.ghost is not None
    }
    catalogue = [entry.ghost for entry in CATALOG if entry.ghost is not None]
    known = [name for name in catalogue if name in names]
    return tuple(known + sorted(names - set(known)))


def suspect_lines(finding: Finding) -> tuple[str, ...]:
    """The items after ``Suspect lines: `` in the fix text, or ``()``."""
    _, marker, tail = finding.fix.rpartition(_SUSPECT_MARKER)
    if not marker:
        return ()
    tail = tail.strip().removesuffix(".")
    return tuple(item for item in tail.split(", ") if item)


@dataclass(frozen=True, slots=True)
class AppSummary:
    slug: str
    name: str
    description: str
    target_column: str
    metric_name: str
    threshold: float
    runs: tuple[Verdict, ...]
    first: Verdict
    last: Verdict
    first_rejected: Verdict | None
    last_approved: Verdict | None
    ghosts: tuple[str, ...]


def summarize_app(slug: str, source: AppSource) -> AppSummary:
    runs = source.history
    rejected = [run for run in runs if run.status is Status.REJECTED]
    approved = [run for run in runs if run.status is Status.APPROVED]
    return AppSummary(
        slug=slug,
        name=source.config.app.name,
        description=source.config.app.description,
        target_column=source.config.target.column,
        metric_name=source.config.metric.name,
        threshold=source.config.metric.threshold,
        runs=runs,
        first=runs[0],
        last=runs[-1],
        first_rejected=rejected[0] if rejected else None,
        last_approved=approved[-1] if approved else None,
        ghosts=ghosts_found(runs),
    )


@dataclass(frozen=True, slots=True)
class Hero:
    app: AppSummary
    rejected: Verdict  # the first REJECTED run of the app
    approved: Verdict | None  # the last APPROVED run of the app


def _shown(value: float) -> float:
    """The score as a reader sees it (four decimals)."""
    return float(score_text(value))


def _hero_key(summary: AppSummary, index: int) -> tuple[float, float, int]:
    rejected = summary.first_rejected
    assert rejected is not None and rejected.metric.value is not None
    score = _shown(rejected.metric.value)
    approved = summary.last_approved
    if approved is None or approved.metric.value is None:
        drop = -math.inf
    else:
        drop = round(score - _shown(approved.metric.value), 4)
    return (score, drop, -index)


def choose_hero(summaries: Sequence[AppSummary]) -> Hero | None:
    """The app with the largest ``(first rejected score, drop, -index)``.

    A rejected run without a score is never a candidate (Requirement 2.6).
    """
    candidates = [
        (_hero_key(summary, index), summary)
        for index, summary in enumerate(summaries)
        if summary.first_rejected is not None
        and summary.first_rejected.metric.value is not None
    ]
    if not candidates:
        return None
    _, best = max(candidates, key=lambda pair: pair[0])
    assert best.first_rejected is not None
    return Hero(app=best, rejected=best.first_rejected, approved=best.last_approved)
