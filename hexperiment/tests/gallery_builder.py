"""Builds verdicts and app sources in memory for the gallery tests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hexperiment.config import (
    AppConfig,
    ChecksConfig,
    DataConfig,
    HexConfig,
    MetricConfig,
    RunConfig,
    SplitConfig,
    TargetConfig,
)
from hexperiment.gallery.records import AppSource
from hexperiment.models import (
    CheckSummary,
    DataSummary,
    Finding,
    MetricResult,
    Outcome,
    RunMode,
    Severity,
    SplitSignature,
    Status,
    Verdict,
)

THRESHOLD = 0.85


def make_finding(
    *,
    check: str = "target_proxy",
    ghost: str | None = "The Possession",
    severity: Severity = Severity.ERROR,
    location: str = "reservation_status",
    message: str = "reservation_status alone predicts the target with AUC 1.0000",
    fix: str = "Check whether this column is recorded after the outcome.",
    waiver_reason: str | None = None,
) -> Finding:
    return Finding(
        check=check,
        ghost=ghost,
        severity=severity,
        location=location,
        message=message,
        evidence={},
        fix=fix,
        waiver_reason=waiver_reason,
    )


def make_verdict(
    status: Status | str = Status.APPROVED,
    mode: RunMode | str = RunMode.FULL,
    value: float | None = 0.9,
    findings: Sequence[Finding] = (),
    created_at: str = "2026-11-11T19:50:51Z",
    run_id: str = "20261111-195051-abcdef",
    *,
    app: str = "demo",
    threshold: float = THRESHOLD,
    test_set_uses: int = 1,
    time_column: str | None = None,
) -> Verdict:
    return Verdict(
        status=Status(status),
        mode=RunMode(mode),
        app=app,
        run_id=run_id,
        created_at=created_at,
        hexperiment_version="0.1.0",
        metric=MetricResult(
            name="roc_auc",
            value=value,
            threshold=threshold,
            passed=value is not None and value >= threshold,
        ),
        checks=(CheckSummary("metric", None, Outcome.PASSED),),
        findings=tuple(findings),
        waivers=(),
        test_set_uses=test_set_uses,
        split=SplitSignature("data.csv", 0.2, 0, time_column),
        data=DataSummary(80, 20, 0.5, 0.5, None, None),
    )


def make_config(
    name: str = "demo",
    *,
    description: str = "A demo app.",
    target: str = "target",
    threshold: float = THRESHOLD,
) -> HexConfig:
    return HexConfig(
        app=AppConfig(name=name, description=description),
        data=DataConfig("data.csv"),
        target=TargetConfig(target),
        metric=MetricConfig("roc_auc", threshold),
        split=SplitConfig(),
        checks=ChecksConfig(),
        run=RunConfig(),
        waivers=(),
        app_dir=Path("app"),
    )


def make_source(
    name: str = "demo",
    runs: Sequence[Verdict] | None = None,
    *,
    description: str = "A demo app.",
    target: str = "target",
    haunted: str | None = None,
    train: str = "print('train')\n",
) -> AppSource:
    history = tuple(runs) if runs is not None else (make_verdict(),)
    has_approved = any(run.status is Status.APPROVED for run in history)
    return AppSource(
        config=make_config(name, description=description, target=target),
        history=history,
        last_verdict_json=b"{}\n",
        approved_card=b"# Model card\n" if has_approved else None,
        train_py=train if haunted is not None else None,
        haunted_py=haunted,
    )


def rejected(value: float | None, *, ghost: str = "The Possession") -> Verdict:
    """A REJECTED run with one error finding."""
    return make_verdict(
        Status.REJECTED,
        value=value,
        findings=[make_finding(ghost=ghost)],
        run_id="20261111-195051-000001",
    )


def approved(value: float) -> Verdict:
    return make_verdict(Status.APPROVED, value=value, run_id="20261111-195052-000002")


HEX_TOML = """\
[app]
name = "{name}"
description = "A demo app."

[data]
path = "data.csv"

[target]
column = "target"

[metric]
name = "roc_auc"
threshold = {threshold}
"""


def _encode(text: str, newline: str) -> bytes:
    return text.replace("\n", newline).encode("utf-8")


def write_app(
    root: Path,
    name: str,
    runs: Sequence[Verdict],
    haunted: str | None = None,
    newline: str = "\n",
    *,
    train: str = "x = 1\n",
) -> Path:
    """Write an app folder under ``root`` and return it.

    Everything is written in binary mode, so ``newline="\\r\\n"`` really gives CRLF
    files, as a Windows checkout with ``core.autocrlf=true`` does.
    """
    from hexperiment.card import render
    from hexperiment.verdict import to_json

    app = root / name
    (app / "runs").mkdir(parents=True)
    (app / "hex.toml").write_bytes(
        _encode(HEX_TOML.format(name=name, threshold=THRESHOLD), newline)
    )
    (app / "train.py").write_bytes(_encode(train, newline))
    if haunted is not None:
        (app / "haunted").mkdir()
        (app / "haunted" / "train.py").write_bytes(_encode(haunted, newline))
    ordered = sorted(runs, key=lambda v: (v.created_at, v.run_id))
    for index, verdict in enumerate(ordered):
        folder = app / "runs" / verdict.run_id
        folder.mkdir()
        (folder / "verdict.json").write_bytes(_encode(to_json(verdict), newline))
        card = render(verdict, ordered[:index], description="A demo app.")
        (folder / "model_card.md").write_bytes(_encode(card, newline))
    return app
