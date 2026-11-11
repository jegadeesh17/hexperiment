"""Builders for the MCP tests: apps, verdicts, run folders and a stub runner.

Shared by ``test_mcp_service.py`` and ``test_mcp_server.py``. A stub ``RunApp``
stands in for ``cli.run_app``, so no MCP test starts a training run.
"""

import hashlib
from pathlib import Path

from hexperiment.models import (
    Finding,
    MetricResult,
    RunMode,
    RunOutcome,
    Severity,
    SplitSignature,
    Status,
    Verdict,
)
from hexperiment.verdict import to_json

HEX_TOML = """\
[app]
name = "{name}"

[data]
path = "data.csv"

[target]
column = "target"

[metric]
name = "roc_auc"
threshold = 0.8
"""

OLD = "20260101-000000-aaaaaa"
NEW = "20260102-000000-bbbbbb"


class StubRun:
    """A ``RunApp`` that records its calls and returns a prepared outcome."""

    def __init__(self, outcome: RunOutcome | None = None) -> None:
        self.outcome = outcome or RunOutcome(3, Status.UNVERIFIED, "summary text", "")
        self.calls: list[tuple[Path, bool, float | None]] = []

    def __call__(
        self, app_dir: Path, *, full: bool, timeout_cap: float | None = None
    ) -> RunOutcome:
        self.calls.append((app_dir, full, timeout_cap))
        return self.outcome


def make_app(folder: Path, name: str = "demo") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "hex.toml").write_text(HEX_TOML.format(name=name), encoding="utf-8")
    (folder / "train.py").write_text("print('train')\n", encoding="utf-8")
    return folder


def finding(check: str, ghost: str | None, severity: Severity = Severity.ERROR):
    return Finding(
        check=check,
        ghost=ghost,
        severity=severity,
        location="somewhere",
        message="something is wrong",
        evidence={},
        fix="fix it",
    )


def make_verdict(
    run_id: str = OLD,
    *,
    created_at: str = "2026-01-01T00:00:00Z",
    status: Status = Status.UNVERIFIED,
    findings: tuple[Finding, ...] = (),
    value: float | None = 0.85,
) -> Verdict:
    return Verdict(
        status=status,
        mode=RunMode.FAST,
        app="demo",
        run_id=run_id,
        created_at=created_at,
        hexperiment_version="0.1.0",
        metric=MetricResult("roc_auc", value, 0.8, True),
        checks=(),
        findings=findings,
        waivers=(),
        test_set_uses=1,
        split=SplitSignature("data.csv", 0.2, 0, None),
        data=None,
    )


def write_run(app: Path, verdict: Verdict, card: str | None = None) -> Path:
    folder = app / "runs" / verdict.run_id
    folder.mkdir(parents=True)
    (folder / "verdict.json").write_text(to_json(verdict), encoding="utf-8")
    text = f"# Model card of {verdict.run_id}\n" if card is None else card
    (folder / "model_card.md").write_text(text, encoding="utf-8")
    return folder


def write_app_verdict(app: Path, verdict: Verdict) -> Path:
    path = app / "verdict.json"
    path.write_text(to_json(verdict), encoding="utf-8")
    return path


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        digest.update(path.relative_to(root).as_posix().encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()
