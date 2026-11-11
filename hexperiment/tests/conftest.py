from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from hypothesis import settings

from hexperiment import runs
from hexperiment.canary import (
    ExecutionFailed,
    ExecutionTimedOut,
    LoadMarker,
    Snapshot,
    write_load_marker,
    write_result,
    write_snapshot,
)
from hexperiment.config import ExecMode
from hexperiment.models import DataAuditResult

settings.register_profile("hexperiment", max_examples=60, deadline=None)
settings.load_profile("hexperiment")


class FakeRunner:
    """A ``Runner`` that writes prepared files instead of starting Python.

    Each mode has a ``Plan``. A mode without a plan writes nothing, like a script
    that never called ``hexperiment.audit``. ``calls`` records the modes run, in
    order, and ``timeouts`` the timeout passed for each.
    """

    @dataclass(frozen=True)
    class Plan:
        snapshot: Snapshot | None = None
        marker: LoadMarker | None = None
        result: DataAuditResult | None = None
        written_run_id: str | None = None  # written in place of the real run id
        fail: str | None = None  # raise ExecutionFailed, like a non-zero exit
        timed_out: bool = False  # raise ExecutionTimedOut, like a stopped script

    def __init__(
        self,
        *,
        normal: Plan | None = None,
        canary: Plan | None = None,
        control: Plan | None = None,
    ) -> None:
        self.plans = {
            ExecMode.NORMAL: normal,
            ExecMode.CANARY: canary,
            ExecMode.CONTROL: control,
        }
        self.calls: list[ExecMode] = []
        self.timeouts: list[float] = []

    def __call__(
        self, app_dir: Path, mode: ExecMode, run_id: str, timeout_seconds: float
    ) -> None:
        self.calls.append(mode)
        self.timeouts.append(timeout_seconds)
        plan = self.plans[mode]
        if plan is None:
            return
        written = plan.written_run_id or run_id
        if plan.snapshot is not None:
            write_snapshot(app_dir, replace(plan.snapshot, mode=mode, run_id=written))
        if plan.marker is not None:
            write_load_marker(app_dir, replace(plan.marker, mode=mode, run_id=written))
        if plan.result is not None:
            write_result(app_dir, plan.result, written)
        if plan.fail is not None:
            raise ExecutionFailed(plan.fail)
        if plan.timed_out:
            raise ExecutionTimedOut("train.py ran longer than the timeout")


@pytest.fixture
def fake_runner() -> type[FakeRunner]:
    return FakeRunner


@pytest.fixture
def anyio_backend() -> str:
    """The MCP tests run on asyncio only."""
    return "asyncio"


@pytest.fixture
def refuse_unlink(monkeypatch):
    """Make ``<app_dir>/verdict.json`` fail to delete, like a file held open.

    ``install(app_dir, failures=n)`` fails the first ``n`` deletes (``None``: all of
    them). It returns the list of attempts and the list of pauses between them; the
    pauses are recorded, not slept.
    """

    def install(app_dir: Path, *, failures: int | None):
        target = app_dir.resolve() / "verdict.json"
        original = Path.unlink
        attempts: list[int] = []
        pauses: list[float] = []

        def unlink(self, missing_ok=False):
            if self != target:
                return original(self, missing_ok=missing_ok)
            attempts.append(1)
            if failures is None or len(attempts) <= failures:
                raise PermissionError(13, "used by another process")
            return original(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", unlink)
        monkeypatch.setattr(runs.time, "sleep", pauses.append)
        return attempts, pauses

    return install
