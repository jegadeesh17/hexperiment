import json
import re
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hexperiment import runs
from hexperiment.canary import CanaryOutcome, ExecutionReport, skipped_outcome
from hexperiment.config import HexConfig, parse_config
from hexperiment.errors import HexInputError
from hexperiment.models import (
    CheckResult,
    CheckSummary,
    DataAuditResult,
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
from hexperiment.runs import (
    app_lock,
    clear_app_verdict,
    count_test_set_uses,
    finalize_run,
    format_run_id,
    load_history,
    new_run_id,
)
from hexperiment.verdict import to_json

RUN_ID_FORMAT = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
SIGNATURE = SplitSignature("data/churn.csv", 0.2, 0, None)
DESCRIPTION = "Predict which subscribers cancel (naïve baseline, ≥ 30 days)."


def make_config(app_dir: Path, **split: object) -> HexConfig:
    raw = {
        "app": {"name": "churn", "description": DESCRIPTION},
        "data": {"path": "data/churn.csv"},
        "target": {"column": "y"},
        "metric": {"name": "roc_auc", "threshold": 0.8},
        "split": {"test_fraction": 0.2, "seed": 0} | split,
    }
    return parse_config(raw, source=app_dir / "hex.toml", app_dir=app_dir)


def input_error_finding() -> Finding:
    return Finding(
        check="input",
        ghost=None,
        severity=Severity.ERROR,
        location="audit()",
        message="y_score has the wrong length",
        evidence={},
        fix="pass one score per test row",
    )


def clean_result() -> DataAuditResult:
    return DataAuditResult(
        run_id=None,
        metric=MetricResult("roc_auc", 0.85, 0.8, True),
        checks=(CheckResult("split_order", None, Outcome.PASSED),),
        extra_findings=(),
        data=DataSummary(80, 20, 0.5, 0.5, None, None),
        input_error=None,
    )


def input_error_result() -> DataAuditResult:
    return DataAuditResult(
        run_id=None,
        metric=None,
        checks=(),
        extra_findings=(),
        data=None,
        input_error=input_error_finding(),
    )


def full_report() -> ExecutionReport:
    passed = CanaryOutcome(Outcome.PASSED, Outcome.PASSED, ())
    return ExecutionReport(clean_result(), passed, RunMode.FULL)


def fast_report() -> ExecutionReport:
    return ExecutionReport(clean_result(), skipped_outcome(), RunMode.FAST)


def input_error_report() -> ExecutionReport:
    return ExecutionReport(input_error_result(), skipped_outcome(), RunMode.FULL)


def make_verdict(
    run_id: str = "20251101-090000-aaaaaa",
    created_at: str = "2025-11-01T09:00:00Z",
    status: Status = Status.UNVERIFIED,
    split: SplitSignature = SIGNATURE,
    findings: tuple[Finding, ...] = (),
    mode: RunMode = RunMode.FAST,
) -> Verdict:
    return Verdict(
        status=status,
        mode=mode,
        app="churn",
        run_id=run_id,
        created_at=created_at,
        hexperiment_version="0.1.0",
        metric=MetricResult("roc_auc", 0.85, 0.8, True),
        checks=(CheckSummary("split_order", None, Outcome.PASSED),),
        findings=findings,
        waivers=(),
        test_set_uses=1,
        split=split,
        data=None,
    )


def store(app_dir: Path, verdict: Verdict) -> Path:
    folder = app_dir / "runs" / verdict.run_id
    folder.mkdir(parents=True)
    (folder / "verdict.json").write_text(to_json(verdict), encoding="utf-8")
    return folder


# --- run ids -------------------------------------------------------------------------


def test_format_run_id():
    now = datetime(2025, 11, 1, 9, 30, 12)
    assert format_run_id(now, "a1b2c3") == "20251101-093012-a1b2c3"


def test_format_run_id_pads_small_values():
    run_id = format_run_id(datetime(7, 1, 2, 3, 4, 5), "000000")
    assert run_id == "00070102-030405-000000"


def test_format_run_id_converts_an_aware_time_to_utc():
    zone = timezone(timedelta(hours=2))
    now = datetime(2025, 11, 1, 1, 30, 12, tzinfo=zone)
    assert format_run_id(now, "a1b2c3") == "20251031-233012-a1b2c3"


@pytest.mark.parametrize("token", ["", "a1b2c", "a1b2c3d", "A1B2C3", "g1b2c3"])
def test_format_run_id_rejects_a_bad_token(token):
    with pytest.raises(HexInputError, match="6 lowercase hex"):
        format_run_id(datetime(2025, 11, 1), token)


def test_new_run_id_uses_the_given_clock_and_token():
    assert new_run_id(datetime(2025, 11, 1, 9, 30, 12), "a1b2c3") == (
        "20251101-093012-a1b2c3"
    )


def test_new_run_ids_are_well_formed_and_unique():
    ids = [new_run_id() for _ in range(2)]
    assert all(RUN_ID_FORMAT.match(i) for i in ids)
    assert ids[0] != ids[1]


@given(
    st.datetimes(),
    st.text(alphabet="0123456789abcdef", min_size=6, max_size=6),
)
def test_format_run_id_always_has_the_documented_shape(now, token):
    assert RUN_ID_FORMAT.match(format_run_id(now, token))


# --- count_test_set_uses -------------------------------------------------------------


def test_count_counts_fast_full_rejected_and_unverified_runs():
    history = [
        make_verdict("20251101-090000-aaaaaa", status=Status.UNVERIFIED),
        make_verdict("20251101-090100-bbbbbb", status=Status.REJECTED),
        make_verdict(
            "20251101-090200-cccccc", status=Status.APPROVED, mode=RunMode.FULL
        ),
        make_verdict(
            "20251101-090300-dddddd", status=Status.REJECTED, mode=RunMode.FULL
        ),
    ]
    assert count_test_set_uses(history, SIGNATURE) == 4


def test_count_ignores_input_error_runs():
    history = [
        make_verdict("20251101-090000-aaaaaa"),
        make_verdict(
            "20251101-090100-bbbbbb",
            status=Status.REJECTED,
            findings=(input_error_finding(),),
        ),
    ]
    assert count_test_set_uses(history, SIGNATURE) == 1


def test_count_ignores_runs_with_another_split_block():
    other = [
        replace(SIGNATURE, data_path="data/other.csv"),
        replace(SIGNATURE, test_fraction=0.3),
        replace(SIGNATURE, seed=1),
        replace(SIGNATURE, time_column="signup_date"),
    ]
    history = [
        make_verdict(f"20251101-09000{i}-aaaaaa", split=s) for i, s in enumerate(other)
    ]
    history.append(make_verdict("20251101-090100-bbbbbb"))
    assert count_test_set_uses(history, SIGNATURE) == 1


def test_count_of_an_empty_history_is_zero():
    assert count_test_set_uses([], SIGNATURE) == 0


@given(st.integers(min_value=0, max_value=6), st.integers(min_value=0, max_value=6))
def test_adding_input_error_verdicts_never_changes_the_count(scored, failed):
    scored_runs = [make_verdict(f"20251101-0900{i:02d}-aaaaaa") for i in range(scored)]
    failed_runs = [
        make_verdict(
            f"20251101-0910{i:02d}-bbbbbb",
            status=Status.REJECTED,
            findings=(input_error_finding(),),
        )
        for i in range(failed)
    ]
    assert count_test_set_uses(scored_runs, SIGNATURE) == scored
    assert count_test_set_uses(scored_runs + failed_runs, SIGNATURE) == scored


# --- load_history --------------------------------------------------------------------


def test_history_of_an_app_without_runs_is_empty(tmp_path):
    assert load_history(tmp_path) == []
    (tmp_path / "runs").mkdir()
    assert load_history(tmp_path) == []


def test_history_is_ordered_by_created_at_then_run_id(tmp_path):
    verdicts = [
        make_verdict("20251101-090000-ffffff", "2025-11-01T09:00:01Z"),
        make_verdict("20251101-090000-bbbbbb", "2025-11-01T09:00:00Z"),
        make_verdict("20251101-090000-aaaaaa", "2025-11-01T09:00:00Z"),
        make_verdict("20251031-120000-cccccc", "2025-10-31T12:00:00Z"),
    ]
    for verdict in verdicts:
        store(tmp_path, verdict)
    history = load_history(str(tmp_path))
    assert [v.run_id for v in history] == [
        "20251031-120000-cccccc",
        "20251101-090000-aaaaaa",
        "20251101-090000-bbbbbb",
        "20251101-090000-ffffff",
    ]
    assert history[1] == verdicts[2]


def test_history_ignores_a_run_folder_without_a_verdict(tmp_path):
    store(tmp_path, make_verdict())
    (tmp_path / "runs" / "20251101-091500-eeeeee").mkdir()
    (tmp_path / "runs" / "notes.txt").write_text("not a run", encoding="utf-8")
    assert len(load_history(tmp_path)) == 1


@pytest.mark.parametrize(
    "content",
    [b"{not json", b"[1, 2]", b'{"status": "APPROVED"}', b"\xff\xfe\x00"],
)
def test_malformed_history_file_raises_naming_the_file(tmp_path, content):
    folder = store(tmp_path, make_verdict())
    bad = tmp_path / "runs" / "20251101-091500-eeeeee"
    bad.mkdir()
    (bad / "verdict.json").write_bytes(content)
    with pytest.raises(HexInputError) as caught:
        load_history(tmp_path)
    assert str(bad / "verdict.json") in str(caught.value)
    assert folder.exists()


# --- clear_app_verdict ---------------------------------------------------------------


def test_clear_app_verdict_deletes_the_file_only(tmp_path):
    (tmp_path / "verdict.json").write_text("{}", encoding="utf-8")
    folder = store(tmp_path, make_verdict())
    clear_app_verdict(tmp_path)
    assert not (tmp_path / "verdict.json").exists()
    assert (folder / "verdict.json").exists()


def test_clear_app_verdict_without_a_file_does_nothing(tmp_path):
    clear_app_verdict(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_clear_app_verdict_retries_a_delete_that_fails_twice(tmp_path, refuse_unlink):
    (tmp_path / "verdict.json").write_text("{}", encoding="utf-8")
    attempts, pauses = refuse_unlink(tmp_path, failures=2)
    clear_app_verdict(tmp_path)
    assert len(attempts) == 3
    assert pauses == [0.2, 0.2]
    assert not (tmp_path / "verdict.json").exists()


def test_clear_app_verdict_gives_up_after_five_retries(tmp_path, refuse_unlink):
    (tmp_path / "verdict.json").write_text("{}", encoding="utf-8")
    attempts, pauses = refuse_unlink(tmp_path, failures=None)
    with pytest.raises(OSError) as caught:
        clear_app_verdict(tmp_path)
    message = str(caught.value)
    assert "could not be removed" in message
    assert "must not be trusted" in message
    assert str(tmp_path / "verdict.json") in message
    assert isinstance(caught.value.__cause__, PermissionError)
    assert len(attempts) == 6
    assert pauses == [0.2] * 5
    assert (tmp_path / "verdict.json").exists()


# --- finalize_run --------------------------------------------------------------------

RUN_A = "20251101-090000-aaaaaa"
RUN_B = "20251101-090100-bbbbbb"


def test_finalize_writes_verdict_card_and_app_copy(tmp_path):
    verdict = finalize_run(tmp_path, make_config(tmp_path), full_report(), run_id=RUN_A)
    folder = tmp_path / "runs" / RUN_A
    assert verdict.status is Status.APPROVED
    assert verdict.run_id == RUN_A
    assert verdict.hexperiment_version == version("hexperiment")
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", verdict.created_at)
    assert (folder / "verdict.json").read_text(encoding="utf-8") == to_json(verdict)
    card = (folder / "model_card.md").read_text(encoding="utf-8")
    assert "Generated by hexperiment. Do not edit by hand." in card
    assert DESCRIPTION in card
    run_copy = (folder / "verdict.json").read_bytes()
    assert (tmp_path / "verdict.json").read_bytes() == run_copy
    assert json.loads(run_copy.decode("utf-8"))["run_id"] == RUN_A


def test_finalize_leaves_no_temporary_file(tmp_path):
    finalize_run(tmp_path, make_config(tmp_path), full_report(), run_id=RUN_A)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["runs", "verdict.json"]


def test_finalize_replaces_an_older_app_level_verdict(tmp_path):
    (tmp_path / "verdict.json").write_text('{"status": "APPROVED"}', encoding="utf-8")
    finalize_run(tmp_path, make_config(tmp_path), fast_report(), run_id=RUN_A)
    stored = json.loads((tmp_path / "verdict.json").read_text(encoding="utf-8"))
    assert stored["run_id"] == RUN_A
    assert stored["mode"] == "fast"


def test_finalize_counts_the_test_set_uses_of_earlier_runs(tmp_path):
    config = make_config(tmp_path)
    first = finalize_run(tmp_path, config, fast_report(), run_id=RUN_A)
    second = finalize_run(tmp_path, config, full_report(), run_id=RUN_B)
    assert (first.test_set_uses, second.test_set_uses) == (1, 2)


def test_finalize_does_not_count_an_input_error_run(tmp_path):
    config = make_config(tmp_path)
    finalize_run(tmp_path, config, fast_report(), run_id=RUN_A)
    failed = finalize_run(tmp_path, config, input_error_report(), run_id=RUN_B)
    after = finalize_run(
        tmp_path, config, fast_report(), run_id="20251101-090200-cccccc"
    )
    assert failed.status is Status.REJECTED
    assert failed.test_set_uses == 1
    assert after.test_set_uses == 2


def test_finalize_does_not_count_runs_with_another_split(tmp_path):
    finalize_run(tmp_path, make_config(tmp_path), fast_report(), run_id=RUN_A)
    other = make_config(tmp_path, seed=7)
    verdict = finalize_run(tmp_path, other, fast_report(), run_id=RUN_B)
    assert verdict.test_set_uses == 1


def test_card_lists_earlier_runs_but_not_the_current_one(tmp_path):
    config = make_config(tmp_path)
    finalize_run(tmp_path, config, fast_report(), run_id=RUN_A)
    finalize_run(tmp_path, config, full_report(), run_id=RUN_B)
    card = (tmp_path / "runs" / RUN_B / "model_card.md").read_text(encoding="utf-8")
    history = card.split("## Run history")[1]
    assert RUN_A in history
    assert RUN_B not in history


def test_finalize_never_overwrites_an_earlier_run_folder(tmp_path):
    config = make_config(tmp_path)
    finalize_run(tmp_path, config, fast_report(), run_id=RUN_A)
    folder = tmp_path / "runs" / RUN_A
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    app_before = (tmp_path / "verdict.json").read_bytes()
    with pytest.raises(FileExistsError):
        finalize_run(tmp_path, config, full_report(), run_id=RUN_A)
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
    assert (tmp_path / "verdict.json").read_bytes() == app_before


def test_finalize_keeps_every_earlier_run_folder(tmp_path):
    config = make_config(tmp_path)
    for run_id in (RUN_A, RUN_B, "20251101-090200-cccccc"):
        finalize_run(tmp_path, config, fast_report(), run_id=run_id)
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == [
        RUN_A,
        RUN_B,
        "20251101-090200-cccccc",
    ]
    assert [v.run_id for v in load_history(tmp_path)] == [
        RUN_A,
        RUN_B,
        "20251101-090200-cccccc",
    ]


@pytest.mark.parametrize("run_id", ["../escape", "run-1", "", RUN_A + "/x"])
def test_finalize_rejects_a_run_id_that_is_not_well_formed(tmp_path, run_id):
    with pytest.raises(HexInputError, match="YYYYMMDD-HHMMSS"):
        finalize_run(tmp_path, make_config(tmp_path), fast_report(), run_id=run_id)
    assert list(tmp_path.iterdir()) == []


def test_finalize_with_a_malformed_history_writes_nothing(tmp_path):
    bad = tmp_path / "runs" / "20251101-091500-eeeeee"
    bad.mkdir(parents=True)
    (bad / "verdict.json").write_text("{", encoding="utf-8")
    with pytest.raises(HexInputError):
        finalize_run(tmp_path, make_config(tmp_path), fast_report(), run_id=RUN_A)
    assert not (tmp_path / "runs" / RUN_A).exists()
    assert not (tmp_path / "verdict.json").exists()


def test_failed_app_copy_removes_its_temporary_file(tmp_path, monkeypatch):
    def refuse(source, target):
        raise PermissionError("locked")

    monkeypatch.setattr(runs.os, "replace", refuse)
    with pytest.raises(PermissionError):
        finalize_run(tmp_path, make_config(tmp_path), fast_report(), run_id=RUN_A)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["runs"]
    assert (tmp_path / "runs" / RUN_A / "verdict.json").exists()


# --- finalize_run: the run folder is complete or absent (Requirement 15.1) ----------


def failing_write(monkeypatch, filename: str) -> None:
    original = runs._write_text

    def write(path, text):
        if path.name == filename:
            raise OSError("disk full")
        original(path, text)

    monkeypatch.setattr(runs, "_write_text", write)


@pytest.mark.parametrize("filename", ["model_card.md", "verdict.json"])
def test_a_write_error_leaves_no_run_folder_and_history_still_works(
    tmp_path, monkeypatch, filename
):
    config = make_config(tmp_path)
    finalize_run(tmp_path, config, fast_report(), run_id=RUN_A)
    app_before = (tmp_path / "verdict.json").read_bytes()
    failing_write(monkeypatch, filename)
    with pytest.raises(OSError, match="disk full"):
        finalize_run(tmp_path, config, full_report(), run_id=RUN_B)
    assert not (tmp_path / "runs" / RUN_B).exists()
    assert [v.run_id for v in load_history(tmp_path)] == [RUN_A]
    assert (tmp_path / "verdict.json").read_bytes() == app_before


def test_the_card_is_written_before_the_verdict(tmp_path, monkeypatch):
    written = []
    original = runs._write_text

    def write(path, text):
        written.append(path.name)
        original(path, text)

    monkeypatch.setattr(runs, "_write_text", write)
    finalize_run(tmp_path, make_config(tmp_path), fast_report(), run_id=RUN_A)
    assert written == ["model_card.md", "verdict.json"]


def test_a_leftover_partial_folder_is_ignored_then_replaced(tmp_path):
    config = make_config(tmp_path)
    finalize_run(tmp_path, config, fast_report(), run_id=RUN_A)
    leftover = tmp_path / "runs" / f".partial-{RUN_B}"
    leftover.mkdir()
    (leftover / "verdict.json").write_text("{not json", encoding="utf-8")
    (leftover / "stale.txt").write_text("old", encoding="utf-8")
    assert [v.run_id for v in load_history(tmp_path)] == [RUN_A]
    verdict = finalize_run(tmp_path, config, full_report(), run_id=RUN_B)
    assert verdict.run_id == RUN_B
    assert not leftover.exists()
    folder = tmp_path / "runs" / RUN_B
    assert sorted(p.name for p in folder.iterdir()) == ["model_card.md", "verdict.json"]
    assert [v.run_id for v in load_history(tmp_path)] == [RUN_A, RUN_B]


def test_a_partial_folder_left_by_a_write_error_is_replaced_by_the_next_run(
    tmp_path, monkeypatch
):
    config = make_config(tmp_path)
    with monkeypatch.context() as patch:
        failing_write(patch, "verdict.json")
        with pytest.raises(OSError):
            finalize_run(tmp_path, config, fast_report(), run_id=RUN_A)
    partial = tmp_path / "runs" / f".partial-{RUN_A}"
    assert (partial / "model_card.md").is_file()
    assert load_history(tmp_path) == []
    finalize_run(tmp_path, config, fast_report(), run_id=RUN_A)
    assert not partial.exists()
    assert [v.run_id for v in load_history(tmp_path)] == [RUN_A]


def test_a_run_folder_that_appears_during_the_write_is_not_overwritten(
    tmp_path, monkeypatch
):
    taken = tmp_path / "runs" / RUN_A
    original = runs._write_text

    def write(path, text):
        original(path, text)
        if path.name == "verdict.json":
            taken.mkdir()
            (taken / "note.txt").write_text("keep", encoding="utf-8")

    monkeypatch.setattr(runs, "_write_text", write)
    with pytest.raises(FileExistsError, match=RUN_A):
        finalize_run(tmp_path, make_config(tmp_path), fast_report(), run_id=RUN_A)
    assert [p.name for p in taken.iterdir()] == ["note.txt"]
    assert not (tmp_path / "verdict.json").exists()


# --- app_lock (Requirement 8) ---------------------------------------------------------

HOLDER = (
    "import sys\n"
    "from hexperiment.runs import app_lock\n"
    "with app_lock(sys.argv[1]):\n"
    "    print('locked', flush=True)\n"
    "    sys.stdin.read()\n"
)


def start_holder(app: Path) -> subprocess.Popen:
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(app)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "locked"
    return holder


def test_app_lock_creates_the_hex_folder_and_the_lock_file(tmp_path):
    with app_lock(tmp_path):
        assert (tmp_path / ".hex" / "run.lock").is_file()


def test_app_lock_can_be_taken_again_after_it_is_released(tmp_path):
    with app_lock(tmp_path):
        pass
    with app_lock(tmp_path):
        pass


def test_app_lock_is_released_when_the_body_raises(tmp_path):
    with pytest.raises(RuntimeError), app_lock(tmp_path):
        raise RuntimeError("boom")
    with app_lock(tmp_path):
        pass


def test_app_lock_refuses_a_second_holder_in_the_same_process(tmp_path):
    with app_lock(tmp_path):
        with pytest.raises(HexInputError) as caught, app_lock(tmp_path):
            pass
    assert str(caught.value) == (
        f"another hexperiment run is in progress on {tmp_path}; "
        "wait for it to finish, then run again"
    )


def test_a_refused_lock_leaves_the_first_holder_in_place(tmp_path):
    with app_lock(tmp_path):
        with pytest.raises(HexInputError), app_lock(tmp_path):
            pass
        with pytest.raises(HexInputError), app_lock(tmp_path):
            pass


def test_app_lock_is_refused_while_another_process_holds_it(tmp_path):
    holder = start_holder(tmp_path)
    try:
        with (
            pytest.raises(HexInputError, match="another hexperiment run"),
            app_lock(tmp_path),
        ):
            pass
    finally:
        holder.communicate("")
    with app_lock(tmp_path):
        pass


def test_app_lock_is_released_when_the_holder_is_killed(tmp_path):
    holder = start_holder(tmp_path)
    try:
        with pytest.raises(HexInputError), app_lock(tmp_path):
            pass
    finally:
        holder.kill()
        holder.communicate()
    # Windows frees the locks of a killed process a moment after it ends, later
    # when the machine is busy, so the lock is retried for a while (Requirement 8.2).
    deadline = time.monotonic() + 10
    while True:
        try:
            with app_lock(tmp_path):
                break
        except HexInputError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)


def test_app_lock_does_not_touch_run_history_or_verdict(tmp_path):
    (tmp_path / "verdict.json").write_text("{}", encoding="utf-8")
    with app_lock(tmp_path):
        pass
    assert (tmp_path / "verdict.json").read_text(encoding="utf-8") == "{}"
    assert not (tmp_path / "runs").exists()
