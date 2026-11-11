"""Tests of ``audit`` with real apps in ``tmp_path`` (Requirements 2.1 to 2.5, 4.7)."""

import copy
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import hexperiment
from hexperiment import HexInputError, audit
from hexperiment.canary import build_snapshot, hex_dir, read_result, read_snapshot
from hexperiment.config import ExecMode
from hexperiment.models import Outcome, RunMode, Severity, Status

RUN = "20251101-093012-a1b2c3"
FEATURES = ["x", "z"]

HEX_TOML = """
[app]
name = "demo"
description = "A demo app."

[data]
path = "data.csv"

[target]
column = "target"

[metric]
name = "roc_auc"
threshold = {threshold}
"""

TRAIN_PY = """\
import hexperiment

df = hexperiment.load()
train, test = hexperiment.split(df)
"""


def write_app(root: Path, threshold: float = 0.6) -> Path:
    app = root / "app"
    app.mkdir()
    toml = HEX_TOML.format(threshold=threshold)
    (app / "hex.toml").write_text(toml, encoding="utf-8")
    (app / "train.py").write_text(TRAIN_PY, encoding="utf-8")
    return app / "train.py"


def make_inputs(n_train: int = 120, n_test: int = 40, seed: int = 0):
    """A train and a test frame, scores, and the features the model used."""
    rng = np.random.default_rng(seed)

    def frame(first_id: int, n: int) -> tuple[pd.DataFrame, np.ndarray]:
        x = rng.normal(size=n)
        target = (x + 1.0 * rng.normal(size=n) > 0).astype(int)
        target[:2] = [0, 1]
        data = {
            "x": x,
            "z": rng.normal(size=n),
            "target": target,
        }
        scores = 1.0 / (1.0 + np.exp(-x))
        return pd.DataFrame(data, index=pd.RangeIndex(first_id, first_id + n)), scores

    train, score_train = frame(0, n_train)
    test, score = frame(n_train, n_test)
    return train, test, score, score_train, FEATURES


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    monkeypatch.delenv("HEX_MODE", raising=False)
    monkeypatch.delenv("HEX_RUN_ID", raising=False)


@pytest.fixture
def script(tmp_path) -> Path:
    return write_app(tmp_path)


def run_in(monkeypatch, mode: str | None, run_id: str | None) -> None:
    if mode is not None:
        monkeypatch.setenv("HEX_MODE", mode)
    if run_id is not None:
        monkeypatch.setenv("HEX_RUN_ID", run_id)


def app_files(app_dir: Path) -> list[str]:
    return sorted(
        path.relative_to(app_dir).as_posix()
        for path in app_dir.rglob("*")
        if path.is_file()
    )


# --- normal mode, started by hexperiment run ---------------------------------------


def test_normal_mode_with_a_run_id_writes_the_snapshot_and_result_but_no_verdict(
    script, monkeypatch
):
    run_in(monkeypatch, None, RUN)
    train, test, score, score_train, features = make_inputs()
    returned = audit(train, test, score, features, script, score_train)
    assert returned is None
    assert app_files(script.parent) == [
        ".hex/normal/meta.json",
        ".hex/normal/result.json",
        ".hex/normal/snapshot.npz",
        "hex.toml",
        "train.py",
    ]


def test_the_normal_snapshot_records_what_training_saw(script, monkeypatch):
    run_in(monkeypatch, "normal", RUN)
    train, test, score, score_train, features = make_inputs()
    audit(train, test, score, features, script, score_train)
    snapshot = read_snapshot(script.parent, ExecMode.NORMAL, RUN)
    expected = build_snapshot(ExecMode.NORMAL, RUN, train, test, features, score_train)
    assert snapshot.features == ("x", "z")
    assert snapshot.train_ids.tolist() == train.index.tolist()
    assert snapshot.test_ids.tolist() == test.index.tolist()
    assert snapshot.train_row_hash.tolist() == expected.train_row_hash.tolist()
    assert snapshot.y_score_train.tolist() == score_train.tolist()


def test_the_result_holds_the_checks_the_metric_and_the_data_summary(
    script, monkeypatch
):
    run_in(monkeypatch, None, RUN)
    train, test, score, score_train, features = make_inputs()
    audit(train, test, score, features, script, score_train)
    result = read_result(script.parent, RUN)
    assert result.run_id == RUN
    assert result.input_error is None
    assert [c.name for c in result.checks] == [
        "row_overlap",
        "target_proxy",
        "temporal",
        "split_order",
    ]
    assert result.metric.name == "roc_auc"
    assert result.metric.threshold == 0.6
    assert (result.data.n_train, result.data.n_test) == (120, 40)


def test_the_split_order_hint_reads_the_script_that_called_audit(script, monkeypatch):
    run_in(monkeypatch, None, RUN)
    train, test, score, score_train, features = make_inputs()
    audit(train, test, score, features, script, score_train)
    checks = {c.name: c for c in read_result(script.parent, RUN).checks}
    assert checks["split_order"].outcome is Outcome.PASSED

    script.unlink()
    audit(train, test, score, features, script, score_train)
    checks = {c.name: c for c in read_result(script.parent, RUN).checks}
    assert checks["split_order"].outcome is Outcome.INCONCLUSIVE


def test_the_app_is_the_folder_of_the_script_even_for_a_relative_path(
    script, monkeypatch
):
    run_in(monkeypatch, None, RUN)
    monkeypatch.chdir(script.parent)
    train, test, score, score_train, features = make_inputs()
    audit(train, test, score, features, "train.py", score_train)
    assert read_result(script.parent, RUN).run_id == RUN


def test_normal_mode_is_the_default_when_hex_mode_is_unset(script, monkeypatch):
    run_in(monkeypatch, None, RUN)
    train, test, score, score_train, features = make_inputs()
    audit(train, test, score, features, str(script), score_train)
    assert hex_dir(script.parent, ExecMode.NORMAL).is_dir()
    assert not hex_dir(script.parent, ExecMode.CANARY).exists()


# --- standalone mode, python train.py without HEX_RUN_ID ---------------------------


def test_a_standalone_run_finalizes_a_fast_verdict(script, capsys):
    train, test, score, score_train, features = make_inputs()
    verdict = audit(train, test, score, features, script, score_train)
    app = script.parent
    assert verdict.mode is RunMode.FAST
    assert verdict.status is Status.UNVERIFIED
    assert verdict.app == "demo"
    assert verdict.test_set_uses == 1
    (run_dir,) = (app / "runs").iterdir()
    assert run_dir.name == verdict.run_id
    assert (run_dir / "model_card.md").is_file()
    written = json.loads((run_dir / "verdict.json").read_text(encoding="utf-8"))
    assert written["status"] == "UNVERIFIED"
    run_copy = (run_dir / "verdict.json").read_bytes()
    assert (app / "verdict.json").read_bytes() == run_copy
    assert "UNVERIFIED (fast)" in capsys.readouterr().out


def test_a_standalone_run_that_misses_the_threshold_is_rejected(tmp_path):
    script = write_app(tmp_path, threshold=0.999)
    train, test, score, score_train, features = make_inputs()
    verdict = audit(train, test, score, features, script, score_train)
    assert verdict.status is Status.REJECTED
    assert verdict.metric.passed is False


def test_every_standalone_run_counts_as_a_use_of_the_test_set(script):
    train, test, score, score_train, features = make_inputs()
    first = audit(train, test, score, features, script, score_train)
    second = audit(train, test, score, features, script, score_train)
    third = audit(train, test, score, features, script, score_train)
    assert [v.test_set_uses for v in (first, second, third)] == [1, 2, 3]
    assert len({first.run_id, second.run_id, third.run_id}) == 3
    history = hexperiment.load_history(script.parent)
    assert {v.run_id for v in history} == {first.run_id, second.run_id, third.run_id}


def test_a_standalone_run_writes_its_snapshot_and_result_without_a_run_id(script):
    train, test, score, score_train, features = make_inputs()
    audit(train, test, score, features, script, score_train)
    meta = json.loads(
        (hex_dir(script.parent, ExecMode.NORMAL) / "meta.json").read_text("utf-8")
    )
    assert meta["run_id"] is None
    assert meta["mode"] == "normal"
    assert (hex_dir(script.parent, ExecMode.NORMAL) / "result.json").is_file()


# --- canary and control mode -------------------------------------------------------


@pytest.mark.parametrize("run_id", [RUN, None])
@pytest.mark.parametrize("mode", ["canary", "control"])
def test_canary_and_control_mode_write_only_the_snapshot(
    script, monkeypatch, mode, run_id
):
    run_in(monkeypatch, mode, run_id)
    train, test, score, score_train, features = make_inputs()
    returned = audit(train, test, score, features, script, score_train)
    assert returned is None
    assert app_files(script.parent) == [
        f".hex/{mode}/meta.json",
        f".hex/{mode}/snapshot.npz",
        "hex.toml",
        "train.py",
    ]


@pytest.mark.parametrize("mode", [ExecMode.CANARY, ExecMode.CONTROL])
def test_the_snapshot_of_canary_and_control_mode_has_the_training_rows(
    script, monkeypatch, mode
):
    run_in(monkeypatch, mode.value, RUN)
    train, test, score, score_train, features = make_inputs()
    audit(train, test, score, features, script, score_train)
    snapshot = read_snapshot(script.parent, mode, RUN)
    expected = build_snapshot(mode, RUN, train, test, features, score_train)
    assert snapshot.mode is mode
    assert snapshot.train_ids.tolist() == train.index.tolist()
    assert snapshot.train_row_hash.tolist() == expected.train_row_hash.tolist()
    assert snapshot.y_score_train.tolist() == score_train.tolist()


def test_canary_mode_does_not_check_what_only_scoring_needs(script, monkeypatch):
    run_in(monkeypatch, "canary", RUN)
    train, test, _, score_train, features = make_inputs()
    test["target"] = 1  # a single class: the metric would be undefined
    del test["z"]  # a feature the test frame lacks
    audit(train, test, [0.5, float("nan")], features, script, score_train)
    assert read_snapshot(script.parent, ExecMode.CANARY, RUN).test_ids.size == 40


# --- input errors (Requirements 4.1 to 4.7) ---------------------------------------


def test_an_input_error_in_normal_mode_is_recorded_and_raised_again(
    script, monkeypatch
):
    run_in(monkeypatch, "normal", RUN)
    train, test, _, score_train, features = make_inputs()
    with pytest.raises(HexInputError, match="y_score has 3 value"):
        audit(train, test, [0.1, 0.2, 0.3], features, script, score_train)
    app = script.parent
    assert app_files(app) == [".hex/normal/result.json", "hex.toml", "train.py"]
    result = read_result(app, RUN)
    error = result.input_error
    assert (error.check, error.ghost, error.severity) == (
        "input",
        None,
        Severity.ERROR,
    )
    assert error.location == "audit()"
    assert error.message.startswith("y_score has 3 value(s)")
    assert "one score per test row" in error.fix
    assert result.metric is None


def test_a_standalone_input_error_leaves_a_rejected_verdict_and_is_raised_again(
    script,
):
    train, test, _, score_train, features = make_inputs()
    with pytest.raises(HexInputError, match="y_score has 3 value"):
        audit(train, test, [0.1, 0.2, 0.3], features, script, score_train)
    app = script.parent
    verdict = json.loads((app / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["status"] == "REJECTED"
    assert verdict["mode"] == "fast"
    assert [f["check"] for f in verdict["findings"]] == ["input"]
    assert verdict["metric"]["value"] is None
    assert verdict["test_set_uses"] == 0
    (run_dir,) = (app / "runs").iterdir()
    assert (run_dir / "verdict.json").is_file()
    assert (run_dir / "model_card.md").is_file()


def test_an_input_error_found_while_building_the_snapshot_is_recorded_too(
    script, monkeypatch
):
    run_in(monkeypatch, "normal", RUN)
    train, test, score, score_train, features = make_inputs()
    repeated = pd.concat([train, train.iloc[:3]])
    repeated_scores = np.concatenate([score_train, score_train[:3]])
    with pytest.raises(HexInputError, match="appear more than once"):
        audit(repeated, test, score, features, script, repeated_scores)
    assert read_result(script.parent, RUN).input_error.check == "input"


@pytest.mark.parametrize("run_id", [RUN, None])
def test_an_input_error_raised_by_a_check_is_recorded_and_raised_again(
    script, monkeypatch, run_id
):
    # Requirement 4.7: a check such as the time parse can raise HexInputError.
    run_in(monkeypatch, "normal", run_id)

    def failing_check(*args, **kwargs):
        raise HexInputError("the time column cannot be parsed.", "fix the column.")

    monkeypatch.setattr("hexperiment.auditor.run_data_checks", failing_check)
    train, test, score, score_train, features = make_inputs()
    with pytest.raises(HexInputError, match="time column cannot be parsed"):
        audit(train, test, score, features, script, score_train)
    app = script.parent
    result = read_result(app, run_id) if run_id else None
    if run_id:
        assert result.input_error.check == "input"
        assert result.input_error.message == "the time column cannot be parsed."
        assert result.input_error.fix == "fix the column."
        assert result.metric is None
    else:
        verdict = json.loads((app / "verdict.json").read_text(encoding="utf-8"))
        assert verdict["status"] == "REJECTED"
        assert [f["check"] for f in verdict["findings"]] == ["input"]


def test_a_byte_order_mark_in_the_script_does_not_break_the_hint(script, monkeypatch):
    run_in(monkeypatch, None, RUN)
    script.write_bytes(b"\xef\xbb\xbf" + TRAIN_PY.encode("utf-8"))
    train, test, score, score_train, features = make_inputs()
    audit(train, test, score, features, script, score_train)
    checks = {c.name: c for c in read_result(script.parent, RUN).checks}
    assert checks["split_order"].outcome is Outcome.PASSED
    assert checks["split_order"].findings == ()


def test_the_summary_is_printed_even_when_the_console_cannot_encode_it(
    script, monkeypatch
):
    raw = io.BytesIO()
    console = io.TextIOWrapper(raw, encoding="ascii", write_through=True)
    monkeypatch.setattr(sys, "stdout", console)
    monkeypatch.setattr("hexperiment.auditor.format_summary", lambda v: "naïve ✓ ok")
    train, test, score, score_train, features = make_inputs()
    assert audit(train, test, score, features, script, score_train) is not None
    assert raw.getvalue() == f"na?ve ? ok{os.linesep}".encode()


def test_a_summary_that_the_console_can_encode_is_printed_unchanged(
    script, monkeypatch
):
    raw = io.BytesIO()
    console = io.TextIOWrapper(raw, encoding="utf-8", write_through=True)
    monkeypatch.setattr(sys, "stdout", console)
    monkeypatch.setattr("hexperiment.auditor.format_summary", lambda v: "naïve ✓ ok")
    train, test, score, score_train, features = make_inputs()
    audit(train, test, score, features, script, score_train)
    assert raw.getvalue() == f"naïve ✓ ok{os.linesep}".encode()


def test_a_single_class_test_target_is_an_input_error_in_normal_mode(
    script, monkeypatch
):
    run_in(monkeypatch, "normal", RUN)
    train, test, score, score_train, features = make_inputs()
    test["target"] = 1
    with pytest.raises(HexInputError, match="single class"):
        audit(train, test, score, features, script, score_train)
    assert "single class" in read_result(script.parent, RUN).input_error.message


@pytest.mark.parametrize("mode", ["canary", "control"])
def test_the_training_side_is_still_validated_in_canary_and_control_mode(
    script, monkeypatch, mode
):
    run_in(monkeypatch, mode, RUN)
    train, test, score, score_train, features = make_inputs()
    bad_scores = score_train.copy()
    bad_scores[3] = np.nan
    with pytest.raises(HexInputError, match="y_score_train has 1 NaN"):
        audit(train, test, score, features, script, bad_scores)
    with pytest.raises(HexInputError, match="train is empty"):
        audit(train.iloc[:0], test, score, features, script, None)
    with pytest.raises(HexInputError, match="'nope'"):
        audit(train, test, score, ["x", "nope"], script, score_train)
    assert app_files(script.parent) == ["hex.toml", "train.py"]


def test_an_invalid_hex_mode_raises_before_anything_is_written(script, monkeypatch):
    run_in(monkeypatch, "sideways", RUN)
    train, test, score, score_train, features = make_inputs()
    with pytest.raises(HexInputError, match="HEX_MODE"):
        audit(train, test, score, features, script, score_train)
    assert app_files(script.parent) == ["hex.toml", "train.py"]


def test_a_missing_hex_toml_raises_before_anything_is_written(script, monkeypatch):
    run_in(monkeypatch, None, RUN)
    (script.parent / "hex.toml").unlink()
    train, test, score, score_train, features = make_inputs()
    with pytest.raises(HexInputError, match="hex.toml"):
        audit(train, test, score, features, script, score_train)
    assert app_files(script.parent) == ["train.py"]


# --- arguments stay unchanged, and y_score_train is optional -----------------------


def copy_of(value):
    if isinstance(value, pd.DataFrame | pd.Series):
        clone = value.copy(deep=True)
        clone.attrs = copy.deepcopy(value.attrs)
        return clone
    if isinstance(value, np.ndarray):
        return value.copy()
    return list(value)


def assert_unchanged(after, before) -> None:
    if isinstance(after, pd.DataFrame):
        pd.testing.assert_frame_equal(after, before, check_exact=True)
        assert after.attrs == before.attrs
    elif isinstance(after, pd.Series):
        pd.testing.assert_series_equal(after, before, check_exact=True)
        assert after.attrs == before.attrs
    elif isinstance(after, np.ndarray):
        assert after.dtype == before.dtype
        assert np.array_equal(after, before, equal_nan=True)
    else:
        assert after == before


@pytest.mark.parametrize("container", ["array", "list", "series"])
@pytest.mark.parametrize(
    ("mode", "run_id"),
    [("normal", RUN), (None, None), ("canary", RUN), ("control", RUN)],
)
def test_audit_never_modifies_its_arguments(
    script, monkeypatch, mode, run_id, container
):
    run_in(monkeypatch, mode, run_id)
    train, test, score, score_train, _ = make_inputs()
    for frame in (train, test):
        frame["cat"] = np.where(frame["x"] > 0, "a", "b")
        frame.attrs["hexperiment"] = {"app_dir": "somewhere", "cutoff": [1, 2]}
    train.loc[train.index[:5], "z"] = np.nan
    features = ["x", "z", "cat"]
    wrap = {"array": lambda a: a, "list": list, "series": pd.Series}[container]
    arguments = [train, test, wrap(score), wrap(score_train), features]
    if container == "array":
        for array in arguments[2:4]:
            array.setflags(write=False)  # a write would raise
    before = [copy_of(a) for a in arguments]

    audit(arguments[0], arguments[1], arguments[2], features, script, arguments[3])

    for after, original in zip(arguments, before, strict=True):
        assert_unchanged(after, original)


@pytest.mark.parametrize("mode", ["normal", "canary", "control"])
def test_without_training_scores_the_snapshot_has_none(script, monkeypatch, mode):
    run_in(monkeypatch, mode, RUN)
    train, test, score, _, features = make_inputs()
    assert audit(train, test, score, features, script) is None
    app = script.parent
    assert read_snapshot(app, ExecMode(mode), RUN).y_score_train is None
    meta = json.loads((hex_dir(app, ExecMode(mode)) / "meta.json").read_text("utf-8"))
    assert meta["has_y_score_train"] is False


def test_a_standalone_run_without_training_scores_is_still_audited(script):
    train, test, score, _, features = make_inputs()
    verdict = audit(train, test, score, features, script)
    assert verdict.status is Status.UNVERIFIED


# --- property: canary and control mode accept any scoring-side input ---------------

TRAIN_ROWS = 6


def fixed_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [0.5, 1.5, np.nan, 3.5, 4.5, 5.5],
            "z": ["a", "b", "a", None, "b", "c"],
            "target": [0, 1, 0, 1, 0, 1],
        },
        index=pd.RangeIndex(TRAIN_ROWS),
    )


@st.composite
def any_test_frames(draw) -> pd.DataFrame:
    rows = draw(st.integers(0, 6))
    names = draw(st.lists(st.sampled_from(["target", "x", "z", "w"]), unique=True))
    values = st.one_of(
        st.integers(-3, 3), st.floats(), st.text(max_size=3), st.none(), st.booleans()
    )
    data = {n: draw(st.lists(values, min_size=rows, max_size=rows)) for n in names}
    return pd.DataFrame(data, index=pd.RangeIndex(100, 100 + rows))


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    mode=st.sampled_from(["canary", "control"]),
    test=any_test_frames(),
    y_score=st.lists(st.floats(), max_size=9),
    features=st.lists(st.sampled_from(["x", "z"]), min_size=1, unique=True),
    y_score_train=st.one_of(
        st.none(),
        st.lists(st.floats(0.0, 1.0), min_size=TRAIN_ROWS, max_size=TRAIN_ROWS),
    ),
)
def test_canary_and_control_mode_write_a_snapshot_for_any_scoring_side_input(
    monkeypatch, mode, test, y_score, features, y_score_train
):
    run_in(monkeypatch, mode, RUN)
    train = fixed_train()
    with tempfile.TemporaryDirectory() as folder:
        script = write_app(Path(folder))
        assert audit(train, test, y_score, features, script, y_score_train) is None
        app = script.parent
        snapshot = read_snapshot(app, ExecMode(mode), RUN)
        assert snapshot.train_ids.tolist() == list(range(TRAIN_ROWS))
        assert snapshot.test_ids.tolist() == test.index.tolist()
        assert snapshot.features == tuple(features)
        assert (snapshot.y_score_train is None) == (y_score_train is None)
        assert app_files(app) == [
            f".hex/{mode}/meta.json",
            f".hex/{mode}/snapshot.npz",
            "hex.toml",
            "train.py",
        ]
