"""Tests of ``hexperiment init``, the templates and the drift check.

Requirements 5.1 to 5.4 and 7.1. Only ``test_the_generated_app_is_approved_end_to_end``
starts real Python processes.
"""

import ast
import json
import tempfile
import tomllib
from importlib import resources
from pathlib import Path, PurePath

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hexperiment import cli, init
from hexperiment.cli import build_parser, main
from hexperiment.config import parse_config
from hexperiment.errors import HexInputError
from hexperiment.init import (
    APP_FILES,
    KIRO_FILES,
    InitReport,
    format_report,
    init_app,
    init_kiro,
)

PACKAGE_DIR = Path(init.__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = PROJECT_DIR.parent
KIRO_REFERENCE_DIR = REPOSITORY_DIR / ".kiro"

KIRO_TARGETS = [
    "hooks/hexperiment.json",
    "workflows/exorcise.workflow.json",
    "agents/ghost-hunter.json",
    "steering/ml-integrity.md",
]


def template_bytes(source: str) -> bytes:
    return (TEMPLATES_DIR / source).read_bytes()


def relative_names(paths: tuple[Path, ...], base: Path) -> list[str]:
    return sorted(path.relative_to(base).as_posix() for path in paths)


# --- init_app (Requirement 5.1, 5.3) -------------------------------------------------


def test_init_app_creates_the_folder_and_both_files(tmp_path):
    app = tmp_path / "apps" / "churn"
    report = init_app(app)
    assert relative_names(report.created, app) == ["hex.toml", "train.py"]
    assert report.skipped == ()
    assert (app / "hex.toml").read_bytes() == template_bytes("app/hex.toml.tmpl")
    assert (app / "train.py").read_bytes() == template_bytes("app/train.py.tmpl")


def test_init_app_in_an_existing_folder_creates_only_the_missing_file(tmp_path):
    (tmp_path / "hex.toml").write_text("mine = 1\n", encoding="utf-8")
    report = init_app(tmp_path)
    assert relative_names(report.created, tmp_path) == ["train.py"]
    assert relative_names(report.skipped, tmp_path) == ["hex.toml"]
    assert (tmp_path / "hex.toml").read_text(encoding="utf-8") == "mine = 1\n"


def test_init_app_leaves_existing_files_untouched(tmp_path):
    original = {"hex.toml": b"a = 1\r\n", "train.py": b"\xff\x00 not text"}
    for name, data in original.items():
        (tmp_path / name).write_bytes(data)
    report = init_app(tmp_path)
    assert report.created == ()
    assert relative_names(report.skipped, tmp_path) == ["hex.toml", "train.py"]
    for name, data in original.items():
        assert (tmp_path / name).read_bytes() == data


def test_init_app_never_opens_an_existing_file_for_writing(tmp_path, monkeypatch):
    for item in APP_FILES:
        (tmp_path / item.target).write_text("keep\n", encoding="utf-8")
    existing = {tmp_path / item.target for item in APP_FILES}
    opened: list[tuple[Path, str]] = []
    original = Path.open

    def spy(self, mode="r", *args, **kwargs):
        opened.append((self, mode))
        return original(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    init_app(tmp_path)
    assert [path for path, _ in opened if path in existing] == []


def test_init_app_refuses_a_file_as_the_app_folder(tmp_path):
    target = tmp_path / "apps"
    target.write_text("not a folder", encoding="utf-8")
    with pytest.raises(HexInputError, match="is not a folder"):
        init_app(target)
    assert target.read_text(encoding="utf-8") == "not a folder"


def test_init_app_refuses_a_folder_where_a_file_belongs_and_writes_nothing(tmp_path):
    (tmp_path / "hex.toml").mkdir()
    with pytest.raises(HexInputError, match="hex.toml is a folder"):
        init_app(tmp_path)
    assert not (tmp_path / "train.py").exists()


def test_init_app_writes_nothing_when_a_template_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(init, "TEMPLATE_DIR", "no-such-templates")
    app = tmp_path / "app"
    with pytest.raises(HexInputError, match="[Rr]einstall"):
        init_app(app)
    assert not app.exists()


def test_the_files_are_written_as_bytes_without_newline_translation(tmp_path):
    init_app(tmp_path)
    assert b"\r" not in (tmp_path / "hex.toml").read_bytes()
    assert b"\r" not in (tmp_path / "train.py").read_bytes()


# --- init_kiro (Requirement 5.2, 5.3) ------------------------------------------------


def test_init_kiro_creates_the_four_files_below_dot_kiro(tmp_path):
    report = init_kiro(tmp_path)
    kiro = tmp_path / ".kiro"
    assert relative_names(report.created, kiro) == sorted(KIRO_TARGETS)
    assert report.skipped == ()
    for target in KIRO_TARGETS:
        assert (kiro / target).is_file()


def test_init_kiro_skips_existing_files_and_keeps_their_bytes(tmp_path):
    hook = tmp_path / ".kiro" / "hooks" / "hexperiment.json"
    hook.parent.mkdir(parents=True)
    hook.write_bytes(b"{}\r\n")
    report = init_kiro(tmp_path)
    assert relative_names(report.skipped, tmp_path) == [".kiro/hooks/hexperiment.json"]
    assert len(report.created) == 3
    assert hook.read_bytes() == b"{}\r\n"


def test_the_kiro_files_are_the_four_templates_below_templates_kiro():
    on_disk = {
        path.relative_to(TEMPLATES_DIR).as_posix()
        for path in (TEMPLATES_DIR / "kiro").rglob("*")
        if path.is_file()
    }
    assert {item.source for item in KIRO_FILES} == on_disk
    assert sorted(item.target for item in KIRO_FILES) == sorted(KIRO_TARGETS)
    assert sorted(item.target for item in APP_FILES) == ["hex.toml", "train.py"]


# --- the report ----------------------------------------------------------------------


def test_the_report_lists_created_and_skipped_files_relative_to_the_base(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(init.shutil, "which", lambda name: "/bin/hexperiment")
    app = tmp_path / "apps" / "churn"
    app.mkdir(parents=True)
    (app / "train.py").write_text("x = 1\n", encoding="utf-8")
    text = format_report(init_app(app), tmp_path)
    lines = text.splitlines()
    assert lines[0] == "created  apps/churn/hex.toml"
    assert lines[1] == "skipped  apps/churn/train.py (already exists)"


def test_the_notes_of_an_app_below_apps_name_the_next_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(init.shutil, "which", lambda name: "/bin/hexperiment")
    text = format_report(init_app(tmp_path / "apps" / "churn"), tmp_path)
    assert "hex.toml and edit its CHANGE lines" in text
    assert "hexperiment run apps/churn" in text
    assert ".hex/" in text and ".gitignore" in text
    assert "ghost hunter" not in text


def test_the_notes_of_an_app_outside_apps_warn_about_the_ghost_hunter(tmp_path):
    text = format_report(init_app(tmp_path / "churn"), tmp_path)
    assert "hexperiment run churn" in text
    assert "ghost hunter agent can only edit Python files under apps/" in text


def test_the_kiro_notes_warn_when_hexperiment_is_not_on_the_path(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(init.shutil, "which", lambda name: None)
    text = format_report(init_kiro(tmp_path), tmp_path)
    assert "`hexperiment` is not on your PATH" in text
    assert "pipx" in text


def test_the_kiro_notes_are_silent_when_hexperiment_is_on_the_path(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(init.shutil, "which", lambda name: "/bin/hexperiment")
    text = format_report(init_kiro(tmp_path), tmp_path)
    assert "PATH" not in text
    assert text.count("\n") == 3  # four file lines and nothing else


def test_the_notes_follow_from_skipped_files_too(tmp_path):
    init_app(tmp_path / "apps" / "churn")
    report = init_app(tmp_path / "apps" / "churn")
    assert report.created == ()
    assert "hexperiment run apps/churn" in format_report(report, tmp_path)


def test_a_report_of_no_files_is_empty(tmp_path):
    assert format_report(InitReport(created=(), skipped=()), tmp_path) == ""


# --- the templates -------------------------------------------------------------------


def test_the_config_template_is_valid_toml_that_the_config_loader_accepts():
    raw = tomllib.loads(template_bytes("app/hex.toml.tmpl").decode("utf-8"))
    config = parse_config(raw, source=Path("hex.toml"), app_dir=Path("my-app"))
    assert config.app.name == "my-model"
    assert config.metric.threshold == 0.75
    assert config.waivers == ()
    assert config.split.time_column is None


def test_the_training_template_is_valid_python_that_ends_with_an_audit():
    tree = ast.parse(template_bytes("app/train.py.tmpl").decode("utf-8"))
    last = tree.body[-1]
    assert isinstance(last, ast.Expr) and isinstance(last.value, ast.Call)
    assert ast.unparse(last.value.func) == "hexperiment.audit"
    keywords = {kw.arg for kw in last.value.keywords}
    assert {"y_score", "features", "script", "y_score_train"} <= keywords
    calls = [
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]
    assert calls.count("hexperiment.audit") == 1
    assert calls.index("hexperiment.load") < calls.index("hexperiment.split")
    assert calls.index("hexperiment.split") < calls.index("model.fit")
    assert "random_state=0" in ast.unparse(tree)


def test_the_package_data_pattern_covers_every_template_file():
    pyproject = tomllib.loads((PROJECT_DIR / "pyproject.toml").read_text("utf-8"))
    patterns = pyproject["tool"]["setuptools"]["package-data"]["hexperiment"]
    assert "templates/**/*" in patterns
    files = [p for p in TEMPLATES_DIR.rglob("*") if p.is_file()]
    assert len(files) == len(APP_FILES) + len(KIRO_FILES)
    for path in files:
        relative = PurePath(path.relative_to(PACKAGE_DIR).as_posix())
        assert any(relative.full_match(pattern) for pattern in patterns), relative


def test_the_templates_folder_is_not_a_python_package():
    assert not list(TEMPLATES_DIR.rglob("__init__.py"))


def test_the_templates_are_reachable_as_package_data():
    data = resources.files("hexperiment").joinpath("templates", "app", "train.py.tmpl")
    assert data.read_bytes() == template_bytes("app/train.py.tmpl")


# --- properties ----------------------------------------------------------------------


@given(
    existing=st.dictionaries(
        st.sampled_from(KIRO_TARGETS), st.binary(max_size=50), max_size=4
    )
)
def test_init_kiro_keeps_existing_files_and_a_second_call_creates_nothing(existing):
    with tempfile.TemporaryDirectory() as name:
        project = Path(name)
        kiro = project / ".kiro"
        for target, data in existing.items():
            (kiro / target).parent.mkdir(parents=True, exist_ok=True)
            (kiro / target).write_bytes(data)
        first = init_kiro(project)
        assert relative_names(first.skipped, kiro) == sorted(existing)
        assert len(first.created) == len(KIRO_TARGETS) - len(existing)
        for target, data in existing.items():
            assert (kiro / target).read_bytes() == data
        second = init_kiro(project)
        assert second.created == ()
        assert len(second.skipped) == len(KIRO_TARGETS)


@given(existing=st.dictionaries(st.sampled_from(["hex.toml", "train.py"]), st.binary()))
def test_init_app_keeps_existing_files_and_a_second_call_creates_nothing(existing):
    with tempfile.TemporaryDirectory() as name:
        app = Path(name) / "app"
        app.mkdir()
        for target, data in existing.items():
            (app / target).write_bytes(data)
        first = init_app(app)
        assert relative_names(first.skipped, app) == sorted(existing)
        for target, data in existing.items():
            assert (app / target).read_bytes() == data
        assert init_app(app).created == ()


# --- the command ---------------------------------------------------------------------


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init.shutil, "which", lambda name: "/bin/hexperiment")
    return tmp_path


def test_the_parser_takes_an_optional_app_folder_and_the_kiro_flag():
    parser = build_parser()
    args = parser.parse_args(["init", "apps/x", "--kiro"])
    assert (args.command, args.app_dir, args.kiro) == ("init", "apps/x", True)
    args = parser.parse_args(["init", "--kiro"])
    assert (args.app_dir, args.kiro) == (None, True)
    args = parser.parse_args(["init"])
    assert (args.app_dir, args.kiro) == (None, False)


def test_init_creates_an_app_and_prints_each_file(project, capsys):
    assert main(["init", "apps/churn"]) == 0
    captured = capsys.readouterr()
    assert "created  apps/churn/hex.toml" in captured.out.splitlines()
    assert "created  apps/churn/train.py" in captured.out.splitlines()
    assert captured.err == ""
    assert (project / "apps" / "churn" / "train.py").is_file()


def test_init_kiro_writes_the_kiro_files_into_the_current_folder(project, capsys):
    assert main(["init", "--kiro"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert sorted(lines) == sorted(f"created  .kiro/{t}" for t in KIRO_TARGETS)
    assert (project / ".kiro" / "agents" / "ghost-hunter.json").is_file()


def test_init_with_both_does_both_and_a_second_run_skips_everything(project, capsys):
    assert main(["init", "apps/churn", "--kiro"]) == 0
    first = capsys.readouterr().out
    assert first.count("created  ") == 6
    assert main(["init", "apps/churn", "--kiro"]) == 0
    second = capsys.readouterr().out
    assert second.count("skipped  ") == 6
    assert "created  " not in second


def test_init_without_arguments_exits_2_with_a_message(project, capsys):
    assert main(["init"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("hexperiment: init needs an app folder")
    assert list(project.iterdir()) == []


def test_init_with_a_file_as_the_app_folder_exits_2_with_a_message(project, capsys):
    (project / "apps").write_text("a file", encoding="utf-8")
    assert main(["init", "apps"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("hexperiment: ")
    assert "is not a folder" in captured.err


def test_init_reports_a_file_system_error_with_exit_code_2(project, capsys):
    (project / "apps").write_text("a file", encoding="utf-8")
    assert main(["init", "apps/churn"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("hexperiment: ")


# --- the generated app runs end to end -----------------------------------------------


def write_dataset(path: Path, rows: int = 600) -> None:
    """Three numeric features; the target is a noisy function of the first two."""
    rng = np.random.default_rng(20251101)
    x1, x2, x3 = rng.normal(size=(3, rows))
    latent = x1 + 0.7 * x2 + rng.normal(size=rows)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": latent > 0})
    frame["target"] = frame["target"].astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_the_generated_app_is_approved_end_to_end(tmp_path, capsys):
    app = tmp_path / "app"
    init_app(app)
    write_dataset(app / "data" / "dataset.csv")
    assert cli.main(["run", str(app)]) == 0
    verdict = json.loads((app / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["status"] == "APPROVED"
    assert capsys.readouterr().out.startswith("APPROVED (full)")


# --- drift check (Requirement 5.4) ---------------------------------------------------


def reference_file(item: init.TemplateFile) -> Path:
    path = KIRO_REFERENCE_DIR / item.target
    if not path.is_file():
        pytest.fail(
            f"{path} is missing: the reference file of the template {item.source} "
            "must exist in the repository's .kiro/ folder.",
            pytrace=False,
        )
    return path


def normalised(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


@pytest.mark.parametrize("item", KIRO_FILES, ids=lambda item: item.target)
def test_the_template_equals_the_reference_file_in_dot_kiro(item):
    reference = reference_file(item)
    template = template_bytes(item.source)
    assert normalised(template) == normalised(reference.read_bytes()), (
        f"{item.source} differs from {reference}: copy the reference file over the "
        "template"
    )


@pytest.mark.parametrize("item", KIRO_FILES, ids=lambda item: item.target)
def test_init_kiro_writes_the_bytes_of_the_template(item, tmp_path):
    init_kiro(tmp_path)
    written = (tmp_path / ".kiro" / item.target).read_bytes()
    assert written == template_bytes(item.source), item.source


def test_the_drift_check_sees_a_difference(tmp_path):
    reference = b'{"a": 1}\n'
    assert normalised(b'{"a": 1}\r\n') == reference
    assert normalised(b'{"a": 2}\n') != reference
