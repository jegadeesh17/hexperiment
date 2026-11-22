"""Pure helpers and file handling of ``apps/verify.py`` (Requirement 5.2).

Nothing here runs ``hexperiment``: the tests build a small fake app in a temporary
folder and call the helpers directly.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path
from types import ModuleType

import pytest
from apps_helpers import APPS_DIR, FLIGHTS, HOTELS

POSSESSION_LINE = (
    '    "reservation_status",  # LEAK: The Possession. Filled in at the end.\n'
)
SPLIT_LINE = "    train, test = other_split(df)  # LEAK: random split\n"
HAUNTED_SOURCE = (
    "CATEGORICAL = [\n"
    '    "hotel",\n'
    + POSSESSION_LINE
    + "]\n"
    "def main():\n"
    + SPLIT_LINE
)
HONEST_SOURCE = 'CATEGORICAL = ["hotel"]\n'


@pytest.fixture
def possession_fixed_source(verify_module: ModuleType):
    return verify_module.possession_fixed_source


@pytest.fixture
def fake_app(tmp_path: Path) -> Path:
    """An app folder with every kind of entry that the scratch copy must leave out."""
    app = tmp_path / "source" / "fake-app"
    for folder in ("runs/20260101-000000", ".hex", "haunted", "__pycache__", "data"):
        (app / folder).mkdir(parents=True)
    (app / "runs" / "20260101-000000" / "verdict.json").write_text("{}")
    (app / ".hex" / "snapshot.json").write_text("{}")
    (app / "__pycache__" / "train.cpython-313.pyc").write_bytes(b"\0")
    (app / "haunted" / "train.py").write_text(HAUNTED_SOURCE, encoding="utf-8")
    (app / "train.py").write_text(HONEST_SOURCE, encoding="utf-8")
    (app / "hex.toml").write_text("[app]\n", encoding="utf-8")
    (app / "fetch_data.py").write_text("# fetch\n", encoding="utf-8")
    (app / "verdict.json").write_text("{}", encoding="utf-8")
    (app / "data" / "prepared.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (app / "data" / "hotels.csv").write_text("source\n", encoding="utf-8")
    (app / "data" / "flights.csv").write_text("source\n", encoding="utf-8")
    return app


def listing(folder: Path) -> set[str]:
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*")}


# --- make_scratch -------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant", ["haunted", "possession-fixed", "split-fixed", "honest"]
)
def test_make_scratch_leaves_out_history_snapshots_and_sources(
    verify_module: ModuleType, fake_app: Path, tmp_path: Path, variant: str
) -> None:
    root = tmp_path / "scratch"
    root.mkdir()
    scratch = verify_module.make_scratch(fake_app, root, variant)
    assert scratch == root / "fake-app"
    assert listing(scratch) == {
        "train.py",
        "hex.toml",
        "fetch_data.py",
        "data",
        "data/prepared.csv",
    }


def test_make_scratch_keeps_prepared_data_and_leaves_the_app_untouched(
    verify_module: ModuleType, fake_app: Path, tmp_path: Path
) -> None:
    before = listing(fake_app)
    root = tmp_path / "scratch"
    root.mkdir()
    scratch = verify_module.make_scratch(fake_app, root, "honest")
    assert (scratch / "data" / "prepared.csv").read_text() == "a,b\n1,2\n"
    assert listing(fake_app) == before


def test_make_scratch_writes_the_variant_as_train_py(
    verify_module: ModuleType, fake_app: Path, tmp_path: Path
) -> None:
    root = tmp_path / "scratch"
    root.mkdir()
    scratch = verify_module.make_scratch(fake_app, root, "haunted")
    assert (scratch / "train.py").read_text(encoding="utf-8") == HAUNTED_SOURCE


def test_every_name_in_the_ignore_list_is_left_out(verify_module: ModuleType) -> None:
    assert set(verify_module.COPY_IGNORE) >= {
        "runs",
        ".hex",
        "verdict.json",
        "haunted",
        "__pycache__",
        "hotels.csv",
        "flights.csv",
    }


# --- variants -----------------------------------------------------------------------


def test_the_variants_of_each_app(verify_module: ModuleType) -> None:
    assert verify_module.VARIANTS[HOTELS] == ("haunted", "possession-fixed", "honest")
    assert verify_module.VARIANTS[FLIGHTS] == ("haunted", "split-fixed", "honest")


def test_the_real_possession_fixed_variant_drops_only_that_leak(
    verify_module: ModuleType,
) -> None:
    app = APPS_DIR / HOTELS
    haunted = (app / "haunted" / "train.py").read_text(encoding="utf-8")
    fixed = verify_module.variant_source(app, "possession-fixed")
    constants = {
        n.value
        for n in ast.walk(ast.parse(fixed))
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    assert "reservation_status" not in constants
    assert "country_cancel_rate" in constants
    assert len(fixed.splitlines()) == len(haunted.splitlines()) - 1


def test_an_unknown_variant_is_refused(verify_module: ModuleType) -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        verify_module.variant_source(APPS_DIR / HOTELS, "nonsense")


# --- possession_fixed_source (pure) ---------------------------------------------------


def test_possession_fixed_source_removes_only_the_marked_line(
    possession_fixed_source,
) -> None:
    fixed = possession_fixed_source(HAUNTED_SOURCE)
    assert fixed == HAUNTED_SOURCE.replace(POSSESSION_LINE, "")
    assert "reservation_status" not in fixed


def test_possession_fixed_source_keeps_a_similar_column_name(
    possession_fixed_source,
) -> None:
    source = HAUNTED_SOURCE + '    "reservation_status_date",\n'
    assert '"reservation_status_date"' in possession_fixed_source(source)


@pytest.mark.parametrize("count", [0, 2])
def test_possession_fixed_source_needs_exactly_one_marked_line(
    possession_fixed_source, count: int
) -> None:
    source = POSSESSION_LINE * count + "y = 2\n"
    with pytest.raises(ValueError, match="exactly one"):
        possession_fixed_source(source)


def test_possession_fixed_source_needs_the_column_and_the_marker(
    possession_fixed_source,
) -> None:
    only_column = '    "reservation_status",\n'
    only_marker = "    x = 1  # LEAK: The Possession.\n"
    with pytest.raises(ValueError, match="exactly one"):
        possession_fixed_source(only_column + only_marker)


def test_possession_fixed_source_refuses_a_line_with_other_code(
    possession_fixed_source,
) -> None:
    source = 'CATEGORICAL += ["reservation_status"]  # LEAK: The Possession.\n'
    with pytest.raises(ValueError, match="own line"):
        possession_fixed_source(source)


# --- read_verdict -------------------------------------------------------------------


def test_read_verdict_reads_a_json_object(
    verify_module: ModuleType, tmp_path: Path
) -> None:
    (tmp_path / "verdict.json").write_text('{"status": "APPROVED"}', encoding="utf-8")
    assert verify_module.read_verdict(tmp_path) == {"status": "APPROVED"}


def test_read_verdict_without_a_file_is_none(
    verify_module: ModuleType, tmp_path: Path
) -> None:
    assert verify_module.read_verdict(tmp_path) is None


@pytest.mark.parametrize("text", ["{not json", "", "[1, 2]", "null"])
def test_read_verdict_with_invalid_json_is_none(
    verify_module: ModuleType, tmp_path: Path, text: str
) -> None:
    (tmp_path / "verdict.json").write_text(text, encoding="utf-8")
    assert verify_module.read_verdict(tmp_path) is None


def test_read_verdict_with_bytes_that_are_not_utf8_is_none(
    verify_module: ModuleType, tmp_path: Path
) -> None:
    (tmp_path / "verdict.json").write_bytes(b"\xff\xfe\x00{")
    assert verify_module.read_verdict(tmp_path) is None


# --- remove_scratch -----------------------------------------------------------------


def test_remove_scratch_deletes_the_folder(
    verify_module: ModuleType, tmp_path: Path
) -> None:
    folder = tmp_path / "scratch"
    (folder / "inner").mkdir(parents=True)
    verify_module.remove_scratch(folder)
    assert not folder.exists()


def test_remove_scratch_warns_with_the_folder_name_when_it_cannot_delete(
    verify_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    folder = tmp_path / "hexverify-locked"
    folder.mkdir()

    def refuse(path: object, *args: object, **kwargs: object) -> None:
        raise PermissionError("in use")

    monkeypatch.setattr(shutil, "rmtree", refuse)
    verify_module.remove_scratch(folder)
    err = capsys.readouterr().err
    assert "warning" in err
    assert str(folder) in err
    assert folder.exists()
