import hashlib
import json
import tomllib
from pathlib import Path, PurePath

import pytest

from gallery_builder import make_finding, make_verdict, write_app
from hexperiment import runs as runs_module
from hexperiment.errors import HexInputError
from hexperiment.gallery import sources
from hexperiment.gallery.sources import (
    RUN_ID_PATTERN,
    read_app,
    read_benchmark,
    read_fonts,
    read_stylesheet,
)
from hexperiment.models import Severity, Status
from test_gallery_benchmark import valid

REJECTED_ID = "20261111-195051-000001"
APPROVED_ID = "20261111-195052-000002"


def history():
    return [
        make_verdict(
            Status.REJECTED,
            value=1.0,
            findings=[make_finding(severity=Severity.ERROR)],
            run_id=REJECTED_ID,
            created_at="2026-11-11T19:50:51Z",
        ),
        make_verdict(
            Status.APPROVED,
            value=0.9096,
            run_id=APPROVED_ID,
            created_at="2026-11-11T19:50:52Z",
        ),
    ]


def make_app(tmp_path, name="demo", **kwargs):
    return write_app(tmp_path, name, history(), **kwargs)


def tree_hash(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def refusal(path: Path) -> str:
    with pytest.raises(HexInputError) as caught:
        read_app(path)
    return str(caught.value)


def edit_verdict(app: Path, folder: str, **changes) -> None:
    path = app / "runs" / folder / "verdict.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


# read_app


def test_a_good_app_is_read(tmp_path):
    source = read_app(make_app(tmp_path, haunted="a = 1\n", train="a = 2\n"))
    assert [v.run_id for v in source.history] == [REJECTED_ID, APPROVED_ID]
    assert source.config.app.name == "demo"
    assert json.loads(source.last_verdict_json)["run_id"] == APPROVED_ID
    assert source.approved_card is not None
    assert (source.train_py, source.haunted_py) == ("a = 2\n", "a = 1\n")


def test_no_haunted_script_gives_none_for_both(tmp_path):
    source = read_app(make_app(tmp_path))
    assert source.train_py is None
    assert source.haunted_py is None


def test_an_app_without_an_approved_run_has_no_card(tmp_path):
    app = write_app(tmp_path, "pending", history()[:1])
    assert read_app(app).approved_card is None


def test_the_card_is_the_one_of_the_last_approved_run(tmp_path):
    app = make_app(tmp_path)
    card = (app / "runs" / APPROVED_ID / "model_card.md").read_bytes()
    assert read_app(app).approved_card == card


def test_a_folder_that_is_not_an_app_is_refused(tmp_path):
    message = refusal(tmp_path)
    assert "is not an app folder" in message
    assert "hex.toml and train.py" in message


def test_an_invalid_hex_toml_is_refused_with_the_core_message(tmp_path):
    app = make_app(tmp_path)
    (app / "hex.toml").write_text("[app\n", encoding="utf-8")
    assert "hex.toml" in refusal(app)


def test_an_app_with_no_recorded_run_is_refused(tmp_path):
    app = write_app(tmp_path, "empty", [])
    message = refusal(app)
    assert "has no recorded run" in message
    assert "hexperiment run" in message


def test_a_damaged_history_file_is_refused(tmp_path):
    app = make_app(tmp_path)
    (app / "runs" / APPROVED_ID / "verdict.json").write_text("{", encoding="utf-8")
    assert "verdict.json" in refusal(app)


def test_a_hostile_run_id_never_reaches_a_path(tmp_path, monkeypatch):
    app = make_app(tmp_path)
    (tmp_path / "x").mkdir(exist_ok=True)
    edit_verdict(app, APPROVED_ID, run_id="..\\..\\x")
    opened: list[Path] = []
    original = Path.read_bytes

    def spy(self):
        opened.append(self)
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", spy)
    message = refusal(app)
    assert "not of the form" in message
    assert not any(".." in path.parts or "x" in path.parts for path in opened)


def test_a_missing_run_folder_is_refused(tmp_path):
    app = make_app(tmp_path)
    edit_verdict(app, APPROVED_ID, run_id="20261111-195052-ffffff")
    message = refusal(app)
    assert "20261111-195052-ffffff" in message
    assert "missing" in message


def test_a_missing_model_card_is_refused(tmp_path):
    app = make_app(tmp_path)
    (app / "runs" / APPROVED_ID / "model_card.md").unlink()
    message = refusal(app)
    assert "model_card.md" in message
    assert "Restore it from version control" in message


@pytest.mark.parametrize("which", ["train.py", "haunted/train.py"])
def test_a_script_that_is_not_utf8_is_refused(tmp_path, which):
    app = make_app(tmp_path, haunted="a = 1\n")
    (app / which).write_bytes(b"x = '\xff\xfe'\n")
    message = refusal(app)
    assert "UTF-8" in message
    assert which.replace("/", "\\") in message or which in message


def test_the_run_id_pattern_equals_the_one_of_runs_py():
    assert RUN_ID_PATTERN.pattern == runs_module._RUN_ID_PATTERN.pattern


# run notes


def test_notes_are_read_aligned_with_the_history(tmp_path):
    app = make_app(tmp_path, notes={APPROVED_ID: b"  Made by hand.\r\n\r\n"})
    assert read_app(app).notes == (None, "Made by hand.")


def test_an_app_without_notes_has_a_none_for_every_run(tmp_path):
    assert read_app(make_app(tmp_path)).notes == (None, None)


def test_a_note_of_exactly_300_characters_is_accepted(tmp_path):
    text = "é" * 300
    app = make_app(tmp_path, notes={REJECTED_ID: (text + "\n").encode("utf-8")})
    assert read_app(app).notes == (text, None)


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (b"caf\xe9\n", "not valid UTF-8"),
        (b"", "is empty"),
        (b" \r\n\n \n", "is empty"),
        (b"one\ntwo\n", "more than one line"),
        (b"one\r\ntwo\r\n", "more than one line"),
        (b"one\rtwo\n", "more than one line"),
        (("x" * 301 + "\n").encode(), "301 characters"),
    ],
)
def test_a_bad_note_is_refused_and_names_the_file(tmp_path, data, reason):
    app = make_app(tmp_path, notes={APPROVED_ID: data})
    message = refusal(app)
    assert str(app / "runs" / APPROVED_ID / "note.txt") in message
    assert reason in message


def test_a_note_that_cannot_be_read_is_refused(tmp_path):
    app = make_app(tmp_path)
    (app / "runs" / APPROVED_ID / "note.txt").mkdir()
    assert "note.txt" in refusal(app)


def test_a_bad_note_exits_with_code_2(tmp_path):
    from hexperiment.cli import main

    app = make_app(tmp_path, notes={REJECTED_ID: b"one\ntwo\n"})
    out = tmp_path / "site"
    assert main(["gallery", str(app), "--out", str(out)]) == 2
    assert not out.exists()


def test_the_history_reader_ignores_note_files(tmp_path):
    app = make_app(tmp_path)
    before = runs_module.load_history(app)
    for run_id in (REJECTED_ID, APPROVED_ID):
        (app / "runs" / run_id / "note.txt").write_bytes(b"A note.\n")
    assert runs_module.load_history(app) == before
    (app / "runs" / "20261111-195099-00000a").mkdir()
    (app / "runs" / "20261111-195099-00000a" / "note.txt").write_bytes(b"x\n")
    assert runs_module.load_history(app) == before


# line endings


def test_a_crlf_app_and_an_lf_app_give_equal_sources(tmp_path):
    lf = read_app(
        write_app(tmp_path / "a", "demo", history(), haunted="a\nb\n", train="a\nc\n")
    )
    crlf = read_app(
        write_app(
            tmp_path / "b",
            "demo",
            history(),
            haunted="a\nb\n",
            train="a\nc\n",
            newline="\r\n",
        )
    )
    assert crlf.train_py == lf.train_py
    assert crlf.haunted_py == lf.haunted_py
    assert crlf.last_verdict_json == lf.last_verdict_json
    assert crlf.approved_card == lf.approved_card
    for value in (crlf.train_py, crlf.haunted_py):
        assert "\r" not in value
    for blob in (crlf.last_verdict_json, crlf.approved_card):
        assert b"\r" not in blob


def test_a_lone_cr_in_a_script_is_normalised(tmp_path):
    app = make_app(tmp_path, haunted="a = 1\n")
    (app / "train.py").write_bytes(b"a = 1\rb = 2\r\nc = 3\n")
    assert read_app(app).train_py == "a = 1\nb = 2\nc = 3\n"


# stylesheet


def test_the_stylesheet_has_lf_line_endings_and_is_not_empty():
    css = read_stylesheet()
    assert css
    assert b"\r" not in css


def test_the_stylesheet_is_listed_as_package_data():
    pyproject = Path(sources.__file__).resolve().parents[3] / "pyproject.toml"
    table = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    patterns = table["tool"]["setuptools"]["package-data"]["hexperiment.gallery"]
    assert any(PurePath("assets/gallery.css").full_match(p) for p in patterns)


# fonts

FONT_FILES = [
    "LICENSE-SpecialElite.txt",
    "OFL-Archivo.txt",
    "archivo-latin.woff2",
    "special-elite-latin.woff2",
]


def test_read_fonts_returns_the_four_files_sorted_by_name():
    names = [name for name, _ in read_fonts()]
    assert names == sorted(FONT_FILES)
    assert set(names) == set(FONT_FILES)


def test_the_woff2_files_are_woff2_and_the_licences_have_lf_endings():
    for name, data in read_fonts():
        if name.endswith(".woff2"):
            assert data.startswith(b"wOF2"), name
        else:
            assert data, name
            assert b"\r" not in data, name


def test_the_woff2_bytes_are_returned_as_they_are():
    folder = Path(sources.__file__).parent / "assets" / "fonts"
    for name, data in read_fonts():
        if name.endswith(".woff2"):
            assert data == (folder / name).read_bytes()


def fake_package(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(sources.resources, "files", lambda package: root)


def test_a_missing_fonts_folder_is_refused(tmp_path, monkeypatch):
    (tmp_path / "assets").mkdir()
    fake_package(monkeypatch, tmp_path)
    with pytest.raises(HexInputError, match="fonts are missing") as caught:
        read_fonts()
    assert "Reinstall the package" in str(caught.value)


def test_a_fonts_folder_without_a_woff2_file_is_refused(tmp_path, monkeypatch):
    fonts = tmp_path / "assets" / "fonts"
    fonts.mkdir(parents=True)
    (fonts / "OFL-Archivo.txt").write_bytes(b"licence\n")
    fake_package(monkeypatch, tmp_path)
    with pytest.raises(HexInputError, match="fonts are missing"):
        read_fonts()


def test_licence_text_is_normalised(tmp_path, monkeypatch):
    fonts = tmp_path / "assets" / "fonts"
    fonts.mkdir(parents=True)
    (fonts / "b.woff2").write_bytes(b"wOF2\r\n")
    (fonts / "a.txt").write_bytes(b"one\r\ntwo\rthree\n")
    fake_package(monkeypatch, tmp_path)
    assert read_fonts() == (("a.txt", b"one\ntwo\nthree\n"), ("b.woff2", b"wOF2\r\n"))


# benchmark


def test_a_missing_benchmark_file_is_refused(tmp_path):
    with pytest.raises(HexInputError, match="cannot be read"):
        read_benchmark(tmp_path / "nope.json")


def test_a_benchmark_file_that_is_not_utf8_is_refused(tmp_path):
    path = tmp_path / "results.json"
    path.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(HexInputError, match="not UTF-8"):
        read_benchmark(path)


def test_a_valid_benchmark_file_is_read_with_a_bom(tmp_path):
    path = tmp_path / "results.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(valid()).encode("utf-8"))
    assert read_benchmark(path).day == "2026-11-11"


def test_a_wrong_benchmark_file_names_its_path(tmp_path):
    path = tmp_path / "results.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(HexInputError, match="results.json"):
        read_benchmark(path)


# nothing is written


def test_reading_leaves_the_app_folder_byte_identical(tmp_path):
    app = make_app(tmp_path, haunted="a = 1\n")
    before = tree_hash(app)
    read_app(app)
    assert tree_hash(app) == before
    assert not (app / ".hex").exists()
