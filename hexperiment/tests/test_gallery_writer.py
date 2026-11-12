import hashlib
import os
from pathlib import Path

import pytest

from hexperiment.errors import HexInputError
from hexperiment.gallery import writer
from hexperiment.gallery.writer import (
    install_new,
    remove_old,
    set_old_aside,
    stage_new,
    temporary_names,
    write_site,
)

MARKER = ".hexperiment-gallery"
NEW_FILES = {
    MARKER: b"marker\n",
    "index.html": b"<p>new</p>\n",
    "app/index.html": b"line\n",
    "app/verdict.json": b"{}\n",
}
OLD_FILES = {MARKER: b"marker\n", "index.html": b"<p>old</p>\n", "only-old.txt": b"x"}


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def tree_hash(root: Path) -> dict[str, str]:
    return {k: hashlib.sha256(v).hexdigest() for k, v in snapshot(root).items()}


def make_old(parent: Path, files=OLD_FILES) -> Path:
    out = parent / "site"
    for key, data in files.items():
        (out / key).parent.mkdir(parents=True, exist_ok=True)
        (out / key).write_bytes(data)
    return out


def leftovers(parent: Path) -> list[str]:
    return sorted(p.name for p in parent.iterdir() if p.name != "site")


def fail_rename(monkeypatch, *, when, error=None, times=None):
    """Make ``os.rename`` raise ``error`` for the calls where ``when(src, dst)``."""
    error = OSError("boom") if error is None else error
    real = os.rename
    calls = {"n": 0}

    def fake(src, dst, *args, **kwargs):
        if when(Path(src), Path(dst)) and (times is None or calls["n"] < times):
            calls["n"] += 1
            raise error
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(writer.os, "rename", fake)
    return calls


def is_install(src, dst):
    return ".new-" in src.name and dst.name == "site"


def is_aside(src, dst):
    return src.name == "site" and ".old-" in dst.name


def is_restore(src, dst):
    return ".old-" in src.name and dst.name == "site"


# the normal cases


def test_a_new_folder_is_written_as_given(tmp_path):
    out = tmp_path / "site"
    assert write_site(NEW_FILES, out) == ""
    assert snapshot(out) == NEW_FILES
    assert leftovers(tmp_path) == []


def test_bytes_are_written_without_changing_line_endings(tmp_path):
    out = tmp_path / "site"
    files = {MARKER: b"m", "crlf.txt": b"a\r\nb\r\n", "lf.txt": b"a\nb\n"}
    write_site(files, out)
    assert snapshot(out) == files


def test_an_empty_existing_folder_is_accepted(tmp_path):
    out = tmp_path / "site"
    out.mkdir()
    write_site(NEW_FILES, out)
    assert snapshot(out) == NEW_FILES
    assert leftovers(tmp_path) == []


def test_a_marked_folder_is_replaced_and_old_only_files_are_gone(tmp_path):
    out = make_old(tmp_path)
    write_site(NEW_FILES, out)
    assert snapshot(out) == NEW_FILES
    assert leftovers(tmp_path) == []


def test_files_are_written_sorted(tmp_path, monkeypatch):
    order: list[str] = []
    real = Path.write_bytes

    def spy(self, data):
        order.append(
            "/".join(self.parts[-2:]) if self.parent.name == "app" else self.name
        )
        return real(self, data)

    monkeypatch.setattr(Path, "write_bytes", spy)
    write_site(NEW_FILES, tmp_path / "site")
    assert order == sorted(NEW_FILES)


def test_a_folder_without_the_marker_is_refused_and_left_alone(tmp_path):
    out = make_old(tmp_path, {"mine.txt": b"precious"})
    before = tree_hash(out)
    with pytest.raises(HexInputError, match="was not built by hexperiment gallery"):
        write_site(NEW_FILES, out)
    assert tree_hash(out) == before
    assert leftovers(tmp_path) == []


# bad keys


@pytest.mark.parametrize(
    "key", ["../x", "/x", "C:/x", "a/../b", "a\\b", "", "a//b", "./x"]
)
def test_a_bad_key_raises_before_anything_is_written(tmp_path, key):
    with pytest.raises(ValueError, match="not allowed"):
        write_site({MARKER: b"m", key: b"x"}, tmp_path / "site")
    assert list(tmp_path.iterdir()) == []


# the steps, one by one


def test_after_w2_out_dir_is_missing_and_old_and_new_are_whole(tmp_path):
    out = make_old(tmp_path)
    new, old = temporary_names(out)
    stage_new(NEW_FILES, new)
    assert set_old_aside(out, old) is True
    assert not out.exists()
    assert snapshot(old) == OLD_FILES
    assert snapshot(new) == NEW_FILES


def test_a_new_build_after_a_stop_at_w2_succeeds_and_leaves_the_leftovers(tmp_path):
    out = make_old(tmp_path)
    new, old = temporary_names(out)
    stage_new(NEW_FILES, new)
    set_old_aside(out, old)
    write_site({MARKER: b"m", "index.html": b"fresh"}, out)
    assert snapshot(out) == {MARKER: b"m", "index.html": b"fresh"}
    assert snapshot(old) == OLD_FILES
    assert snapshot(new) == NEW_FILES


def test_the_steps_finish_the_swap(tmp_path):
    out = make_old(tmp_path)
    new, old = temporary_names(out)
    stage_new(NEW_FILES, new)
    set_old_aside(out, old)
    install_new(new, out)
    assert remove_old(old) == ""
    assert snapshot(out) == NEW_FILES
    assert leftovers(tmp_path) == []


def test_the_temporary_names_are_hidden_siblings(tmp_path):
    new, old = temporary_names(tmp_path / "site")
    assert new.parent == old.parent == tmp_path
    assert new.name.startswith(".site.new-")
    assert old.name.startswith(".site.old-")
    assert new.name.split("-")[-1] == old.name.split("-")[-1]


# injected failures


def test_a_failure_at_w1_leaves_the_previous_gallery(tmp_path, monkeypatch):
    out = make_old(tmp_path)
    before = tree_hash(out)
    real = Path.write_bytes

    def fail(self, data):
        if self.name == "verdict.json":
            raise OSError("disk full")
        return real(self, data)

    monkeypatch.setattr(Path, "write_bytes", fail)
    with pytest.raises(OSError, match="disk full"):
        write_site(NEW_FILES, out)
    assert tree_hash(out) == before
    assert leftovers(tmp_path) == []


def test_a_failure_at_w2_leaves_the_previous_gallery(tmp_path, monkeypatch):
    out = make_old(tmp_path)
    before = tree_hash(out)
    fail_rename(monkeypatch, when=is_aside)
    with pytest.raises(OSError, match="boom"):
        write_site(NEW_FILES, out)
    assert tree_hash(out) == before
    assert leftovers(tmp_path) == []


def test_a_failure_at_w3_restores_the_previous_gallery(tmp_path, monkeypatch):
    out = make_old(tmp_path)
    before = tree_hash(out)
    fail_rename(monkeypatch, when=is_install)
    with pytest.raises(OSError, match="boom"):
        write_site(NEW_FILES, out)
    assert tree_hash(out) == before
    assert leftovers(tmp_path) == []


def test_an_interrupt_at_w3_restores_the_previous_gallery(tmp_path, monkeypatch):
    out = make_old(tmp_path)
    before = tree_hash(out)
    fail_rename(monkeypatch, when=is_install, error=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        write_site(NEW_FILES, out)
    assert tree_hash(out) == before
    assert leftovers(tmp_path) == []


def test_a_failing_restore_names_old_and_new(tmp_path, monkeypatch):
    out = make_old(tmp_path)
    fail_rename(monkeypatch, when=lambda s, d: is_install(s, d) or is_restore(s, d))
    with pytest.raises(HexInputError) as caught:
        write_site(NEW_FILES, out)
    message = str(caught.value)
    assert ".site.old-" in message
    assert ".site.new-" in message
    olds = [p for p in tmp_path.iterdir() if ".old-" in p.name]
    news = [p for p in tmp_path.iterdir() if ".new-" in p.name]
    assert len(olds) == 1
    assert len(news) == 1
    assert snapshot(olds[0]) == OLD_FILES
    assert snapshot(news[0]) == NEW_FILES


# retry


def test_a_permission_error_is_retried(tmp_path, monkeypatch):
    out = make_old(tmp_path)
    sleeps: list[float] = []
    monkeypatch.setattr(writer.time, "sleep", sleeps.append)
    calls = fail_rename(
        monkeypatch, when=is_aside, error=PermissionError("in use"), times=2
    )
    write_site(NEW_FILES, out)
    assert calls["n"] == 2
    assert sleeps == [writer.RETRY_PAUSE_SECONDS] * 2
    assert snapshot(out) == NEW_FILES


def test_a_permission_error_that_stays_is_reported_after_five_retries(
    tmp_path, monkeypatch
):
    out = make_old(tmp_path)
    before = tree_hash(out)
    sleeps: list[float] = []
    monkeypatch.setattr(writer.time, "sleep", sleeps.append)
    fail_rename(monkeypatch, when=is_install, error=PermissionError("in use"))
    with pytest.raises(HexInputError, match="could not be replaced"):
        write_site(NEW_FILES, out)
    assert len(sleeps) == 5
    assert tree_hash(out) == before
    assert leftovers(tmp_path) == []


# W4


def test_a_failure_at_w4_is_a_warning_not_an_error(tmp_path, monkeypatch):
    out = make_old(tmp_path)
    real = writer.shutil.rmtree

    def fail(path, *args, **kwargs):
        if ".old-" in Path(path).name:
            raise OSError("locked")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(writer.shutil, "rmtree", fail)
    warning = write_site(NEW_FILES, out)
    assert warning.startswith("hexperiment: warning:")
    assert ".site.old-" in warning
    assert "delete it by hand" in warning
    assert snapshot(out) == NEW_FILES


# nothing else is touched


def test_nothing_outside_out_dir_is_touched(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "other.txt").write_bytes(b"keep")
    (parent / "sibling").mkdir()
    (parent / "sibling" / "f.txt").write_bytes(b"keep too")
    before = tree_hash(parent)
    write_site(NEW_FILES, parent / "site")
    write_site(NEW_FILES, parent / "site")
    after = tree_hash(parent)
    kept = {k: v for k, v in after.items() if not k.startswith("site/")}
    assert kept == before
