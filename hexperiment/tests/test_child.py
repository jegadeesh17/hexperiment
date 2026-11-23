"""The bootstrap that starts train.py and counts opens of the data file."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from hexperiment import canary, child
from hexperiment.child import normalize_path, opens_data_file


@pytest.fixture
def data_file(tmp_path, monkeypatch) -> str:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.csv").write_text("a\n1\n", encoding="utf-8")
    return normalize_path(tmp_path / "data.csv")


# --- opens_data_file ---------------------------------------------------------------


def test_a_relative_path_is_resolved_against_the_working_directory(data_file):
    assert opens_data_file("data.csv", data_file)
    assert opens_data_file("./data.csv", data_file)
    assert opens_data_file("sub/../data.csv", data_file)


def test_an_absolute_path_matches(data_file, tmp_path):
    assert opens_data_file(str(tmp_path / "data.csv"), data_file)


def test_a_bytes_path_matches(data_file):
    assert opens_data_file(b"data.csv", data_file)
    assert opens_data_file(os.fsencode(os.path.abspath("data.csv")), data_file)


def test_a_path_like_matches(data_file, tmp_path):
    assert opens_data_file(Path("data.csv"), data_file)
    assert opens_data_file(tmp_path / "data.csv", data_file)


def test_a_file_descriptor_never_matches(data_file):
    assert not opens_data_file(0, data_file)
    assert not opens_data_file(3, data_file)
    assert not opens_data_file(True, data_file)


def test_another_file_does_not_match(data_file, tmp_path):
    assert not opens_data_file("lookup.csv", data_file)
    assert not opens_data_file(tmp_path / "data.csv.bak", data_file)
    assert not opens_data_file(tmp_path / "other" / "data.csv", data_file)
    assert not opens_data_file(None, data_file)
    assert not opens_data_file(1.5, data_file)


@pytest.mark.skipif(os.name != "nt", reason="Windows paths ignore case")
def test_the_case_of_a_path_is_ignored_on_windows(data_file):
    assert opens_data_file("DATA.CSV", data_file)


def test_a_symbolic_link_to_the_data_file_matches(data_file, tmp_path):
    link = tmp_path / "link.csv"
    try:
        link.symlink_to(tmp_path / "data.csv")
    except OSError:
        pytest.skip("symbolic links are not available")
    assert opens_data_file("link.csv", data_file)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(name=st.text(alphabet="abcdefgh_.", min_size=1, max_size=12))
def test_a_name_other_than_the_data_file_never_matches(data_file, name):
    expected = os.path.normcase(name) == os.path.normcase("data.csv")
    assert opens_data_file(name, data_file) == expected


# --- without a watch ---------------------------------------------------------------


def test_without_a_watch_nothing_is_counted_and_own_read_is_a_no_op():
    assert child.data_reads_outside_load() == 0
    with child.own_read():
        pass
    assert child.data_reads_outside_load() == 0


# --- the process ------------------------------------------------------------------


def run_child(
    app: Path, script: str, bootstrap: str = "import hexperiment.child as c; c.main()"
) -> subprocess.CompletedProcess[str]:
    (app / "train.py").write_text(textwrap.dedent(script), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-c", bootstrap, "train.py"],
        cwd=app,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


@pytest.fixture
def app(tmp_path) -> Path:
    (tmp_path / "hex.toml").write_text('[data]\npath = "data.csv"\n', encoding="utf-8")
    (tmp_path / "data.csv").write_text("a\n1\n", encoding="utf-8")
    (tmp_path / "other.csv").write_text("a\n2\n", encoding="utf-8")
    return tmp_path


def test_the_script_sees_a_plain_run(app):
    done = run_child(
        app,
        """
        import os, sys
        print(__name__, __file__, sys.argv, sys.path[0] == os.getcwd())
        """,
    )
    assert done.returncode == 0
    assert done.stdout.split() == [
        "__main__",
        "train.py",
        "['train.py']",
        "True",
    ]


def test_the_exit_code_of_the_script_is_kept(app):
    assert run_child(app, "import sys; sys.exit(7)").returncode == 7


def test_an_exception_gives_exit_code_1_and_a_traceback_of_the_script_alone(app):
    done = run_child(app, "x = 1\nraise ValueError('boom')\n")
    assert done.returncode == 1
    assert "train.py\", line 2, in <module>" in done.stderr
    assert "ValueError: boom" in done.stderr
    assert "runpy" not in done.stderr
    assert "child.py" not in done.stderr


def test_the_count_ignores_other_files_and_counts_opens_of_the_data_file(app):
    script = """
        import hexperiment.child as child
        open("other.csv").close()
        open(1, "w", closefd=False).close()
        print(child.data_reads_outside_load())
        open("data.csv").close()
        print(child.data_reads_outside_load() > 0)
        with child.own_read():
            before = child.data_reads_outside_load()
            open("data.csv").close()
            print(child.data_reads_outside_load() == before)
        """
    done = run_child(app, script)
    assert done.stdout.split() == ["0", "True", "True"]


def test_the_hook_never_raises_for_unusual_open_arguments(app):
    done = run_child(
        app,
        """
        import sys
        sys.audit("open", object(), None, None)
        sys.audit("open")
        print("done")
        """,
    )
    assert done.returncode == 0
    assert done.stdout.strip() == "done"


def test_a_missing_or_broken_config_starts_the_script_without_a_watch(app):
    (app / "hex.toml").write_text("not = [valid", encoding="utf-8")
    done = run_child(
        app,
        """
        import hexperiment.child as child
        open("data.csv").close()
        print(child.data_reads_outside_load())
        """,
    )
    assert done.returncode == 0
    assert done.stdout.strip() == "0"


@pytest.mark.parametrize("module", ["tomllib", "runpy", "traceback", "threading"])
def test_an_app_file_named_like_a_standard_module_does_not_switch_the_watch_off(
    app, module
):
    (app / f"{module}.py").write_text("raise RuntimeError('shadowed')\n")
    done = run_child(
        app,
        """
        import hexperiment.child as child
        open("data.csv").close()
        print(child.data_reads_outside_load())
        """,
        bootstrap=canary._BOOTSTRAP,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "1"


def test_the_app_folder_is_first_on_the_path_of_the_script_in_the_real_bootstrap(app):
    done = run_child(
        app,
        """
        import os, sys
        print(sys.path[0] == os.getcwd(), "" in sys.path)
        """,
        bootstrap=canary._BOOTSTRAP,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["True", "False"]


def test_starting_the_bootstrap_loads_neither_pandas_nor_mcp(app):
    done = run_child(
        app,
        """
        import sys
        print([m for m in ("pandas", "numpy", "sklearn", "mcp") if m in sys.modules])
        """,
    )
    assert done.stdout.strip() == "[]"


@pytest.mark.parametrize(
    "event",
    [
        "shutil.copyfile",
        "_winapi.CopyFile2",
        "os.link",
        "os.symlink",
        "os.rename",
        "os.replace",
    ],
)
def test_a_copy_link_or_move_of_the_data_file_is_counted(app, event):
    done = run_child(
        app,
        f"""
        import hexperiment.child as child
        sys_audit = __import__("sys").audit
        sys_audit("other.event", "data.csv", "copy.csv")
        print(child.data_reads_outside_load())
        sys_audit({event!r}, "other.csv", "copy.csv")
        print(child.data_reads_outside_load())
        sys_audit({event!r}, "data.csv", "copy.csv")
        print(child.data_reads_outside_load())
        sys_audit({event!r}, 3, "copy.csv")
        sys_audit({event!r}, None)
        sys_audit({event!r})
        print(child.data_reads_outside_load())
        """,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["0", "0", "1", "1"]


def test_the_real_calls_that_copy_link_or_move_the_data_file_are_counted(app):
    done = run_child(
        app,
        """
        import os, shutil
        import hexperiment.child as child
        counts = []
        shutil.copy2("data.csv", "c1.csv")
        counts.append(child.data_reads_outside_load())
        shutil.copyfile("data.csv", "c2.csv")
        counts.append(child.data_reads_outside_load())
        os.link("data.csv", "l.csv")
        counts.append(child.data_reads_outside_load())
        os.replace("data.csv", "moved.csv")
        counts.append(child.data_reads_outside_load())
        print([b > a for a, b in zip([0] + counts, counts)])
        """,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[True, True, True, True]"


def test_an_open_in_another_thread_while_load_runs_is_counted(app):
    done = run_child(
        app,
        """
        import threading
        import hexperiment.child as child

        def read_in_thread():
            open("data.csv").close()

        with child.own_read():
            worker = threading.Thread(target=read_in_thread)
            worker.start()
            worker.join()
            own = child.data_reads_outside_load()
            open("data.csv").close()
            print(own, child.data_reads_outside_load() == own)
        """,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["1", "True"]
