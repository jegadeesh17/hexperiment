"""Tests of the workspace guard (Requirements 1.2, 4.2, 4.3, 4.5).

Real hook events carry ``c:\\...`` while the working directory is ``C:\\...``;
the Windows tests below use both spellings.
"""

import os
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hexperiment.errors import HexInputError
from hexperiment.workspace import (
    ENV_WORKSPACE,
    AppRef,
    discover_apps,
    find_app_for_file,
    is_app,
    relative_posix,
    resolve_app,
    resolve_inside,
    resolve_workspace,
)

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

windows_only = pytest.mark.skipif(
    os.name != "nt", reason="drive letters are case-insensitive only on Windows"
)


def make_app(folder: Path, name: str | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "hex.toml").write_text(
        HEX_TOML.format(name=name or folder.name), encoding="utf-8"
    )
    (folder / "train.py").write_text("print('train')\n", encoding="utf-8")
    return folder


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x = 1\n", encoding="utf-8")
    return path


def link_to(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"this platform refuses to create symbolic links ({error})")


def with_lower_drive(path: Path) -> str:
    text = str(path)
    return text[0].lower() + text[1:]


# --- resolve_workspace -------------------------------------------------------------


def test_the_working_directory_is_the_workspace_when_nothing_is_set(tmp_path):
    assert resolve_workspace({}, tmp_path) == tmp_path.resolve()


def test_an_empty_variable_counts_as_not_set(tmp_path):
    assert resolve_workspace({ENV_WORKSPACE: "  "}, tmp_path) == tmp_path.resolve()


def test_the_variable_overrides_the_working_directory(tmp_path):
    chosen = tmp_path / "chosen"
    chosen.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    assert resolve_workspace({ENV_WORKSPACE: str(chosen)}, other) == chosen.resolve()


def test_a_relative_variable_is_taken_relative_to_the_working_directory(tmp_path):
    (tmp_path / "sub").mkdir()
    assert (
        resolve_workspace({ENV_WORKSPACE: "sub"}, tmp_path)
        == (tmp_path / "sub").resolve()
    )


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_a_variable_that_is_not_a_folder_names_the_variable(kind, tmp_path):
    target = tmp_path / "nothing"
    if kind == "file":
        target = touch(tmp_path / "a_file.txt")
    with pytest.raises(HexInputError) as caught:
        resolve_workspace({ENV_WORKSPACE: str(target)}, tmp_path)
    assert ENV_WORKSPACE in str(caught.value)
    assert "not an existing folder" in str(caught.value)


def test_the_defaults_read_the_environment_and_the_working_directory(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(ENV_WORKSPACE, raising=False)
    monkeypatch.chdir(tmp_path)
    assert resolve_workspace() == tmp_path.resolve()
    (tmp_path / "ws").mkdir()
    monkeypatch.setenv(ENV_WORKSPACE, "ws")
    assert resolve_workspace() == (tmp_path / "ws").resolve()


# --- is_app and relative_posix -----------------------------------------------------


def test_a_folder_with_both_files_is_an_app(tmp_path):
    assert is_app(make_app(tmp_path / "x"))


@pytest.mark.parametrize("missing", ["hex.toml", "train.py"])
def test_a_folder_missing_one_file_is_not_an_app(missing, tmp_path):
    app = make_app(tmp_path / "x")
    (app / missing).unlink()
    assert not is_app(app)


def test_a_folder_named_train_py_does_not_make_an_app(tmp_path):
    app = tmp_path / "x"
    app.mkdir()
    (app / "hex.toml").write_text("", encoding="utf-8")
    (app / "train.py").mkdir()
    assert not is_app(app)


def test_relative_posix_uses_forward_slashes_and_dot_for_the_workspace(tmp_path):
    app = make_app(tmp_path / "apps" / "churn")
    assert relative_posix(app, tmp_path) == "apps/churn"
    assert relative_posix(tmp_path, tmp_path) == "."


def test_relative_posix_refuses_a_path_outside_the_workspace(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    with pytest.raises(HexInputError, match="not inside the workspace"):
        relative_posix(tmp_path, workspace)


# --- find_app_for_file -------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


def test_a_file_at_the_app_root_finds_the_app(workspace):
    app = make_app(workspace / "apps" / "churn")
    assert find_app_for_file(app / "train.py", workspace) == app.resolve()


def test_a_nested_helper_file_finds_the_enclosing_app(workspace):
    app = make_app(workspace / "apps" / "churn")
    helper = touch(app / "features" / "deep" / "encode.py")
    assert find_app_for_file(helper, workspace) == app.resolve()


def test_a_file_that_does_not_exist_yet_still_finds_the_app(workspace):
    app = make_app(workspace / "apps" / "churn")
    assert find_app_for_file(app / "new_module.py", workspace) == app.resolve()


def test_the_nearest_app_wins_over_an_outer_one(workspace):
    outer = make_app(workspace / "apps" / "outer")
    inner = make_app(outer / "sub" / "inner")
    assert find_app_for_file(inner / "train.py", workspace) == inner.resolve()
    assert find_app_for_file(outer / "other.py", workspace) == outer.resolve()


def test_an_app_at_the_workspace_root_is_found(workspace):
    make_app(workspace)
    helper = touch(workspace / "lib" / "util.py")
    assert find_app_for_file(helper, workspace) == workspace.resolve()


def test_a_file_in_no_app_gives_none(workspace):
    make_app(workspace / "apps" / "churn")
    assert (
        find_app_for_file(touch(workspace / "scripts" / "tool.py"), workspace) is None
    )


def test_the_walk_never_goes_above_the_workspace(workspace):
    make_app(workspace.parent)
    assert find_app_for_file(touch(workspace / "tool.py"), workspace) is None


def test_a_file_outside_the_workspace_gives_none_even_inside_an_app(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    foreign = make_app(tmp_path / "foreign")
    assert find_app_for_file(foreign / "train.py", workspace) is None


def test_the_workspace_itself_is_not_a_file(workspace):
    make_app(workspace.parent)
    assert find_app_for_file(workspace, workspace) is None


def test_dot_dot_segments_that_leave_the_workspace_give_none(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    make_app(tmp_path / "foreign")
    sneaky = workspace / ".." / "foreign" / "train.py"
    assert sneaky.exists()
    assert find_app_for_file(sneaky, workspace) is None


def test_a_relative_path_is_taken_relative_to_the_workspace(workspace):
    app = make_app(workspace / "apps" / "churn")
    assert find_app_for_file(Path("apps/churn/train.py"), workspace) == app.resolve()
    assert find_app_for_file(Path("apps/churn/new.py"), workspace) == app.resolve()


def test_a_relative_path_does_not_use_the_process_working_directory(
    workspace, tmp_path, monkeypatch
):
    make_app(workspace / "apps" / "churn")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    found = find_app_for_file(Path("apps/churn/train.py"), workspace)
    assert found == (workspace / "apps" / "churn").resolve()


def test_a_symlinked_file_that_points_out_of_the_workspace_gives_none(tmp_path):
    workspace = tmp_path / "project"
    app = make_app(workspace / "apps" / "churn")
    outside = touch(make_app(tmp_path / "foreign") / "evil.py")
    link_to(app / "linked.py", outside)
    assert find_app_for_file(app / "linked.py", workspace) is None


def test_a_symlinked_folder_that_points_out_of_the_workspace_gives_none(tmp_path):
    workspace = tmp_path / "project"
    (workspace / "apps").mkdir(parents=True)
    foreign = make_app(tmp_path / "foreign")
    link_to(workspace / "apps" / "door", foreign)
    assert (
        find_app_for_file(workspace / "apps" / "door" / "train.py", workspace) is None
    )


def test_a_symlink_inside_the_workspace_finds_the_app_at_its_target(tmp_path):
    workspace = tmp_path / "project"
    app = make_app(workspace / "apps" / "churn")
    link_to(workspace / "alias", app)
    found = find_app_for_file(workspace / "alias" / "train.py", workspace)
    assert found == app.resolve()


@pytest.mark.skipif(os.name != "nt", reason="junctions exist only on Windows")
def test_a_junction_that_points_out_of_the_workspace_gives_none(tmp_path):
    import _winapi

    workspace = tmp_path / "project"
    (workspace / "apps").mkdir(parents=True)
    foreign = make_app(tmp_path / "foreign")
    _winapi.CreateJunction(str(foreign), str(workspace / "apps" / "door"))
    assert (
        find_app_for_file(workspace / "apps" / "door" / "train.py", workspace) is None
    )


@pytest.mark.skipif(os.name != "nt", reason="backslash is a separator only on Windows")
def test_a_windows_path_with_backslashes_finds_the_app(workspace):
    app = make_app(workspace / "apps" / "churn")
    helper = touch(app / "features" / "encode.py")
    relative = "apps\\churn\\features\\encode.py"
    assert find_app_for_file(Path(relative), workspace) == app.resolve()
    assert find_app_for_file(Path(str(helper)), workspace) == app.resolve()


@windows_only
def test_a_lower_case_drive_letter_in_the_file_matches_the_workspace(workspace):
    app = make_app(workspace / "apps" / "churn")
    event_path = Path(with_lower_drive(app / "train.py"))
    upper_workspace = Path(str(workspace)[0].upper() + str(workspace)[1:])
    assert find_app_for_file(event_path, upper_workspace) == app.resolve()


@windows_only
def test_a_fully_lower_case_file_path_matches_the_workspace(workspace):
    app = make_app(workspace / "apps" / "churn")
    event_path = Path(str(app / "train.py").lower())
    assert find_app_for_file(event_path, workspace) is not None
    missing = Path(str(app / "New_Module.py").lower())
    assert not missing.exists()
    assert find_app_for_file(missing, workspace) is not None


@windows_only
def test_a_lower_case_workspace_matches_an_upper_case_file(workspace):
    app = make_app(workspace / "apps" / "churn")
    lower_workspace = Path(str(workspace).lower())
    assert find_app_for_file(app / "train.py", lower_workspace) == app.resolve()


@windows_only
def test_the_real_event_spelling_is_accepted_and_a_foreign_drive_path_is_not(
    workspace, tmp_path
):
    app = make_app(workspace / "apps" / "churn")
    assert find_app_for_file(Path(with_lower_drive(app / "train.py")), workspace)
    sibling = make_app(tmp_path / "project2" / "apps" / "churn")
    assert (
        find_app_for_file(Path(with_lower_drive(sibling / "train.py")), workspace)
        is None
    )


@windows_only
def test_a_sibling_folder_that_shares_the_workspace_prefix_is_outside(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    sibling = make_app(tmp_path / "project-copy" / "apps" / "x")
    assert (
        find_app_for_file(Path(with_lower_drive(sibling / "train.py")), workspace)
        is None
    )


def test_a_sibling_folder_that_shares_the_workspace_prefix_is_outside_everywhere(
    tmp_path,
):
    workspace = tmp_path / "project"
    workspace.mkdir()
    sibling = make_app(tmp_path / "project-copy" / "apps" / "x")
    assert find_app_for_file(sibling / "train.py", workspace) is None


@windows_only
def test_relative_posix_accepts_the_other_drive_letter_case(workspace):
    app = make_app(workspace / "apps" / "churn")
    assert relative_posix(Path(with_lower_drive(app)), workspace) == "apps/churn"


# --- resolve_app -------------------------------------------------------------------


def test_resolve_app_returns_the_resolved_folder(workspace):
    app = make_app(workspace / "apps" / "churn")
    assert resolve_app(workspace, "apps/churn") == app.resolve()


def test_resolve_app_accepts_an_absolute_path_inside_the_workspace(workspace):
    app = make_app(workspace / "apps" / "churn")
    assert resolve_app(workspace, str(app)) == app.resolve()


def test_resolve_app_accepts_a_name_with_a_space(workspace):
    app = make_app(workspace / "apps" / "my model")
    assert resolve_app(workspace, "apps/my model") == app.resolve()


def test_resolve_app_accepts_the_workspace_itself_when_it_is_an_app(workspace):
    make_app(workspace)
    assert resolve_app(workspace, ".") == workspace.resolve()


def test_resolve_app_accepts_dot_dot_that_stays_inside(workspace):
    app = make_app(workspace / "apps" / "churn")
    (workspace / "apps" / "other").mkdir()
    assert resolve_app(workspace, "apps/other/../churn") == app.resolve()


@pytest.mark.parametrize("app", ["", "   ", "apps/x\x00y"])
def test_resolve_app_refuses_an_unusable_name(app, workspace):
    with pytest.raises(HexInputError, match="not a usable folder name"):
        resolve_app(workspace, app)


def test_resolve_app_refuses_a_value_that_is_not_text(workspace):
    with pytest.raises(HexInputError, match="not a usable folder name"):
        resolve_app(workspace, 5)  # type: ignore[arg-type]


def test_resolve_app_refuses_dot_dot_that_leaves_the_workspace(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    make_app(tmp_path / "foreign")
    with pytest.raises(HexInputError) as caught:
        resolve_app(workspace, "../foreign")
    message = str(caught.value)
    assert "'../foreign'" in message
    assert "outside the workspace" in message
    assert "Pass an app folder below the workspace" in message


def test_resolve_app_refuses_an_absolute_path_outside_the_workspace(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    foreign = make_app(tmp_path / "foreign")
    with pytest.raises(HexInputError, match="outside the workspace"):
        resolve_app(workspace, str(foreign))


def test_resolve_app_refuses_a_sibling_that_shares_the_workspace_prefix(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    sibling = make_app(tmp_path / "project-copy")
    with pytest.raises(HexInputError, match="outside the workspace"):
        resolve_app(workspace, str(sibling))


def test_resolve_app_refuses_a_link_that_leaves_the_workspace(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    foreign = make_app(tmp_path / "foreign")
    link_to(workspace / "door", foreign)
    with pytest.raises(HexInputError, match="outside the workspace"):
        resolve_app(workspace, "door")


def test_resolve_app_refuses_a_folder_that_does_not_exist(workspace):
    with pytest.raises(HexInputError) as caught:
        resolve_app(workspace, "apps/nowhere")
    message = str(caught.value)
    assert "'apps/nowhere'" in message
    assert "is not a folder" in message
    assert "list_apps" in message


def test_resolve_app_refuses_a_file(workspace):
    touch(workspace / "apps" / "file.py")
    with pytest.raises(HexInputError, match="is not a folder"):
        resolve_app(workspace, "apps/file.py")


@pytest.mark.parametrize("missing", ["hex.toml", "train.py"])
def test_resolve_app_names_the_missing_file(missing, workspace):
    app = make_app(workspace / "apps" / "churn")
    (app / missing).unlink()
    with pytest.raises(HexInputError) as caught:
        resolve_app(workspace, "apps/churn")
    message = str(caught.value)
    assert f"has no {missing}" in message
    assert "'apps/churn'" in message
    assert "hexperiment init" in message


@pytest.fixture(scope="module")
def shared_workspace(tmp_path_factory) -> Path:
    outer = tmp_path_factory.mktemp("outer")
    for name in ("a", "b", "x y"):
        make_app(outer / name)
    root = outer / "ws"
    make_app(root / "a")
    make_app(root / "a" / "b")
    make_app(root / "x y")
    (root / "b").mkdir()
    return root


@given(
    segments=st.lists(
        st.sampled_from(["a", "b", ".", "..", "x y"]), min_size=1, max_size=7
    )
)
def test_resolve_app_never_returns_a_path_outside_the_workspace(
    segments, shared_workspace
):
    root = shared_workspace.resolve()
    try:
        folder = resolve_app(shared_workspace, "/".join(segments))
    except HexInputError:
        return
    assert folder.is_relative_to(root)
    assert is_app(folder)


# --- discover_apps -----------------------------------------------------------------


def test_discover_apps_returns_nothing_for_an_empty_workspace(workspace):
    assert discover_apps(workspace) == ()


def test_discover_apps_lists_name_path_and_relative_path(workspace):
    app = make_app(workspace / "apps" / "churn", name="Churn model")
    (ref,) = discover_apps(workspace)
    assert ref == AppRef(
        name="Churn model", path=app.resolve(), relative="apps/churn", problem=None
    )


def test_discover_apps_sorts_by_relative_path(workspace):
    for folder in ("apps/zeta", "apps/alpha", "other/mid", "apps/beta"):
        make_app(workspace / folder)
    found = [app.relative for app in discover_apps(workspace)]
    assert found == ["apps/alpha", "apps/beta", "apps/zeta", "other/mid"]
    assert found == sorted(found)


def test_discover_apps_finds_an_app_at_the_workspace_root(workspace):
    make_app(workspace, name="root app")
    (ref,) = discover_apps(workspace)
    assert (ref.name, ref.relative) == ("root app", ".")


def test_discover_apps_skips_hidden_folders(workspace):
    make_app(workspace / ".hidden" / "app")
    make_app(workspace / "apps" / ".cache" / "app")
    visible = make_app(workspace / "apps" / "shown")
    assert [a.path for a in discover_apps(workspace)] == [visible.resolve()]


@pytest.mark.parametrize("name", ["node_modules", "__pycache__"])
def test_discover_apps_skips_folders_that_never_hold_apps(name, workspace):
    make_app(workspace / name / "app")
    make_app(workspace / "apps" / name / "app")
    assert discover_apps(workspace) == ()


def test_discover_apps_skips_virtual_environments(workspace):
    venv = workspace / "env"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
    make_app(venv / "lib" / "site" / "app")
    make_app(workspace / "apps" / "real")
    assert [a.relative for a in discover_apps(workspace)] == ["apps/real"]


def test_discover_apps_does_not_descend_into_an_app(workspace):
    outer = make_app(workspace / "apps" / "outer")
    make_app(outer / "nested")
    make_app(outer / "runs" / "copy")
    assert [a.relative for a in discover_apps(workspace)] == ["apps/outer"]


def test_a_broken_hex_toml_gives_a_problem_and_keeps_the_other_apps(workspace):
    broken = make_app(workspace / "apps" / "broken")
    (broken / "hex.toml").write_text("[app\n", encoding="utf-8")
    make_app(workspace / "apps" / "fine", name="Fine one")
    refs = {a.relative: a for a in discover_apps(workspace)}
    assert set(refs) == {"apps/broken", "apps/fine"}
    assert refs["apps/broken"].name == "broken"
    assert refs["apps/broken"].problem is not None
    assert "hex.toml" in refs["apps/broken"].problem
    assert refs["apps/fine"].problem is None
    assert refs["apps/fine"].name == "Fine one"


def test_discover_apps_does_not_follow_symbolic_links(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    foreign = make_app(tmp_path / "foreign")
    link_to(workspace / "door", foreign)
    assert discover_apps(workspace) == ()


@pytest.mark.skipif(os.name != "nt", reason="junctions exist only on Windows")
def test_discover_apps_does_not_follow_junctions(tmp_path):
    import _winapi

    workspace = tmp_path / "project"
    workspace.mkdir()
    foreign = make_app(tmp_path / "foreign")
    _winapi.CreateJunction(str(foreign), str(workspace / "door"))
    assert discover_apps(workspace) == ()


def test_discover_apps_ignores_folders_with_only_one_of_the_files(workspace):
    half = workspace / "apps" / "half"
    half.mkdir(parents=True)
    (half / "hex.toml").write_text(HEX_TOML.format(name="half"), encoding="utf-8")
    assert discover_apps(workspace) == ()


# --- resolve_inside ------------------------------------------------------------------


def test_resolve_inside_returns_the_resolved_path_of_a_file_that_does_not_exist(
    tmp_path,
):
    root = tmp_path / "project"
    root.mkdir()
    found = resolve_inside(root / "apps" / ".." / "verdict.json", root, "the file")
    assert found == (root / "verdict.json").resolve()


def test_resolve_inside_refuses_a_path_that_leaves_the_workspace(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    with pytest.raises(HexInputError, match="outside the workspace") as caught:
        resolve_inside(root / ".." / "other.json", root, "the verdict file")
    assert "the verdict file" in str(caught.value)


def test_resolve_inside_refuses_a_link_that_leaves_the_workspace(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    target = tmp_path / "other.json"
    target.write_text("{}", encoding="utf-8")
    try:
        (root / "link.json").symlink_to(target)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"this platform refuses to create file links ({error!r})")
    with pytest.raises(HexInputError, match="outside the workspace"):
        resolve_inside(root / "link.json", root, "the file")
