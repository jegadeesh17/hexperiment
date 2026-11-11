"""Tests of the SDK-free logic of the MCP tools (Requirements 4.3, 4.4, 4.5, 4.7).

Workspaces are temporary folders with prepared ``verdict.json`` and ``runs/``
files. A stub ``RunApp`` stands in for ``cli.run_app``: no test here starts a
training run.
"""

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hexperiment.errors import HexInputError
from hexperiment.ghosts import find_ghost, to_dict
from hexperiment.mcp_server import service
from hexperiment.models import RunOutcome, Severity, Status
from hexperiment.runs import load_history
from hexperiment.verdict import to_json
from service_builder import (
    NEW,
    OLD,
    StubRun,
    finding,
    make_app,
    make_verdict,
    tree_hash,
    write_app_verdict,
    write_run,
)


@pytest.fixture
def workspace(tmp_path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root.resolve()


# --- list_apps -----------------------------------------------------------------------


def test_list_apps_of_an_empty_workspace_has_a_hint_that_names_the_workspace(
    workspace,
):
    result = service.list_apps(workspace)
    assert result["workspace"] == str(workspace)
    assert result["apps"] == []
    assert str(workspace) in result["hint"]
    assert "hexperiment init --kiro" in result["hint"]
    assert ".kiro/settings/mcp.json" in result["hint"]
    assert "HEXPERIMENT_WORKSPACE" in result["hint"]


def test_list_apps_has_no_hint_when_an_app_is_found(workspace):
    make_app(workspace / "apps" / "churn")
    result = service.list_apps(workspace)
    assert len(result["apps"]) == 1
    assert "hint" not in result


def test_list_apps_has_a_hint_when_the_only_folder_is_not_an_app(workspace):
    (workspace / "apps" / "empty").mkdir(parents=True)
    result = service.list_apps(workspace)
    assert result["apps"] == []
    assert str(workspace) in result["hint"]


def test_list_apps_gives_name_path_status_and_problem_sorted_by_path(workspace):
    approved = make_app(workspace / "apps" / "b-app", name="Beta")
    make_app(workspace / "apps" / "a-app", name="Alpha")
    write_app_verdict(approved, make_verdict(status=Status.APPROVED))
    result = service.list_apps(workspace)
    assert result["workspace"] == str(workspace)
    assert result["apps"] == [
        {"name": "Alpha", "path": "apps/a-app", "status": "none", "problem": None},
        {
            "name": "Beta",
            "path": "apps/b-app",
            "status": "APPROVED",
            "problem": None,
        },
    ]


def test_the_status_is_the_current_verdict_not_the_newest_run(workspace):
    app = make_app(workspace / "apps" / "churn")
    write_run(app, make_verdict(OLD, status=Status.APPROVED))
    [entry] = service.list_apps(workspace)["apps"]
    assert entry["status"] == "none"


def test_an_unparsable_verdict_gives_status_none_and_a_problem(workspace):
    app = make_app(workspace / "apps" / "churn")
    (app / "verdict.json").write_text("{not json", encoding="utf-8")
    [entry] = service.list_apps(workspace)["apps"]
    assert entry["status"] == "none"
    assert "verdict.json" in entry["problem"]


def test_a_verdict_without_a_status_gives_status_none_and_a_problem(workspace):
    app = make_app(workspace / "apps" / "churn")
    (app / "verdict.json").write_text('{"status": 3}', encoding="utf-8")
    [entry] = service.list_apps(workspace)["apps"]
    assert entry["status"] == "none"
    assert "no status" in entry["problem"]


def test_a_verdict_json_that_is_a_folder_gives_status_none_and_a_problem(workspace):
    app = make_app(workspace / "apps" / "churn")
    (app / "verdict.json").mkdir()
    [entry] = service.list_apps(workspace)["apps"]
    assert entry["status"] == "none"
    assert entry["problem"]


def test_a_broken_hex_toml_gives_a_problem_and_keeps_the_other_apps(workspace):
    broken = make_app(workspace / "apps" / "broken")
    (broken / "hex.toml").write_text("this is [not toml", encoding="utf-8")
    make_app(workspace / "apps" / "fine", name="Fine")
    entries = service.list_apps(workspace)["apps"]
    assert [e["path"] for e in entries] == ["apps/broken", "apps/fine"]
    assert entries[0]["name"] == "broken"
    assert entries[0]["problem"]
    assert entries[1]["problem"] is None


def test_hidden_folders_virtual_environments_and_node_modules_are_skipped(workspace):
    make_app(workspace / "apps" / "churn")
    make_app(workspace / ".hidden" / "app")
    make_app(workspace / "node_modules" / "app")
    make_app(workspace / "env" / "app")
    (workspace / "env" / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
    assert [e["path"] for e in service.list_apps(workspace)["apps"]] == ["apps/churn"]


# --- get_verdict ---------------------------------------------------------------------


def test_get_verdict_returns_the_parsed_verdict_json_of_the_app(workspace):
    app = make_app(workspace / "apps" / "churn")
    verdict = make_verdict(status=Status.APPROVED)
    write_app_verdict(app, verdict)
    result = service.get_verdict(workspace, "apps/churn")
    assert result == {"found": True, "verdict": verdict.to_dict()}


def test_get_verdict_returns_the_app_verdict_not_the_newest_run(workspace):
    app = make_app(workspace / "apps" / "churn")
    write_app_verdict(app, make_verdict(OLD, status=Status.APPROVED))
    write_run(
        app,
        make_verdict(NEW, created_at="2026-01-02T00:00:00Z", status=Status.REJECTED),
    )
    result = service.get_verdict(workspace, "apps/churn")
    assert result["found"] is True
    assert result["verdict"]["status"] == "APPROVED"
    assert result["verdict"]["run_id"] == OLD


def test_get_verdict_without_a_verdict_says_so_even_when_runs_exist(workspace):
    app = make_app(workspace / "apps" / "churn")
    write_run(app, make_verdict(OLD, status=Status.APPROVED))
    result = service.get_verdict(workspace, "apps/churn")
    assert result["found"] is False
    assert "no current verdict" in result["message"]
    assert "verdict" not in result


@pytest.mark.parametrize("text", ["{not json", "[1, 2]", "3", ""])
def test_get_verdict_raises_for_a_file_that_is_not_a_json_object(workspace, text):
    app = make_app(workspace / "apps" / "churn")
    (app / "verdict.json").write_text(text, encoding="utf-8")
    with pytest.raises(HexInputError, match="verdict.json"):
        service.get_verdict(workspace, "apps/churn")


def test_get_verdict_raises_for_a_verdict_json_that_is_a_folder(workspace):
    app = make_app(workspace / "apps" / "churn")
    (app / "verdict.json").mkdir()
    with pytest.raises(HexInputError, match="verdict.json"):
        service.get_verdict(workspace, "apps/churn")


# --- get_history ---------------------------------------------------------------------


def test_get_history_lists_runs_oldest_first_with_status_metric_and_ghosts(workspace):
    app = make_app(workspace / "apps" / "churn")
    write_run(
        app,
        make_verdict(
            NEW,
            created_at="2026-01-02T00:00:00Z",
            status=Status.REJECTED,
            value=0.97,
            findings=(
                finding("target_proxy", "The Possession"),
                finding("row_overlap", "The Doppelganger"),
                finding("target_proxy", "The Possession", Severity.WAIVED),
                finding("metric", None, Severity.WARNING),
            ),
        ),
    )
    write_run(app, make_verdict(OLD, value=None))
    assert service.get_history(workspace, "apps/churn") == {
        "app": "apps/churn",
        "runs": [
            {"run_id": OLD, "status": "UNVERIFIED", "metric_value": None, "ghosts": []},
            {
                "run_id": NEW,
                "status": "REJECTED",
                "metric_value": 0.97,
                "ghosts": ["The Doppelganger", "The Possession"],
            },
        ],
    }


def test_get_history_of_an_app_without_runs_is_empty(workspace):
    make_app(workspace / "apps" / "churn")
    assert service.get_history(workspace, "apps/churn") == {
        "app": "apps/churn",
        "runs": [],
    }


def test_get_history_names_a_malformed_history_file(workspace):
    app = make_app(workspace / "apps" / "churn")
    folder = write_run(app, make_verdict(OLD))
    (folder / "verdict.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(HexInputError, match="verdict.json"):
        service.get_history(workspace, "apps/churn")


def history_cases():
    return st.lists(
        st.tuples(st.integers(0, 3), st.integers(0, 0xFFF)),
        unique=True,
        max_size=6,
    ).flatmap(lambda runs: st.permutations(runs))


@given(history_cases())
def test_get_history_keeps_the_order_of_load_history(runs):
    with tempfile.TemporaryDirectory() as folder:
        workspace = Path(folder).resolve()
        app = make_app(workspace / "app")
        for minute, token in runs:
            run_id = f"20260101-00{minute:02d}00-{token:06x}"
            created_at = f"2026-01-01T00:{minute:02d}:00Z"
            write_run(app, make_verdict(run_id, created_at=created_at))
        listed = [r["run_id"] for r in service.get_history(workspace, "app")["runs"]]
        expected = sorted(
            f"20260101-00{minute:02d}00-{token:06x}" for minute, token in runs
        )
        assert listed == [v.run_id for v in load_history(app)]
        assert listed == expected


# --- explain_ghost -------------------------------------------------------------------


def test_explain_ghost_returns_the_catalogue_entry():
    assert service.explain_ghost("The Possession") == to_dict(
        find_ghost("target_proxy")
    )


def test_explain_ghost_raises_for_an_unknown_name():
    with pytest.raises(HexInputError, match="banshee"):
        service.explain_ghost("banshee")


# --- check_audit_request -------------------------------------------------------------


def test_check_audit_request_returns_the_folder_and_the_mode(workspace):
    app = make_app(workspace / "apps" / "churn")
    assert service.check_audit_request(workspace, "apps/churn", "full") == (app, True)
    assert service.check_audit_request(workspace, "apps/churn", "fast") == (app, False)


@pytest.mark.parametrize("mode", ["", "FULL", "Fast", "slow", "full "])
def test_check_audit_request_refuses_an_unknown_mode(workspace, mode):
    make_app(workspace / "apps" / "churn")
    with pytest.raises(HexInputError, match="full.*fast"):
        service.check_audit_request(workspace, "apps/churn", mode)


# --- run_audit -----------------------------------------------------------------------


RUN_CASES = [
    (RunOutcome(0, Status.APPROVED, "APPROVED text", ""), "APPROVED", "APPROVED text"),
    (RunOutcome(1, Status.REJECTED, "REJECTED text", ""), "REJECTED", "REJECTED text"),
    (RunOutcome(2, None, "", "hexperiment: boom"), "none", "hexperiment: boom"),
    (
        RunOutcome(3, Status.UNVERIFIED, "UNVERIFIED text", ""),
        "UNVERIFIED",
        "UNVERIFIED text",
    ),
]


@pytest.mark.parametrize(("outcome", "status", "summary"), RUN_CASES)
def test_run_audit_reports_the_outcome_of_every_exit_code(
    workspace, outcome, status, summary
):
    app = make_app(workspace / "apps" / "churn")
    run = StubRun(outcome)
    result = service.run_audit(workspace, "apps/churn", "full", run)
    assert result == {
        "app": "apps/churn",
        "mode": "full",
        "status": status,
        "exit_code": outcome.exit_code,
        "summary": summary,
    }
    assert run.calls == [(app, True, None)]


def test_run_audit_in_fast_mode_runs_the_app_without_the_canary(workspace):
    app = make_app(workspace / "apps" / "churn")
    run = StubRun()
    result = service.run_audit(workspace, "apps/churn", "fast", run)
    assert result["mode"] == "fast"
    assert run.calls == [(app, False, None)]


@pytest.mark.parametrize(
    ("app", "mode"),
    [
        ("../elsewhere", "full"),
        ("apps/missing", "full"),
        ("apps/no_train", "full"),
        ("apps/churn", "turbo"),
        ("", "full"),
    ],
)
def test_run_audit_refuses_a_bad_request_without_running_anything(workspace, app, mode):
    make_app(workspace / "apps" / "churn")
    no_train = make_app(workspace / "apps" / "no_train")
    (no_train / "train.py").unlink()
    run = StubRun()
    with pytest.raises(HexInputError):
        service.run_audit(workspace, app, mode, run)
    assert run.calls == []


# --- the workspace guard -------------------------------------------------------------


@pytest.mark.parametrize("function_name", ["get_verdict", "get_history", "model_card"])
def test_the_read_functions_refuse_an_app_outside_the_workspace(
    tmp_path, workspace, function_name
):
    make_app(tmp_path / "outside")
    function = getattr(service, function_name)
    with pytest.raises(HexInputError, match="outside the workspace"):
        function(workspace, "../outside")
    with pytest.raises(HexInputError, match="outside the workspace"):
        function(workspace, str(tmp_path / "outside"))


# --- model_card ----------------------------------------------------------------------


def test_model_card_is_the_card_of_the_newest_run(workspace):
    app = make_app(workspace / "apps" / "churn")
    write_run(app, make_verdict(OLD), card="old card")
    write_run(
        app,
        make_verdict(NEW, created_at="2026-01-02T00:00:00Z"),
        card="# Churn card\n\nnaïve baseline ≥ 30 days\n",
    )
    assert service.model_card(workspace, "apps/churn") == (
        "# Churn card\n\nnaïve baseline ≥ 30 days\n"
    )


def test_model_card_raises_when_the_app_has_no_run(workspace):
    make_app(workspace / "apps" / "churn")
    with pytest.raises(HexInputError, match="no run yet"):
        service.model_card(workspace, "apps/churn")


def test_model_card_raises_when_the_card_file_is_missing(workspace):
    app = make_app(workspace / "apps" / "churn")
    folder = write_run(app, make_verdict(OLD))
    (folder / "model_card.md").unlink()
    with pytest.raises(HexInputError, match="model_card.md"):
        service.model_card(workspace, "apps/churn")


@pytest.mark.parametrize("run_id", ["..", "../../secret", "a/b"])
def test_model_card_refuses_a_run_id_that_is_not_a_folder_name(workspace, run_id):
    app = make_app(workspace / "apps" / "churn")
    folder = app / "runs" / "bad"
    folder.mkdir(parents=True)
    (folder / "verdict.json").write_text(
        to_json(make_verdict(run_id)), encoding="utf-8"
    )
    with pytest.raises(HexInputError):
        service.model_card(workspace, "apps/churn")


# --- resource ids --------------------------------------------------------------------


@pytest.mark.parametrize(
    "app_relative", ["apps/churn", "apps/my model", "apps/100%", "a b/c%20d", "."]
)
def test_the_resource_id_round_trips(app_relative):
    uri = service.resource_uri(app_relative)
    assert uri.startswith("hexperiment://model-card/")
    resource_id = uri.removeprefix("hexperiment://model-card/")
    assert "/" not in resource_id
    assert " " not in resource_id
    assert service.app_from_resource_id(resource_id) == app_relative


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1))
def test_the_resource_id_round_trips_for_any_text(app_relative):
    uri = service.resource_uri(app_relative)
    resource_id = uri.removeprefix("hexperiment://model-card/")
    assert "/" not in resource_id
    assert service.app_from_resource_id(resource_id) == app_relative


# --- the service never writes --------------------------------------------------------


def test_every_read_only_function_leaves_the_workspace_byte_identical(workspace):
    app = make_app(workspace / "apps" / "churn")
    write_app_verdict(app, make_verdict(NEW, status=Status.REJECTED))
    write_run(app, make_verdict(OLD))
    write_run(app, make_verdict(NEW, created_at="2026-01-02T00:00:00Z"))
    make_app(workspace / "apps" / "other")
    before = tree_hash(workspace)

    service.list_apps(workspace)
    service.get_verdict(workspace, "apps/churn")
    service.get_verdict(workspace, "apps/other")
    service.get_history(workspace, "apps/churn")
    service.model_card(workspace, "apps/churn")
    service.explain_ghost("canary")
    service.check_audit_request(workspace, "apps/churn", "full")
    with pytest.raises(HexInputError):
        service.get_verdict(workspace, "../nowhere")

    assert tree_hash(workspace) == before


def test_the_verdict_json_content_is_passed_through_unchanged(workspace):
    app = make_app(workspace / "apps" / "churn")
    verdict = make_verdict(status=Status.REJECTED)
    write_app_verdict(app, verdict)
    shown = service.get_verdict(workspace, "apps/churn")["verdict"]
    assert json.dumps(shown, sort_keys=True) == json.dumps(
        verdict.to_dict(), sort_keys=True
    )


# --- files that are not valid UTF-8 (Requirement 4.5) --------------------------------

NOT_UTF8 = b"\xff\xfe not utf-8 \xe9"


def test_get_verdict_names_a_verdict_json_that_is_not_utf8(workspace):
    app = make_app(workspace / "apps" / "churn")
    (app / "verdict.json").write_bytes(NOT_UTF8)
    with pytest.raises(HexInputError, match="UTF-8") as caught:
        service.get_verdict(workspace, "apps/churn")
    assert str(app / "verdict.json") in str(caught.value)


def test_get_history_names_a_history_file_that_is_not_utf8(workspace):
    app = make_app(workspace / "apps" / "churn")
    folder = write_run(app, make_verdict(OLD))
    (folder / "verdict.json").write_bytes(NOT_UTF8)
    with pytest.raises(HexInputError) as caught:
        service.get_history(workspace, "apps/churn")
    assert str(folder / "verdict.json") in str(caught.value)


def test_model_card_names_a_card_that_is_not_utf8(workspace):
    app = make_app(workspace / "apps" / "churn")
    folder = write_run(app, make_verdict(OLD))
    (folder / "model_card.md").write_bytes(NOT_UTF8)
    with pytest.raises(HexInputError, match="UTF-8") as caught:
        service.model_card(workspace, "apps/churn")
    assert str(folder / "model_card.md") in str(caught.value)


def test_model_card_names_a_history_file_that_is_not_utf8(workspace):
    app = make_app(workspace / "apps" / "churn")
    folder = write_run(app, make_verdict(OLD))
    (folder / "verdict.json").write_bytes(NOT_UTF8)
    with pytest.raises(HexInputError) as caught:
        service.model_card(workspace, "apps/churn")
    assert str(folder / "verdict.json") in str(caught.value)


def test_list_apps_keeps_the_other_apps_when_one_hex_toml_is_not_utf8(workspace):
    bad = make_app(workspace / "apps" / "bad")
    (bad / "hex.toml").write_bytes(b'[app]\nname = "caf\xe9"\n')
    make_app(workspace / "apps" / "fine", name="Fine")
    entries = service.list_apps(workspace)["apps"]
    assert [e["path"] for e in entries] == ["apps/bad", "apps/fine"]
    assert entries[0]["name"] == "bad"
    assert "UTF-8" in entries[0]["problem"]
    assert str(bad / "hex.toml") in entries[0]["problem"]
    assert entries[1] == {
        "name": "Fine",
        "path": "apps/fine",
        "status": "none",
        "problem": None,
    }


def test_list_apps_reports_a_verdict_json_that_is_not_utf8_as_a_problem(workspace):
    app = make_app(workspace / "apps" / "churn")
    (app / "verdict.json").write_bytes(NOT_UTF8)
    [entry] = service.list_apps(workspace)["apps"]
    assert entry["status"] == "none"
    assert "verdict.json" in entry["problem"]


# --- links that lead out of the workspace (Requirements 4.2, 4.5) --------------------


def make_link(link: Path, target: Path) -> None:
    """Create a file link, or skip the test when the platform refuses to."""
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"this platform refuses to create file links ({error!r})")


@pytest.fixture
def outside(tmp_path) -> Path:
    folder = tmp_path / "outside"
    folder.mkdir()
    return folder


def test_get_verdict_refuses_a_verdict_json_that_links_outside_the_workspace(
    workspace, outside
):
    app = make_app(workspace / "apps" / "churn")
    secret = outside / "secret.json"
    secret.write_text(to_json(make_verdict(status=Status.APPROVED)), encoding="utf-8")
    make_link(app / "verdict.json", secret)
    with pytest.raises(HexInputError, match="outside the workspace") as caught:
        service.get_verdict(workspace, "apps/churn")
    assert str(secret.resolve()) in str(caught.value)


def test_get_verdict_refuses_a_link_to_a_missing_file_outside_the_workspace(
    workspace, outside
):
    app = make_app(workspace / "apps" / "churn")
    make_link(app / "verdict.json", outside / "missing.json")
    with pytest.raises(HexInputError, match="outside the workspace"):
        service.get_verdict(workspace, "apps/churn")


def test_get_verdict_follows_a_link_that_stays_inside_the_workspace(workspace):
    app = make_app(workspace / "apps" / "churn")
    verdict = make_verdict(status=Status.APPROVED)
    real = workspace / "shared" / "verdict.json"
    real.parent.mkdir()
    real.write_text(to_json(verdict), encoding="utf-8")
    make_link(app / "verdict.json", real)
    result = service.get_verdict(workspace, "apps/churn")
    assert result == {"found": True, "verdict": verdict.to_dict()}


def test_model_card_refuses_a_card_that_links_outside_the_workspace(
    workspace, outside
):
    app = make_app(workspace / "apps" / "churn")
    folder = write_run(app, make_verdict(OLD))
    secret = outside / "secret.md"
    secret.write_text("# not for the agent\n", encoding="utf-8")
    (folder / "model_card.md").unlink()
    make_link(folder / "model_card.md", secret)
    with pytest.raises(HexInputError, match="outside the workspace") as caught:
        service.model_card(workspace, "apps/churn")
    assert str(secret.resolve()) in str(caught.value)


def test_model_card_follows_a_link_that_stays_inside_the_workspace(workspace):
    app = make_app(workspace / "apps" / "churn")
    folder = write_run(app, make_verdict(OLD))
    real = workspace / "shared" / "card.md"
    real.parent.mkdir()
    real.write_text("# shared card\n", encoding="utf-8")
    (folder / "model_card.md").unlink()
    make_link(folder / "model_card.md", real)
    assert service.model_card(workspace, "apps/churn") == "# shared card\n"
