"""Tests of the MCP server through an in-memory client (Requirements 4.1 to 4.7, 7.1).

``build_server`` gets a stub runner, so no test here starts a training run. Every
test runs on asyncio (the ``anyio_backend`` fixture of ``conftest.py``).
"""

import json
import sys
import threading
import time
from pathlib import Path

import anyio
import pytest
from mcp import Client, MCPError

import hexperiment.mcp_server as mcp_package
from hexperiment.mcp_server import service
from hexperiment.mcp_server.server import build_server
from hexperiment.models import RunOutcome, Severity, Status
from service_builder import (
    NEW,
    OLD,
    StubRun,
    finding,
    make_app,
    make_verdict,
    write_app_verdict,
    write_run,
)

pytestmark = pytest.mark.anyio

TOOLS = {"list_apps", "run_audit", "get_verdict", "get_history", "explain_ghost"}
CARD_PREFIX = "hexperiment://model-card/"


def payload(result) -> dict:
    """The structured content of a tool result, or its text parsed as JSON."""
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


def error_text(result) -> str:
    assert result.is_error
    return "\n".join(block.text for block in result.content)


class OverlapRun:
    """A ``RunApp`` that sleeps and records how many calls ran at the same time."""

    def __init__(self, pause: float = 0.3) -> None:
        self.pause = pause
        self.active = 0
        self.most_active = 0
        self.calls = 0
        self._guard = threading.Lock()

    def __call__(
        self, app_dir: Path, *, full: bool, timeout_cap: float | None = None
    ) -> RunOutcome:
        with self._guard:
            self.calls += 1
            self.active += 1
            self.most_active = max(self.most_active, self.active)
        time.sleep(self.pause)
        with self._guard:
            self.active -= 1
        return RunOutcome(3, Status.UNVERIFIED, "done", "")


@pytest.fixture
def workspace(tmp_path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root.resolve()


@pytest.fixture
def churn(workspace) -> Path:
    return make_app(workspace / "apps" / "churn", name="Churn")


# --- the tools -----------------------------------------------------------------------


async def test_the_server_has_exactly_the_five_tools(workspace):
    async with Client(build_server(workspace, run=StubRun())) as client:
        tools = (await client.list_tools()).tools
    assert {tool.name for tool in tools} == TOOLS
    run_audit = next(tool for tool in tools if tool.name == "run_audit")
    assert run_audit.input_schema["required"] == ["app"]
    # The mode is a plain string, so a bad mode is refused by the service with a
    # message that names the allowed values.
    assert run_audit.input_schema["properties"]["mode"]["type"] == "string"


async def test_list_apps_reports_the_workspace_and_the_apps(workspace, churn):
    write_app_verdict(churn, make_verdict(status=Status.APPROVED))
    async with Client(build_server(workspace, run=StubRun())) as client:
        result = await client.call_tool("list_apps", {})
    assert not result.is_error
    assert payload(result) == {
        "workspace": str(workspace),
        "apps": [
            {
                "name": "Churn",
                "path": "apps/churn",
                "status": "APPROVED",
                "problem": None,
            }
        ],
    }


async def test_run_audit_runs_the_app_in_full_mode_by_default(workspace, churn):
    run = StubRun(RunOutcome(0, Status.APPROVED, "APPROVED text", ""))
    async with Client(build_server(workspace, run=run)) as client:
        result = await client.call_tool("run_audit", {"app": "apps/churn"})
    assert not result.is_error
    assert payload(result) == {
        "app": "apps/churn",
        "mode": "full",
        "status": "APPROVED",
        "exit_code": 0,
        "summary": "APPROVED text",
    }
    assert run.calls == [(churn.resolve(), True, None)]


async def test_run_audit_in_fast_mode(workspace, churn):
    run = StubRun()
    async with Client(build_server(workspace, run=run)) as client:
        result = await client.call_tool(
            "run_audit", {"app": "apps/churn", "mode": "fast"}
        )
    assert payload(result)["mode"] == "fast"
    assert run.calls == [(churn.resolve(), False, None)]


async def test_a_failed_run_is_a_normal_result_with_the_message(workspace, churn):
    run = StubRun(RunOutcome(2, None, "", "hexperiment: train.py exited with code 1"))
    async with Client(build_server(workspace, run=run)) as client:
        result = await client.call_tool("run_audit", {"app": "apps/churn"})
    assert not result.is_error
    data = payload(result)
    assert data["status"] == "none"
    assert data["exit_code"] == 2
    assert data["summary"] == "hexperiment: train.py exited with code 1"


async def test_get_verdict_with_and_without_a_current_verdict(workspace, churn):
    async with Client(build_server(workspace, run=StubRun())) as client:
        missing = await client.call_tool("get_verdict", {"app": "apps/churn"})
        write_app_verdict(churn, make_verdict(status=Status.REJECTED))
        present = await client.call_tool("get_verdict", {"app": "apps/churn"})
    assert payload(missing)["found"] is False
    data = payload(present)
    assert data["found"] is True
    assert data["verdict"]["status"] == "REJECTED"


async def test_get_history_lists_the_runs_oldest_first(workspace, churn):
    write_run(
        churn,
        make_verdict(
            NEW,
            created_at="2026-01-02T00:00:00Z",
            status=Status.REJECTED,
            findings=(finding("target_proxy", "The Possession", Severity.ERROR),),
        ),
    )
    write_run(churn, make_verdict(OLD))
    async with Client(build_server(workspace, run=StubRun())) as client:
        result = await client.call_tool("get_history", {"app": "apps/churn"})
    runs = payload(result)["runs"]
    assert [run["run_id"] for run in runs] == [OLD, NEW]
    assert runs[1]["ghosts"] == ["The Possession"]


async def test_explain_ghost_answers_for_a_ghost_and_for_a_check(workspace):
    async with Client(build_server(workspace, run=StubRun())) as client:
        by_ghost = await client.call_tool("explain_ghost", {"name": "The Prophet"})
        by_check = await client.call_tool("explain_ghost", {"name": "temporal"})
    assert payload(by_ghost) == payload(by_check)
    assert payload(by_ghost)["ghost"] == "The Prophet"
    assert payload(by_ghost)["how_to_fix"]


# --- refusals (Requirement 4.5) ------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "arguments", "expected"),
    [
        ("run_audit", {"app": "../outside"}, "outside the workspace"),
        ("run_audit", {"app": "apps/no_train"}, "train.py"),
        ("run_audit", {"app": "apps/missing"}, "not a folder"),
        ("run_audit", {"app": "apps/churn", "mode": "turbo"}, "turbo"),
        ("get_verdict", {"app": "../outside"}, "outside the workspace"),
        ("get_verdict", {"app": "apps/no_train"}, "train.py"),
        ("get_history", {"app": "../outside"}, "outside the workspace"),
        ("get_history", {"app": "apps/no_train"}, "train.py"),
        ("explain_ghost", {"name": "banshee"}, "banshee"),
    ],
)
async def test_a_bad_request_is_an_error_result_and_runs_nothing(
    workspace, churn, tool, arguments, expected
):
    no_train = make_app(workspace / "apps" / "no_train")
    (no_train / "train.py").unlink()
    run = StubRun()
    async with Client(build_server(workspace, run=run)) as client:
        result = await client.call_tool(tool, arguments)
    assert expected in error_text(result)
    assert run.calls == []


async def test_an_absolute_path_outside_the_workspace_is_refused(
    tmp_path, workspace, churn
):
    outside = make_app(tmp_path / "elsewhere")
    run = StubRun()
    async with Client(build_server(workspace, run=run)) as client:
        result = await client.call_tool("run_audit", {"app": str(outside)})
    assert "outside the workspace" in error_text(result)
    assert run.calls == []


# --- one audit at a time (Requirement 4.6) -------------------------------------------


async def test_two_concurrent_audits_do_not_overlap(workspace, churn):
    run = OverlapRun()
    results = []

    async with Client(build_server(workspace, run=run)) as client:

        async def call() -> None:
            results.append(await client.call_tool("run_audit", {"app": "apps/churn"}))

        with anyio.fail_after(30):
            async with anyio.create_task_group() as group:
                group.start_soon(call)
                group.start_soon(call)

    assert run.calls == 2
    assert run.most_active == 1
    assert [payload(result)["status"] for result in results] == ["UNVERIFIED"] * 2


async def test_a_bad_request_is_refused_while_an_audit_is_running(workspace, churn):
    started = threading.Event()
    release = threading.Event()

    def blocking_run(
        app_dir: Path, *, full: bool, timeout_cap: float | None = None
    ) -> RunOutcome:
        started.set()
        release.wait(20)
        return RunOutcome(3, Status.UNVERIFIED, "done", "")

    async with Client(build_server(workspace, run=blocking_run)) as client:
        async with anyio.create_task_group() as group:

            async def slow() -> None:
                await client.call_tool("run_audit", {"app": "apps/churn"})

            group.start_soon(slow)
            with anyio.fail_after(10):
                while not started.is_set():
                    await anyio.sleep(0.01)
                # The refusal comes before the lock, so it does not wait for the
                # running audit. If it did, this call would time out.
                with anyio.fail_after(5):
                    refused = await client.call_tool(
                        "run_audit", {"app": "apps/churn", "mode": "turbo"}
                    )
                assert refused.is_error
            release.set()


# --- model cards as resources (Requirement 4.4) --------------------------------------


async def test_the_model_card_of_each_app_is_listed_and_readable(workspace, churn):
    spaced = make_app(workspace / "apps" / "my model 100%", name="Spaced")
    write_run(churn, make_verdict(OLD), card="# old card\n")
    write_run(
        churn, make_verdict(NEW, created_at="2026-01-02T00:00:00Z"), card="# new card\n"
    )
    write_run(spaced, make_verdict(OLD), card="# spaced card\n")
    async with Client(build_server(workspace, run=StubRun())) as client:
        listed = (await client.list_resources()).resources
        uris = {str(resource.uri) for resource in listed}
        churn_card = await client.read_resource(service.resource_uri("apps/churn"))
        spaced_card = await client.read_resource(
            service.resource_uri("apps/my model 100%")
        )
    assert uris == {
        service.resource_uri("apps/churn"),
        service.resource_uri("apps/my model 100%"),
    }
    assert churn_card.contents[0].text == "# new card\n"
    assert churn_card.contents[0].mime_type == "text/markdown"
    assert spaced_card.contents[0].text == "# spaced card\n"


async def test_an_app_created_after_the_start_is_readable_through_the_template(
    workspace, churn
):
    async with Client(build_server(workspace, run=StubRun())) as client:
        later = make_app(workspace / "apps" / "later app")
        write_run(later, make_verdict(OLD), card="# later card\n")
        templates = (await client.list_resource_templates()).resource_templates
        card = await client.read_resource(service.resource_uri("apps/later app"))
    assert [t.uri_template for t in templates] == [CARD_PREFIX + "{app}"]
    assert card.contents[0].text == "# later card\n"


async def test_a_card_that_does_not_exist_is_an_error_naming_the_problem(
    workspace, churn
):
    async with Client(build_server(workspace, run=StubRun())) as client:
        with pytest.raises(MCPError, match="no run yet"):
            await client.read_resource(service.resource_uri("apps/churn"))
        with pytest.raises(MCPError, match="not a folder"):
            await client.read_resource(service.resource_uri("apps/missing"))


async def test_a_card_outside_the_workspace_cannot_be_read(tmp_path, workspace, churn):
    outside = make_app(tmp_path / "outside")
    write_run(outside, make_verdict(OLD), card="# secret card\n")
    async with Client(build_server(workspace, run=StubRun())) as client:
        for app in ("../outside", str(outside)):
            with pytest.raises(MCPError):
                await client.read_resource(service.resource_uri(app))


async def test_an_app_at_the_workspace_root_has_a_readable_card(workspace):
    make_app(workspace)
    write_run(workspace, make_verdict(OLD), card="# root card\n")
    async with Client(build_server(workspace, run=StubRun())) as client:
        card = await client.read_resource(service.resource_uri("."))
        tool = await client.call_tool("get_history", {"app": "."})
    assert card.contents[0].text == "# root card\n"
    assert payload(tool)["app"] == "."


# --- standard output carries the protocol --------------------------------------------


async def test_a_session_writes_nothing_to_standard_output(workspace, churn, capfd):
    write_run(churn, make_verdict(OLD))
    run = StubRun()
    async with Client(build_server(workspace, run=run)) as client:
        await client.list_tools()
        await client.call_tool("list_apps", {})
        await client.call_tool("run_audit", {"app": "apps/churn"})
        await client.call_tool("get_verdict", {"app": "apps/churn"})
        await client.call_tool("get_history", {"app": "apps/churn"})
        await client.call_tool("explain_ghost", {"name": "canary"})
        await client.call_tool("run_audit", {"app": "../outside"})
        await client.read_resource(service.resource_uri("apps/churn"))
    assert capfd.readouterr().out == ""


# --- main (Requirement 4.1) ----------------------------------------------------------


def hide_sdk(monkeypatch) -> None:
    """Make ``import mcp`` fail, as in an environment without the extra."""
    names = [n for n in sys.modules if n == "mcp" or n.startswith("mcp.")]
    for name in names:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.delitem(sys.modules, mcp_package.SERVER_MODULE, raising=False)


def test_main_explains_a_missing_sdk_and_exits_with_code_2(monkeypatch, capsys):
    hide_sdk(monkeypatch)
    with pytest.raises(SystemExit) as caught:
        mcp_package.main()
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert 'pip install "hexperiment[mcp]"' in captured.err


def test_main_reports_an_invalid_workspace_and_exits_with_code_2(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setenv("HEXPERIMENT_WORKSPACE", str(tmp_path / "no such folder"))
    with pytest.raises(SystemExit) as caught:
        mcp_package.main()
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "HEXPERIMENT_WORKSPACE" in captured.err


def test_main_lets_an_unrelated_import_error_through(monkeypatch):
    monkeypatch.delitem(sys.modules, mcp_package.SERVER_MODULE, raising=False)
    monkeypatch.setitem(sys.modules, "hexperiment.workspace", None)
    with pytest.raises(ImportError):
        mcp_package.main()
