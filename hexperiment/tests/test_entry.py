"""Tests for the light entry point of the ``hexperiment`` command (Requirement 9.6).

Two tests start a Python process, to see what the stop check really loads. None
starts a training run.
"""

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from hexperiment import cli, entry
from hexperiment import hook as hook_module

SESSION = "sess_entry"
STOP_EVENT = json.dumps({"session_id": SESSION, "hook_event_name": "Stop"})
HEAVY = ("pandas", "numpy", "sklearn")

RUN_STOP = (
    "import sys\n"
    "from hexperiment.entry import main\n"
    "code = main(['hook', '--stop'])\n"
    f"loaded = [m for m in {HEAVY!r} if m in sys.modules]\n"
    "cli = 'hexperiment.cli' in sys.modules\n"
    "sys.stderr.write(f'\\nLOADED={loaded} CLI={cli}')\n"
    "sys.exit(code)\n"
)


def run_stop(workspace: Path, event: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "HEXPERIMENT_WORKSPACE"}
    return subprocess.run(
        [sys.executable, "-c", RUN_STOP],
        input=event,
        capture_output=True,
        text=True,
        cwd=workspace,
        env=env,
        check=False,
    )


def test_hook_stop_calls_stop_command_without_importing_cli(monkeypatch):
    monkeypatch.setattr(hook_module, "stop_command", lambda: 7)
    assert entry.main(["hook", "--stop"]) == 7


def test_every_other_argument_list_goes_to_cli_main_unchanged(monkeypatch):
    seen: list[list[str]] = []

    def stub(argv):
        seen.append(argv)
        return 5

    monkeypatch.setattr(cli, "main", stub)
    for args in (
        ["--help"],
        ["hook"],
        ["hook", "--stop", "extra"],
        ["run", "app", "--fast"],
        [],
    ):
        assert entry.main(args) == 5
    assert seen == [
        ["--help"],
        ["hook"],
        ["hook", "--stop", "extra"],
        ["run", "app", "--fast"],
        [],
    ]


def test_no_arguments_are_read_from_sys_argv_when_none_are_given(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["hexperiment", "hook", "--stop"])
    monkeypatch.setattr(hook_module, "stop_command", lambda: 9)
    assert entry.main() == 9


def test_the_stop_check_with_no_message_exits_0_and_loads_nothing_heavy(tmp_path):
    result = run_stop(tmp_path, STOP_EVENT)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr.strip() == "LOADED=[] CLI=False"


def test_the_stop_check_names_an_unverified_app_on_stderr_and_exits_1(tmp_path):
    app = tmp_path / "apps" / "churn"
    (app / ".hex").mkdir(parents=True)
    (app / "hex.toml").write_text("", encoding="utf-8")
    (app / "train.py").write_text("", encoding="utf-8")
    (app / ".hex" / "agent-session").write_text(SESSION + "\n", encoding="utf-8")
    (app / "verdict.json").write_text('{"status": "UNVERIFIED"}', encoding="utf-8")
    result = run_stop(tmp_path, STOP_EVENT)
    assert result.returncode == 1
    assert result.stdout == ""
    message, _, marker = result.stderr.rpartition("\nLOADED=")
    assert message.startswith("Hexperiment: not approved yet.\n- apps/churn:")
    assert marker == "[] CLI=False"


def test_pyproject_declares_the_light_entry_point():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["scripts"]
    assert scripts["hexperiment"] == "hexperiment.entry:main"
    assert scripts["hexperiment-mcp"] == "hexperiment.mcp_server:main"


def test_entry_imports_only_sys_and_collections_abc_at_module_level():
    source = Path(entry.__file__).read_text(encoding="utf-8")
    top_level = [
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    ]
    assert top_level == [
        "from __future__ import annotations",
        "import sys",
        "from collections.abc import Sequence",
    ]


def test_cli_keeps_the_event_limit_as_a_name_for_the_same_value():
    assert cli.MAX_EVENT_BYTES == hook_module.MAX_EVENT_BYTES
