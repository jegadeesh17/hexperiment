"""The no-network guard covers the apps tests and only the apps tests (Requirement 5.5)."""

from __future__ import annotations

import socket
import subprocess
import sys

import pytest
from conftest import APPS_DIR

REPO_ROOT = APPS_DIR.parent
MCP_TESTS = "hexperiment/tests/test_mcp_server.py"
MCP_TEST = "test_the_server_has_exactly_the_five_tools"
OTHER_APPS_TEST = "apps/tests/test_hex_toml.py"


def test_the_guard_does_not_reach_tests_outside_the_apps_folder() -> None:
    # The child run takes the root folder from the first argument (the kit), so the
    # apps conftest lies outside it, as in a run of both folders. It runs one MCP
    # test, which opens only asyncio's loopback socket pair (no outside network).
    # ``-k`` deselects every test of the apps file, so this test is not collected
    # again and cannot recurse. ``-n0`` is used because ``-p no:xdist`` would make
    # the kit's ``-n`` option an error.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n0",
            "-p",
            "no:cacheprovider",
            "-k",
            MCP_TEST,
            MCP_TESTS,
            OTHER_APPS_TEST,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = result.stdout + result.stderr
    assert "a test tried to use the network" not in output, output
    assert result.returncode == 0, output
    assert "1 passed" in output, output


def test_a_connection_attempt_in_an_apps_test_is_refused() -> None:
    with socket.socket() as sock:
        with pytest.raises(RuntimeError, match="a test tried to use the network"):
            sock.connect(("127.0.0.1", 9))
        with pytest.raises(RuntimeError, match="a test tried to use the network"):
            sock.connect_ex(("127.0.0.1", 9))
