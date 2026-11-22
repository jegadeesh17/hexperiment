"""The no-network guard covers the apps tests and only the apps tests (Requirement 5.5)."""

from __future__ import annotations

import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from apps_helpers import APPS_DIR

REPO_ROOT = APPS_DIR.parent
OTHER_APPS_TEST = "apps/tests/test_hex_toml.py"

# One test that opens a listening socket on the loopback address and connects a
# second socket to it. It goes through ``socket.socket.connect`` on every operating
# system, and it never leaves the machine.
PROBE = """\
import socket


def test_probe() -> None:
    with socket.socket() as server, socket.socket() as client:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        client.connect(server.getsockname())
"""


def test_the_guard_does_not_reach_tests_outside_the_apps_folder(tmp_path: Path) -> None:
    # The probe folder has its own pytest configuration, so the child run takes it
    # as its root folder, and the apps conftest lies outside that root (the same
    # situation as a run of both test folders). If the guard ignored the scope, it
    # would patch ``connect`` for the probe and the probe would fail. ``-k``
    # deselects the apps file's tests, so the guard tests are not run again and
    # cannot recurse.
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\n", encoding="utf-8"
    )
    probe = tmp_path / "test_probe.py"
    probe.write_text(PROBE, encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(probe),
            OTHER_APPS_TEST,
            "-k",
            "test_probe",
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
    assert re.search(r"\b1 passed\b", output), output


def test_a_connection_attempt_in_an_apps_test_is_refused() -> None:
    with socket.socket() as sock:
        with pytest.raises(RuntimeError, match="a test tried to use the network"):
            sock.connect(("127.0.0.1", 9))
        with pytest.raises(RuntimeError, match="a test tried to use the network"):
            sock.connect_ex(("127.0.0.1", 9))


def test_a_datagram_send_in_an_apps_test_is_refused() -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock,
        pytest.raises(RuntimeError, match="a test tried to use the network"),
    ):
        sock.sendto(b"ping", ("127.0.0.1", 9))


def test_a_host_name_lookup_in_an_apps_test_is_refused() -> None:
    with pytest.raises(RuntimeError, match="a test tried to use the network"):
        socket.getaddrinfo("example.com", 80)


def test_a_gethostbyname_lookup_in_an_apps_test_is_refused() -> None:
    with pytest.raises(RuntimeError, match="a test tried to use the network"):
        socket.gethostbyname("example.com")


def test_a_gethostbyname_ex_lookup_in_an_apps_test_is_refused() -> None:
    with pytest.raises(RuntimeError, match="a test tried to use the network"):
        socket.gethostbyname_ex("example.com")


def test_a_getnameinfo_lookup_in_an_apps_test_is_refused() -> None:
    with pytest.raises(RuntimeError, match="a test tried to use the network"):
        socket.getnameinfo(("127.0.0.1", 80), 0)


@pytest.mark.skipif(
    not hasattr(socket.socket, "sendmsg"), reason="this platform has no socket.sendmsg"
)
def test_a_sendmsg_datagram_in_an_apps_test_is_refused() -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock,
        pytest.raises(RuntimeError, match="a test tried to use the network"),
    ):
        sock.sendmsg([b"ping"], [], 0, ("127.0.0.1", 9))
