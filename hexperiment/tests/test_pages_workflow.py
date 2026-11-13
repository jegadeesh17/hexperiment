"""The Pages workflow keeps its safety rules (Requirement 8.3).

The workflow is read as text; there is no YAML dependency. Small helpers split
the text into blocks by indentation. Negative tests feed the checks a copy of
the text with one defect each, so a weak check cannot pass.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "pages.yml"
BENCHMARK = ROOT / "hexperiment" / "benchmark" / "results.json"

BASE_COMMAND = (
    "hexperiment gallery apps/hotel-cancellations apps/flight-delays "
    '--out site --repo-url "$REPO_URL"'
)
BENCHMARK_FLAG = " --benchmark hexperiment/benchmark/results.json"

PINNED = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40} # v\d+(\.\d+)*$")


# --- helpers -----------------------------------------------------------------


def read_workflow() -> str:
    if not WORKFLOW.is_file():
        pytest.fail(f"The Pages workflow is missing: expected the file {WORKFLOW}")
    return WORKFLOW.read_text(encoding="utf-8").replace("\r\n", "\n")


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def is_content(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def code_lines(text: str) -> list[str]:
    """Lines that are not blank and not comments."""
    return [line for line in text.split("\n") if is_content(line)]


def child_block(lines: list[str], key: str, indent: int) -> list[str] | None:
    """Lines nested under `key:` at `indent`, or None if the key is absent."""
    start = None
    for i, line in enumerate(lines):
        if (
            indent_of(line) == indent
            and line.strip().rstrip(":") == key
            and line.rstrip().endswith(":")
        ):
            start = i
            break
    if start is None:
        return None
    block: list[str] = []
    for line in lines[start + 1 :]:
        if indent_of(line) <= indent:
            break
        block.append(line)
    return block


def top_block(text: str, key: str) -> list[str] | None:
    return child_block(code_lines(text), key, 0)


def job_block(text: str, name: str) -> list[str]:
    jobs = top_block(text, "jobs")
    assert jobs is not None, "The workflow has no top-level `jobs` block."
    block = child_block(jobs, name, 2)
    assert block is not None, f"The workflow has no `{name}` job."
    return block


def entries(block: list[str]) -> list[str]:
    return [line.strip() for line in block]


def raw_uses_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.split("\n") if is_content(ln) and "uses:" in ln]


def step_after(lines: list[str], index: int) -> list[str]:
    """Lines of the step that starts at lines[index] (a `- ` line)."""
    step_indent = indent_of(lines[index])
    body = [lines[index]]
    for line in lines[index + 1 :]:
        if indent_of(line) <= step_indent:
            break
        body.append(line)
    return body


# --- checks (each raises AssertionError on a defect) -------------------------


def check_pins(text: str) -> None:
    lines = raw_uses_lines(text)
    assert lines, "No `uses:` line found."
    for line in lines:
        value = line.split("uses:", 1)[1].strip()
        assert PINNED.match(value), (
            f"`uses:` must be owner/name@<40 lowercase hex> # vX.Y.Z, got: {line}"
        )


def check_checkout_credentials(text: str) -> None:
    lines = code_lines(text)
    seen = 0
    for i, line in enumerate(lines):
        if "uses: actions/checkout@" in line:
            seen += 1
            step = [s.strip() for s in step_after(lines, _step_start(lines, i))]
            assert "persist-credentials: false" in step, (
                "actions/checkout must be followed by `persist-credentials: false`."
            )
    assert seen, "No actions/checkout step found."


def _step_start(lines: list[str], index: int) -> int:
    """Index of the `- ` line that opens the step containing lines[index]."""
    i = index
    while i >= 0 and not lines[i].lstrip().startswith("- "):
        i -= 1
    return max(i, 0)


def check_top_permissions(text: str) -> None:
    block = top_block(text, "permissions")
    assert block is not None, "The workflow needs a top-level `permissions` block."
    assert entries(block) == ["contents: read"], (
        f"Top-level permissions must be exactly `contents: read`, got {entries(block)}"
    )


def check_build_has_no_permissions(text: str) -> None:
    block = job_block(text, "build")
    keys = [ln.strip() for ln in block if indent_of(ln) == 4]
    assert not any(k.startswith("permissions:") for k in keys), (
        "The `build` job must not have a `permissions` key."
    )


def check_deploy(text: str) -> None:
    block = job_block(text, "deploy")
    keys = [ln.strip() for ln in block if indent_of(ln) == 4]
    assert "needs: build" in keys, "The `deploy` job must have `needs: build`."
    ifs = [k for k in keys if k.startswith("if:")]
    assert ifs, "The `deploy` job needs an `if:` line."
    assert "github.event.repository.private" in ifs[0], (
        "The `deploy` job `if:` must test github.event.repository.private."
    )
    perms = child_block(block, "permissions", 4)
    assert perms is not None, "The `deploy` job needs a `permissions` block."
    assert sorted(entries(perms)) == ["id-token: write", "pages: write"], (
        "Deploy permissions must be exactly pages: write and id-token: write, "
        f"got {entries(perms)}"
    )


def build_command(text: str) -> str:
    lines = code_lines(text)
    for i, line in enumerate(lines):
        if line.strip() == "run: >-":
            base = indent_of(line)
            parts = []
            for follow in lines[i + 1 :]:
                if indent_of(follow) <= base:
                    break
                parts.append(follow.strip())
            command = " ".join(parts)
            if command.startswith("hexperiment gallery"):
                return command
    raise AssertionError("No folded `run: >-` step runs `hexperiment gallery`.")


def check_command(text: str, benchmark_exists: bool) -> None:
    expected = BASE_COMMAND + (BENCHMARK_FLAG if benchmark_exists else "")
    assert build_command(text) == expected, f"The build command must equal: {expected}"


def check_pip_install(text: str) -> None:
    installs = [ln for ln in code_lines(text) if "pip install" in ln]
    assert installs, "The workflow must run `pip install`."
    for line in installs:
        assert not re.search(r"\s(-e|--editable)(\s|$)", line), (
            f"`pip install` must not use -e: {line.strip()}"
        )


def check_no_secrets(text: str) -> None:
    for line in code_lines(text):
        assert "secrets." not in line, (
            f"The workflow must not use secrets: {line.strip()}"
        )


def check_triggers(text: str) -> None:
    block = top_block(text, "on")
    assert block is not None, "The workflow needs an `on:` block."
    push = child_block(block, "push", 2)
    assert push is not None, "The workflow must trigger on `push`."
    assert [e for e in entries(push)] == ["branches: [main]"], (
        f"`push` must be limited to `branches: [main]`, got {entries(push)}"
    )
    assert any(ln.strip() == "workflow_dispatch:" for ln in block), (
        "The workflow must trigger on `workflow_dispatch`."
    )
    assert "pull_request_target" not in "\n".join(code_lines(text)), (
        "The workflow must not use `pull_request_target`."
    )


ALL_CHECKS = (
    check_pins,
    check_checkout_credentials,
    check_top_permissions,
    check_build_has_no_permissions,
    check_deploy,
    lambda t: check_command(t, BENCHMARK.is_file()),
    check_pip_install,
    check_no_secrets,
    check_triggers,
)


# --- tests on the committed workflow -----------------------------------------


def test_workflow_file_exists() -> None:
    assert WORKFLOW.is_file(), f"The Pages workflow is missing: expected {WORKFLOW}"


def test_every_uses_line_is_pinned_with_a_version_comment() -> None:
    check_pins(read_workflow())


def test_checkout_does_not_persist_credentials() -> None:
    check_checkout_credentials(read_workflow())


def test_top_level_permissions_are_contents_read_only() -> None:
    check_top_permissions(read_workflow())


def test_build_job_has_no_permissions_key() -> None:
    check_build_has_no_permissions(read_workflow())


def test_deploy_job_is_gated_and_least_privilege() -> None:
    check_deploy(read_workflow())


def test_build_command_is_the_expected_one() -> None:
    check_command(read_workflow(), BENCHMARK.is_file())


def test_pip_install_is_not_editable() -> None:
    check_pip_install(read_workflow())


def test_no_secrets_are_used() -> None:
    check_no_secrets(read_workflow())


def test_triggers_are_push_to_main_and_manual_only() -> None:
    check_triggers(read_workflow())


def test_committed_workflow_passes_every_check() -> None:
    text = read_workflow()
    for check in ALL_CHECKS:
        check(text)


# --- negative tests: one defect each -----------------------------------------


def mutate(old: str, new: str, count: int = 1) -> str:
    text = read_workflow()
    assert old in text, f"The test needs {old!r} in the workflow to build a defect."
    return text.replace(old, new, count)


def test_a_tag_instead_of_a_sha_is_caught() -> None:
    text = re.sub(
        r"(actions/setup-python)@[0-9a-f]{40}", r"\1@v7", read_workflow(), count=1
    )
    with pytest.raises(AssertionError):
        check_pins(text)


def test_a_missing_version_comment_is_caught() -> None:
    text = re.sub(
        r"(actions/deploy-pages@[0-9a-f]{40}) # v\S+", r"\1", read_workflow(), count=1
    )
    with pytest.raises(AssertionError):
        check_pins(text)


def test_an_uppercase_or_short_sha_is_caught() -> None:
    text = re.sub(
        r"(actions/checkout)@[0-9a-f]{40}", r"\1@3d3c42e", read_workflow(), count=1
    )
    with pytest.raises(AssertionError):
        check_pins(text)


def test_editable_install_is_caught() -> None:
    text = mutate("pip install ./hexperiment", "pip install -e ./hexperiment")
    with pytest.raises(AssertionError):
        check_pip_install(text)


def test_a_write_permission_in_build_is_caught() -> None:
    text = mutate(
        "  build:\n    runs-on: ubuntu-latest\n",
        "  build:\n    runs-on: ubuntu-latest\n"
        "    permissions:\n      contents: write\n",
    )
    with pytest.raises(AssertionError):
        check_build_has_no_permissions(text)


def test_a_missing_if_on_deploy_is_caught() -> None:
    text = mutate("    if: ${{ !github.event.repository.private }}\n", "")
    with pytest.raises(AssertionError):
        check_deploy(text)


def test_a_missing_persist_credentials_is_caught() -> None:
    text = mutate("        with:\n          persist-credentials: false\n", "")
    with pytest.raises(AssertionError):
        check_checkout_credentials(text)


def test_persist_credentials_true_is_caught() -> None:
    text = mutate("persist-credentials: false", "persist-credentials: true")
    with pytest.raises(AssertionError):
        check_checkout_credentials(text)


def test_extra_top_level_permission_is_caught() -> None:
    text = mutate(
        "permissions:\n  contents: read\n",
        "permissions:\n  contents: read\n  pages: write\n",
    )
    with pytest.raises(AssertionError):
        check_top_permissions(text)


def test_extra_deploy_permission_is_caught() -> None:
    text = mutate(
        "      id-token: write\n", "      id-token: write\n      contents: write\n"
    )
    with pytest.raises(AssertionError):
        check_deploy(text)


def test_deploy_without_needs_is_caught() -> None:
    text = mutate("    needs: build\n", "")
    with pytest.raises(AssertionError):
        check_deploy(text)


def test_a_changed_command_is_caught() -> None:
    text = mutate("--out site", "--out public")
    with pytest.raises(AssertionError):
        check_command(text, benchmark_exists=False)


FLAG_LINE = "\n          --benchmark hexperiment/benchmark/results.json"


def workflow_without_flag() -> str:
    return read_workflow().replace(FLAG_LINE, "")


def workflow_with_flag() -> str:
    text = workflow_without_flag()
    old = '"$REPO_URL"\n'
    assert old in text, f"The test needs {old!r} in the workflow to add the flag."
    return text.replace(old, '"$REPO_URL"' + FLAG_LINE + "\n", 1)


def test_missing_benchmark_flag_is_caught_when_the_file_exists() -> None:
    text = workflow_without_flag()
    check_command(text, benchmark_exists=False)
    with pytest.raises(AssertionError):
        check_command(text, benchmark_exists=True)


def test_benchmark_flag_without_the_file_is_caught() -> None:
    text = workflow_with_flag()
    with pytest.raises(AssertionError):
        check_command(text, benchmark_exists=False)
    check_command(text, benchmark_exists=True)


def test_a_secret_is_caught() -> None:
    text = mutate("REPO_URL: ", "TOKEN: ${{ secrets.TOKEN }}\n          REPO_URL: ")
    with pytest.raises(AssertionError):
        check_no_secrets(text)


def test_pull_request_target_is_caught() -> None:
    text = mutate(
        "  workflow_dispatch:\n", "  workflow_dispatch:\n  pull_request_target:\n"
    )
    with pytest.raises(AssertionError):
        check_triggers(text)


def test_a_push_to_any_branch_is_caught() -> None:
    text = mutate("branches: [main]", "branches: [main, dev]")
    with pytest.raises(AssertionError):
        check_triggers(text)


def test_a_missing_workflow_dispatch_is_caught() -> None:
    text = mutate("  workflow_dispatch:\n", "")
    with pytest.raises(AssertionError):
        check_triggers(text)
