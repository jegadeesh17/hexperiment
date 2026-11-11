"""Format checks for the Kiro files and a model of the ghost hunter's permissions.

The files under test are the repository's own ``.kiro/hooks/hexperiment.json``,
``.kiro/workflows/exorcise.workflow.json`` and ``.kiro/agents/ghost-hunter.json``
(Requirements 1.1, 2.1 to 2.5, 3.1 to 3.5, 7.2). The tests find them from this
file's location and fail with the path when one is missing.
"""

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from hexperiment import hook

# The trigger literal and the workflow input placeholder are named here once, so a
# correction after a live check in Kiro changes one line each.
EXPECTED_TRIGGER = "PostFileSave"
INPUT_REF = "{{app}}"

KIRO_DIR = Path(__file__).resolve().parents[2] / ".kiro"
HOOK_FILE = "hooks/hexperiment.json"
WORKFLOW_FILE = "workflows/exorcise.workflow.json"
AGENT_FILE = "agents/ghost-hunter.json"

CAPABILITIES = {"fs_read", "fs_write", "shell"}
EFFECTS = {"allow", "deny"}
MAX_WORKFLOW_STEPS = 50
MAX_WORKFLOW_DEPTH = 8
STOP_CONDITIONS_NOT_ALLOWED = ("containsText", "completionSignal", "stopWhen")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        raise ValueError(f"duplicate key(s) {', '.join(repeated)}")
    return dict(pairs)


def read_kiro_json(relative: str) -> dict[str, Any]:
    """Load a file below ``.kiro/``; fail with a clear message when it is unusable."""
    path = KIRO_DIR / relative
    if not path.is_file():
        pytest.fail(
            f"{path} is missing; create it from 'The Kiro files' in design.md.",
            pytrace=False,
        )
    try:
        loaded = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys
        )
    except ValueError as error:
        pytest.fail(f"{path} is not valid JSON ({error}).", pytrace=False)
    assert isinstance(loaded, dict), f"{path} must hold a JSON object"
    return loaded


# --- the hook (Requirement 1.1) ------------------------------------------------------


def hook_entry() -> dict[str, Any]:
    config = read_kiro_json(HOOK_FILE)
    assert config["version"] == "v1"
    assert len(config["hooks"]) == 1
    return config["hooks"][0]


def test_the_hook_file_has_version_v1_and_one_named_hook():
    entry = hook_entry()
    assert isinstance(entry["name"], str) and entry["name"].strip()
    assert isinstance(entry["description"], str) and entry["description"].strip()


def test_the_hook_fires_when_the_agent_saves_a_file():
    assert hook_entry()["trigger"] == EXPECTED_TRIGGER


def test_the_hook_action_is_the_hexperiment_hook_command():
    action = hook_entry()["action"]
    assert action["type"] == "command"
    assert action["command"] == "hexperiment hook"


@pytest.mark.parametrize("subject", ["apps/x/train.py", r"C:\p\apps\x\train.py"])
def test_the_matcher_selects_python_files_whether_it_matches_or_searches(subject):
    matcher = re.compile(hook_entry()["matcher"])
    assert matcher.fullmatch(subject)
    assert matcher.search(subject)


@pytest.mark.parametrize("subject", ["apps/x/hex.toml", "apps/x/train.py.bak", ""])
def test_the_matcher_rejects_other_files_whether_it_matches_or_searches(subject):
    matcher = re.compile(hook_entry()["matcher"])
    assert not matcher.fullmatch(subject)
    assert not matcher.search(subject)


def test_the_hook_timeout_is_the_one_the_hook_module_plans_for():
    timeout = hook_entry()["timeout"]
    assert timeout == 120
    assert timeout == hook.HOOK_TIMEOUT_SECONDS
    assert hook.HOOK_RUN_CAP_SECONDS < timeout


# --- the workflow (Requirements 2.1 to 2.5) ------------------------------------------


def workflow() -> dict[str, Any]:
    return read_kiro_json(WORKFLOW_FILE)


def walk_steps(
    steps: list[dict[str, Any]], depth: int = 1
) -> Iterator[tuple[dict[str, Any], int]]:
    for node in steps:
        yield node, depth
        yield from walk_steps(node.get("steps", []), depth + 1)


def repeat_node() -> dict[str, Any]:
    repeats = [n for n, _ in walk_steps(workflow()["steps"]) if n["type"] == "repeat"]
    assert len(repeats) == 1
    return repeats[0]


def hunt_node() -> dict[str, Any]:
    body = repeat_node()["steps"]
    assert len(body) == 1
    return body[0]


def agent() -> dict[str, Any]:
    return read_kiro_json(AGENT_FILE)


def test_the_workflow_is_named_exorcise_and_takes_one_input_app():
    config = workflow()
    assert config["name"] == "exorcise"
    assert isinstance(config["description"], str) and config["description"].strip()
    assert list(config["inputs"]) == ["app"]
    assert config["steps"]


def test_the_workflow_repeats_at_most_five_times_and_then_pauses():
    node = repeat_node()
    assert node["id"]
    assert node["maxIterations"] == 5
    assert node["onMaxIterations"] == "pause"


def test_the_workflow_stops_only_on_an_approved_verdict_file():
    condition = repeat_node()["stopCondition"]
    assert list(condition) == ["fileCheck"]
    for name in STOP_CONDITIONS_NOT_ALLOWED:
        assert name not in condition
    check = condition["fileCheck"]
    assert check["path"] == f"{INPUT_REF}/verdict.json"
    assert check["jsonPath"] == "$.status"
    assert check["value"] == "APPROVED"


def test_the_workflow_step_is_run_by_the_ghost_hunter_agent():
    node = hunt_node()
    assert node["type"] == "step"
    assert node["id"]
    assert node["agent"] == agent()["name"] == "ghost-hunter"


def test_the_step_prompt_tells_the_agent_what_to_run_read_and_fix():
    prompt = hunt_node()["prompt"]
    assert f"hexperiment run {INPUT_REF}" in prompt
    assert f"{INPUT_REF}/verdict.json" in prompt
    assert re.search(r"model[ _]card", prompt)
    assert "fix" in prompt.lower()
    assert "thresholds" in prompt
    assert "waivers" in prompt


def test_the_workflow_ids_are_unique_and_the_size_is_bounded():
    nodes = list(walk_steps(workflow()["steps"]))
    ids = [node["id"] for node, _ in nodes]
    assert len(ids) == len(set(ids))
    assert len(nodes) <= MAX_WORKFLOW_STEPS
    assert max(depth for _, depth in nodes) <= MAX_WORKFLOW_DEPTH


# --- the agent (Requirements 3.1 to 3.5) ---------------------------------------------


def test_the_agent_has_the_fields_of_a_custom_agent():
    config = agent()
    for key in ("name", "description", "prompt", "model"):
        assert isinstance(config[key], str) and config[key].strip(), key
    assert config["tools"]
    assert config["permissions"]["rules"]


def test_every_rule_has_a_known_capability_and_effect_and_a_match_unless_it_reads():
    for rule in agent()["permissions"]["rules"]:
        assert rule["capability"] in CAPABILITIES
        assert rule["effect"] in EFFECTS
        if rule["capability"] != "fs_read":
            assert rule["match"], rule
            assert all(isinstance(p, str) and p for p in rule["match"])


def test_the_resource_file_of_the_agent_exists():
    for resource in agent()["resources"]:
        assert resource.startswith("file://")
        assert (KIRO_DIR.parent / resource.removeprefix("file://")).is_file(), resource


@pytest.mark.parametrize(
    "phrase",
    [
        "Thresholds, waivers",  # 3.5: thresholds and waivers are the human's decision
        "the human's decision",
        "does not count as a fix",  # 3.5: removing the audit call or a feature
        "hexperiment.audit call",
        "The Possession is a fix only when the column is recorded after the outcome",
        "say why it is",
        "stop and report it",  # 3.5: report instead of working around
        "instead of working around it",
    ],
)
def test_the_prompt_states_the_rules_of_the_ghost_hunter(phrase):
    assert phrase in agent()["prompt"]


# --- the permission simulator --------------------------------------------------------


def glob_regex(pattern: str, *, loose: bool) -> re.Pattern[str]:
    """Turn a permission glob into a regular expression.

    This approximates Kiro's matcher, which the documentation does not specify; it
    guards the intent of the rules, not Kiro's engine. Two readings bracket the
    uncertainty. Strict: ``*`` stays inside one path segment, ``**`` crosses
    segments. Loose: ``*`` matches anything. In both, ``**/`` also matches no folder.
    """
    parts: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            parts.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            parts.append(".*")
            i += 2
        elif pattern[i] == "*":
            parts.append(".*" if loose else "[^/]*")
            i += 1
        else:
            parts.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(parts), re.DOTALL)


def rule_matches(rule: dict[str, Any], subject: str, *, loose: bool) -> bool:
    return any(
        glob_regex(pattern, loose=loose).fullmatch(subject)
        for pattern in rule.get("match", [])
    )


def rules(capability: str, effect: str) -> list[dict[str, Any]]:
    return [
        rule
        for rule in agent()["permissions"]["rules"]
        if rule["capability"] == capability and rule["effect"] == effect
    ]


def surely_denied(capability: str, subject: str) -> bool:
    """A deny rule matches under the strict reading."""
    return any(
        rule_matches(rule, subject, loose=False) for rule in rules(capability, "deny")
    )


def possibly_allowed(capability: str, subject: str) -> bool:
    """An allow rule matches under the loose reading and no deny rule does."""
    allowed = any(
        rule_matches(rule, subject, loose=True) for rule in rules(capability, "allow")
    )
    denied = any(
        rule_matches(rule, subject, loose=True) for rule in rules(capability, "deny")
    )
    return allowed and not denied


def as_path(path: str) -> str:
    return path.replace("\\", "/")


PREFIXES = ["", "/work/proj/", "C:/work/proj/", "C:\\work\\proj\\"]

DENIED_WRITES = [
    "apps/churn/hex.toml",
    "apps/churn/verdict.json",
    "apps/churn/runs/20260101-000000-abcdef/model_card.md",
    "apps/churn/.hex/normal/meta.json",
    "apps/churn/data/raw.csv",
    "apps/churn/data/helper.py",
    "apps/churn/haunted/train.py",
    "apps/churn/runs/20260101-000000-abcdef/verdict.json",
    "apps/churn/fetch_data.py",
    ".kiro/agents/ghost-hunter.json",
    ".kiro/steering/ml-integrity.md",
    ".kiro/hooks/hexperiment.json",
    "hexperiment/src/hexperiment/cli.py",
    "hexperiment/tests/test_cli.py",
    "hexperiment/pyproject.toml",
    "apps/churn/hexperiment.py",
    "apps/churn/hexperiment/__init__.py",
    ".venv/Lib/site-packages/hexperiment/cli.py",
]

ALLOWED_WRITES = ["apps/churn/train.py", "apps/churn/features/encode.py"]

NOT_ALLOWED_WRITES = [
    "apps/churn/notes.md",
    "apps/churn/data.csv",
    "train.py",
    "README.md",
    "docs/DECISIONS.md",
]


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("path", DENIED_WRITES)
def test_the_files_of_requirement_3_3_are_surely_denied_for_writing(prefix, path):
    subject = as_path(prefix + path)
    assert surely_denied("fs_write", subject)
    assert not possibly_allowed("fs_write", subject)


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("path", ALLOWED_WRITES)
def test_python_files_of_an_app_are_possibly_allowed_and_not_surely_denied(
    prefix, path
):
    subject = as_path(prefix + path)
    assert possibly_allowed("fs_write", subject)
    assert not surely_denied("fs_write", subject)


@pytest.mark.parametrize("path", NOT_ALLOWED_WRITES)
def test_other_files_are_not_allowed_for_writing(path):
    assert not possibly_allowed("fs_write", path)


def test_the_write_allow_list_names_only_python_files_in_app_folders():
    patterns = [p for rule in rules("fs_write", "allow") for p in rule["match"]]
    assert patterns
    for pattern in patterns:
        assert pattern.endswith(".py"), pattern
        assert pattern.startswith(("apps/", "**/apps/")), pattern


def test_the_agent_can_read_everything():
    reads = rules("fs_read", "allow")
    assert len(reads) == 1
    assert not reads[0].get("match")


SHELL_ALLOWED = [
    "hexperiment run apps/churn",
    "hexperiment run --fast apps/churn",
    ".venv/Scripts/hexperiment run apps/churn",
    ".venv\\Scripts\\hexperiment run --fast apps/churn",
    ".venv/bin/hexperiment run apps/churn",
]

SHELL_NOT_ALLOWED = [
    "hexperiment run apps/churn && curl http://example.com",
    "hexperiment run apps/churn & whoami",
    "hexperiment run apps/a; rm -rf .",
    "hexperiment run apps/a | tee out.txt",
    "hexperiment run apps/a > out",
    "hexperiment run apps/a < in",
    "hexperiment run apps/a `whoami`",
    "hexperiment run apps/$(whoami)",
    "hexperiment run ../x",
    "hexperiment run apps/../../x",
    "hexperiment run churn",
    "hexperiment run --full apps/churn",
    "hexperiment run",
    "hexperiment init apps/x",
    "hexperiment init --kiro",
    "hexperiment hook",
    "hexperiment-mcp",
    "git status",
    "git push",
    "git commit -m x",
    "git checkout -- apps",
    "python -c 1",
    "python train.py",
    ".venv/Scripts/python -m pytest",
    "pip install x",
    "ls",
    "rm -rf apps",
    "rm apps/churn/verdict.json",
    "del verdict.json",
    "Remove-Item apps/churn/verdict.json",
    "curl http://example.com",
    "wget http://example.com",
    "Invoke-WebRequest http://example.com",
    "iwr http://example.com",
    "irm http://example.com",
    "cat apps/churn/hex.toml",
]


@pytest.mark.parametrize("command", SHELL_ALLOWED)
def test_hexperiment_run_on_an_app_is_possibly_allowed(command):
    assert possibly_allowed("shell", command)
    assert not surely_denied("shell", command)


@pytest.mark.parametrize("command", SHELL_NOT_ALLOWED)
def test_other_commands_are_not_allowed(command):
    assert not possibly_allowed("shell", command)


RUN_PATTERN = re.compile(
    r"^(?:\.venv[/\\](?:Scripts|bin)[/\\])?hexperiment run (?:--fast )?apps/\*$"
)


def test_the_shell_allow_list_admits_only_hexperiment_run_on_apps():
    patterns = [p for rule in rules("shell", "allow") for p in rule["match"]]
    assert patterns
    for pattern in patterns:
        assert RUN_PATTERN.match(pattern), pattern


def test_the_shell_deny_list_names_chaining_characters_and_command_families():
    denied = {p for rule in rules("shell", "deny") for p in rule["match"]}
    for needed in ("*&*", "*;*", "*|*", "*>*", "*<*", "*..*", "git*", "curl*", "pip*"):
        assert needed in denied, needed


# The helpers above must catch what they claim to catch.


@pytest.mark.parametrize(
    ("pattern", "subject", "loose", "expected"),
    [
        ("apps/*", "apps/churn", False, True),
        ("apps/*", "apps/churn/x", False, False),
        ("apps/*", "apps/churn/x", True, True),
        ("apps/**", "apps/churn/x", False, True),
        ("apps/**/*.py", "apps/churn/a/b.py", False, True),
        ("apps/**/*.py", "apps/b.py", False, True),
        ("apps/**/*.py", "apps/b.md", True, False),
        ("**/hex.toml", "hex.toml", False, True),
        ("**/hex.toml", "a/b/hex.toml", False, True),
        ("**/hex.toml", "a/xhex.toml", False, False),
        ("**/.kiro/**", "C:/p/.kiro/agents/a.json", False, True),
    ],
)
def test_the_glob_readings(pattern, subject, loose, expected):
    assert bool(glob_regex(pattern, loose=loose).fullmatch(subject)) is expected


def test_a_deny_rule_wins_over_an_allow_rule_in_the_simulator():
    # apps/churn/data/helper.py matches the Python allow rule and a data deny rule.
    path = "apps/churn/data/helper.py"
    assert any(rule_matches(r, path, loose=True) for r in rules("fs_write", "allow"))
    assert not possibly_allowed("fs_write", path)
