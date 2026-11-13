"""Format checks for the Kiro files and a model of the ghost hunter's permissions.

The files under test are the repository's own ``.kiro/hooks/hexperiment.json``,
``.kiro/workflows/exorcise.workflow.json`` and ``.kiro/agents/ghost-hunter.json``
(Requirements 1.1, 2.1 to 2.5, 3.1 to 3.5, 7.2), and the power in
``power/hexperiment/`` (Requirements 6.1 to 6.3, 7.2). The tests find them from this
file's location and fail with the path when one is missing.
"""

import json
import re
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from hexperiment import hook
from hexperiment.ghosts import CATALOG

# The trigger literal and the workflow input placeholder are named here once, so a
# correction after a live check in Kiro changes one line each.
EXPECTED_TRIGGER = "PostFileSave"
INPUT_REF = "{{app}}"

KIRO_DIR = Path(__file__).resolve().parents[2] / ".kiro"
PACKAGE_DIR = Path(__file__).resolve().parents[1]
POWER_DIR = Path(__file__).resolve().parents[2] / "power" / "hexperiment"
STEERING_FILE = KIRO_DIR / "steering" / "ml-integrity.md"
HOOK_FILE = "hooks/hexperiment.json"
WORKFLOW_FILE = "workflows/exorcise.workflow.json"
AGENT_FILE = "agents/ghost-hunter.json"

CAPABILITIES = {"fs_read", "fs_write", "mcp"}
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


def hook_entry(name: str = "hexperiment-check-on-save") -> dict[str, Any]:
    config = read_kiro_json(HOOK_FILE)
    assert config["version"] == "v1"
    assert [h["name"] for h in config["hooks"]] == [
        "hexperiment-check-on-save",
        "hexperiment-check-on-stop",
    ]
    return next(h for h in config["hooks"] if h["name"] == name)


def stop_entry() -> dict[str, Any]:
    return hook_entry("hexperiment-check-on-stop")


def test_the_hook_file_has_version_v1_and_two_named_hooks():
    for name in ("hexperiment-check-on-save", "hexperiment-check-on-stop"):
        entry = hook_entry(name)
        assert isinstance(entry["description"], str) and entry["description"].strip()


def test_the_stop_hook_runs_the_stop_check_with_a_30_second_timeout():
    entry = stop_entry()
    assert entry["trigger"] == "Stop"
    assert entry["action"] == {"type": "command", "command": "hexperiment hook --stop"}
    assert entry["timeout"] == 30
    assert "matcher" not in entry


def test_the_stop_hook_leaves_the_save_hook_unchanged():
    entry = hook_entry()
    assert entry["trigger"] == EXPECTED_TRIGGER
    assert entry["action"] == {"type": "command", "command": "hexperiment hook"}
    assert entry["timeout"] == 120
    assert entry["matcher"] == ".*\\.py$"


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
    assert INPUT_REF in prompt
    assert "run_audit" in prompt
    assert f"hexperiment run {INPUT_REF}" not in prompt
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


SHELL_TOOLS = ("shell", "execute_pwsh", "run_command", "*", "@builtin")


def test_the_agent_has_no_shell_rule_and_no_shell_tool():
    config = agent()
    assert all(rule["capability"] != "shell" for rule in config["permissions"]["rules"])
    for tool in SHELL_TOOLS:
        assert tool not in config["tools"], tool


def test_the_agent_reaches_the_hexperiment_server_through_the_workspace_mcp_json():
    config = agent()
    assert "@hexperiment" in config["tools"]
    assert config["includeMcpJson"] is True


def test_the_agent_has_exactly_one_mcp_rule_that_allows_the_hexperiment_tools():
    mcp_rules = [r for r in agent()["permissions"]["rules"] if r["capability"] == "mcp"]
    assert len(mcp_rules) == 1
    assert mcp_rules[0]["effect"] == "allow"
    assert mcp_rules[0]["match"] == ["hexperiment/*"]


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
        "You have no shell",  # 3.4: audits run only through run_audit
        "the hexperiment MCP tool run_audit",
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
    segments. Loose: ``*`` matches anything. In both, ``**/`` also matches no folder,
    and no star matches a line feed or a carriage return (a live check showed that a
    command with a line break matches no allow rule).
    """
    anything = r"[^\r\n]*"
    one_segment = r"[^/\r\n]*"
    parts: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            parts.append(f"(?:{anything}/)?")
            i += 3
        elif pattern.startswith("**", i):
            parts.append(anything)
            i += 2
        elif pattern[i] == "*":
            parts.append(anything if loose else one_segment)
            i += 1
        else:
            parts.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(parts))


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
    "apps/verify.py",  # directly in apps/, not in an app folder
    "apps/tests/test_x.py",  # a test file
    "apps/x/tests/test_a.py",
    "apps/x/conftest.py",
    "apps/x/../y.py",  # ".." is not resolved before matching
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
        assert pattern.startswith(("apps/*/", "**/apps/*/")), pattern


def test_the_write_deny_list_names_dot_dot_and_test_files():
    denied = {p for rule in rules("fs_write", "deny") for p in rule["match"]}
    for needed in ("*..*", "**/tests/**", "**/test_*.py", "**/conftest.py"):
        assert needed in denied, needed


def test_no_write_or_shell_pattern_contains_a_control_character():
    """A pattern with a line feed made every shell call fail in Kiro CLI 3.2.0."""
    for capability in ("fs_write", "mcp"):
        for effect in ("allow", "deny"):
            for rule in rules(capability, effect):
                for pattern in rule["match"]:
                    bad = [c for c in pattern if ord(c) < 0x20]
                    assert not bad, f"{capability} {effect} pattern {pattern!r}"


def test_the_agent_can_read_everything():
    reads = rules("fs_read", "allow")
    assert len(reads) == 1
    assert not reads[0].get("match")


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
        ("apps/*", "apps/a\nb", True, False),
        ("apps/*", "apps/a\rb", True, False),
        ("apps/*", "apps/a\nb", False, False),
        ("apps/**", "apps/a/b\nc", False, False),
        ("**/x.py", "a\nb/x.py", True, False),
        ("a*", "a\r\n", True, False),
        ("a*", "ab c", True, True),
    ],
)
def test_the_glob_readings(pattern, subject, loose, expected):
    assert bool(glob_regex(pattern, loose=loose).fullmatch(subject)) is expected


def test_a_deny_rule_wins_over_an_allow_rule_in_the_simulator():
    # apps/churn/data/helper.py matches the Python allow rule and a data deny rule.
    path = "apps/churn/data/helper.py"
    assert any(rule_matches(r, path, loose=True) for r in rules("fs_write", "allow"))
    assert not possibly_allowed("fs_write", path)


# --- the power (Requirements 6.1 to 6.3) ---------------------------------------------

SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
REQUIRED_KEYWORDS = (
    "data leakage",
    "train test split",
    "model card",
    "too good to be true",
)
SKILL_NAME = "hexperiment"
RULE_LINE = re.compile(r"^(\d+)\. \*\*(.+?)\*\*", re.MULTILINE)


def read_power_json(name: str) -> dict[str, Any]:
    """Load a file of the power; fail with a clear message when it is unusable."""
    path = POWER_DIR / name
    if not path.is_file():
        pytest.fail(
            f"{path} is missing; create it from 'power/hexperiment/' in design.md.",
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


def package_version() -> str:
    pyproject = tomllib.loads((PACKAGE_DIR / "pyproject.toml").read_text("utf-8"))
    return pyproject["project"]["version"]


def read_skill() -> tuple[dict[str, str], str]:
    """The front matter and the body of the skill, with ``\\n`` line ends."""
    path = POWER_DIR / "skills" / SKILL_NAME / "SKILL.md"
    if not path.is_file():
        pytest.fail(
            f"{path} is missing; create it from 'power/hexperiment/' in design.md.",
            pytrace=False,
        )
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert text.startswith("---\n"), f"{path} must start with front matter"
    head, separator, body = text[4:].partition("\n---\n")
    assert separator, f"{path} front matter is not closed with ---"
    front: dict[str, str] = {}
    for line in head.splitlines():
        key, colon, value = line.partition(":")
        assert colon, f"{path} front matter line {line!r} is not 'key: value'"
        value = value.strip()
        front[key.strip()] = json.loads(value) if value.startswith('"') else value
    return front, body


def steering_rules() -> list[tuple[str, str, str]]:
    """Number, bold title and full line of each rule of ``ml-integrity.md``."""
    text = STEERING_FILE.read_text(encoding="utf-8").replace("\r\n", "\n")
    lines = {line for line in text.splitlines() if RULE_LINE.match(line)}
    rules_found = []
    for line in sorted(lines):
        match = RULE_LINE.match(line)
        assert match
        rules_found.append((match.group(1), match.group(2), line))
    return rules_found


def test_plugin_json_has_the_fields_of_the_agent_plugins_format():
    config = read_power_json("plugin.json")
    assert config["name"] == "hexperiment"
    for key in ("description", "license"):
        assert isinstance(config[key], str) and config[key].strip(), key
    assert isinstance(config["author"]["name"], str) and config["author"]["name"]


def test_the_plugin_version_is_semver_and_equals_the_package_version():
    version = read_power_json("plugin.json")["version"]
    assert SEMVER.match(version), version
    assert version == package_version()


def test_the_plugin_keywords_include_the_activation_phrases():
    keywords = read_power_json("plugin.json")["keywords"]
    assert all(isinstance(keyword, str) and keyword for keyword in keywords)
    assert len(keywords) == len(set(keywords))
    for needed in REQUIRED_KEYWORDS:
        assert needed in keywords, needed


def test_mcp_json_starts_one_stdio_server_with_the_hexperiment_mcp_command():
    config = read_power_json("mcp.json")
    servers = config["mcpServers"]
    assert list(servers) == ["hexperiment"]
    server = servers["hexperiment"]
    assert server["type"] == "stdio"
    assert server["command"] == "hexperiment-mcp"
    assert server["args"] == []


def test_mcp_json_needs_no_secrets():
    read_power_json("mcp.json")
    text = (POWER_DIR / "mcp.json").read_text(encoding="utf-8")
    assert "${" not in text
    assert read_power_json("mcp.json")["mcpServers"]["hexperiment"]["env"] == {}


def test_the_plugin_files_hold_no_variable_placeholders():
    for name in ("plugin.json", "mcp.json"):
        read_power_json(name)
        assert "${" not in (POWER_DIR / name).read_text(encoding="utf-8"), name


def test_the_skill_front_matter_name_equals_its_folder():
    front, _ = read_skill()
    assert front["name"] == SKILL_NAME
    assert (POWER_DIR / "skills" / front["name"] / "SKILL.md").is_file()
    assert front["description"].strip()


def test_the_steering_file_has_the_eight_numbered_rules():
    numbers = [number for number, _, _ in steering_rules()]
    assert numbers == [str(n) for n in range(8)]


@pytest.mark.parametrize("number", [str(n) for n in range(8)])
def test_every_rule_title_of_the_steering_file_appears_in_the_skill(number):
    _, body = read_skill()
    (title,) = [t for n, t, _ in steering_rules() if n == number]
    assert f"{number}. **{title}**" in body


@pytest.mark.parametrize("number", [str(n) for n in range(8)])
def test_the_skill_states_each_rule_as_the_steering_file_does(number):
    # This includes rule 2: drop the test rows that also appear in the training rows.
    _, body = read_skill()
    (line,) = [full for n, _, full in steering_rules() if n == number]
    assert line in body.splitlines()


def test_every_ghost_and_check_of_the_catalogue_appears_in_the_skill():
    _, body = read_skill()
    for entry in CATALOG:
        assert f"`{entry.check}`" in body, entry.check
        if entry.ghost is not None:
            assert entry.ghost in body, entry.ghost


def test_the_skill_names_the_three_statuses_and_the_exit_codes():
    _, body = read_skill()
    for status in ("APPROVED", "REJECTED", "UNVERIFIED"):
        assert status in body
    assert "Exit codes" in body


def test_the_skill_says_that_only_hexperiment_run_produces_approved():
    _, body = read_skill()
    assert "Only `hexperiment run` in full mode produces an APPROVED verdict" in body


def test_the_skill_explains_how_to_start_with_hexperiment_init():
    _, body = read_skill()
    assert "hexperiment init apps/<name>" in body
    assert "hexperiment init --kiro" in body


def test_the_skill_says_what_to_do_when_list_apps_finds_no_app():
    _, body = read_skill()
    assert "`list_apps` returns no apps" in body
    assert "hexperiment init --kiro" in body
    assert ".kiro/settings/mcp.json" in body


def test_the_skill_does_not_tell_the_agent_to_ignore_a_fix_hint():
    """The fix hint of The Doppelganger is right now; nothing needs overriding."""
    _, body = read_skill()
    assert "may still say" not in body
    assert "duplicates before splitting" not in body
    assert "Follow rule 2 instead" not in body


def test_the_skill_says_that_thresholds_and_waivers_belong_to_the_human():
    _, body = read_skill()
    assert "Thresholds and waivers belong to the human" in body


def test_the_skill_names_the_command_that_starts_the_workflow():
    _, body = read_skill()
    assert "/workflow run exorcise --app apps/<name>" in body


def test_the_skill_says_hooks_run_hexperiment_and_kiro_starts_hexperiment_mcp():
    _, body = read_skill()
    assert "Hooks run the bare command `hexperiment`." in body
    assert "Kiro starts the bare command `hexperiment-mcp`" in body


def test_the_skill_says_the_ghost_hunter_audits_through_run_audit_and_has_no_shell():
    _, body = read_skill()
    assert "runs the audit through the MCP tool `run_audit`" in body
    assert "it has no shell" in body
    assert "`runs/`, `.hex/`, `data/`, `haunted/`, `.kiro/`" in body
