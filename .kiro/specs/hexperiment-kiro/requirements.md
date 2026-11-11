# Requirements Document: hexperiment-kiro

## Introduction

`hexperiment-core` gives a training script a verdict. This spec makes that verdict part of how Kiro works on ML code, so an agent cannot hand over a haunted model as finished work:

- a **hook** re-checks an app every time an agent saves its training code, and puts the result in the agent's context;
- a **workflow** loops an agent on an app until `hexperiment run` writes an `APPROVED` verdict, with a restricted agent that can change the training code but never the verdict, the thresholds or the waivers;
- an **MCP server** lets any agent list apps, run audits, read verdicts and history, and ask what a ghost means;
- a **power** packages the kit so another Kiro user can install it in one step, and `hexperiment init` writes the hook, the workflow, the agent and the steering into their project.

Scope: the `hexperiment hook` and `hexperiment init` commands, the Kiro hook, workflow and agent files, the MCP server, and the power. The example apps, the gallery and the leak benchmark are covered by later specs. File formats follow the Kiro documentation as of November 2026 (hooks `.kiro/hooks/*.json` with `trigger` and `action`; workflows `.kiro/workflows/*.workflow.json`; powers in the Agent Plugins format with `plugin.json`).

## Glossary

- **Workspace:** the folder Kiro has open; apps are found below it.
- **App:** as in `hexperiment-core`: a folder with `hex.toml` and `train.py`.
- **Hook event:** the JSON document Kiro writes to a command hook's standard input.
- **Ghost hunter:** the restricted custom agent that the workflow uses to fix an app.
- **Power:** the installable package in `power/hexperiment/`.

## Requirements

### Requirement 1: Check on save

**User Story:** As a data scientist working with an agent, I want every agent edit to my training code to be audited right away, so that a leak is reported in the same turn it was introduced.

#### Acceptance Criteria

1. THE repository SHALL contain `.kiro/hooks/hexperiment.json` with one hook that triggers when the agent saves a file, whose matcher selects Python files, and whose action is the command `hexperiment hook` (which decides whether the file belongs to an app).
2. WHEN `hexperiment hook` runs THEN the system SHALL read the hook event from standard input, find the saved file's path in it, and locate the app as the nearest parent folder of that file that contains `hex.toml` and `train.py`, without leaving the workspace.
3. WHEN an app is found THEN the system SHALL run that app in fast mode, exactly as `hexperiment run --fast <app>` does, and write the summary to standard output, followed by one line telling the agent to run the full check (`hexperiment run <app>`) before calling the work done.
4. IF no app is found, the event cannot be parsed, or the saved file is not a Python file THEN the system SHALL write nothing and exit with code `0`.
5. THE `hexperiment hook` command SHALL always exit with code `0`, so that a failing or haunted run is reported to the agent and never blocks the agent's turn; a run that fails (exit code `2` of `hexperiment run`) SHALL be reported on standard output with the failure message.
6. THE hook SHALL finish within its configured timeout (`120` seconds), and a fast run SHALL never start the canary execution.
7. A run started by the hook SHALL be recorded in the app's run history and SHALL count as a use of the test set, like any other run, because the agent sees the test metric.

### Requirement 2: Loop until approved

**User Story:** As a data scientist, I want to hand an app to an agent and get it back only when the full audit approves it, so that "done" means "proven leak-free", not "the agent says so".

#### Acceptance Criteria

1. THE repository SHALL contain `.kiro/workflows/exorcise.workflow.json`, a workflow with one input `app` (the app folder, relative to the workspace).
2. THE workflow SHALL repeat a step run by the ghost hunter agent, and SHALL stop only when the app's `verdict.json` exists and its `status` is `APPROVED`.
3. THE stop condition SHALL read the `verdict.json` that `hexperiment run` writes in the app folder; because `hexperiment run` deletes that file before every run, a failed or interrupted run SHALL never leave an earlier `APPROVED` verdict in place.
4. THE workflow SHALL run the step at most `5` times and then pause for the human, reporting the latest status and findings.
5. EACH step SHALL tell the agent to run `hexperiment run <app>` (full mode), read `verdict.json` and the latest model card, and, unless the status is `APPROVED`, change the app's code to address each finding using its fix hint, then run `hexperiment run <app>` again.

### Requirement 3: The ghost hunter cannot cheat

**User Story:** As a reviewer, I want the agent that fixes the model to be unable to fake or relax the verdict, so that an `APPROVED` from the loop can be trusted.

#### Acceptance Criteria

1. THE repository SHALL contain `.kiro/agents/ghost-hunter.json`, a custom agent in the Kiro CLI V3 permission format, used by the workflow.
2. THE ghost hunter SHALL be allowed to write only Python files inside app folders.
3. THE ghost hunter SHALL be denied writing `hex.toml`, `verdict.json`, `model_card.md`, anything under an app's `runs/`, `.hex/` or `data/` folders, anything under `.kiro/`, and the `hexperiment` package itself; deny rules SHALL take precedence over allow rules.
4. THE ghost hunter's shell access SHALL be limited to `hexperiment run` and `hexperiment run --fast` on an app, and SHALL deny every other command, including version control, deletion and network commands.
5. THE ghost hunter's prompt SHALL state that thresholds and waivers are the human's decision, that removing features or the `audit` call does not count as a fix, and that it must report a finding it cannot fix instead of working around it.
6. THE agent file SHALL pass `kiro-cli agent validate`.

### Requirement 4: MCP server

**User Story:** As an agent (in Kiro or any MCP client), I want tools to list, run and read hexperiment audits, so that I can check ML work without knowing the file layout.

#### Acceptance Criteria

1. THE package SHALL provide an MCP server over stdio, built with the official MCP Python SDK, installed with the optional extra `hexperiment[mcp]` and started by the console command `hexperiment-mcp`; the core package SHALL NOT import the SDK.
2. THE server SHALL treat `HEXPERIMENT_WORKSPACE`, when set, or else its working directory as the workspace, and SHALL only read or run apps whose resolved folder lies inside the workspace and contains `hex.toml` and `train.py`.
3. THE server SHALL expose the tools:
   - `list_apps()`: every app below the workspace (skipping hidden folders, virtual environments and `node_modules`), with its name, relative path and latest status or `none`;
   - `run_audit(app, mode="full")`: runs `hexperiment run` on the app in `full` or `fast` mode and returns the status, exit code and summary text; this is the only tool that executes code;
   - `get_verdict(app)`: the content of the app's `verdict.json`, or a message that the app has no current verdict;
   - `get_history(app)`: one entry per earlier run (run id, status, metric value, ghosts found), oldest first;
   - `explain_ghost(name)`: for a ghost name or check name, what it means, why it matters and how to fix it.
4. THE server SHALL expose each app's latest model card as a resource.
5. IF a tool gets an app outside the workspace, an app without `hex.toml` and `train.py`, or an unknown mode or ghost name THEN it SHALL return an error result naming the problem, and SHALL NOT run any code.
6. THE server SHALL run at most one audit at a time, and an audit SHALL respect the app's `[run] timeout_seconds`.
7. THE server SHALL never write to an app folder except through `hexperiment run`.

### Requirement 5: Setting up a project

**User Story:** As a Kiro user who just installed the power, I want one command that adds the kit to my project, so that I do not copy files by hand.

#### Acceptance Criteria

1. THE package SHALL provide `hexperiment init <app_dir>` that creates `<app_dir>` with a commented `hex.toml` template and a `train.py` template that loads with `hexperiment.load()`, splits with `hexperiment.split()`, fits on training rows only, and calls `hexperiment.audit(...)` with `y_score_train`.
2. THE command `hexperiment init --kiro` SHALL write the hook, the workflow, the ghost hunter agent and the ML-integrity steering file into the current project's `.kiro/` folder.
3. THE `init` command SHALL never overwrite an existing file; it SHALL list each file it created and each file it skipped.
4. THE files written by `init --kiro` SHALL be identical to the ones in this repository's `.kiro/` folder, and a test SHALL fail if they drift apart.

### Requirement 6: The power

**User Story:** As a Kiro user, I want to install Hexperiment as a power, so that Kiro knows how to audit ML work whenever I talk about leakage, model cards or suspiciously good scores.

#### Acceptance Criteria

1. THE repository SHALL contain `power/hexperiment/plugin.json` in the Agent Plugins 1.0.0 format, with `name` `hexperiment`, a semver `version`, a `description`, an `author` and activation `keywords` (including "data leakage", "train test split", "model card" and "too good to be true").
2. THE power SHALL contain an `mcp.json` that starts the MCP server with `hexperiment-mcp` over stdio and needs no secrets.
3. THE power SHALL contain a skill (`skills/<name>/SKILL.md`) that tells the agent when to use the kit, the ML-integrity rules, how to read a verdict and each ghost, how to start with `hexperiment init`, and that only `hexperiment run` can produce an `APPROVED` verdict.
4. THE power SHALL install with `kiro-cli powers install power/hexperiment` without errors.

### Requirement 7: Quality

**User Story:** As a maintainer, I want the Kiro layer tested like the core, so that it keeps working as the core changes.

#### Acceptance Criteria

1. THE tests SHALL cover: event parsing and app lookup in `hexperiment hook` (including events with no path, paths outside the workspace and non-Python files), the exit code and output of the hook command for an approved, a rejected and a failed run, each MCP tool through an in-memory MCP client session, the workspace guard, `init` creating and skipping files, and the drift check between `init --kiro` and `.kiro/`.
2. THE hook, workflow, agent and power JSON files SHALL be checked by tests for the fields their Kiro formats require.
3. THE core package's runtime dependencies SHALL stay `pandas`, `numpy` and `scikit-learn`; the MCP SDK SHALL be pinned in the `mcp` extra.
4. THE full test suite, including the new tests, SHALL pass with `.venv/Scripts/python -m pytest hexperiment/tests -q` in under 90 seconds, and `ruff` SHALL report no problems.
