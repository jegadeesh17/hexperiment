# Requirements Document: hexperiment-kiro

## Introduction

`hexperiment-core` gives a training script a verdict. This spec makes that verdict part of how Kiro works on ML code, so an agent cannot hand over a haunted model as finished work:

- a **hook** re-checks an app every time an agent saves its training code; the check clears the app's verdict and is never `APPROVED`, so a saved change replaces an earlier `APPROVED` (Kiro's documentation says the hook's output is also added to the agent's context; Kiro CLI 3.2.0 and Kiro IDE 1.4.154 do not do this, see `design.md`, Open question 14);
- a **stop check** tells the agent, when it ends a turn, which apps it changed in this session that are not `APPROVED` (Kiro CLI 3.3.0 gives the agent the message of a Stop hook that exits with code `1`; checked live, see `design.md`);
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

1. THE repository SHALL contain `.kiro/hooks/hexperiment.json` with a save hook that triggers when the agent saves a file, whose matcher selects Python files, and whose action is the command `hexperiment hook` (which decides whether the file belongs to an app).
2. WHEN `hexperiment hook` runs THEN the system SHALL read the hook event from standard input, find the saved file's path in it, and locate the app as the nearest parent folder of that file that contains `hex.toml` and `train.py`, without leaving the workspace.
3. WHEN an app is found THEN the system SHALL run that app in fast mode, exactly as `hexperiment run --fast <app>` does, and write the summary to standard output, followed by one line telling the agent to run the full check (`hexperiment run <app>`) before calling the work done.
4. IF no app is found, the event cannot be parsed, or the saved file is not a Python file THEN the system SHALL write nothing and exit with code `0`.
5. THE `hexperiment hook` command SHALL always exit with code `0`, so that a failing or haunted run never blocks the agent's turn, and SHALL print its summary on standard output, which Kiro's documentation says is added to the agent's context; a run that fails (exit code `2` of `hexperiment run`) SHALL be reported on standard output with the failure message.
6. THE hook SHALL finish within its configured timeout (`120` seconds), and a fast run SHALL never start the canary execution.
7. A run started by the hook SHALL be recorded in the app's run history and SHALL count as a use of the test set, like any other run, because the agent can read the test metric in `verdict.json`.
8. IF another `hexperiment run` holds the app's run lock THEN the hook SHALL print that run's refusal message and the full-check notice and exit with code `0`.

### Requirement 2: Loop until approved

**User Story:** As a data scientist, I want to hand an app to an agent and get it back only when the full audit approves it, so that "done" means "proven leak-free", not "the agent says so".

#### Acceptance Criteria

1. THE repository SHALL contain `.kiro/workflows/exorcise.workflow.json`, a workflow with one input `app` (the app folder, relative to the workspace).
2. THE workflow SHALL repeat a step run by the ghost hunter agent, and SHALL stop only when the app's `verdict.json` exists and its `status` is `APPROVED`.
3. THE stop condition SHALL read the `verdict.json` that `hexperiment run` writes in the app folder; because `hexperiment run` deletes that file before every run, a failed or interrupted run SHALL never leave an earlier `APPROVED` verdict in place.
4. THE workflow SHALL run the step at most `5` times and then pause for the human, reporting the latest status and findings.
5. EACH step SHALL tell the agent to run the full audit with the MCP tool `run_audit` of the workspace `hexperiment` server, read `verdict.json` and the latest model card, and, unless the status is `APPROVED`, change the app's code to address each finding using its fix hint, then call `run_audit` again.

### Requirement 3: The ghost hunter cannot cheat

**User Story:** As a reviewer, I want the agent that fixes the model to be unable to fake or relax the verdict, so that an `APPROVED` from the loop can be trusted.

#### Acceptance Criteria

1. THE repository SHALL contain `.kiro/agents/ghost-hunter.json`, a custom agent in the Kiro CLI V3 permission format, used by the workflow.
2. THE ghost hunter SHALL be allowed to write only Python files inside app folders.
3. THE ghost hunter SHALL be denied writing `hex.toml`, `verdict.json`, `model_card.md`, `fetch_data.py`, anything under an app's `runs/`, `.hex/`, `data/` or `haunted/` folders, anything under `.kiro/`, and the `hexperiment` package itself; deny rules SHALL take precedence over allow rules.
4. THE ghost hunter SHALL have no shell tool. It SHALL run audits only through the MCP tool `run_audit` of the workspace `hexperiment` server; its other tools SHALL be file reading, file writing (limited by 3.2 and 3.3) and the read-only tools of that server; the Python code `run_audit` executes is not sandboxed.
5. THE ghost hunter's prompt SHALL state that thresholds and waivers are the human's decision; that removing the `audit` call, or a feature that is not the cause of a finding, does not count as a fix; that dropping a column flagged by The Possession is a fix only when the column is recorded after the outcome, and the agent must say why it is; and that it must report a finding it cannot fix instead of working around it.
6. THE agent file SHALL pass `kiro-cli agent validate`.
7. THE ghost hunter SHALL NOT write outside `<app>/` except to paths listed in its allow rules, proven by a headless write probe whose log is committed. The probe SHALL use neutral file names and SHALL include root files, case variants of denied folders, `..` segments and an absolute path outside the workspace.
8. WHERE Kiro supports blocking a tool call from a `preToolUse` agent hook, THE ghost hunter SHALL run `hexperiment guard-write` before every file write. The command SHALL resolve the path (`..`, absolute paths, letter case) and allow only Python files inside `apps/<app>/`, where `<app>` is not `tests`, outside the app's `haunted/`, `data/`, `runs/`, `.hex/` and `hexperiment/` folders, and not named `fetch_data.py` or `hexperiment.py`. It SHALL refuse every other path, and any hook input it cannot read, with exit code 2. WHERE Kiro does not support it, the README Known limits SHALL list the paths the deny rules cannot refuse.

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
8. THE server SHALL start quickly: importing its module SHALL NOT load scikit-learn or SciPy, and `import hexperiment` SHALL NOT load pandas, scikit-learn or SciPy until one of the package's exported functions is used.
9. IF the workspace holds no app THEN `list_apps` SHALL add a `hint` that names the workspace it used and says how to point the server at the project: the `.kiro/settings/mcp.json` entry that `hexperiment init --kiro` writes, or `HEXPERIMENT_WORKSPACE`.

### Requirement 5: Setting up a project

**User Story:** As a Kiro user who just installed the power, I want one command that adds the kit to my project, so that I do not copy files by hand.

#### Acceptance Criteria

1. THE package SHALL provide `hexperiment init <app_dir>` that creates `<app_dir>` with a commented `hex.toml` template and a `train.py` template that loads with `hexperiment.load()`, splits with `hexperiment.split()`, fits on training rows only, and calls `hexperiment.audit(...)` with `y_score_train`.
2. THE command `hexperiment init --kiro` SHALL write the hook, the workflow, the ghost hunter agent and the ML-integrity steering file into the current project's `.kiro/` folder. WHEN `.kiro/settings/mcp.json` does not exist THEN it SHALL also create it with a `hexperiment` server entry, so that Kiro starts `hexperiment-mcp` in the project folder; WHEN the file exists without a `hexperiment` entry THEN it SHALL leave the file unchanged and print the entry to add.
3. THE `init` command SHALL never overwrite an existing file; it SHALL list each file it created and each file it skipped.
4. THE four files written by `init --kiro` SHALL be identical to the ones in this repository's `.kiro/` folder, the `hexperiment` entry it writes into `.kiro/settings/mcp.json` SHALL equal the one in this repository's `.kiro/settings/mcp.json`, and a test SHALL fail if they drift apart.

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

1. THE tests SHALL cover: event parsing and app lookup in `hexperiment hook` (including events with no path, paths outside the workspace and non-Python files), the exit code and output of the hook command for a clean (`UNVERIFIED`), a rejected and a failed run, each MCP tool through an in-memory MCP client session, the workspace guard, `init` creating and skipping files, and the drift check between `init --kiro` and `.kiro/`.
2. THE hook, workflow, agent and power JSON files SHALL be checked by tests for the fields their Kiro formats require.
3. THE core package's runtime dependencies SHALL stay `pandas`, `numpy` and `scikit-learn`; the MCP SDK SHALL be pinned in the `mcp` extra.
4. THE full test suite, including the new tests, SHALL pass with `.venv/Scripts/python -m pytest hexperiment/tests -q` and the time it took SHALL be reported (There is no time limit: the time depends on the machine's background load. On the reference laptop (AMD Ryzen 7 3750H, 4 cores, 8 threads), on the power plan it is left on (Power saver) with the Kiro IDE, a browser and an editor open, the suite took 135 to 213 s on 2026-11-12; 84 tests that train real models take most of it.), and `ruff` SHALL report no problems.

### Requirement 8: One run per app at a time

**User Story:** As a user whose agent, hook and MCP server can all start runs, I want two runs on one app never to overlap, so that their `.hex/` exchange files cannot mix.

#### Acceptance Criteria

1. WHILE a `hexperiment run` is running on an app THEN a second `hexperiment run` on the same app SHALL exit with code `2` and a message that names the app, without writing a run folder or changing `verdict.json`.
2. THE lock SHALL be held by the operating system for the life of the first process, so that it is released when that process ends, also when it is killed.

### Requirement 9: Check before the agent finishes

**User Story:** As a data scientist working with an agent, I want the agent to be told, when it ends its turn, that an app it changed is not approved, so that it runs the full audit or says plainly that the work is not done.

#### Acceptance Criteria

1. THE file `.kiro/hooks/hexperiment.json` SHALL also contain a hook that triggers when the agent stops (trigger `Stop`), whose action is the command `hexperiment hook --stop`, with a timeout of `30` seconds.
2. WHEN `hexperiment hook` (the save hook) finds an app and the event has a valid `session_id` THEN the system SHALL write that id to the app's `.hex/agent-session` before the run; a failed write SHALL NOT change the hook's output or exit code.
3. WHEN `hexperiment hook --stop` runs THEN the system SHALL read the event from standard input and select every app of the workspace whose `.hex/agent-session` holds the event's `session_id` and whose `verdict.json` is missing or has a status other than `APPROVED`.
4. WHEN at least one app is selected THEN the system SHALL write a message to standard error that names each app with its state and tells the agent what to do next (run the full audit for an app with no verdict or an `UNVERIFIED` one; fix the findings or tell the user plainly that the app is `REJECTED` for a rejected one), and SHALL exit with code `1`.
5. WHEN no app is selected, the event has no valid `session_id`, or anything fails THEN the system SHALL write nothing and exit with code `0`.
6. THE stop check SHALL NOT run `train.py`, start a run, write a run record or count as a test-set use, and SHALL NOT import pandas, NumPy or scikit-learn.
