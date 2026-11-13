# How Kiro was used

Kiro plays two parts in Hexperiment. It **built** the kit, spec first, with custom agents that could each touch only their own files. And it is **part of the product**: the hooks, the MCP server, the ghost hunter agent and the `exorcise` workflow are how Hexperiment stops a leaky result from counting as done. This page describes both, and what we learned about Kiro along the way.

## 1. Building Hexperiment spec first

Every part of the kit was written from a Kiro spec in [.kiro/specs/](../.kiro/specs/): requirements in EARS form ("WHEN … THE system SHALL …"), a design, and an ordered task list.

| Spec | Requirements | Tasks | What it covers |
|---|---:|---:|---|
| [hexperiment-core](../.kiro/specs/hexperiment-core/) | 17 | 18 | `load`, `split`, `audit`, the checks, the canary test, verdicts, model cards, the CLI |
| [hexperiment-kiro](../.kiro/specs/hexperiment-kiro/) | 9 | 15 | hooks, MCP server, ghost hunter, `exorcise` workflow, power, `init --kiro` |
| [hexperiment-apps](../.kiro/specs/hexperiment-apps/) | 5 | 6 | the two example apps and their haunted scripts |
| [hexperiment-gallery](../.kiro/specs/hexperiment-gallery/) | 8 | 17 | the static gallery site |
| [hexperiment-benchmark](../.kiro/specs/hexperiment-benchmark/) | 6 | 19 | the 13-mechanism leak benchmark |

We wrote the requirements and the steering; Kiro wrote the designs, the task lists and the code. Steering in [.kiro/steering/](../.kiro/steering/) held the rules every session followed: `product.md`, `structure.md`, `tech.md`, the ML-integrity rules in `ml-integrity.md` (which ship with the product too), and `gallery-design.md` for the site.

### Three build agents, each boxed in

The build ran through three custom agents in [.kiro/agents/](../.kiro/agents/), mostly headless in Kiro CLI (`kiro-cli chat --no-interactive` with a custom agent), one task per session:

| Agent | May write | May run | Denied |
|---|---|---|---|
| `spec-writer` | only `design.md` and `tasks.md` of a spec | nothing (no shell) | requirements and steering |
| `builder` | the kit, the apps, the power, the hooks, the workflow, the ghost hunter, and the task ticks | the project's Python and `hexperiment`, and read-only git | requirements, steering, docs, `.git`, secrets, the other agents' files; `git commit`, `git push`, deleting files, downloads |
| `reviewer` | nothing | `pytest`, `ruff`, read-only git | every write |

Every task was then verified outside Kiro: the full test suite and the linter were run again, and the task was committed only when both passed. Nothing was committed or pushed by an agent.

### Moments where the spec process paid off

- **The spec writer refused a statistically wrong requirement.** The first draft said the canary would shuffle test values within each column. The spec writer pointed out that shuffling keeps each column's mean, so a leak through a mean imputer could never be seen. The design was changed to a "tilted resample" of the test rows before any code existed. In the benchmark, the canary later rejected all six apps that leak through a scaler or an imputer fitted on all rows.
- **The reviewer found a loophole.** A training script that reported its training-set predictions only in the normal run, and not in the canary run, could make the leak check pass silently. The builder closed it and showed the fix worked by breaking the code on purpose and watching the tests fail.
- **Mutation checks as a habit.** Across the tasks the builder repeatedly broke its own code on purpose (a wrong boundary, a missing salt, two decision steps swapped) to show that the tests would catch it. The suite now has about 3,100 tests, including property-based tests with Hypothesis.

## 2. Kiro as part of the product

Hexperiment's job is to make "no leakage" a condition the agent must meet before its result counts. Each Kiro feature carries one piece of that:

| Feature | File | Role |
|---|---|---|
| Steering | [.kiro/steering/ml-integrity.md](../.kiro/steering/ml-integrity.md) | Eight rules for any training script: let the kit own the data, split first, no row in both sets, no features from the future, respect time, treat suspicion as a signal, always end with an audit, fix the cause and not the check. |
| Save hook | [.kiro/hooks/hexperiment.json](../.kiro/hooks/hexperiment.json) | When the agent saves a Python file of an app, runs the fast check. The run first clears the app's verdict, so an old `APPROVED` never covers new code. |
| Stop hook | same file | When the agent ends its turn, lists the apps it changed in this session that are not `APPROVED` and tells it to run the full audit or to say the work is not done. |
| MCP server | `hexperiment-mcp`, entry in [.kiro/settings/mcp.json](../.kiro/settings/mcp.json) | `list_apps`, `run_audit`, `get_verdict`, `get_history`, `explain_ghost`, and each app's model card as a resource. |
| Custom agent | [.kiro/agents/ghost-hunter.json](../.kiro/agents/ghost-hunter.json) | Fixes leaks in one app. May edit the app's Python files and call the hexperiment tools; has no shell; may not write `hex.toml`, verdicts, model cards, run history, data, tests, `.kiro` or the kit. |
| Workflow | [.kiro/workflows/exorcise.workflow.json](../.kiro/workflows/exorcise.workflow.json) | Repeats one ghost-hunter step (run the full check, fix each finding at its cause, check again) until `verdict.json` says `APPROVED`. The stop condition reads the file itself, not the agent's reply. |
| Power | [power/hexperiment/](../power/hexperiment/) | Packages the skill and the MCP server for any Kiro session. `hexperiment init --kiro` writes the hooks, workflow, agent, steering and MCP entry into another project. |

### The loop, recorded

Both demo histories in the apps' `runs/` folders were produced by the `exorcise` workflow on the haunted scripts, on 2026-11-11 (UTC), in Kiro CLI 3.2.0. The ghost hunter's sessions are in [kiro/exorcise-hotel-cancellations.md](kiro/exorcise-hotel-cancellations.md) and [kiro/exorcise-flight-delays.md](kiro/exorcise-flight-delays.md), with every tool call and its output.

- **Hotels:** REJECTED at ROC AUC 1.0000 (The Possession on `reservation_status`, The Peeking Spirit on a country cancellation rate built from every label) → the ghost hunter dropped the column and rebuilt the rate from training rows → APPROVED at 0.9090. One round.
- **Flights:** REJECTED at 0.7971 (The Prophet: a random split on time-ordered data) → the ghost hunter switched to the kit's time split and also removed a carrier late rate built from every label → APPROVED at 0.6193. One round.

At the time of those recordings the ghost hunter ran `hexperiment run` through the shell. Live probes then showed that Kiro's shell rules are not a boundary inside a headless workflow (see below), so we took the shell away and gave it the MCP tool `run_audit` instead. A live headless `exorcise` run on a copy of the hotel app on 2026-11-13 (Kiro CLI 3.3.0) went REJECTED 1.0000 → APPROVED 0.9092 in one round, with no shell tool offered or used and nothing denied.

## 3. What we learned about Kiro

We checked how Kiro behaves with live probes wherever the documentation was silent or our design depended on it. The full notes are in [kiro/kiro-formats.md](kiro/kiro-formats.md).

- **Hook output does not reach the agent.** In Kiro CLI 3.2.0 and the Kiro IDE 1.4.154, a command hook on `PostFileSave` or `PostToolUse` ran and finished, but its output was not added to the agent's context, although the documentation says it is. In Kiro CLI 3.3.0 a `PostToolUse` hook's output did not reach the agent either. Our write-up is in [kiro/hook-output-issue.md](kiro/hook-output-issue.md). What does work, in Kiro CLI 3.3.0: a `Stop` hook that exits 1 with its message on standard error. The agent gets the message and one more turn, and the hook fires once per prompt, so it cannot loop. That is why Hexperiment's nudge is a Stop hook.
- **Shell rules are not a boundary in a headless workflow.** Inside a workflow step the shell tool is `run_command`, and once it is trusted, a command that matches no rule runs; deny rules are case-sensitive while PowerShell is not, and a chain with `;` is judged one part at a time. So the ghost hunter has no shell at all. The code it writes is still executed by every audit, which no rule can prevent; the README says so.
- **File-write rules held.** Write globs were case-insensitive and treated `\` and `/` alike, and every attempt to write `hex.toml` was refused. One gap: `..` in a path is not resolved, so the ghost hunter's rules also deny any path containing `..`.
- **MCP tools in custom agents.** With `includeMcpJson: true` and `@hexperiment` in `tools`, the agent finds the server's tools through `tool_load`; the server takes about 13 s to connect, so the step prompt has the agent read the app's files first instead of calling a tool at once.
- **A power's MCP server starts in the power's install folder,** not in the user's project, so it cannot see the project's apps. `hexperiment init --kiro` therefore also registers a workspace-level server, and `list_apps` explains what to do when it finds no app.
- **Trust comes first.** The Kiro IDE runs a project's hooks and workspace MCP servers only after the folder is trusted.

## 4. Numbers

- 5 specs, 45 requirements, 75 tasks. Kiro wrote the code of every task, and each task was verified before it was committed.
- About 3,100 tests, run in 2 to 3 minutes on a 2019 four-core laptop on a power-saving plan.
- Benchmark: 78 apps, 30 of 39 leaky apps rejected as predicted, 0 of 39 false alarms ([hexperiment/benchmark/README.md](../hexperiment/benchmark/README.md)).
