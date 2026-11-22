# How Kiro was used

Kiro plays two essential parts in Hexperiment. It **built** the kit, spec-first, with custom agents that could each touch only their own files. And it is **the execution engine of the product itself**: the steering contracts, the hooks, the MCP server, the sandboxed ghost hunter agent, and the `exorcise` workflow are how Hexperiment prevents an AI coding agent from shipping leaky, fraudulent machine-learning models. This page describes both dimensions and what we discovered about Kiro along the way.

## 1. Building Hexperiment spec first

Every part of the kit was written from a formal Kiro spec in [.kiro/specs/](../.kiro/specs/): requirements in EARS form ("WHEN … THE system SHALL …"), a technical design document, and an ordered task list.

| Spec | Requirements | Tasks | What it covers |
|---|---:|---:|---|
| [hexperiment-core](../.kiro/specs/hexperiment-core/) | 17 | 18 | `load`, `split`, `audit`, the data checks, the canary test, verdicts, model cards, the CLI |
| [hexperiment-kiro](../.kiro/specs/hexperiment-kiro/) | 9 | 15 | hooks, MCP server, ghost hunter, `exorcise` workflow, power, `init --kiro` |
| [hexperiment-apps](../.kiro/specs/hexperiment-apps/) | 5 | 6 | the two example apps and their haunted scripts |
| [hexperiment-gallery](../.kiro/specs/hexperiment-gallery/) | 8 | 17 | the static gallery site |
| [hexperiment-benchmark](../.kiro/specs/hexperiment-benchmark/) | 6 | 19 | the 13-mechanism leak benchmark |

We wrote the requirements and the steering; Kiro wrote the designs, the task lists, and the code. Steering in [.kiro/steering/](../.kiro/steering/) held the invariant rules every session followed: `product.md`, `structure.md`, `tech.md`, the ML-integrity rules in `ml-integrity.md` (which ship with the product too), and `gallery-design.md` for the site.

### Three build agents, each boxed in

The build ran through three custom agents in [.kiro/agents/](../.kiro/agents/), mostly headless in Kiro CLI (`kiro-cli chat --no-interactive` with a custom agent), one task per session:

| Agent | May write | May run | Denied |
|---|---|---|---|
| `spec-writer` | only `design.md` and `tasks.md` of a spec | nothing (no shell) | requirements and steering |
| `builder` | the kit, the apps, the power, the hooks, the workflow, the ghost hunter, and task ticks | project Python and `hexperiment`, read-only git | requirements, steering, docs, `.git`, secrets; `git commit`, `git push`, file deletion, network downloads |
| `reviewer` | nothing | `pytest`, `ruff`, read-only git | every write |

Every task was then verified outside Kiro: the full test suite and linter were run independently, and a task was committed only when both passed. No agent committed or pushed code.

### Moments where the spec process paid off

- **The spec writer refused a statistically wrong requirement.** The first draft said the canary would shuffle test values within each column. The spec writer pointed out that shuffling preserves each column's marginal mean, so a leak through an upstream mean imputer could never be observed. The design was changed to a "tilted resample" of test rows before any code was written. In the 78-app benchmark, the canary later rejected all six apps that leak through a scaler or imputer fitted on all rows.
- **The reviewer caught a subtle loophole.** A training script that reported its training-set predictions only during the normal run, but omitted them in the canary run, could allow the leak check to pass silently. The builder closed it and proved the fix by deliberately breaking the code (mutation check) and watching tests fail.
- **Mutation checks as a habit.** Across the tasks, the builder repeatedly injected deliberate faults (wrong boundary conditions, missing salts, swapped decision steps) to verify that the test suite caught each one. The suite now comprises ~3,400 tests, including Hypothesis property-based tests.

---

## 2. Kiro as part of the product: The Autonomous Exorcism Loop

Hexperiment’s core objective is to make **zero data leakage** a non-negotiable contract an agent must satisfy before its code can be merged or deployed.

```mermaid
graph TD
    subgraph Kiro IDE and CLI
        S[Steering: ml-integrity.md] -->|Directs behavior| A[Coding agent]
        A -->|Writes| F[train.py]
        F -->|File save| H[Save hook: fast run, one training execution]
        H -->|Side branch| X[Clears prior APPROVED]
        U[User starts /workflow run exorcise] --> W[exorcise.workflow.json]
        W -->|Runs step| G[ghost-hunter agent]
        G -->|Calls MCP tool| MCP[hexperiment-mcp run_audit]
        MCP -->|Full run with canary| V2{Verdict?}
        V2 -->|REJECTED: ghost found| G
        G -->|Fixes the code| F
        V2 -->|APPROVED| C[Model card written]
    end
```

Each Kiro platform primitive carries an essential role in this closed-loop architecture:

| Kiro Feature | Source File | Architectural Role |
|---|---|---|
| **Steering** | [.kiro/steering/ml-integrity.md](../.kiro/steering/ml-integrity.md) | **Policy Layer:** Eight rules enforcing clean data ingestion seams (`load`, `split`), temporal ordering, fixed seeds, and prohibiting verdict tampering. |
| **Save Hook** | [.kiro/hooks/hexperiment.json](../.kiro/hooks/hexperiment.json) | **Save-time fast run:** Runs the fast check on save (one normal training run, no canary); immediately invalidates any prior `APPROVED` status so stale passes never mask modified code. |
| **Stop Hook** | same file | **Turn Gate:** When the agent attempts to conclude its turn, checks whether modified apps are unapproved; if so, exits 1 with the findings on stderr, and the agent gets the message and one more turn. It fires once per prompt, so it cannot loop and does not guarantee an approved verdict. |
| **MCP Server** | `hexperiment-mcp` ([.kiro/settings/mcp.json](../.kiro/settings/mcp.json)) | **Structured Tool Interface:** Exposes `list_apps`, `run_audit`, `get_verdict`, `get_history`, `explain_ghost`, and model cards directly to agent context. |
| **Custom Agent** | [.kiro/agents/ghost-hunter.json](../.kiro/agents/ghost-hunter.json) | **Sandboxed Remediation:** Dedicated agent permitted to edit application Python files and call MCP audit tools; denied shell execution, data modification, and threshold editing. |
| **Workflow** | [.kiro/workflows/exorcise.workflow.json](../.kiro/workflows/exorcise.workflow.json) | **Autonomous Loop:** Multi-step workflow repeating *Audit $\rightarrow$ Diagnose $\rightarrow$ Fix $\rightarrow$ Verify* until `verdict.json` evaluates to `APPROVED`. |
| **Power** | [power/hexperiment/](../power/hexperiment/) | **Distribution:** Packages the hexperiment skill and the MCP server for any Kiro session (`kiro-cli powers install`). `hexperiment init --kiro` writes the hooks, workflow, agent, steering and MCP entry into another project. |

### The loop, recorded

Both demo histories in the apps' `runs/` folders were generated autonomously by the `exorcise` workflow executing on haunted scripts (recorded on 2026-11-11 UTC in Kiro CLI 3.2.0). The ghost hunter's recorded sessions are available in [kiro/exorcise-hotel-cancellations.md](kiro/exorcise-hotel-cancellations.md) and [kiro/exorcise-flight-delays.md](kiro/exorcise-flight-delays.md):

- **Hotels:** Initially `REJECTED` at ROC AUC 1.0000 (The Possession on `reservation_status`, The Peeking Spirit on a label-derived country cancellation rate). The ghost hunter dropped the proxy column, recomputed the rate strictly from training rows, and achieved `APPROVED` at an honest 0.9090 in a single round.
- **Flights:** Initially `REJECTED` at 0.7971 (The Prophet: random split across time-ordered flights). The ghost hunter converted the split to the kit's temporal split and removed a label-derived carrier rate, achieving `APPROVED` at 0.6193 in one round.

At the time of those recordings the ghost hunter ran `hexperiment run` through the shell. Live probes then showed that Kiro's shell rules are not a boundary inside a headless workflow (see below), so we took the shell away and gave it the MCP tool `run_audit` instead. A live headless `exorcise` run on a copy of the hotel app on 2026-11-13 (Kiro CLI 3.3.0) went REJECTED 1.0000 → APPROVED 0.9092 in one round, with no shell tool offered or used and nothing denied.

---

## 3. What we learned about Kiro

We probed Kiro's runtime characteristics extensively wherever behavior was undocumented or mission-critical. Detailed logs are preserved in [kiro/kiro-formats.md](kiro/kiro-formats.md):

- **Hook output routing:** In Kiro CLI 3.2.0 and Kiro IDE 1.4.154, command hooks on `PostFileSave` or `PostToolUse` execute successfully, but their stdout/stderr is not injected into agent session context, although the documentation says it is. Our write-up is in [kiro/hook-output-issue.md](kiro/hook-output-issue.md). What *does* reliably communicate with the agent in Kiro CLI 3.3.0 is a `Stop` hook exiting with code 1 and diagnostics on stderr. The agent receives the error and one more turn, and the hook fires once per prompt, so it cannot loop. That is why Hexperiment's nudge is a Stop hook.
- **Shell rules are not a boundary in a headless workflow.** Inside a workflow step the shell tool is `run_command`, and once it is trusted, a command that matches no rule runs; deny rules are case-sensitive while PowerShell is not, and a chain with `;` is judged one part at a time. So we removed the shell tool from `ghost-hunter` entirely and routed its runs through the typed `hexperiment-mcp` server. The code it writes is still executed by every audit, which no rule can prevent; the README says so.
- **File-write rules held.** Write globs were case-insensitive and treated `\` and `/` alike, and every attempt to write `hex.toml` was refused. One gap: `..` in a path is not resolved, so the ghost hunter's rules also deny any path containing `..`.
- **MCP server lifecycle in Powers:** A power's MCP server starts from its installation directory rather than the user's workspace root. `hexperiment init --kiro` configures a workspace-level MCP definition so `hexperiment-mcp` resolves local application directories seamlessly.
- **Workspace Trust:** The Kiro IDE activates hooks and custom MCP servers only after the developer explicitly trusts the workspace folder.

---

## 4. Key Metrics & Evidence

- **5 specs, 45 requirements, 75 tasks:** Kiro authored the code for every single task, verified and committed incrementally.
- **~3,400 automated tests:** Passing with exit code 0 in ~2 minutes on a 2019 4-core laptop on a power-saving profile.
- **Kapoor & Narayanan Benchmark:** 78 applications evaluated; 30 of 39 leaky applications caught as predicted; **0 false alarms** across 39 clean twins ([hexperiment/benchmark/README.md](../hexperiment/benchmark/README.md)).
