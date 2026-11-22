# Hexperiment

**"If your model looks too good to be true, it's haunted."**

Hexperiment catches data leakage in agent-written training code before a result counts. Built by Skeleton Crew for data science teams adopting autonomous coding agents (one kit, two demo apps).

Hexperiment is an ML-integrity kit and Kiro power that checks machine-learning training code for data leakage and flags suspiciously good scores. It checks every training run, issues a verdict (`APPROVED`, `REJECTED`, or `UNVERIFIED`) and an automated model card, and inside Kiro runs a workflow that sends the coding agent back to fix its own leaks until the verdict is `APPROVED` or five rounds have passed.

**Gallery (the two demo cases, step by step):** <https://jegadeesh17.github.io/hexperiment/>

## The problem in plain words

When asked to maximize accuracy, an autonomous AI coding agent behaves like a student looking for shortcuts on an exam: it optimizes the loss function by exploiting whatever data is in its environment. If an agent can peek at the test set, fit a scaler before splitting, or use an ID column that encodes future outcomes, it will do so.

The result is **an illusion of success**: a model that boasts a 1.0000 ROC AUC on paper, but fails on new data in production.

Nothing in the reported metric warns you. A leak makes the score go *up*, misleading developers and managers into a false sense of security. Data leakage is not an isolated bug—Kapoor and Narayanan (*Patterns*, 2023) cataloged leakage errors across 17 distinct scientific disciplines, affecting 294–329 papers (329 in the 2022 preprint, 294 in the published version; [arXiv:2207.07048](https://arxiv.org/abs/2207.07048)).

## How it catches a leak: the canary test

Traditional linters inspect AST syntax, which cannot trace dynamic runtime data flow. Hexperiment takes a behavioral approach:

1. Hexperiment runs your training script normally and records what training produced.
2. It runs the script a second time with the **test rows scrambled** while keeping training rows identical.
3. **The Behavioral Invariant:** If anything training produced changes between the two runs, training mathematically depended on the test rows. That is a leak, and the run is immediately `REJECTED`.

Because the Canary monitors runtime behavior rather than inspecting syntax, it catches test-row leakage that changes training even inside a third-party library, an upstream preprocessor, or a nested helper function, as long as the data comes from `hexperiment.load()` (see Known limits).

### Layered Verification: Solving the Compute Scale Question

A common concern in production ML is compute cost: *Does running training twice double GPU training expenses?* Hexperiment solves this with a **Two-Tier Verification Model**:

| Tier | When It Runs | Verification Mechanism | Compute Overhead | Verdict Impact |
|---|---|---|---|---|
| **Tier 1: Fast Check** | On every file save (Kiro save hook) | Column invariants, proxy correlation scans, split boundary sanity | One normal training run (about 30 s on the demo apps) | Fast feedback; clears prior approval, cannot issue `APPROVED` |
| **Tier 2: Canary Audit** | `hexperiment run`, MCP `run_audit`, or the Kiro `exorcise` workflow | Full behavioral canary test with scrambled test rows | 2 training runs (a third when they differ) | Required to write `APPROVED` and generate official model card |

**Not built yet, ideas for large models:** for deep learning or multi-gigabyte data, where retraining twice is expensive, two possible extensions are a *subsampled canary* (scramble a stratified 5% probe instead of the whole test set) and *pipeline hash auditing* (check the preprocessing graph without retraining the model). Neither exists today; the kit runs CPU-only tabular scripts (see Known limits).

### Zero-Friction Seams via Kiro Steering

The Canary test requires one architectural condition: data must be loaded and partitioned through `hexperiment.load()` and `hexperiment.split()`. 

Rather than imposing manual developer friction, Hexperiment leverages **Kiro Steering** (`.kiro/steering/ml-integrity.md`). When working in Kiro, the AI agent is instructed by steering contracts to automatically generate code adhering to these clean seams from line 1.

### The Five Diagnostic Ghosts

When a run is rejected, Hexperiment translates complex statistical anomalies into five intuitive ghost personas so developers and non-ML managers immediately understand what went wrong:

| Ghost | Statistical Failure Mode | Plain-English Meaning |
|---|---|---|
| **The Possession** | Target Proxy Leakage ($AUC \ge 0.95$) | One column alone gives the answer away (e.g., recorded post-outcome). |
| **The Peeking Spirit** | Pre-processing / Test Leakage | Training calculations saw test rows (detected by the Canary test). |
| **The Prophet** | Temporal Leakage | Training samples originate from a time period after the test period. |
| **The Doppelganger** | Duplicate Rows | The exact same rows are in both training and test. |
| **The Shapeshifter** | Non-Deterministic Instability | Identical runs produce divergent results, meaning the model is unverified. |

## The two demo cases

Both apps use real public data. We planted realistic leaks in each app's training script. Kiro's `exorcise` workflow then ran the `ghost-hunter` agent, which read the verdict, autonomously repaired the code, and re-ran the full check until it passed. Both histories below were recorded by that loop and are committed in each app's `runs/` folder.

| App | First run | After the fix | What was wrong |
|---|---|---|---|
| [hotel-cancellations](apps/hotel-cancellations/README.md): will a booking be cancelled? | `REJECTED`, ROC AUC **1.0000** | `APPROVED`, ROC AUC **0.9096** (threshold 0.85) | `reservation_status` is filled in when the booking ends, so it *is* the answer (The Possession). A per-country cancellation rate was computed from every row's label, test rows included (The Peeking Spirit). |
| [flight-delays](apps/flight-delays/README.md): will a 2013 New York flight arrive >15 min late? | `REJECTED`, ROC AUC **0.7971** | `APPROVED`, ROC AUC **0.6193** (threshold 0.60) | A random split on time-ordered data let the model learn from later flights (The Prophet). A second planted leak (carrier daily late rate computed from all labels) was not flagged, because the first run split with its own code and the canary was not applied; the ghost hunter removed that feature anyway while fixing the split, so the history never shows The Peeking Spirit. |

**Why the honest score is lower:** In real ML, catching a leak drops the metric from an inflated, invalid score to a realistic, deployable one. A model showing 0.79 with temporal leakage will fail in production; a model scoring 0.61 on clean out-of-time data actually works. The gallery shows each case with the code diff, the diagnostic findings, and the generated model cards. The ghost hunter's own hotel fix scored 0.9090; the 0.9096 in the table is the reference script we keep in the repository.

## Scientific Benchmark: 13 Leakage Mechanisms (Kapoor & Narayanan 2023)

To evaluate how well Hexperiment catches leakage across diverse patterns, we built an automated benchmark implementing the formal leakage taxonomy from Kapoor & Narayanan (*Patterns* 4(9), 100804, 2023). We evaluated 13 mechanisms across 3 distinct model architectures (`LogisticRegression`, `RandomForest`, `HistGradientBoosting`), creating 78 total applications with clean twins:

- **30 of 39 leaky applications rejected**, exactly matching predicted ghost assignments.
- **0 of 39 clean twins rejected** (**0% false alarm rate**).
- **A scaler or imputer fitted on all rows moved ROC AUC by $\le 0.008$**—subtle enough that score-comparison alone would never catch it. The Canary rejected all 6 instances.
- **Calibrated Scientific Boundaries (The 3 Predicted Misses):**
  1. *Weak proxy features* (stand-alone AUC ~0.89, below the 0.95 cutoff).
  2. *Undeclared time columns* (where no temporal metadata was provided to `hex.toml`).
  3. *Entity-level leakage* (near-duplicate rows of the same entity, such as a customer or patient, split across train and test; the kit has no notion of an entity).

Full methodology, benchmark runner, and results matrix: [hexperiment/benchmark/README.md](hexperiment/benchmark/README.md).

## How it works inside Kiro

Hexperiment transforms Kiro from a passive code-generation tool into an active, self-correcting ML integrity engine:

```
KIRO CLOSED-LOOP EXORCISM CYCLE

Steering contract (.kiro/steering/ml-integrity.md)
   │
   ▼
Coding agent writes train.py
   │
   └─► on save: hexperiment-check-on-save hook
         fast run (train.py once, no canary); clears any prior APPROVED
         side branch only: it does not start the workflow

You start the loop:  /workflow run exorcise --app <app>
   │
   ▼
exorcise.workflow.json (repeats until verdict.json says APPROVED)
   │
   ▼
ghost-hunter agent (no shell; calls MCP run_audit)
   │
   ▼
Verdict REJECTED (ghost identified) ──► agent fixes the code, audits again
   │
   ▼
Verdict APPROVED ─────────────────────► model card written
```

| Kiro feature | What Hexperiment uses it for |
|---|---|
| **Power** ([power/hexperiment/](power/hexperiment/)) | Installs the skill (rules, diagnostics, verdict handling) and MCP server in one step. |
| **Steering** ([.kiro/steering/ml-integrity.md](.kiro/steering/ml-integrity.md)) | Enforces eight ML-integrity rules: split first, no future features, fix seeds, never tamper with verdicts, never weaken checks. |
| **Hooks** ([.kiro/hooks/hexperiment.json](.kiro/hooks/hexperiment.json)) | **On save:** runs fast check and clears prior approvals so old passes never mask new code. **On stop:** intercepts incomplete work if an edited app lacks approval. |
| **MCP Server** (`hexperiment-mcp`) | Exposes tools `list_apps`, `run_audit`, `get_verdict`, `get_history`, `explain_ghost`, and model cards as live resources. |
| **Custom Agent** ([.kiro/agents/ghost-hunter.json](.kiro/agents/ghost-hunter.json)) | Dedicated repair agent with permission-restricted boundaries (no shell; executed code is not sandboxed): edits Python files inside the app folder and invokes MCP tools. A `preToolUse` hook runs `hexperiment guard-write` before every file write and refuses anything but the app's own Python files, so data, thresholds and verdict files stay out of reach ([write probe](docs/kiro/ghost-hunter-write-probe.md)). |
| **Workflow** ([.kiro/workflows/exorcise.workflow.json](.kiro/workflows/exorcise.workflow.json)) | Autonomous loop executing *Audit $\rightarrow$ Diagnose $\rightarrow$ Refactor $\rightarrow$ Verify* until `verdict.json` reports `APPROVED` (capped at 5 rounds before pausing). |
| **Specs** ([.kiro/specs/](.kiro/specs/)) | Built spec-first in Kiro: requirements (EARS), design, and task verification across core, Kiro layer, apps, gallery, and benchmark. |

More details on the Kiro implementation and lessons learned: [docs/KIRO.md](docs/KIRO.md).

## Quick start

You need Python 3.13 and git. The Kiro steps also need [Kiro](https://kiro.dev) (IDE or CLI).

1. Get the code and install the kit into a virtual environment:

   ```bash
   git clone https://github.com/jegadeesh17/hexperiment
   cd hexperiment
   python -m venv .venv
   source .venv/Scripts/activate      # Windows, Git Bash
   # .venv\Scripts\Activate.ps1       # Windows, PowerShell
   # source .venv/bin/activate        # macOS and Linux
   pip install -c hexperiment/constraints.txt -e "./hexperiment[mcp]"
   ```

2. Download the hotel data (about 17 MB; the script checks its checksum) and run the full check. It trains the model twice, once with the test rows scrambled, and prints the verdict:

   ```bash
   python apps/hotel-cancellations/fetch_data.py
   hexperiment run apps/hotel-cancellations
   ```

   Expected: `APPROVED (full)  roc_auc 0.9096 >= 0.8500`, exit code 0. The verdict is in `apps/hotel-cancellations/verdict.json` and the model card in the newest folder under `apps/hotel-cancellations/runs/`.

3. See the leak get caught. Put the haunted script in place and run the check again:

   ```bash
   cp apps/hotel-cancellations/haunted/train.py apps/hotel-cancellations/train.py
   hexperiment run apps/hotel-cancellations
   ```

   Expected: `REJECTED (full)  roc_auc 1.0000`, with The Possession on `reservation_status` and The Peeking Spirit, exit code 1. `git checkout -- apps/hotel-cancellations/train.py` puts the honest script back.

4. Let Kiro fix it. Start Kiro **from the same activated shell**, so the hook and the MCP server can find the `hexperiment` and `hexperiment-mcp` commands, and trust the folder when Kiro asks (Kiro runs a project's hooks and MCP servers only in a trusted folder). With the haunted script in place, switch workflows on and start the loop:

   ```bash
   kiro-cli settings chat.enableWorkflows true
   kiro-cli chat
   ```

   Then, in the chat: `/workflow run exorcise --app apps/hotel-cancellations`. In the Kiro IDE, switch on Workflows in the workspace configuration and start `exorcise` from the chat. The workflow ends when the full check says `APPROVED`. Every run adds a folder to `runs/`.

To use Hexperiment in your own project instead:

```bash
pip install "hexperiment[mcp] @ git+https://github.com/jegadeesh17/hexperiment#subdirectory=hexperiment"
hexperiment init apps/<name>      # a commented hex.toml and a train.py that follows the rules
hexperiment init --kiro           # the hooks, the exorcise workflow, the ghost hunter, steering and the MCP entry
kiro-cli powers install <path to a clone>/power/hexperiment   # optional: the skill, for any Kiro session
```

Run times depend on CPU power plan and background load: on a 2019 four-core laptop on a power-saving plan with everyday apps open, a full run takes about 45 to 55 s, a fast run about 30 s, and the MCP server connects in about 13 s; on higher power plans or newer hardware, full runs complete in 15 to 25 s.

## Usage

```text
hexperiment run <app>            full check with the canary test; the only way to get APPROVED
hexperiment run --fast <app>     quick feedback without the canary; never APPROVED
hexperiment init <app>           create a new app from the templates
hexperiment init --kiro          add the Kiro files to the current project
hexperiment gallery --out <dir> <app> [<app> ...] [--benchmark <results.json>]
                                 build the static gallery site from the recorded runs
```

Exit codes of `hexperiment run`: `0` approved, `1` rejected, `2` the run failed and there is no verdict, `3` unverified.

An app is a folder with a `hex.toml` (target, metric, threshold, split), a `train.py` that loads and splits with `hexperiment.load()` and `hexperiment.split()` and ends with one call to `hexperiment.audit(...)`, and a `runs/` history written by the kit. See [the hotel app](apps/hotel-cancellations/) for a full example.

## Known limits

- **The canary needs the kit to own the data.** If a script reads its CSV itself, or splits with its own code so that scrambled test rows end up in training, the canary is not applied and the run cannot be approved: it ends `UNVERIFIED`, or `REJECTED` if another check finds a leak. The kit checks this column by column: when the target, or any audited feature with at least two distinct test values, is identical after the scramble, the canary is not applied. A feature computed only from the time column is reported the same way, because the scramble never changes it.
- **Leaks that do not depend on the test rows' values** (a column recorded after the outcome, a weak proxy, an undeclared time column, rows of the same entity on both sides) are left to the other checks, which catch only some of them. The benchmark measures these misses.
- **Save hook misses helper modules outside the app folder:** the on-save hook inspects the app directory and does not re-audit when external library helpers outside the app folder are modified.
- **Undeclared time columns:** the hotel cancellations demo script uses a random split on date-ordered data without a `time_column` declared in `hex.toml`. Because the kit does not guess time columns without explicit configuration, this undeclared-temporal case is not flagged by the temporal check (captured in the benchmark as L3.1).
- **No group split.** The kit has no notion of an entity (a patient, a customer), so it cannot keep all rows of one entity on one side.
- **The training script is run as written; it is not sandboxed.** The ghost hunter has no shell, but the code it writes is executed by every run. Review agent-written code before you run it on a machine that matters.
- **Hook output does not reach the agent.** In Kiro CLI 3.2.0 and the Kiro IDE 1.4.154, the save hook runs and records its fast check, but its output is not added to the agent's context; in CLI 3.3.0 a `PostToolUse` hook's output did not reach the agent either. The Stop hook's message (exit 1) does reach the agent in Kiro CLI 3.3.0; we have not yet checked it in the IDE. The `exorcise` workflow depends on neither hook: the ghost hunter runs the full check itself.
- **The write guard does not know the workflow's app.** It allows Python files in any app folder under `apps/`, not only the one being repaired, because the hook input does not name the app.
- **Leftover `HEX_MODE`:** if `HEX_MODE=canary` is set in the environment, standalone runs may suppress normal stdout output.
- **Not a certificate.** `APPROVED` means the run passed these checks and reached its threshold. It does not prove the model is free of every possible leak.
- **Scope:** tabular data, scikit-learn style scripts, CPU only. No deep learning, notebooks or hyperparameter search.

## Running tests

```bash
pip install -c hexperiment/constraints.txt -e "./hexperiment[dev]"
python -m pytest hexperiment/tests apps/tests -q      # about 3,400 tests, about 2 minutes: many start real training runs
python -m pytest apps/tests -q                        # only the two apps: no network and no downloaded data, about 20 s
python -m pytest hexperiment/tests/test_canary.py -q   # one file
python -m ruff check hexperiment apps
```

The benchmark takes about 40 minutes: `python hexperiment/benchmark/run_benchmark.py --out hexperiment/benchmark/results.json`.

## Prior art

- waldean, "Runtime Data Invariant Checks for ML Pipelines" (2026): runtime invariant checks for data pipelines.
- Yang, Brower-Sinning, Lewis and Kästner, "Data Leakage in Notebooks: Static Detection and Better Processes" ([arXiv:2209.03345](https://arxiv.org/abs/2209.03345)): a static analysis that detects common forms of leakage in data-science notebooks.
- AlOmar et al., "LeakageDetector" ([arXiv:2503.14723](https://arxiv.org/abs/2503.14723)): a PyCharm plugin that identifies leakage in ML code and suggests fixes.

Both read code that a person has written. Hexperiment differs in two ways: the canary test checks behaviour by running the script again, so it does not depend on recognising a code pattern, and the check is a gate inside the agent's workflow in Kiro, so a leaky result does not count as done.

## Project structure

```text
hexperiment/        the kit: Python package, tests and the leak benchmark
apps/               two example apps (hotel-cancellations, flight-delays) with their recorded runs
power/hexperiment/  the Kiro power: plugin.json, mcp.json and the skill
.kiro/              specs, steering, hooks, agents, the exorcise workflow and MCP settings
docs/               documentation index, Kiro write-up and Kiro notes
.github/workflows/  builds the gallery and publishes it to GitHub Pages
```

## Documentation

- [docs/README.md](docs/README.md): where to find what
- [docs/KIRO.md](docs/KIRO.md): how Kiro was used to build and run Hexperiment
- [docs/DECISIONS.md](docs/DECISIONS.md): decisions and why
- [hexperiment/benchmark/README.md](hexperiment/benchmark/README.md): the leak benchmark

## How this was built

Designed by Jegadeesh. Kiro wrote the code from the specs in [.kiro/specs/](.kiro/specs/), using the custom agents in [.kiro/agents/](.kiro/agents/) (a spec writer, a builder limited to the kit, apps and power folders, and a read-only reviewer); [docs/KIRO.md](docs/KIRO.md) describes how. Scoping, the specs' requirements and steering, code review and verification of every task were done with Claude Code and [SoftwareEngineeringAgents](https://github.com/jegadeesh17/SoftwareEngineeringAgents), a multi-agent workflow for Claude Code. I set the scope, the architecture and the key decisions (see [docs/DECISIONS.md](docs/DECISIONS.md)); the agents implemented and reviewed tasks under that plan.

## License

[MIT](LICENSE). The example apps download public datasets with their own licences, listed in each app's README.
