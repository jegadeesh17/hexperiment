# Hexperiment

**If your model looks too good to be true, it's haunted.**

Hexperiment is a Kiro power that catches data leakage in machine-learning training code, whether a person or an AI agent wrote it. It checks every training run, writes a verdict (`APPROVED`, `REJECTED` or `UNVERIFIED`) and a model card, and inside Kiro it sends the agent back to fix the code until the verdict is `APPROVED`.

**Gallery (the two demo cases, step by step):** <https://jegadeesh17.github.io/hexperiment/>

## The problem in plain words

A model is graded on rows it has never seen, the *test set*, the way an exam checks what a student learned. A **data leak** is when the model sees part of the answers while it learns: the test rows, data from the future, or a column that is the answer in disguise. A leaky model gets a brilliant score and then fails in real use.

Nothing in the score warns you. A leak makes the number go *up*, so whoever tunes code for a better metric, a person or an AI agent, gets no signal that anything is wrong. Leakage is a known cause of results that do not hold up: Kapoor and Narayanan found leakage errors reported in 17 fields, affecting 329 papers ([arXiv:2207.07048](https://arxiv.org/abs/2207.07048)).

## How it catches a leak: the canary test

Think of a unit test that changes one input and checks that an unrelated output stays the same.

1. Hexperiment runs your training script normally and records what training produced.
2. It runs the script again with the **test rows scrambled**. The training rows are untouched.
3. If anything training produced changes, training must have looked at the test rows. That is a leak, and the run is `REJECTED`.

The canary watches behaviour instead of reading code, so it finds a test-row leak wherever it hides: a scaler fitted on all rows, a helper function, another file. It needs one thing from the script: the data must be loaded and split through `hexperiment.load()` and `hexperiment.split()`.

Four other checks cover what the canary cannot see. Each one reports its finding as a named ghost, so a failed run reads like a story:

| Ghost | What it means |
|---|---|
| The Doppelganger | The same rows are in both training and test. |
| The Possession | One column alone gives the answer away. |
| The Peeking Spirit | Training saw the test rows (the canary test). |
| The Prophet | Training rows come from after the test period. |
| The Shapeshifter | Identical runs give different results, so nothing can be verified (a warning). |

## The two demo cases

Both apps use real public data. We planted realistic leaks in each app's training script. Kiro's `exorcise` workflow then ran the `ghost-hunter` agent, which read the verdict, fixed the code and re-ran the full check until it passed. Both histories below were recorded by that loop and are committed in each app's `runs/` folder.

| App | First run | After the fix | What was wrong |
|---|---|---|---|
| [hotel-cancellations](apps/hotel-cancellations/README.md): will a booking be cancelled? | `REJECTED`, ROC AUC **1.0000** | `APPROVED`, ROC AUC 0.9096 (threshold 0.85) | `reservation_status` is filled in when the booking ends, so it *is* the answer (The Possession). A per-country cancellation rate was computed from every row's label, test rows included (The Peeking Spirit). |
| [flight-delays](apps/flight-delays/README.md): will a 2013 New York flight arrive more than 15 minutes late? | `REJECTED`, ROC AUC 0.7971 | `APPROVED`, ROC AUC 0.6193 (threshold 0.60) | A random split on time-ordered data let the model learn from later flights (The Prophet). A second planted leak, a per-carrier daily late rate computed from every row's label, is caught by The Peeking Spirit once the split is fixed; the ghost hunter fixed both in one round. |

The honest scores are lower. That is the point: the first numbers were not real. The ghost hunter's own hotel fix scored 0.9090; the 0.9096 above is the reference script we keep in the repository. The gallery shows each case with the code diff, the findings and the model cards.

## Does it catch leaks it was not built around?

We built a benchmark of 13 leak mechanisms taken from the leakage taxonomy of Kapoor and Narayanan (*Patterns* 4(9), 100804, 2023), each with three models and a clean twin: 78 apps, run through the real `hexperiment run` command. We wrote down our predictions before the first run.

- **30 of 39 leaky apps rejected**, each by the ghost we predicted.
- **0 of 39 clean twins rejected** (no false alarms).
- **3 mechanisms missed in every model**, as predicted: a proxy column that is only moderately predictive (stand-alone AUC about 0.89, below the 0.95 cut-off), a time column that was never declared in `hex.toml`, and near-duplicate rows of the same entity split across train and test.
- A scaler or imputer fitted on all rows moved the score by 0.008 or less, so comparing scores would not have shown those leaks. The canary rejected all six.

The cases are synthetic and were written by the same people who wrote the checks, so this shows the checks see the leaks we thought of; it is not a measurement of how often they find leaks in the wild. Full table, method and the misses: [hexperiment/benchmark/README.md](hexperiment/benchmark/README.md).

## How it works inside Kiro

| Kiro feature | What Hexperiment uses it for |
|---|---|
| **Power** ([power/hexperiment/](power/hexperiment/)) | Installs the skill (the rules and how to read a verdict) and the MCP server in one step. |
| **Steering** ([.kiro/steering/ml-integrity.md](.kiro/steering/ml-integrity.md)) | Eight ML-integrity rules the agent follows in every training script: split first, no features from the future, fix seeds, never edit a verdict, never weaken a check. |
| **Hooks** ([.kiro/hooks/hexperiment.json](.kiro/hooks/hexperiment.json)) | On save of a training file: runs the fast check and clears any earlier `APPROVED`, so an old pass never covers new code. When the agent ends its turn: if an app it changed is not approved, tells it to run the full audit or to say the work is not done. |
| **MCP server** (`hexperiment-mcp`) | Tools `list_apps`, `run_audit`, `get_verdict`, `get_history`, `explain_ghost`, and each app's model card as a resource. |
| **Custom agent** ([.kiro/agents/ghost-hunter.json](.kiro/agents/ghost-hunter.json)) | The ghost hunter may edit an app's Python files and call the hexperiment tools. It has no shell, and it may not touch thresholds, verdicts, run history, data or the kit itself. |
| **Workflow** ([.kiro/workflows/exorcise.workflow.json](.kiro/workflows/exorcise.workflow.json)) | `exorcise` repeats "run the full check, fix each finding at its cause, check again" until `verdict.json` says `APPROVED`. |
| **Specs** ([.kiro/specs/](.kiro/specs/)) | The kit was built spec-first in Kiro: requirements, design and tasks for the core, the Kiro layer, the apps, the gallery and the benchmark. |

Only a full run of `hexperiment run` can write `APPROVED`. Thresholds and waivers belong to the person who owns the model, not to the agent. More on how Kiro was used: [docs/KIRO.md](docs/KIRO.md).

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
   pip install -e "./hexperiment[mcp]"
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

Run times on a 2019 four-core laptop on a power-saving plan, with everyday apps open: a full run of either demo app takes about 45 to 55 s, a fast run about 30 s, and the MCP server connects in about 13 s.

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

- **The canary needs the kit to own the data.** If a script reads its CSV itself, or splits with its own code so that scrambled test rows end up in training, the canary is not applied and the run cannot be approved: it ends `UNVERIFIED`, or `REJECTED` if another check finds a leak.
- **Leaks that do not depend on the test rows' values** (a column recorded after the outcome, a weak proxy, an undeclared time column, rows of the same entity on both sides) are left to the other checks, which catch only some of them. The benchmark measures these misses.
- **No group split.** The kit has no notion of an entity (a patient, a customer), so it cannot keep all rows of one entity on one side.
- **The training script is run as written; it is not sandboxed.** The ghost hunter has no shell, but the code it writes is executed by every run. Review agent-written code before you run it on a machine that matters.
- **Hook output does not reach the agent.** In Kiro CLI 3.2.0 and the Kiro IDE 1.4.154, the save hook runs and records its fast check, but its output is not added to the agent's context; in CLI 3.3.0 a `PostToolUse` hook's output did not reach the agent either. The Stop hook's message (exit 1) does reach the agent in Kiro CLI 3.3.0; we have not yet checked it in the IDE. The `exorcise` workflow depends on neither hook: the ghost hunter runs the full check itself.
- **Not a certificate.** `APPROVED` means the run passed these checks and reached its threshold. It does not prove the model is free of every possible leak.
- **Scope:** tabular data, scikit-learn style scripts, CPU only. No deep learning, notebooks or hyperparameter search.

## Running tests

```bash
pip install -e "./hexperiment[dev]"
python -m pytest hexperiment/tests apps/tests -q      # about 3,400 tests, about 2 minutes: many start real training runs
python -m pytest apps/tests -q                        # only the two apps: no network and no downloaded data, about 20 s
python -m pytest hexperiment/tests/test_canary.py -q   # one file
python -m ruff check hexperiment apps
```

The benchmark takes about 40 minutes: `python hexperiment/benchmark/run_benchmark.py --out hexperiment/benchmark/results.json`.

## Prior art

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
