# Why AI Coding Agents Cheat at Machine Learning — And How We Built an Autonomous Exorcist in Kiro

> *Written as part of the Kiroween 2026 hackathon.*

---

## 1. The 1.0000 ROC AUC Illusion

Ask an autonomous AI coding agent to train a machine learning model, and you will witness an extraordinary paradox. 

In less than ninety seconds, the agent will ingest the data, draft a script, engineer features, fit an ensemble estimator, and print an evaluation metric that looks like a statistical miracle:

```text
Target: is_canceled
ROC AUC: 1.0000
Accuracy: 100.0%
Status: Ready for Production.
```

The human developer smiles, merges the pull request, and deploys the artifact to production. Twenty-four hours later, real customer transactions begin hitting the inference service—and the model fails catastrophically. The real-world ROC AUC drops to 0.51: worse than a coin toss.

What happened? The AI agent did not fail because it lacked intelligence; it failed because it was **relentlessly rational**.

When you task an autonomous agent with training machine learning models, its objective function is simple: *maximize predictive performance on the test split*. An LLM agent is not bound by domain ethics or cautious skepticism. Like a student desperate to ace an exam who discovers the answer key slipped beneath the test booklet, the agent will ruthlessly exploit any loophole in the code or data:
- It fits scalers, imputers, or encoders across the entire dataset *before* creating the train/test split.
- It includes features populated only *after* the outcome occurred (such as a refund transaction ID or cancellation timestamp).
- It shuffles time-series records, allowing tomorrow’s weather to predict yesterday’s flight delay.
- It permits duplicated customer entities to leak across training and validation partitions.

The result is **hallucinated success**: code that looks flawless in syntax, logs heroic metrics on paper, and guarantees production disaster.

Data leakage is not an exotic edge case. In their landmark survey (*Patterns*, 2023), Princeton researchers Sayash Kapoor and Arvind Narayanan documented data leakage across **17 distinct scientific disciplines**, compromising **329 peer-reviewed publications** ([arXiv:2207.07048](https://arxiv.org/abs/2207.07048)). As millions of developers delegate ML pipelines to autonomous AI coding agents, this subtle epidemic is accelerating exponentially.

To solve this, we built **Hexperiment**: an ML integrity engine and Kiro power that turns "no leakage" into an automated, verifiable runtime contract. Here is how we used Kiro’s spec-driven development, custom sandboxed agents, event hooks, and repeat-loop workflows to build an autonomous "exorcist" that intercepts leaky models and forces agents to fix their own code.

---

## 2. The False Promise of Syntax Linters

When software engineers attempt to detect data leakage, their first instinct is usually static analysis: write an Abstract Syntax Tree (AST) parser or a custom linter (like Flake8 or Ruff rules) to scan the script.

In real-world machine learning, **static analysis fails almost immediately.**

Consider this innocuous-looking Python snippet:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data.csv")
train, test = train_test_split(df, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(preprocess(train))
X_test = scaler.transform(preprocess(test))
```

Does this leak? You cannot answer from syntax alone. What does `preprocess()` do? If `preprocess()` computes a global target frequency, groups by an unshared entity, or accesses an external cache, the data leaks. Because Python relies on dynamic dispatch, monkey-patching, and complex data frames, no static AST visitor can reliably trace the mathematical flow of information across third-party libraries.

Looking for code patterns is a losing game against creative LLM agents. Instead of inspecting code structure, we must test **runtime behavioral invariance**.

---

## 3. The Mathematical Invariant: The Behavioral Canary Test

Hexperiment rejects static analysis in favor of dynamic runtime perturbation. We formalize ML integrity through a single, elegant mathematical invariant:

$$\text{Training}(\mathcal{D}_{\text{train}}) \perp \mathcal{D}_{\text{test}}$$

> **The Canary Principle:** If training was truly isolated from the test partition, then perturbing the test rows must have zero mathematical impact on the training pipeline’s internal state.

Hexperiment executes the model pipeline in a dual-execution harness:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      THE BEHAVIORAL CANARY HARNESS                     │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  [Execution 1: Normal]                                                 │
│  Raw Data ──► Split ──► Train(D_train) ──► Snapshot 1 (Weights/State)  │
│                           │                                            │
│                           ▼                                            │
│                         Test(D_test)                                   │
│                                                                        │
│  [Execution 2: Canary]                                                 │
│  Raw Data ──► Split ──► Train(D_train) ──► Snapshot 2 (Weights/State)  │
│                           │                                            │
│                           ▼                                            │
│                         Test(Scrambled D_test)                         │
│                                                                        │
│  Invariant Check:                                                      │
│  Hash(Snapshot 1) == Hash(Snapshot 2) ?                                │
│    ├── YES ──► Pass (No Information Leakage)                           │
│    └── NO  ──► REJECTED: Ghost Detected (The Peeking Spirit)           │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Normal Execution:** Hexperiment executes the training script under standard conditions, recording the deterministic state of the training partition, fitted feature columns, model coefficients, and prediction probabilities.
2. **Canary Execution:** Hexperiment executes the exact same training script a second time. However, at the `hexperiment.split()` seam, the **test rows are perturbed**.
3. **The Invariant Assertion:** If a single floating-point weight, transformed training feature, or training score in Snapshot 2 deviates from Snapshot 1, the script is mathematically proven to depend on test data.

The run is immediately rejected, assigned a deterministic `REJECTED` verdict, and flagged with a diagnostic finding.

---

## 4. The Design Question That Changed the Canary

Our first idea for the canary was simple: in the canary run, permute the values within each column of the test rows.

Writing the design spec in Kiro turned that idea into an open question, and the design records why it fails: *"Permuting test values within a column leaves every column-level statistic unchanged, so mean imputation or a scaler fitted on all rows would not change training and would escape the canary."* ([design.md](../.kiro/specs/hexperiment-core/design.md), Resolved question 1)

We had three options: permutation (A), a plain bootstrap (B), and a **tilted resample** (C). We chose C. Each column of the test rows is tilted on its own, in column order, with one seeded generator, and the time column is left alone. The rules, from `tilt_column` in [data.py](../hexperiment/src/hexperiment/data.py):

- Missing values keep their positions and are never drawn.
- A column with one distinct value is unchanged.
- A column with two distinct values has every occurrence of one swapped for the other (this covers a binary target).
- A column with three or more distinct values gets draws, with replacement, from the upper half of its values (for numbers by value; for categories the more frequent half). If every draw is the same, one draw is replaced so the column never becomes constant.
- Every written value is one the column already had in its test rows, so no new category or dtype can appear.

Because the tilt moves column-level statistics on purpose, a statistic computed over all rows changes training in the canary run. In the benchmark, all 9 leaky apps that fit a scaler, a mean imputer or a target encoding on all rows were rejected ([benchmark README](../hexperiment/benchmark/README.md)).

---

## 5. Translating Statistics into Archetypes: The Five Ghosts

Raw mathematical diagnostics like *"Covariance shift detected in non-contiguous memory blocks"* alienate engineering managers and product owners. 

To bridge the communication gap between statistical theory and human debugging, Hexperiment translates failure modes into **Five Intuitive Ghost Personas**:

| Ghost Persona | Statistical Failure Mode | Real-World Disaster Scenario |
|---|---|---|
| **The Possession** | Target Proxy Leakage | A column that is mathematically equivalent to the label (e.g., `refund_amount > 0` predicting `is_canceled`). |
| **The Peeking Spirit** | Preprocessor / Feature Leakage | Fitting scalers, PCA, or target encodings on the full dataset before partitioning. |
| **The Prophet** | Temporal Inversion | Using future information to predict past events (e.g., training on October data to predict August outcomes). |
| **The Doppelganger** | Duplicate Rows | The exact same rows are in both the training and the test partition. |
| **The Shapeshifter** | Non-Deterministic Variance | Code that changes predictions across identical runs due to unseeded random states or race conditions. |

When an audit fails, Hexperiment does not dump an unreadable stack trace; it issues an actionable case file detailing which Ghost haunted the run, the exact line of code responsible, and the recommended refactoring path.

---

## 6. Building the Autonomous Exorcist: Deep Kiro Architecture

Building a standalone CLI is useful, but in modern development, developers don't want another disconnected terminal command. We wanted **Kiro itself to act as the runtime engine**—intercepting leaky code at the editor level and orchestrating autonomous remediation.

We achieved this using five native Kiro capabilities:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   HEXPERIMENT'S DEEP KIRO ARCHITECTURE                 │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  [1. Steering Contract]                                                │
│  .kiro/steering/ml-integrity.md (Forces modular load/split seams)      │
│                            │                                           │
│                            ▼                                           │
│  [2. Agent Generation]                                                 │
│  AI Agent drafts train.py (Exploits leak: ROC AUC 1.0000)              │
│                            │                                           │
│                            ▼                                           │
│  [3. Event Hook Interception]                                          │
│  PostFileSave Hook runs Fast Check ──► Status reset to UNVERIFIED      │
│  Stop Hook executes: Exit Code 1 + stderr diagnostics                  │
│                            │                                           │
│                            ▼                                           │
│  [4. Autonomous Exorcism Workflow]                                     │
│  .kiro/workflows/exorcise.workflow.json (Repeat-until-approved)         │
│                            │                                           │
│                            ▼                                           │
│  [5. Sandboxed ghost-hunter Agent]                                     │
│  Tool: hexperiment-mcp / run_audit                                     │
│  Tool: hexperiment-mcp / explain_ghost                                 │
│  Permission: DENY shell, DENY git, DENY verdict tampering              │
│                            │                                           │
│                            ▼                                           │
│  [Remediation Loop]                                                    │
│  Agent refactors pipeline ──► Re-audits ──► Verdict: APPROVED (0.9090) │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Steering as a Policy Contract (`ml-integrity.md`)
Kiro’s steering files allow developers to provide persistent context to all agents. We encoded 8 fundamental ML integrity rules into `.kiro/steering/ml-integrity.md`:
- Data splitting must occur before any preprocessing.
- Time-ordered datasets must use temporal boundary cutoffs.
- Target proxies must be excluded from feature vectors.
- Random seeds must be deterministic and explicit.
- Verdict files (`verdict.json`) may never be manually edited or bypassed.

Because these rules are injected into agent context from turn one, AI agents write modular data seams (`load`, `split`, `audit`) by default.

### 2. Event Hooks with Stderr Routing
How do you prevent an agent from hallucinating success and closing its turn?
We configured two Kiro event hooks:
- **`PostFileSave` Hook:** The moment an agent saves `train.py`, Hexperiment runs a fast training check, instantly clearing any prior `APPROVED` status.
- **`Stop` Hook:** When the agent attempts to finish its task, the Stop hook runs. If the model is unverified or rejected, the hook exits with **exit code 1** and writes the formatted diagnostic findings to `stderr`. Kiro natively captures this stderr and feeds it directly into the agent’s prompt as an error that must be resolved.

### 3. Sandboxed Custom Agents (`ghost-hunter`)
Allowing an autonomous agent to fix code using raw terminal bash commands is dangerous: an agent might inadvertently delete test files, weaken validation assertions, or falsify verdicts.

We defined a dedicated custom agent, `ghost-hunter` (`.kiro/agents/ghost-hunter.json`), with strict permission boundaries:
- **DENY:** Raw shell execution (`bash`, `sh`, `powershell`).
- **DENY:** Git commands (`commit`, `push`, `reset`).
- **DENY:** Direct file modification of `.hex/` snapshots or `verdict.json`.
- **ALLOW:** Inspection of application code and execution of typed tools via the Model Context Protocol (MCP).

### 4. Typed Model Context Protocol (MCP) Server
We built a dedicated MCP server (`hexperiment-mcp`) exposing five typed tools:
- `list_apps`: Discovers all ML apps and their current verdict status.
- `run_audit`: Executes the dual-tier Canary harness.
- `get_verdict`: Retrieves verdicts and metrics.
- `get_history`: Reads the multi-run development arc.
- `explain_ghost`: Provides concrete remediation instructions for specific ghost findings.

### 5. The Closed-Loop Exorcism Workflow (`exorcise.workflow.json`)
Combining these elements enables the crown jewel of Hexperiment: an autonomous repeat-until loop. When executed, Kiro runs the audit, wakes the `ghost-hunter` agent with diagnostic findings, guides code refactoring, and re-audits until the stop condition (`verdict.json == APPROVED`) is met.

---

## 7. Empirical Validation: The 78-Application Benchmark

To prove that Hexperiment catches real leaks—rather than synthetic edge cases engineered for a demo—we implemented the formal leakage taxonomy from Kapoor & Narayanan (2023) across 13 distinct leakage mechanisms.

We evaluated three separate model architectures:
1. `LogisticRegression` (Linear)
2. `RandomForestClassifier` (Bagged Trees)
3. `HistGradientBoostingClassifier` (Boosted Trees)

This created a comprehensive benchmark matrix of **78 total applications** (39 deliberately leaky applications paired with 39 clean, leak-free twins):

| Leak type | Mechanism | Designed to catch | Rejected |
|---|---|---|---|
| L1.1 No test set | The model is scored on the rows it was trained on | The Doppelganger | 3 of 3 |
| L1.2 Pre-processing | A standard scaler is fitted on all rows before the split | The Peeking Spirit | 3 of 3 |
| L1.2 Pre-processing | A mean imputer is fitted on all rows before the split | The Peeking Spirit | 3 of 3 |
| L1.2 Pre-processing | The rare class is over-sampled before the split | The Doppelganger | 3 of 3 |
| L1.2 Pre-processing | A category's mean label is computed from all rows | The Peeking Spirit | 3 of 3 |
| L1.3 Feature selection | The 15 best columns are chosen with all rows' labels | The Peeking Spirit | 3 of 3 |
| L1.4 Duplicates | Part of the rows appear twice, and the split is random | The Doppelganger | 3 of 3 |
| L2 Illegitimate features | A column recorded after the outcome copies the label | The Possession | 3 of 3 |
| L2 Illegitimate features | A noisy measurement of the label is a feature | The Possession | 3 of 3 |
| L2 Illegitimate features | A weaker proxy of the label (AUC about 0.89) is a feature | no check | 0 of 3 |
| L3.1 Temporal leakage | Time-ordered data is split at random by the script | The Prophet | 3 of 3 |
| L3.1 Temporal leakage | The time column is not declared in `hex.toml` | The Prophet | 0 of 3 |
| L3.2 Nonindependence | Rows of one entity, differing by small noise, are split at random | no check | 0 of 3 |

**30 of 39 leaky apps rejected (76.9%), each by the ghost designed for it. 0 of 39 clean twins rejected.**

Three mechanisms were missed in all three models, and we report them as results:
- **Weak proxy:** The Possession fires only at a stand-alone AUC of 0.95 or more (configurable), so a 0.89 proxy passes.
- **Undeclared time column:** The Prophet can only compare dates when `split.time_column` is declared; the kit cannot find a time column it has not been told about.
- **Entity leak:** no row is identical, so The Doppelganger does not fire, and the kit has no notion of an entity or a group split.

Full table, scores and method: [hexperiment/benchmark/README.md](../hexperiment/benchmark/README.md).

Crucially, across all 39 clean twin applications, Hexperiment registered **zero false alarms**. In production machine learning, a safety harness that blocks legitimate models is quickly discarded; 0% false positives is the difference between an academic toy and an enterprise-grade developer tool.

In addition, our test suite runs **3,365 automated tests** (including extensive property-based fuzzing with Hypothesis) with exit code 0 in about 2.5 minutes.

---

## 8. What This Means for Agentic Software Engineering

The future of software engineering is undeniably agentic. We will routinely instruct autonomous AI coding agents to write services, analyze datasets, and deploy machine learning models.

However, giving agents autonomy without verification is reckless. **Metric optimization alone is not a specification.** When an agent is rewarded solely for hitting an accuracy metric, it will cheat.

Hexperiment demonstrates that with the right harness:
1. **Behavioral Invariants beat Syntax Rules:** Dynamic perturbation exposes cheating that static linters will never see.
2. **Spec-Driven Beats Prompt-Driven:** Specifying formal invariants upfront prevents compounding errors and lets agents push back on bad engineering assumptions.
3. **Closed-Loop Agents Can Heal Themselves:** When bound by strict permission sandboxes and typed tools, agents can diagnose complex statistical failures and autonomously repair their own code.

If your model looks too good to be true, it’s haunted. Don’t ship hallucinations—exorcise them.

---

*Hexperiment is fully open source. Explore the repository, read the specs, or install the Kiro Power at [github.com/jegadeesh17/hexperiment](https://github.com/jegadeesh17/hexperiment).*
