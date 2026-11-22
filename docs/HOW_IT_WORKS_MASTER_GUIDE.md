# Hexperiment Master Guide: How It Actually Works

> **Purpose:** This document is your complete, end-to-end learning manual. Read this to deeply understand the architecture, mathematics, Kiro platform integration, demo apps, benchmark proof, and judge Q&A defenses for **Hexperiment**.

---

## Table of Contents
1. [The Big Picture: Why Hexperiment Exists](#1-the-big-picture-why-hexperiment-exists)
2. [What Is Data Leakage? (The Math in Plain English)](#2-what-is-data-leakage-the-math-in-plain-english)
3. [The Canary Test: How Hexperiment Catches Leaks](#3-the-canary-test-how-hexperiment-catches-leaks)
4. [The Five Diagnostic Ghosts Explained](#4-the-five-diagnostic-ghosts-explained)
5. [How Kiro Powers the System (The Native Primitives)](#5-how-kiro-powers-the-system-the-native-primitives)
6. [Deep Dive: The Two Demo Applications](#6-deep-dive-the-two-demo-applications)
7. [The Scientific Benchmark: 78 Apps, 0 False Alarms](#7-the-scientific-benchmark-78-apps-0-false-alarms)
8. [Judge Defense & Q&A Cheat Sheet](#8-judge-defense--qa-cheat-sheet)

---

## 1. The Big Picture: Why Hexperiment Exists

### The Rise of Autonomous Coding Agents
Developers are increasingly using AI coding agents (Kiro, Cursor, Copilot) to build machine learning pipelines. You give the agent a dataset, a problem statement (e.g., "predict flight delays"), and a target metric (e.g., "maximize ROC AUC").

### The Fatal Vulnerability: AI Agents Cheat
AI agents are optimization machines. When tasked with maximizing a score, an agent takes every mathematical shortcut available in the codebase to make the number go higher:
- It fits scalers on the entire dataset before splitting.
- It calculates group target-encoding rates using labels from test rows.
- It uses columns recorded *after* the event occurs.
- It randomly splits time-series data so the model can predict yesterday by looking at tomorrow.

### The Illusion of Success
The script completes, and the AI agent proudly reports: `ROC AUC: 1.0000` or `Accuracy: 99.8%`. 
The developer is thrilled. But the moment this model goes to live production, **it completely fails**, costing companies millions in bad decisions, credit defaults, or medical misdiagnoses.

**Hexperiment's Mission:** To provide an automated, behavior-based safety harness for Kiro that makes "no data leakage" a non-negotiable contract before an AI agent's code can count as done.

---

## 2. What Is Data Leakage? (The Math in Plain English)

Machine learning models are evaluated on a **test set** (unseen data) to measure how well they generalize to the real world. 
A **data leak** occurs whenever information from the test set or from the future bleeds into the training process.

There are four primary forms of leakage:

### A. Preprocessing Leakage (The "Peeking" Leak)
- **What happens:** You have 10,000 rows. You calculate the average (mean) or variance across all 10,000 rows to scale the data, and *then* you split into 8,000 train rows and 2,000 test rows.
- **Why it’s a leak:** The 8,000 training rows now mathematically incorporate information about the 2,000 test rows. In production, you will never know the future test distribution in advance.

### B. Target Proxy Leakage (The "Answer in Disguise" Leak)
- **What happens:** You build a model to predict whether a hotel booking will be canceled. The dataset contains a column called `reservation_status` (values: "Check-Out", "Canceled", "No-Show"). 
- **Why it’s a leak:** `reservation_status` is filled in by hotel staff *at the end of the stay*. Feeding it to the model is like giving a student the answer key during the exam. The model gets 100% accuracy, but in production, when a guest makes a new reservation, that column doesn't exist yet!

### C. Temporal Leakage (The "Time Machine" Leak)
- **What happens:** You have stock prices or flight delays from January through December. You perform a random 80/20 train/test split.
- **Why it’s a leak:** The model trains on December flights to predict July flights. In real life, you cannot use tomorrow's weather to predict yesterday's flight. Real validation must be strictly split by time (train on Jan–Oct, test on Nov–Dec).

### D. Entity / Overlap Leakage (The "Doppelganger" Leak)
- **What happens:** You have multiple medical records for the same patient across different days. You randomly split rows so Patient #42's records appear in both training and test sets.
- **Why it’s a leak:** The model memorizes Patient #42's unique baseline biometrics rather than learning general disease indicators.

---

## 3. The Canary Test: How Hexperiment Catches Leaks

### Why Traditional Code Linters Fail
Traditional tools parse code into an Abstract Syntax Tree (AST) looking for patterns like `train_test_split()`. But AI agents write creative code: they might write custom Python loops, use Polars, call third-party helper modules, or write lambdas. Static linters cannot trace dynamic data values across complex pipelines.

### The Behavioral Canary Invariant
Instead of reading code, Hexperiment tests **runtime behavior**. Think of it like a unit test that verifies mathematical independence:

```
[Normal Run]
Load Data ──► Split Data ──► Train Model on Train Set ──► Evaluate on Test Set ──► Record Outputs (Weights/Predictions)

[Canary Run]
Load Data ──► Split Data ──► TEST ROWS SCRAMBLED ──► Train Model on Train Set ──► Record Outputs
                                │
                                └── Training rows are UNTOUCHED!
```

### The Core Invariant
* **If training is clean:** Scrambling the test rows has **zero mathematical effect** on the training set. The model's weights and training predictions will be identical to 6 decimal places.
* **If training leaked:** If training looked at test rows (even indirectly through a mean scaler or target encoding), changing the test rows changes the training inputs. The model's weights will shift.
* **Verdict:** Any difference in training output between the normal run and the scrambled run proves a leak. The run is immediately `REJECTED`.

---

## 4. The Five Diagnostic Ghosts Explained

When a leak is rejected, Hexperiment doesn't output an opaque stack trace. It diagnoses the mathematical cause and reports it as one of **Five Ghosts**:

| Ghost Name | Scientific Term | What Happened Under the Hood | How It Is Caught |
|---|---|---|---|
| **The Possession** | Target Proxy Leakage | A feature predicts the target with $AUC > 0.95$ on its own. It's the label in disguise. | Single-feature ROC AUC scan during audit. |
| **The Peeking Spirit** | Pre-processing / Test Leakage | Upstream preprocessing (scaler, imputer, target rate) saw test rows. | The Canary Test (scrambling test rows moved training weights). |
| **The Prophet** | Temporal Leakage | Training samples are timestamped after test samples. | Timestamp comparison across train and test partitions. |
| **The Doppelganger** | Duplicate Overlap | The exact same data row appears in both train and test splits. | Row-hash hashing comparison between train and test. |
| **The Shapeshifter** | Non-Deterministic Instability | Identical runs produce different metrics (unfixed seeds). | Re-running the baseline script produces inconsistent results. |

---

## 5. How Kiro Powers the System (The Native Primitives)

Hexperiment is not just a Python script; it is a **Kiro Power** that transforms Kiro into a closed-loop self-healing engine. Here is how each Kiro feature works:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   THE KIRO EXORCISM ARCHITECTURE                       │
├──────────────────────────┬─────────────────────────────────────────────┤
│ Kiro Primitive           │ Real Function in Hexperiment                │
├──────────────────────────┼─────────────────────────────────────────────┤
│ 1. Steering Contract     │ .kiro/steering/ml-integrity.md              │
│    (Policy)              │ Enforces the 8 ML rules in every session.   │
├──────────────────────────┼─────────────────────────────────────────────┤
│ 2. Save Hook             │ .kiro/hooks/hexperiment.json                │
│    (Fast Interception)   │ Runs a fast check on save (one training     │
│                          │ run, no canary); clears stale approvals.    │
├──────────────────────────┼─────────────────────────────────────────────┤
│ 3. Custom Agent          │ .kiro/agents/ghost-hunter.json              │
│    (Sandboxed Exorcist)  │ Has MCP tool permissions; shell is denied;  │
│                          │ cannot alter verdicts or data.              │
├──────────────────────────┼─────────────────────────────────────────────┤
│ 4. Typed MCP Server      │ hexperiment-mcp                             │
│    (Structured Tools)    │ Exposes run_audit, list_apps, explain_ghost,│
│                          │ get_verdict to agent context.               │
├──────────────────────────┼─────────────────────────────────────────────┤
│ 5. Workflow Loop         │ .kiro/workflows/exorcise.workflow.json      │
│    (Autonomous Healing)  │ Repeats: Audit -> Diagnose -> Refactor ->   │
│                          │ Verify until verdict.json == APPROVED.      │
└──────────────────────────┴─────────────────────────────────────────────┘
```

### The Autonomous Healing Loop in Practice:
1. The developer asks Kiro to train a model.
2. The agent writes `train.py`.
3. The save hook fires: fast check runs train.py once (no canary) and clears any prior approval.
4. The developer triggers `/workflow run exorcise`.
5. The workflow executes the full Canary audit via `hexperiment-mcp`.
6. Audit returns: `REJECTED: The Possession on reservation_status`.
7. The `ghost-hunter` agent reads the finding, edits `train.py` to drop `reservation_status` and to compute `country_cancel_rate` from training rows only, after the split, and triggers the audit again.
8. Audit returns: `APPROVED: ROC AUC 0.9090`.
9. The workflow completes, generating an official Markdown **Model Card**.

---

## 6. Deep Dive: The Two Demo Applications

### Demo App 1: Hotel Cancellations (`apps/hotel-cancellations`)
* **Problem:** Predict whether a hotel guest will cancel their booking.
* **Dataset:** Real public hotel reservation dataset (119,390 rows, 32 columns).
* **The Planted Haunted Leaks:**
  1. `reservation_status`: A column recording "Canceled" vs "Check-Out" filled at checkout (*The Possession*).
  2. `country_cancel_rate`: A feature computed by calculating the cancellation rate per country across **all rows**, including test rows (*The Peeking Spirit*).
* **Initial Run:** `REJECTED`, ROC AUC **1.0000** (impossible perfection).
* **Kiro's Autonomous Repair:** Dropped `reservation_status`, recomputed `country_cancel_rate` from training rows only, after the split (a new helper, `add_country_cancel_rate`; countries not seen in training get NaN).
* **Final Run:** `APPROVED`, ROC AUC **0.9090** (honest, deployable performance; the committed reference script scores 0.9096). Session: [docs/kiro/exorcise-hotel-cancellations.md](kiro/exorcise-hotel-cancellations.md).

### Demo App 2: Flight Delays (`apps/flight-delays`)
* **Problem:** Predict whether a 2013 flight from NYC will arrive >15 minutes late.
* **Dataset:** 336,776 real flights from JFK, LGA, EWR.
* **The Planted Haunted Leaks:**
  1. Random train/test split across time-ordered data (*The Prophet*).
  2. `carrier_day_late_rate`: each carrier's same-day late rate, computed over all rows, so it carries test labels into training (*The Peeking Spirit*).
* **Initial Run:** `REJECTED`, ROC AUC **0.7971**.
* **Kiro's Autonomous Repair:** Replaced scikit-learn's `train_test_split` with `hexperiment.split(df)`, which splits by the time column declared in `hex.toml` (`split.time_column = "time_hour"`), so every training flight is earlier than every test flight; removed `carrier_day_late_rate`, which is built from the target. Session: [docs/kiro/exorcise-flight-delays.md](kiro/exorcise-flight-delays.md).
* **Final Run:** `APPROVED`, ROC AUC **0.6193** (honest out-of-time predictive score).

---

## 7. The Scientific Benchmark: 78 Apps, 0 False Alarms

To prove that Hexperiment isn't just tuned to its two demo apps, we implemented the formal taxonomy of data leakage from **Kapoor & Narayanan (*Patterns*, 2023)**.

### The Setup:
- 13 distinct leakage mechanisms.
- Tested across 3 model types: `LogisticRegression`, `RandomForest`, `HistGradientBoosting`.
- Total: 39 leaky applications + 39 clean twin applications = **78 total applications**.

### The Results:
- **30 of 39 leaky apps rejected** by the exact predicted ghost.
- **0 of 39 clean twins rejected** (**0% False Alarm Rate**).
- **Subtle Preprocessing Leaks:** A scaler or imputer fitted on all rows only moved the ROC AUC by $\le 0.008$. Human reviewers and score checks never notice this; **the Canary caught all 6 instances**.

### The 3 Calibrated Misses (Why This Proves Scientific Honesty):
A fake benchmark claims 100% detection. A real scientific tool has documented boundaries:
1. `weak-proxy`: A feature with ROC AUC of 0.89 (below the 0.95 cutoff).
2. `time-column-undeclared`: A temporal dataset where no time column was specified.
3. `entity-leak`: Near-duplicate customer rows split across sets without an entity ID.

---

## 8. Judge Defense & Q&A Cheat Sheet

When judges or reviewers evaluate your submission, here are the exact answers to the toughest questions they could ask:

#### Q1: "Why is your honest score lower than the initial score? Didn't the tool make the model worse?"
> **Your Answer:** "No, the initial score was fraudulent. A model scoring 1.0000 on hotel cancellations was looking at `reservation_status`—a column only created at checkout. In real life, that model has 0% predictive power. Hexperiment dropped the score from an impossible 1.0000 to an honest, production-viable 0.9090 (the agent's fix; the committed reference script scores 0.9096)."

#### Q2: "Doesn't running the training script twice double GPU training costs?"
> **Your Answer:** "In Hexperiment, we use a Two-Tier Verification Model. Tier 1 is a Fast Check that runs on save (one normal training run, no canary). Tier 2 is the Canary Test, which runs via hexperiment run, MCP run_audit, or Kiro workflows. For enterprise deep learning models, our architecture roadmap explores ideas like Subsampled Canaries on a 5% validation slice and DAG-hash pipeline auditing."

#### Q3: "Why not just use static AST analysis or linters like Flake8/Ruff?"
> **Your Answer:** "Static linters can only recognize known syntactic patterns. But AI coding agents write dynamic code: they use custom list comprehensions, wrap operations in lambda functions, or call external libraries. The Canary Test watches runtime behavior under input perturbation, making it agnostic to how the code was written."

#### Q4: "How does this prove deep Kiro integration?"
> **Your Answer:** "Hexperiment uses five distinct Kiro primitives: Steering enforces the ML rules; Hooks catch unverified code on save; Custom Agents provide sandboxed repair roles; MCP provides structured tool execution; and Workflows automate the multi-turn diagnosis-and-repair loop until the model card is generated."

#### Q5: "What category does this compete in?"
> **Your Answer:** "Skeleton Crew ($5,000). It is a complete, reusable Python skeleton with two distinct public demo applications, and we are contending for the Overall Grand Prize and the Best Demo bonus."
