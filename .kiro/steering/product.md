# Product: Hexperiment

**Motto:** "If your model looks too good to be true, it's haunted."

Hexperiment is a spec-driven, leakage-proof ML kit for Kiro. A data scientist states what a model must achieve (a metric threshold) and what it must never do (leak test information into training). The AI agent writes the training code; Hexperiment checks every run and refuses to approve a result that is good only because it cheated.

## Who it is for
- **Customer:** a data scientist or ML engineer who lets an AI coding agent write training code and needs to trust the number it reports.
- **Also served:** the reviewer who signs off the model, and the people who rely on its predictions.

## Why it matters
Data leakage is the most common cause of ML results that do not reproduce: Kapoor & Narayanan found leakage-type errors in 294 to 329 published papers across 17 fields. An agent optimizing a metric will happily find a leak. Existing leakage tools lint notebooks after a human wrote them; Hexperiment makes "no leakage" a contract the agent must satisfy before its result counts.

## The ghosts (the checks)
Each check reports its finding as a named ghost, so a failed run reads like a story:
| Ghost | Check | What it means |
|---|---|---|
| The Doppelganger | row overlap | identical rows exist in both train and test |
| The Possession | target proxy | one feature alone predicts the target almost perfectly |
| The Peeking Spirit | canary test | scrambling the test rows changed training, so test information reached training |
| The Prophet | temporal leak | training rows come from after test rows |
| The Shapeshifter | determinism (warning) | identical runs give different results, so nothing can be verified |

The canary test is the core idea: the kit re-runs the training script with the test rows scrambled at load time. In a leak-free pipeline nothing about training depends on test rows, so the training predictions must not move. It catches leaks wherever they hide (global statistics, helper functions, other files) without guessing from the code.

## What the kit is
- A Python package, `hexperiment`: `load()` and `split()` for data, `audit(...)` at the end of a training script, and `hexperiment run <app>` (fast on save, full with the canary at the gate). It writes `verdict.json` (APPROVED, REJECTED or UNVERIFIED) and a model card.
- A Kiro layer: steering with ML-integrity rules, a spec template, hooks that run the checks on save, and a workflow that loops fix, retrain, re-check until the verdict is APPROVED. Shipped as a Kiro power.
- Two example applications built on the kit, each with a realistic leak that the kit catches.

## Non-goals
Deep learning, GPU training, notebooks as input (scripts only), hyperparameter tuning, experiment-tracking servers, hosted training.
