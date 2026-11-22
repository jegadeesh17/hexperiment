# Product: Hexperiment

**Motto:** "If your model looks too good to be true, it's haunted."

Hexperiment is a spec-driven ML kit for Kiro that catches data leakage before a result counts. A data scientist states what a model must achieve (a metric threshold) and what it must never do (leak test information into training). The AI agent writes the training code; Hexperiment checks every run and refuses to approve a result whose score depends on test information reaching training.

## Who it is for
- **Customer:** a data scientist or ML engineer who lets an AI coding agent write training code and needs to trust the number it reports.
- **Also served:** the reviewer who signs off the model, and the people who rely on its predictions.

## Why it matters
Data leakage is a common cause of ML results that do not reproduce: a survey by Kapoor & Narayanan found leakage errors in 17 fields, affecting 294–329 papers (329 in the 2022 preprint, 294 in the published version; [arXiv:2207.07048](https://arxiv.org/abs/2207.07048)). A leak raises the score, so whoever tunes for the metric, a person or an agent, gets no signal that something is wrong. Existing leakage tools lint notebooks after a human wrote them; Hexperiment makes "no leakage" a contract the agent must satisfy before its result counts.

## The ghosts (the checks)
Each check reports its finding as a named ghost, so a failed run reads like a story:
| Ghost | Check | What it means |
|---|---|---|
| The Doppelganger | row overlap | identical rows exist in both train and test |
| The Possession | target proxy | one feature alone predicts the target almost perfectly |
| The Peeking Spirit | canary test | scrambling the test rows changed training, so test information reached training |
| The Prophet | temporal leak | training rows come from after test rows |
| The Shapeshifter | determinism (warning) | identical runs give different results, so nothing can be verified |

The canary test is the core idea: the kit re-runs the training script with the test rows scrambled at load time. In a leak-free pipeline nothing about training depends on test rows, so the training predictions must not move. It finds a leak through the test rows wherever it sits (global statistics, helper functions, other files) without reading the code, as long as the data goes through `hexperiment.load()` and `split()`. A leak that does not depend on the test rows' values (a column recorded after the outcome, for example) is left to the other checks, which catch only some kinds.

## What the kit is
- A Python package, `hexperiment`: `load()` and `split()` for data, `audit(...)` at the end of a training script, and `hexperiment run <app>` (fast on save, full with the canary at the gate). It writes `verdict.json` (APPROVED, REJECTED or UNVERIFIED) and a model card.
- A Kiro layer: steering with ML-integrity rules; a hook that runs the fast check when the agent saves training code, and one that tells the agent at the end of its turn which changed apps are not approved; an MCP server with the audit tools; a restricted `ghost-hunter` agent with no shell; and the `exorcise` workflow that loops fix, retrain, re-check until the verdict is APPROVED. Shipped as a Kiro power, and written into a project by `hexperiment init --kiro`.
- Two example applications built on the kit, each with a realistic leak that the kit catches.

## Non-goals
Deep learning, GPU training, notebooks as input (scripts only), hyperparameter tuning, experiment-tracking servers, hosted training.
