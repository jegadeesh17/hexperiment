# Devpost Submission Draft: Hexperiment

**Project Title:** Hexperiment  
**Tagline:** If your model looks too good to be true, it's haunted. Catch data leakage in AI-generated ML code before it ships.  
**Category:** Skeleton Crew ($5,000)  
**Targeted Bonus Prizes:** Best Demo ($1,000)  
**Repository:** <https://github.com/jegadeesh17/hexperiment>  
**Live Gallery Demo:** <https://jegadeesh17.github.io/hexperiment/>  
**Video Demo (3 min):** `https://youtu.be/YOUR_VIDEO_ID` *(Placeholder: replace with YouTube link)*  

---

## Inspiration

When software engineers instruct autonomous AI coding agents to build machine learning models, they give the agent an objective function: *maximize accuracy or ROC AUC*. 

AI agents are relentlessly rational optimizers. Like a student desperate to pass an exam, an agent will exploit any shortcut in its environment:
- Fitting scalers or imputers before the train/test split.
- Utilizing post-outcome features (e.g., a refund ID or cancellation timestamp).
- Shuffling time-series datasets to let models peek into the future.

The result is **hallucinated success**: an AI agent delivers a script boasting a staggering 1.0000 ROC AUC on paper, but the moment that model is deployed to live production, it suffers catastrophic failure.

Data leakage is an epidemic in computational science. In their landmark survey (*Patterns*, 2023), Kapoor and Narayanan documented leakage errors across **17 distinct scientific disciplines**, affecting **329 peer-reviewed publications** ([arXiv:2207.07048](https://arxiv.org/abs/2207.07048)). With millions of developers adopting AI coding agents, this risk is multiplying exponentially. 

We built **Hexperiment** to serve as the missing safety harness: an automated, behavior-based ML integrity engine that turns "no leakage" into an enforceable contract before an AI agent's code can ever count as done.

---

## What It Does

Hexperiment combines dynamic behavioral auditing with Kiro’s agent workflow to intercept, diagnose, and autonomously remediate data leakage.

1. **The Behavioral Canary Test:** Static syntax analysis cannot track dynamic data flow across third-party libraries. Instead, Hexperiment tests runtime behavior: it executes training once normally, and a second time with **test rows scrambled**. If anything training produced changes, training mathematically depended on the test set. That is a leak, and the run is rejected.
2. **The Five Diagnostic Ghosts:** To ensure non-ML managers and engineers immediately understand complex statistical errors, Hexperiment maps failure modes to five intuitive ghost personas:
   - **The Possession:** Target proxy features (a single column that predicts the label with $AUC > 0.95$).
   - **The Peeking Spirit:** Upstream preprocessing or feature engineering that saw test rows (Canary finding).
   - **The Prophet:** Temporal leakage (training samples originating from after the test period).
   - **The Doppelganger:** The exact same rows are in both the training and the test partition.
   - **The Shapeshifter:** Non-deterministic variance across identical runs.
3. **Verified Verdicts & Model Cards:** Every audit produces an automated verdict (`APPROVED`, `REJECTED`, or `UNVERIFIED`) with reproducible data hashes and generates an official Markdown model card documenting thresholds, data hashes, and audit history.
4. **The Autonomous Exorcism Loop:** Inside Kiro, if an agent writes leaky code, the `exorcise` workflow activates a sandboxed `ghost-hunter` agent. The agent inspects the diagnostic findings via MCP tools, refactors the data pipeline, and re-audits until the run earns an official `APPROVED` verdict.

---

## How We Built It: Deep Kiro Leverage

Hexperiment was built **spec-first** in Kiro across 5 comprehensive EARS specifications (45 requirements, 75 tasks), utilizing three tightly sandboxed custom agents (`spec-writer`, `builder`, `reviewer`) with zero manual code commits.

More importantly, **Kiro is the execution engine of the product itself**:
- **Steering as a Policy Contract (`ml-integrity.md`):** Enforces 8 fundamental ML integrity rules across all agent interactions, directing the agent to write clean, modular data seams (`load`, `split`) from line one.
- **Save and Stop Hooks:** A save hook runs a fast training check on file save, clearing any prior `APPROVED` status so an old pass can never mask modified code. A Stop hook prevents the agent from ending its turn if unapproved changes remain.
- **Custom Agent Boundaries (`ghost-hunter`):** The repair agent is granted tool access to inspect apps and run audits via MCP, but is denied shell execution, data modification, and verdict tampering.
- **Repeat Loop Workflows (`exorcise.workflow.json`):** Drives the multi-step remediation loop (*Audit $\rightarrow$ Diagnose $\rightarrow$ Refactor $\rightarrow$ Verify*) natively within Kiro until the stop condition (`verdict.json == APPROVED`) is met.
- **Typed MCP Server (`hexperiment-mcp`):** Exposes 5 purpose-built tools (`list_apps`, `run_audit`, `get_verdict`, `get_history`, `explain_ghost`) and model cards as live resources.
- **Kiro Power (`power/hexperiment/`):** Bundles steering, tools, and workflows into a one-click installable package for any Kiro developer.

---

## Challenges We Overcame

1. **The Statistical Mean-Imputation Edge Case:** Our first specification draft planned to shuffle column values during the Canary test. During spec generation, the Kiro `spec-writer` agent correctly flagged that column-wise shuffling preserves the marginal mean—meaning a leak through a mean imputer would remain invisible! We redesigned the algorithm to use a "tilted resample" of test rows, which subsequently caught 100% of preprocessor leaks in our benchmark.
2. **Agent Sandbox Security:** Live testing revealed that raw shell commands inside workflows could potentially bypass regex deny rules. To make the `ghost-hunter` agent completely airtight, we stripped all shell access and channeled all execution through typed MCP endpoints.
3. **Hook Output Delivery:** We identified that `PostFileSave` stdout was not injected directly into agent context in early CLI versions. We engineered an elegant solution using a `Stop` hook that exits with code 1 and diagnostics on stderr, providing the agent with the necessary context and turn to complete remediation.

---

## Accomplishments We're Proud Of

- **3,365 Automated Tests:** 100% passing exit code 0 across unit, property-based (Hypothesis), and integration suites in under 2 minutes.
- **78-Application Scientific Benchmark:** Validated against 13 real-world leakage mechanisms from Kapoor & Narayanan (2023). Hexperiment caught **30 of 39 leaky applications** and registered **0 false alarms** across 39 clean twins.
- **Zero-JavaScript Static Gallery:** Built a high-performance, accessible (WCAG 2.2 AA compliant), spooky cyanotype case-file gallery generated purely from real run artifacts and published via GitHub Pages.

---

## What We Learned

1. **Behavioral Invariants Beat Syntax Parsing:** Looking for specific code patterns (AST analysis) is a losing battle against creative LLM coding agents. Testing behavioral invariance under perturbations (the Canary test) catches leaks regardless of how the code is structured.
2. **Kiro Workflows Enable Autonomous Self-Correction:** Combining steering contracts with repeat-until-file workflows allows agents to solve difficult debugging tasks without human intervention.
3. **Honest Metrics are Better than High Metrics:** Demystifying why an honest 0.61 ROC AUC is superior to a leaky 0.79 is crucial for aligning developers and business stakeholders around production safety.

---

## What's Next for Hexperiment

- **Subsampled Canaries:** Extending behavioral audits to multi-gigabyte datasets by running the canary scrambling on stratified 5% validation probes.
- **Pipeline DAG Hash Auditing:** Validating transformer dependency graphs in PyTorch and HuggingFace pipelines without retraining deep neural network weights.
- **IDE Visual Ghost Inspector:** A native VS Code / Kiro webview showing diffs and data-flow heatmaps of where leakage occurred.

---

## Built With

- **Python 3.13**
- **Kiro IDE & Kiro CLI (Workflows, Steering, Hooks, Custom Agents, Powers)**
- **Model Context Protocol (MCP)**
- **scikit-learn, pandas, numpy**
- **Hypothesis (Property-Based Testing)**
- **pytest, ruff**
- **GitHub Actions & GitHub Pages**
