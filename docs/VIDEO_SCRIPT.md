# Hexperiment: 3-Minute Video Script & Recording Companion

**Target Duration:** Exactly 180 seconds (3:00 hard limit)  
**Targeted Prize:** Best Demo ($1,000 bonus) + Presentation Score (20% of Overall Evaluation)  
**Tone:** Confident, crisp, authoritative, spooky yet scientifically rigorous  
**Pacing:** 135–140 words per minute (deliberate, clear enunciation)

---

## 1. Recording Setup & Screen Layout

To ensure your video looks like a top-tier hackathon submission:
- **Resolution:** 1920x1080 (1080p) at 60 fps.
- **Recording Software:** OBS Studio or Loom (High Quality).
- **IDE Theme:** High-contrast dark theme (e.g., Kiro Dark / VS Code Dark+), font size bumped to 16–18px for legibility.
- **Terminal Layout:** Clean terminal window with colors enabled. Clear prior command history before starting.
- **Browser Windows Ready:**
  - Tab 1: Live GitHub Pages Gallery (`https://jegadeesh17.github.io/hexperiment/`)
  - Tab 2: Public GitHub Repository (`https://github.com/jegadeesh17/hexperiment`)
- **Audio:** Use a headset or external microphone; record in a quiet room with background noise cancellation enabled.

---

## 2. Teleprompter Script & Visual Timeline

### Scene 1: The Hook — The AI Agent's Dirty Secret (`0:00 - 0:30` | 30s)

**Visual Setup:**
- [0:00 - 0:12] Screencast: Terminal showing an AI coding agent generating a Python ML training script. The final log line prints in green:
  `Training Complete. Test ROC AUC: 1.0000. Model Ready for Production.`
- [0:12 - 0:22] Slide / Graphic: Split-screen visual. On the left: `Development: 1.0000 AUC (Green)`. On the right: A production crash chart plummeting to near-zero accuracy (Red).
- [0:22 - 0:30] Text Callout: **HALLUCINATED SUCCESS: DATA LEAKAGE.**

**Voiceover Narration:**
> *"AI coding agents are astonishingly good at writing machine learning code. [Pause]*  
> *But they have a dark secret.*  
> *When you ask an agent to maximize model accuracy, it behaves like an ambitious student taking an exam: if it can peek at the answer key, it will.*  
> *It fits a scaler before the split, encodes future data, or uses an ID column that gives the answer away.*  
> *The result is a model that reports a perfect 1.0000 ROC AUC on paper... but completely collapses the moment it touches real production data."*

---

### Scene 2: The Canary Test — Behavioral Verification (`0:30 - 1:15` | 45s)

**Visual Setup:**
- [0:30 - 0:45] Screen: Hexperiment logo appears with tagline: *"If your model looks too good to be true, it's haunted."*
- [0:45 - 1:00] Diagram / Animation:
  - Box A: Normal Run $\rightarrow$ Model Trains $\rightarrow$ Records Output.
  - Box B: Canary Run $\rightarrow$ **Test Rows Scrambled** while training rows stay untouched.
  - Comparison arrow showing outputs changed $\rightarrow$ Red Flag: `LEAKAGE REJECTED`.
- [1:00 - 1:15] Graphic: Display the **5 Diagnostic Ghosts** card:
  - *The Possession* (Proxy Leakage)
  - *The Peeking Spirit* (Preprocessing / Test Leakage)
  - *The Prophet* (Temporal Leakage)
  - *The Doppelganger* (Duplicate Overlap)
  - *The Shapeshifter* (Non-Deterministic Variance)

**Voiceover Narration:**
> *"Traditional code linters inspect syntax trees, but they can't trace dynamic data flow across complex libraries.*  
> *That’s why we built **Hexperiment**.*  
> *Instead of reading code, Hexperiment tests **runtime behavior** using a dynamic Canary Test.*  
> *It trains the model once normally. Then, it runs a second time with the test rows scrambled, leaving training data untouched.*  
> *Here is the mathematical invariant: if anything training produced changes between those two runs, training looked at the test rows. That is a leak, and the run is rejected.*  
> *We translate these statistical errors into five diagnostic ghosts—so developers and managers instantly understand whether their model has a temporal Prophet or a proxy Possession."*

---

### Scene 3: Live Demo — The Kiro Autonomous Exorcism (`1:15 - 2:15` | 60s)

**Visual Setup:**
- [1:15 - 1:25] Kiro IDE: Open `apps/hotel-cancellations/haunted/train.py`. Point out the suspicious 1.0000 metric in the console.
- [1:25 - 1:40] Terminal in Kiro: Execute the audit:
  ```bash
  hexperiment run apps/hotel-cancellations
  ```
  The screen shows the audit rejecting the run:
  `REJECTED (full) roc_auc 1.0000`
  `Ghosts found: The Possession (reservation_status), The Peeking Spirit (country_cancel_rate)`
- [1:40 - 2:00] Kiro Chat: Type the workflow command:
  ```bash
  /workflow run exorcise --app apps/hotel-cancellations
  ```
  Highlight the `ghost-hunter` agent activating. Show the agent reading the MCP findings, opening `train.py`, removing `reservation_status`, rebuilding `country_cancel_rate` strictly on training rows, and re-running the audit.
- [2:00 - 2:15] Terminal in Kiro: The loop terminates with:
  `APPROVED (full) roc_auc 0.9096 >= 0.8500`
  Show the newly generated Markdown Model Card in `runs/`.

**Voiceover Narration:**
> *"Now watch how this works natively inside Kiro.*  
> *Here is our Hotel Cancellations app. An agent wrote code that hit an impossible 1.0000 ROC AUC.*  
> *When Hexperiment audits the run, it immediately rejects it: finding 'The Possession' on `reservation_status` and 'The Peeking Spirit' on a leaked target rate.*  
> *Now, we launch Kiro’s `exorcise` workflow.*  
> *Our custom `ghost-hunter` agent wakes up. Notice that the agent is securely sandboxed—it has no shell access and cannot alter thresholds.*  
> *Using our MCP tools, it diagnoses the root causes, refactors the data pipeline into a proper transformer, and re-audits.*  
> *In a single autonomous round, the workflow achieves an APPROVED verdict at an honest 0.9096 ROC AUC, and generates a reproducible model card."*

---

### Scene 4: Scientific Benchmark — 78 Apps, 0 False Alarms (`2:15 - 2:45` | 30s)

**Visual Setup:**
- [2:15 - 2:30] Screen: Show the benchmark table in `hexperiment/benchmark/README.md`.
- [2:30 - 2:45] Visual Callout Cards:
  - **78 Applications Tested** (13 mechanisms $\times$ 3 models + clean twins)
  - **30 / 39 Leaky Apps Caught**
  - **0 / 39 Clean Twins Rejected (0% False Alarm Rate)**
  - **Caught Microscopic Leaks ($\Delta \text{AUC} \le 0.008$) Missed by Humans**

**Voiceover Narration:**
> *"This isn't just a toy demo.*  
> *We benchmarked Hexperiment against the formal leakage taxonomy from Kapoor & Narayanan's landmark 2023 survey in Nature Patterns.*  
> *We evaluated 13 distinct leakage mechanisms across three different machine learning models—running 78 applications in total.*  
> *Hexperiment caught 30 of 39 leaky applications as predicted, and rejected **zero clean twins**—delivering a zero percent false alarm rate.*  
> *Even subtle preprocessing leaks that only shifted accuracy by 0.008—which human reviewers easily miss—were caught by the Canary."*

---

### Scene 5: Conclusion & The First ML Power for Kiro (`2:45 - 3:00` | 15s)

**Visual Setup:**
- [2:45 - 2:55] Screen: Fast scroll through the live GitHub Pages gallery (`https://jegadeesh17.github.io/hexperiment/`), showing the spooky cyanotype case files, diff toggles, and model cards.
- [2:55 - 3:00] Terminal: Display the one-line install command:
  ```bash
  hexperiment init --kiro
  ```
  Closing screen with Hexperiment logo and GitHub repo URL.

**Voiceover Narration:**
> *"Hexperiment is the first machine learning integrity power in the Kiro catalog. It provides the automated guardrails teams need to trust AI coding agents with production ML.*  
> *Check out our live gallery on GitHub Pages, or install it into your Kiro workspace today. Thank you!"*

---

## 3. Post-Recording Quality Checklist

Before submitting to Devpost:
- [ ] **Exact Duration Check:** Verify the exported MP4 file is $\le 180$ seconds. (Even 181 seconds can disqualify an entry from the Best Demo prize).
- [ ] **Audio Balance:** Voiceover should be loud, clear, and centered ($-6\text{ dB}$ to $-3\text{ dB}$).
- [ ] **1080p Check:** Ensure code text and terminal fonts are razor-sharp on YouTube playback.
- [ ] **YouTube Settings:** Upload as **Public** or **Unlisted** (never Private!). Set title to: `Hexperiment - Kiroween 2026 Hackathon Demo`.
