# Requirements Document: hexperiment-gallery

## Introduction

An audit is only convincing if people can see it. This spec adds `hexperiment gallery`, a command that turns the recorded run history of one or more apps into a static website: a landing page that shows the score that was too good to be true next to the honest one, and one case page per app with its ordered run record, every finding, the code change that removed the leak and the model card of the last approved run. The project publishes the gallery of its two example apps on GitHub Pages, built from the committed run history, so the public page always shows real runs.

The command is part of the product: any user can publish the audit history of their own apps the same way. The visual design follows `.kiro/steering/gallery-design.md`, which is part of these requirements.

Scope: the `hexperiment gallery` command and its package module, the Pages workflow, and tests. The leak benchmark that produces the optional benchmark table is covered by a later spec; this spec fixes only the file format the gallery reads.

## Glossary

- **App, run, verdict, history:** as in `hexperiment-core`. The history of an app is the list returned by `hexperiment.load_history`, oldest first.
- **Gallery folder:** the output folder of `hexperiment gallery`.
- **Marker file:** `.hexperiment-gallery`, written into every gallery folder, so that a later build knows the folder is its own.
- **Haunted script:** `<app>/haunted/train.py`, when it exists: the version of the training script before the fix.
- **Benchmark results file:** a JSON file in the format of Requirement 6.

## Requirements

### Requirement 1: The command

**User Story:** As a data scientist, I want one command that builds a website from my apps' audit history, so that I can show reviewers what was found and fixed without writing a page by hand.

#### Acceptance Criteria

1. THE command `hexperiment gallery APP_DIR [APP_DIR ...] --out OUT_DIR` SHALL write a static site into `OUT_DIR`: `index.html`, one folder per app (its case folder, Requirement 1.7) with an `index.html`, the folder `assets/` with the stylesheet (and at most one small script), and, per app, copies of the last run's `verdict.json` and the last approved run's `model_card.md`. It SHALL print the path of `index.html` and exit with code `0`.
2. THE options `--repo-url URL` (a link shown on every page) and `--benchmark FILE` (a benchmark results file) SHALL be optional. A page SHALL NOT show the benchmark section or the repository link when the option is not given.
3. IF an `APP_DIR` is not an app folder, has no recorded run, or two apps have the same name THEN the command SHALL write nothing, print one message on standard error that names the problem and the fix, and exit with code `2`.
4. IF `OUT_DIR` exists and is not empty and does not contain the marker file THEN the command SHALL refuse with exit code `2` and write nothing. IF it contains the marker file THEN the command SHALL replace its contents. THE command SHALL NOT delete or write anything outside `OUT_DIR` other than its own temporary folders of Requirement 1.5, and SHALL refuse an `OUT_DIR` that is inside one of the app folders or contains one, is a file or a link, or whose parent folder does not exist.
5. THE command SHALL build the whole site in a temporary folder next to `OUT_DIR` and move it into place only when every page was written, so a failed build leaves the previous gallery unchanged.
6. THE command SHALL only read the apps: it SHALL NOT run `train.py`, change any file in an app folder or count as a use of the test set.
7. THE case folder of an app SHALL be the slug of its `[app] name`: lower case, every run of characters other than `a`-`z` and `0`-`9` replaced by one `-`, and leading and trailing `-` removed. IF a slug is empty, is `assets` or a Windows device name (`con`, `prn`, `aux`, `nul`, `com1` to `com9`, `lpt1` to `lpt9`), or equals the slug of another app given THEN the command SHALL refuse as in Requirement 1.3.
8. THE command SHALL accept `--repo-url` only as an `http` or `https` URL with a host and without whitespace, control characters, quotes or angle brackets; otherwise it SHALL refuse as in Requirement 1.3. The link SHALL read "Source on GitHub" when the host is `github.com` and "Source code" otherwise.

### Requirement 2: The landing page

**User Story:** As a judge or an ML engineer arriving from a link, I want to see within seconds what Hexperiment caught and why it matters, so that I decide to read on.

#### Acceptance Criteria

1. THE landing page SHALL open with the haunted number as described in the design direction: the score of the first `REJECTED` run of the app whose first rejected score is the highest (ties go to the larger drop to its last approved score, then to the app given first), next to that app's last approved score, with one sentence written from the findings of that run (ghost names and their locations) and the motto "If your model looks too good to be true, it's haunted."
2. THE landing page SHALL show one summary per app: its name and description, the status and score of its first run, the ghosts found across its history, the score and status of its last run, the number of recorded runs, and a link to its case page.
3. THE landing page SHALL explain the canary check in three ordered steps (train normally; train again with the test rows scrambled; compare what the model learned) in at most 80 words in total.
4. THE landing page SHALL list every entry of `hexperiment.ghosts.CATALOG` that has a ghost name, with that name, what it means, why it matters and how to fix it, using the catalogue's own texts. Text between backticks in these texts and in finding texts SHALL be shown as code.
5. WHEN `--benchmark` is given THEN the landing page SHALL show the benchmark table of Requirement 6.
6. IF no app has an approved run THEN the hero SHALL show the app chosen by the rule of Requirement 2.1 with the sentence "No approved run yet." and no honest score. IF no app has a rejected run with a score THEN the hero SHALL show no number, only the motto and the sentence "No run has been rejected yet." A rejected run without a score (it stopped on an input error) SHALL NOT be chosen.
7. THE motto SHALL be the landing page's only `h1`; the haunted number is a figure, not a heading.

### Requirement 3: The case page

**User Story:** As a reviewer, I want to read one app's full audit record, so that I can check every claim the landing page makes.

#### Acceptance Criteria

1. THE case page SHALL show the app's name, description, target column, metric name, threshold and number of recorded runs, read from `hex.toml` and the history.
2. THE case page SHALL show every recorded run in history order as an ordered list. Each entry SHALL show the run number, status, mode, score, the run's creation time in UTC, the number of test-set uses, and every finding with its ghost name (or its check name when it has none), severity, location, message and fix; a waived finding SHALL be marked as waived and show its waiver reason.
3. WHEN the app has a haunted script THEN the case page SHALL show a line diff of the haunted script against `train.py`, with line numbers, removed and added lines marked by both colour and a `-` or `+` sign. WHEN it has none THEN the section SHALL be left out.
4. WHEN the app has an approved run THEN the case page SHALL show the model card of the last approved run, built from that run's verdict (status, metric and threshold, split, data summary, checks and their outcomes, waivers) and link the copied `model_card.md`. WHEN it has none THEN the page SHALL say "No approved run yet."
5. THE case page SHALL link the copied `verdict.json` of the last run.

### Requirement 4: Safe and portable output

**User Story:** As a maintainer publishing a gallery on a public host, I want the pages to be safe and self-contained, so that a strange column name or finding text cannot break or attack a page.

#### Acceptance Criteria

1. THE command SHALL escape every text that comes from an app (names, descriptions, column names, locations, messages, fix hints, script lines) for HTML, so that a value such as `<script>alert(1)</script>` appears as text.
2. THE pages SHALL contain no script, no inline event handlers, no `style` attributes and no `javascript:` or `data:` URLs. The only external reference in the pages SHALL be the Google Fonts stylesheet (which loads its own font files), with no `preconnect` hint, and every page SHALL stay readable with the system font fallbacks and without any stylesheet.
3. EVERY link between pages and to assets SHALL be relative, so the site works when served from a sub-path such as `https://user.github.io/repo/`, and every link SHALL point to a file the build wrote (or to `--repo-url`).
4. THE output SHALL be deterministic: building twice from the same inputs SHALL give byte-identical files, whatever the folder the apps are in and whatever the line endings of the input files (a `train.py` with CRLF line endings, as Git writes it on Windows, SHALL give the same pages and the same diff as the same file with LF). Pages SHALL contain no build time; the only times shown are the runs' own times. Output files SHALL be UTF-8 without a byte order mark, with LF line endings.
5. THE package SHALL keep its runtime dependencies (`pandas`, `numpy`, `scikit-learn`): the gallery SHALL use only the Python standard library and the package's own modules. No gallery module SHALL import pandas, scikit-learn, scipy or `mcp` itself, and building a gallery SHALL NOT load scikit-learn, scipy or `mcp` (pandas is loaded by the core modules the gallery reads, such as `runs`).

### Requirement 5: Design and accessibility

**User Story:** As a visitor on any device, I want pages that are easy to read and use, so that the evidence is accessible to everyone.

#### Acceptance Criteria

1. THE pages SHALL follow `.kiro/steering/gallery-design.md`: its colours, typefaces, type scale, layouts, components, motion and copy rules.
2. THE pages SHALL meet WCAG 2.2 AA: text contrast, a visible keyboard focus, a skip link, landmarks, `lang="en"`, a distinct title per page, and status never shown by colour alone.
3. THE pages SHALL have no horizontal page scroll from 360px wide; the code diff and wide tables SHALL scroll inside their own area.
4. WHEN the visitor prefers reduced motion THEN the page SHALL show no animation.

### Requirement 6: Benchmark results file

**User Story:** As a judge, I want the measured catch rate of the checks next to the case studies, so that I can see how well the kit works beyond two examples.

#### Acceptance Criteria

1. THE benchmark results file SHALL be in the format of `hexperiment-benchmark`, Requirement 4 (`format` `"hexperiment-benchmark/1"`). THE gallery SHALL validate every key it shows and ignore keys it does not show.
2. THE benchmark section SHALL show one table row per mechanism, grouped by leakage type in file order: the leakage type (ID and the paper's name), the mechanism, the ghosts designed to catch it (or "No check is designed for it"), "rejected N of M" with the rate in percent, and the ghosts or checks that rejected it with their counts. A last row SHALL show the clean twins: "N false alarms in M scripts" with the rate. Below the table it SHALL list every not-measured leakage type with its reason, and state the taxonomy citation, the `hexperiment`, Python and scikit-learn versions, the date of the run and the number of runs.
3. IF the benchmark results file cannot be read, does not match the format, or has `complete` set to false THEN the command SHALL exit with code `2` and name the first problem.

### Requirement 7: Publishing the example gallery

**User Story:** As the project owner, I want the public gallery rebuilt from the committed history on every push, so that it can never drift from the real runs.

#### Acceptance Criteria

1. THE repository SHALL contain `.github/workflows/pages.yml` (written by the owner): on a push to `main` (and on manual start) a build job SHALL install the package without `-e`, run `hexperiment gallery apps/hotel-cancellations apps/flight-delays --out site --repo-url <the repository URL>` (adding `--benchmark hexperiment/benchmark/results.json` once that file is committed), and upload `site` as the Pages artifact; a deploy job SHALL publish it with the official GitHub Pages actions. It SHALL need no secret and no dataset download.
2. WHILE the repository is private, the build job SHALL still run and the deploy job SHALL be skipped, not failed.
3. THE workflow SHALL pin every action to a full commit SHA with the version in a comment. The workflow and the build job SHALL have only `contents: read`; only the deploy job SHALL add `pages: write` and `id-token: write`.

### Requirement 8: Quality

**User Story:** As a maintainer, I want the gallery covered by tests like the rest of the kit, so that it keeps working.

#### Acceptance Criteria

1. THE tests SHALL cover: each page builder on verdicts built in the test; escaping of a hostile app name, column name, finding text and script line; the hero choice (Requirement 2.1) and the no-approved-run case; the diff markup; the benchmark table and every format error of the benchmark file; the command's exit codes and messages for each refusal of Requirement 1; that a refused or failed build leaves an existing gallery unchanged; determinism; that every relative link of every page resolves to a written file; and that every page parses as HTML with balanced tags (with `html.parser`).
2. THE tests SHALL build the gallery of the two committed example apps and check that it states their real first and last statuses, scores and ghost names.
3. THE tests SHALL check Requirement 4.5 (no gallery module imports `pandas`, `sklearn`, `scipy` or `mcp`; building a gallery loads none of `sklearn`, `scipy`, `mcp`), that the runtime dependencies are unchanged, and that `.github/workflows/pages.yml` meets Requirement 7 (the test reads it as text and does not skip).
4. THE full test suite SHALL pass with `.venv/Scripts/python -m pytest hexperiment/tests -q` in under 2 minutes in everyday use (as defined in `hexperiment-kiro`, Requirement 7.4), and `ruff` SHALL report no problems.
