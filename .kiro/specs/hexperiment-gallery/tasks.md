# Implementation Plan: hexperiment-gallery

## How to use this plan

- Work the tasks in order. Each task ends with the whole suite green and `ruff` clean, so a later task never has to repair an earlier one.
- Run all commands from the repository root. Every task ends with the same two full commands, `.venv/Scripts/python -m pytest hexperiment/tests -q` and `.venv/Scripts/python -m ruff check hexperiment`; the "Verify" line of a task names the focused test files to run first.
- Tests live next to the module they cover (`tests/test_gallery_<module>.py`) and are written in the same task as the module. Where `design.md` lists a Hypothesis property for a module, the task adds it.
- Signatures, algorithms, the markup of every page and the test lists are in `design.md`. This plan names what to build and where, not how.
- The order is: pure modules first (tasks 1 to 8), then the reader, the writer and the command (tasks 9 to 11), then the stylesheet (12), then the redesign approved by the owner (12a to 12d: fonts and art, the new stylesheet, the landing page, the case page), then the tests on the example apps and the workflow (13 and 14), then the final check (15). The owner tasks are 16 and 17; the last one is the owner's check in a browser.
- The core rules still hold: pure modules stay pure, `tests/test_architecture.py` stays green, no module-level mutable state, only `canary.py` imports `subprocess`, `hexperiment/__init__.py` exports the same seven names, and the runtime dependencies stay `pandas`, `numpy` and `scikit-learn`. Nothing under `apps/` is changed and nothing under `.github/` is written by the builder.
- Do not edit `requirements.md`, the steering files or `design.md`. If the design is ambiguous or contradicts itself, record the question in the report at the end of the task instead of guessing.
- Line endings: every reader of an app file goes through `newlines.py` first (design, "`newlines.py`"). A test that writes files for the gallery writes them in binary mode, so Windows cannot change the line endings the test asked for.
- **Test time.** The gallery adds about 330 small tests, all in process, each a few milliseconds (Hypothesis tests use `max_examples=50`). Exactly one new test starts a Python process: the import-isolation test of task 11. Each task says how many tests it adds and whether any starts a process. The whole suite must stay under 2 minutes in everyday use (`hexperiment-kiro`, Requirement 7.4). If a run is above 110 seconds, report the slowest tests with `--durations=10` instead of lowering Hypothesis settings.
- **Owner tasks** (16 and 17) are done by the project owner after the builder finishes: the builder cannot write `.github/` and cannot open a browser. The builder skips them and lists them in its report.

## Tasks

- [x] 1. The package, line endings and the HTML builder
  - Create `hexperiment/src/hexperiment/gallery/__init__.py` (a docstring, no imports).
  - Create `gallery/newlines.py` (pure): `normalize_text` and `normalize_bytes`, as in `design.md` (`\r\n` first, then `\r`; nothing else changes).
  - Create `gallery/markup.py` (pure): `Markup`, `esc`, `literal`, `el`, `prose`, `code`, `spaced`, `document`, with the tag allowlist, the attribute rules (`on*`, `style`, `srcdoc` refused; `href` and `src` relative or `http(s)`), the control-character and lone-surrogate replacement, the code spans and the layout rules of the design.
  - Create `gallery/records.py` (pure): the frozen dataclasses `AppSource`, `BenchmarkMechanism`, `NotMeasured`, `Benchmark` and `GalleryInputs`.
  - Create `hexperiment/tests/test_gallery_newlines.py` (the cases and the idempotence property of "Other tests" in `design.md`) and `hexperiment/tests/test_gallery_markup.py` (escaping of every character class, `prose` with and without partner backticks and with `` `<script>` ``, every refusal of `el` (`script`, `style`, `iframe`, `onclick`, `javascript:`, `data:`, `//host`, an `int` child, a bad attribute name), void tags, block layout and inline text exactness, `document` bytes without BOM and without `\r`, and Hypothesis properties: any text round-trips through `esc` and an `HTMLParser` as the text with controls replaced; `prose` output never contains an unescaped `<` other than the two code tags).
  - In `hexperiment/tests/test_architecture.py` add `newlines.py`, `markup.py` and `records.py` to `PURE_FILES`; add the rule that no gallery module imports `hexperiment.cli`, `pandas`, `sklearn`, `scipy` or `mcp`, and the rule that every call to `literal(...)` in the package has a string literal as its argument (checked with `ast`). Later tasks add their modules to `PURE_FILES` in the task that creates them.
  - _Requirements: 2.4 (code spans), 4.1, 4.2, 4.4, 4.5_
  - Tests added: about 48 (4 Hypothesis, 4 architecture). None starts a Python process.
  - Verify: `.venv/Scripts/python -m pytest hexperiment/tests/test_gallery_newlines.py hexperiment/tests/test_gallery_markup.py hexperiment/tests/test_architecture.py -q`, then `.venv/Scripts/python -m pytest hexperiment/tests -q` and `.venv/Scripts/python -m ruff check hexperiment`.

- [x] 2. Test helpers, slugs, summaries and the hero rule
  - Create `hexperiment/tests/gallery_builder.py` with `make_verdict(status, mode, value, findings, created_at, run_id, ...)` and `make_source(...)` (a `Verdict` and an `AppSource` built in memory, like `app_builder.py`). `write_app` follows in task 9.
  - Create `gallery/summary.py` (pure): `MARKER_FILENAME`, `RESERVED_SLUGS`, `slugify`, `assign_slugs`, `score_text`, `ghosts_found`, `suspect_lines`, `AppSummary`, `summarize_app`, `Hero` and `choose_hero` with the single rule of `design.md` (largest `(score, drop, -index)`; a rejected run without a score is never chosen; `None` when there is no candidate; `approved=None` when the chosen app has no approved run).
  - Create `hexperiment/tests/test_gallery_summary.py`: slug examples and every refusal message of Requirement 1.7 (empty, `assets`, a Windows device name, a repeated slug, the same folder twice), `score_text` (four decimals, `no score`), `ghosts_found` order, `suspect_lines` on the real fix text format, and the hero rows of the coverage table in `design.md` (highest score; tie to the larger drop; tie to the app given first; rounding to four decimals decides a tie; no approved run anywhere; an app without an approved run beats a lower score with one; no rejected run; a rejected run without a score is skipped). Hypothesis: the slug matches `^([a-z0-9]+(-[a-z0-9]+)*)?$` and is idempotent; for any list of apps the hero is a member and no other candidate has a larger key.
  - Add `summary.py` to `PURE_FILES`.
  - _Requirements: 1.7, 2.1, 2.2, 2.6_
  - Tests added: about 30 (3 Hypothesis). None starts a Python process.
  - Verify: `.venv/Scripts/python -m pytest hexperiment/tests/test_gallery_summary.py hexperiment/tests/test_architecture.py -q`, then the two full commands.

- [x] 3. The script diff
  - Create `gallery/difftext.py` (pure): `DiffLine`, `Context`, `Change`, `Gap`, `ScriptDiff`, `diff_scripts` and `diff_markup`. `diff_scripts` calls `normalize_text` on both arguments first, splits on `\n` only (never `splitlines`), uses `difflib.SequenceMatcher(None, old, new, autojunk=False)` and `get_grouped_opcodes(context)`; `diff_markup` writes the structure of `design.md` (`role="region"`, label, `tabindex="0"`, `.removed` and `.added` with real `-` and `+` signs, line numbers on both sides).
  - Create `hexperiment/tests/test_gallery_diff.py`: the cases of the coverage table (equal texts, pure insert, pure delete, unequal replace, far-apart changes with gaps and counts, line numbers on both sides, empty input, a script of more than 200 lines compared without autojunk), CRLF and lone-CR input giving the same `ScriptDiff` as LF input, a text with `\x0c` and `\u2028` not split, the markup checks (signs, one occurrence of each line, region attributes) and a hostile line `<script>alert(1)</script>` shown as text. Hypothesis: replaying context plus added lines rebuilds the new text, context plus removed lines the old text; the diff of any text against its CRLF version is empty.
  - Add `difftext.py` to `PURE_FILES`.
  - _Requirements: 3.3, 4.1, 4.4, 5.3_
  - Tests added: about 24 (2 Hypothesis). None starts a Python process.
  - Verify: `.venv/Scripts/python -m pytest hexperiment/tests/test_gallery_diff.py hexperiment/tests/test_architecture.py -q`, then the two full commands.

- [x] 4. The model card section
  - Create `gallery/modelcard.py` (pure): `model_card_section(verdict, *, card_href)` with the contents and number formats listed in `design.md` (key and value `dl`, checks table, waivers table or "No waivers applied.", test-set uses, link to `model_card.md`; tables inside `<div class="scroll" role="region" tabindex="0" aria-label="...">`). Do not import the private helpers of `card.py`.
  - Create `hexperiment/tests/test_gallery_modelcard.py`: every value of Requirement 3.4 appears; waivers with their reasons; a verdict with and without a time column; the comparison with `card.render` (every formatted value of the section also appears in the card text); a hostile waiver reason is escaped.
  - Add `modelcard.py` to `PURE_FILES`.
  - _Requirements: 3.4, 4.1_
  - Tests added: about 10. None starts a Python process.
  - Verify: `.venv/Scripts/python -m pytest hexperiment/tests/test_gallery_modelcard.py hexperiment/tests/test_architecture.py -q`, then the two full commands.

- [x] 5. The benchmark section
  - Create `gallery/benchmark.py`: `BENCHMARK_FORMAT`, `parse_benchmark(text, *, source)`, `run_count`, `percent` and `benchmark_section(benchmark, ghost_names)`. `parse_benchmark` reports the first problem in the order of the 21-row table in `design.md` ("`benchmark.py`"), each message starting with `<source>: ` and the key path, including `format` other than `"hexperiment-benchmark/1"`, `complete` missing or false (the message that names `--only`), and every key the section shows. The module imports `datetime` and stays out of `PURE_FILES`; add the comment in `test_architecture.py` that says why (as in `design.md`, "Dependency rules").
  - Create `hexperiment/tests/test_gallery_benchmark.py`: a valid file built in the test as a dictionary and dumped with `json`; the table (grouping by leakage type with the `rowspan` cell, a file that interleaves two types, designed ghosts or "No check is designed for it", "rejected N of M" with the rate, `by_ghost` order, the clean row in the singular and the plural, the not-measured list, citation, three versions, date, run count); one parametrised test per row of the error table (21), each with one mutation of the valid dictionary, the expected first problem, the prefix `<source>: ` and the key path; the order of two problems (`complete` false before a bad `created_at`); bool, `3.0` and a string are not integers; `NaN` refused; a BOM accepted; extra keys (`statuses`, `by_ghost` of `clean`) ignored; hostile `mechanism`, `reason` and `source` escaped; a name that is not a catalogue ghost is not set as a ghost.
  - _Requirements: 2.5, 6.1, 6.2, 6.3, 4.1_
  - Tests added: about 40 (1 Hypothesis: a valid dictionary with random counts within bounds always parses and renders a table with one row per mechanism). None starts a Python process.
  - Verify: `.venv/Scripts/python -m pytest hexperiment/tests/test_gallery_benchmark.py hexperiment/tests/test_architecture.py -q`, then the two full commands.

- [x] 6. The page shell and the case page
  - Create `gallery/pages.py` (pure) with `FONTS_URL`, `relative_href`, `repo_link_label`, `status_mark`, `page_shell` and `case_page` (sections and markup of "Case page" in `design.md`: back link, `h1`, fact line, the ordered record with status mark, mode, score, UTC time, test-set uses and every finding (errors, then waived, then warnings with `finding-warning`), the fix section with `diff_scripts` and `diff_markup` or nothing, the model card section or "No approved run yet.", the raw-file links). The landing page comes in task 7.
  - Create `hexperiment/tests/test_gallery_pages.py` (the case page part): shell (`lang`, `charset` first, viewport, title, skip link first in the body, `header`, `main id="main"`, `footer`, no `preconnect`, the repository link and its label for `github.com` and for another host, absent without `--repo-url`), the status mark markup for the three statuses (word as the verdict writes it, mark `aria-hidden`), every field of a timeline entry, a run with no findings, a run without a score, a waived finding with its reason, a finding without a ghost showing its check name, "No approved run yet.", the identical-scripts sentence, `relative_href` examples, and a whole case page built from a source whose name, description, target, finding location, message, fix, waiver reason and script lines are the hostile strings of the coverage table (parsed with `html.parser`: only allowlisted start tags, no `on*` or `style` attribute, the text appears with controls replaced).
  - Add `pages.py` to `PURE_FILES`.
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.3, 5.1, 5.2, 1.8 (label)_
  - Tests added: about 30. None starts a Python process.
  - Verify: `.venv/Scripts/python -m pytest hexperiment/tests/test_gallery_pages.py hexperiment/tests/test_architecture.py -q`, then the two full commands.

- [x] 7. The landing page
  - In `gallery/pages.py` add `landing_page` and `hero_sentence`: the hero (haunted figure with `data-value`, the generated sentence, the honest score or "No approved run yet.", the motto as the only `h1`; without a `Hero` only the motto and "No run has been rejected yet."), the cases as an unnumbered `<ul class="cases">` with the summary items of `design.md`, the three steps as the page's only `<ol>` (at most 80 words with the heading), the benchmark section only when a `Benchmark` is given, the ghost catalogue as a `<dl>` through `prose`, and the "How it was built" paragraph.
  - Extend `hexperiment/tests/test_gallery_pages.py` with the landing part: the hero for the example shapes (rejected 1.0000 and approved 0.9096), the sentence from the error findings in verdict order with the suspect lines, a hero without error findings ("The audit rejected it."), no approved run, no rejected run, exactly one `h1` and it is the motto, the figure not in a heading, the cases are a `ul` and the page has exactly one `ol`, every summary field, the word count of the steps, every catalogue entry that has a ghost name with its three texts and code spans, the benchmark section absent without `Benchmark` and present with it, the repository link absent without `--repo-url`, and the same hostile-string site test on the landing page.
  - _Requirements: 1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 5.2_
  - Tests added: about 26. None starts a Python process.
  - Verify: `.venv/Scripts/python -m pytest hexperiment/tests/test_gallery_pages.py -q`, then the two full commands.

- [x] 8. Links and the site
  - Create `gallery/links.py` (pure): `page_links` and `broken_links`. Create `gallery/site.py` (pure): `MARKER_TEXT` and `build_site`, with the file mapping, the order of the insertions and the `RuntimeError` for a broken link, as in `design.md`.
  - Create `hexperiment/tests/test_gallery_site.py`: the file set for one app, two apps, an app with no approved run (no `model_card.md`); `build_site` twice gives equal dictionaries; two different `tmp_path` folders give equal bytes; no page contains the current year; UTF-8 without BOM and without `\r`; the independent link check of the coverage table on the three sites (its own `HTMLParser`, not `links.py`); the negative test for `broken_links` and for `build_site` raising `RuntimeError`; the balanced-tags check (stack, void tags, unique ids, one `main`, `header` and `footer` per page, one `h1`, a first focusable skip link with an existing target, `lang="en"`, distinct titles, no skipped heading level); the hostile site is parsed again here.
  - Add `links.py` and `site.py` to `PURE_FILES`.
  - _Requirements: 1.1, 4.1, 4.3, 4.4, 5.2, 8.1_
  - Tests added: about 24. None starts a Python process.
  - Verify: `.venv/Scripts/python -m pytest hexperiment/tests/test_gallery_site.py hexperiment/tests/test_architecture.py -q`, then the two full commands.

- [x] 9. Reading the apps
  - Create `gallery/sources.py` (edge): `read_app`, `read_stylesheet` and `read_benchmark`, with the five steps of `design.md`: `is_app`, `load_config` and `load_history`, the run-id pattern before any path is built, the last verdict and the last approved card read as bytes and passed through `normalize_bytes`, and `train.py` and `haunted/train.py` read as bytes, decoded as strict UTF-8 and passed through `normalize_text` before anything else. Never use `read_text` for these files. Nothing is opened for writing.
  - Create `gallery/assets/gallery.css` as a **placeholder** (one comment line and the six colour custom properties) so that `read_stylesheet` works; task 12 replaces it. Add `"hexperiment.gallery" = ["assets/*.css"]` to `[tool.setuptools.package-data]` in `hexperiment/pyproject.toml`.
  - Extend `hexperiment/tests/gallery_builder.py` with `write_app(root, name, runs, haunted=..., newline="\n")`, which writes `hex.toml`, `train.py`, the optional `haunted/train.py`, `runs/<id>/verdict.json` (through `verdict.to_json`) and `model_card.md` (through `card.render`) in binary mode; with `newline="\r\n"` the scripts, `verdict.json` and `model_card.md` get CRLF line endings.
  - Create `hexperiment/tests/test_gallery_sources.py`: every refusal of the error table that `read_app` owns (not an app folder, invalid `hex.toml`, no recorded run, damaged history file, a hostile run id such as `..\..\x` that never reaches a path, a missing run folder, a missing `model_card.md`, a script that is not UTF-8) with the message text and fix; the run-id pattern equals the one in `runs.py`; **a CRLF app and an LF app give equal `AppSource` values** (`train_py`, `haunted_py`, `last_verdict_json`, `approved_card`) and none contains `\r`; a lone-CR script is normalised; `read_stylesheet` returns LF bytes and the file is in the package data pattern (`PurePath.full_match` against `pyproject.toml`); `read_benchmark` for a missing file, a non-UTF-8 file and a valid file; the app folder is byte-identical (hash of every file) before and after.
  - _Requirements: 1.3, 1.6, 3.3, 3.4, 4.4_
  - Tests added: about 22. None starts a Python process.
  - Verify: `.venv/Scripts/python -m pytest hexperiment/tests/test_gallery_sources.py -q`, then the two full commands.

- [x] 10. Writing the site
  - Create `gallery/writer.py` (edge): `write_site(files, out_dir)` as the steps W1 to W4 of `design.md`, each step a small function so that tests can stop between them; the key defence (`ValueError` for a key with `..`, a drive or a leading `/`); the temporary names `.<name>.new-<token>` and `.<name>.old-<token>`; the retry of `PermissionError` on W2 and W3 (5 times, 0.2 seconds); the restore of the old gallery when W3 fails; the warning when W4 fails.
  - Create `hexperiment/tests/test_gallery_writer.py`: a new folder, an empty existing folder (accepted), a marked folder (replaced; files only the old gallery had are gone); files are written as given (bytes, binary mode) and sorted; the states of the "What is left" table built by calling the steps one by one (after W2 `OUT_DIR` is missing and `OLD` and `NEW` hold whole galleries; a new build then succeeds and leaves them alone); injected failures at W1, W2 and W3 (monkeypatched `os.rename` and `Path.write_bytes`) leave the previous gallery unchanged with no `.new-` or `.old-` folder, and a failure at W3 restores it; a failing restore names `OLD` and `NEW`; the retry with a patched `time.sleep`; a failure at W4 gives the warning and no error; bad keys raise `ValueError`; nothing outside `OUT_DIR` is touched (hash of the parent folder's other entries).
  - _Requirements: 1.4, 1.5_
  - Tests added: about 18. None starts a Python process.
  - Verify: `.venv/Scripts/python -m pytest hexperiment/tests/test_gallery_writer.py -q`, then the two full commands.

- [x] 11. The command, `cli.py` and the isolation test
  - Create `gallery/command.py` (edge): `build_gallery`, `validate_repo_url` and `check_out_dir` in the order of checks of `design.md`; every `HexInputError` and `OSError` becomes `RunOutcome(2, None, "", "hexperiment: <message>")`, any other `Exception` an internal error with a traceback; success returns the path of `index.html` and the writer's warning, if any.
  - In `hexperiment/src/hexperiment/cli.py` add the `gallery` subparser to `build_parser`, `_gallery_command` (the import of `hexperiment.gallery.command` inside the function) and `_emit`; make the `run` command use `_emit` without changing its output or exit codes. Do not edit any existing test.
  - Create `hexperiment/tests/test_gallery_command.py`: success (exit 0, standard output is the path of `index.html`, the files of Requirement 1.1 exist, the repository link on both pages and absent without the option, `--benchmark` section present with a valid file); every refusal of Requirement 1 and 6.3 as one parametrised test (not an app folder, no recorded run, the same name twice, the same folder twice, empty slug, `assets`, a device name, `OUT_DIR` non-empty without the marker, inside an app folder, holding an app folder, a file, a link where the platform allows it, a missing parent, bad `--repo-url` (`javascript:`, no host, whitespace, quote), a benchmark file that is wrong or has `complete` false): exit 2, one line on standard error that starts with `hexperiment: `, nothing on standard output, and the file tree under `tmp_path` unchanged (hash before and after); an existing marked gallery unchanged after each refusal; `argparse` errors give 2; **determinism**: the same app built once with LF and once with CRLF scripts, `verdict.json` and `model_card.md` (`write_app(..., newline=...)`) through `build_gallery` gives two output trees with the same paths and the same bytes and no `\r` in any file; the app folders are unchanged and `test_set_uses` did not change (Requirement 1.6).
  - In `hexperiment/tests/test_architecture.py` add: `cli.py` has no module-level import of `hexperiment.gallery` (checked on `tree.body`); no gallery module calls `print`; no gallery module imports `pandas`, `sklearn`, `scipy` or `mcp`; and the **import-isolation test** (one new process): write a one-run app with `write_app`, start a fresh interpreter that calls `build_gallery` on it, and assert that `sklearn`, `scipy`, `mcp` and `anyio` are not in `sys.modules` afterwards.
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 4.4, 4.5, 6.3, 8.1, 8.3_
  - Tests added: about 34, of which **one starts a Python process** (the isolation test, about 2 seconds).
  - Verify: `.venv/Scripts/python -m pytest hexperiment/tests/test_gallery_command.py hexperiment/tests/test_architecture.py hexperiment/tests/test_cli.py -q`, then the two full commands.

- [x] 11a. Plain language for readers outside machine learning
  - In `gallery/pages.py` add `METRIC_RANGE`, `GHOST_GLOSS` and `EXPLAINER`, change `hero_sentence` to the plain-language version and add the `explain` paragraph after the motto, exactly as in "Plain-language version" of `design.md` (under the landing page's hero).
  - Update the tests in `hexperiment/tests/test_gallery_pages.py` that pin the old sentence, and add: the sentence for the example shapes (first run and a later run; with and without a gloss; a metric without a range entry); every catalogue ghost has a gloss of at most eight words; the explainer is at most 70 words, follows the `h1` and appears in the hero without a `Hero`; a hostile ghost name or location in the sentence is escaped.
  - Add the class `explain` to the classes the stylesheet test of task 12 expects.
  - _Requirements: 2.1, 2.6, 2.8, 2.9_
  - Tests added: about 8. None starts a Python process.
  - Verify: `.venv/Scripts/python -m pytest hexperiment/tests/test_gallery_pages.py hexperiment/tests/test_gallery_site.py -q`, then the two full commands.

- [x] 12. The stylesheet
  - Replace the placeholder `gallery/assets/gallery.css` with the stylesheet described in "The stylesheet and the haunted number" of `design.md` and in `.kiro/steering/gallery-design.md`: custom properties for the six colours and the spacing steps, base and focus (`2px` Paper outline, `2px` offset), skip link, one-column layout (`72rem`, 16px gutter, 32px from 768px), headings and scale, the cases list (two columns with one vertical Faded cyan rule from 768px), the record rail, status marks (Lantern square with `4px` corners, Verdigris circle, dashed hollow Faded cyan circle), findings, tables and `.scroll`, `.diff` (Plex Mono 14px on Deep print, left bars, side by side from 1024px), `.haunted` with the `::after` ghost copy, `.honest`, and the reduced-motion rule that wraps the only `@keyframes`. Exactly the six colours, no gradient, no shadow, no `transition`, no `@import`, no `url()` to another origin, no `text-transform`.
  - Create `hexperiment/tests/test_gallery_assets.py`: the file ships as package data (pattern in `pyproject.toml`); the hex colours are exactly the six of the steering file; none of `gradient`, `box-shadow`, `text-shadow`, `transition`, `@import`, `text-transform`; the focus rule; `prefers-reduced-motion` wraps the only `@keyframes` and the base `.haunted::after` rule already has the offset; `overflow-x: auto` on `.diff` and `.scroll`; the `min-width` queries 768px and 1024px; the font stacks; `font-variant-numeric` on figures; the status mark rules; the contrast of the six pairs on Prussian and on Deep print computed from the hex values with the WCAG formula, at least 4.5; the class names used by `pages.py` and `difftext.py` (collected from a built site) all have a rule or are listed as structural in the test.
  - _Requirements: 1.1, 4.2, 5.1, 5.2, 5.3, 5.4_
  - Tests added: about 16. None starts a Python process.
  - Verify: `.venv/Scripts/python -m pytest hexperiment/tests/test_gallery_assets.py hexperiment/tests/test_gallery_sources.py -q`, then the two full commands.

- [x] 12a. Fonts and art
  - Read "Redesign of the pages" in `design.md` first; it supersedes the older sections it names. The reference pages are `.kiro/specs/hexperiment-gallery/mockups/landing.html` and `case.html`.
  - In `hexperiment/pyproject.toml`, set `"hexperiment.gallery" = ["assets/*.css", "assets/fonts/*"]`. The font files are already in `gallery/assets/fonts/`; do not change them.
  - Add `read_fonts()` to `gallery/sources.py`, the `fonts` field to `GalleryInputs` in `gallery/records.py`, the font files to `build_site` in `gallery/site.py` and `fonts=read_fonts()` to `_build` in `gallery/command.py`, as in "Fonts plumbing".
  - Create `gallery/art.py` with the constants of the table in "`art.py`", copied from the reference pages character for character, and add `"art.py"` to `PURE_FILES` in `hexperiment/tests/test_architecture.py`.
  - In `gallery/markup.py`, add `ART_TAGS` and add `strong` to `TAGS`.
  - Tests: create `hexperiment/tests/test_gallery_art.py`; extend `test_gallery_markup.py`, `test_gallery_sources.py`, `test_gallery_site.py` and `test_gallery_command.py`; and add `assets/fonts/*` to the package-data test in `test_gallery_assets.py`. All as listed in "Tests" of the redesign section, except the rules that belong to the later tasks: the stylesheet rules (12b), and the `test_gallery_site.py` check for `http` links and `fonts.googleapis.com` (12c, because the pages link Google Fonts until then).
  - _Requirements: 1.1, 4.2_
  - Tests added: about 14. None starts a Python process.
  - Verify: `.venv/Scripts/python -m pytest hexperiment/tests/test_gallery_art.py hexperiment/tests/test_gallery_markup.py hexperiment/tests/test_gallery_sources.py hexperiment/tests/test_gallery_site.py hexperiment/tests/test_gallery_command.py -q`, then the two full commands.

- [x] 12b. The stylesheet
  - Rewrite `gallery/assets/gallery.css` as in "Stylesheet" of the redesign section: the two `@font-face` rules, the CSS of both reference pages merged, and the extra rules listed there.
  - Rewrite `hexperiment/tests/test_gallery_assets.py` for the rules of "Stylesheet" and the assets test of "Tests" in the redesign section. The pages still use some classes of the first design until the next two tasks: list exactly those in a frozenset `OLD_PAGE_CLASSES` in the test, with the comment "Classes of the first page design, removed when the pages are rebuilt.", and let the class check skip them. Do not add rules for them to the stylesheet.
  - _Requirements: 4.2, 5.1, 5.2, 5.4, 5.5_
  - Tests added: about 18 (replacing the old stylesheet tests). None starts a Python process.
  - Verify: `.venv/Scripts/python -m pytest hexperiment/tests/test_gallery_assets.py -q`, then the two full commands.

- [x] 12c. The landing page
  - In `gallery/pages.py`, rebuild the shell and the landing page as in "Shell" and "Landing page" of the redesign section; remove `FONTS_URL`, `METRIC_RANGE`, `GHOST_GLOSS`, `EXPLAINER`, `hero_sentence`, `CHECK_STEPS` and `HOW_IT_WAS_BUILT`; add `GHOST_LINE` and the fixed copy; change `MOTTO` to use U+2019. Add `canary_counts` to `gallery/summary.py`. In `gallery/links.py`, remove the import and use of `FONTS_URL`: the only external link a page may hold is now the repository link.
  - Since the shell no longer writes `<header>`, `<main>` and `<footer>`, the case page writes them itself in this task, around its current content: its current header, then `<main>` holding its current sections, with `id="content"` and `tabindex="-1"` on the element that holds its `h1`, then its current footer. The new case-page markup is the next task.
  - Tests: in `test_gallery_pages.py`, remove the tests of the removed names and add the landing-page and shell tests of "Tests" in the redesign section; add the `canary_counts` tests to `test_gallery_summary.py`; add the `http` and `fonts.googleapis.com` check to `test_gallery_site.py`. Remove from `OLD_PAGE_CLASSES` in `test_gallery_assets.py` every class the landing page no longer uses.
  - _Requirements: 2.1 to 2.11, 5.2_
  - Tests added: about 30. None starts a Python process.
  - Verify: `.venv/Scripts/python -m pytest hexperiment/tests/test_gallery_pages.py hexperiment/tests/test_gallery_summary.py hexperiment/tests/test_gallery_site.py hexperiment/tests/test_gallery_assets.py -q`, then the two full commands.

- [x] 12d. The case page
  - In `gallery/pages.py`, rebuild the case page as in "Case page" of the redesign section (head with scores and facts, record with run head, run meta and findings, the fix, the model card with the raw-files line, the end block). In `gallery/modelcard.py`, regroup the section as described there, without changing its content.
  - Tests: add the case-page tests of "Tests" in the redesign section to `test_gallery_pages.py` and update the ones that pin the old case markup; update `test_gallery_modelcard.py`. Delete `OLD_PAGE_CLASSES` and its use from `test_gallery_assets.py`: every class of the built site must now have a rule.
  - _Requirements: 3.1 to 3.7, 5.2_
  - Tests added: about 16. None starts a Python process.
  - Verify: `.venv/Scripts/python -m pytest hexperiment/tests/test_gallery_pages.py hexperiment/tests/test_gallery_modelcard.py hexperiment/tests/test_gallery_assets.py hexperiment/tests/test_gallery_site.py -q`, then the two full commands.

- [ ] 13. The gallery of the two example apps
  - Create `hexperiment/tests/test_gallery_examples.py`: build the gallery of `apps/hotel-cancellations` and `apps/flight-delays` with `read_app`, `build_site` and `write_site` into `tmp_path`. Read the expected values straight from `runs/*/verdict.json` with `json` and a plain sort on `created_at`, not with the code under test: the first and last status and score of each app and the ghost names of its error findings. Check that the landing page and each case page state them; that the hero is the app with the highest first rejected score and shows the last approved score of that app; that each timeline has four entries in order; that the diff appears on both case pages; that the copied `verdict.json` equals the last run's file byte for byte after `\r\n` is turned into `\n` (with the test's own `replace`, not `newlines.py`, because a Windows checkout may have CRLF); that the model card of the last approved run is shown; and that the hash of the two app folders is the same before and after (Requirement 1.6). Skip nothing: both apps are committed.
  - _Requirements: 8.2, 1.6_
  - Tests added: about 8. None starts a Python process.
  - Verify: `.venv/Scripts/python -m pytest hexperiment/tests/test_gallery_examples.py -q`, then the two full commands.

- [ ] 14. The Pages workflow test
  - Create `hexperiment/tests/test_pages_workflow.py`. It reads the committed `.github/workflows/pages.yml` as text (no YAML dependency) and does **not** skip: a missing file is a failure whose message names the path. It checks everything listed for it in "Other tests" of `design.md`: the 40-character SHAs with a version comment on every `uses:`; `persist-credentials: false` after `actions/checkout`; the top-level `permissions` block holding only `contents: read`; no `permissions` key in the `build` job; the `deploy` job with `needs: build`, an `if:` that contains `github.event.repository.private`, and exactly `pages: write` and `id-token: write`; the folded `run: >-` command (lines joined with one space) equal to `hexperiment gallery apps/hotel-cancellations apps/flight-delays --out site --repo-url "$REPO_URL"`, plus ` --benchmark hexperiment/benchmark/results.json` when that file exists in the repository; `pip install` without `-e`; no `secrets.`; the triggers `push` on `main` and `workflow_dispatch` and no `pull_request_target`. The block-splitting helpers live in the test file; negative tests feed them a copy of the text with one defect each (a tag instead of a SHA, `-e`, a write permission in `build`, a missing `if`, a missing `persist-credentials`).
  - Do not edit `.github/workflows/pages.yml`. If the test finds a defect in it, report the defect; the fix is an owner step (task 16).
  - _Requirements: 7.1, 7.2, 7.3, 8.3_
  - Tests added: about 16. None starts a Python process.
  - Verify: `.venv/Scripts/python -m pytest hexperiment/tests/test_pages_workflow.py -q`, then the two full commands.

- [ ] 15. Final verification
  - Run the whole suite with `.venv/Scripts/python -m pytest hexperiment/tests -q --durations=10` and confirm it passes in under 2 minutes in everyday use (the definition of `hexperiment-kiro`, Requirement 7.4). Report the time and the ten slowest tests. Do not lower Hypothesis settings or add a separate stage.
  - Run `.venv/Scripts/python -m ruff check hexperiment` and confirm it reports no problems.
  - Build the real gallery once by hand: `.venv/Scripts/hexperiment gallery apps/hotel-cancellations apps/flight-delays --out <a temporary folder outside the repository> --repo-url https://github.com/example/example`; confirm exit code 0, that standard output is the path of `index.html`, that the folder holds `index.html`, `assets/gallery.css`, the four files of `assets/fonts/`, `.hexperiment-gallery` and one folder per app, and delete the temporary folder. Confirm that `hexperiment gallery --help` prints usage, that the runtime dependencies are still `pandas`, `numpy` and `scikit-learn`, and that `git status` shows no change under `apps/`.
  - Report anything the tests cannot show (layout, colour, motion) as owner checks for task 17.
  - _Requirements: 4.5, 8.3, 8.4_
  - Tests added: none.
  - Verify: the commands above.

- [ ] 16. **Owner:** the workflow on GitHub
  - The builder cannot do this: it is under `.github/` and needs the hosted service. The workflow file is already committed.
  - Confirm that `hexperiment/benchmark/results.json` exists; once it does, edit the build step of `.github/workflows/pages.yml` to add `--benchmark hexperiment/benchmark/results.json` (task 14's test fails until you do). Commit the change.
  - Push to `main`. Open the Actions tab and confirm that `build` succeeds and that `deploy` is **skipped** while the repository is private (Requirement 7.2).
  - Once the repository is public (GitHub Free offers Pages only for public repositories), set Settings, Pages, Source to "GitHub Actions", run the workflow from "Run workflow" and confirm that `deploy` publishes the site at `https://<user>.github.io/<repo>/`.
  - Open the published address and confirm that the landing page, both case pages, `verdict.json` and `model_card.md` load from the sub-path (Requirement 4.3), and that the "Source on GitHub" link opens the repository.
  - _Requirements: 7.1, 7.2, 7.3_

- [ ] 17. **Owner:** check the built gallery in a browser
  - The builder cannot do this: the tests prove that the rules exist, not how the pages look. Build the gallery (`hexperiment gallery apps/hotel-cancellations apps/flight-delays --out site`) and open `site/index.html`, then both case pages, in a desktop browser. Compare them with the reference pages in `.kiro/specs/hexperiment-gallery/mockups/`.
  - At **1366×657, 1440×900 and 1831×843**: every screen of the landing page fits the window with nothing cut off; the moon and tree stay clear of the text; there is no band or seam between the first and second screen; scrolling settles gently on each screen.
  - **Motion on:** the ghost rises out of the claimed score and bobs twice, and is still within 5 seconds; while scrolling, each screen fades in, the leak ghost slips through, the rows turn orange, and each screen at rest shows its final state. **Reduced motion on** (the operating system setting, or the browser's rendering emulation): nothing moves and every screen shows its final state.
  - **At 360px wide:** no horizontal page scroll; the diff and the tables scroll inside their own box and can be reached with the keyboard.
  - **Stylesheet disabled:** every page reads in order: the question, the definition, the numbers with their labels, every screen's text, the cases, the record with its run numbers, the findings, the diff lines with their `-` and `+` signs and the model card.
  - Also check: Tab first shows the skip link, then a visible 3px lime focus outline on each link; the fonts load from `assets/fonts/` (the network panel shows no request to another site); status words are always written next to their marks.
  - Report any defect to the builder as a change to the stylesheet or the pages; do not edit the generated files by hand.
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
