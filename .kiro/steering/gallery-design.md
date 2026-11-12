---
inclusion: fileMatch
fileMatchPattern: "hexperiment/src/hexperiment/gallery/**"
---

# Gallery design direction

The gallery is the public face of Hexperiment: judges and ML engineers open it from a link, often on a phone, and decide in a few seconds whether the kit is real. It must look like evidence, not like a Halloween party page.

## Concept: the cyanotype case file

- **Material.** Cyanotype prints (the 1840s Prussian-blue photographs in which objects appear as pale ghosts) give the palette: a mid-dark Prussian blue page, pale paper-white type. It is Halloween-flavoured through its subject (ghosts, prints, spirit photography), never through pumpkins, bats, cobwebs, dripping fonts or neon.
- **Structure.** Each app is a case: what the model claimed, which ghosts the audit found and where, how the code was fixed, and the clean verdict at the end. The history is a real sequence, so it is shown as an ordered timeline.
- **The one bold element: the haunted number.** The score that was too good to be true (for example `1.0000`) is set very large, as a double exposure: the figure plus a faint, slightly offset, blurred second copy of itself, like a ghost in a spirit photograph. The honest score next to it is crisp, with no ghost. Everything else on every page stays quiet so that this is what people remember.

## Tokens

### Colour

| Name | Hex | Use |
|---|---|---|
| Prussian | `#17406D` | page background |
| Deep print | `#123359` | raised areas: code diff, tables, the timeline rail |
| Paper | `#F1F6F9` | body text, headings, the haunted number |
| Faded cyan | `#A9C4D8` | secondary text, rules, captions |
| Lantern | `#F4B04F` | REJECTED, removed diff lines, ghost names |
| Verdigris | `#8FE0C4` | APPROVED, added diff lines |

- All six meet WCAG 2.2 AA on Prussian and on Deep print (paper about 9.7:1, faded cyan about 5.8:1, lantern about 5.6:1, verdigris about 6.8:1). Do not add colours; use opacity of Paper for the ghost copy only.
- No gradients, no glows, no drop shadows. Depth comes from Deep print areas and 1px Faded cyan rules.
- UNVERIFIED has no colour of its own: it is Faded cyan with a dashed outline, because it is "not checked yet", not a third kind of result.

### Type

- **Spectral** (Google Fonts, weights 200, 400, 600 and italic 400) for everything that is not code. Its name is a happy accident; it was chosen because it is a screen serif with clear figures. Fallback: `Georgia, "Times New Roman", serif`.
- **IBM Plex Mono** (400, 500) only for code: the diff, file paths, column names and commands. Never for labels, numbers or metadata. Fallback: `ui-monospace, Consolas, monospace`.
- Load both from Google Fonts with `display=swap`; the pages must read correctly with the fallbacks.
- Scale (px, on a 16px root): 14, 16, 18 (body), 21, 26, 34, 48, and the hero figure at `clamp(5rem, 18vw, 12rem)`. Body 18px with line height 1.6; headings line height 1.15. Measure at most 68 characters for prose.
- Figures: `font-variant-numeric: tabular-nums lining-nums` wherever scores, counts and times appear, so columns of numbers align.
- Headings are sentence case, weight 400 (h1 weight 200 at 48px). Ghost names are set in Spectral italic 400 in Lantern; they are names, like species.

### Layout and spacing

- One column, max width 72rem, left-aligned text; 16px side gutter on phones, 32px from 768px. Spacing steps: 4, 8, 16, 24, 40, 64, 104px.
- Corners: 0 for rules and tables, 4px only for the diff block and status marks. No cards with shadows.
- Works from 360px wide with no horizontal page scroll; the diff and wide tables scroll inside their own box.

## Pages

### Landing page (`index.html`)

```
Hexperiment                                         Source on GitHub
-----------------------------------------------------------------------
        1.0000      <- haunted number, double exposure, very large
  This hotel-cancellation model scored a perfect ROC AUC. The audit
  found a column recorded after the outcome and a scaler that had seen
  the test rows. Its honest score is
        0.9096      <- crisp, about a third of the size, Verdigris rule
  If your model looks too good to be true, it's haunted.
-----------------------------------------------------------------------
 Two cases                 | 
 hotel-cancellations       | flight-delays
 REJECTED on its first run | REJECTED on its first run
 The Possession, The       | The Prophet
 Peeking Spirit            |
 from 1.0000 to 0.9096     | from 0.7971 to 0.6193
 4 recorded runs           | 4 recorded runs
 Read the case             | Read the case
-----------------------------------------------------------------------
 How the check works   (three numbered steps: a real sequence)
 1 Train normally   2 Train again with the test rows scrambled   3 Compare
-----------------------------------------------------------------------
 Benchmark table (only when results are given)
-----------------------------------------------------------------------
 The ghosts: definition list of the catalogue (name, what it is, fix)
-----------------------------------------------------------------------
 How it was built (Kiro specs, hooks, workflow, power) | MIT licence
```

- The hero uses the app whose first REJECTED score is the highest (the most too-good-to-be-true number), with that app's last APPROVED score as the honest one; the sentence under it is written from that run's findings.
- The two cases sit side by side from 768px, divided by a single vertical Faded cyan rule, and stack below that. They are not cards.

### Case page (`<app>/index.html`)

```
All cases
hotel-cancellations
Predict whether a hotel booking will be cancelled.
target is_canceled, metric roc_auc, threshold 0.85, 4 recorded runs
-----------------------------------------------------------------------
 The record (ordered timeline, rail on the left)
 1 | REJECTED   full   1.0000   The Possession on reservation_status
   |                            The Peeking Spirit, suspect train.py:100
 2 | UNVERIFIED fast   0.9090
 3 | APPROVED   full   0.9090
 4 | APPROVED   full   0.9096
-----------------------------------------------------------------------
 The fix: haunted/train.py against train.py (side by side from 1024px,
 unified below; removed lines Lantern, added lines Verdigris)
-----------------------------------------------------------------------
 Model card of the last approved run (rendered from the verdict)
 Raw files: verdict.json, model_card.md
```

- Each timeline entry shows status (word plus mark), mode, score, the date and time in UTC, and every finding: ghost name or check name, location, and the fix hint. Warnings are listed under errors in Faded cyan.

## Components

- **Status mark.** A small shape before the word: filled square in Lantern for REJECTED, filled circle in Verdigris for APPROVED, dashed hollow circle in Faded cyan for UNVERIFIED. The word is always written out; colour is never the only signal.
- **Haunted number.** The figure is real text. The ghost copy comes from a pseudo-element reading a `data-value` attribute, is `aria-hidden` by construction, offset by about `0.05em 0.03em`, blurred 2px, Paper at 30% opacity.
- **Code diff.** Plex Mono 14px on Deep print, line numbers in Faded cyan, removed lines marked with a Lantern left bar and a minus sign, added lines with a Verdigris left bar and a plus sign, so it also reads in greyscale.

## Motion

Exactly one orchestrated moment: on the landing page, the ghost copy of the haunted number drifts from on top of the figure to its offset over about 1.6 s, once, after load. Nothing else animates; no hover transitions beyond a colour change and underline on links. With `prefers-reduced-motion: reduce` the ghost is shown at its offset with no movement.

## Copy

- Plain, specific sentences from the reader's side: "Read the case", "Source on GitHub", "Raw files". No puns in headings beyond the ghost names themselves.
- Not allowed: all-caps labels, small labels above headings, meta strings joined with middle dots, an arrow appended to link text, one word in a heading set apart in another colour or style, exclamation marks.
- Numbers keep four decimals for scores, as in the verdicts.

## Quality floor

Visible keyboard focus (2px Paper outline, 2px offset), skip link to the main content, landmarks (`header`, `main`, `footer`), `lang="en"`, a distinct `<title>` per page, alt text or `aria-label` for every non-text mark, and readable pages with CSS and JavaScript disabled.
