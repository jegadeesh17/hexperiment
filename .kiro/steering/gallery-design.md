---
inclusion: fileMatch
fileMatchPattern: "hexperiment/src/hexperiment/gallery/**"
---

# Gallery design direction

The gallery is the public face of Hexperiment. Hackathon judges and developers open it from a link on a desktop screen and decide within one screen whether the kit is real. Most of them do not work in machine learning. The page has to tell them what Hexperiment is, show the numbers that prove it, and then walk them through the problem, the method, the proof and the Kiro integration, one screen at a time.

The reference pages `.kiro/specs/hexperiment-gallery/mockups/landing.html` and `mockups/case.html` are the approved design. They are normative for structure, class names, art and CSS. This file says why they look the way they do, so that later changes stay in the same world.

## Concept: a moonlit night

- **World.** One continuous night sky in deep purple. The first screen is a landscape: a pale moon, a bare tree, a far hill, a few bats. The ground of that landscape has the exact colour of the page, so the first screen flows into the rest with no seam. Halloween comes from the night, the ghost and the silhouettes, never from pumpkins, cobwebs, dripping letters, neon or puns.
- **Story.** The first screen asks a question ("Is your model too good to be true?"), says in one sentence what Hexperiment is, and shows up to three real numbers from a recorded run: the score a leaky model claimed, how many training rows changed when only the test rows were scrambled, and the honest score. Each later screen answers one question a judge has: what is the problem, how does the check work, what does it find, does it work on real apps, how does it use Kiro, what does it do about a leak.
- **The one bold element: the rising ghost.** A small ghost rises once out of the claimed score on the first screen. It is the only motion that starts on its own, and it ends within five seconds.

## Tokens

### Colour

| Token | Hex | Use |
|---|---|---|
| `--sky` | `#23153A` | Page background on every page and every screen, and the near ground of the landscape. |
| `--night` | `#0B0612` | Silhouettes and the darkest ground. |
| `--well` | `#1A0F2A` | The code diff background on the case page. |
| `--bone` | `#EDE6F5` | Headings, text that must stand out, the moon and the ghost. |
| `--soft` | `#D2C8E1` | Body text. |
| `--muted` | `#BDB2CF` | Secondary text: labels, keys, step numbers, meta lines. |
| `--dash` | `#7A6893` | Decorative dashes and dashed outlines only, never text. |
| `--rows` | `#8A5CCB` | Training rows in the drawings, never text. |
| `--orange` | `#FF8A1F` | A leak: the claimed score, REJECTED, ghost names in findings, leaked rows. |
| `--orange-soft` | `#FFC08A` | Removed lines in the diff. |
| `--lime` | `#A6FF4D` | An honest result: the honest score, APPROVED, the call-to-action, keyboard focus. |
| `--lime-soft` | `#C9FF94` | Added lines in the diff. |
| `--rule` | `rgba(237, 230, 245, 0.16)` | Hairline rules between items. |

- Orange always means "leak or rejected" and lime always means "honest or approved". Do not use them for anything else.
- The SVG drawings use their own shades of the same night (the sky gradient `#150B24` to `#2D1A48`, the far hill `#2A1843`, the back bats `#4E3769`, the stones `#3A2852`, the fence `#160D24`, the moon craters `#DCD2E8`, the ghost's eyes `#170D27`). Those colours live only in the SVG markup, never in the stylesheet.
- Every text colour meets WCAG 2.2 AA on `--sky` and on `--well`. No gradients in the stylesheet, no glows, no box or text shadows.

### Type

- **Special Elite** (a typewriter face, Apache License 2.0) for headings, the large numbers and the wordmark. It gives the case-file voice. Fallback: `"Courier New", serif`.
- **Archivo** (variable weight 100 to 900, SIL Open Font License 1.1) for body text, labels and links. Fallback: `system-ui, sans-serif`.
- The system monospace (`ui-monospace, "Cascadia Mono", Consolas, monospace`) only for code: the diff, file paths, column names, run ids and commands.
- Both fonts are self-hosted: the latin `woff2` files and their licences ship in `gallery/assets/fonts/` and are copied to `assets/fonts/` in the site. `@font-face` uses `font-display: swap`. The pages make no request to another origin.
- Sizes scale with the height of the screen as well as its width (`clamp(min, min(vw, vh), max)`), so a page never looks zoomed on a short laptop screen. The first-screen question is the largest text.
- Figures use `font-variant-numeric: tabular-nums lining-nums` wherever scores, counts and times appear.
- Headings are sentence case. Status words are written as the verdict writes them (`REJECTED`, `APPROVED`, `UNVERIFIED`): they are values, so no `text-transform` is used anywhere.
- Apostrophes in page copy are typographic (U+2019).

### Layout

- **Page-wise.** Every section is one screen (`.page`, `min-height: 100svh`) with its content centred vertically. On a screen at least 861px wide and 600px tall, scrolling snaps gently to each screen (`scroll-snap-type: y proximity`). Below that, the pages simply follow each other.
- Content width up to about 72rem, with the gutter `clamp(16px, 5vw, 64px)` and the page padding `clamp(56px, 9vh, 96px)`. Text is left-aligned; prose is at most about 68 characters wide.
- No cards, no boxes with shadows, no rounded panels. Groups are separated by space and one hairline rule.
- Every page fits one screen at 1366×657, 1440×900 and 1831×843 with no content cut off. From 360px wide there is no horizontal page scroll; the diff and wide tables scroll inside their own region.

## Art

All art is inline SVG, written once as string constants in `gallery/art.py`:

- the **sprite**: the symbols `bat`, `ghost`, `tree` and `stone`, defined once per page and drawn with `<use>`;
- the **landscape** of the first screen (sky, moon, far hill, tree, near ground, two layers of bats), anchored to the bottom right (`xMaxYMax slice`) so the moon and tree stay clear of the text at every width;
- the **leak figure** of the problem screen: training rows, the split, test rows, and a ghost slipping through;
- the **rows** of the canary steps: eight training rows and two test rows as small bars;
- the **peeking ghost** at the edge of the ghost screen;
- the **corner moon** of the case page;
- the **graveyard** at the foot of the last screen of both pages: fence, tree, stones, a small ghost.

Every drawing is decorative (`aria-hidden="true"`) except the leak figure, which has `role="img"` and an `aria-label` that says what it shows. The pages read completely without the art.

## Motion

- **On load (the only motion that starts by itself):** the ghost rises out of the claimed score (1.4 s, after 0.9 s), then bobs twice (1.3 s each). Everything ends 4.9 s after load (WCAG 2.2.2: nothing moves on its own for more than 5 seconds).
- **Linked to scrolling** (inside `@supports (animation-timeline: view())`): the content of each page fades in as the page arrives and out as it leaves; on the problem screen the ghost slips through the split and three training rows turn orange; on the canary screen the test rows rattle and the changed training rows turn orange; the ghost on the ghost screen peeks in. Each of these finishes before its page is fully in view, so a page at rest always shows its final state.
- Browsers without scroll-linked animation show the final state. With `prefers-reduced-motion: reduce`, nothing moves and smooth scrolling is off.
- No bats fly. No motion on hover beyond a colour change. No `transition`.

## Components

- **Number trail** (first screen): up to three large numbers joined by short dashed lines, each with a one-line label under it. Claimed score in orange, rows changed in bone, honest score in lime.
- **Status mark:** a small shape before the status word. REJECTED: a filled orange square. APPROVED: a filled lime circle. UNVERIFIED: a hollow dashed circle in `--dash`, word in `--muted`, because it means "not checked yet", not a third kind of result. The word is always written; the mark is `aria-hidden`.
- **Score pair** (cases screen and case page head): first score, a dash or arrow, last score, each with a small label that names the run and its status in words, so colour is never the only signal. A score is orange when its run was rejected, lime when approved and muted when unverified.
- **Finding:** what (the ghost name in orange italics, or the check name) and where (in code), a small severity pill, the message, then the fix. Warnings are set quieter than errors.
- **Call-to-action:** at most one lime button-like link per screen ("Open the case file"). Other links are underlined text.
- **Focus:** a 3px lime outline with a 3px offset on every link; on the lime call-to-action the outline is bone.

## Copy

- Plain, specific sentences from the reader's side, for developers who do not work in machine learning. Say what a thing does, not how clever it is.
- Every number on a page comes from a recorded run or from the benchmark file. Fixed copy may describe how Hexperiment works; it never states a result.
- Not allowed: all-caps labels, small labels above headings, meta strings joined with middle dots, an arrow appended to link text, one word of a heading set apart in another colour or style, exclamation marks, puns beyond the ghost names.
- Not allowed in any copy: "incentive to cheat", "proves an AI cheated", "certified honest", invented figures, fines or customer names.
- Scores keep four decimals, as in the verdicts. Row counts use thousands separators.

## Quality floor

- WCAG 2.2 AA: contrast, visible focus, a skip link, landmarks, `lang="en"`, one `h1` per page, a distinct title per page, status never by colour alone.
- Static HTML and CSS only. No script, no `style` attribute, no request to another origin.
- Readable with the stylesheet off: every page reads in order as plain text.
