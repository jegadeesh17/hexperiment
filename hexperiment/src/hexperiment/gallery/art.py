"""The inline SVG art of the pages (pure).

Every constant is trusted markup written once here, copied from the approved
reference pages. ``literal`` only accepts string literals (an architecture test
checks it), so no app data can ever reach the art.
"""

# The art is copied character for character, so its lines keep their length.
# ruff: noqa: E501

from __future__ import annotations

from hexperiment.gallery.markup import literal

# The symbols drawn with <use>. Defined once per page.
SPRITE = literal("""<svg class="sprite" width="0" height="0" aria-hidden="true">
  <defs>
    <symbol id="bat" viewBox="0 0 64 30"><path d="M32 12 L29.6 6.5 L28.4 10.6 C23 4.5 12 3 1 8.5 C7 10 9.5 14 9.5 18.5 C13.5 15.5 18 15.8 20.5 19.5 C23.5 16.5 27.5 17.5 29.5 22 L32 25.5 L34.5 22 C36.5 17.5 40.5 16.5 43.5 19.5 C46 15.8 50.5 15.5 54.5 18.5 C54.5 14 57 10 63 8.5 C52 3 41 4.5 35.6 10.6 L34.4 6.5 Z"/></symbol>
    <symbol id="ghost" viewBox="0 0 100 120"><path fill="#EDE6F5" d="M50 4 C24 4 12 24 12 50 L12 104 C18 98 24 98 30 106 C36 114 42 114 50 104 C58 114 64 114 70 106 C76 98 82 98 88 104 L88 50 C88 24 76 4 50 4 Z"/><ellipse cx="38" cy="46" rx="6" ry="9" fill="#170D27"/><ellipse cx="62" cy="46" rx="6" ry="9" fill="#170D27"/><ellipse cx="50" cy="70" rx="5" ry="7" fill="#170D27"/></symbol>
    <symbol id="tree" viewBox="0 0 300 400" overflow="visible">
      <path fill="#0B0612" d="M100 400 C108 340 110 300 104 250 C100 215 106 185 100 160 L112 158 C120 190 118 220 126 252 C134 300 132 345 144 400 Z"/>
      <g fill="none" stroke="#0B0612" stroke-linecap="round">
        <path stroke-width="9" d="M112 245 C150 220 170 200 210 190"/>
        <path stroke-width="5" d="M170 205 C185 180 190 165 205 150"/>
        <path stroke-width="4" d="M195 193 C220 195 240 185 260 170"/>
        <path stroke-width="8" d="M104 215 C80 195 60 190 30 170"/>
        <path stroke-width="4" d="M60 188 C50 165 52 150 40 130"/>
        <path stroke-width="3" d="M40 177 C25 175 15 165 0 160"/>
        <path stroke-width="7" d="M106 165 C110 130 130 110 125 80"/>
        <path stroke-width="3.5" d="M125 95 C140 80 160 75 170 55"/>
        <path stroke-width="3.5" d="M122 110 C105 95 95 80 85 60"/>
        <path stroke-width="2" d="M170 55 C175 45 178 40 185 35 M85 60 C82 50 76 45 70 38 M260 170 C270 165 278 160 290 158 M205 150 C210 140 220 135 228 128 M40 130 C35 120 30 115 22 110 M125 80 C123 65 128 55 124 42"/>
      </g>
    </symbol>
    <symbol id="stone" viewBox="0 0 100 130"><path d="M4 130 L4 48 C4 20 26 4 50 4 C74 4 96 20 96 48 L96 130 Z"/></symbol>
  </defs>
</svg>""")

# The landscape of the first screen.
LANDSCAPE = literal("""<svg class="scene" viewBox="0 0 1600 900" preserveAspectRatio="xMaxYMax slice" aria-hidden="true">
    <defs><linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#150B24"/><stop offset="1" stop-color="#2D1A48"/></linearGradient></defs>
    <rect x="-3000" width="7600" height="900" fill="url(#sky)"/>
    <circle cx="1110" cy="330" r="92" fill="#EDE6F5"/>
    <circle cx="1080" cy="308" r="14" fill="#DCD2E8"/><circle cx="1136" cy="362" r="9" fill="#DCD2E8"/><circle cx="1142" cy="298" r="6" fill="#DCD2E8"/>
    <path fill="#2A1843" d="M-3000 770 L0 770 C300 715 600 745 900 725 C1200 705 1400 735 1600 715 L1600 900 L-3000 900Z"/>
    <use href="#tree" x="1170" y="290" width="435" height="580"/>
    <path fill="#23153A" d="M-3000 835 L0 835 C400 805 800 828 1200 808 C1400 798 1500 812 1600 805 L1600 900 L-3000 900Z"/>
    <g fill="#0B0612"><use href="#bat" x="1050" y="268" width="76" height="36"/><use href="#bat" x="1100" y="370" width="44" height="21"/></g>
    <g fill="#4E3769"><use href="#bat" x="1260" y="250" width="54" height="25"/><use href="#bat" x="930" y="225" width="36" height="17"/><use href="#bat" x="1330" y="300" width="34" height="16"/><use href="#bat" x="870" y="400" width="28" height="13"/></g>
  </svg>""")

# The small ghost that rises out of the claimed score.
GHOST_RISE = literal(
    """<span class="ghost-wrap" aria-hidden="true"><svg class="ghost-rise" viewBox="0 0 100 120"><use href="#ghost"/></svg></span>"""
)

# The leak figure of the problem screen: the one drawing with a text alternative.
LEAK_FIGURE = literal("""<div class="figure">
        <svg viewBox="0 0 520 340" role="img" aria-label="A ghost slips from the test rows through the split into the training rows.">
          <g fill="#8A5CCB">
            <rect x="20" y="40" width="200" height="14" rx="3"/><rect x="20" y="64" width="200" height="14" rx="3"/>
            <rect class="leak l1" x="20" y="88" width="200" height="14" rx="3" fill="#FF8A1F"/><rect x="20" y="112" width="200" height="14" rx="3"/>
            <rect class="leak l2" x="20" y="136" width="200" height="14" rx="3" fill="#FF8A1F"/><rect x="20" y="160" width="200" height="14" rx="3"/>
            <rect x="20" y="184" width="200" height="14" rx="3"/><rect class="leak l3" x="20" y="208" width="200" height="14" rx="3" fill="#FF8A1F"/>
            <rect x="20" y="232" width="200" height="14" rx="3"/><rect x="20" y="256" width="200" height="14" rx="3"/>
          </g>
          <g fill="none" stroke="#D2C8E1">
            <rect x="320.5" y="40.5" width="180" height="13" rx="3"/><rect x="320.5" y="64.5" width="180" height="13" rx="3"/>
            <rect x="320.5" y="88.5" width="180" height="13" rx="3"/><rect x="320.5" y="112.5" width="180" height="13" rx="3"/>
          </g>
          <line x1="270" y1="20" x2="270" y2="280" stroke="#7A6893" stroke-dasharray="6 6"/>
          <path d="M330 120 C320 170 300 175 285 160" fill="none" stroke="#EDE6F5" stroke-dasharray="3 6" stroke-linecap="round"/>
          <use class="leak-ghost" href="#ghost" x="226" y="112" width="58" height="70" transform="rotate(-18 255 147)"/>
          <g font-family="Archivo, sans-serif" font-size="16" fill="#BDB2CF">
            <text x="20" y="306">Training rows</text>
            <text x="320" y="306">Test rows: the answers</text>
            <text x="270" y="12" text-anchor="middle">the split</text>
          </g>
        </svg>
      </div>""")

# The rows of the three canary steps: trained, scrambled, compared.
ROWS_TRAIN = literal(
    """<div class="rows" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i class="t t1"></i><i class="t"></i></div>"""
)
ROWS_SCRAMBLED = literal(
    """<div class="rows" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i class="s t1"></i><i class="s"></i></div>"""
)
ROWS_CHANGED = literal(
    """<div class="rows" aria-hidden="true"><i class="m"></i><i class="m"></i><i></i><i class="m"></i><i class="m"></i><i class="m"></i><i></i><i class="m"></i><i class="s t1"></i><i class="s"></i></div>"""
)

# The ghost that peeks in from the edge of the ghost screen.
PEEK = literal(
    """<svg class="peek" viewBox="0 0 100 120" aria-hidden="true"><use href="#ghost"/></svg>"""
)

# The graveyard at the foot of the landing page (with two bats).
GRAVEYARD_LANDING = literal("""<svg class="graveyard" viewBox="0 0 1600 200" preserveAspectRatio="xMidYMax slice" aria-hidden="true">
      <defs><pattern id="fence" width="30" height="80" patternUnits="userSpaceOnUse"><path d="M11 80 L11 14 L15 4 L19 14 L19 80 Z" fill="#160D24"/></pattern></defs>
      <rect x="0" y="110" width="1600" height="80" fill="url(#fence)"/>
      <rect x="0" y="126" width="1600" height="5" fill="#160D24"/><rect x="0" y="156" width="1600" height="5" fill="#160D24"/>
      <use href="#tree" x="1280" y="-20" width="165" height="220"/>
      <g fill="#3A2852"><use href="#stone" x="210" y="120" width="54" height="70"/><use href="#stone" x="300" y="140" width="40" height="52"/><use href="#stone" x="1050" y="128" width="50" height="64"/></g>
      <use href="#ghost" x="246" y="96" width="34" height="41" transform="rotate(14 263 116)" opacity=".9"/>
      <g fill="#4E3769"><use href="#bat" x="700" y="40" width="40" height="19"/><use href="#bat" x="760" y="70" width="28" height="13"/></g>
      <path fill="#0B0612" d="M0 175 C300 165 600 172 900 166 C1200 160 1400 170 1600 164 L1600 200 L0 200Z"/>
    </svg>""")

# The moon in the corner of the case page head.
CORNER_MOON = literal("""<svg class="corner" viewBox="0 0 260 200" aria-hidden="true">
      <circle cx="150" cy="80" r="54" fill="#EDE6F5"/><circle cx="132" cy="66" r="8" fill="#DCD2E8"/><circle cx="168" cy="98" r="5" fill="#DCD2E8"/>
      <g fill="#0B0612"><use href="#bat" x="104" y="58" width="46" height="22"/></g>
      <g fill="#4E3769"><use href="#bat" x="214" y="40" width="32" height="15"/><use href="#bat" x="40" y="120" width="26" height="12"/></g>
    </svg>""")

# The graveyard at the foot of the case page (no bats).
GRAVEYARD_CASE = literal("""<svg class="graveyard" viewBox="0 0 1600 200" preserveAspectRatio="xMidYMax slice" aria-hidden="true">
      <defs><pattern id="fence" width="30" height="80" patternUnits="userSpaceOnUse"><path d="M11 80 L11 14 L15 4 L19 14 L19 80 Z" fill="#160D24"/></pattern></defs>
      <rect x="0" y="110" width="1600" height="80" fill="url(#fence)"/>
      <rect x="0" y="126" width="1600" height="5" fill="#160D24"/><rect x="0" y="156" width="1600" height="5" fill="#160D24"/>
      <use href="#tree" x="1280" y="-20" width="165" height="220"/>
      <g fill="#3A2852"><use href="#stone" x="210" y="120" width="54" height="70"/><use href="#stone" x="300" y="140" width="40" height="52"/><use href="#stone" x="1050" y="128" width="50" height="64"/></g>
      <use href="#ghost" x="246" y="96" width="34" height="41" transform="rotate(14 263 116)" opacity=".9"/>
      <path fill="#0B0612" d="M0 175 C300 165 600 172 900 166 C1200 160 1400 170 1600 164 L1600 200 L0 200Z"/>
    </svg>""")
