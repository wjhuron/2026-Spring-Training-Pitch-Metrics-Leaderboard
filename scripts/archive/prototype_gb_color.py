#!/usr/bin/env python3
"""Swatch comparison for the single-game batted-ball GROUND BALL color.

The current GB color (#00d4ff, neon cyan) was tuned for the old black card.
On warm paper it reads digital. This renders the current color next to a few
calmer blues, each shown the way they actually appear on a card: a donut + a
stacked-bar segment row beside the fixed LD / FB / PU colors, on warm paper.
Output: ~/Downloads/gb_color_compare.png
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

BG = '#f0e8d8'
TEXT_PRIMARY = '#1a1612'
TEXT_MUTED = '#6a5f55'
TRACK = '#d8ccb4'

# Fixed (unchanged) batted-ball colors
LD = '#FF6B6B'   # line drive  (coral)
FB = '#7B68EE'   # fly ball    (slate purple)
PU = '#FF9F43'   # popup       (amber)

# Ground-ball candidates
CANDS = [
    ('Current — Neon Cyan', '#00d4ff'),
    ('A · Steel Teal',      '#2E8FA8'),
    ('B · Cerulean',        '#3494C0'),
    ('C · Deep Teal',       '#1F7A8C'),
]


def lum(hexc):
    r, g, b = (int(hexc[i:i+2], 16) / 255 for i in (1, 3, 5))
    return 0.299 * r + 0.587 * g + 0.114 * b


def tcol(hexc):
    return '#1a1612' if lum(hexc) > 0.55 else '#ffffff'


fig = plt.figure(figsize=(11, 8.2), dpi=150)
fig.patch.set_facecolor(BG)
fig.text(0.5, 0.965, 'GROUND BALL color on warm paper', ha='center',
         fontsize=17, fontweight='bold', color=TEXT_PRIMARY)
fig.text(0.5, 0.93, 'GB shown with the fixed LD / FB / PU. Counts illustrative (GB 5 · LD 2 · FB 1 · PU 1).',
         ha='center', fontsize=10, color=TEXT_MUTED)

rows = len(CANDS)
top, bot = 0.88, 0.05
rh = (top - bot) / rows
for i, (name, gb) in enumerate(CANDS):
    y0 = top - (i + 1) * rh
    yc = y0 + rh / 2
    # label
    fig.text(0.04, yc, name, fontsize=12, fontweight='bold', color=TEXT_PRIMARY, va='center')
    fig.text(0.04, yc - 0.045, gb, fontsize=9, color=TEXT_MUTED, va='center', family='monospace')

    # donut
    axd = fig.add_axes([0.30, yc - 0.075, 0.13, 0.15]); axd.set_facecolor(BG)
    vals = [5, 2, 1, 1]; cols = [gb, LD, FB, PU]
    axd.pie(vals, colors=cols, startangle=90, counterclock=False,
            wedgeprops=dict(width=0.32, edgecolor=BG, linewidth=2.0))
    axd.text(0, 0, '9\nBIP', ha='center', va='center', fontsize=9,
             fontweight='bold', color=TEXT_PRIMARY, linespacing=1.1)

    # stacked-bar segment row
    axb = fig.add_axes([0.47, yc - 0.05, 0.49, 0.10])
    axb.set_xlim(0, 1); axb.set_ylim(0, 1); axb.axis('off')
    axb.add_patch(Rectangle((0, 0.30), 1.0, 0.40, facecolor=TRACK, edgecolor='none'))
    left = 0.0
    for cnt, c in zip(vals, cols):
        w = cnt / sum(vals)
        axb.add_patch(Rectangle((left, 0.30), w, 0.40, facecolor=c, edgecolor=BG, linewidth=0.8))
        axb.text(left + w / 2, 0.50, str(cnt), ha='center', va='center',
                 fontsize=10, fontweight='bold', color=tcol(c))
        left += w

out = os.path.expanduser('~/Downloads/gb_color_compare.png')
fig.savefig(out, dpi=150, facecolor=BG, bbox_inches='tight', pad_inches=0.2)
print('Saved:', out)
