#!/usr/bin/env python3
"""Differentiate Sinker (SI) vs Cutter (FC) pitch colors.

Current SI #E0A81E (gold) and FC #FFA500 (orange) share nearly the same hue,
so they blur together on the small movement-plot dots. This renders the current
pair next to a few options, each shown as badge chips + a dot cluster the way
they appear on the movement plot, on warm paper.
Output: ~/Downloads/si_fc_color_compare.png
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG = '#f0e8d8'
PANEL = '#e8dfcb'
TEXT_PRIMARY = '#1a1612'
TEXT_MUTED = '#6a5f55'

# Fixed dot positions (same layout reused per row so only color changes)
SI_PTS = [(-19, 5), (-18, 6.5), (-17.5, 4), (-16.5, 5.5), (-18.5, 3), (-17, 7), (-16, 4.5)]
FC_PTS = [(1, 6), (2.5, 5), (3.5, 6.5), (2, 4), (4, 5.5), (3, 3.5)]

OPTIONS = [
    ('Current',                 '#E0A81E', '#FFA500'),
    ('A · Cutter Brown',        '#E0A81E', '#8B5A2B'),
    ('B · Cutter Burnt Sienna', '#E0A81E', '#C2671C'),
    ('C · Yellow Sinker',       '#EAB308', '#FFA500'),
]


def lum(h):
    r, g, b = (int(h[i:i+2], 16) / 255 for i in (1, 3, 5))
    return 0.299 * r + 0.587 * g + 0.114 * b


def tcol(h):
    return '#1a1612' if lum(h) > 0.55 else '#ffffff'


fig = plt.figure(figsize=(12, 8.6), dpi=150)
fig.patch.set_facecolor(BG)
fig.text(0.5, 0.965, 'SINKER vs CUTTER — color separation', ha='center',
         fontsize=18, fontweight='bold', color=TEXT_PRIMARY)
fig.text(0.5, 0.925, 'Badges + movement-plot dot clusters on warm paper. SI cluster left, FC cluster right.',
         ha='center', fontsize=10.5, color=TEXT_MUTED)

rows = len(OPTIONS)
top, bot = 0.87, 0.04
rh = (top - bot) / rows
for i, (name, si, fc) in enumerate(OPTIONS):
    y0 = top - (i + 1) * rh
    yc = y0 + rh / 2
    fig.text(0.035, yc + 0.018, name, fontsize=13, fontweight='bold', color=TEXT_PRIMARY, va='center')
    fig.text(0.035, yc - 0.028, f'SI {si}   FC {fc}', fontsize=9, color=TEXT_MUTED,
             va='center', family='monospace')

    # badge chips
    axb = fig.add_axes([0.24, yc - rh * 0.30, 0.12, rh * 0.6])
    axb.set_xlim(0, 1); axb.set_ylim(0, 1); axb.axis('off')
    for cx, lab, c in [(0.28, 'SI', si), (0.72, 'FC', fc)]:
        axb.add_patch(FancyBboxPatch((cx - 0.16, 0.32), 0.32, 0.36,
                      boxstyle='round,pad=0.02', facecolor=c, edgecolor='none'))
        axb.text(cx, 0.5, lab, ha='center', va='center', fontsize=11,
                 fontweight='bold', color=tcol(c))

    # dot cluster panel (mimics the movement plot)
    axp = fig.add_axes([0.40, yc - rh * 0.40, 0.56, rh * 0.8])
    axp.set_facecolor(PANEL)
    axp.set_xlim(-25, 12); axp.set_ylim(-2, 10)
    axp.set_xticks([]); axp.set_yticks([])
    for sp in axp.spines.values():
        sp.set_edgecolor('#cfc4ad')
    for (x, y) in SI_PTS:
        axp.scatter(x, y, s=130, c=si, edgecolors=BG, linewidths=0.8, zorder=3)
    for (x, y) in FC_PTS:
        axp.scatter(x, y, s=130, c=fc, edgecolors=BG, linewidths=0.8, zorder=3)

out = os.path.expanduser('~/Downloads/si_fc_color_compare.png')
fig.savefig(out, dpi=150, facecolor=BG, bbox_inches='tight', pad_inches=0.2)
print('Saved:', out)
