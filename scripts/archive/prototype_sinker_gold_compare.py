#!/usr/bin/env python3
"""Compare the two Sinker golds in card context.

Cards.py uses SI #E0A81E; the website + other generators use SI #FFD700.
This renders each gold the way it appears on a card: the four-pitch badge row
and a movement-plot scatter (with the new brown cutter), on warm paper.
Output: ~/Downloads/sinker_gold_compare.png
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG = '#f0e8d8'
PANEL = '#e8dfcb'
TEXT_PRIMARY = '#1a1612'
TEXT_MUTED = '#6a5f55'

FF = '#4488FF'   # fastball blue
FC = '#8B5A2B'   # cutter brown (new)
ST = '#FF1493'   # sweeper pink

# movement-plot clusters (IVB vs HB), LHP-style like Palmquist
FF_PTS = [(-12, 12), (-11, 12.5), (-13, 11), (-12.5, 13), (-10.5, 11.5), (-11.5, 10.5)]
SI_PTS = [(-19, 5), (-18, 6.5), (-17.5, 4), (-16.5, 5.5), (-18.5, 3), (-17, 7), (-16, 4.5), (-19.5, 4)]
FC_PTS = [(1, 6), (2.5, 5), (3.5, 6.5), (2, 4), (4, 5.5), (3, 3.5)]
ST_PTS = [(11, 1), (12, 0), (13, -1), (11.5, -0.5), (13.5, 0.5), (12.5, -2)]

ROWS = [
    ('Cards.py  ·  SI #E0A81E', '#E0A81E'),
    ('Site & other files  ·  SI #FFD700', '#FFD700'),
]


def lum(h):
    r, g, b = (int(h[i:i+2], 16) / 255 for i in (1, 3, 5))
    return 0.299 * r + 0.587 * g + 0.114 * b


def tcol(h):
    return '#1a1612' if lum(h) > 0.55 else '#ffffff'


fig = plt.figure(figsize=(12, 7.2), dpi=150)
fig.patch.set_facecolor(BG)
fig.text(0.5, 0.96, 'SINKER gold  —  #E0A81E  vs  #FFD700', ha='center',
         fontsize=18, fontweight='bold', color=TEXT_PRIMARY)
fig.text(0.5, 0.915, 'Each gold shown as the badge row + movement-plot dots (cutter now brown), on warm paper.',
         ha='center', fontsize=10.5, color=TEXT_MUTED)

top, bot = 0.85, 0.05
rh = (top - bot) / len(ROWS)
for i, (name, si) in enumerate(ROWS):
    y0 = top - (i + 1) * rh
    yc = y0 + rh / 2
    fig.text(0.035, yc + rh * 0.32, name, fontsize=13, fontweight='bold',
             color=TEXT_PRIMARY, va='center')

    # badge row: FF SI FC ST
    axb = fig.add_axes([0.035, yc - rh * 0.12, 0.26, rh * 0.34])
    axb.set_xlim(0, 4); axb.set_ylim(0, 1); axb.axis('off')
    for j, (lab, c) in enumerate([('FF', FF), ('SI', si), ('FC', FC), ('ST', ST)]):
        axb.add_patch(FancyBboxPatch((j + 0.12, 0.2), 0.76, 0.6,
                      boxstyle='round,pad=0.03', facecolor=c, edgecolor='none'))
        axb.text(j + 0.5, 0.5, lab, ha='center', va='center', fontsize=12,
                 fontweight='bold', color=tcol(c))

    # movement-plot scatter
    axp = fig.add_axes([0.36, yc - rh * 0.40, 0.60, rh * 0.82])
    axp.set_facecolor(PANEL)
    axp.set_xlim(-24, 20); axp.set_ylim(-6, 16)
    axp.axhline(0, color='#cfc4ad', lw=0.8); axp.axvline(0, color='#cfc4ad', lw=0.8)
    axp.set_xticks([]); axp.set_yticks([])
    for sp in axp.spines.values():
        sp.set_edgecolor('#cfc4ad')
    for pts, c in [(FF_PTS, FF), (SI_PTS, si), (FC_PTS, FC), (ST_PTS, ST)]:
        for (x, y) in pts:
            axp.scatter(x, y, s=120, c=c, edgecolors=BG, linewidths=0.8, zorder=3)
    axp.text(-23, 14.5, 'IVB vs HB', fontsize=8, color=TEXT_MUTED, style='italic')

out = os.path.expanduser('~/Downloads/sinker_gold_compare.png')
fig.savefig(out, dpi=150, facecolor=BG, bbox_inches='tight', pad_inches=0.2)
print('Saved:', out)
