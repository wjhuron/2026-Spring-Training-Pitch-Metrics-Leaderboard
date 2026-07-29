#!/usr/bin/env python3
"""Can we pull the arm-side trio (SI/CH/FS) apart for colorblind viewers at
500-each density? Compares the recommended palette vs a tweak that gives the
changeup a darker green and the splitter a blue-leaning magenta (both survive
the deuteranopia transform better than gold/green/pink).

2x2: {Normal, Deuteranopia} x {Recommended, Tweaked}, 500 of each pitch type.
Prints the closest color pair under deuteranopia for each palette.
Output: ~/Downloads/dense500_compare.png
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

np.random.seed(7)
BG = '#f0e8d8'; PLOT_PANEL = '#e8dfcb'; GRID = '#cfc4ad'
TEXT_PRIMARY = '#1a1612'; TEXT_MUTED = '#6a5f55'

NAMES = {'FF': '4-Seam', 'SI': 'Sinker', 'FC': 'Cutter', 'CH': 'Changeup', 'FS': 'Splitter',
         'SL': 'Slider', 'ST': 'Sweeper', 'SV': 'Slurve', 'CU': 'Curveball'}
ORDER = ['FF', 'SI', 'FC', 'CH', 'FS', 'SL', 'ST', 'SV', 'CU']

REC   = {'FF': '#0072B2', 'SI': '#E0A81E', 'FC': '#8B5A2B', 'CH': '#009E73', 'FS': '#CC79A7',
         'SL': '#D55E00', 'ST': '#56B4E9', 'SV': '#882255', 'CU': '#332288'}
# Tweak: changeup -> darker forest green (lightness gap from amber); splitter ->
# saturated magenta (blue content survives the red-green transform).
TWEAK = dict(REC, CH='#117733', FS='#AA4499')

DEUT = np.array([[0.367322, 0.860646, -0.227968],
                 [0.280085, 0.672501, 0.047413],
                 [-0.011820, 0.042940, 0.968881]])
hex2rgb = lambda h: np.array([int(h[i:i+2], 16) / 255 for i in (1, 3, 5)])
srgb2lin = lambda c: np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
lin2srgb = lambda c: np.where(np.clip(c, 0, 1) <= 0.0031308, np.clip(c, 0, 1) * 12.92,
                              1.055 * np.clip(c, 0, 1) ** (1 / 2.4) - 0.055)
deut = lambda h: tuple(lin2srgb(DEUT @ srgb2lin(hex2rgb(h))))

with open(os.path.join('data', 'all_pitches_rs_cache.pkl'), 'rb') as f:
    P = pickle.load(f)
pts = defaultdict(list)
for p in P:
    if p.get('Throws') != 'R' or p.get('Pitch Type') not in NAMES:
        continue
    try:
        pts[p['Pitch Type']].append((float(p['HorzBrk']), float(p['IndVertBrk'])))
    except (TypeError, ValueError):
        continue
samples = {}
for pt in ORDER:
    a = np.array(pts[pt])
    samples[pt] = np.random.multivariate_normal(a.mean(0), np.cov(a.T) * 0.5, 500)


def draw(ax, cmap, title):
    ax.set_facecolor(PLOT_PANEL); ax.set_xlim(-25, 25); ax.set_ylim(-25, 25)
    ax.set_xticks(range(-25, 26, 5)); ax.set_yticks(range(-25, 26, 5))
    ax.axhline(0, color=GRID, ls='--', lw=0.6); ax.axvline(0, color=GRID, ls='--', lw=0.6)
    ax.grid(True, alpha=0.5, color=GRID); ax.tick_params(labelsize=7, colors=TEXT_MUTED)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_title(title, fontsize=12, fontweight='bold', color=TEXT_PRIMARY, pad=6)
    for pt in ORDER:
        xy = samples[pt]
        ax.scatter(xy[:, 0], xy[:, 1], color=cmap[pt], s=7, alpha=0.5, edgecolors='none')
    handles = [mpatches.Patch(color=cmap[pt], label=f'{pt}·{NAMES[pt]}') for pt in ORDER]
    leg = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.11),
                    ncol=5, fontsize=7, frameon=False, handlelength=1.0, columnspacing=0.9)
    for t in leg.get_texts():
        t.set_color(TEXT_PRIMARY)


fig, ax = plt.subplots(2, 2, figsize=(16, 16.5), dpi=145)
fig.patch.set_facecolor(BG)
fig.suptitle('500 of every pitch type — Recommended vs Tweaked (arm-side trio recolored)',
             fontsize=17, fontweight='bold', color=TEXT_PRIMARY, y=0.985)
draw(ax[0, 0], REC, 'Recommended — Normal vision')
draw(ax[0, 1], TWEAK, 'Tweaked — Normal vision')
draw(ax[1, 0], {pt: deut(REC[pt]) for pt in ORDER}, 'Recommended — Deuteranopia')
draw(ax[1, 1], {pt: deut(TWEAK[pt]) for pt in ORDER}, 'Tweaked — Deuteranopia')
fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.06, hspace=0.20, wspace=0.14)
out = os.path.expanduser('~/Downloads/dense500_compare.png')
fig.savefig(out, dpi=145, facecolor=BG); print('Saved:', out)

# quantify: closest color pair under deuteranopia
for nm, pal in [('Recommended', REC), ('Tweaked', TWEAK)]:
    sim = {pt: np.array(deut(pal[pt])) for pt in ORDER}
    pairs = sorted((float(np.linalg.norm(np.clip(sim[a], 0, 1) - np.clip(sim[b], 0, 1))), a, b)
                   for i, a in enumerate(ORDER) for b in ORDER[i + 1:])
    print(f'{nm:12s} closest under deuteranopia:', ', '.join(f'{a}-{b} ({d:.2f})' for d, a, b in pairs[:3]))
