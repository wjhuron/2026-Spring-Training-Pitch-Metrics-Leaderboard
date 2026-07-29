#!/usr/bin/env python3
"""Colorblind-friendly pitch-color palettes, tested on Seth Lugo's real season
movement plot (he throws all 9 tracked pitch types).

Renders the current palette + 3 candidates as a 2x2 grid of IVB-vs-HB plots
(matching the card: -25..25, dashed 1.5-sigma cluster ellipses, legend below),
and saves the palette definitions as JSON.

Outputs:
  ~/Downloads/lugo_palette_compare.png
  ~/Downloads/pitch_palettes.json
"""
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from collections import defaultdict

BG = '#f0e8d8'
PLOT_PANEL = '#e8dfcb'
GRID = '#cfc4ad'
TEXT_PRIMARY = '#1a1612'
TEXT_MUTED = '#6a5f55'

PITCH_NAMES = {'FF': '4-Seam', 'SI': 'Sinker', 'FC': 'Cutter', 'CH': 'Changeup',
               'FS': 'Splitter', 'SL': 'Slider', 'ST': 'Sweeper', 'SV': 'Slurve',
               'CU': 'Curveball'}
LEGEND_ORDER = ['FF', 'SI', 'FC', 'CH', 'FS', 'SL', 'ST', 'SV', 'CU']

# ── palettes ───────────────────────────────────────────────────────────────
CURRENT = {'FF': '#4488FF', 'SI': '#E0A81E', 'FC': '#8B5A2B', 'SL': '#9E9E9E',
           'ST': '#FF1493', 'CU': '#E03030', 'SV': '#32CD32', 'CH': '#CC66EE', 'FS': '#35BCAF'}

# A — Okabe-Ito based: vivid, CB-safe, keeps intuitive anchors (FF blue, SI
# orange, CH green, FC brown). Arm-side trio {SI,CH,FS}=orange/green/purple;
# glove-side {SL,ST,SV,CU}=vermillion/sky/wine/indigo.
OKABE = {'FF': '#0072B2', 'SI': '#E69F00', 'FC': '#8B5A2B', 'CH': '#009E73',
         'FS': '#CC79A7', 'SL': '#D55E00', 'ST': '#56B4E9', 'SV': '#882255', 'CU': '#332288'}

# B — Paul Tol "Muted": the canonical 9-color CB-safe qualitative set, softer.
TOL_MUTED = {'FF': '#88CCEE', 'SI': '#DDCC77', 'FC': '#999933', 'CH': '#117733',
             'FS': '#882255', 'SL': '#CC6677', 'ST': '#44AA99', 'SV': '#AA4499', 'CU': '#332288'}

# C — Familiar: keeps the identities you already use (blue/gold/brown/pink/red/
# purple/teal); only fixes the red-green clash (slurve green -> purple) and the
# dull grey slider (-> steel blue). Most learnable, partial CB.
FAMILIAR = {'FF': '#4488FF', 'SI': '#E0A81E', 'FC': '#8B5A2B', 'SL': '#5A7AB8',
            'ST': '#FF1493', 'CU': '#E03030', 'SV': '#9B59B6', 'CH': '#CC66EE', 'FS': '#2BB3A6'}

PALETTES = [
    ('Current', CURRENT),
    ('A · Okabe-Ito  (vivid, CB-safe)', OKABE),
    ('B · Paul Tol Muted  (CB-safe)', TOL_MUTED),
    ('C · Familiar  (keeps your identities)', FAMILIAR),
]


def load_lugo_groups():
    with open(os.path.join('data', 'all_pitches_rs_cache.pkl'), 'rb') as f:
        P = pickle.load(f)
    groups = defaultdict(list)
    for p in P:
        if p.get('Pitcher') != 'Lugo, Seth':
            continue
        pt = p.get('Pitch Type', '')
        try:
            hb = float(p.get('HorzBrk')); ivb = float(p.get('IndVertBrk'))
        except (TypeError, ValueError):
            continue
        if pt in PITCH_NAMES:
            groups[pt].append((hb, ivb))
    return groups


def draw_plot(ax, groups, palette, title):
    ax.set_facecolor(PLOT_PANEL)
    ax.set_xlim(-25, 25); ax.set_ylim(-25, 25)
    ax.set_xticks(range(-25, 26, 5)); ax.set_yticks(range(-25, 26, 5))
    ax.axhline(0, color=GRID, ls='--', lw=0.6); ax.axvline(0, color=GRID, ls='--', lw=0.6)
    ax.grid(True, alpha=0.5, color=GRID)
    ax.tick_params(labelsize=7, colors=TEXT_MUTED)
    ax.set_xlabel('Horizontal Break (in)', fontsize=9, color=TEXT_MUTED, fontweight='bold')
    ax.set_ylabel('Induced Vertical Break (in)', fontsize=9, color=TEXT_MUTED, fontweight='bold')
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_title(title, fontsize=13, fontweight='bold', color=TEXT_PRIMARY, pad=8)

    # draw big clusters first so small ones (e.g. FS) stay visible on top
    for pt in sorted(groups, key=lambda k: -len(groups[k])):
        xs, ys = zip(*groups[pt]); c = palette[pt]
        ax.scatter(xs, ys, c=c, s=42, alpha=1.0, edgecolors='none', zorder=3)
        if len(groups[pt]) >= 6:
            cov = np.cov(xs, ys); vals, vecs = np.linalg.eigh(cov)
            if vals[0] > 0 and vals[1] > 0:
                ax.add_patch(Ellipse((np.mean(xs), np.mean(ys)),
                    2 * 1.5 * np.sqrt(vals[1]), 2 * 1.5 * np.sqrt(vals[0]),
                    angle=np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])),
                    fill=False, edgecolor=c, lw=1.2, ls='--', alpha=0.7, zorder=2))

    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=palette[pt], label=f'{pt} · {PITCH_NAMES[pt]}')
               for pt in LEGEND_ORDER if pt in groups]
    leg = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.14),
                    ncol=5, fontsize=7.5, frameon=False, handlelength=1.1, columnspacing=1.0)
    for t in leg.get_texts():
        t.set_color(TEXT_PRIMARY)


def main():
    groups = load_lugo_groups()
    n = {pt: len(groups[pt]) for pt in LEGEND_ORDER if pt in groups}
    print('Lugo pitch counts:', n, '| total', sum(n.values()))

    fig, axes = plt.subplots(2, 2, figsize=(15, 15.5), dpi=150)
    fig.patch.set_facecolor(BG)
    fig.suptitle('Seth Lugo · 2026 season movement — pitch-color palettes',
                 fontsize=19, fontweight='bold', color=TEXT_PRIMARY, y=0.985)
    for ax, (name, pal) in zip(axes.flat, PALETTES):
        draw_plot(ax, groups, pal, name)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.06, hspace=0.32, wspace=0.18)

    img = os.path.expanduser('~/Downloads/lugo_palette_compare.png')
    fig.savefig(img, dpi=150, facecolor=BG)
    print('Saved:', img)

    pj = os.path.expanduser('~/Downloads/pitch_palettes.json')
    with open(pj, 'w') as f:
        json.dump({'current': CURRENT, 'okabe_ito': OKABE,
                   'tol_muted': TOL_MUTED, 'familiar': FAMILIAR}, f, indent=2)
    print('Saved:', pj)


if __name__ == '__main__':
    main()
