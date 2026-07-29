#!/usr/bin/env python3
"""Mockup: season-card movement plot with an arm-angle spoke + league ghost rings.

- Muted spoke from the origin at the pitcher's average arm angle (Q1 for RHP,
  Q4 for LHP), degree label at the rim.
- Faint gray open rings at the league-average movement (same hand) for each of
  the pitcher's pitch types — ghost anchors for reading shape vs league.

Real Lugo (KCR, RHP, ~29.8 deg) data from data/all_pitches_rs_cache.pkl.
Run from repo root. Output: ~/Downloads/mockup_movement_spokes.png
"""
import os
import pickle
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, Circle

# ── Card theme (Cards.py warm paper) ──
BG = '#f0e8d8'; PLOT_PANEL = '#e8dfcb'; GRID_COLOR = '#c5b89f'
TEXT_PRIMARY = '#1a1612'; TEXT_SECONDARY = '#3a3530'; TEXT_MUTED = '#6a5f55'
TEXT_FAINT = '#8a7f75'
PITCH_COLORS = {'FF': '#0072B2', 'SI': '#E0A81E', 'FC': '#8B5A2B', 'SL': '#D55E00',
                'ST': '#56B4E9', 'CU': '#332288', 'SV': '#882255', 'CH': '#009E73',
                'FS': '#CC79A7', 'KN': '#9A9A9A'}
PITCH_NAMES = {'FF': 'Fastball', 'SI': 'Sinker', 'FC': 'Cutter', 'SL': 'Slider',
               'ST': 'Sweeper', 'CU': 'Curveball', 'SV': 'Slurve', 'CH': 'Changeup',
               'FS': 'Splitter', 'KN': 'Knuckleball'}
PITCH_ORDER = ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'SV', 'CH', 'FS', 'KN']
GHOST = '#9a9186'   # warm gray for the league ghost anchors + spoke


def sf(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


with open(os.path.join('data', 'all_pitches_rs_cache.pkl'), 'rb') as f:
    P = pickle.load(f)

# Lugo movement groups + arm angle
groups = defaultdict(list)
arm_angles = []
for p in P:
    if p.get('Pitcher') != 'Lugo, Seth' or p.get('PTeam') != 'KCR':
        continue
    pt = p.get('Pitch Type')
    hb = sf(p.get('xHorzBrk') if p.get('xHorzBrk') not in (None, '') else p.get('HorzBrk'))
    ivb = sf(p.get('xIndVrtBrk') if p.get('xIndVrtBrk') not in (None, '') else p.get('IndVertBrk'))
    if pt and hb is not None and ivb is not None:
        groups[pt].append((hb, ivb))
    aa = sf(p.get('ArmAngle'))
    if aa is not None:
        arm_angles.append(aa)
arm_angle = np.mean(arm_angles)
throws = 'R'
print(f'Lugo: {sum(len(v) for v in groups.values())} pitches, arm angle {arm_angle:.1f}')

# League-average movement per pitch type, same hand (MLB RHP), Lugo's types only
lg_sum = defaultdict(lambda: [0.0, 0.0, 0])
for p in P:
    if p.get('_source') != 'MLB' or p.get('Throws') != throws:
        continue
    pt = p.get('Pitch Type')
    if pt not in groups:
        continue
    hb, ivb = sf(p.get('HorzBrk')), sf(p.get('IndVertBrk'))
    if hb is None or ivb is None:
        continue
    s = lg_sum[pt]; s[0] += hb; s[1] += ivb; s[2] += 1
league_avg = {pt: (s[0]/s[2], s[1]/s[2]) for pt, s in lg_sum.items() if s[2] >= 100}

sorted_types = [pt for pt in PITCH_ORDER if pt in groups]

fig = plt.figure(figsize=(10.6, 10.9), dpi=100)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0.09, 0.115, 0.86, 0.80])
fig.text(0.52, 0.965, 'PITCH MOVEMENT — ARM-ANGLE SPOKE + LEAGUE GHOST ANCHORS',
         ha='center', va='center', fontsize=19, fontweight='bold',
         color=TEXT_SECONDARY, fontfamily='DIN Condensed')
fig.text(0.52, 0.938, 'Seth Lugo · KCR · RHP · 2026 Season', ha='center', va='center',
         fontsize=11, color=TEXT_MUTED, fontfamily='Avenir Next')

ax.set_xlim(-25, 25); ax.set_ylim(-25, 25)
ax.axhline(0, color=GRID_COLOR, ls='--', lw=0.6)
ax.axvline(0, color=GRID_COLOR, ls='--', lw=0.6)
ax.set_xlabel('Horizontal Break (in)', fontsize=10, color=TEXT_MUTED,
              fontweight='bold', fontfamily='Avenir Next')
ax.set_ylabel('Induced Vertical Break (in)', fontsize=10, color=TEXT_MUTED,
              fontweight='bold', fontfamily='Avenir Next')
ax.tick_params(labelsize=8, colors=TEXT_MUTED)
ax.set_xticks(range(-25, 26, 5)); ax.set_yticks(range(-25, 26, 5))
ax.grid(True, alpha=0.5, color=GRID_COLOR); ax.set_facecolor(PLOT_PANEL)
for sp in ax.spines.values():
    sp.set_color(TEXT_FAINT)

# ── Arm-angle spoke: origin -> rim at the avg arm angle. Q1 (x>0) for RHP,
#    Q4-mirrored (x<0) for LHP so the spoke always points to the arm side. ──
theta = np.radians(arm_angle)
R = 24.0
dx = np.cos(theta) * (1 if throws == 'R' else -1)
dy = np.sin(theta)
ax.plot([0, R*dx], [0, R*dy], color=GHOST, lw=1.8, ls=(0, (5, 4)),
        alpha=0.85, zorder=2, solid_capstyle='round')
ax.annotate(f'{arm_angle:.1f}°', xy=(R*dx, R*dy),
            xytext=(R*dx - 2.2, R*dy + 1.1), fontsize=11, fontweight='bold',
            color=TEXT_MUTED, ha='center', va='center', fontfamily='Avenir Next')
ax.text(R*dx - 2.2, R*dy - 0.6, 'arm angle', fontsize=7.5, color=TEXT_FAINT,
        ha='center', va='top', fontfamily='Avenir Next', fontstyle='italic')

# ── League ghost anchors: faint open rings at same-hand league avg movement ──
for pt, (lx, ly) in league_avg.items():
    ax.add_patch(Circle((lx, ly), 1.35, fill=False, edgecolor=GHOST,
                        linewidth=1.5, alpha=0.75, zorder=2))
    ax.text(lx, ly - 2.0, pt, fontsize=7.5, color=GHOST, alpha=0.95,
            ha='center', va='top', fontweight='bold', fontfamily='Avenir Next')

# ── Scatter + covariance ellipses (same as the shipped card) ──
for pt in PITCH_ORDER:
    if pt not in groups:
        continue
    xs, ys = zip(*groups[pt]); c = PITCH_COLORS[pt]
    ax.scatter(xs, ys, c=c, s=65, alpha=1.0, edgecolors=PLOT_PANEL, linewidths=0.5, zorder=3)
    if len(groups[pt]) >= 6:
        cov = np.cov(xs, ys); vals, vecs = np.linalg.eigh(cov)
        if vals[0] > 0 and vals[1] > 0:
            ax.add_patch(Ellipse((np.mean(xs), np.mean(ys)),
                                 2*1.5*np.sqrt(vals[1]), 2*1.5*np.sqrt(vals[0]),
                                 angle=np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])),
                                 fill=False, edgecolor=c, lw=1.2, ls='--', alpha=0.7))

handles = [mpatches.Patch(color=PITCH_COLORS[pt], label=f'{pt} - {PITCH_NAMES[pt]}')
           for pt in sorted_types]
leg = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.075),
                ncol=min(len(sorted_types), 5), fontsize=8, frameon=False,
                handlelength=1.2, columnspacing=1.2)
for t in leg.get_texts():
    t.set_color(TEXT_SECONDARY)

fig.text(0.52, 0.022,
         'Dashed spoke = avg arm angle from release · Gray open rings = league-average movement '
         'per pitch type (RHP only)',
         ha='center', fontsize=9, color=TEXT_MUTED, fontfamily='Avenir Next', fontweight='bold')

out = os.path.expanduser('~/Downloads/mockup_movement_spokes.png')
plt.savefig(out, dpi=150, facecolor=BG, bbox_inches='tight', pad_inches=0.15)
print('saved', out)
