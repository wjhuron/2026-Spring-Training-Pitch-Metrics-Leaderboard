#!/usr/bin/env python3
"""Two previews of Option 3 (hairline edge):
  1) Card pitch plot (warm paper)  -> ~/Downloads/card_dots_option3.png
  2) LA x Spray hitter chart (wOBAcon heatmap bg) -> ~/Downloads/la_spray_option3.png
Each shows Current (borderless) vs Option 3 side by side, on real data.
"""
import os, math, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from collections import defaultdict

P = pickle.load(open('data/all_pitches_rs_cache.pkl', 'rb'))

# ---------------- Preview 1: CARD movement plot (warm paper) ----------------
BG, PANEL, GRID, TXT, DIM = '#f0e8d8', '#e8dfcb', '#cfc4ad', '#1a1612', '#6a5f55'
CARD = {'FF': '#0072B2', 'SI': '#E0A81E', 'FC': '#8B5A2B', 'SL': '#D55E00', 'ST': '#56B4E9',
        'CU': '#332288', 'SV': '#882255', 'CH': '#009E73', 'FS': '#CC79A7'}
NM = {'FF': '4-Seam', 'SI': 'Sinker', 'FC': 'Cutter', 'SL': 'Slider', 'ST': 'Sweeper',
      'CU': 'Curve', 'SV': 'Slurve', 'CH': 'Change', 'FS': 'Split'}
ORDER = ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'SV', 'CH', 'FS']
g = defaultdict(list)
for p in P:
    if p.get('Pitcher') != 'Lugo, Seth':
        continue
    try:
        g[p['Pitch Type']].append((float(p['HorzBrk']), float(p['IndVertBrk'])))
    except (TypeError, ValueError):
        continue


def card_panel(ax, edge, title):
    ax.set_facecolor(PANEL); ax.set_xlim(-25, 25); ax.set_ylim(-25, 25)
    ax.set_xticks(range(-25, 26, 5)); ax.set_yticks(range(-25, 26, 5))
    ax.grid(True, color=GRID, lw=0.4); ax.axhline(0, color=GRID, ls='--', lw=0.6); ax.axvline(0, color=GRID, ls='--', lw=0.6)
    ax.tick_params(labelsize=6, colors=DIM); [s.set_color(GRID) for s in ax.spines.values()]
    ax.set_title(title, color=TXT, fontsize=12, fontweight='bold', pad=6)
    for pt in ORDER:
        xs, ys = zip(*g[pt]); c = CARD[pt]
        if edge:
            ax.scatter(xs, ys, c=c, s=46, edgecolors=edge, linewidths=0.5, zorder=3)
        else:
            ax.scatter(xs, ys, c=c, s=46, edgecolors='none', zorder=3)
        if len(g[pt]) >= 6:
            cov = np.cov(xs, ys); v, ve = np.linalg.eigh(cov)
            if v[0] > 0 and v[1] > 0:
                ax.add_patch(Ellipse((np.mean(xs), np.mean(ys)), 2 * 1.5 * np.sqrt(v[1]), 2 * 1.5 * np.sqrt(v[0]),
                             angle=np.degrees(np.arctan2(ve[1, 1], ve[0, 1])), fill=False, edgecolor=c, lw=1.2, ls='--', alpha=0.7))


fig, ax = plt.subplots(1, 2, figsize=(15, 7.6), dpi=150); fig.patch.set_facecolor(BG)
fig.suptitle('Card pitch plot (warm paper) — current vs Option 3 hairline', color=TXT, fontsize=15, fontweight='bold', y=0.97)
card_panel(ax[0], None, 'Current (borderless — blobby)')
card_panel(ax[1], PANEL, 'Option 3 (hairline edge = panel color)')
h = [mpatches.Patch(color=CARD[pt], label=NM[pt]) for pt in ORDER]
leg = fig.legend(handles=h, loc='lower center', ncol=9, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.01))
[t.set_color(TXT) for t in leg.get_texts()]
fig.subplots_adjust(bottom=0.13, top=0.9, wspace=0.16, left=0.05, right=0.97)
fig.savefig(os.path.expanduser('~/Downloads/card_dots_option3.png'), dpi=150, facecolor=BG)
print('Saved card preview')

# ---------------- Preview 2: LA x SPRAY (hitter, heatmap bg) ----------------
def outcome_color(ev):
    e = str(ev).lower()
    if 'home_run' in e: return '#dc143c'
    if 'triple' in e: return '#20b2aa'
    if 'double' in e and 'play' not in e: return '#7b68ee'
    if 'single' in e: return '#ff8c00'
    return '#888888'        # outs / other


def ev_radius(ev):
    if ev is None: return 7
    for lim, r in [(80, 7), (90, 8), (95, 9), (100, 10), (105, 11)]:
        if ev < lim: return r
    return 12


pts = []
for p in P:
    if str(p.get('Description')) != 'In Play' or p.get('Batter') != 'Arraez, Luis':
        continue
    try:
        la = float(p['LaunchAngle']); hx = float(p['HC_X']); hy = float(p['HC_Y'])
    except (TypeError, ValueError):
        continue
    try:
        ev = float(p['ExitVelo'])
    except (TypeError, ValueError):
        ev = None
    spray = math.degrees(math.atan2(hx - 125.42, 198.27 - hy))
    pts.append((spray, la, ev, outcome_color(p.get('Event'))))
print('Arraez BIP:', len(pts))

# approximate wOBAcon heatmap (blue low -> red high), peak in the barrel band
sx = np.linspace(-45, 45, 140); sy = np.linspace(-30, 55, 140)
SX, SY = np.meshgrid(sx, sy)
la_fac = np.exp(-((SY - 18) / 16.0) ** 2)
pull = np.clip(SX / 45.0, 0, 1) * np.clip((SY - 5) / 25.0, 0, 1)
woba = np.clip(la_fac * (0.45 + 0.7 * pull), 0, 1.35)


def spray_panel(ax, edge, title):
    ax.set_xlim(-45, 45); ax.set_ylim(-30, 55)
    ax.imshow(woba, extent=[-45, 45, -30, 55], origin='lower', aspect='auto', cmap='RdYlBu_r', alpha=0.92, zorder=0)
    ax.set_title(title, color='#eee', fontsize=12, fontweight='bold', pad=6)
    ax.set_xlabel('Spray angle (deg)', color='#bbb', fontsize=9); ax.set_ylabel('Launch angle (deg)', color='#bbb', fontsize=9)
    ax.tick_params(labelsize=7, colors='#bbb'); [s.set_color('#444') for s in ax.spines.values()]
    for (sp, la, ev, c) in pts:
        r = ev_radius(ev)
        if edge:
            ax.scatter(sp, la, s=r * r, c=c, edgecolors=edge, linewidths=0.6, zorder=3)
        else:
            ax.scatter(sp, la, s=r * r, c=c, edgecolors='none', zorder=3)


fig2, ax2 = plt.subplots(1, 2, figsize=(15, 7.4), dpi=150); fig2.patch.set_facecolor('#15171c')
fig2.suptitle('LA x Spray (Arraez) on wOBAcon heatmap — current vs Option 3 hairline (dark edge)', color='#eee', fontsize=13.5, fontweight='bold', y=0.97)
spray_panel(ax2[0], None, 'Current (borderless)')
spray_panel(ax2[1], (0, 0, 0, 0.5), 'Option 3 (hairline dark edge)')
oh = [mpatches.Patch(color=c, label=l) for l, c in [('Out', '#888888'), ('Single', '#ff8c00'), ('Double', '#7b68ee'), ('Triple', '#20b2aa'), ('HR', '#dc143c')]]
leg2 = fig2.legend(handles=oh, loc='lower center', ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.01))
[t.set_color('#eee') for t in leg2.get_texts()]
fig2.subplots_adjust(bottom=0.13, top=0.9, wspace=0.16, left=0.05, right=0.97)
fig2.savefig(os.path.expanduser('~/Downloads/la_spray_option3.png'), dpi=150, facecolor='#15171c')
print('Saved LA x spray preview')
