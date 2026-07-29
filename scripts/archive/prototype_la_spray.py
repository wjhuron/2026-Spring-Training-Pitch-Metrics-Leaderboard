#!/usr/bin/env python3
"""Side-by-side mockup: the CURRENT LA x Spray panel vs OPTION B.

LEFT  = faithful reproduction of the production hitter-card panel
        (wOBAcon value-map zones + outcome-colored EV-sized dots + zone grid
         + Avg Placement marker).
OPTION B (right) = BIP DENSITY heat map + damage-zone outlines + BARRELS as
        red dots (all other dots removed).

This is a COMPARISON MOCKUP ONLY — it does not touch the production cards.
Output -> ~/Downloads/_la_spray_current_vs_optionB.png

    python3 scripts/prototype_la_spray.py "Wood, James"
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

import HitterCards as HC
from HitterCards import (load_pitch_data, load_metadata, build_sacq_lookup,
                         spray_angle, la_bin_idx, spray_direction, WOBA_CMAP,
                         LA_BINS, sf, BG, GRID_COLOR, TEXT_PRIMARY, TEXT_SECONDARY,
                         TEXT_MUTED, TEXT_FAINT, MARKER_ACCENT)

NAME = sys.argv[1] if len(sys.argv) > 1 else 'Wood, James'
THROUGH = '2026-06-02'
OUT = '/Users/wallyhuron/Downloads/_la_spray_current_vs_optionB.png'
XLIM = (-50, 50); YLIM = (-20, 60)
BARREL_COLOR = '#b81d24'


def zone_bounds(bats):
    if bats == 'L':
        return {'pull': (30, 50), 'pull_side': (15, 30), 'center_pull': (0, 15),
                'center_oppo': (-15, 0), 'oppo_side': (-30, -15), 'oppo': (-50, -30)}
    return {'pull': (-50, -30), 'pull_side': (-30, -15), 'center_pull': (-15, 0),
            'center_oppo': (0, 15), 'oppo_side': (15, 30), 'oppo': (30, 50)}


# ── Faithful current-panel zone fills (mirrors HitterCards 1398-1433) ──
def draw_value_zones(ax, bats, hand_zones, pool_zones, soft=False):
    bounds = zone_bounds(bats)
    def fill(sd, lb, z, count):
        if sd not in bounds or lb >= len(LA_BINS): return
        v = z.get('wobacon') if z.get('wobacon') is not None else z.get('woba')
        if v is None: return
        bx = bounds[sd]; ly = LA_BINS[lb]
        lo = max(-20, ly[0]); hi = min(60, ly[1])
        col = WOBA_CMAP(min(1.0, v / 1.0))
        a = (0.22 if count < 20 else 0.70)
        if soft: a *= 0.6
        ax.add_patch(Rectangle((min(bx), lo), abs(bx[1]-bx[0]), hi-lo,
                               facecolor=col, alpha=a, edgecolor=GRID_COLOR, linewidth=0.3))
    for (sd, lb), z_hand in hand_zones.items():
        z_pool = pool_zones.get((sd, lb))
        z = z_hand if z_hand.get('count', 0) >= 20 else z_pool
        if not z: continue
        fill(sd, lb, z, z.get('count', 0))
    for (sd, lb), z in pool_zones.items():
        if (sd, lb) in hand_zones: continue
        fill(sd, lb, z, z.get('count', 0))


def draw_grid_edges(ax, bats):
    b = zone_bounds(bats)
    for x in sorted({e for v in b.values() for e in v}):
        if XLIM[0] < x < XLIM[1]:
            ax.axvline(x, color=GRID_COLOR, lw=1.0, alpha=0.75, zorder=2)
    for y in sorted({e for r in LA_BINS for e in r}):
        if YLIM[0] < y < YLIM[1]:
            ax.axhline(y, color=GRID_COLOR, lw=1.0, alpha=0.75, zorder=2)


# ── Outcome dot styling (verbatim from HitterCards) ──
OUTCOME_COLORS = {'Out': '#6e6557', '1B': '#e0892b', '2B': '#9a4eaf', '3B': '#188a8a', 'HR': '#a8261e'}
OUTCOME_ALPHA = {'Out': 0.62, '1B': 0.95, '2B': 0.95, '3B': 0.95, 'HR': 0.95}
_OUTCOME_Z = {'Out': 0, '1B': 1, '2B': 2, '3B': 3, 'HR': 4}

def _cat(ev_event):
    return {'Single': '1B', 'Double': '2B', 'Triple': '3B', 'Home Run': 'HR'}.get(ev_event, 'Out')

def outcome_color(event):
    c = OUTCOME_COLORS[_cat(event)]; a = OUTCOME_ALPHA[_cat(event)]
    return (int(c[1:3],16)/255, int(c[3:5],16)/255, int(c[5:7],16)/255, a)

def ev_size(ev):
    if ev is None: return 110
    if ev < 80:  return 110
    if ev < 90:  return 175
    if ev < 95:  return 250
    if ev < 100: return 340
    if ev < 105: return 430
    return 540


def style_axes(ax, title):
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_facecolor(BG)
    ax.set_title(title, fontsize=16, fontweight='bold', color=TEXT_SECONDARY,
                 fontfamily='DIN Condensed', pad=10)
    ax.set_xlabel('Oppo   •   Spray Angle   •   Pull', fontsize=10, color=TEXT_MUTED)
    ax.set_ylabel('Launch Angle', fontsize=10, color=TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    for s in ax.spines.values(): s.set_color(TEXT_FAINT)


def kde_grid(pts, hx=7.0, hy=5.0, nx=130, ny=110):
    gx = np.linspace(*XLIM, nx); gy = np.linspace(*YLIM, ny)
    GX, GY = np.meshgrid(gx, gy); Z = np.zeros_like(GX)
    for (x, y) in pts:
        Z += np.exp(-(((GX - x) / hx) ** 2 + ((GY - y) / hy) ** 2) / 2.0)
    return gx, gy, Z


# ── Load ──
print(f'Loading {NAME} (through {THROUGH}) …')
allp = load_pitch_data()
meta = load_metadata()
lb = HC.load_hitter_leaderboard()
hrow = next((r for r in lb if r.get('hitter') == NAME), {})
bats = hrow.get('stands') or 'L'

bip = []
for p in allp:
    if p.get('Batter') != NAME: continue
    if str(p.get('Game Date', '')) > THROUGH: continue
    if p.get('Description') != 'In Play': continue
    bbt = str(p.get('BBType', '')).strip()
    if not bbt or bbt.startswith('bunt'): continue
    la = sf(p.get('LaunchAngle')); ang = spray_angle(sf(p.get('HC_X')), sf(p.get('HC_Y')))
    if la is None or ang is None: continue
    bip.append({'x': ang, 'y': max(-20, min(60, la)), 'ev': sf(p.get('ExitVelo')),
                'la': la, 'event': p.get('Event', '') or '',
                'barrel': str(p.get('Barrel', '')).strip() == '6'})
barrels = [(b['x'], b['y'], b['ev']) for b in bip if b['barrel']]
allpts = [(b['x'], b['y']) for b in bip]
print(f'  bats={bats}  BIP={len(bip)}  barrels={len(barrels)}')

_, hand_zones, pool_zones = build_sacq_lookup(meta, bats)
# Avg placement = median of the through-date BIP (consistent with shown data).
med_x = float(np.median([b['x'] for b in bip])); med_y = float(np.median([b['y'] for b in bip]))
cells = []
bounds = zone_bounds(bats)
for (sd, lb_), z in {**pool_zones, **hand_zones}.items():
    v = z.get('wobacon') if z.get('wobacon') is not None else z.get('woba')
    if v is None or sd not in bounds or lb_ >= len(LA_BINS): continue
    bx = bounds[sd]; ly = LA_BINS[lb_]
    cells.append((min(bx), max(-20, ly[0]), abs(bx[1]-bx[0]), min(60, ly[1])-max(-20, ly[0]), v))

# ── Figure ──
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 8.6))
fig.patch.set_facecolor(BG)

# LEFT — current panel (faithful)
style_axes(axL, 'CURRENT — value map + every BIP (outcome color, EV size)')
draw_value_zones(axL, bats, hand_zones, pool_zones, soft=False)
draw_grid_edges(axL, bats)
for b in sorted(bip, key=lambda r: _OUTCOME_Z[_cat(r['event'])]):
    axL.scatter([b['x']], [b['y']], s=ev_size(b['ev']), c=[outcome_color(b['event'])],
                edgecolors='#1a1612', linewidths=0.6, zorder=3 + _OUTCOME_Z[_cat(b['event'])])
axL.scatter([med_x], [med_y], s=420, c='white', zorder=10, alpha=0.95, edgecolors='black', linewidths=0.5)
axL.scatter([med_x], [med_y], s=240, c=MARKER_ACCENT, edgecolors='black', linewidths=2, zorder=11)
axL.legend(handles=[
    Line2D([0],[0], marker='o', color='none', markerfacecolor=OUTCOME_COLORS['Out'], markersize=9, label='Out/E/FC'),
    Line2D([0],[0], marker='o', color='none', markerfacecolor=OUTCOME_COLORS['1B'], markersize=9, label='1B'),
    Line2D([0],[0], marker='o', color='none', markerfacecolor=OUTCOME_COLORS['2B'], markersize=9, label='2B'),
    Line2D([0],[0], marker='o', color='none', markerfacecolor=OUTCOME_COLORS['HR'], markersize=9, label='HR'),
    Line2D([0],[0], marker='o', color='none', markerfacecolor=MARKER_ACCENT, markersize=10, label='Avg Placement'),
], loc='upper right', fontsize=8, frameon=False, labelcolor=TEXT_MUTED, handletextpad=0.3)

# RIGHT — Option B (recommended): NEUTRAL gray density + table-colored damage
# zones, so frequency (gray) and value (warm) read independently.
style_axes(axR, 'OPTION B — gray density + table-colored damage zones + barrels')
GRAY_CMAP = LinearSegmentedColormap.from_list(
    'gray', [(0.0, (0.95, 0.93, 0.88)), (0.45, (0.64, 0.64, 0.67)),
             (1.0, (0.20, 0.20, 0.24))], N=256)
gx, gy, Z = kde_grid(allpts)
Zn = Z / Z.max() if Z.max() > 0 else Z
Zn = Zn ** 1.25   # bump contrast so his hot spot reads strongly under the color
axR.imshow(Zn, origin='lower', extent=[*XLIM, *YLIM], aspect='auto', cmap=GRAY_CMAP, alpha=0.95, zorder=1)
for (x0, y0, w, h, v) in cells:
    if v >= 0.50:   # damage zones → table color (WOBA_CMAP), like the current panel
        axR.add_patch(Rectangle((x0, y0), w, h, facecolor=HC.WOBA_CMAP(min(1.0, v)),
                                alpha=0.62, edgecolor=GRID_COLOR, linewidth=0.4, zorder=3))
for (x, y, ev) in barrels:
    s = 70 + (max(85, min(115, ev)) - 85) * 6 if ev else 90
    axR.scatter([x], [y], s=s, c=BARREL_COLOR, edgecolors='#1a1612', linewidths=0.7, zorder=10, alpha=0.95)
axR.legend(handles=[
    Line2D([0],[0], marker='o', color='none', markerfacecolor=BARREL_COLOR, markersize=10, label='Barrel'),
    Line2D([0],[0], marker='s', color='none', markerfacecolor=(0.85,0.30,0.22), markersize=10, label='Damage zone (table color)'),
    Line2D([0],[0], marker='s', color='none', markerfacecolor=(0.42,0.42,0.45), markersize=10, label='Darker = more BIP'),
], loc='upper right', fontsize=8, frameon=False, labelcolor=TEXT_MUTED, handletextpad=0.5)

fig.suptitle(f'{NAME}  ·  LA × Spray: current vs Option B  ·  {len(bip)} BIP through Jun 2, 2026  ·  {len(barrels)} barrels',
             fontsize=15, fontweight='bold', color=TEXT_PRIMARY, fontfamily='DIN Condensed', y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=140, facecolor=BG, bbox_inches='tight')
print(f'wrote {OUT}')
