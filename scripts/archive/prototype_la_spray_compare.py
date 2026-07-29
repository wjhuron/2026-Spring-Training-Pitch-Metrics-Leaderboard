#!/usr/bin/env python3
"""Compare two hitters in the recommended Option B LA x Spray format
(gray contact density + table-colored damage zones + red barrels).

    python3 scripts/prototype_la_spray_compare.py "Alvarez, Yordan" "Chapman, Matt"

Output -> ~/Downloads/_la_spray_compare.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import HitterCards as HC
from HitterCards import (load_pitch_data, load_metadata, build_sacq_lookup,
                         spray_angle, LA_BINS, sf, BG, GRID_COLOR, TEXT_PRIMARY,
                         TEXT_SECONDARY, TEXT_MUTED, TEXT_FAINT)

N1 = sys.argv[1] if len(sys.argv) > 1 else 'Alvarez, Yordan'
N2 = sys.argv[2] if len(sys.argv) > 2 else 'Chapman, Matt'
THROUGH = '2026-06-02'
OUT = '/Users/wallyhuron/Downloads/_la_spray_compare.png'
XLIM = (-50, 50); YLIM = (-20, 60); BARREL_COLOR = '#b81d24'
GRAY_CMAP = LinearSegmentedColormap.from_list(
    'gray', [(0.0, (0.95, 0.93, 0.88)), (0.45, (0.64, 0.64, 0.67)), (1.0, (0.20, 0.20, 0.24))], N=256)


def zone_bounds(bats):
    if bats == 'L':
        return {'pull':(30,50),'pull_side':(15,30),'center_pull':(0,15),
                'center_oppo':(-15,0),'oppo_side':(-30,-15),'oppo':(-50,-30)}
    return {'pull':(-50,-30),'pull_side':(-30,-15),'center_pull':(-15,0),
            'center_oppo':(0,15),'oppo_side':(15,30),'oppo':(30,50)}


def kde_grid(pts, hx=7.0, hy=5.0, nx=130, ny=110):
    gx=np.linspace(*XLIM,nx); gy=np.linspace(*YLIM,ny); GX,GY=np.meshgrid(gx,gy)
    Z=np.zeros_like(GX)
    for (x,y) in pts: Z+=np.exp(-(((GX-x)/hx)**2+((GY-y)/hy)**2)/2.0)
    return gx,gy,Z


def load_hitter(name, allp, meta, lb):
    row = next((r for r in lb if r.get('hitter') == name), {})
    bats = row.get('stands') or 'R'
    bip = []
    for p in allp:
        if p.get('Batter') != name or str(p.get('Game Date','')) > THROUGH: continue
        if p.get('Description') != 'In Play': continue
        bbt = str(p.get('BBType','')).strip()
        if not bbt or bbt.startswith('bunt'): continue
        la = sf(p.get('LaunchAngle')); ang = spray_angle(sf(p.get('HC_X')), sf(p.get('HC_Y')))
        if la is None or ang is None: continue
        bip.append((ang, max(-20,min(60,la)), sf(p.get('ExitVelo')), str(p.get('Barrel','')).strip()=='6'))
    _, hand_zones, pool_zones = build_sacq_lookup(meta, bats)
    bounds = zone_bounds(bats); cells = []
    for (sd,lb_),z in {**pool_zones,**hand_zones}.items():
        v = z.get('wobacon') if z.get('wobacon') is not None else z.get('woba')
        if v is None or sd not in bounds or lb_>=len(LA_BINS): continue
        bx=bounds[sd]; ly=LA_BINS[lb_]
        cells.append((min(bx),max(-20,ly[0]),abs(bx[1]-bx[0]),min(60,ly[1])-max(-20,ly[0]),v))
    return row, bats, bip, cells


def render_optionb(ax, name, row, bats, bip, cells):
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_facecolor(BG)
    barrels = [(b[0],b[1],b[2]) for b in bip if b[3]]
    allpts = [(b[0],b[1]) for b in bip]
    sub = (f"{name.split(', ')[1]} {name.split(', ')[0]}  ·  {bats}HH  ·  {len(bip)} BIP  ·  "
           f"{len(barrels)} barrels  ·  xwOBAsp {('%.3f'%row.get('xwOBAsp')).lstrip('0') if row.get('xwOBAsp') else '—'}"
           f"  ·  wRC+ {row.get('wRCplus','—')}")
    ax.set_title(sub, fontsize=13, fontweight='bold', color=TEXT_SECONDARY, fontfamily='DIN Condensed', pad=10)
    ax.set_xlabel('Oppo   •   Spray Angle   •   Pull', fontsize=10, color=TEXT_MUTED)
    ax.set_ylabel('Launch Angle', fontsize=10, color=TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    for s in ax.spines.values(): s.set_color(TEXT_FAINT)
    gx,gy,Z = kde_grid(allpts); Zn = (Z/Z.max() if Z.max()>0 else Z) ** 1.25
    ax.imshow(Zn, origin='lower', extent=[*XLIM,*YLIM], aspect='auto', cmap=GRAY_CMAP, alpha=0.95, zorder=1)
    for (x0,y0,w,h,v) in cells:
        if v >= 0.50:
            ax.add_patch(Rectangle((x0,y0),w,h, facecolor=HC.WOBA_CMAP(min(1.0,v)),
                                   alpha=0.62, edgecolor=GRID_COLOR, linewidth=0.4, zorder=3))
    for (x,y,ev) in barrels:
        s = 70 + (max(85,min(115,ev))-85)*6 if ev else 90
        ax.scatter([x],[y], s=s, c=BARREL_COLOR, edgecolors='#1a1612', linewidths=0.7, zorder=10, alpha=0.95)
    ax.legend(handles=[
        Line2D([0],[0],marker='o',color='none',markerfacecolor=BARREL_COLOR,markersize=10,label='Barrel'),
        Line2D([0],[0],marker='s',color='none',markerfacecolor=(0.85,0.30,0.22),markersize=10,label='Damage zone (table color)'),
        Line2D([0],[0],marker='s',color='none',markerfacecolor=(0.42,0.42,0.45),markersize=10,label='Darker = more BIP'),
    ], loc='upper right', fontsize=8, frameon=False, labelcolor=TEXT_MUTED, handletextpad=0.5)


allp = load_pitch_data(); meta = load_metadata(); lb = HC.load_hitter_leaderboard()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8.6)); fig.patch.set_facecolor(BG)
for ax, nm in ((ax1, N1), (ax2, N2)):
    row, bats, bip, cells = load_hitter(nm, allp, meta, lb)
    print(f'{nm}: bats={bats} BIP={len(bip)} barrels={sum(1 for b in bip if b[3])}')
    render_optionb(ax, nm, row, bats, bip, cells)
fig.suptitle('LA × Spray (Option B) — contact density + damage zones + barrels   ·   through Jun 2, 2026',
             fontsize=15, fontweight='bold', color=TEXT_PRIMARY, fontfamily='DIN Condensed', y=0.985)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(OUT, dpi=140, facecolor=BG, bbox_inches='tight')
print(f'wrote {OUT}')
