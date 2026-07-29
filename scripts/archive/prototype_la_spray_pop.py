#!/usr/bin/env python3
"""Shadow density over the FULL, popping value map (like the current table).

Full WOBA_CMAP value map (all zones, current alphas — pops off the cream) +
density as a neutral dark transparency-ramped SHADOW on top + red barrels.
The shadow is hue-neutral, so it darkens his hot spots without clashing with
the value colors.

    python3 scripts/prototype_la_spray_pop.py "Alvarez, Yordan" "Chapman, Matt"
Output -> ~/Downloads/_la_spray_pop.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import HitterCards as HC
from HitterCards import (load_pitch_data, load_metadata, build_sacq_lookup,
                         spray_angle, LA_BINS, sf, BG, GRID_COLOR, TEXT_PRIMARY,
                         TEXT_SECONDARY, TEXT_MUTED, TEXT_FAINT)

N1 = sys.argv[1] if len(sys.argv) > 1 else 'Alvarez, Yordan'
N2 = sys.argv[2] if len(sys.argv) > 2 else 'Chapman, Matt'
THROUGH = '2026-06-02'
OUT = '/Users/wallyhuron/Downloads/_la_spray_pop.png'
XLIM = (-50, 50); YLIM = (-20, 60); BARREL_COLOR = '#b81d24'


def zone_bounds(bats):
    if bats == 'L':
        return {'pull':(30,50),'pull_side':(15,30),'center_pull':(0,15),'center_oppo':(-15,0),'oppo_side':(-30,-15),'oppo':(-50,-30)}
    return {'pull':(-50,-30),'pull_side':(-30,-15),'center_pull':(-15,0),'center_oppo':(0,15),'oppo_side':(15,30),'oppo':(30,50)}


def kde_grid(pts, hx=7.0, hy=5.0, nx=160, ny=130):
    gx=np.linspace(*XLIM,nx); gy=np.linspace(*YLIM,ny); GX,GY=np.meshgrid(gx,gy); Z=np.zeros_like(GX)
    for (x,y) in pts: Z+=np.exp(-(((GX-x)/hx)**2+((GY-y)/hy)**2)/2.0)
    return Z


def draw_full_value_map(ax, bats, hand_zones, pool_zones):
    """Reproduce the current card's full value map (every zone, WOBA_CMAP,
    0.70 alpha for ≥20-BIP zones else 0.22) so it POPS off the cream."""
    bounds = zone_bounds(bats)
    def fill(sd, lb, z):
        if sd not in bounds or lb >= len(LA_BINS): return
        v = z.get('wobacon') if z.get('wobacon') is not None else z.get('woba')
        if v is None: return
        bx = bounds[sd]; ly = LA_BINS[lb]; lo = max(-20, ly[0]); hi = min(60, ly[1])
        a = 0.22 if z.get('count', 0) < 20 else 0.70
        ax.add_patch(Rectangle((min(bx), lo), abs(bx[1]-bx[0]), hi-lo,
                               facecolor=HC.WOBA_CMAP(min(1.0, v)), alpha=a,
                               edgecolor=GRID_COLOR, linewidth=0.3, zorder=1))
    for (sd, lb), z_hand in hand_zones.items():
        z = z_hand if z_hand.get('count', 0) >= 20 else pool_zones.get((sd, lb))
        if z: fill(sd, lb, z)
    for (sd, lb), z in pool_zones.items():
        if (sd, lb) not in hand_zones: fill(sd, lb, z)
    # zone-edge grid
    for x in sorted({e for v in bounds.values() for e in v}):
        if XLIM[0] < x < XLIM[1]: ax.axvline(x, color=GRID_COLOR, lw=1.0, alpha=0.6, zorder=2)
    for y in sorted({e for r in LA_BINS for e in r}):
        if YLIM[0] < y < YLIM[1]: ax.axhline(y, color=GRID_COLOR, lw=1.0, alpha=0.6, zorder=2)


def render(ax, name, allp, meta, lb):
    row = next((r for r in lb if r.get('hitter') == name), {}); bats = row.get('stands') or 'R'
    bip = []
    for p in allp:
        if p.get('Batter') != name or str(p.get('Game Date','')) > THROUGH: continue
        if p.get('Description') != 'In Play': continue
        bbt = str(p.get('BBType','')).strip()
        if not bbt or bbt.startswith('bunt'): continue
        la = sf(p.get('LaunchAngle')); ang = spray_angle(sf(p.get('HC_X')), sf(p.get('HC_Y')))
        if la is None or ang is None: continue
        bip.append((ang, max(-20,min(60,la)), sf(p.get('ExitVelo')), str(p.get('Barrel','')).strip()=='6'))
    barrels = [(b[0],b[1],b[2]) for b in bip if b[3]]; allpts = [(b[0],b[1]) for b in bip]
    _, hand_zones, pool_zones = build_sacq_lookup(meta, bats)
    print(f'{name}: bats={bats} BIP={len(bip)} barrels={len(barrels)}')

    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_facecolor(BG)
    sub = (f"{name.split(', ')[1]} {name.split(', ')[0]}  ·  {bats}HH  ·  {len(bip)} BIP  ·  {len(barrels)} barrels"
           f"  ·  xwOBAsp {('%.3f'%row['xwOBAsp']).lstrip('0') if row.get('xwOBAsp') else '—'}  ·  wRC+ {row.get('wRCplus','—')}")
    ax.set_title(sub, fontsize=13, fontweight='bold', color=TEXT_SECONDARY, fontfamily='DIN Condensed', pad=10)
    ax.set_xlabel('Oppo   •   Spray Angle   •   Pull', fontsize=10, color=TEXT_MUTED)
    ax.set_ylabel('Launch Angle', fontsize=10, color=TEXT_MUTED); ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    for s in ax.spines.values(): s.set_color(TEXT_FAINT)

    draw_full_value_map(ax, bats, hand_zones, pool_zones)
    Z = kde_grid(allpts); Zn = Z / Z.max() if Z.max() > 0 else Z
    shadow = np.zeros((*Zn.shape, 4)); shadow[...,0]=0.10; shadow[...,1]=0.10; shadow[...,2]=0.13
    shadow[...,3] = (Zn ** 0.9) * 0.55
    ax.imshow(shadow, origin='lower', extent=[*XLIM,*YLIM], aspect='auto', zorder=5, interpolation='bilinear')
    for (x,y,ev) in barrels:
        s = 70 + (max(85,min(115,ev))-85)*6 if ev else 90
        ax.scatter([x],[y], s=s, c=BARREL_COLOR, edgecolors='#1a1612', linewidths=0.7, zorder=10, alpha=0.95)
    ax.legend(handles=[
        Line2D([0],[0],marker='o',color='none',markerfacecolor=BARREL_COLOR,markersize=10,label='Barrel'),
        Line2D([0],[0],marker='s',color='none',markerfacecolor=(0.15,0.15,0.18),markersize=10,label='Shadow = more BIP'),
    ], loc='upper right', fontsize=8, frameon=False, labelcolor=TEXT_MUTED, handletextpad=0.5)


allp = load_pitch_data(); meta = load_metadata(); lb = HC.load_hitter_leaderboard()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 8.6)); fig.patch.set_facecolor(BG)
render(a1, N1, allp, meta, lb); render(a2, N2, allp, meta, lb)
fig.suptitle('LA × Spray — full value map (popping) + density shadow + barrels   ·   through Jun 2, 2026',
             fontsize=15, fontweight='bold', color=TEXT_PRIMARY, fontfamily='DIN Condensed', y=0.985)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(OUT, dpi=140, facecolor=BG, bbox_inches='tight')
print(f'wrote {OUT}')
