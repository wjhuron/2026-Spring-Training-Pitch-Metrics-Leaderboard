#!/usr/bin/env python3
"""Zone-alpha sweep on the ACTUAL Option B: gray contact density + table-colored
DAMAGE ZONES ONLY (not the full blue->red value map) + red barrels.

Only the damage-zone fill alpha changes: A's 0.42, middle 0.55, current 0.70.

    python3 scripts/prototype_la_spray_alpha.py "Alvarez, Yordan"
Output -> ~/Downloads/_la_spray_alpha.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import HitterCards as HC
from HitterCards import (load_pitch_data, load_metadata, build_sacq_lookup,
                         spray_angle, LA_BINS, sf, BG, GRID_COLOR, TEXT_PRIMARY,
                         TEXT_SECONDARY, TEXT_MUTED, TEXT_FAINT)

NAME = sys.argv[1] if len(sys.argv) > 1 else 'Alvarez, Yordan'
THROUGH = '2026-06-02'; OUT = '/Users/wallyhuron/Downloads/_la_spray_alpha.png'
XLIM = (-50, 50); YLIM = (-20, 60); BARREL_COLOR = '#b81d24'
ALPHAS = [(0.42, "A's alpha (0.42)"), (0.55, 'middle (0.55)'), (0.70, 'current / pop (0.70)')]
GRAY_CMAP = LinearSegmentedColormap.from_list(
    'gray', [(0.0, (0.95, 0.93, 0.88)), (0.45, (0.64, 0.64, 0.67)), (1.0, (0.20, 0.20, 0.24))], N=256)

def zone_bounds(bats):
    if bats == 'L':
        return {'pull':(30,50),'pull_side':(15,30),'center_pull':(0,15),'center_oppo':(-15,0),'oppo_side':(-30,-15),'oppo':(-50,-30)}
    return {'pull':(-50,-30),'pull_side':(-30,-15),'center_pull':(-15,0),'center_oppo':(0,15),'oppo_side':(15,30),'oppo':(30,50)}

def kde_grid(pts, hx=7.0, hy=5.0, nx=160, ny=130):
    gx=np.linspace(*XLIM,nx); gy=np.linspace(*YLIM,ny); GX,GY=np.meshgrid(gx,gy); Z=np.zeros_like(GX)
    for (x,y) in pts: Z+=np.exp(-(((GX-x)/hx)**2+((GY-y)/hy)**2)/2.0)
    return Z

# Load
allp=load_pitch_data(); meta=load_metadata()
row=next((r for r in HC.load_hitter_leaderboard() if r.get('hitter')==NAME),{}); bats=row.get('stands') or 'R'
bip=[]
for p in allp:
    if p.get('Batter')!=NAME or str(p.get('Game Date',''))>THROUGH: continue
    if p.get('Description')!='In Play': continue
    bbt=str(p.get('BBType','')).strip()
    if not bbt or bbt.startswith('bunt'): continue
    la=sf(p.get('LaunchAngle')); ang=spray_angle(sf(p.get('HC_X')),sf(p.get('HC_Y')))
    if la is None or ang is None: continue
    bip.append((ang,max(-20,min(60,la)),sf(p.get('ExitVelo')),str(p.get('Barrel','')).strip()=='6'))
barrels=[(b[0],b[1],b[2]) for b in bip if b[3]]; allpts=[(b[0],b[1]) for b in bip]
_,hand_zones,pool_zones=build_sacq_lookup(meta,bats); bounds=zone_bounds(bats)
# damage zones only (v >= .500)
dmg=[]
for (sd,lb_),z in {**pool_zones,**hand_zones}.items():
    v=z.get('wobacon') if z.get('wobacon') is not None else z.get('woba')
    if v is None or v<0.50 or sd not in bounds or lb_>=len(LA_BINS): continue
    bx=bounds[sd]; ly=LA_BINS[lb_]
    dmg.append((min(bx),max(-20,ly[0]),abs(bx[1]-bx[0]),min(60,ly[1])-max(-20,ly[0]),v))
Z=kde_grid(allpts); Zn=(Z/Z.max() if Z.max()>0 else Z)**1.25
print(f'{NAME}: bats={bats} BIP={len(bip)} barrels={len(barrels)} damage-zones={len(dmg)}')

def panel(ax, zone_alpha, label):
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_facecolor(BG)
    ax.set_title(label, fontsize=14, fontweight='bold', color=TEXT_SECONDARY, fontfamily='DIN Condensed', pad=10)
    ax.set_xlabel('Oppo   •   Spray Angle   •   Pull', fontsize=9, color=TEXT_MUTED); ax.tick_params(colors=TEXT_MUTED, labelsize=7)
    for s in ax.spines.values(): s.set_color(TEXT_FAINT)
    # gray contact density UNDER
    ax.imshow(Zn, origin='lower', extent=[*XLIM,*YLIM], aspect='auto', cmap=GRAY_CMAP, alpha=0.95, zorder=1)
    # table-colored DAMAGE ZONES ONLY, alpha varies
    for (x0,y0,w,h,v) in dmg:
        ax.add_patch(Rectangle((x0,y0),w,h,facecolor=HC.WOBA_CMAP(min(1.0,v)),alpha=zone_alpha,edgecolor=GRID_COLOR,linewidth=0.4,zorder=3))
    for (x,y,ev) in barrels:
        s=70+(max(85,min(115,ev))-85)*6 if ev else 90
        ax.scatter([x],[y],s=s,c=BARREL_COLOR,edgecolors='#1a1612',linewidths=0.7,zorder=10,alpha=0.95)

fig,axes=plt.subplots(1,3,figsize=(19,8.3)); fig.patch.set_facecolor(BG)
for ax,(a,lbl) in zip(axes,ALPHAS): panel(ax,a,lbl)
fig.suptitle(f'{NAME}  ·  Option B (gray density + damage-zones-ONLY + barrels) — damage-zone alpha sweep',
             fontsize=15,fontweight='bold',color=TEXT_PRIMARY,fontfamily='DIN Condensed',y=0.985)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(OUT,dpi=135,facecolor=BG,bbox_inches='tight')
print(f'wrote {OUT}')
