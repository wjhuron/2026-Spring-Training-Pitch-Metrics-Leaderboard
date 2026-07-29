#!/usr/bin/env python3
"""Option B layering: density UNDER the damage zones (current) vs density ON TOP.

LEFT  = density under the bins (current — bins cover it).
RIGHT = density ON TOP of the bins, as a transparency-ramped charcoal overlay
        (clear where he doesn't hit, darkens where he concentrates), so his
        contact reads in the foreground while the bins still show through.

Damage-zones-only (table color) + barrels, both panels.
    python3 scripts/prototype_la_spray_ontop.py "Alvarez, Yordan"
Output -> ~/Downloads/_la_spray_ontop.png
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
THROUGH = '2026-06-02'; OUT = '/Users/wallyhuron/Downloads/_la_spray_ontop.png'
XLIM = (-50, 50); YLIM = (-20, 60); BARREL_COLOR = '#b81d24'; ZONE_ALPHA = 0.62
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
dmg=[]
for (sd,lb_),z in {**pool_zones,**hand_zones}.items():
    v=z.get('wobacon') if z.get('wobacon') is not None else z.get('woba')
    if v is None or v<0.50 or sd not in bounds or lb_>=len(LA_BINS): continue
    bx=bounds[sd]; ly=LA_BINS[lb_]
    dmg.append((min(bx),max(-20,ly[0]),abs(bx[1]-bx[0]),min(60,ly[1])-max(-20,ly[0]),v))
Z=kde_grid(allpts); Zn=(Z/Z.max() if Z.max()>0 else Z)
print(f'{NAME}: bats={bats} BIP={len(bip)} barrels={len(barrels)}')

def setup(ax,title):
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_facecolor(BG)
    ax.set_title(title, fontsize=14, fontweight='bold', color=TEXT_SECONDARY, fontfamily='DIN Condensed', pad=10)
    ax.set_xlabel('Oppo   •   Spray Angle   •   Pull', fontsize=9, color=TEXT_MUTED); ax.tick_params(colors=TEXT_MUTED, labelsize=7)
    for s in ax.spines.values(): s.set_color(TEXT_FAINT)
def zones(ax):
    for (x0,y0,w,h,v) in dmg: ax.add_patch(Rectangle((x0,y0),w,h,facecolor=HC.WOBA_CMAP(min(1.0,v)),alpha=ZONE_ALPHA,edgecolor=GRID_COLOR,linewidth=0.4,zorder=3))
def barr(ax):
    for (x,y,ev) in barrels:
        s=70+(max(85,min(115,ev))-85)*6 if ev else 90
        ax.scatter([x],[y],s=s,c=BARREL_COLOR,edgecolors='#1a1612',linewidths=0.7,zorder=12,alpha=0.95)

fig,(a1,a2)=plt.subplots(1,2,figsize=(15,8.6)); fig.patch.set_facecolor(BG)

# LEFT — panel 1 (vivid bins, gray density under) for reference
setup(a1,'panel 1 — bins vivid (density under)')
a1.imshow(Zn**1.25,origin='lower',extent=[*XLIM,*YLIM],aspect='auto',cmap=GRAY_CMAP,alpha=0.95,zorder=1)
zones(a1); barr(a1)

# RIGHT — density of 3 ON TOP, but DARK (hue-preserving) so the bins keep their
# exact colors (just shaded darker where he concentrates), never grayed.
setup(a2,'density ON TOP (dark), bins keep their colors')
zones(a2)
dens=np.zeros((*Zn.shape,4)); dens[...,0]=0.12; dens[...,1]=0.12; dens[...,2]=0.15
dens[...,3]=(Zn**0.7)*0.80
a2.imshow(dens,origin='lower',extent=[*XLIM,*YLIM],aspect='auto',zorder=6,interpolation='bilinear')
barr(a2)

fig.suptitle(f'{NAME}  ·  Option B — density under vs ON TOP of the damage zones  ·  {len(bip)} BIP / {len(barrels)} barrels',
             fontsize=15,fontweight='bold',color=TEXT_PRIMARY,fontfamily='DIN Condensed',y=0.985)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(OUT,dpi=140,facecolor=BG,bbox_inches='tight')
print(f'wrote {OUT}')
