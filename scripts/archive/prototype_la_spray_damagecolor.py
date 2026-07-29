#!/usr/bin/env python3
"""Option B variant test: can the damage zones be colored like the current table?

LEFT  = naive — keep the WARM density and color the damage zones with the
        table's WOBA_CMAP. Warm-on-warm: the value color muddies into the
        density (worst exactly where he hits most).
RIGHT = fix  — NEUTRAL (gray) density so the warm value-colored damage zones
        pop cleanly. gray = where he hits, warm-red = valuable zones.

Both keep barrels as red dots. Comparison mockup only.
Output -> ~/Downloads/_la_spray_damagecolor.png
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

NAME = sys.argv[1] if len(sys.argv) > 1 else 'Wood, James'
THROUGH = '2026-06-02'
OUT = '/Users/wallyhuron/Downloads/_la_spray_damagecolor.png'
XLIM = (-50, 50); YLIM = (-20, 60); BARREL_COLOR = '#b81d24'

def zone_bounds(bats):
    if bats == 'L':
        return {'pull': (30,50),'pull_side':(15,30),'center_pull':(0,15),
                'center_oppo':(-15,0),'oppo_side':(-30,-15),'oppo':(-50,-30)}
    return {'pull':(-50,-30),'pull_side':(-30,-15),'center_pull':(-15,0),
            'center_oppo':(0,15),'oppo_side':(15,30),'oppo':(30,50)}

def kde_grid(pts, hx=7.0, hy=5.0, nx=130, ny=110):
    gx=np.linspace(*XLIM,nx); gy=np.linspace(*YLIM,ny); GX,GY=np.meshgrid(gx,gy)
    Z=np.zeros_like(GX)
    for (x,y) in pts: Z+=np.exp(-(((GX-x)/hx)**2+((GY-y)/hy)**2)/2.0)
    return gx,gy,Z

def style(ax,title):
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_facecolor(BG)
    ax.set_title(title,fontsize=15,fontweight='bold',color=TEXT_SECONDARY,fontfamily='DIN Condensed',pad=10)
    ax.set_xlabel('Oppo   •   Spray Angle   •   Pull',fontsize=10,color=TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED,labelsize=8)
    for s in ax.spines.values(): s.set_color(TEXT_FAINT)

# Load
allp=load_pitch_data(); meta=load_metadata()
hrow=next((r for r in HC.load_hitter_leaderboard() if r.get('hitter')==NAME),{})
bats=hrow.get('stands') or 'L'
bip=[]
for p in allp:
    if p.get('Batter')!=NAME or str(p.get('Game Date',''))>THROUGH: continue
    if p.get('Description')!='In Play': continue
    bbt=str(p.get('BBType','')).strip()
    if not bbt or bbt.startswith('bunt'): continue
    la=sf(p.get('LaunchAngle')); ang=spray_angle(sf(p.get('HC_X')),sf(p.get('HC_Y')))
    if la is None or ang is None: continue
    bip.append((ang,max(-20,min(60,la)),sf(p.get('ExitVelo')),str(p.get('Barrel','')).strip()=='6'))
allpts=[(b[0],b[1]) for b in bip]; barrels=[(b[0],b[1],b[2]) for b in bip if b[3]]
print(f'{NAME}: bats={bats} BIP={len(bip)} barrels={len(barrels)}')

_,hand_zones,pool_zones=build_sacq_lookup(meta,bats); bounds=zone_bounds(bats)
cells=[]
for (sd,lb_),z in {**pool_zones,**hand_zones}.items():
    v=z.get('wobacon') if z.get('wobacon') is not None else z.get('woba')
    if v is None or sd not in bounds or lb_>=len(LA_BINS): continue
    bx=bounds[sd]; ly=LA_BINS[lb_]
    cells.append((min(bx),max(-20,ly[0]),abs(bx[1]-bx[0]),min(60,ly[1])-max(-20,ly[0]),v))

gx,gy,Z=kde_grid(allpts); Zn=Z/Z.max() if Z.max()>0 else Z

WARM=LinearSegmentedColormap.from_list('warm',[(0,(0.94,0.91,0.85)),(0.35,(0.90,0.78,0.55)),
        (0.7,(0.84,0.55,0.20)),(1,(0.55,0.20,0.10))],N=256)
GRAY=LinearSegmentedColormap.from_list('gray',[(0,(0.95,0.93,0.88)),(0.45,(0.66,0.66,0.69)),
        (1,(0.26,0.26,0.30))],N=256)

def draw_barrels(ax):
    for (x,y,ev) in barrels:
        s=70+(max(85,min(115,ev))-85)*6 if ev else 90
        ax.scatter([x],[y],s=s,c=BARREL_COLOR,edgecolors='#1a1612',linewidths=0.7,zorder=10,alpha=0.95)

def damage_fills(ax,alpha):
    for (x0,y0,w,h,v) in cells:
        if v>=0.50:
            ax.add_patch(Rectangle((x0,y0),w,h,facecolor=HC.WOBA_CMAP(min(1.0,v)),
                         alpha=alpha,edgecolor=GRID_COLOR,linewidth=0.4,zorder=3))

fig,(axA,axB)=plt.subplots(1,2,figsize=(15,8.6)); fig.patch.set_facecolor(BG)

# A — naive: warm density + colored damage zones
style(axA,'NAIVE — warm density + table-colored damage zones (clash)')
axA.imshow(Zn,origin='lower',extent=[*XLIM,*YLIM],aspect='auto',cmap=WARM,alpha=0.92,zorder=1)
damage_fills(axA,0.55); draw_barrels(axA)

# B — fix: gray density + colored damage zones
style(axB,'FIX — neutral gray density + table-colored damage zones')
axB.imshow(Zn,origin='lower',extent=[*XLIM,*YLIM],aspect='auto',cmap=GRAY,alpha=0.92,zorder=1)
damage_fills(axB,0.68); draw_barrels(axB)
axB.legend(handles=[
    Line2D([0],[0],marker='o',color='none',markerfacecolor=BARREL_COLOR,markersize=10,label='Barrel'),
    Line2D([0],[0],marker='s',color='none',markerfacecolor=(0.85,0.30,0.22),markersize=10,label='Damage zone (table color)'),
    Line2D([0],[0],marker='s',color='none',markerfacecolor=(0.4,0.4,0.43),markersize=10,label='Darker = more BIP'),
],loc='lower right',fontsize=8,frameon=False,labelcolor=TEXT_MUTED,handletextpad=0.5)

fig.suptitle(f'{NAME}  ·  Option B with colored damage zones: warm density clashes, gray density works',
             fontsize=14,fontweight='bold',color=TEXT_PRIMARY,fontfamily='DIN Condensed',y=0.985)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(OUT,dpi=140,facecolor=BG,bbox_inches='tight')
print(f'wrote {OUT}')
