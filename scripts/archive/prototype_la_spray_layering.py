#!/usr/bin/env python3
"""How to show BOTH the colored damage zones AND the contact density without
one burying the other. Three layerings, one hitter (default Alvarez):

  1 CURRENT  — colored zones painted OVER the gray density (zones win).
  2 SHADOW   — colored zones as the base; density as a transparency-ramped
               DARK shadow on top (opacity = frequency, clear where he doesn't
               hit). Zones keep full color; his hot spots darken. <- recommended
  3 CONTOUR  — colored zones (full); density as contour LINES on top.

    python3 scripts/prototype_la_spray_layering.py "Alvarez, Yordan"
Output -> ~/Downloads/_la_spray_layering.png
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
THROUGH = '2026-06-02'
OUT = '/Users/wallyhuron/Downloads/_la_spray_layering.png'
XLIM = (-50, 50); YLIM = (-20, 60); BARREL_COLOR = '#b81d24'
GRAY_CMAP = LinearSegmentedColormap.from_list(
    'gray', [(0.0, (0.95, 0.93, 0.88)), (0.45, (0.64, 0.64, 0.67)), (1.0, (0.20, 0.20, 0.24))], N=256)

def zone_bounds(bats):
    if bats == 'L':
        return {'pull':(30,50),'pull_side':(15,30),'center_pull':(0,15),'center_oppo':(-15,0),'oppo_side':(-30,-15),'oppo':(-50,-30)}
    return {'pull':(-50,-30),'pull_side':(-30,-15),'center_pull':(-15,0),'center_oppo':(0,15),'oppo_side':(15,30),'oppo':(30,50)}

def kde_grid(pts, hx=7.0, hy=5.0, nx=160, ny=130):
    gx=np.linspace(*XLIM,nx); gy=np.linspace(*YLIM,ny); GX,GY=np.meshgrid(gx,gy); Z=np.zeros_like(GX)
    for (x,y) in pts: Z+=np.exp(-(((GX-x)/hx)**2+((GY-y)/hy)**2)/2.0)
    return gx,gy,Z

# Load hitter
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
allpts=[(b[0],b[1]) for b in bip]; barrels=[(b[0],b[1],b[2]) for b in bip if b[3]]
_,hand_zones,pool_zones=build_sacq_lookup(meta,bats); bounds=zone_bounds(bats)
cells=[]
for (sd,lb_),z in {**pool_zones,**hand_zones}.items():
    v=z.get('wobacon') if z.get('wobacon') is not None else z.get('woba')
    if v is None or sd not in bounds or lb_>=len(LA_BINS): continue
    bx=bounds[sd]; ly=LA_BINS[lb_]
    cells.append((min(bx),max(-20,ly[0]),abs(bx[1]-bx[0]),min(60,ly[1])-max(-20,ly[0]),v))
gx,gy,Z=kde_grid(allpts); Zn=(Z/Z.max() if Z.max()>0 else Z)
print(f'{NAME}: bats={bats} BIP={len(bip)} barrels={len(barrels)}')

def base(ax,title):
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_facecolor(BG)
    ax.set_title(title,fontsize=14,fontweight='bold',color=TEXT_SECONDARY,fontfamily='DIN Condensed',pad=10)
    ax.set_xlabel('Oppo   •   Spray Angle   •   Pull',fontsize=9,color=TEXT_MUTED); ax.tick_params(colors=TEXT_MUTED,labelsize=7)
    for s in ax.spines.values(): s.set_color(TEXT_FAINT)
def zones(ax,a):
    for (x0,y0,w,h,v) in cells:
        if v>=0.50: ax.add_patch(Rectangle((x0,y0),w,h,facecolor=HC.WOBA_CMAP(min(1.0,v)),alpha=a,edgecolor=GRID_COLOR,linewidth=0.4,zorder=3))
def barr(ax):
    for (x,y,ev) in barrels:
        s=70+(max(85,min(115,ev))-85)*6 if ev else 90
        ax.scatter([x],[y],s=s,c=BARREL_COLOR,edgecolors='#1a1612',linewidths=0.7,zorder=10,alpha=0.95)

fig,(a1,a2,a3)=plt.subplots(1,3,figsize=(19,8.3)); fig.patch.set_facecolor(BG)

# 1 CURRENT: zones over gray density
base(a1,'1 — CURRENT: zones OVER density (zones bury it)')
a1.imshow(Zn**1.25,origin='lower',extent=[*XLIM,*YLIM],aspect='auto',cmap=GRAY_CMAP,alpha=0.95,zorder=1)
zones(a1,0.62); barr(a1)

# 2 SHADOW: zones base + dark transparency-ramped density on top
base(a2,'2 — SHADOW: density over zones as a dark, transparent overlay')
zones(a2,0.66)
shadow=np.zeros((*Zn.shape,4)); shadow[...,0]=0.12; shadow[...,1]=0.12; shadow[...,2]=0.15
shadow[...,3]=(Zn**0.9)*0.55                      # opacity = frequency, peak 0.55
a2.imshow(shadow,origin='lower',extent=[*XLIM,*YLIM],aspect='auto',zorder=5,interpolation='bilinear')
barr(a2)

# 3 CONTOUR: zones full + density contour lines
base(a3,'3 — CONTOUR: zones full color + density lines')
zones(a3,0.66)
lv=np.linspace(Zn.max()*0.22,Zn.max()*0.9,4)
a3.contour(gx,gy,Zn,levels=lv,colors='#ffffff',linewidths=2.6,alpha=0.85,zorder=5)
a3.contour(gx,gy,Zn,levels=lv,colors='#1a1612',linewidths=1.1,alpha=0.9,zorder=6)
barr(a3)

fig.suptitle(f'{NAME}  ·  showing density WITHOUT burying the damage zones  ·  {len(bip)} BIP / {len(barrels)} barrels',
             fontsize=15,fontweight='bold',color=TEXT_PRIMARY,fontfamily='DIN Condensed',y=0.985)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(OUT,dpi=135,facecolor=BG,bbox_inches='tight')
print(f'wrote {OUT}')
