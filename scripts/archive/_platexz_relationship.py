"""Characterize sheet(=feed) vs Savant per-pitch difference for PlateX & PlateZ.
READ-ONLY. Uses the cached full pkl + auto-ball-aware feed numbering."""
import os, sys, time, warnings
warnings.filterwarnings('ignore')
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
import pandas as pd, gspread, backfill_supplement as B
from collections import defaultdict

df=pd.read_pickle('data/_statcast2026_full.pkl')
print("pkl rows:", len(df))
cols=['game_pk','at_bat_number','pitch_number','description','plate_x','plate_z','release_speed']
bypa=defaultdict(list)
for r in df[cols].itertuples(index=False):
    try: bypa[(int(r.game_pk),int(r.at_bat_number))].append((int(r.pitch_number),r))
    except: continue
aligned={}
for k,evs in bypa.items():
    evs.sort(key=lambda t:t[0]); feed=0
    for pn,r in evs:
        if 'automatic' in str(r.description or '').lower(): continue
        feed+=1; aligned[(k[0],k[1],feed)]=r

def sf(x):
    try:
        v=float(x); return v if v==v else None
    except: return None

gc=gspread.service_account()
dx=[]; dz=[]; tailx=0; tailz=0; both=0; matched=0
for label,sid in B.SPREADSHEET_IDS.items():
    sh=gc.open_by_key(sid)
    for ws in sh.worksheets():
        t=ws.title.upper()
        if t not in B.ALL_TRACKED_TEAMS or t in ('ROC','AAA','FCL'): continue
        time.sleep(0.6); vals=ws.get_all_values()
        if not vals or 'PitchID' not in vals[0]: continue
        ci={n:j for j,n in enumerate(vals[0]) if n}
        for r in vals[1:]:
            pid=r[ci['PitchID']] if ci['PitchID']<len(r) else ''
            p=pid.split('_')
            if len(p)!=3: continue
            try: key=(int(p[0]),int(p[1]),int(p[2]))
            except: continue
            sr=aligned.get(key)
            if sr is None: continue
            px=sf(r[ci['PlateX']]); pz=sf(r[ci['PlateZ']]); vv=sf(r[ci['Velocity']])
            spx=sf(sr.plate_x); spz=sf(sr.plate_z); svv=sf(sr.release_speed)
            # validate join by velocity
            if vv is not None and svv is not None and abs(vv-svv)>3: continue
            matched+=1
            ax=az=None
            if px is not None and spx is not None: ax=px-spx; dx.append(ax); tailx+= (abs(ax)>0.2)
            if pz is not None and spz is not None: az=pz-spz; dz.append(az); tailz+= (abs(az)>0.2)
            if ax is not None and az is not None and abs(ax)>0.2 and abs(az)>0.2: both+=1

def stats(a,name):
    a=sorted(a); n=len(a)
    def q(p): return a[int(p*(n-1))]
    body=[v for v in a if abs(v)<0.2]; nb=len(body)
    import statistics as st
    print(f"\n{name}: n={n}")
    print(f"  ALL   median={st.median(a):+.4f} mean={st.mean(a):+.4f}  p05={q(.05):+.3f} p95={q(.95):+.3f}")
    print(f"  BODY(|d|<0.2, n={nb}={100*nb/n:.1f}%): median={st.median(body):+.4f} mean={st.mean(body):+.4f} sd={st.pstdev(body):.4f}")
    within=lambda t: 100*sum(abs(v)<=t for v in a)/n
    print(f"  within 0.02={within(0.02):.1f}%  0.05={within(0.05):.1f}%  0.10={within(0.10):.1f}%  |d|>0.2 (corrupt tail)={sum(abs(v)>0.2 for v in a)}")

print(f"\nmatched (velo-validated): {matched}")
stats(dx,"PlateX  (sheet-Savant)")
stats(dz,"PlateZ  (sheet-Savant)")
print(f"\ntail counts: PlateX>0.2={tailx}  PlateZ>0.2={tailz}  BOTH>0.2={both}")
