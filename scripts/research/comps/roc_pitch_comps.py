"""
Pitch-mix comps for ROC pitchers vs MLB pitchers.
Match on shape (velocity, IVB, HB, relz, relx, extension), tag-agnostic.
Full-arsenal usage-weighted distance, soft handedness preference,
lefties mirrored into a common RHP frame so mirror-image shapes match.
"""
import json, math
from collections import defaultdict

D = json.load(open('data/pitch_leaderboard_rs.json'))

MLB30 = {'ARI','ATH','ATL','BAL','BOS','CHC','CIN','CLE','COL','CWS','DET','HOU',
         'KCR','LAA','LAD','MIA','MIL','MIN','NYM','NYY','PHI','PIT','SDP','SEA',
         'SFG','STL','TBR','TEX','TOR','WSH'}
AGG = {'2TM','3TM'}

FEATS = ['velocity','indVertBrk','horzBrk','relPosZ','relPosX','extension']

TARGETS = ['Champlain, Chandler','Cranz, Robert','Kent, Jackson','Kent, Zak',
           'Lara, Andry','Penrod, Zach','Perales, Luis','Sinclair, Jack',
           'Tolman, Erik','Yean, Eddy','Young, Luke']

def valid(r):
    return all(r.get(f) is not None for f in FEATS)

def normframe(r):
    """Return feature vector in RHP frame (mirror HB & relX for LHP)."""
    hb = r['horzBrk']; rx = r['relPosX']
    if r['throws'] == 'L':
        hb = -hb; rx = -rx
    return {'velocity':r['velocity'],'indVertBrk':r['indVertBrk'],'horzBrk':hb,
            'relPosZ':r['relPosZ'],'relPosX':rx,'extension':r['extension']}

# ---- Build candidate MLB arsenals (one per mlbId) ----
by_id = defaultdict(list)
for r in D:
    by_id[r['mlbId']].append(r)

target_ids = set()
for t in TARGETS:
    for r in D:
        if r['pitcher']==t and r['team']=='ROC':
            target_ids.add(r['mlbId'])

candidates = {}   # mlbId -> {'name','throws','pitches':[{pt,usage,vec,count}]}
for mid, rows in by_id.items():
    if mid in target_ids:
        continue
    teams = set(r['team'] for r in rows)
    if teams & AGG:
        use = [r for r in rows if r['team'] in AGG]
    elif teams & MLB30:
        use = [r for r in rows if r['team'] in MLB30]
    else:
        continue  # not an MLB pitcher
    use = [r for r in use if valid(r) and r['count'] >= 15]
    total = sum(r['count'] for r in use)
    if total < 200 or not use:
        continue
    pitches = [{'pt':r['pitchType'],'usage':r['usagePct'],'count':r['count'],
                'vec':normframe(r),'throws':r['throws']} for r in use]
    # renormalize usage over kept pitches
    us = sum(p['usage'] for p in pitches)
    for p in pitches: p['w'] = p['usage']/us if us else 0
    candidates[mid] = {'name':use[0]['pitcher'],'throws':use[0]['throws'],'pitches':pitches}

# ---- Standardization: mean/SD per feature over candidate pitch rows ----
pool = []
for c in candidates.values():
    for p in c['pitches']:
        pool.append(p['vec'])
def msd(f):
    xs=[v[f] for v in pool]; m=sum(xs)/len(xs)
    sd=(sum((x-m)**2 for x in xs)/len(xs))**0.5
    return m,sd
STATS = {f:msd(f) for f in FEATS}

def z(vec):
    return {f:(vec[f]-STATS[f][0])/STATS[f][1] for f in FEATS}

# ---- Build target (ROC) arsenals ----
targets = {}
for t in TARGETS:
    rows=[r for r in D if r['pitcher']==t and r['team']=='ROC' and valid(r)]
    total=sum(r['count'] for r in rows)
    pitches=[{'pt':r['pitchType'],'usage':r['usagePct'],'count':r['count'],
              'vec':normframe(r)} for r in rows]
    us=sum(p['usage'] for p in pitches)
    for p in pitches: p['w']=p['usage']/us if us else 0
    targets[t]={'throws':rows[0]['throws'],'pitches':pitches}

# ---- Distance ----
def pdist(za, zb, W):
    return math.sqrt(sum(W[f]*(za[f]-zb[f])**2 for f in FEATS))

def arsenal_dist(tgt, cand, W):
    # R -> M : each ROC pitch to closest cand pitch, usage weighted
    tz=[(p['w'], z(p['vec'])) for p in tgt['pitches']]
    cz=[(p['w'], z(p['vec'])) for p in cand['pitches']]
    fwd=sum(w*min(pdist(zv,czv,W) for _,czv in cz) for w,zv in tz)
    rev=sum(w*min(pdist(czv,zv,W) for _,zv in tz) for w,czv in cz)
    return 0.65*fwd + 0.35*rev

def run(W, hand_penalty=1.08, k=0.55, label=''):
    print(f"\n{'='*70}\nWEIGHTS {label}: "+", ".join(f'{f}={W[f]}' for f in FEATS))
    print('='*70)
    out={}
    for t in TARGETS:
        tgt=targets[t]
        scored=[]
        for mid,c in candidates.items():
            d=arsenal_dist(tgt,c,W)
            if c['throws']!=tgt['throws']:
                d*=hand_penalty
            sim=100*math.exp(-k*d)
            scored.append((sim,d,c['name'],c['throws']))
        scored.sort(reverse=True)
        out[t]=scored[:3]
        armix=", ".join(f"{p['pt']}{int(round(p['usage']*100))}" for p in sorted(tgt['pitches'],key=lambda x:-x['usage']))
        print(f"\n{t} ({tgt['throws']}) [{armix}]")
        for sim,d,nm,th in scored[:3]:
            print(f"   {sim:5.1f}  {nm} ({th})")
    return out

# Shape-forward primary
W_shape={'velocity':1.0,'indVertBrk':1.0,'horzBrk':1.0,'relPosZ':0.6,'relPosX':0.6,'extension':0.6}
run(W_shape, label='SHAPE-FORWARD')

def detail(t, cand, W):
    tgt=targets[t]
    print(f"\n--- {t} ({tgt['throws']})  vs  {cand['name']} ({cand['throws']}) ---")
    print(f"{'ROC pitch':>10} {'use':>4} | {'best MLB':>9} | {'velo':>10} {'ivb':>10} {'hb(RHPfrm)':>10} {'relz':>9} {'relx':>9} {'ext':>8}")
    for p in sorted(tgt['pitches'],key=lambda x:-x['usage']):
        zp=z(p['vec'])
        best=min(cand['pitches'],key=lambda c:pdist(zp,z(c['vec']),W))
        pv=p['vec']; bv=best['vec']
        print(f"{p['pt']:>10} {int(round(p['usage']*100)):>3}% | {best['pt']:>9} | "
              f"{pv['velocity']:>4.1f}/{bv['velocity']:<4.1f} {pv['indVertBrk']:>4.1f}/{bv['indVertBrk']:<4.1f} "
              f"{pv['horzBrk']:>4.1f}/{bv['horzBrk']:<4.1f} {pv['relPosZ']:>3.1f}/{bv['relPosZ']:<3.1f} "
              f"{pv['relPosX']:>3.1f}/{bv['relPosX']:<3.1f} {pv['extension']:>3.1f}/{bv['extension']:<3.1f}")

# Detail for the #1 comp of each (shape-forward)
print("\n\n############ PITCH-BY-PITCH DETAIL (#1 comp, RHP frame) ############")
cand_by_name={c['name']:c for c in candidates.values()}
for t in TARGETS:
    scored=[]
    for mid,c in candidates.items():
        d=arsenal_dist(targets[t],c,W_shape)
        if c['throws']!=targets[t]['throws']: d*=1.08
        scored.append((d,c))
    scored.sort(key=lambda x:x[0])
    detail(t, scored[0][1], W_shape)
