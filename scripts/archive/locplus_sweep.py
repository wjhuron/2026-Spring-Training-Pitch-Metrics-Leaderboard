"""locplus_sweep.py — hyperparameter sweep for the V3 decomposition Loc+.

Decisions locked in with Wally:
  - true count level, NO demean
  - V3 (decomposition) is the model to perfect; V2 (unified grid) is the
    benchmark to beat/match on reliability
  - goal of this sweep: lift V3 reliability toward V2 by reliability-weighting
    the components (per THT: contact suppression is ~luck, R^2=0.10) and by
    smoothing the sparse surfaces harder, WITHOUT losing V3's predictive edge

Design for speed: accumulate raw num/den grids ONCE; only re-run the kernel
smoother per config (kprior / bandwidth are the swept knobs).
"""
import pickle, math, collections, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline_sdplus import classify_zone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
LG_WOBA, WOBA_SCALE = 0.3169, 1.2393

PTG = {'FF':'FF','FA':'FF','SI':'SI','FC':'FC','CF':'FC','SL':'SL','ST':'SL',
       'SV':'SL','SW':'SL','CU':'CB','KC':'CB','CS':'CB','CH':'CH','FS':'CH'}
def ptg(pt): return PTG.get(pt,'OTHER') if pt else None
def sf(x):
    try: return float(x)
    except (TypeError,ValueError): return None

SWING_DESC={'Swinging Strike','Foul','In Play'}
TAKE_DESC={'Ball','Called Strike'}
EXCLUDE_DESC={'Hit By Pitch','Foul Bunt','Missed Bunt','Pitchout','Swinging Pitchout'}
BUNT_BB={'bunt','bunt_grounder','bunt_popup','bunt_line_drive'}
COUNTS=[(b,s) for b in range(4) for s in range(3)]
HANDS=('L','R')
GROUPS=[(g,bh,ph) for g in ['FF','SI','FC','SL','CB','CH','OTHER']
        for bh in HANDS for ph in HANDS]

# fixed 2" grid (bin-size tested separately at the end)
X_MIN,X_MAX,XW=-1.5,1.5,1.0/6.0
NX=int(round((X_MAX-X_MIN)/XW))
Z_MIN,Z_MAX,ZW=-0.6,1.6,0.1
NZ=int(round((Z_MAX-Z_MIN)/ZW))
def xbin(px): return min(max(int((px-X_MIN)/XW),0),NX-1)
def zbin(zn): return min(max(int((zn-Z_MIN)/ZW),0),NZ-1)

def get_count(p):
    c=p.get('Count')
    if not isinstance(c,str) or '-' not in c: return None
    try: b,s=c.split('-',1); b,s=int(b),int(s)
    except (TypeError,ValueError): return None
    return (b,s) if (0<=b<=3 and 0<=s<=2) else None
def znorm(p):
    pz,top,bot=sf(p.get('PlateZ')),sf(p.get('SzTop')),sf(p.get('SzBot'))
    if None in (pz,top,bot) or top<=bot: return None
    return (pz-bot)/(top-bot)
def xrv_hitter(p):
    if p.get('Description')=='In Play':
        xw=sf(p.get('xwOBA'))
        if xw is not None: return (xw-LG_WOBA)/WOBA_SCALE
    re=sf(p.get('RunExp'))
    return -re if re is not None else None
def scorable(p):
    if p.get('_source')!='MLB': return False
    if p.get('Description') in EXCLUDE_DESC: return False
    if p.get('BBType') in BUNT_BB: return False
    if p.get('Event')=='Intent Walk': return False
    if znorm(p) is None or sf(p.get('PlateX')) is None: return False
    if ptg(p.get('Pitch Type')) is None: return False
    if p.get('Bats') not in HANDS or p.get('Throws') not in HANDS: return False
    if get_count(p) is None: return False
    return True
def pearson(xs,ys):
    n=len(xs)
    if n<3: return None
    mx,my=sum(xs)/n,sum(ys)/n
    sx=sum((x-mx)**2 for x in xs); sy=sum((y-my)**2 for y in ys)
    if sx<=0 or sy<=0: return None
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/math.sqrt(sx*sy)

def make_kernel(bw,win=3):
    K=[]
    for di in range(-win,win+1):
        for dj in range(-win,win+1):
            K.append((di,dj,math.exp(-0.5*((di/bw)**2+(dj/bw)**2))))
    return K
def smooth(num,den,prior,kprior,K):
    out={}; pd=isinstance(prior,dict)
    for i in range(NX):
        for j in range(NZ):
            sn=sd=0.0
            for di,dj,w in K:
                ii,jj=i+di,j+dj
                if 0<=ii<NX and 0<=jj<NZ:
                    c=(ii,jj)
                    if c in den:
                        sn+=w*num.get(c,0.0); sd+=w*den[c]
            pr=prior.get((i,j),0.0) if pd else prior
            out[(i,j)]=(sn+kprior*pr)/(sd+kprior) if (sd+kprior)>0 else pr
    return out

# ═══ load + precompute ═══
print("loading ...",flush=True)
with open(PKL,'rb') as f: ALL=pickle.load(f)
P=[p for p in ALL if scorable(p)]
print(f"  {len(P)} scorable",flush=True)
for p in P:
    p['_g']=ptg(p.get('Pitch Type')); p['_c']=get_count(p)
    p['_i']=xbin(sf(p.get('PlateX'))); p['_j']=zbin(znorm(p))
    p['_bh']=p['Bats']; p['_ph']=p['Throws']; p['_xrv']=xrv_hitter(p)
    p['_z']=classify_zone(p)

# ═══ count value scalars ═══
def count_scalars():
    acc={k:collections.defaultdict(lambda:[0.0,0]) for k in ('whiff','foul','cs','ball')}
    for p in P:
        re=sf(p.get('RunExp'))
        if re is None: continue
        d=p.get('Description'); c=p['_c']; v=-re
        key={'Swinging Strike':'whiff','Foul':'foul','Called Strike':'cs','Ball':'ball'}.get(d)
        if key: acc[key][c][0]+=v; acc[key][c][1]+=1
    return {k:{c:(s/n if n else 0.0) for c,(s,n) in dd.items()} for k,dd in acc.items()}
RV=count_scalars()

# ═══ V0 baseline (fixed) ═══
ZCELL=collections.defaultdict(lambda:[0.0,0]); ZMARG=collections.defaultdict(lambda:[0.0,0])
for p in P:
    if p['_xrv'] is None or p['_z'] is None: continue
    g,c=p['_g'],p['_c']; v=p['_xrv']
    ZCELL[(p['_z'],c,g,p['_bh'],p['_ph'])][0]+=v; ZCELL[(p['_z'],c,g,p['_bh'],p['_ph'])][1]+=1
    ZMARG[(p['_z'],c,g)][0]+=v; ZMARG[(p['_z'],c,g)][1]+=1
ZMARG={k:s/n for k,(s,n) in ZMARG.items()}
for p in P:
    z=p['_z']
    if z is None: p['_v0']=None; continue
    s,n=ZCELL.get((z,p['_c'],p['_g'],p['_bh'],p['_ph']),[0.0,0])
    m=ZMARG.get((z,p['_c'],p['_g']),0.0); p['_v0']=(s+50*m)/(n+50)

# ═══ accumulate raw grids ONCE (for V2 + V3) ═══
print("accumulating raw grids ...",flush=True)
# V2 grid xRV
gx_num=collections.defaultdict(lambda:collections.defaultdict(float)); gx_den=collections.defaultdict(lambda:collections.defaultdict(float))
gxc_num=collections.defaultdict(lambda:collections.defaultdict(float)); gxc_den=collections.defaultdict(lambda:collections.defaultdict(float))
# V3 surfaces
sw_num=collections.defaultdict(lambda:collections.defaultdict(float)); sw_den=collections.defaultdict(lambda:collections.defaultdict(float))
wh_num=collections.defaultdict(lambda:collections.defaultdict(float)); fl_num=collections.defaultdict(lambda:collections.defaultdict(float))
bip_num=collections.defaultdict(lambda:collections.defaultdict(float)); bip_den=collections.defaultdict(lambda:collections.defaultdict(float))
swc_num=collections.defaultdict(lambda:collections.defaultdict(float)); swc_den=collections.defaultdict(lambda:collections.defaultdict(float))
cs_num=collections.defaultdict(float); cs_den=collections.defaultdict(float)
for p in P:
    key=(p['_g'],p['_bh'],p['_ph']); cell=(p['_i'],p['_j']); d=p.get('Description'); c=p['_c']
    if p['_xrv'] is not None:
        gx_num[key][cell]+=p['_xrv']; gx_den[key][cell]+=1
        gxc_num[(key,c)][cell]+=p['_xrv']; gxc_den[(key,c)][cell]+=1
    sw_den[key][cell]+=1; swc_den[(key,c)][cell]+=1
    if d in SWING_DESC:
        sw_num[key][cell]+=1; swc_num[(key,c)][cell]+=1
        if d=='Swinging Strike': wh_num[key][cell]+=1
        elif d=='Foul': fl_num[key][cell]+=1
        elif d=='In Play':
            xw=sf(p.get('xwOBA'))
            v=(xw-LG_WOBA)/WOBA_SCALE if xw is not None else None
            if v is not None: bip_num[key][cell]+=v; bip_den[key][cell]+=1
    if d in TAKE_DESC:
        cs_den[cell]+=1
        if d=='Called Strike': cs_num[cell]+=1

# group-level global rates (priors)
GR={}
for key in GROUPS:
    swd=sum(sw_den[key].values()); swn=sum(sw_num[key].values()); bipd=sum(bip_den[key].values())
    GR[key]={'sw':swn/swd if swd else 0.0,'wh':sum(wh_num[key].values())/max(swn,1),
             'fl':sum(fl_num[key].values())/max(swn,1),'bip':sum(bip_num[key].values())/max(bipd,1),
             'gx':(sum(gx_num[key].values())/sum(gx_den[key].values())) if sum(gx_den[key].values()) else 0.0}
glob_cs=sum(cs_num.values())/max(sum(cs_den.values()),1)

# ═══ static eval infra ═══
MIN_HALF, MIN_PRED = 150, 200
dates_by_p=collections.defaultdict(set)
for p in P: dates_by_p[(p['Pitcher'],p['Throws'])].add(p.get('Game Date'))
half_of={}
for k,ds in dates_by_p.items():
    for idx,d in enumerate(sorted(ds)): half_of[(k,d)]=idx%2
H=[[],[]]
firsts=[];
for p in P:
    H[half_of[((p['Pitcher'],p['Throws']),p.get('Game Date'))]].append(p)
    if (p.get('Game Date') or '')<'2026-05-01': firsts.append(p)
stf=collections.defaultdict(lambda:{'sw':0,'wh':0,'vs':0.0,'vn':0})
for p in P:
    k=(p['Pitcher'],p['Throws']); d=p.get('Description')
    if d in SWING_DESC:
        stf[k]['sw']+=1
        if d=='Swinging Strike': stf[k]['wh']+=1
    if p['_g']=='FF':
        v=sf(p.get('Velocity'))
        if v is not None: stf[k]['vs']+=v; stf[k]['vn']+=1
whiffrate={k:s['wh']/s['sw'] for k,s in stf.items() if s['sw']>=50}
ffvelo={k:s['vs']/s['vn'] for k,s in stf.items() if s['vn']>=30}
sec=collections.defaultdict(list)
for p in P:
    if (p.get('Game Date') or '')>='2026-05-01' and p['_xrv'] is not None:
        sec[(p['Pitcher'],p['Throws'])].append(p['_xrv'])
sec_xrv={k:sum(v)/len(v) for k,v in sec.items() if len(v)>=MIN_PRED}

def agg(field,pitches):
    out=collections.defaultdict(list)
    for p in pitches:
        v=p.get(field)
        if v is not None: out[(p['Pitcher'],p['Throws'])].append(v)
    return {k:(sum(v)/len(v),len(v)) for k,v in out.items()}
def evaluate(field):
    sa=agg(field,H[0]); sb=agg(field,H[1])
    common=[k for k in sa if k in sb and sa[k][1]>=MIN_HALF and sb[k][1]>=MIN_HALF]
    rel=pearson([sa[k][0] for k in common],[sb[k][0] for k in common])
    full=agg(field,P); qual={k:val for k,(val,n) in full.items() if n>=MIN_PRED}
    kw=[k for k in qual if k in whiffrate]; rw=pearson([qual[k] for k in kw],[whiffrate[k] for k in kw])
    kv=[k for k in qual if k in ffvelo]; rv=pearson([qual[k] for k in kv],[ffvelo[k] for k in kv])
    fa=agg(field,firsts); fq={k:val for k,(val,n) in fa.items() if n>=MIN_PRED}
    kp=[k for k in fq if k in sec_xrv]; rp=pearson([fq[k] for k in kp],[sec_xrv[k] for k in kp])
    return rel,abs(rw),abs(rv),rp,len(common),len(kp)

# fixed-field evals
for p in P: p['_v0v']=p['_v0']
print("\nbaseline V0:", tuple(round(x,3) if x is not None else None for x in evaluate('_v0v')[:4]))

# ═══ build functions (per cfg) ═══
def build_v2(bw,win=3):
    K=make_kernel(bw,win); GRIDV={}
    for key in GROUPS:
        coll=smooth(gx_num[key],gx_den[key],GR[key]['gx'],8,K)
        GRIDV[key]={c:smooth(gxc_num[(key,c)],gxc_den[(key,c)],coll,25,K) for c in COUNTS}
    for p in P: p['_v2']=GRIDV[(p['_g'],p['_bh'],p['_ph'])][p['_c']][(p['_i'],p['_j'])]
def build_v3(bw,xwcon_k,swc_k,wh_k=8,fl_k=8,cs_k=10,win=3):
    K=make_kernel(bw,win)
    WH,FL,XW,SW,PCS={},{},{},{},smooth(cs_num,cs_den,glob_cs,cs_k,K)
    for key in GROUPS:
        WH[key]=smooth(wh_num[key],sw_num[key],GR[key]['wh'],wh_k,K)
        FL[key]=smooth(fl_num[key],sw_num[key],GR[key]['fl'],fl_k,K)
        XW[key]=smooth(bip_num[key],bip_den[key],GR[key]['bip'],xwcon_k,K)
        coll=smooth(sw_num[key],sw_den[key],GR[key]['sw'],6,K)
        SW[key]={c:smooth(swc_num[(key,c)],swc_den[(key,c)],coll,swc_k,K) for c in COUNTS}
    for p in P:
        key=(p['_g'],p['_bh'],p['_ph']); cell=(p['_i'],p['_j']); c=p['_c']
        psw=SW[key][c][cell]; pwh=WH[key][cell]; pfl=FL[key][cell]
        pbip=max(0.0,1-pwh-pfl); vbip=XW[key][cell]; pcs=PCS[cell]
        sv=pwh*RV['whiff'].get(c,0)+pfl*RV['foul'].get(c,0)+pbip*vbip
        tv=pcs*RV['cs'].get(c,0)+(1-pcs)*RV['ball'].get(c,0)
        p['_v3']=psw*sv+(1-psw)*tv

# ═══ SWEEP ═══
print(f"\n{'config':40s} {'reliab':>7s} {'rWhf':>6s} {'rVel':>6s} {'pred':>6s}")
print("-"*72)
build_v2(1.8,win=5); rel,rw,rv,rp,_,_=evaluate('_v2')
print(f"{'V2 grid bw=1.8 win5':40s} {rel:7.3f} {rw:6.3f} {rv:6.3f} {rp:6.3f}")
print("-"*72)
# V3 bandwidth push (wider window so high bw isn't truncated)
for bw in (1.8,2.4,3.0,3.6):
    build_v3(bw,200,20,win=5); rel,rw,rv,rp,_,_=evaluate('_v3')
    print(f"{'V3 bw='+str(bw)+' xwK=200 swK=20 win5':40s} {rel:7.3f} {rw:6.3f} {rv:6.3f} {rp:6.3f}")
print("-"*72)
# best V3 so far + V2, then blends
build_v2(1.8,win=5)
build_v3(2.4,200,20,win=5)
# pitcher-level correlation V2 vs V3
fv2=agg('_v2',P); fv3=agg('_v3',P)
ck=[k for k in fv2 if k in fv3 and fv2[k][1]>=MIN_PRED]
print(f"corr(V2,V3) across pitchers = {pearson([fv2[k][0] for k in ck],[fv3[k][0] for k in ck]):.3f}")
for a in (0.3,0.5,0.7):
    for p in P: p['_blend']=a*p['_v2']+(1-a)*p['_v3']
    rel,rw,rv,rp,_,_=evaluate('_blend')
    print(f"{'blend '+str(a)+'*V2+'+str(round(1-a,1))+'*V3':40s} {rel:7.3f} {rw:6.3f} {rv:6.3f} {rp:6.3f}")
print("-"*72)
print(f"n_reliab≈{evaluate('_v0v')[4]}  n_pred≈{evaluate('_v0v')[5]}")
print("reliab↑  rWhf↓ rVel↓ (stuff leak)  pred↑")
