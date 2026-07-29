"""locplus_groups.py — settle pitch-type grouping (incl. cutters) and show
real-pitcher Loc+ examples under a few scale conventions.

Wally's grouping: FF | SI | CH(+FS) | BREAKING(all breakers except cutters) |
cutters = TBD | OTHER. We test whether merging all breakers hurts overall
accuracy vs keeping slider/curve split, where cutters belong (surface
similarity to FF vs breaking), and what the score distribution looks like.

Locked scoring config: 2in x 0.10z grid, 4.5in/0.22 smoothing, xwK=200,
swK=20, V3 decomposition, true count level. n_prior=107 (measured).
"""
import pickle, math, collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline_sdplus import classify_zone
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL=os.path.join(ROOT,'data','all_pitches_rs_cache.pkl')
LG,SCALE=0.3169,1.2393
def sf(x):
    try: return float(x)
    except (TypeError,ValueError): return None
SWING_DESC={'Swinging Strike','Foul','In Play'}; TAKE_DESC={'Ball','Called Strike'}
EXCLUDE={'Hit By Pitch','Foul Bunt','Missed Bunt','Pitchout','Swinging Pitchout'}
BUNT={'bunt','bunt_grounder','bunt_popup','bunt_line_drive'}
COUNTS=[(b,s) for b in range(4) for s in range(3)]; HANDS=('L','R')
X_MIN,X_MAX=-1.5,1.5; Z_MIN,Z_MAX=-0.6,1.6; N_PRIOR=107

# ── group schemes ──
FINE={'FF':'FF','FA':'FF','SI':'SI','FC':'FC','CF':'FC','SL':'SL','ST':'SL','SV':'SL',
      'SW':'SL','CU':'CB','KC':'CB','CS':'CB','CH':'CH','FS':'CH'}
def make_group_fn(cutter):
    """cutter in {'FC','FF','BRK'}"""
    m={'FF':'FF','FA':'FF','SI':'SI',
       'SL':'BRK','ST':'BRK','SV':'BRK','SW':'BRK','CU':'BRK','KC':'BRK','CS':'BRK',
       'CH':'CH','FS':'CH','FC':cutter,'CF':cutter}
    return lambda pt: (m.get(pt,'OTHER') if pt else None)
def fine_fn(pt): return FINE.get(pt,'OTHER') if pt else None

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
        if xw is not None: return (xw-LG)/SCALE
    re=sf(p.get('RunExp')); return -re if re is not None else None
def scorable(p):
    if p.get('_source')!='MLB' or p.get('Description') in EXCLUDE: return False
    if p.get('BBType') in BUNT or p.get('Event')=='Intent Walk': return False
    if znorm(p) is None or sf(p.get('PlateX')) is None: return False
    if p.get('Pitch Type') in (None,''): return False
    if p.get('Bats') not in HANDS or p.get('Throws') not in HANDS: return False
    return get_count(p) is not None
def pearson(xs,ys):
    n=len(xs)
    if n<3: return None
    mx,my=sum(xs)/n,sum(ys)/n
    sx=sum((x-mx)**2 for x in xs); sy=sum((y-my)**2 for y in ys)
    if sx<=0 or sy<=0: return None
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/math.sqrt(sx*sy)

print("loading ...",flush=True)
with open(PKL,'rb') as f: ALL=pickle.load(f)
P=[p for p in ALL if scorable(p)]
print(f"  {len(P)} scorable",flush=True)
for p in P:
    p['_pt']=p.get('Pitch Type'); p['_c']=get_count(p)
    p['_px']=sf(p.get('PlateX')); p['_zn']=znorm(p)
    p['_bh']=p['Bats']; p['_ph']=p['Throws']; p['_d']=p.get('Description'); p['_xw']=sf(p.get('xwOBA'))

acc={k:collections.defaultdict(lambda:[0.0,0]) for k in ('whiff','foul','cs','ball')}
for p in P:
    re=sf(p.get('RunExp'))
    if re is None: continue
    key={'Swinging Strike':'whiff','Foul':'foul','Called Strike':'cs','Ball':'ball'}.get(p['_d'])
    if key: acc[key][p['_c']][0]+=-re; acc[key][p['_c']][1]+=1
RV={k:{c:(s/n if n else 0.0) for c,(s,n) in dd.items()} for k,dd in acc.items()}

# eval infra
MIN_HALF,MIN_PRED=150,200
dbp=collections.defaultdict(set)
for p in P: dbp[(p['Pitcher'],p['Throws'])].add(p.get('Game Date'))
half_of={}
for k,ds in dbp.items():
    for idx,d in enumerate(sorted(ds)): half_of[(k,d)]=idx%2
H=[[],[]]; firsts=[]
for p in P:
    H[half_of[((p['Pitcher'],p['Throws']),p.get('Game Date'))]].append(p)
    if (p.get('Game Date') or '')<'2026-05-01': firsts.append(p)
stf=collections.defaultdict(lambda:{'sw':0,'wh':0,'vs':0.0,'vn':0})
for p in P:
    k=(p['Pitcher'],p['Throws'])
    if p['_d'] in SWING_DESC:
        stf[k]['sw']+=1
        if p['_d']=='Swinging Strike': stf[k]['wh']+=1
    if fine_fn(p['_pt'])=='FF':
        v=sf(p.get('Velocity'))
        if v is not None: stf[k]['vs']+=v; stf[k]['vn']+=1
whiffrate={k:s['wh']/s['sw'] for k,s in stf.items() if s['sw']>=50}
ffvelo={k:s['vs']/s['vn'] for k,s in stf.items() if s['vn']>=30}
sec=collections.defaultdict(list)
for p in P:
    if (p.get('Game Date') or '')>='2026-05-01' and xrv_hitter(p) is not None:
        sec[(p['Pitcher'],p['Throws'])].append(xrv_hitter(p))
sec_xrv={k:sum(v)/len(v) for k,v in sec.items() if len(v)>=MIN_PRED}
def aggf(pitches):
    out=collections.defaultdict(list)
    for p in pitches:
        v=p.get('_v3')
        if v is not None: out[(p['Pitcher'],p['Throws'])].append(v)
    return out
def evaluate():
    a0={k:(sum(v)/len(v),len(v)) for k,v in aggf(H[0]).items()}
    a1={k:(sum(v)/len(v),len(v)) for k,v in aggf(H[1]).items()}
    com=[k for k in a0 if k in a1 and a0[k][1]>=MIN_HALF and a1[k][1]>=MIN_HALF]
    rel=pearson([a0[k][0] for k in com],[a1[k][0] for k in com])
    full={k:(sum(v)/len(v),len(v)) for k,v in aggf(P).items()}
    qual={k:val for k,(val,n) in full.items() if n>=MIN_PRED}
    kw=[k for k in qual if k in whiffrate]; rw=pearson([qual[k] for k in kw],[whiffrate[k] for k in kw])
    kv=[k for k in qual if k in ffvelo]; rv=pearson([qual[k] for k in kv],[ffvelo[k] for k in kv])
    fa={k:(sum(v)/len(v),len(v)) for k,v in aggf(firsts).items()}
    fq={k:val for k,(val,n) in fa.items() if n>=MIN_PRED}
    kp=[k for k in fq if k in sec_xrv]; rp=pearson([fq[k] for k in kp],[sec_xrv[k] for k in kp])
    return rel,abs(rw),abs(rv),rp

# ── locked smoother ──
BIN_X_IN,BIN_Z=2.0,0.10; PHX,PHZ=4.5,0.22; XWK,SWK=200,20
bx=BIN_X_IN/12.0; NX=int(round((X_MAX-X_MIN)/bx)); NZ=int(round((Z_MAX-Z_MIN)/BIN_Z))
def xb(px): return min(max(int((px-X_MIN)/bx),0),NX-1)
def zb(zn): return min(max(int((zn-Z_MIN)/BIN_Z),0),NZ-1)
def k1d(bw):
    win=max(1,int(math.ceil(3*bw))); return [(d,math.exp(-0.5*(d/bw)**2)) for d in range(-win,win+1)]
kx=k1d(PHX/BIN_X_IN); kz=k1d(PHZ/BIN_Z)
def zeros(): return [[0.0]*NZ for _ in range(NX)]
def smooth2d(num,den,prior,kp):
    tn=zeros(); td=zeros()
    for i in range(NX):
        ni,di_,tni,tdi=num[i],den[i],tn[i],td[i]
        for j in range(NZ):
            sn=sd=0.0
            for dj,w in kz:
                jj=j+dj
                if 0<=jj<NZ: sn+=w*ni[jj]; sd+=w*di_[jj]
            tni[j]=sn; tdi[j]=sd
    out=zeros(); pdict=not isinstance(prior,(int,float))
    for i in range(NX):
        oi=out[i]
        for j in range(NZ):
            sn=sd=0.0
            for di2,w in kx:
                ii=i+di2
                if 0<=ii<NX: sn+=w*tn[ii][j]; sd+=w*td[ii][j]
            pr=prior[i][j] if pdict else prior; s=sd+kp
            oi[j]=(sn+kp*pr)/s if s>0 else pr
    return out
def gsum(arr): return sum(sum(r) for r in arr)

def build_score(group_fn, return_surfaces=False):
    GRP={}  # (group,bh,ph)->accs ; build lazily
    def acc0(): return {'swn':zeros(),'swd':zeros(),'whn':zeros(),'fln':zeros(),'bipn':zeros(),'bipd':zeros()}
    A=collections.defaultdict(acc0)
    AC=collections.defaultdict(lambda:{'swn':zeros(),'swd':zeros()})
    csn=zeros(); csd=zeros()
    for p in P:
        g=group_fn(p['_pt'])
        if g is None: p['_v3']=None; continue
        key=(g,p['_bh'],p['_ph']); c=p['_c']; i=xb(p['_px']); j=zb(p['_zn']); d=p['_d']
        a=A[key]; ac=AC[(key,c)]; a['swd'][i][j]+=1; ac['swd'][i][j]+=1
        if d in SWING_DESC:
            a['swn'][i][j]+=1; ac['swn'][i][j]+=1
            if d=='Swinging Strike': a['whn'][i][j]+=1
            elif d=='Foul': a['fln'][i][j]+=1
            elif d=='In Play' and p['_xw'] is not None: a['bipn'][i][j]+=(p['_xw']-LG)/SCALE; a['bipd'][i][j]+=1
        if d in TAKE_DESC:
            csd[i][j]+=1
            if d=='Called Strike': csn[i][j]+=1
    PCS=smooth2d(csn,csd,gsum(csn)/max(gsum(csd),1),10)
    WH={};FL={};XW={};SW={}
    for key,a in A.items():
        swd=gsum(a['swd']); swn=gsum(a['swn']); bipd=gsum(a['bipd'])
        WH[key]=smooth2d(a['whn'],a['swn'],gsum(a['whn'])/max(swn,1),8)
        FL[key]=smooth2d(a['fln'],a['swn'],gsum(a['fln'])/max(swn,1),8)
        XW[key]=smooth2d(a['bipn'],a['bipd'],gsum(a['bipn'])/max(bipd,1),XWK)
        coll=smooth2d(a['swn'],a['swd'],swn/swd if swd else 0.0,6)
        SW[key]={c:smooth2d(AC[(key,c)]['swn'],AC[(key,c)]['swd'],coll,SWK) for c in COUNTS}
    for p in P:
        g=group_fn(p['_pt'])
        if g is None: continue
        key=(g,p['_bh'],p['_ph']); c=p['_c']; i=xb(p['_px']); j=zb(p['_zn'])
        if key not in WH: p['_v3']=None; continue
        psw=SW[key][c][i][j]; pwh=WH[key][i][j]; pfl=FL[key][i][j]
        pbip=max(0.0,1-pwh-pfl); vbip=XW[key][i][j]; pcs=PCS[i][j]
        sv=pwh*RV['whiff'].get(c,0)+pfl*RV['foul'].get(c,0)+pbip*vbip
        tv=pcs*RV['cs'].get(c,0)+(1-pcs)*RV['ball'].get(c,0)
        p['_v3']=psw*sv+(1-psw)*tv
    if return_surfaces: return WH,XW,A
    return None

# ═══ 1. grouping scheme comparison ═══
print(f"\n{'scheme':34s} {'reliab':>7s} {'rWhf':>6s} {'rVel':>6s} {'pred':>6s}")
print("-"*64)
build_score(fine_fn); print(f"{'fine (FF SI FC SL CB CH)':34s} "+" ".join(f"{x:6.3f}" for x in evaluate()))
for cut in ('FC','FF','BRK'):
    build_score(make_group_fn(cut))
    print(f"{'wally, cutter->'+cut:34s} "+" ".join(f"{x:6.3f}" for x in evaluate()))

# ═══ 2. cutter surface similarity (RHP) ═══
WH,XW,A=build_score(fine_fn, return_surfaces=True)
def gridcorr(s1,s2,wkey):
    a1=A.get(('FC','R','R')); # weight by min sample of FC vs other already implicit
    xs=[];ys=[]
    for i in range(NX):
        for j in range(NZ):
            xs.append(s1[i][j]); ys.append(s2[i][j])
    return pearson(xs,ys)
def avg_surface(groups):  # average a surface dict over hand combos for RHP set
    pass
print("\ncutter (FC) surface similarity, RHH vs RHP:")
for nm,sd in (('whiff',WH),('xwOBAcon',XW)):
    fc=sd.get(('FC','R','R')); ff=sd.get(('FF','R','R')); sl=sd.get(('SL','R','R')); cb=sd.get(('CB','R','R'))
    if fc and ff and sl:
        print(f"  {nm:9s}: corr(FC,FF)={gridcorr(fc,ff,nm):+.3f}   corr(FC,SL)={gridcorr(fc,sl,nm):+.3f}"
              + (f"   corr(FC,CB)={gridcorr(fc,cb,nm):+.3f}" if cb else ""))

# ═══ 3. scale examples (wally grouping, cutter own) ═══
build_score(make_group_fn('FC'))
full=collections.defaultdict(lambda:[0.0,0,None])
for p in P:
    if p.get('_v3') is None: continue
    k=(p['Pitcher'],p['Throws']); full[k][0]+=p['_v3']; full[k][1]+=1
raw={k:(s/n) for k,(s,n,_) in full.items() if n>0}
npit={k:full[k][1] for k in full}
pool={k for k in raw if npit[k]>=400}
lg=sum(raw[k] for k in pool)/len(pool)
adj={k:(npit[k]*raw[k]+N_PRIOR*lg)/(npit[k]+N_PRIOR) for k in raw}
pool_adj=[adj[k] for k in pool]
mu=sum(pool_adj)/len(pool_adj)
sigma=math.sqrt(sum((v-mu)**2 for v in pool_adj)/len(pool_adj))
def pct(vals,q):
    s=sorted(vals); return s[min(len(s)-1,int(q*len(s)))]
zs=sorted((adj[k]-mu)/sigma for k in pool)
print(f"\npool={len(pool)} pitchers (>=400 pitches).  raw lg={lg:+.4f}  sigma(adj)={sigma:.4f}")
print("z percentiles (pool):", {q:round((adj_to_z) ,2) for q,adj_to_z in
      [(p2,pct(zs,p2)) for p2 in (0.01,0.05,0.25,0.5,0.75,0.95,0.99)]})
def show(K,label):
    locp={k:100-K*((adj[k]-mu)/sigma) for k in pool}
    order=sorted(locp,key=lambda k:-locp[k])
    print(f"\n--- {label} (K={K}/SD) ---  range {locp[order[-1]]:.0f} .. {locp[order[0]]:.0f}")
    print("  TOP:", ", ".join(f"{k[0]} {locp[k]:.0f}" for k in order[:8]))
    print("  BOT:", ", ".join(f"{k[0]} {locp[k]:.0f}" for k in order[-6:]))
for K in (8,10,12):
    show(K,f"+stat z*{K}")
# run-denominated alternative: location runs / 100 pitches above avg (pitcher persp)
locrv={k:-(adj[k]-lg)*100 for k in pool}
o=sorted(locrv,key=lambda k:-locrv[k])
print(f"\n--- alt: Location Runs / 100 pitches (pitcher persp, +good) ---  range {locrv[o[-1]]:+.2f} .. {locrv[o[0]]:+.2f}")
print("  TOP:", ", ".join(f"{k[0]} {locrv[k]:+.2f}" for k in o[:8]))
print("  BOT:", ", ".join(f"{k[0]} {locrv[k]:+.2f}" for k in o[-6:]))
