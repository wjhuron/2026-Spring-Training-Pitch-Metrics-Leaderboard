"""locplus_final_validate.py — exhaustive final tuning pass.

Accumulates raw grids ONCE (grid bins fixed at the verified-immaterial 2"/0.10z),
then re-smooths/re-scores per config. Reports, for each hyperparameter, a fine
one-axis sweep around the current locked value, plus a multi-seed n_prior
estimate (overall + per group). Objective: stay stuff-independent (low whiff
corr), then maximize reliability + predictive validity. Round where flat.

Current locked: phys 4.5/0.22, xwK=200, swK=20, kWhiff=kFoul=8, kCS=10,
kSwingColl=6, n_prior=107 / 150.
"""
import pickle, math, collections, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL=os.path.join(ROOT,'data','all_pitches_rs_cache.pkl')
LG,SCALE=0.3169,1.2393
GROUP={'FF':'FF','FA':'FF','SI':'SI','FC':'FC','CF':'FC',
       'SL':'SL','ST':'SL','SW':'SL','SV':'SL','CU':'CU','KC':'CU','CS':'CU',
       'CH':'CH','FS':'CH'}
def grp(pt): return GROUP.get(pt,'OTHER') if pt else None
def sf(x):
    try: return float(x)
    except (TypeError,ValueError): return None
SWING_DESC={'Swinging Strike','Foul','In Play'}; TAKE_DESC={'Ball','Called Strike'}
EXCLUDE={'Hit By Pitch','Foul Bunt','Missed Bunt','Pitchout','Swinging Pitchout'}
BUNT={'bunt','bunt_grounder','bunt_popup','bunt_line_drive'}
COUNTS=[(b,s) for b in range(4) for s in range(3)]; HANDS=('L','R')
X_MIN,X_MAX=-1.5,1.5; Z_MIN,Z_MAX=-0.6,1.6
BIN_X_IN,BIN_Z=2.0,0.10
bx=BIN_X_IN/12.0; NX=int(round((X_MAX-X_MIN)/bx)); NZ=int(round((Z_MAX-Z_MIN)/BIN_Z))
def xb(px): return min(max(int((px-X_MIN)/bx),0),NX-1)
def zb(zn): return min(max(int((zn-Z_MIN)/BIN_Z),0),NZ-1)
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
def scorable(p):
    if p.get('_source')!='MLB' or p.get('Description') in EXCLUDE: return False
    if p.get('BBType') in BUNT or p.get('Event')=='Intent Walk': return False
    if znorm(p) is None or sf(p.get('PlateX')) is None: return False
    if grp(p.get('Pitch Type')) is None: return False
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
    p['_g']=grp(p.get('Pitch Type')); p['_c']=get_count(p)
    p['_i']=xb(sf(p.get('PlateX'))); p['_j']=zb(znorm(p))
    p['_bh']=p['Bats']; p['_ph']=p['Throws']; p['_d']=p.get('Description'); p['_xw']=sf(p.get('xwOBA'))

acc={k:collections.defaultdict(lambda:[0.0,0]) for k in ('whiff','foul','cs','ball')}
for p in P:
    re=sf(p.get('RunExp'))
    if re is None: continue
    key={'Swinging Strike':'whiff','Foul':'foul','Called Strike':'cs','Ball':'ball'}.get(p['_d'])
    if key: acc[key][p['_c']][0]+=-re; acc[key][p['_c']][1]+=1
RV={k:{c:(s/n if n else 0.0) for c,(s,n) in dd.items()} for k,dd in acc.items()}

def zeros(): return [[0.0]*NZ for _ in range(NX)]
def acc0(): return {k:zeros() for k in ('swn','swd','whn','fln','bipn','bipd')}
A=collections.defaultdict(acc0); AC=collections.defaultdict(lambda:{'swn':zeros(),'swd':zeros()})
csn=zeros(); csd=zeros()
for p in P:
    key=(p['_g'],p['_bh'],p['_ph']); c=p['_c']; i=p['_i']; j=p['_j']; d=p['_d']
    a=A[key]; ac=AC[(key,c)]; a['swd'][i][j]+=1; ac['swd'][i][j]+=1
    if d in SWING_DESC:
        a['swn'][i][j]+=1; ac['swn'][i][j]+=1
        if d=='Swinging Strike': a['whn'][i][j]+=1
        elif d=='Foul': a['fln'][i][j]+=1
        elif d=='In Play' and p['_xw'] is not None: a['bipn'][i][j]+=(p['_xw']-LG)/SCALE; a['bipd'][i][j]+=1
    if d in TAKE_DESC:
        csd[i][j]+=1
        if d=='Called Strike': csn[i][j]+=1
def gsum(a): return sum(sum(r) for r in a)
KEYS=list(A.keys())

def k1d(bw):
    win=max(1,int(math.ceil(3*bw))); return [(d,math.exp(-0.5*(d/bw)**2)) for d in range(-win,win+1)]
def smooth2d(num,den,prior,kp,kx,kz):
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

def build_score(phx=4.5,phz=0.22,xwK=200,swK=20,kwh=8,kfl=8,kcs=10,kcoll=6):
    kx=k1d(phx/BIN_X_IN); kz=k1d(phz/BIN_Z)
    PCS=smooth2d(csn,csd,gsum(csn)/max(gsum(csd),1),kcs,kx,kz)
    WH={};FL={};XW={};SW={}
    for key,a in A.items():
        swn=gsum(a['swn']); swd=gsum(a['swd']); bipd=gsum(a['bipd'])
        WH[key]=smooth2d(a['whn'],a['swn'],gsum(a['whn'])/max(swn,1),kwh,kx,kz)
        FL[key]=smooth2d(a['fln'],a['swn'],gsum(a['fln'])/max(swn,1),kfl,kx,kz)
        XW[key]=smooth2d(a['bipn'],a['bipd'],gsum(a['bipn'])/max(bipd,1),xwK,kx,kz)
        coll=smooth2d(a['swn'],a['swd'],swn/swd if swd else 0.0,kcoll,kx,kz)
        SW[key]={c:smooth2d(AC[(key,c)]['swn'],AC[(key,c)]['swd'],coll,swK,kx,kz) for c in COUNTS}
    for p in P:
        key=(p['_g'],p['_bh'],p['_ph']); c=p['_c']; i=p['_i']; j=p['_j']
        psw=SW[key][c][i][j]; pwh=WH[key][i][j]; pfl=FL[key][i][j]
        pbip=max(0.0,1-pwh-pfl); vbip=XW[key][i][j]; pcs=PCS[i][j]
        sv=pwh*RV['whiff'].get(c,0)+pfl*RV['foul'].get(c,0)+pbip*vbip
        tv=pcs*RV['cs'].get(c,0)+(1-pcs)*RV['ball'].get(c,0)
        p['_v3']=psw*sv+(1-psw)*tv

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
    if p['_g']=='FF':
        v=sf(p.get('Velocity'))
        if v is not None: stf[k]['vs']+=v; stf[k]['vn']+=1
whiffrate={k:s['wh']/s['sw'] for k,s in stf.items() if s['sw']>=50}
ffvelo={k:s['vs']/s['vn'] for k,s in stf.items() if s['vn']>=30}
sec=collections.defaultdict(list)
for p in P:
    if (p.get('Game Date') or '')>='2026-05-01':
        d=p['_d']; xw=p['_xw']
        v=(xw-LG)/SCALE if (d=='In Play' and xw is not None) else (-sf(p.get('RunExp')) if sf(p.get('RunExp')) is not None else None)
        if v is not None: sec[(p['Pitcher'],p['Throws'])].append(v)
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

CUR=dict(phx=4.5,phz=0.22,xwK=200,swK=20,kwh=8,kfl=8,kcs=10,kcoll=6)
def run(**over):
    cfg=dict(CUR); cfg.update(over); build_score(**cfg); return evaluate()

# ── Part A: multi-seed n_prior ──
build_score(**CUR)
def nprior(by_p,Ns,seeds=25):
    sums=[0.0]*len(Ns); cnts=[0]*len(Ns)
    for s in range(seeds):
        rnd=random.Random(1000+s)
        shuffled={k:v[:] for k,v in by_p.items()}
        for v in shuffled.values(): rnd.shuffle(v)
        for idx,N in enumerate(Ns):
            xs=[];ys=[]
            for v in shuffled.values():
                if len(v)>=2*N: xs.append(sum(v[:N])/N); ys.append(sum(v[N:2*N])/N)
            r=pearson(xs,ys)
            if r is not None: sums[idx]+=r; cnts[idx]+=1
    rs=[(sums[i]/cnts[i] if cnts[i] else None) for i in range(len(Ns))]
    cross=None; prev=None
    for N,r in zip(Ns,rs):
        if r is None: continue
        if prev and cross is None and prev[1]<0.5<=r:
            (N0,r0)=prev; cross=N0+(0.5-r0)*(N-N0)/(r-r0)
        prev=(N,r)
    return rs,cross
bp=collections.defaultdict(list)
for p in P: bp[(p['Pitcher'],p['Throws'])].append(p['_v3'])
Ns=[60,80,100,120,150,200,300]
rs,cross=nprior(bp,Ns,seeds=25)
print("\n=== n_prior (25-seed averaged split-half r) ===")
print("  N   :", "  ".join(f"{N}:{r:.3f}" for N,r in zip(Ns,rs)))
print(f"  OVERALL n_prior (r=0.5 crossing) = {cross:.0f}")
for G in ['FF','SI','FC','SL','CU','CH']:
    bpg=collections.defaultdict(list)
    for p in P:
        if p['_g']==G: bpg[(p['Pitcher'],p['Throws'])].append(p['_v3'])
    rsg,cg=nprior(bpg,[40,60,80,100,150,200],seeds=25)
    print(f"  {G:3s} n_prior ≈ {('%.0f'%cg) if cg else '>range'}   ("+
          " ".join(f"{N}:{r:.2f}" for N,r in zip([40,60,80,100,150,200],rsg) if r is not None)+")")

# ── Part B: one-axis sweeps ──
def sweep(name,axis,values):
    print(f"\n=== {name} (current {CUR[axis]}) ===   reliab  rWhf   rVel   pred")
    for val in values:
        rel,rw,rv,rp=run(**{axis:val})
        star=" *cur" if val==CUR[axis] else ""
        print(f"   {axis}={val:<6}        {rel:6.3f} {rw:6.3f} {rv:6.3f} {rp:6.3f}{star}")
sweep("smoothing horizontal (in)","phx",[3.5,4.0,4.5,5.0,5.5])
sweep("smoothing vertical (zone)","phz",[0.17,0.20,0.22,0.25,0.28])
sweep("contact shrink xwK","xwK",[100,150,200,300,500])
sweep("swing-per-count shrink swK","swK",[10,15,20,30,50])
sweep("whiff/foul shrink (kwh=kfl)","kwh",[4,8,16,32])  # note: only kwh varied
sweep("called-strike shrink kcs","kcs",[5,10,20,40])
sweep("swing-collapsed shrink kcoll","kcoll",[3,6,10,20])
print("\nNOTE: kwh sweep varies whiff only (foul held at 8); fine for sensitivity read.")
print("Objective: keep rWhf low (stuff-independent), then maximize reliab+pred. Round where flat.")
