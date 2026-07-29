"""locplus_binsweep.py — find the best grid bin size for V3 Loc+.

Bin size and smoothing bandwidth are coupled: bandwidth is in CELLS, so a
finer grid needs a wider cell-kernel for the same PHYSICAL smoothing. To
test bin size fairly we hold the physical smoothing scale fixed (in inches
horizontally, zone-fractions vertically) and derive per-axis cell bandwidths.
Kernel is anisotropic + separable (Gaussian), so fine grids stay fast.

V3 config locked from prior sweep: contact-shrink xwK=200, swing-prob swK=20,
true count level, no demean. Only grid + physical bandwidth vary here.
"""
import pickle, math, collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline_sdplus import classify_zone

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL=os.path.join(ROOT,'data','all_pitches_rs_cache.pkl')
LG,SCALE=0.3169,1.2393
PTG={'FF':'FF','FA':'FF','SI':'SI','FC':'FC','CF':'FC','SL':'SL','ST':'SL','SV':'SL',
     'SW':'SL','CU':'CB','KC':'CB','CS':'CB','CH':'CH','FS':'CH'}
def ptg(pt): return PTG.get(pt,'OTHER') if pt else None
def sf(x):
    try: return float(x)
    except (TypeError,ValueError): return None
SWING_DESC={'Swinging Strike','Foul','In Play'}; TAKE_DESC={'Ball','Called Strike'}
EXCLUDE={'Hit By Pitch','Foul Bunt','Missed Bunt','Pitchout','Swinging Pitchout'}
BUNT={'bunt','bunt_grounder','bunt_popup','bunt_line_drive'}
COUNTS=[(b,s) for b in range(4) for s in range(3)]; HANDS=('L','R')
GROUPS=[(g,bh,ph) for g in ['FF','SI','FC','SL','CB','CH','OTHER'] for bh in HANDS for ph in HANDS]
X_MIN,X_MAX=-1.5,1.5; Z_MIN,Z_MAX=-0.6,1.6

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
    if ptg(p.get('Pitch Type')) is None: return False
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
    p['_g']=ptg(p.get('Pitch Type')); p['_c']=get_count(p)
    p['_px']=sf(p.get('PlateX')); p['_zn']=znorm(p)
    p['_bh']=p['Bats']; p['_ph']=p['Throws']; p['_xrv']=xrv_hitter(p)
    p['_d']=p.get('Description'); p['_xw']=sf(p.get('xwOBA')); p['_z']=classify_zone(p)

# count scalars (bin-independent)
acc={k:collections.defaultdict(lambda:[0.0,0]) for k in ('whiff','foul','cs','ball')}
for p in P:
    re=sf(p.get('RunExp'))
    if re is None: continue
    key={'Swinging Strike':'whiff','Foul':'foul','Called Strike':'cs','Ball':'ball'}.get(p['_d'])
    if key: acc[key][p['_c']][0]+=-re; acc[key][p['_c']][1]+=1
RV={k:{c:(s/n if n else 0.0) for c,(s,n) in dd.items()} for k,dd in acc.items()}

# V0 baseline (bin-independent)
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
    p['_v0']=(s+50*ZMARG.get((z,p['_c'],p['_g']),0.0))/(n+50)

# eval infra (bin-independent)
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
    if (p.get('Game Date') or '')>='2026-05-01' and p['_xrv'] is not None:
        sec[(p['Pitcher'],p['Throws'])].append(p['_xrv'])
sec_xrv={k:sum(v)/len(v) for k,v in sec.items() if len(v)>=MIN_PRED}
def aggf(field,pitches):
    out=collections.defaultdict(list)
    for p in pitches:
        v=p.get(field)
        if v is not None: out[(p['Pitcher'],p['Throws'])].append(v)
    return {k:(sum(v)/len(v),len(v)) for k,v in out.items()}
def evaluate(field):
    sa=aggf(field,H[0]); sb=aggf(field,H[1])
    com=[k for k in sa if k in sb and sa[k][1]>=MIN_HALF and sb[k][1]>=MIN_HALF]
    rel=pearson([sa[k][0] for k in com],[sb[k][0] for k in com])
    full=aggf(field,P); qual={k:val for k,(val,n) in full.items() if n>=MIN_PRED}
    kw=[k for k in qual if k in whiffrate]; rw=pearson([qual[k] for k in kw],[whiffrate[k] for k in kw])
    kv=[k for k in qual if k in ffvelo]; rv=pearson([qual[k] for k in kv],[ffvelo[k] for k in kv])
    fa=aggf(field,firsts); fq={k:val for k,(val,n) in fa.items() if n>=MIN_PRED}
    kp=[k for k in fq if k in sec_xrv]; rp=pearson([fq[k] for k in kp],[sec_xrv[k] for k in kp])
    return rel,abs(rw),abs(rv),rp

print("baseline V0:", tuple(round(x,3) for x in evaluate('_v0')))

# ── separable anisotropic Gaussian smoother ──
def k1d(bw):
    win=max(1,int(math.ceil(3*bw)))
    return [(d,math.exp(-0.5*(d/bw)**2)) for d in range(-win,win+1)]
def zeros(NX,NZ): return [[0.0]*NZ for _ in range(NX)]
def smooth2d(num,den,prior,kp,kx,kz,NX,NZ):
    tn=zeros(NX,NZ); td=zeros(NX,NZ)
    for i in range(NX):
        ni,di_,tni,tdi=num[i],den[i],tn[i],td[i]
        for j in range(NZ):
            sn=sd=0.0
            for dj,w in kz:
                jj=j+dj
                if 0<=jj<NZ: sn+=w*ni[jj]; sd+=w*di_[jj]
            tni[j]=sn; tdi[j]=sd
    out=zeros(NX,NZ); pdict=not isinstance(prior,(int,float))
    for i in range(NX):
        oi=out[i]
        for j in range(NZ):
            sn=sd=0.0
            for di2,w in kx:
                ii=i+di2
                if 0<=ii<NX: sn+=w*tn[ii][j]; sd+=w*td[ii][j]
            pr=prior[i][j] if pdict else prior
            s=sd+kp
            oi[j]=(sn+kp*pr)/s if s>0 else pr
    return out

def run(bin_x_in, bin_z_frac, phys_x_in, phys_z_frac, xwK=200, swK=20):
    bx=bin_x_in/12.0; NX=int(round((X_MAX-X_MIN)/bx)); NZ=int(round((Z_MAX-Z_MIN)/bin_z_frac))
    def xb(px): return min(max(int((px-X_MIN)/bx),0),NX-1)
    def zb(zn): return min(max(int((zn-Z_MIN)/bin_z_frac),0),NZ-1)
    kx=k1d(phys_x_in/bin_x_in); kz=k1d(phys_z_frac/bin_z_frac)
    # accumulators
    A={key:{'swn':zeros(NX,NZ),'swd':zeros(NX,NZ),'whn':zeros(NX,NZ),'fln':zeros(NX,NZ),
            'bipn':zeros(NX,NZ),'bipd':zeros(NX,NZ)} for key in GROUPS}
    AC={(key,c):{'swn':zeros(NX,NZ),'swd':zeros(NX,NZ)} for key in GROUPS for c in COUNTS}
    csn=zeros(NX,NZ); csd=zeros(NX,NZ)
    for p in P:
        key=(p['_g'],p['_bh'],p['_ph']); c=p['_c']; i=xb(p['_px']); j=zb(p['_zn']); d=p['_d']
        a=A[key]; ac=AC[(key,c)]
        a['swd'][i][j]+=1; ac['swd'][i][j]+=1
        if d in SWING_DESC:
            a['swn'][i][j]+=1; ac['swn'][i][j]+=1
            if d=='Swinging Strike': a['whn'][i][j]+=1
            elif d=='Foul': a['fln'][i][j]+=1
            elif d=='In Play' and p['_xw'] is not None:
                a['bipn'][i][j]+=(p['_xw']-LG)/SCALE; a['bipd'][i][j]+=1
        if d in TAKE_DESC:
            csd[i][j]+=1
            if d=='Called Strike': csn[i][j]+=1
    def gsum(arr): return sum(sum(r) for r in arr)
    glob_cs=gsum(csn)/max(gsum(csd),1)
    PCS=smooth2d(csn,csd,glob_cs,10,kx,kz,NX,NZ)
    WH={};FL={};XW={};SW={}
    for key in GROUPS:
        a=A[key]; swd=gsum(a['swd']); swn=gsum(a['swn']); bipd=gsum(a['bipd'])
        g_sw=swn/swd if swd else 0.0; g_wh=gsum(a['whn'])/max(swn,1)
        g_fl=gsum(a['fln'])/max(swn,1); g_bip=gsum(a['bipn'])/max(bipd,1)
        WH[key]=smooth2d(a['whn'],a['swn'],g_wh,8,kx,kz,NX,NZ)
        FL[key]=smooth2d(a['fln'],a['swn'],g_fl,8,kx,kz,NX,NZ)
        XW[key]=smooth2d(a['bipn'],a['bipd'],g_bip,xwK,kx,kz,NX,NZ)
        coll=smooth2d(a['swn'],a['swd'],g_sw,6,kx,kz,NX,NZ)
        SW[key]={c:smooth2d(AC[(key,c)]['swn'],AC[(key,c)]['swd'],coll,swK,kx,kz,NX,NZ) for c in COUNTS}
    for p in P:
        key=(p['_g'],p['_bh'],p['_ph']); c=p['_c']; i=xb(p['_px']); j=zb(p['_zn'])
        psw=SW[key][c][i][j]; pwh=WH[key][i][j]; pfl=FL[key][i][j]
        pbip=max(0.0,1-pwh-pfl); vbip=XW[key][i][j]; pcs=PCS[i][j]
        sv=pwh*RV['whiff'].get(c,0)+pfl*RV['foul'].get(c,0)+pbip*vbip
        tv=pcs*RV['cs'].get(c,0)+(1-pcs)*RV['ball'].get(c,0)
        p['_v3']=psw*sv+(1-psw)*tv
    return evaluate('_v3'),NX,NZ

print(f"\n{'config':46s} {'grid':>9s} {'reliab':>7s} {'rWhf':>6s} {'rVel':>6s} {'pred':>6s}")
print("-"*86)
# Pass 1: fix physical smoothing (~4.5in x, 0.22 zone z), vary bin resolution
PHX,PHZ=4.5,0.22
for bx_in,bz in [(1.0,0.05),(1.5,0.075),(2.0,0.10),(3.0,0.15),(4.0,0.20)]:
    (rel,rw,rv,rp),NX,NZ=run(bx_in,bz,PHX,PHZ)
    print(f"{'bins '+str(bx_in)+'in x '+str(bz)+'z  phys='+str(PHX)+'/'+str(PHZ):46s} "
          f"{str(NX)+'x'+str(NZ):>9s} {rel:7.3f} {rw:6.3f} {rv:6.3f} {rp:6.3f}",flush=True)
print("-"*86)
# Pass 2: at fine bins, vary physical smoothing
for PHX2,PHZ2 in [(3.5,0.17),(4.5,0.22),(5.5,0.27),(6.5,0.32)]:
    (rel,rw,rv,rp),NX,NZ=run(1.5,0.075,PHX2,PHZ2)
    print(f"{'bins 1.5in/0.075z  phys='+str(PHX2)+'/'+str(PHZ2):46s} "
          f"{str(NX)+'x'+str(NZ):>9s} {rel:7.3f} {rw:6.3f} {rv:6.3f} {rp:6.3f}",flush=True)
print("-"*86)
print("reliab↑  rWhf↓ rVel↓ (stuff leak)  pred↑   [physical smoothing held constant within each pass]")
