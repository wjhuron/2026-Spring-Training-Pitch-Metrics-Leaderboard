"""locplus_stabilize.py — measure the regression constant (n_prior) for V3 Loc+.

Builds V3 at the LOCKED config (2in x 0.10z grid, 4.5in/0.22 physical
smoothing, xwK=200, swK=20) then measures split-half reliability as a
function of pitches-per-estimate. The N at which split-half r = 0.5 IS the
Bayesian regression constant (true-score model: rel(n)=n/(n+k); r=.5 -> k=n).
"""
import pickle, math, collections, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline_sdplus import classify_zone
random.seed(17)

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
    p['_bh']=p['Bats']; p['_ph']=p['Throws']; p['_d']=p.get('Description'); p['_xw']=sf(p.get('xwOBA'))

acc={k:collections.defaultdict(lambda:[0.0,0]) for k in ('whiff','foul','cs','ball')}
for p in P:
    re=sf(p.get('RunExp'))
    if re is None: continue
    key={'Swinging Strike':'whiff','Foul':'foul','Called Strike':'cs','Ball':'ball'}.get(p['_d'])
    if key: acc[key][p['_c']][0]+=-re; acc[key][p['_c']][1]+=1
RV={k:{c:(s/n if n else 0.0) for c,(s,n) in dd.items()} for k,dd in acc.items()}

# ── locked config ──
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

print("building V3 surfaces (locked config) ...",flush=True)
A={key:{'swn':zeros(),'swd':zeros(),'whn':zeros(),'fln':zeros(),'bipn':zeros(),'bipd':zeros()} for key in GROUPS}
AC={(key,c):{'swn':zeros(),'swd':zeros()} for key in GROUPS for c in COUNTS}
csn=zeros(); csd=zeros()
for p in P:
    key=(p['_g'],p['_bh'],p['_ph']); c=p['_c']; i=xb(p['_px']); j=zb(p['_zn']); d=p['_d']
    a=A[key]; ac=AC[(key,c)]; a['swd'][i][j]+=1; ac['swd'][i][j]+=1
    if d in SWING_DESC:
        a['swn'][i][j]+=1; ac['swn'][i][j]+=1
        if d=='Swinging Strike': a['whn'][i][j]+=1
        elif d=='Foul': a['fln'][i][j]+=1
        elif d=='In Play' and p['_xw'] is not None: a['bipn'][i][j]+=(p['_xw']-LG)/SCALE; a['bipd'][i][j]+=1
    if d in TAKE_DESC:
        csd[i][j]+=1
        if d=='Called Strike': csn[i][j]+=1
def gsum(arr): return sum(sum(r) for r in arr)
PCS=smooth2d(csn,csd,gsum(csn)/max(gsum(csd),1),10)
WH={};FL={};XW={};SW={}
for key in GROUPS:
    a=A[key]; swd=gsum(a['swd']); swn=gsum(a['swn']); bipd=gsum(a['bipd'])
    WH[key]=smooth2d(a['whn'],a['swn'],gsum(a['whn'])/max(swn,1),8)
    FL[key]=smooth2d(a['fln'],a['swn'],gsum(a['fln'])/max(swn,1),8)
    XW[key]=smooth2d(a['bipn'],a['bipd'],gsum(a['bipn'])/max(bipd,1),XWK)
    coll=smooth2d(a['swn'],a['swd'],swn/swd if swd else 0.0,6)
    SW[key]={c:smooth2d(AC[(key,c)]['swn'],AC[(key,c)]['swd'],coll,SWK) for c in COUNTS}
for p in P:
    key=(p['_g'],p['_bh'],p['_ph']); c=p['_c']; i=xb(p['_px']); j=zb(p['_zn'])
    psw=SW[key][c][i][j]; pwh=WH[key][i][j]; pfl=FL[key][i][j]
    pbip=max(0.0,1-pwh-pfl); vbip=XW[key][i][j]; pcs=PCS[i][j]
    sv=pwh*RV['whiff'].get(c,0)+pfl*RV['foul'].get(c,0)+pbip*vbip
    tv=pcs*RV['cs'].get(c,0)+(1-pcs)*RV['ball'].get(c,0)
    p['_v3']=psw*sv+(1-psw)*tv

# ── stabilization curve ──
by_p=collections.defaultdict(list)
for p in P: by_p[(p['Pitcher'],p['Throws'])].append(p['_v3'])
for v in by_p.values(): random.shuffle(v)
print("\n  N/half   n_pitchers   split-half r   implied rel(full≈2N)")
prev=None; cross=None
for N in [50,75,100,150,200,300,400,500,600,800]:
    xs=[];ys=[]
    for vals in by_p.values():
        if len(vals)>=2*N:
            xs.append(sum(vals[:N])/N); ys.append(sum(vals[N:2*N])/N)
    r=pearson(xs,ys)
    if r is None: continue
    sb=2*r/(1+r)  # Spearman-Brown: reliability of a 2N-pitch estimate
    print(f"  {N:5d}   {len(xs):8d}     {r:6.3f}        {sb:6.3f}")
    if prev and cross is None and prev[1]<0.5<=r:
        # linear interpolate N where r=0.5
        (N0,r0)=prev; cross=N0+(0.5-r0)*(N-N0)/(r-r0)
    prev=(N,r)
print("\n  => split-half r crosses 0.5 at N ≈", round(cross) if cross else ">range",
      "pitches per half  ==  regression constant n_prior")
