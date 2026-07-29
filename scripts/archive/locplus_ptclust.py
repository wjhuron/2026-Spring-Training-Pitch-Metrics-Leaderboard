"""locplus_ptclust.py — data-driven check of pitch-type grouping.

For each RAW pitch type (RHP vs RHH, the most-populated frame), build the
count-collapsed location-value surface (smoothed mean hitter-xRV: integrates
whiff, contact, and take). Then compute the sample-weighted correlation of
every pair of value surfaces. Pitch types whose value geography matches
(high corr) can share a group; distinct ones need their own.

This tests whether {FF | SI | CH+FS | all-breakers | FC} is actually right.
"""
import pickle, math, collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL=os.path.join(ROOT,'data','all_pitches_rs_cache.pkl')
LG,SCALE=0.3169,1.2393
def sf(x):
    try: return float(x)
    except (TypeError,ValueError): return None
EXCLUDE={'Hit By Pitch','Foul Bunt','Missed Bunt','Pitchout','Swinging Pitchout'}
BUNT={'bunt','bunt_grounder','bunt_popup','bunt_line_drive'}
X_MIN,X_MAX=-1.5,1.5; Z_MIN,Z_MAX=-0.6,1.6
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
    return p.get('Pitch Type') not in (None,'')

print("loading ...",flush=True)
with open(PKL,'rb') as f: ALL=pickle.load(f)
# RHP vs RHH frame for comparability
P=[p for p in ALL if scorable(p) and p.get('Throws')=='R' and p.get('Bats')=='R']
print(f"  {len(P)} RvR scorable",flush=True)

BIN_X_IN,BIN_Z=2.0,0.10; PHX,PHZ=4.5,0.22
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
    out=zeros()
    for i in range(NX):
        oi=out[i]
        for j in range(NZ):
            sn=sd=0.0
            for di2,w in kx:
                ii=i+di2
                if 0<=ii<NX: sn+=w*tn[ii][j]; sd+=w*td[ii][j]
            s=sd+kp; oi[j]=(sn+kp*prior)/s if s>0 else prior
    return out

# accumulate per raw pitch type
num=collections.defaultdict(zeros); den=collections.defaultdict(zeros); cnt=collections.Counter()
for p in P:
    v=xrv_hitter(p)
    if v is None: continue
    pt=p.get('Pitch Type'); i=xb(sf(p.get('PlateX'))); j=zb(znorm(p))
    num[pt][i][j]+=v; den[pt][i][j]+=1; cnt[pt]+=1

MIN_N=1500
types=[t for t in cnt if cnt[t]>=MIN_N]
# order by family for readability
order=['FF','FA','SI','FC','SL','ST','SW','SV','CU','KC','CS','CH','FS','KN','SC','EP']
types=sorted(types, key=lambda t: order.index(t) if t in order else 99)
print("\npitch types (RvR, n>=%d):"%MIN_N, {t:cnt[t] for t in types})
print("dropped (rare):", {t:cnt[t] for t in cnt if cnt[t]<MIN_N})

surf={}; dens={}
for t in types:
    g=sum(num[t][i][j] for i in range(NX) for j in range(NZ))/max(cnt[t],1)
    surf[t]=smooth2d(num[t],den[t],g,12); dens[t]=den[t]
def wcorr(a,b):
    da,db=dens[a],dens[b]; sa,sb=surf[a],surf[b]
    W=ma=mb=0.0
    for i in range(NX):
        for j in range(NZ):
            w=math.sqrt(da[i][j]*db[i][j])
            if w>0: W+=w; ma+=w*sa[i][j]; mb+=w*sb[i][j]
    if W<=0: return None
    ma/=W; mb/=W; cov=va=vb=0.0
    for i in range(NX):
        for j in range(NZ):
            w=math.sqrt(da[i][j]*db[i][j])
            if w>0:
                da_=sa[i][j]-ma; db_=sb[i][j]-mb
                cov+=w*da_*db_; va+=w*da_*da_; vb+=w*db_*db_
    return cov/math.sqrt(va*vb) if va>0 and vb>0 else None

print("\nsample-weighted correlation of location-value surfaces (RvR):")
print("      "+" ".join(f"{t:>5s}" for t in types))
M={}
for a in types:
    row=[]
    for b in types:
        c=wcorr(a,b); M[(a,b)]=c
        row.append("  -  " if c is None else f"{c:+.2f}")
    print(f"{a:>5s} "+" ".join(f"{x:>5s}" for x in row))

# nearest neighbor for each type
print("\nnearest neighbor (most similar other type):")
for a in types:
    best=max((b for b in types if b!=a), key=lambda b: (M[(a,b)] if M[(a,b)] is not None else -9))
    print(f"  {a}: {best} ({M[(a,best)]:+.2f})")

# greedy single-link clusters at thresholds
def clusters(th):
    parent={t:t for t in types}
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    for a in types:
        for b in types:
            if a<b and M[(a,b)] is not None and M[(a,b)]>=th:
                parent[find(a)]=find(b)
    comp=collections.defaultdict(list)
    for t in types: comp[find(t)].append(t)
    return list(comp.values())
for th in (0.90,0.85,0.80):
    print(f"\nsingle-link clusters at corr>={th}: "+
          " | ".join("{"+",".join(c)+"}" for c in clusters(th)))
