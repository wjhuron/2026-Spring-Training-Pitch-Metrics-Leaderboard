"""locplus_lab.py — experimentation harness for the next-gen Loc+ metric.

Goal: compare candidate Loc+ formulas on three objective functions and
print a leaderboard so we can pick the best construction empirically.

Variants
--------
  V0  current   : 5 attack zones x count x pt x hand, raw cell xRV (mirrors
                  pipeline_locplus exactly via imported classify_zone)
  V1  demean    : V0 minus the (count, pt) baseline -> within-count location
  V2  grid      : fine (PlateX, z_norm) grid xRV per (pt, hand, count),
                  kernel-smoothed; raw and count-demeaned
  V3  decomp    : full decomposition
                  ExpRV = Pswing*[Pwhiff*rvWhiff + Pfoul*rvFoul + Pbip*xwOBAcon]
                        + (1-Pswing)*[Pcs*rvCS + (1-Pcs)*rvBall]
                  surfaces count-independent, value weights count-dependent;
                  raw and count-demeaned

Objective functions (printed per variant)
  reliability  : split-half (odd/even game dates) Pearson of per-pitcher raw
  stuff-indep  : |Pearson(raw, whiff/swing)| and |Pearson(raw, FF velo)|
  predictive   : Pearson(first-half raw, second-half actual xRV allowed)

All raw scores are HITTER perspective (lower = better for the pitcher).
"""
import pickle, math, collections, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline_sdplus import classify_zone, get_count as sd_get_count  # faithful V0 baseline

PKL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'data', 'all_pitches_rs_cache.pkl')
LG_WOBA = 0.3169
WOBA_SCALE = 1.2393

PITCH_TYPE_GROUPS = {
    'FF': 'FF', 'FA': 'FF', 'SI': 'SI', 'FC': 'FC', 'CF': 'FC',
    'SL': 'SL', 'ST': 'SL', 'SV': 'SL', 'SW': 'SL',
    'CU': 'CB', 'KC': 'CB', 'CS': 'CB', 'CH': 'CH', 'FS': 'CH',
}
def ptg(pt):
    return PITCH_TYPE_GROUPS.get(pt, 'OTHER') if pt else None

def sf(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

SWING_DESC = {'Swinging Strike', 'Foul', 'In Play'}
WHIFF_DESC = {'Swinging Strike'}
FOUL_DESC  = {'Foul'}
TAKE_DESC  = {'Ball', 'Called Strike'}
EXCLUDE_DESC = {'Hit By Pitch', 'Foul Bunt', 'Missed Bunt', 'Pitchout', 'Swinging Pitchout'}
BUNT_BB = {'bunt', 'bunt_grounder', 'bunt_popup', 'bunt_line_drive'}

COUNTS = [(b, s) for b in range(4) for s in range(3)]
HANDS = ('L', 'R')

# ── grid ────────────────────────────────────────────────────────────────
X_MIN, X_MAX, XW = -1.5, 1.5, 1.0 / 6.0          # 2-inch bins
NX = int(round((X_MAX - X_MIN) / XW))            # 18
Z_MIN, Z_MAX, ZW = -0.6, 1.6, 0.1               # zone-normalized
NZ = int(round((Z_MAX - Z_MIN) / ZW))            # 22

def xbin(px):
    return min(max(int((px - X_MIN) / XW), 0), NX - 1)
def zbin(zn):
    return min(max(int((zn - Z_MIN) / ZW), 0), NZ - 1)

def znorm(p):
    pz, top, bot = sf(p.get('PlateZ')), sf(p.get('SzTop')), sf(p.get('SzBot'))
    if None in (pz, top, bot) or top <= bot:
        return None
    return (pz - bot) / (top - bot)

# ── kernel ──────────────────────────────────────────────────────────────
BW = 1.3
WIN = 3
KERNEL = []
for di in range(-WIN, WIN + 1):
    for dj in range(-WIN, WIN + 1):
        KERNEL.append((di, dj, math.exp(-0.5 * ((di / BW) ** 2 + (dj / BW) ** 2))))

def smooth(num, den, prior, kprior):
    """Nadaraya-Watson kernel regression on the grid with a prior pseudo-count.
    num/den: dict[(i,j)]->float. prior: scalar or dict[(i,j)]->rate.
    Returns dict[(i,j)]->smoothed rate for every cell."""
    out = {}
    pd = isinstance(prior, dict)
    for i in range(NX):
        for j in range(NZ):
            sn = sd = 0.0
            for di, dj, w in KERNEL:
                ii, jj = i + di, j + dj
                if 0 <= ii < NX and 0 <= jj < NZ:
                    c = (ii, jj)
                    if c in den:
                        sn += w * num.get(c, 0.0)
                        sd += w * den[c]
            pr = prior.get((i, j), 0.0) if pd else prior
            out[(i, j)] = (sn + kprior * pr) / (sd + kprior) if (sd + kprior) > 0 else pr
    return out

# ── eligibility / value ─────────────────────────────────────────────────
def scorable(p):
    if p.get('_source') != 'MLB':
        return False
    if p.get('Description') in EXCLUDE_DESC:
        return False
    if p.get('BBType') in BUNT_BB:
        return False
    if p.get('Event') == 'Intent Walk':
        return False
    if znorm(p) is None or sf(p.get('PlateX')) is None:
        return False
    if ptg(p.get('Pitch Type')) is None:
        return False
    if p.get('Bats') not in HANDS or p.get('Throws') not in HANDS:
        return False
    if get_count(p) is None:
        return False
    return True

def get_count(p):
    c = p.get('Count')
    if not isinstance(c, str) or '-' not in c:
        return None
    try:
        b, s = c.split('-', 1)
        b, s = int(b), int(s)
    except (TypeError, ValueError):
        return None
    if 0 <= b <= 3 and 0 <= s <= 2:
        return (b, s)
    return None

def xrv_hitter(p):
    """Per-pitch luck-neutral hitter-perspective RV (higher = better hitter)."""
    if p.get('Description') == 'In Play':
        xw = sf(p.get('xwOBA'))
        if xw is not None:
            return (xw - LG_WOBA) / WOBA_SCALE
    re = sf(p.get('RunExp'))
    return -re if re is not None else None

def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx <= 0 or sy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sx * sy)


# ═════════════════════════════════════════════════════════════════════════
#  LOAD
# ═════════════════════════════════════════════════════════════════════════
print("loading pickle ...", flush=True)
with open(PKL, 'rb') as f:
    ALL = pickle.load(f)
PITCHES = [p for p in ALL if scorable(p)]
print(f"  {len(ALL)} total -> {len(PITCHES)} scorable MLB pitches", flush=True)

# precompute per-pitch fields we reuse
for p in PITCHES:
    p['_g']  = ptg(p.get('Pitch Type'))
    p['_c']  = get_count(p)
    p['_i']  = xbin(sf(p.get('PlateX')))
    p['_j']  = zbin(znorm(p))
    p['_bh'] = p['Bats']
    p['_ph'] = p['Throws']
    p['_xrv'] = xrv_hitter(p)


# ═════════════════════════════════════════════════════════════════════════
#  COUNT-VALUE SCALARS  (hitter perspective)
# ═════════════════════════════════════════════════════════════════════════
def count_scalars():
    acc = {k: collections.defaultdict(lambda: [0.0, 0])
           for k in ('whiff', 'foul', 'cs', 'ball')}
    for p in PITCHES:
        re = sf(p.get('RunExp'))
        if re is None:
            continue
        d = p.get('Description')
        c = p['_c']
        v = -re
        if d == 'Swinging Strike':
            acc['whiff'][c][0] += v; acc['whiff'][c][1] += 1
        elif d == 'Foul':
            acc['foul'][c][0] += v; acc['foul'][c][1] += 1
        elif d == 'Called Strike':
            acc['cs'][c][0] += v; acc['cs'][c][1] += 1
        elif d == 'Ball':
            acc['ball'][c][0] += v; acc['ball'][c][1] += 1
    out = {}
    for k, dd in acc.items():
        out[k] = {c: (s / n if n else 0.0) for c, (s, n) in dd.items()}
    return out

RV = count_scalars()
print("\ncount value scalars (hitter persp, + = good for hitter):")
print("  count  rvBall   rvCS    rvWhiff  rvFoul")
for c in COUNTS:
    print(f"  {c[0]}-{c[1]}   {RV['ball'].get(c,0):+.4f} {RV['cs'].get(c,0):+.4f} "
          f"{RV['whiff'].get(c,0):+.4f} {RV['foul'].get(c,0):+.4f}")


# ═════════════════════════════════════════════════════════════════════════
#  SURFACES (decomposition)
# ═════════════════════════════════════════════════════════════════════════
print("\nbuilding surfaces ...", flush=True)

# Pcs: global geometric called-strike probability over taken pitches
cs_num = collections.defaultdict(float)
cs_den = collections.defaultdict(float)
for p in PITCHES:
    d = p.get('Description')
    if d in TAKE_DESC:
        cell = (p['_i'], p['_j'])
        cs_den[cell] += 1
        if d == 'Called Strike':
            cs_num[cell] += 1
glob_cs = sum(cs_num.values()) / max(sum(cs_den.values()), 1)
PCS = smooth(cs_num, cs_den, glob_cs, kprior=10)

# per (ptg,bh,ph): whiff/swing, foul/swing, xwOBAcon value, swing/all (per count)
groups = [(g, bh, ph) for g in ['FF','SI','FC','SL','CB','CH','OTHER']
          for bh in HANDS for ph in HANDS]

WHIFF, FOUL, XWCON = {}, {}, {}
SWING = {}   # SWING[(g,bh,ph)][count] = smoothed grid

# accumulators
acc_sw_num = collections.defaultdict(lambda: collections.defaultdict(float))  # [(g,bh,ph)][cell]
acc_sw_den = collections.defaultdict(lambda: collections.defaultdict(float))
acc_wh_num = collections.defaultdict(lambda: collections.defaultdict(float))
acc_fl_num = collections.defaultdict(lambda: collections.defaultdict(float))
acc_bip_num = collections.defaultdict(lambda: collections.defaultdict(float))
acc_bip_den = collections.defaultdict(lambda: collections.defaultdict(float))
# swing per count: [(g,bh,ph,count)][cell]
acc_swc_num = collections.defaultdict(lambda: collections.defaultdict(float))
acc_swc_den = collections.defaultdict(lambda: collections.defaultdict(float))

for p in PITCHES:
    key = (p['_g'], p['_bh'], p['_ph'])
    cell = (p['_i'], p['_j'])
    d = p.get('Description')
    is_swing = d in SWING_DESC
    acc_sw_den[key][cell] += 1
    acc_swc_den[(key, p['_c'])][cell] += 1
    if is_swing:
        acc_sw_num[key][cell] += 1
        acc_swc_num[(key, p['_c'])][cell] += 1
        if d == 'Swinging Strike':
            acc_wh_num[key][cell] += 1
        elif d == 'Foul':
            acc_fl_num[key][cell] += 1
        elif d == 'In Play':
            xw = sf(p.get('xwOBA'))
            v = (xw - LG_WOBA) / WOBA_SCALE if xw is not None else (
                -sf(p.get('RunExp')) if sf(p.get('RunExp')) is not None else None)
            if v is not None:
                acc_bip_num[key][cell] += v
                acc_bip_den[key][cell] += 1

# global rates per group for priors
for key in groups:
    swd = sum(acc_sw_den[key].values())
    if swd == 0:
        WHIFF[key] = {(i, j): 0.0 for i in range(NX) for j in range(NZ)}
        FOUL[key]  = dict(WHIFF[key]); XWCON[key] = dict(WHIFF[key])
        SWING[key] = {c: dict(WHIFF[key]) for c in COUNTS}
        continue
    swn = sum(acc_sw_num[key].values())
    g_sw = swn / swd
    g_wh = sum(acc_wh_num[key].values()) / max(swn, 1)
    g_fl = sum(acc_fl_num[key].values()) / max(swn, 1)
    bipd = sum(acc_bip_den[key].values())
    g_bip = sum(acc_bip_num[key].values()) / max(bipd, 1)

    # whiff/foul are conditioned on swings (den = swings)
    WHIFF[key] = smooth(acc_wh_num[key], acc_sw_num[key], g_wh, kprior=8)
    FOUL[key]  = smooth(acc_fl_num[key], acc_sw_num[key], g_fl, kprior=8)
    XWCON[key] = smooth(acc_bip_num[key], acc_bip_den[key], g_bip, kprior=12)
    # swing prob per count, prior = group-collapsed swing surface
    coll_sw = smooth(acc_sw_num[key], acc_sw_den[key], g_sw, kprior=6)
    SWING[key] = {}
    for c in COUNTS:
        SWING[key][c] = smooth(acc_swc_num[(key, c)], acc_swc_den[(key, c)],
                               coll_sw, kprior=20)

print("  surfaces built", flush=True)


def exprv_decomp(p):
    key = (p['_g'], p['_bh'], p['_ph'])
    cell = (p['_i'], p['_j'])
    c = p['_c']
    psw = SWING[key][c][cell]
    pwh = WHIFF[key][cell]
    pfl = FOUL[key][cell]
    pbip = max(0.0, 1.0 - pwh - pfl)
    vbip = XWCON[key][cell]
    pcs = PCS[cell]
    swing_val = pwh * RV['whiff'].get(c, 0) + pfl * RV['foul'].get(c, 0) + pbip * vbip
    take_val  = pcs * RV['cs'].get(c, 0) + (1 - pcs) * RV['ball'].get(c, 0)
    return psw * swing_val + (1 - psw) * take_val


# ═════════════════════════════════════════════════════════════════════════
#  V0 / V1  — faithful 5-zone cell table (current metric)
# ═════════════════════════════════════════════════════════════════════════
print("building zone tables (V0/V1) ...", flush=True)
ZCELL = collections.defaultdict(lambda: [0.0, 0])           # (zone,count,g,bh,ph)
ZMARG = collections.defaultdict(lambda: [0.0, 0])           # (zone,count,g)
CBASE = collections.defaultdict(lambda: [0.0, 0])           # (count,g) baseline
for p in PITCHES:
    if p['_xrv'] is None:
        continue
    z = classify_zone(p)
    if z is None:
        continue
    g, c = p['_g'], p['_c']
    v = p['_xrv']
    ZCELL[(z, c, g, p['_bh'], p['_ph'])][0] += v; ZCELL[(z, c, g, p['_bh'], p['_ph'])][1] += 1
    ZMARG[(z, c, g)][0] += v; ZMARG[(z, c, g)][1] += 1
    CBASE[(c, g)][0] += v; CBASE[(c, g)][1] += 1
ZMARG = {k: (s / n) for k, (s, n) in ZMARG.items()}
CBASE = {k: (s / n) for k, (s, n) in CBASE.items()}
def zcell_val(z, c, g, bh, ph, k=50):
    s, n = ZCELL.get((z, c, g, bh, ph), [0.0, 0])
    m = ZMARG.get((z, c, g), 0.0)
    return (s + k * m) / (n + k) if (n + k) else 0.0


# ═════════════════════════════════════════════════════════════════════════
#  V2 — fine-grid xRV per (g,bh,ph,count)
# ═════════════════════════════════════════════════════════════════════════
print("building grid xRV tables (V2) ...", flush=True)
gx_num = collections.defaultdict(lambda: collections.defaultdict(float))   # [(g,bh,ph)][cell]
gx_den = collections.defaultdict(lambda: collections.defaultdict(float))
gxc_num = collections.defaultdict(lambda: collections.defaultdict(float))  # [(g,bh,ph,count)][cell]
gxc_den = collections.defaultdict(lambda: collections.defaultdict(float))
for p in PITCHES:
    if p['_xrv'] is None:
        continue
    key = (p['_g'], p['_bh'], p['_ph']); cell = (p['_i'], p['_j'])
    gx_num[key][cell] += p['_xrv']; gx_den[key][cell] += 1
    gxc_num[(key, p['_c'])][cell] += p['_xrv']; gxc_den[(key, p['_c'])][cell] += 1
GRIDV = {}   # [(g,bh,ph)][count] -> smoothed grid
for key in groups:
    den = sum(gx_den[key].values())
    g_mean = sum(gx_num[key].values()) / den if den else 0.0
    coll = smooth(gx_num[key], gx_den[key], g_mean, kprior=8)
    GRIDV[key] = {}
    for c in COUNTS:
        GRIDV[key][c] = smooth(gxc_num[(key, c)], gxc_den[(key, c)], coll, kprior=25)
def grid_val(p):
    return GRIDV[(p['_g'], p['_bh'], p['_ph'])][p['_c']][(p['_i'], p['_j'])]


# ═════════════════════════════════════════════════════════════════════════
#  PRECOMPUTE per-pitch variant values (once), then score by summing
# ═════════════════════════════════════════════════════════════════════════
VARIANTS = ['V0', 'V1', 'V2', 'V2d', 'V3', 'V3d']
print("precomputing per-pitch variant values ...", flush=True)
for p in PITCHES:
    g, c = p['_g'], p['_c']
    cbase = CBASE.get((c, g), 0.0)
    z = classify_zone(p)
    v0 = zcell_val(z, c, g, p['_bh'], p['_ph']) if z is not None else None
    gv = grid_val(p)
    dv = exprv_decomp(p)
    p['_val'] = {
        'V0':  v0,
        'V1':  (v0 - cbase) if v0 is not None else None,
        'V2':  gv,
        'V2d': gv - cbase,
        'V3':  dv,
        'V3d': dv - cbase,
    }

def score(variant, pitches):
    out = collections.defaultdict(list)
    for p in pitches:
        v = p['_val'][variant]
        if v is not None:
            out[(p['Pitcher'], p['Throws'])].append(v)
    return {k: (sum(vs) / len(vs), len(vs)) for k, vs in out.items()}


# ═════════════════════════════════════════════════════════════════════════
#  OBJECTIVE FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════
MIN_HALF = 150
MIN_FULL = 250

# split halves by alternating game date
dates_by_p = collections.defaultdict(set)
for p in PITCHES:
    dates_by_p[(p['Pitcher'], p['Throws'])].add(p.get('Game Date'))
half_of = {}
for k, ds in dates_by_p.items():
    for idx, d in enumerate(sorted(ds)):
        half_of[(k, d)] = idx % 2
def half_pitches(h):
    return [p for p in PITCHES if half_of.get(((p['Pitcher'], p['Throws']), p.get('Game Date'))) == h]

H0, H1 = half_pitches(0), half_pitches(1)
firsts  = [p for p in PITCHES if (p.get('Game Date') or '') < '2026-05-01']
seconds = [p for p in PITCHES if (p.get('Game Date') or '') >= '2026-05-01']

# stuff proxies (full sample): whiff/swing, FF velo
stf = collections.defaultdict(lambda: {'sw': 0, 'wh': 0, 'vsum': 0.0, 'vn': 0})
for p in PITCHES:
    k = (p['Pitcher'], p['Throws'])
    d = p.get('Description')
    if d in SWING_DESC:
        stf[k]['sw'] += 1
        if d == 'Swinging Strike':
            stf[k]['wh'] += 1
    if p['_g'] == 'FF':
        v = sf(p.get('Velocity'))
        if v is not None:
            stf[k]['vsum'] += v; stf[k]['vn'] += 1
whiffrate = {k: (s['wh'] / s['sw']) for k, s in stf.items() if s['sw'] >= 50}
ffvelo    = {k: (s['vsum'] / s['vn']) for k, s in stf.items() if s['vn'] >= 30}

# second-half actual xRV allowed
sec_actual = collections.defaultdict(list)
for p in seconds:
    if p['_xrv'] is not None:
        sec_actual[(p['Pitcher'], p['Throws'])].append(p['_xrv'])
sec_xrv = {k: sum(vs) / len(vs) for k, vs in sec_actual.items() if len(vs) >= MIN_FULL}

print("\n" + "=" * 78)
print(f"{'variant':7s} {'reliab':>8s} {'|r:whiff|':>9s} {'|r:velo|':>9s} {'predict':>8s}  notes")
print("-" * 78)

base_full = score('V0', PITCHES)  # for n reference
for v in VARIANTS:
    sa = score(v, H0); sb = score(v, H1)
    common = [k for k in sa if k in sb and sa[k][1] >= MIN_HALF and sb[k][1] >= MIN_HALF]
    rel = pearson([sa[k][0] for k in common], [sb[k][0] for k in common])

    full = score(v, PITCHES)
    qual = {k: val for k, (val, n) in full.items() if n >= MIN_FULL}

    kk = [k for k in qual if k in whiffrate]
    rw = pearson([qual[k] for k in kk], [whiffrate[k] for k in kk])
    kv = [k for k in qual if k in ffvelo]
    rv = pearson([qual[k] for k in kv], [ffvelo[k] for k in kv])

    fa = score(v, firsts)
    fq = {k: val for k, (val, n) in fa.items() if n >= MIN_FULL}
    kp = [k for k in fq if k in sec_xrv]
    rp = pearson([fq[k] for k in kp], [sec_xrv[k] for k in kp])

    print(f"{v:7s} {rel:8.3f} {abs(rw):9.3f} {abs(rv):9.3f} {rp:8.3f}  "
          f"n_rel={len(common)} n_pred={len(kp)}")

# reference: does past actual xRV predict future actual xRV?
fa_actual = collections.defaultdict(list)
for p in firsts:
    if p['_xrv'] is not None:
        fa_actual[(p['Pitcher'], p['Throws'])].append(p['_xrv'])
fxrv = {k: sum(vs)/len(vs) for k, vs in fa_actual.items() if len(vs) >= MIN_FULL}
kp = [k for k in fxrv if k in sec_xrv]
print("-" * 78)
print(f"{'ref:xRV':7s} {'':8s} {'':9s} {'':9s} {pearson([fxrv[k] for k in kp],[sec_xrv[k] for k in kp]):8.3f}"
      f"  (first-half actual xRV -> second-half actual xRV)")
print("=" * 78)
print("reliab: split-half Pearson (higher=better). |r:whiff|,|r:velo|: stuff leakage (lower=better).")
print("predict: first-half score vs second-half actual xRV allowed (higher=better).")
