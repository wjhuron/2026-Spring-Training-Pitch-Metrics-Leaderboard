"""locplus_multiseason_test.py — does building Loc+ league surfaces from MULTIPLE
seasons (2021-2026) improve the metric? And do count-SPECIFIC physical surfaces
(now affordable with the extra data) help on top?

Loc+ scores each pitch against league-average location-value SURFACES built from
`_source=='MLB'` (currently the partial 2026 season only). The physical surfaces
(whiff/foul/contact/called-strike) are forced count-INDEPENDENT for sample size.
This test rebuilds those surfaces from more data and re-measures the 3 shipped
objectives, EVALUATED ON 2026 (only the surface-baseline changes):
  reliability      split-half (odd/even games) of per-pitcher Loc value
  stuff-indep      |corr| with whiff% and FF velo (must stay LOW)
  predictive       first-half Loc vs second-half actual xRV allowed

Baseline sources use the raw Statcast caches (2021-25) mapped to Wally's Loc+
schema; run-value count weights + all evaluation stay on 2026 to isolate the
surface-quality effect.

Usage: python3 scripts/locplus_multiseason_test.py
"""
import pickle, math, collections, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
LG, SCALE = 0.3172, 1.2343                       # live 2026 guts

GROUP = {'FF': 'FF', 'FA': 'FF', 'SI': 'SI', 'FC': 'FC', 'CF': 'FC',
         'SL': 'SL', 'ST': 'SL', 'SW': 'SL', 'SV': 'SL', 'CU': 'CU', 'KC': 'CU', 'CS': 'CU',
         'CH': 'CH', 'FS': 'CH'}
def grp(pt): return GROUP.get(pt, 'OTHER') if pt else None
def sf(x):
    try: return float(x)
    except (TypeError, ValueError): return None
SWING_DESC = {'Swinging Strike', 'Foul', 'In Play'}; TAKE_DESC = {'Ball', 'Called Strike'}
EXCLUDE = {'Hit By Pitch', 'Foul Bunt', 'Missed Bunt', 'Pitchout', 'Swinging Pitchout'}
COUNTS = [(b, s) for b in range(4) for s in range(3)]; HANDS = ('L', 'R')
X_MIN, X_MAX = -1.5, 1.5; Z_MIN, Z_MAX = -0.6, 1.6
BIN_X_IN, BIN_Z = 2.0, 0.10
bx = BIN_X_IN / 12.0; NX = int(round((X_MAX - X_MIN) / bx)); NZ = int(round((Z_MAX - Z_MIN) / BIN_Z))
def xb(px): return min(max(int((px - X_MIN) / bx), 0), NX - 1)
def zb(zn): return min(max(int((zn - Z_MIN) / BIN_Z), 0), NZ - 1)

# raw Statcast description -> Wally's Loc+ vocabulary
DMAP = {'ball': 'Ball', 'blocked_ball': 'Ball', 'called_strike': 'Called Strike',
        'swinging_strike': 'Swinging Strike', 'swinging_strike_blocked': 'Swinging Strike',
        'foul_tip': 'Swinging Strike', 'foul': 'Foul', 'hit_into_play': 'In Play',
        'hit_into_play_no_out': 'In Play', 'hit_into_play_score': 'In Play',
        'hit_by_pitch': 'Hit By Pitch', 'foul_bunt': 'Foul Bunt', 'missed_bunt': 'Missed Bunt',
        'bunt_foul_tip': 'Foul Bunt', 'pitchout': 'Pitchout'}


def znorm_v(pz, top, bot):
    if None in (pz, top, bot) or top <= bot: return None
    return (pz - bot) / (top - bot)

# ---- feature tuple extraction: (g, bh, ph, c, i, j, d, xw, re) ----
def feat_dict(p):
    g = grp(p.get('Pitch Type'))
    if g is None: return None
    d = p.get('Description')
    if d in EXCLUDE: return None
    bh, ph = p.get('Bats'), p.get('Throws')
    if bh not in HANDS or ph not in HANDS: return None
    px = sf(p.get('PlateX')); zn = znorm_v(sf(p.get('PlateZ')), sf(p.get('SzTop')), sf(p.get('SzBot')))
    if px is None or zn is None: return None
    c = p.get('Count')
    if not isinstance(c, str) or '-' not in c: return None
    try: b, s = c.split('-', 1); c = (int(b), int(s))
    except (TypeError, ValueError): return None
    if not (0 <= c[0] <= 3 and 0 <= c[1] <= 2): return None
    return (g, bh, ph, c, xb(px), zb(zn), d, sf(p.get('xwOBA')), sf(p.get('RunExp')))


def feat_cacherow(r):
    g = grp(r.pitch_type)
    if g is None: return None
    d = DMAP.get(r.description)
    if d is None or d in EXCLUDE: return None
    bh, ph = r.stand, r.p_throws
    if bh not in HANDS or ph not in HANDS: return None
    px = sf(r.plate_x); zn = znorm_v(sf(r.plate_z), sf(r.sz_top), sf(r.sz_bot))
    if px is None or zn is None: return None
    b, s = sf(r.balls), sf(r.strikes)
    if b is None or s is None or not (0 <= b <= 3 and 0 <= s <= 2): return None
    c = (int(b), int(s))
    re = sf(getattr(r, 'delta_pitcher_run_exp', None))
    if re is None:
        dre = sf(r.delta_run_exp); re = -dre if dre is not None else None
    xw = sf(r.estimated_woba_using_speedangle) if d == 'In Play' else None
    return (g, bh, ph, c, xb(px), zb(zn), d, xw, re)


def zeros(): return [[0.0] * NZ for _ in range(NX)]
def acc0(): return {k: zeros() for k in ('swn', 'swd', 'whn', 'fln', 'bipn', 'bipd')}


def accumulate(feat_iters):
    """Build surface accumulators from one or more iterables of feature tuples."""
    A = collections.defaultdict(acc0)
    AC = collections.defaultdict(lambda: {'swn': zeros(), 'swd': zeros(),
                                          'whn': zeros(), 'fln': zeros(), 'bipn': zeros(), 'bipd': zeros()})
    csn, csd = zeros(), zeros()
    n = 0
    for it in feat_iters:
        for t in it:
            if t is None: continue
            g, bh, ph, c, i, j, d, xw, re = t
            n += 1
            key = (g, bh, ph); a = A[key]; ac = AC[(key, c)]
            a['swd'][i][j] += 1; ac['swd'][i][j] += 1
            if d in SWING_DESC:
                a['swn'][i][j] += 1; ac['swn'][i][j] += 1
                if d == 'Swinging Strike': a['whn'][i][j] += 1; ac['whn'][i][j] += 1
                elif d == 'Foul': a['fln'][i][j] += 1; ac['fln'][i][j] += 1
                elif d == 'In Play' and xw is not None:
                    v = (xw - LG) / SCALE; a['bipn'][i][j] += v; a['bipd'][i][j] += 1
                    ac['bipn'][i][j] += v; ac['bipd'][i][j] += 1
            elif d in TAKE_DESC:
                csd[i][j] += 1
                if d == 'Called Strike': csn[i][j] += 1
    return A, AC, csn, csd, n


def k1d(bw):
    win = max(1, int(math.ceil(3 * bw))); return [(d, math.exp(-0.5 * (d / bw) ** 2)) for d in range(-win, win + 1)]
def gsum(a): return sum(sum(r) for r in a)
def smooth2d(num, den, prior, kp, kx, kz):
    tn = zeros(); td = zeros()
    for i in range(NX):
        ni, di_, tni, tdi = num[i], den[i], tn[i], td[i]
        for j in range(NZ):
            sn = sd = 0.0
            for dj, w in kz:
                jj = j + dj
                if 0 <= jj < NZ: sn += w * ni[jj]; sd += w * di_[jj]
            tni[j] = sn; tdi[j] = sd
    out = zeros(); pdict = not isinstance(prior, (int, float))
    for i in range(NX):
        oi = out[i]
        for j in range(NZ):
            sn = sd = 0.0
            for di2, w in kx:
                ii = i + di2
                if 0 <= ii < NX: sn += w * tn[ii][j]; sd += w * td[ii][j]
            pr = prior[i][j] if pdict else prior; s = sd + kp
            oi[j] = (sn + kp * pr) / s if s > 0 else pr
    return out


def build_surfaces(A, AC, csn, csd, RV, count_specific=False,
                   phx=4.5, phz=0.22, xwK=200, swK=20, kwh=8, kfl=8, kcs=10, kcoll=6, kcnt=40):
    kx = k1d(phx / BIN_X_IN); kz = k1d(phz / BIN_Z)
    PCS = smooth2d(csn, csd, gsum(csn) / max(gsum(csd), 1), kcs, kx, kz)
    WH = {}; FL = {}; XW = {}; SW = {}
    WHc = {}; FLc = {}; XWc = {}
    for key, a in A.items():
        swn = gsum(a['swn']); swd = gsum(a['swd']); bipd = gsum(a['bipd'])
        wh = smooth2d(a['whn'], a['swn'], gsum(a['whn']) / max(swn, 1), kwh, kx, kz)
        fl = smooth2d(a['fln'], a['swn'], gsum(a['fln']) / max(swn, 1), kfl, kx, kz)
        xw = smooth2d(a['bipn'], a['bipd'], gsum(a['bipn']) / max(bipd, 1), xwK, kx, kz)
        WH[key] = wh; FL[key] = fl; XW[key] = xw
        coll = smooth2d(a['swn'], a['swd'], swn / swd if swd else 0.0, kcoll, kx, kz)
        SW[key] = {c: smooth2d(AC[(key, c)]['swn'], AC[(key, c)]['swd'], coll, swK, kx, kz) for c in COUNTS}
        if count_specific:
            WHc[key] = {}; FLc[key] = {}; XWc[key] = {}
            for c in COUNTS:
                ac = AC[(key, c)]
                WHc[key][c] = smooth2d(ac['whn'], ac['swn'], wh, kcnt, kx, kz)   # shrink toward count-indep surface
                FLc[key][c] = smooth2d(ac['fln'], ac['swn'], fl, kcnt, kx, kz)
                XWc[key][c] = smooth2d(ac['bipn'], ac['bipd'], xw, xwK, kx, kz)
    return dict(PCS=PCS, WH=WH, FL=FL, XW=XW, SW=SW, WHc=WHc, FLc=FLc, XWc=XWc,
                RV=RV, count_specific=count_specific)


def score(pitches, S):
    RV = S['RV']; cs_ = S['count_specific']
    for p in pitches:
        t = p.get('_ft')
        if t is None: p['_v3'] = None; continue
        g, bh, ph, c, i, j, d, xw, re = t
        key = (g, bh, ph)
        psw = S['SW'][key][c][i][j]
        if cs_:
            pwh = S['WHc'][key][c][i][j]; pfl = S['FLc'][key][c][i][j]; vbip = S['XWc'][key][c][i][j]
        else:
            pwh = S['WH'][key][i][j]; pfl = S['FL'][key][i][j]; vbip = S['XW'][key][i][j]
        pbip = max(0.0, 1 - pwh - pfl); pcs = S['PCS'][i][j]
        sv = pwh * RV['whiff'].get(c, 0) + pfl * RV['foul'].get(c, 0) + pbip * vbip
        tv = pcs * RV['cs'].get(c, 0) + (1 - pcs) * RV['ball'].get(c, 0)
        p['_v3'] = psw * sv + (1 - psw) * tv


def pearson(xs, ys):
    n = len(xs)
    if n < 3: return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs); sy = sum((y - my) ** 2 for y in ys)
    if sx <= 0 or sy <= 0: return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sx * sy)


MIN_HALF, MIN_PRED = 150, 200
def build_eval_infra(P):
    dbp = collections.defaultdict(set)
    for p in P: dbp[(p['Pitcher'], p['Throws'])].add(p.get('Game Date'))
    half_of = {}
    for k, ds in dbp.items():
        for idx, dd in enumerate(sorted(ds)): half_of[(k, dd)] = idx % 2
    H = [[], []]; firsts = []
    for p in P:
        H[half_of[((p['Pitcher'], p['Throws']), p.get('Game Date'))]].append(p)
        if (p.get('Game Date') or '') < '2026-05-01': firsts.append(p)
    stf = collections.defaultdict(lambda: {'sw': 0, 'wh': 0, 'vs': 0.0, 'vn': 0})
    for p in P:
        k = (p['Pitcher'], p['Throws']); d = p.get('Description')
        if d in SWING_DESC:
            stf[k]['sw'] += 1
            if d == 'Swinging Strike': stf[k]['wh'] += 1
        if grp(p.get('Pitch Type')) == 'FF':
            v = sf(p.get('Velocity'))
            if v is not None: stf[k]['vs'] += v; stf[k]['vn'] += 1
    whiffrate = {k: s['wh'] / s['sw'] for k, s in stf.items() if s['sw'] >= 50}
    ffvelo = {k: s['vs'] / s['vn'] for k, s in stf.items() if s['vn'] >= 30}
    sec = collections.defaultdict(list)
    for p in P:
        if (p.get('Game Date') or '') >= '2026-05-01':
            d = p.get('Description'); xw = sf(p.get('xwOBA')); re = sf(p.get('RunExp'))
            v = (xw - LG) / SCALE if (d == 'In Play' and xw is not None) else (-re if re is not None else None)
            if v is not None: sec[(p['Pitcher'], p['Throws'])].append(v)
    sec_xrv = {k: sum(v) / len(v) for k, v in sec.items() if len(v) >= MIN_PRED}
    return H, firsts, whiffrate, ffvelo, sec_xrv

def aggf(pitches):
    out = collections.defaultdict(list)
    for p in pitches:
        v = p.get('_v3')
        if v is not None: out[(p['Pitcher'], p['Throws'])].append(v)
    return out

def evaluate(P, infra):
    H, firsts, whiffrate, ffvelo, sec_xrv = infra
    a0 = {k: (sum(v) / len(v), len(v)) for k, v in aggf(H[0]).items()}
    a1 = {k: (sum(v) / len(v), len(v)) for k, v in aggf(H[1]).items()}
    com = [k for k in a0 if k in a1 and a0[k][1] >= MIN_HALF and a1[k][1] >= MIN_HALF]
    rel = pearson([a0[k][0] for k in com], [a1[k][0] for k in com])
    full = {k: (sum(v) / len(v), len(v)) for k, v in aggf(P).items()}
    qual = {k: val for k, (val, n) in full.items() if n >= MIN_PRED}
    kw = [k for k in qual if k in whiffrate]; rw = pearson([qual[k] for k in kw], [whiffrate[k] for k in kw])
    kv = [k for k in qual if k in ffvelo]; rv = pearson([qual[k] for k in kv], [ffvelo[k] for k in kv])
    fa = {k: (sum(v) / len(v), len(v)) for k, v in aggf(firsts).items()}
    fq = {k: val for k, (val, n) in fa.items() if n >= MIN_PRED}
    kp = [k for k in fq if k in sec_xrv]; rp = pearson([fq[k] for k in kp], [sec_xrv[k] for k in kp])
    return rel, abs(rw or 0), abs(rv or 0), rp, len(com)


def cache_feats(year):
    df = pickle.load(open(os.path.join(ROOT, 'data', f'_statcast{year}_cache.pkl'), 'rb'))
    for r in df.itertuples(index=False):
        yield feat_cacherow(r)


def rv_from(P):
    acc = {k: collections.defaultdict(lambda: [0.0, 0]) for k in ('whiff', 'foul', 'cs', 'ball')}
    for p in P:
        t = p.get('_ft')
        if t is None: continue
        _, _, _, c, _, _, d, _, re = t
        if re is None: continue
        slot = {'Swinging Strike': 'whiff', 'Foul': 'foul', 'Called Strike': 'cs', 'Ball': 'ball'}.get(d)
        if slot: acc[slot][c][0] += -re; acc[slot][c][1] += 1
    return {k: {c: (s / n if n else 0.0) for c, (s, n) in dd.items()} for k, dd in acc.items()}


def main():
    print('loading 2026 eval ...', flush=True)
    P = [p for p in pickle.load(open(PKL, 'rb')) if p.get('_source') == 'MLB']
    for p in P: p['_ft'] = feat_dict(p)
    P = [p for p in P if p['_ft'] is not None]
    print(f'  {len(P)} scorable 2026 pitches', flush=True)
    RV = rv_from(P)
    infra = build_eval_infra(P)

    def run(baseline_years, count_specific=False, label=''):
        feats = [(p['_ft'] for p in P)]
        for y in baseline_years:
            feats.append(cache_feats(y))
        A, AC, csn, csd, n = accumulate(feats)
        S = build_surfaces(A, AC, csn, csd, RV, count_specific=count_specific)
        score(P, S)
        rel, rw, rv, rp, ncom = evaluate(P, infra)
        print(f"{label:32s} surf_n={n/1e6:5.2f}M  reliab={rel:.3f}  rWhf={rw:.3f}  rVel={rv:.3f}  pred={rp:.3f}  (n={ncom})", flush=True)
        return rel, rw, rv, rp

    print(f"\n{'variant':32s} {'baseline pitches':>10s}   objectives (rWhf/rVel = stuff-leak, keep LOW)\n")
    base = run([], False, 'BASELINE 2026-only')
    run([2024, 2025], False, 'surfaces +2024-25')
    ms = run([2021, 2022, 2023, 2024, 2025], False, 'surfaces +2021-25 (all)')
    print()
    run([], True, 'count-specific (2026-only)')
    csms = run([2021, 2022, 2023, 2024, 2025], True, 'count-specific +2021-25')
    print(f"\nbaseline reliab {base[0]:.3f} pred {base[3]:.3f} | rounding where objectives flat, stuff-leak must stay low")


if __name__ == '__main__':
    main()
