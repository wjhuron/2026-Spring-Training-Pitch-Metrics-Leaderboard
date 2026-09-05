"""war_velo_bias_checks2.py — is the pitcher-level velocity residual selection, and what does an
out-of-sample criterion say (2026-09-05)?

war_velo_bias_checks.py: at the BIP level contact off fast pitches runs BELOW xwOBA, but at the
pitcher level fast arms allow MORE actual runs than their deserved rate in the same season
(slope +.23 runs/9 per 10 mph), and the correction widens that. Same-season residuals are bent by
selection on outcomes (CLAUDE.md), and survival is velocity-dependent: a slow arm is kept only
when his results are good, a hard thrower is kept anyway. So:
  1. same-season residual by velocity tercile, split by workload (30-120 IP vs 120+): a selection
     signature concentrates in the low-workload group
  2. rest-of-season criterion: second-half actual RA9 minus the first-half deserved rate, by
     velocity tercile, under k = 0 / .5 / 1 / 2 (h2 luck is fresh; h1 results are in the rate)
  3. next-season criterion: RA9(y+1) minus the deserved rate(y), 60 IP both sides (aging confounds
     this one: soft tossers are old)
  4. within-pitcher band deltas: residual centered on the pitcher-season mean, then by band
     (the pitch, not the pitcher)
Usage: python3 scripts/research/era/war_velo_bias_checks2.py
Output: console + data/_war_velo_bias_checks2.json
"""
import gc, json, math, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from pipeline.eraplus import N0_XW, POOL_MIN_OUTS, WAR_PARK_PASS, DH_B
import war_rate_validation as W
import war_velo_bias_followup as VF

SEASONS = W.SEASONS; BANDS = VF.BANDS; KS = [0.0, 0.5, 1.0, 2.0]


def rate_fn(rows, lg):
    """production-form deserved rate over a frame: anchor(pool ERA) + DH_B z(shrunk x) + gap - park; returns dict pid -> rate."""
    pool = [q for q in rows if q['outs'] >= POOL_MIN_OUTS]
    sh = np.array([q['sh'] for q in pool]); mu, sd = sh.mean(), sh.std(); anchor = np.mean([q['era'] for q in pool])
    return {q['pid']: anchor + DH_B * (q['sh'] - mu) / sd + (lg['ra9'] - lg['era']) - WAR_PARK_PASS * (q['exp'] - 1) * lg['ra9'] for q in rows}


def frames(Py, t, lg, k, add, sc):
    mask = np.ones(len(Py), bool) if sc == 'full' else (Py['h1'].values if sc == 'h1' else ~Py['h1'].values)
    g = pd.DataFrame(dict(pid=Py['pid'].values[mask], x=(Py['x'].values + k * add)[mask])).groupby('pid')['x'].agg(['mean', 'size'])
    lgx = float((g['mean'] * g['size']).sum() / g['size'].sum())
    rows = []
    for pid, r in t.items():
        if pid not in g.index:
            continue
        if sc == 'full':
            o, rr, er = r['outs'], r['r'], r['er']
        else:
            h = r[sc]; o = h.get('outs') or 0; rr = h.get('r', 0); er = h.get('er', 0)
        if o <= 0:
            continue
        den = g.loc[pid, 'size']; rows.append(dict(pid=pid, outs=o, ra9=rr * 27 / o, era=er * 27 / o, exp=r['exp'],
                                                   sh=(g.loc[pid, 'mean'] * den + N0_XW * lgx) / (den + N0_XW)))
    return rows


def tercile_line(v, res, o):
    q1, q2 = np.percentile(v, [33.3, 66.7])
    tm = [float(np.average(res[m], weights=o[m])) for m in (v < q1, (v >= q1) & (v < q2), v >= q2)]
    return tm, W.wls_slope(v, res, o) * 10


def main():
    out = {}
    TAB = {y: W.season_table(y) for y in SEASONS}; LG = {y: W.league(TAB[y]) for y in SEASONS}
    P = {}
    for y in SEASONS:
        P[y] = VF.pa_savant(y) if y < 2026 else VF.pa_sheet(y); gc.collect()
    LOSO, ADD, VEL = {}, {}, {}
    for y in SEASONS:
        others = pd.concat([P[s] for s in SEASONS if s != y and s < 2026])
        LOSO[y] = {c: v[0] for c, v in VF.cell_table(others, 'velo').items()}
        ADD[y] = np.array([LOSO[y].get(c, 0.0) if c is not None else 0.0 for c in P[y]['velo'].values])
        VEL[y] = pd.DataFrame(dict(pid=P[y]['pid'].values, rs=P[y]['rs'].values)).groupby('pid')['rs'].mean().dropna()

    print("1. SAME-SEASON RESIDUAL (actual RA9 - deserved rate, k=0) BY VELOCITY TERCILE, split by workload")
    res_lo, res_hi = [], []
    for y in SEASONS:
        rows = frames(P[y], TAB[y], LG[y], 0.0, ADD[y], 'full'); R = rate_fn(rows, LG[y])
        for lab, lo, hi, acc in (('30-120 IP', 90, 360, res_lo), ('120+ IP', 360, 10 ** 9, res_hi)):
            sub = [q for q in rows if lo <= q['outs'] < hi and q['pid'] in VEL[y].index]
            v = np.array([VEL[y][q['pid']] for q in sub]); r = np.array([q['ra9'] - R[q['pid']] for q in sub]); o = np.array([q['outs'] for q in sub], float)
            tm, sl = tercile_line(v, r, o); acc.append((tm, sl, len(sub)))
    for lab, acc in (('30-120 IP', res_lo), ('120+ IP', res_hi)):
        tm = np.mean([a[0] for a in acc], axis=0); sl = np.array([a[1] for a in acc])
        print(f"  {lab:10} n/season {int(np.mean([a[2] for a in acc]))}: slow/mid/fast {tm[0]:+.3f}/{tm[1]:+.3f}/{tm[2]:+.3f}   slope {sl.mean():+.3f} ± {sl.std(ddof=1) / math.sqrt(len(sl)):.3f} per 10 mph   by season " + " ".join(f"{s:+.2f}" for s in sl))
    out['same_season_by_ip'] = {'lo': [a[:2] for a in res_lo], 'hi': [a[:2] for a in res_hi]}

    print("\n2. REST-OF-SEASON CRITERION: RA9(h2) - deserved rate(h1), both halves >= 30 IP, by velocity tercile, under k")
    ros = {k: [] for k in KS}
    for y in SEASONS:
        for k in KS:
            r1 = frames(P[y], TAB[y], LG[y], k, ADD[y], 'h1'); R1 = rate_fn(r1, LG[y])
            r2 = {q['pid']: q for q in frames(P[y], TAB[y], LG[y], k, ADD[y], 'h2')}
            sub = [q for q in r1 if q['outs'] >= 90 and q['pid'] in r2 and r2[q['pid']]['outs'] >= 90 and q['pid'] in VEL[y].index]
            v = np.array([VEL[y][q['pid']] for q in sub]); r = np.array([r2[q['pid']]['ra9'] - R1[q['pid']] for q in sub]); o = np.array([r2[q['pid']]['outs'] for q in sub], float)
            ros[k].append(tercile_line(v, r, o) + (len(sub),))
    for k in KS:
        tm = np.mean([a[0] for a in ros[k]], axis=0); sl = np.array([a[1] for a in ros[k]])
        print(f"  k={k:.1f}: slow/mid/fast {tm[0]:+.3f}/{tm[1]:+.3f}/{tm[2]:+.3f}   slope {sl.mean():+.3f} ± {sl.std(ddof=1) / math.sqrt(len(sl)):.3f} per 10 mph   by season " + " ".join(f"{s:+.2f}" for s in sl) + f"   n {int(np.mean([a[2] for a in ros[k]]))}")
    s0 = np.array([a[1] for a in ros[0.0]]); s1 = np.array([a[1] for a in ros[1.0]]); ks = -s0 / (s1 - s0)
    print(f"  k* (zero slope) per season: " + " ".join(f"{y} {v:+.2f}" for y, v in zip(SEASONS, ks)) + f"   mean {ks.mean():+.2f} ± {ks.std(ddof=1) / math.sqrt(len(ks)):.2f}")
    out['ros'] = {str(k): [a[:2] for a in ros[k]] for k in KS}; out['ros_kstar'] = ks.tolist()

    print("\n3. NEXT-SEASON CRITERION: RA9(y+1) - deserved rate(y), 60 IP both sides, by velocity tercile, under k (aging confounds this)")
    nxt = {k: [] for k in KS}
    for y in SEASONS[:-1]:
        for k in KS:
            r1 = frames(P[y], TAB[y], LG[y], k, ADD[y], 'full'); R1 = rate_fn(r1, LG[y])
            t2 = TAB[y + 1]
            sub = [q for q in r1 if q['outs'] >= 180 and q['pid'] in t2 and t2[q['pid']]['outs'] >= 180 and q['pid'] in VEL[y].index]
            v = np.array([VEL[y][q['pid']] for q in sub]); r = np.array([t2[q['pid']]['r'] * 27 / t2[q['pid']]['outs'] - R1[q['pid']] for q in sub])
            o = np.array([t2[q['pid']]['outs'] for q in sub], float)
            nxt[k].append(tercile_line(v, r, o) + (len(sub),))
    for k in KS:
        tm = np.mean([a[0] for a in nxt[k]], axis=0); sl = np.array([a[1] for a in nxt[k]])
        print(f"  k={k:.1f}: slow/mid/fast {tm[0]:+.3f}/{tm[1]:+.3f}/{tm[2]:+.3f}   slope {sl.mean():+.3f} ± {sl.std(ddof=1) / math.sqrt(len(sl)):.3f} per 10 mph   by season " + " ".join(f"{s:+.2f}" for s in sl) + f"   n {int(np.mean([a[2] for a in nxt[k]]))}")
    out['nxt'] = {str(k): [a[:2] for a in nxt[k]] for k in KS}

    print("\n4. WITHIN-PITCHER BAND DELTAS: residual centered on the pitcher-season mean (pooled 2021-2025)")
    ALL = pd.concat([P[y].assign(ys=str(y)) for y in SEASONS if y < 2026])
    d = ALL[ALL['bip'] & ALL['velo'].notna()].copy(); d['r'] = d['w'] - d['x']
    d['key'] = d['ys'] + d['pid']; d['rc'] = d['r'] - d.groupby('key')['r'].transform('mean')
    raw = d.groupby('velo')['r'].mean() - d['r'].mean(); within = d.groupby('velo')['rc'].mean() - d['rc'].mean()
    print("    band      raw      within-pitcher")
    for b in BANDS:
        print(f"    {b:9} {raw[b]:+.4f}   {within[b]:+.4f}")
    out['within_pitcher'] = dict(raw=raw.to_dict(), within=within.to_dict())
    json.dump(out, open(os.path.join(ROOT, 'data', '_war_velo_bias_checks2.json'), 'w'), indent=1, default=float)
    print("wrote data/_war_velo_bias_checks2.json")


if __name__ == '__main__':
    main()
