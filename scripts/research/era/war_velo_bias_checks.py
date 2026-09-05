"""war_velo_bias_checks.py — three checks on the velocity bias before it can be a decision (2026-09-05).

war_velo_bias_followup.py: the strength curve is monotone in k, which makes rel/nxt gameable
(velocity is a stable pitcher trait, so more of it always reads more reliable). The constant
must be pinned by a CALIBRATION criterion instead. Checks:
  1. weights: the band deltas under three linear-weight sets (battery .89/1.27/1.61/2.10,
     FanGraphs 2021, the pipeline fallback). A bias that moves with the HR weight is a weight
     artifact, not an xwOBA property.
  2. home runs out: the band deltas on BIP that are not home runs.
  3. pitcher-level calibration: for >= 30 IP arms, actual RA9 minus the park-adjusted deserved
     rate, by tercile of the pitcher's BIP velocity, under ship and under k = 0.5 / 1 / 2 of the
     LOSO band table; the IP-weighted slope of that residual on velocity, and the k that zeros
     it (k*). Also the LOSO DH_B refit on the corrected z at k = 1.
Usage: python3 scripts/research/era/war_velo_bias_checks.py
Output: console + data/_war_velo_bias_checks.json
"""
import gc, json, math, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from pipeline.utils import BB_EVENTS, HBP_EVENTS
from pipeline.eraplus import N0_XW, POOL_MIN_OUTS, WAR_PARK_PASS, DH_B
import war_rate_validation as W
import war_velo_bias_followup as VF

SEASONS = W.SEASONS; BANDS = VF.BANDS
WSETS = {'battery': (.69, .72, .89, 1.27, 1.61, 2.10), 'fg2021': (.692, .722, .883, 1.244, 1.569, 2.004), 'fallback': (.692, .723, .884, 1.256, 1.591, 2.048)}
KS = [0.0, 0.5, 1.0, 2.0]


def main():
    out = {}
    TAB = {y: W.season_table(y) for y in SEASONS}; LG = {y: W.league(TAB[y]) for y in SEASONS}
    P = {}
    for y in SEASONS:
        P[y] = VF.pa_savant(y) if y < 2026 else VF.pa_sheet(y)
        gc.collect()
    # the event token is needed for the weight sets; rebuild w per set from the battery w (invertible: w in {0, BB, HBP, 1B, 2B, 3B, HR})
    ALL = pd.concat([P[y] for y in SEASONS if y < 2026])
    print("1. BAND DELTAS UNDER THREE WEIGHT SETS (pooled 2021-2025, BIP, recentered)")
    inv = {v: k for k, v in zip(('bb', 'hbp', '1b', '2b', '3b', 'hr'), WSETS['battery'])}
    bip = ALL[ALL['bip'] & ALL['velo'].notna()]
    tok = bip['w'].round(3).map({round(v, 3): k for k, v in zip(('bb', 'hbp', '1b', '2b', '3b', 'hr'), WSETS['battery'])}).fillna('out')
    out['weights'] = {}
    print("    set       " + "  ".join(f"{b:>8}" for b in BANDS))
    for nm, ws in WSETS.items():
        wmap = dict(zip(('bb', 'hbp', '1b', '2b', '3b', 'hr'), ws)); wmap['out'] = 0.0
        r = tok.map(wmap).values - bip['x'].values; base = r.mean()
        dl = {b: float(r[bip['velo'].values == b].mean() - base) for b in BANDS}
        out['weights'][nm] = dl
        print(f"    {nm:9} " + "  ".join(f"{dl[b]:+.4f}" for b in BANDS))
    print("2. BAND DELTAS WITH HOME RUNS REMOVED (both sides), and the HR share of BIP by band")
    nohr = bip[tok.values != 'hr']; r = (nohr['w'] - nohr['x']).values; base = r.mean()
    dl = {b: float(r[nohr['velo'].values == b].mean() - base) for b in BANDS}
    hs = {b: float((tok.values[bip['velo'].values == b] == 'hr').mean()) for b in BANDS}
    out['no_hr'] = dl; out['hr_share'] = hs
    print("    no-HR     " + "  ".join(f"{dl[b]:+.4f}" for b in BANDS))
    print("    HR share  " + "  ".join(f"{hs[b]:8.4f}" for b in BANDS))

    # LOSO band tables
    LOSO = {}
    for y in SEASONS:
        others = pd.concat([P[s] for s in SEASONS if s != y and s < 2026])
        LOSO[y] = {c: v[0] for c, v in VF.cell_table(others, 'velo').items()}

    print("\n3. PITCHER-LEVEL CALIBRATION: actual RA9 - park-adjusted deserved rate, >= 30 IP, by BIP-velocity tercile (IP-weighted), and its slope per mph")
    out['calib'] = {}
    slopes = {k: [] for k in KS}; terc = {k: [] for k in KS}; dhb = []
    for y in SEASONS:
        Py = P[y]; t = TAB[y]; lg = LG[y]
        base_x = Py['x'].values
        add = np.array([LOSO[y].get(c, 0.0) if c is not None else 0.0 for c in Py['velo'].values])
        vel = pd.DataFrame(dict(pid=Py['pid'].values, rs=Py['rs'].values)).groupby('pid')['rs'].mean()
        for k in KS:
            g = pd.DataFrame(dict(pid=Py['pid'].values, x=base_x + k * add)).groupby('pid')['x'].agg(['mean', 'size'])
            rows = []
            for pid, r in t.items():
                if pid not in g.index or r['outs'] < POOL_MIN_OUTS or pid not in vel.index or np.isnan(vel[pid]):
                    continue
                den = g.loc[pid, 'size']; sh = (g.loc[pid, 'mean'] * den + N0_XW * lg['xw']) / (den + N0_XW)
                rows.append(dict(pid=pid, sh=sh, outs=r['outs'], ra9=r['r'] * 27 / r['outs'], era=r['er'] * 27 / r['outs'], exp=r['exp'], v=vel[pid]))
            sh = np.array([q['sh'] for q in rows]); mu, sd = sh.mean(), sh.std(); anchor = np.mean([q['era'] for q in rows])
            z = (sh - mu) / sd
            rate = anchor + DH_B * z + (lg['ra9'] - lg['era']) - WAR_PARK_PASS * (np.array([q['exp'] for q in rows]) - 1) * lg['ra9']
            res = np.array([q['ra9'] for q in rows]) - rate; o = np.array([q['outs'] for q in rows], float); v = np.array([q['v'] for q in rows])
            slope = W.wls_slope(v, res, o); slopes[k].append(slope)
            q1, q2 = np.percentile(v, [33.3, 66.7]); tm = [float(np.average(res[m], weights=o[m])) for m in (v < q1, (v >= q1) & (v < q2), v >= q2)]
            terc[k].append(tm)
            if k == 1.0:
                # LOSO DH_B refit on the corrected z: slope of ERA on z, this season's pool contributes to the other folds; report the pooled fit here
                dhb.append(float(np.polyfit(z, np.array([q['era'] for q in rows]) - anchor, 1)[0]))
        print(f"  {y}: " + "  ".join(f"k={k:.1f} terciles {terc[k][-1][0]:+.3f}/{terc[k][-1][1]:+.3f}/{terc[k][-1][2]:+.3f} slope {slopes[k][-1] * 10:+.3f}/10mph" for k in KS))
    print("  mean over seasons:")
    for k in KS:
        tm = np.mean(terc[k], axis=0); s = np.array(slopes[k])
        print(f"    k={k:.1f}: terciles slow/mid/fast {tm[0]:+.3f}/{tm[1]:+.3f}/{tm[2]:+.3f} runs/9   slope {s.mean() * 10:+.3f} ± {s.std(ddof=1) / math.sqrt(len(s)) * 10:.3f} per 10 mph  "
              f"(negative = fast arms beat their deserved rate)")
    s0, s1 = np.array(slopes[0.0]), np.array(slopes[1.0]); kstar = -s0 / (s1 - s0)
    print(f"  k* that zeros the residual slope, per season: " + " ".join(f"{y} {v:.2f}" for y, v in zip(SEASONS, kstar)) + f"   mean {kstar.mean():.2f} ± {kstar.std(ddof=1) / math.sqrt(len(kstar)):.2f}")
    print(f"  ERA-on-z slope per season on the corrected z (k=1): {np.round(dhb, 3).tolist()} mean {np.mean(dhb):.3f} (shipped DH_B {DH_B}, same fit on ship z gives the fold values in war_error_bar.py)")
    out['calib'] = dict(slopes={str(k): slopes[k] for k in KS}, terciles={str(k): terc[k] for k in KS}, kstar=kstar.tolist(), dhb_corrected=dhb)
    json.dump(out, open(os.path.join(ROOT, 'data', '_war_velo_bias_checks.json'), 'w'), indent=1, default=float)
    print("wrote data/_war_velo_bias_checks.json")


if __name__ == '__main__':
    main()
