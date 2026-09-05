"""war_park_pass_within.py — WAR_PARK_PASS re-measured WITHIN pitcher, in hdERA's own currency (2026-09-05).

The shipped .91 is the across-pitcher slope of hdR9 on club park exposure (war_rate_validation),
a design where club quality rides on the park axis (Seattle's staff is good and pitches in a
pitcher park). Here each pitcher is his own control, home vs road in the same season:
    d_hd   = (xwOBA against at home - on the road) x DH_B / sd(pool)      runs per 9, hdERA's slope
    d_park = (PF_home - PA-weighted PF of the road venues) / 100 x lgRA9    runs per 9, the full factor
    pass   = PA-weighted slope of d_hd on d_park (weight = harmonic mean of home and road PA),
             >= 100 PA each side, per season and LOSO; 1.0 = the full published runs factor
Home park = the venue with the most of his PA, kept when it holds >= 40% (single-club seasons).
Also the actual wOBA-against version at the linear-weights conversion, for context.
Usage: python3 scripts/research/era/war_park_pass_within.py
Output: console + data/_war_park_pass_within.json
"""
import json, math, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research', 'hitter'))
from pipeline.eraplus import DH_B
import war_rate_validation as W
import hwar_park_pass_within as HP

SEASONS = W.SEASONS; T = W.T; PF = W.PF; SCALE = HP.SCALE
MIN_SIDE, HOME_SHARE = 100, 0.40


def main():
    out = {}
    P = {y: HP.table(y) for y in SEASONS}
    rows = {}
    for y in SEASONS:
        t = W.season_table(y); lg = W.league(t); mu, sd, anchor = W.pool_stats(t, lg)
        ph = T[str(y)]['pitchers']; rpa = sum(v['r'] for v in ph.values()) / sum(v['bf'] for v in ph.values()); pa9 = lg['ra9'] / rpa
        rec = []
        for pid, g in P[y].groupby('pid'):
            vc = g['venue'].value_counts()
            if vc.iloc[0] / len(g) < HOME_SHARE:
                continue
            home = vc.index[0]; hm = g['venue'] == home
            if hm.sum() < MIN_SIDE or (~hm).sum() < MIN_SIDE:
                continue
            dpf = PF[str(y)].get(home, 100.0) / 100.0 - float(g.loc[~hm, 'pf'].mean())
            d_xw = float(g.loc[hm, 'xw'].mean() - g.loc[~hm, 'xw'].mean()); d_w = float(g.loc[hm, 'woba'].mean() - g.loc[~hm, 'woba'].mean())
            rec.append(dict(d_park=dpf * lg['ra9'], d_hd=d_xw * DH_B / sd, d_lw=d_w * pa9 / SCALE[y], w=2 * hm.sum() * (~hm).sum() / len(g)))
        rows[y] = pd.DataFrame(rec); out[f'consts_{y}'] = dict(sd_pool=sd, lg_ra9=lg['ra9'], pa9=pa9, runs_per_xw_hd=DH_B / sd, runs_per_xw_lw=pa9 / SCALE[y])
    print("WITHIN-PITCHER PASS-THROUGH, runs per 9: hdERA currency (xwOBA against x DH_B / sd) and actual wOBA against at linear weights")
    print("  season    n   sd(d_park)   hdERA   actual-lw   [runs/9 per xwOBA point: hd / lw]")
    res = {'hd': [], 'lw': []}
    for y in SEASONS:
        d = rows[y]; s_hd = W.wls_slope(d['d_park'].values, d['d_hd'].values, d['w'].values); s_lw = W.wls_slope(d['d_park'].values, d['d_lw'].values, d['w'].values)
        res['hd'].append(s_hd); res['lw'].append(s_lw); c = out[f'consts_{y}']
        print(f"  {y}    {len(d):4d}   {d['d_park'].std():.3f}       {s_hd:.3f}   {s_lw:.3f}      [{c['runs_per_xw_hd']:.1f} / {c['runs_per_xw_lw']:.1f}]")
    loso = {'hd': [], 'lw': []}
    for hold in SEASONS:
        d = pd.concat([rows[y] for y in SEASONS if y != hold])
        loso['hd'].append(W.wls_slope(d['d_park'].values, d['d_hd'].values, d['w'].values)); loso['lw'].append(W.wls_slope(d['d_park'].values, d['d_lw'].values, d['w'].values))
    for k, lab in (('hd', 'hdERA currency'), ('lw', 'actual, linear weights')):
        print(f"  {lab:24} per-season mean {np.mean(res[k]):.3f} ± {np.std(res[k], ddof=1) / math.sqrt(len(res[k])):.3f}   LOSO folds " + " ".join(f"{v:.3f}" for v in loso[k]) + f"  mean {np.mean(loso[k]):.3f}")
    print(f"  shipped WAR_PARK_PASS .91 (across-pitcher club design, war_rate_validation.py)")
    out['per_season'] = res; out['loso'] = loso
    json.dump(out, open(os.path.join(ROOT, 'data', '_war_park_pass_within.json'), 'w'), indent=1, default=float)
    print("wrote data/_war_park_pass_within.json")


if __name__ == '__main__':
    main()
