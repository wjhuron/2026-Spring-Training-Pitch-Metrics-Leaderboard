"""hwar_hitter_rate_desc.py — the VALUE-metric criteria for the hitter batting rate (2026-09-05).

hwar_hitter_rate_validation.py chose by prediction (xw edges xhb by .003 r). A WAR batting
component is a value metric: what his plate appearances were worth THIS season. Two criteria:
  1. descriptive fit: r of the park-adjusted rate with actual same-season wOBA, >= 300 PA,
     per season (the objective the shipped xwRC+ pulled-air term was chosen on)
  2. the calibrated shrink: the N0 where the LOSO slope of actual wOBA on the shrunk,
     park-adjusted rate is 1.0, so batting runs on the linear-weights scale need no fitted
     slope (the hitter selection gradient is mild: .89/.82/.82 by PA tercile)
Park pass-through at the within-batter xw value (.35).
Usage: python3 scripts/research/hitter/hwar_hitter_rate_desc.py
Output: console + data/_hwar_hitter_rate_desc.json
"""
import json, math, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research', 'era'))
import war_rate_validation as W
import hwar_hitter_rate_validation as HR

SEASONS = HR.SEASONS; T = HR.T; SCALE = HR.SCALE; CANDS = HR.CANDS
PASS = 0.35
N0_GRID = [0, 25, 50, 65, 75, 100, 125, 150, 200]


def main():
    out = {}
    P = {y: (HR.pa_savant(y) if y < 2026 else HR.pa_sheet(y))[0] for y in SEASONS}
    RPA = {y: (lambda ph: sum(v['r'] for v in ph.values()) / sum(v['bf'] for v in ph.values()))(T[str(y)]['pitchers']) for y in SEASONS}
    LG = {y: {c: float(P[y][c].mean()) for c in CANDS} for y in SEASONS}
    S = {}
    for y in SEASONS:
        g = P[y].groupby('bid').agg(woba=('woba', 'mean'), xw=('xw', 'mean'), xhb=('xhb', 'mean'), n=('woba', 'size'), pf=('pf', 'mean'))
        S[y] = g[g['n'] >= 300].copy()
        S[y]['park'] = PASS * (S[y]['pf'].fillna(1.0) - 1) * RPA[y] * SCALE[y]

    def rate(y, c, n0):
        g = S[y]; return (g[c] * g['n'] + n0 * LG[y][c]) / (g['n'] + n0) - (g['park'] if c != 'woba' else 0.0)

    print("1. DESCRIPTIVE FIT: r of the rate with actual same-season wOBA, >= 300 PA (woba is its own target: 1.0 by construction)")
    out['desc'] = {}
    for c in ('xw', 'xhb'):
        rs = [W.pear(rate(y, c, 0).values, S[y]['woba'].values) for y in SEASONS]; out['desc'][c] = rs
        print(f"  {c:4} " + " ".join(f"{r:.4f}" for r in rs) + f"  mean {np.mean(rs):.4f}")
    d = np.array(out['desc']['xhb']) - np.array(out['desc']['xw'])
    print(f"  xhb - xw: {d.mean():+.4f} ± {d.std(ddof=1) / math.sqrt(len(d)):.4f}  ({int((d > 0).sum())}/{len(d)})")

    print("\n2. CALIBRATED SHRINK: LOSO slope of actual wOBA on the shrunk park-adjusted rate (fit on the other five seasons, read on the held-out one)")
    out['calib'] = {}
    for c in ('xw', 'xhb'):
        print(f"  {c}:")
        rows = {}
        for n0 in N0_GRID:
            folds = []
            for hold in SEASONS:
                x, yv, w = [], [], []
                for y in SEASONS:
                    if y == hold:
                        continue
                    r = rate(y, c, n0); x += list(r - LG[y][c]); yv += list(S[y]['woba'] - LG[y]['woba']); w += list(S[y]['n'])
                b = W.wls_slope(np.array(x), np.array(yv), np.array(w, float))
                r = rate(hold, c, n0); folds.append(b)
            rows[n0] = folds
            print(f"    N0={n0:4d}  slope folds " + " ".join(f"{f:.3f}" for f in folds) + f"  mean {np.mean(folds):.3f}")
        out['calib'][c] = rows
        ns = sorted(rows); ms = [np.mean(rows[n]) for n in ns]
        for i in range(len(ns) - 1):
            if (ms[i] - 1) * (ms[i + 1] - 1) <= 0:
                n_star = ns[i] + (1 - ms[i]) / (ms[i + 1] - ms[i]) * (ns[i + 1] - ns[i])
                print(f"    slope = 1 crossing at N0 = {n_star:.0f} PA"); out['calib'][f'{c}_n0_star'] = float(n_star); break
    # spread implication for a 600-PA hitter at +.040 raw xhb
    for n0 in (0, 65, 100):
        sh = (0.040 * 600) / (600 + n0)
        print(f"  a +.040 xhb hitter over 600 PA: shrunk +{sh:.4f} -> {sh * 600 / 1.24:+.1f} batting runs at N0 {n0}")
    json.dump(out, open(os.path.join(ROOT, 'data', '_hwar_hitter_rate_desc.json'), 'w'), indent=1, default=float)
    print("wrote data/_hwar_hitter_rate_desc.json")


if __name__ == '__main__':
    main()
