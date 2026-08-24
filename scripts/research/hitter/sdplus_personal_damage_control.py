"""sdplus_personal_damage_control.py — contamination adjudication for the
personalized-SD+ battery.

The main battery showed pred deltas of +0.10-0.13 r with partials ~+0.35
given BASE. That is too large to be decision skill (base SD+ itself only
predicts 0.14-0.21): the centered shape still leaks overall damage
through the swing/take imbalance weights (damage residuals in swing-heavy
zones enter positively, take-heavy zones negatively). The registered
control (partial given BASE) tests duplication of BASE, not smuggled
outcome quality.

This adjudicator recomputes full-season scores and adds the missing
control: partial r of the personalized score with next-season wOBA given
BOTH the base score AND the hitter's overall swing-damage residual
(mean rv on swings minus the league swing cell value — the quantity the
centering was supposed to exclude). If the personalized gain collapses
under that control, the variant is outcome contamination and is REJECTED.

Usage: PYTHONHASHSEED=0 python3 scripts/research/hitter/sdplus_personal_damage_control.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import pipeline.sdplus as sd
from sdplus_personal_damage import (atoms, league_tables, hitter_aggs,
                                    score, load_season, guts, pearson,
                                    PAIRS, FULL_MIN_DEC, K_GRID)


def resid_corr(y, x, controls):
    """r(y, x | controls) via OLS residualization (pure python)."""
    n = len(y)
    X = [[1.0] + [c[i] for c in controls] for i in range(n)]
    k = len(X[0])

    def ols_resid(target):
        XtX = [[sum(X[i][a] * X[i][b] for i in range(n))
                for b in range(k)] for a in range(k)]
        XtY = [sum(X[i][a] * target[i] for i in range(n)) for a in range(k)]
        for col in range(k):
            piv = max(range(col, k), key=lambda r2: abs(XtX[r2][col]))
            if abs(XtX[piv][col]) < 1e-12:
                return None
            XtX[col], XtX[piv] = XtX[piv], XtX[col]
            XtY[col], XtY[piv] = XtY[piv], XtY[col]
            d = XtX[col][col]
            XtX[col] = [v / d for v in XtX[col]]
            XtY[col] /= d
            for r2 in range(k):
                if r2 != col and XtX[r2][col]:
                    f2 = XtX[r2][col]
                    XtX[r2] = [a - f2 * b for a, b in zip(XtX[r2], XtX[col])]
                    XtY[r2] -= f2 * XtY[col]
        beta = XtY
        return [target[i] - sum(beta[j] * X[i][j] for j in range(k))
                for i in range(n)]

    ry, rx = ols_resid(y), ols_resid(x)
    if ry is None or rx is None:
        return None
    return pearson(rx, ry)


def main():
    seasons = sorted({y for p in PAIRS for y in p})
    scores, woba, damage = {}, {}, {}
    for y in seasons:
        lg, sc = guts(y)
        rv_fn = sd.make_rv_xrv(lg, sc)
        rows, wt = atoms(load_season(y), rv_fn)
        woba[y] = wt
        tbl, zw = league_tables(rows)
        ag = hitter_aggs(rows, tbl)
        scores[y] = {K: score(ag, zw, K) for K in [None] + K_GRID}
        dmg = {}
        for h, zones in ag.items():
            rs = sum(z[3] for z in zones.values())
            ns = sum(z[4] for z in zones.values())
            if ns >= 65:
                dmg[h] = rs / ns
        damage[y] = dmg
        del rows
        print(f'{y}: scored {len(ag)} hitters, damage for {len(dmg)}',
              flush=True)

    print('\npair        K     r(pers)  |BASE    |BASE+damage')
    for y0, y1 in PAIRS:
        base = scores[y0][None]
        nxt = woba[y1]
        hs = [h for h in base if base[h][1] >= FULL_MIN_DEC
              and h in nxt and h in damage[y0]]
        yv = [nxt[h] for h in hs]
        bv = [base[h][0] for h in hs]
        dv = [damage[y0][h] for h in hs]
        print(f'  {y0}->{y1} n={len(hs)}  '
              f'r(BASE)={pearson(bv, yv):+.3f}  '
              f'r(damage)={pearson(dv, yv):+.3f}')
        for K in K_GRID:
            pv = [scores[y0][K][h][0] for h in hs]
            print(f'    K{K:<4} r={pearson(pv, yv):+.3f}  '
                  f'|base={resid_corr(yv, pv, [bv]):+.3f}  '
                  f'|base+dmg={resid_corr(yv, pv, [bv, dv]):+.3f}')


if __name__ == '__main__':
    main()
