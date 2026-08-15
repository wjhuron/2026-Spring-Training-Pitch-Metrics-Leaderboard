"""Test the expected-movement RESIDUAL (IVBOE/HBOE), not just model fit.

R^2 is a gameable objective here: conditioning on more of the movement's own
causes always raises it and always shrinks the residual. What actually decides
whether an OE metric is worth shipping:

  C1 DISTINCTNESS  corr(OE, raw movement) across pitchers. If it is ~1 the
                   metric is a relabelled copy of the column next to it.
  C2 RELIABILITY   split-half (random halves of a pitcher-season's pitches),
                   Spearman-Brown corrected, at the RENDERED unit.
  C3 PERSISTENCE   year-over-year corr for the same pitcher x pitch type.
  C4 VALUE         does OE carry run-value signal beyond the raw movement?

Models are fit on 2021-2024 and scored on 2025 so no pitcher helps define his
own expectation.
"""
import os, math, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from xmove_audit import load, add_axis_trig, FEATURE_SETS, PITCH_TYPES, MIN_N

RNG = np.random.default_rng(17)
UNIT_MIN = 50     # pitches per (pitcher, season, pitch type) -- the card gate


def fit_reference(train, feats):
    """Per (pitch type, hand) OLS on the training seasons. Returns coef dict."""
    models = {}
    for (pt, thr), g in train.groupby(['pt', 'thr']):
        if len(g) < MIN_N:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[f].values for f in feats])
        b = {}
        for t in ('ivb', 'hb_s'):
            beta, *_ = np.linalg.lstsq(X, g[t].values, rcond=None)
            b[t] = beta
        models[(pt, thr)] = b
    return models


def score(test, models, feats):
    """Attach xIVB/xHB and residuals to the test frame."""
    out = test.copy()
    out['x_ivb'] = np.nan
    out['x_hb'] = np.nan
    for (pt, thr), g in out.groupby(['pt', 'thr']):
        m = models.get((pt, thr))
        if m is None:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[f].values for f in feats])
        out.loc[g.index, 'x_ivb'] = X @ m['ivb']
        out.loc[g.index, 'x_hb'] = X @ m['hb_s']
    out['ivb_oe'] = out.ivb - out.x_ivb
    out['hb_oe'] = out.hb_s - out.x_hb
    return out.dropna(subset=['ivb_oe', 'hb_oe'])


def unit_agg(scored):
    scored = scored.copy()
    scored['half'] = RNG.integers(0, 2, len(scored))
    g = scored.groupby(['pitcher', 'thr', 'pt', 'season'])
    agg = g.agg(n=('ivb', 'size'), ivb=('ivb', 'mean'), hb=('hb_s', 'mean'),
                ivb_oe=('ivb_oe', 'mean'), hb_oe=('hb_oe', 'mean'),
                x_ivb=('x_ivb', 'mean'), x_hb=('x_hb', 'mean'),
                spin=('spin', 'mean'), dev=('dev', 'mean')).reset_index()
    h = scored.groupby(['pitcher', 'thr', 'pt', 'season', 'half']).agg(
        ivb_oe=('ivb_oe', 'mean'), hb_oe=('hb_oe', 'mean'),
        ivb=('ivb', 'mean'), hb=('hb_s', 'mean'), n=('ivb', 'size')).reset_index()
    hp = h.pivot_table(index=['pitcher', 'thr', 'pt', 'season'], columns='half',
                       values=['ivb_oe', 'hb_oe', 'ivb', 'hb', 'n'])
    hp.columns = [f'{a}_{b}' for a, b in hp.columns]
    return agg.merge(hp.reset_index(), on=['pitcher', 'thr', 'pt', 'season'])


def sb(r):
    """Spearman-Brown: half-sample correlation -> full-sample reliability."""
    return 2 * r / (1 + r) if r is not None and not np.isnan(r) and r > -1 else np.nan


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def report(name, u, u_prev=None):
    print(f'\n--- {name} ---')
    print(f"{'pt':>4} {'units':>6} {'corr(ivbOE,IVB)':>16} {'corr(hbOE,HB)':>14} "
          f"{'rel ivbOE':>10} {'rel hbOE':>9} {'YoY ivbOE':>10} {'YoY hbOE':>9}")
    for pt in PITCH_TYPES:
        g = u[(u.pt == pt) & (u.n >= UNIT_MIN)]
        if len(g) < 30:
            continue
        r_i = corr(g.ivb_oe.values, g.ivb.values)
        r_h = corr(g.hb_oe.values, g.hb.values)
        gh = g[(g.n_0 >= UNIT_MIN / 2) & (g.n_1 >= UNIT_MIN / 2)]
        rel_i = sb(corr(gh.ivb_oe_0.values, gh.ivb_oe_1.values))
        rel_h = sb(corr(gh.hb_oe_0.values, gh.hb_oe_1.values))
        yi = yh = np.nan
        if u_prev is not None:
            p = u_prev[(u_prev.pt == pt) & (u_prev.n >= UNIT_MIN)]
            mg = g.merge(p, on=['pitcher', 'thr', 'pt'], suffixes=('', '_p'))
            yi = corr(mg.ivb_oe.values, mg.ivb_oe_p.values)
            yh = corr(mg.hb_oe.values, mg.hb_oe_p.values)
        print(f'{pt:>4} {len(g):>6} {r_i:>16.3f} {r_h:>14.3f} {rel_i:>10.3f} '
              f'{rel_h:>9.3f} {yi:>10.3f} {yh:>9.3f}')
    g = u[u.n >= UNIT_MIN]
    print(f"  ALL  {len(g):>5} {corr(g.ivb_oe.values, g.ivb.values):>16.3f} "
          f"{corr(g.hb_oe.values, g.hb.values):>14.3f}")


if __name__ == '__main__':
    df = add_axis_trig(load())
    train = df[df.season <= 2023]
    test25 = df[df.season == 2025]
    test24 = df[df.season == 2024]
    sets = ['S1 shipped (aa,ext,v)', 'S3 +spin,axis']
    if len(sys.argv) > 1:
        sets = [s for s in FEATURE_SETS if s.startswith(tuple(sys.argv[1:]))]
    for name in sets:
        feats = FEATURE_SETS[name]
        models = fit_reference(train, feats)
        u25 = unit_agg(score(test25, models, feats))
        u24 = unit_agg(score(test24, models, feats))
        report(name, u25, u24)
