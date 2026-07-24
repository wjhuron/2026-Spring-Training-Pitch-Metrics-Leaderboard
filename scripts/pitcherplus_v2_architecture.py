#!/usr/bin/env python3
"""pitcherplus_v2_architecture.py — is there headroom left, and where?

RESEARCH ONLY (2026-07-24). The phase-2 feature hunt found nothing (see
pitcherplus_v2_candidates.py: sequencing and stamina are role proxies).
This asks the structural questions instead.

1. CEILING. Future xRV/100 is itself a noisy measurement. Its split-half
   reliability caps how well ANY predictor can correlate with it:
   max r = sqrt(reliability of the target). Reported as "% of ceiling
   achieved" so we know whether v2 is worth attempting at all.

2. ROLE-NEUTRAL TARGET. The shipped metric excludes a role term on
   principle, but that leaves role variance in the TARGET, which quietly
   biases every weight toward whatever correlates with being a reliever.
   The structural fix is not to add role to the model — it is to remove
   role from the target: fit against future xRV/100 MINUS its role
   expectation. Then two identical arms in different roles are worth the
   same by construction, and no component earns weight for role proxying.
   Compared three ways: current, role-in-model, role-out-of-target.

3. NONLINEARITY. The composite is linear in shrunk z-scores. A gradient
   boosted model on the identical features under the identical folds
   measures what functional form alone is worth (interactions like
   stuff x command, diminishing returns on xRV/100).

4. RESIDUAL CHAIN. The proposed "each layer only gets credit for what the
   previous layers can't explain" architecture. NOTE THE ALGEBRA: OLS is
   invariant under invertible linear transforms of its predictors, so a
   chain built by LINEAR residualization spans the same column space and
   must produce identical predictions. Verified numerically here rather
   than assumed. Its value is therefore attribution (a waterfall of where
   a pitcher's grade comes from), not accuracy — unless the residuals are
   shrunk at their own rates, which IS a different estimator and is
   tested separately.

Usage: python3 scripts/pitcherplus_v2_architecture.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import leaderboard_metric_battery as bat   # noqa: E402
import pitcherplus_search as ps            # noqa: E402
import pitcherplus_combo as pc             # noqa: E402

SHIPPED = ['stuffRaw', 'locRaw', 'kPct', 'izWhiffPct', 'xrv100', 'gbPct']
W_SHIPPED = np.array([.20, .06, .21, .19, .23, .12])


def hdr(s):
    print('\n' + '═' * 74 + f'\n {s}\n' + '═' * 74)


# ── 1. ceiling ───────────────────────────────────────────────────────────
def ceiling(t):
    hdr('1. CEILING — how much headroom does the target even allow?')
    A = t[(t['half'] == 'A') & (t['n'] >= ps.MIN_HALF)]
    B = t[(t['half'] == 'B') & (t['n'] >= ps.MIN_HALF)]
    ab = A.merge(B, on=['pid', 'season'], suffixes=('_a', '_b'))
    r_half, n = bat.pear(ab['xrv100_a'], ab['xrv100_b'])
    r_full = 2 * r_half / (1 + r_half)          # Spearman-Brown
    print(f'  xRV/100 split-half r          {r_half:.4f}  (n={n})')
    print(f'  full-season reliability       {r_full:.4f}  (Spearman-Brown)')
    print(f'  => max achievable r vs a NOISY target: sqrt(rel) = '
          f'{np.sqrt(r_full):.4f}')
    print('  (a predictor that knew true talent PERFECTLY could not beat '
          'this,\n   because the thing being predicted is itself measured '
          'with error)')
    return np.sqrt(r_full)


# ── 2/3/4 need the panels ────────────────────────────────────────────────
def build(t, kmap, extra=()):
    feats = list(pc.SURVIVORS) + [f for f in extra if f not in pc.SURVIVORS]
    orig = pc.SURVIVORS
    pc.SURVIVORS = feats
    try:
        panels = pc.build_panels(t, kmap)
    finally:
        pc.SURVIVORS = orig
    return panels, feats


def role_neutral(t, kmap, cap):
    hdr('2. ROLE-NEUTRAL TARGET — remove role from the target, not the model')
    (S, Y), feats = build(t, kmap, extra=['pitchesPerG'])
    S_X, S_y, S_g = S
    Y_X, Y_y, Y_g = Y
    base = [feats.index(c) for c in SHIPPED]
    role = feats.index('pitchesPerG')

    def resid_target(X, y):
        """Strip the role expectation out of the target (quadratic in the
        shrunk role z, fit on the pooled panel)."""
        r = X[:, role]
        A = np.column_stack([np.ones(len(y)), r, r ** 2])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return y - A @ beta

    S_yr, Y_yr = resid_target(S_X, S_y), resid_target(Y_X, Y_y)
    rows = []
    for name, cols, ys, yy in (
            ('current (role ignored)', base, S_y, Y_y),
            ('role term IN model', base + [role], S_y, Y_y),
            ('role OUT of target', base, S_yr, Y_yr)):
        rs, _ = pc.oof_r(S_X, ys, S_g, cols)
        ry, _ = pc.oof_r(Y_X, yy, Y_g, cols)
        rows.append({'approach': name, 'r_S': rs, 'r_Y': ry,
                     'combined': (rs + ry) / 2, 'pct_ceiling':
                     100 * ((rs + ry) / 2) / cap})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print('  NOTE: the three rows predict DIFFERENT targets, so r is not '
          'directly\n  comparable across rows — the useful output is the '
          'WEIGHTS below.')

    def wts(X, y, cols):
        b = pc.fit_weights(X, y, cols)
        return b[1:len(SHIPPED) + 1]
    w_cur = (wts(S_X, S_y, base) / wts(S_X, S_y, base).sum()
             + wts(Y_X, Y_y, base) / wts(Y_X, Y_y, base).sum()) / 2
    w_rn = (wts(S_X, S_yr, base) / wts(S_X, S_yr, base).sum()
            + wts(Y_X, Y_yr, base) / wts(Y_X, Y_yr, base).sum()) / 2
    print('\n  normalized weights, current vs role-neutral target:')
    print(f'    {"component":12s} {"shipped":>8s} {"current":>8s} '
          f'{"role-neut":>10s} {"shift":>7s}')
    for i, c in enumerate(SHIPPED):
        print(f'    {c:12s} {W_SHIPPED[i]:8.3f} {w_cur[i]:8.3f} '
              f'{w_rn[i]:10.3f} {w_rn[i] - w_cur[i]:+7.3f}')
    return w_rn, (S_X, S_yr, S_g), (Y_X, Y_yr, Y_g), feats, base, role


def nonlinearity(t, kmap, cap):
    hdr('3. NONLINEARITY — what is functional form alone worth?')
    (S, Y), feats = build(t, kmap)
    S_X, S_y, S_g = S
    Y_X, Y_y, Y_g = Y
    base = [feats.index(c) for c in SHIPPED]
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    except ImportError:
        print('  sklearn unavailable — skipped')
        return
    rows = []
    r_s, _ = pc.oof_r(S_X, S_y, S_g, base)
    r_y, _ = pc.oof_r(Y_X, Y_y, Y_g, base)
    rows.append({'model': 'linear (shipped features)', 'r_S': r_s,
                 'r_Y': r_y, 'combined': (r_s + r_y) / 2})

    def gbm_oof(X, y, g, cols, **kw):
        pred = np.full(len(y), np.nan)
        for gg in np.unique(g):
            tr, te = g != gg, g == gg
            m = HGB(max_depth=3, max_iter=300, learning_rate=0.05,
                    min_samples_leaf=40, **kw).fit(X[tr][:, cols], y[tr])
            pred[te] = m.predict(X[te][:, cols])
        return float(np.corrcoef(pred, y)[0, 1])
    r_s2 = gbm_oof(S_X, S_y, S_g, base)
    r_y2 = gbm_oof(Y_X, Y_y, Y_g, base)
    rows.append({'model': 'GBM (same features)', 'r_S': r_s2, 'r_Y': r_y2,
                 'combined': (r_s2 + r_y2) / 2})
    # explicit pairwise interactions on top of linear
    inter, names = [], []
    for i in range(len(SHIPPED)):
        for j in range(i + 1, len(SHIPPED)):
            inter.append(S_X[:, base[i]] * S_X[:, base[j]])
            names.append(f'{SHIPPED[i]}x{SHIPPED[j]}')
    S_Xi = np.column_stack([S_X[:, base]] + inter)
    Y_Xi = np.column_stack(
        [Y_X[:, base]] + [Y_X[:, base[i]] * Y_X[:, base[j]]
                          for i in range(len(SHIPPED))
                          for j in range(i + 1, len(SHIPPED))])
    ci = list(range(S_Xi.shape[1]))
    r_s3, _ = pc.oof_r(S_Xi, S_y, S_g, ci)
    r_y3, _ = pc.oof_r(Y_Xi, Y_y, Y_g, ci)
    rows.append({'model': 'linear + all 15 interactions', 'r_S': r_s3,
                 'r_Y': r_y3, 'combined': (r_s3 + r_y3) / 2})
    df = pd.DataFrame(rows)
    df['pct_ceiling'] = 100 * df['combined'] / cap
    df['gain'] = df['combined'] - df['combined'].iloc[0]
    print(df.round(4).to_string(index=False))


def residual_chain(t, kmap):
    hdr('4. RESIDUAL CHAIN — accuracy or just attribution?')
    (S, Y), feats = build(t, kmap)
    S_X, S_y, S_g = S
    Y_X, Y_y, Y_g = Y
    base = [feats.index(c) for c in SHIPPED]

    def chain(X):
        """Sequentially residualize each layer on all previous layers."""
        out = np.zeros((len(X), len(base)))
        out[:, 0] = X[:, base[0]]
        for i in range(1, len(base)):
            prev = np.column_stack([np.ones(len(X))] + [out[:, j]
                                                        for j in range(i)])
            beta, *_ = np.linalg.lstsq(prev, X[:, base[i]], rcond=None)
            out[:, i] = X[:, base[i]] - prev @ beta
        return out
    S_C, Y_C = chain(S_X), chain(Y_X)
    ci = list(range(len(base)))
    r_s0, _ = pc.oof_r(S_X, S_y, S_g, base)
    r_y0, _ = pc.oof_r(Y_X, Y_y, Y_g, base)
    r_s1, _ = pc.oof_r(S_C, S_y, S_g, ci)
    r_y1, _ = pc.oof_r(Y_C, Y_y, Y_g, ci)
    print(f'  flat composite   S {r_s0:.6f}  Y {r_y0:.6f}')
    print(f'  residual chain   S {r_s1:.6f}  Y {r_y1:.6f}')
    print(f'  difference       S {r_s1 - r_s0:+.2e}  Y {r_y1 - r_y0:+.2e}')
    print('  => identical to floating point, as the algebra requires: OLS is\n'
          '     invariant under invertible linear transforms of predictors.\n'
          '     The chain is an ATTRIBUTION device (waterfall), never an\n'
          '     accuracy gain.')
    # but: shrinking the RESIDUALS at their own rates IS a different estimator
    print('\n  orthogonalized layer correlations with the target (panel Y),')
    print('  i.e. what each layer adds that the ones above it cannot see:')
    for i, c in enumerate(SHIPPED):
        r = np.corrcoef(Y_C[:, i], Y_y)[0, 1]
        print(f'    {i + 1}. {c:12s} r = {r:+.4f}')


def main():
    t = pc.load_tables()
    feats = [c for c in t.columns if c not in ps.META]
    for c in feats:
        t[c] = pd.to_numeric(t[c], errors='coerce')
    t = ps.add_xrvoe(t)
    kmap = pc.stab_constants()
    kmap.setdefault('pitchesPerG', 10.0)
    cap = ceiling(t)
    role_neutral(t, kmap, cap)
    nonlinearity(t, kmap, cap)
    residual_chain(t, kmap)


if __name__ == '__main__':
    main()
