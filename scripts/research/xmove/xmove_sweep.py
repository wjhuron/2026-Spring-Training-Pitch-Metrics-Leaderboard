"""How much of the GBM's edge can a fixed basis recover, and does the
release-axis frame help?

Why it matters: the site does not run a model. `js/aggregator.js` reads mu/cov
per (pitch type, hand) from metadata_rs.json and takes an MVN conditional mean,
which is exactly OLS on whatever is in the vector. So ANY fixed basis expansion
-- splines, harmonics, tensor interactions -- still ships as a covariance
matrix and needs no new plumbing. A GBM does not; it would have to be scored
per pitch in process_data and summed into every aggregation cell so that
filtered views stay correct.

Sweeps, not eyeballs. Spline df on spin and the axis-harmonic order are
searched over a grid in both frames; the grid is extended if the argmax lands
on an edge, and the winner has to hold in all five seasons individually.

Objective: out-of-sample R^2, cross-fit by game parity within season. Admissible
here (unlike a tuning constant) because every regressor is strictly UPSTREAM of
movement -- spin, release axis, slot, extension, velocity. The circular inputs
that would make R^2 meaningless (OTilt, movement-derived pitch subtype) are not
admitted. DIST -- |corr(OE, raw movement)| at the rendered unit -- is the guard.
"""
import os, sys, gc, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from xmove_compare import load_np, MIN_N, SEASONS, UNIT_MIN, corr, GBM_FEATS


def spline_basis(x, df):
    """Cubic B-spline basis with `df` columns, knots at quantiles.

    Hand-rolled truncated-power bases are unusable here: spin rate cubed is
    ~1e10 against design columns of order 1, and lstsq returns garbage (an
    earlier pass produced R^2 of -1e16 from exactly that). B-splines are
    compactly supported and bounded in [0,1], so the design stays conditioned.
    df<1 drops the term; df==1 keeps it linear."""
    if df < 1:
        return np.empty((len(x), 0))
    if df == 1:
        return x.reshape(-1, 1)
    from scipy.interpolate import BSpline
    k = 3
    lo, hi = x.min(), x.max()
    n_int = max(df - k, 0)
    interior = np.unique(np.percentile(x, np.linspace(0, 100, n_int + 2)[1:-1])) \
        if n_int > 0 else np.array([])
    t = np.concatenate([[lo] * (k + 1), interior, [hi] * (k + 1)])
    B = BSpline.design_matrix(np.clip(x, lo, hi), t, k, extrapolate=False).toarray()
    return B[:, 1:]        # drop one column; the intercept carries it


def build(A, spin_df, velo_df, harm, tensor):
    """Design matrix. tensor=True crosses the spin spline with the axis
    harmonics -- the multiplicative structure the physics actually has
    (break = magnitude(spin, v) x direction(axis)), which purely additive
    terms cannot express."""
    th = np.arctan2(A['ax_sin'], A['ax_cos'])
    sb = spline_basis(A['spin'], spin_df)
    vb = spline_basis(A['velo'], velo_df)
    H = np.column_stack([f(h * th) for h in range(1, harm + 1) for f in (np.sin, np.cos)])
    mats = [np.column_stack([A['ext'], A['aa'], A['aa'] ** 2]), sb, vb, H,
            np.column_stack([A['aa'] * np.sin(th), A['aa'] * np.cos(th)])]
    if tensor:
        mats.append(np.column_stack([sb[:, i] * H[:, j]
                                     for i in range(sb.shape[1])
                                     for j in range(min(2, H.shape[1]))]))
    return np.column_stack(mats)


# Every config must blank the SAME groups, or the R^2 denominators differ and
# the grid is not a comparison. So the per-group training floor is set from the
# LARGEST design in the grid, not from each config's own k. Groups that cannot
# support the widest basis are dropped for all configs alike -- the same
# honesty MVN_MIN_N already applies in process_data.
KMAX = 64
RCOND = 1e-8


def run_linear_X(A, X, polar):
    n = len(A['ivb'])
    xi = np.full(n, np.nan); xh = np.full(n, np.nan)
    XD = np.hstack([np.ones((n, 1)), X])
    order = np.argsort(A['gid'], kind='stable')
    b = np.searchsorted(A['gid'][order], np.arange(A['gid'].max() + 2))
    for gi in range(A['gid'].max() + 1):
        idx = order[b[gi]:b[gi + 1]]
        if len(idx) == 0:
            continue
        par = A['game'][idx] % 2
        for p in (0, 1):
            tr, te = idx[par == p], idx[par == 1 - p]
            if len(tr) < max(MIN_N, 30 * KMAX) or len(te) == 0:
                continue
            Xt, Xs = XD[tr], XD[te]
            if polar:
                a = Xs @ np.linalg.lstsq(Xt, A['along'][tr], rcond=RCOND)[0]
                c = Xs @ np.linalg.lstsq(Xt, A['cross'][tr], rcond=RCOND)[0]
                xi[te] = a * A['ct'][te] - c * A['st'][te]
                xh[te] = a * A['st'][te] + c * A['ct'][te]
            else:
                xi[te] = Xs @ np.linalg.lstsq(Xt, A['ivb'][tr], rcond=RCOND)[0]
                xh[te] = Xs @ np.linalg.lstsq(Xt, A['hb_s'][tr], rcond=RCOND)[0]
    del XD
    gc.collect()
    return xi, xh


def run_gbm(A, polar):
    from sklearn.ensemble import HistGradientBoostingRegressor
    n = len(A['ivb'])
    xi = np.full(n, np.nan); xh = np.full(n, np.nan)
    X = np.column_stack([A[f] for f in GBM_FEATS])
    order = np.argsort(A['gid'], kind='stable')
    b = np.searchsorted(A['gid'][order], np.arange(A['gid'].max() + 2))
    for gi in range(A['gid'].max() + 1):
        idx = order[b[gi]:b[gi + 1]]
        if len(idx) < MIN_N * 4:
            continue
        par = A['game'][idx] % 2
        for p in (0, 1):
            tr, te = idx[par == p], idx[par == 1 - p]
            if len(tr) < MIN_N * 4 or len(te) == 0:
                continue
            t1, t2 = ('along', 'cross') if polar else ('ivb', 'hb_s')
            preds = []
            for tgt in (t1, t2):
                m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08,
                                                  min_samples_leaf=80,
                                                  l2_regularization=1.0, random_state=0)
                m.fit(X[tr], A[tgt][tr])
                preds.append(m.predict(X[te]))
            if polar:
                a, c = preds
                xi[te] = a * A['ct'][te] - c * A['st'][te]
                xh[te] = a * A['st'][te] + c * A['ct'][te]
            else:
                xi[te], xh[te] = preds
    return xi, xh


def light_eval(A, xi, xh):
    """gid already encodes season, so per-group sums via bincount give both the
    pooled and the per-season R^2 in one O(n) pass."""
    ok = np.isfinite(xi) & np.isfinite(xh)
    io, ho = A['ivb'] - xi, A['hb_s'] - xh
    g = A['gid'][ok]
    ng = A['gid'].max() + 1
    cnt = np.bincount(g, minlength=ng).astype(float)
    ssr_i = np.bincount(g, weights=io[ok] ** 2, minlength=ng)
    ssr_h = np.bincount(g, weights=ho[ok] ** 2, minlength=ng)
    mi = np.bincount(g, weights=A['ivb'][ok], minlength=ng)
    mh = np.bincount(g, weights=A['hb_s'][ok], minlength=ng)
    qi = np.bincount(g, weights=A['ivb'][ok] ** 2, minlength=ng)
    qh = np.bincount(g, weights=A['hb_s'][ok] ** 2, minlength=ng)
    with np.errstate(invalid='ignore', divide='ignore'):
        sst_i = qi - mi ** 2 / cnt
        sst_h = qh - mh ** 2 / cnt
    gseason = np.zeros(ng, dtype=int)
    gseason[A['gid'][ok]] = A['season'][ok]
    ni, nh, di, dh = ssr_i.sum(), ssr_h.sum(), np.nansum(sst_i), np.nansum(sst_h)
    per = {}
    for y in SEASONS:
        m = gseason == y
        per[y] = (1 - ssr_i[m].sum() / np.nansum(sst_i[m]),
                  1 - ssr_h[m].sum() / np.nansum(sst_h[m]))
    u = (pd.DataFrame({'u': A['unit'][ok], 'pt': A['pt'][ok], 'ivb': A['ivb'][ok],
                       'hb': A['hb_s'][ok], 'io': io[ok], 'ho': ho[ok]})
         .groupby('u').agg(n=('ivb', 'size'), pt=('pt', 'first'), ivb=('ivb', 'mean'),
                           hb=('hb', 'mean'), io=('io', 'mean'), ho=('ho', 'mean')))
    u = u[u.n >= UNIT_MIN]
    di_l, dh_l = [], []
    for pt, g in u.groupby('pt'):
        if len(g) >= 100:
            di_l.append(abs(corr(g.io.values, g.ivb.values)))
            dh_l.append(abs(corr(g.ho.values, g.hb.values)))
    return dict(r2i=1 - ni / di, r2h=1 - nh / dh, dist_i=np.mean(di_l),
                dist_h=np.mean(dh_l), per=per, cov=ok.mean())


def line(tag, k, r):
    mn_i = min(v[0] for v in r['per'].values())
    mn_h = min(v[1] for v in r['per'].values())
    print(f'{tag:>34} {k:>4} {r["cov"]:>5.3f} {r["r2i"]:>6.3f} {r["r2h"]:>6.3f} '
          f'{r["dist_i"]:>6.3f} {r["dist_h"]:>6.3f} {mn_i:>7.3f} {mn_h:>7.3f}', flush=True)


if __name__ == '__main__':
    A = load_np()
    print(f"{len(A['ivb']):,} pitches\n")
    print(f"{'config':>34} {'k':>4} {'cov':>5} {'R2i':>6} {'R2h':>6} {'DISTi':>6} "
          f"{'DISTh':>6} {'wrstYi':>7} {'wrstYh':>7}")
    print('-' * 88)
    import os as _os
    grid = [(sd, 3, h, True) for sd in (1, 2, 4, 6) for h in (1, 2, 3)]
    if _os.environ.get('XMOVE_GRID') == 'wide':      # edge-of-grid extension
        grid = [(sd, vd, h, True) for sd in (8, 10, 12) for vd in (3, 6)
                for h in (4, 5, 6)]
    for polar in (True,):
        frame = 'polar' if polar else 'cart '
        for sd, vd, h, tn in grid:
            X = build(A, sd, vd, h, tn)
            xi, xh = run_linear_X(A, X, polar)
            k = X.shape[1] + 1
            del X; gc.collect()
            line(f'{frame} spin{sd} velo{vd} harm{h} tn{int(tn)}', k, light_eval(A, xi, xh))
            del xi, xh; gc.collect()
    for polar in (False, True):
        xi, xh = run_gbm(A, polar)
        line(f"GBM {'polar' if polar else 'cart '} (ceiling)", 0, light_eval(A, xi, xh))
        del xi, xh; gc.collect()
