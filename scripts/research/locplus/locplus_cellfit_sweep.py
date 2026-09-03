"""locplus_cellfit_sweep.py — out-of-sample CELL FIT of the Loc+ league
surfaces, 2021-2026, all 30 ordered season pairs (2026-09-02).

WHY THIS EXISTS. Every smoothing/shrinkage constant in pipeline/locplus.py
(PHYS_X_IN, PHYS_Z_FRAC, K_WHIFF, K_WH_COUNT, K_FOUL, K_XWCON, K_SWING_COLL,
K_SWING_COUNT, K_CS) was either set on a partial season with reliability
inside the objective, shown flat only on a pitcher-level partial-r objective,
a best-of-3, pinned in every replicate harness, or has no provenance at all.
None was measured on the objective this repo fixed on 2026-08-16 for cell
and surface shrinkage constants (memory: project_cell_shrinkage_audit_2026_08):

    build the table on one FULL season, score the held-out pitches of
    ANOTHER season; log loss for a probability cell, MSE for a mean cell.

That objective cannot be gamed by over-smoothing (a flat surface loses to
the truth on held-out cells) and it is measured at the sample size
production actually runs at (a full season), unlike the quarter-season
replicate harness (locplus_constants_multiseason.py), which builds its
surfaces on a quarter of a season and therefore favours over-regularisation.

WHAT IS MEASURED. Per surface, the smoothing is production's: separable
truncated Gaussian kernels with UNNORMALISED weights exp(-0.5 d^2) over
numerator and denominator separately (no renormalisation at the edges, a
missing neighbour contributes nothing), then

    p = (smoothed_num + K * prior) / (smoothed_den + K)

so every K is in kernel-weighted units and coupled to the bandwidth, which
is why bandwidth and K are swept JOINTLY. The per-count grids are composed
exactly as production composes them: the collapsed grid with its own K,
then the per-count grid shrunk toward collapsed x league count multiplier
(capped at 1). The called-strike surface keeps the count transform on
(logit shift per (hand, count), fitted on the TRAIN season, applied to the
test season). The numpy reimplementation is asserted against lp._smooth on
a real key to 1e-9 before any sweep runs.

Reads the smoothing geometry (bins, bounds, kernel truncation) from
pipeline.locplus so it cannot drift from production.

DATA. 2021-2025 public Statcast caches through base.adapt(); 2026 the
production pitch cache (MLB rows, Savant-denominated PlateX/PlateZ since
2026-08-29). One known asymmetry: adapt() sets Event/BBType to None, so a
BUNT put in play in 2021-2025 is NOT detectable and enters the baseline,
while 2026 excludes it through BBType='bunt'. The count of such pitches is
recorded per season in the counts cache ('bunt_bip_in_source') — it is
about 0.3% of balls in play.

Guts: the 2026 (lg_woba, woba_scale) pair is used for every season, as the
existing harness does. For the contact MSE that is a per-season LEVEL shift
of the standardised value; it adds a constant to every config's loss within
a pair and cannot move the argmin, because the shrinkage target is the
train season's own mean.

Usage:
    python3 scripts/research/locplus/locplus_cellfit_sweep.py accumulate 2021
    ...   (one season per process: the pickles are large and the box has 8 GB)
    python3 scripts/research/locplus/locplus_cellfit_sweep.py sweep
    python3 scripts/research/locplus/locplus_cellfit_sweep.py report
"""
import argparse
import gc
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline.locplus as lp
import locplus_constants_multiseason as base

LG, SCALE = base.LG, base.SCALE
SEASONS = {2021: 'data/_statcast2021_cache.pkl',
           2022: 'data/_statcast2022_cache.pkl',
           2023: 'data/_statcast2023_cache.pkl',
           2024: 'data/_statcast2024_cache.pkl',
           2025: 'data/_statcast2025_full_cache.pkl',
           2026: 'data/all_pitches_rs_cache.pkl'}
COUNTS_PKL = os.path.join(ROOT, 'data', '_loc_cellfit_counts.pkl')
SWEEP_PKL = os.path.join(ROOT, 'data', '_loc_cellfit_sweep_raw.pkl')
OUT_JSON = os.path.join(ROOT, 'data', '_loc_cellfit_sweep.json')
OUT_MD = os.path.join(ROOT, 'data', '_loc_cellfit_sweep.md')

NX, NZ = lp.NX, lp.NZ
NC = len(lp.COUNTS)
COUNT_IDX = {c: i for i, c in enumerate(lp.COUNTS)}
HAND_IDX = {'L': 0, 'R': 1}

BX_GRID = [2.0, 3.0, 4.5, 6.0, 9.0, 13.0, 20.0]        # inches
BZ_GRID = [0.12, 0.17, 0.22, 0.30, 0.40, 0.55]          # zone heights
K_BASE = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
K_INF = 1e9                                             # "prior only"
SHIPPED = {'bx': lp.PHYS_X_IN, 'bz': lp.PHYS_Z_FRAC,
           'K_WHIFF': lp.K_WHIFF, 'K_WH_COUNT': lp.K_WH_COUNT,
           'K_FOUL': lp.K_FOUL, 'K_XWCON': lp.K_XWCON,
           'K_SWING_COLL': lp.K_SWING_COLL, 'K_SWING_COUNT': lp.K_SWING_COUNT,
           'K_CS': lp.K_CS}
MIN_CT_TAKES = 50            # mirrors build_surfaces' count-transform floor
P_CLIP = 1e-6

# surface -> (loss kind, constant names in K-combo order)
SURFACES = {
    'whiff': ('logloss', ('K_WHIFF', 'K_WH_COUNT')),
    'swing': ('logloss', ('K_SWING_COLL', 'K_SWING_COUNT')),
    'foul': ('logloss', ('K_FOUL',)),
    'contact': ('mse', ('K_XWCON',)),
    'cs': ('logloss', ('K_CS',)),
}


def kgrid(name):
    return sorted(set(K_BASE) | {SHIPPED[name]}) + [K_INF]


# ═════════════════════════════════════════════════════════════════════════
#  ACCUMULATE (production semantics, one pass per season)
# ═════════════════════════════════════════════════════════════════════════
def load_season(yr):
    path = os.path.join(ROOT, SEASONS[yr])
    meta = {}
    if yr == 2026:
        D = pickle.load(open(path, 'rb'))
        pitches = [p for p in D if p.get('_source') == 'MLB']
        del D
        gc.collect()
        meta['bunt_bip_detectable'] = True
        meta['bunt_bip_excluded'] = sum(
            1 for p in pitches if p.get('Description') == 'In Play'
            and p.get('BBType') in lp.BUNT_BB)
        meta['dates'] = len({p.get('Game Date') for p in pitches})
    else:
        import pandas as pd
        df = pd.read_pickle(path)
        ev = df['events'].astype(str).str.lower()
        meta['bunt_bip_detectable'] = False
        meta['bunt_bip_in_source'] = int(
            ((df['description'] == 'hit_into_play') & ev.str.contains('bunt')).sum())
        meta['dates'] = int(df['game_date'].astype(str).str[:10].nunique())
        del df, ev
        gc.collect()
        pitches = base.adapt(path)
    meta['rows'] = len(pitches)
    return pitches, meta


DESC_CODE = {'Ball': 0, 'Called Strike': 1, 'Swinging Strike': 2,
             'Foul': 3, 'In Play': 4}


def accumulate(pitches):
    """Raw per-cell counts with production binning and eligibility."""
    keyidx = {}
    K, C, I, J, D, H, XW = [], [], [], [], [], [], []
    n_base = 0
    for p in pitches:
        if not lp.is_eligible_baseline(p):
            continue
        n_base += 1
        key = (lp.group_of(p), p['Bats'], p['Throws'])
        k = keyidx.setdefault(key, len(keyidx))
        c = COUNT_IDX[lp.get_count(p)]
        i = lp._xbin(lp.safe_float(p.get('PlateX')))
        j = lp._zbin(lp._znorm(p))
        d = DESC_CODE.get(p.get('Description'), -1)
        xw = lp.safe_float(p.get('xwOBA')) if d == 4 else None
        K.append(k); C.append(c); I.append(i); J.append(j); D.append(d)
        H.append(HAND_IDX[p['Bats']])
        XW.append(np.nan if xw is None else xw)
    K = np.array(K); C = np.array(C); I = np.array(I); J = np.array(J)
    D = np.array(D); H = np.array(H); XW = np.array(XW, dtype=float)
    nk = len(keyidx)

    def grid4(mask, w=None):
        g = np.zeros((nk, NC, NX, NZ))
        np.add.at(g, (K[mask], C[mask], I[mask], J[mask]),
                  1.0 if w is None else w[mask])
        return g

    def grid3(mask, w=None):
        g = np.zeros((nk, NX, NZ))
        np.add.at(g, (K[mask], I[mask], J[mask]), 1.0 if w is None else w[mask])
        return g

    swing = np.isin(D, (2, 3, 4))
    take = np.isin(D, (0, 1))
    bip = (D == 4) & ~np.isnan(XW)
    v = (XW - LG) / SCALE
    out = {
        'keys': [k for k, _ in sorted(keyidx.items(), key=lambda kv: kv[1])],
        'sw_d': grid4(np.ones(len(K), bool)),      # every baseline pitch
        'sw_n': grid4(swing),                       # swings
        'wh_n': grid4(D == 2),                      # whiffs (den = sw_n)
        'fl_n': grid3(D == 3),                      # fouls  (den = sw_n coll)
        'bip_n': grid3(bip),
        'bip_s1': grid3(bip, v),
        'bip_s2': grid3(bip, v * v),
        'n_base': int(n_base),
        'desc_counts': {int(x): int(n) for x, n in zip(*np.unique(D, return_counts=True))},
    }
    cs_d = np.zeros((2, NC, NX, NZ)); cs_n = np.zeros((2, NC, NX, NZ))
    np.add.at(cs_d, (H[take], C[take], I[take], J[take]), 1.0)
    cs1 = D == 1
    np.add.at(cs_n, (H[cs1], C[cs1], I[cs1], J[cs1]), 1.0)
    out['cs_d'] = cs_d; out['cs_n'] = cs_n
    return out


def cmd_accumulate(yr):
    t0 = time.time()
    pitches, meta = load_season(yr)
    print(f"{yr}: {meta['rows']} rows loaded ({time.time() - t0:.0f}s)", flush=True)
    acc = accumulate(pitches)
    acc['meta'] = meta
    del pitches
    gc.collect()
    print(f"{yr}: baseline {acc['n_base']} pitches, {len(acc['keys'])} keys, "
          f"desc {acc['desc_counts']}  meta {meta}", flush=True)
    cache = pickle.load(open(COUNTS_PKL, 'rb')) if os.path.exists(COUNTS_PKL) else {}
    cache[yr] = acc
    tmp = COUNTS_PKL + '.tmp'
    pickle.dump(cache, open(tmp, 'wb'))
    os.replace(tmp, COUNTS_PKL)
    print(f"{yr}: cached -> {COUNTS_PKL} ({time.time() - t0:.0f}s)", flush=True)


# ═════════════════════════════════════════════════════════════════════════
#  SMOOTHER (numpy twin of lp._smooth)
# ═════════════════════════════════════════════════════════════════════════
def k1d_np(bw):
    """Same taps as lp._k1d: window ceil(3*bw), unnormalised Gaussian."""
    win = max(1, int(math.ceil(3 * bw)))
    d = np.arange(-win, win + 1)
    return np.exp(-0.5 * (d / bw) ** 2)


def conv_axis(G, w, axis):
    """Zero-padded truncated convolution along one axis (missing neighbours
    contribute nothing, exactly as the production loops)."""
    n = G.shape[axis]
    win = len(w) // 2
    out = np.zeros_like(G)
    for t, wt in enumerate(w):
        d = t - win
        if abs(d) >= n:
            continue
        src = [slice(None)] * G.ndim
        dst = [slice(None)] * G.ndim
        if d >= 0:
            dst[axis] = slice(0, n - d); src[axis] = slice(d, n)
        else:
            dst[axis] = slice(-d, n); src[axis] = slice(0, n + d)
        out[tuple(dst)] += wt * G[tuple(src)]
    return out


def smooth_np(G, wx, wz):
    """G has trailing axes (NX, NZ). Z pass then X pass, like _smooth."""
    return conv_axis(conv_axis(G, wz, -1), wx, -2)


def shrink(sn, sd, prior, K):
    """(sn + K*prior) / (sd + K), prior where the denominator is zero."""
    s = sd + K
    prior_b = np.broadcast_to(prior, np.broadcast(sn, prior).shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(s > 0, (sn + K * prior_b) / np.where(s > 0, s, 1.0), prior_b)
    return out


def verify_smoother(cache):
    """Assert the numpy twin reproduces lp._smooth on a real key to 1e-9."""
    yr = max(cache)
    acc = cache[yr]
    k = acc['keys'].index(('FF', 'R', 'R'))
    num = acc['wh_n'][k].sum(0); den = acc['sw_n'][k].sum(0)
    prior = float(num.sum() / max(den.sum(), 1))
    worst = 0.0
    for (bx, bz, K) in ((lp.PHYS_X_IN, lp.PHYS_Z_FRAC, lp.K_WHIFF),
                        (9.0, 0.40, 3), (2.0, 0.12, 0), (20.0, 0.55, 500)):
        kx = lp._k1d(bx / lp.BIN_X_IN); kz = lp._k1d(bz / lp.BIN_Z)
        ref = np.array(lp._smooth(num.tolist(), den.tolist(), prior, K, kx, kz))
        wx = k1d_np(bx / lp.BIN_X_IN); wz = k1d_np(bz / lp.BIN_Z)
        mine = shrink(smooth_np(num, wx, wz), smooth_np(den, wx, wz), prior, K)
        worst = max(worst, float(np.abs(ref - mine).max()))
        # array prior (the per-count composition path)
        pr_arr = np.minimum(1.0, mine * 1.3)
        numc = acc['wh_n'][k][COUNT_IDX[(1, 2)]]; denc = acc['sw_n'][k][COUNT_IDX[(1, 2)]]
        ref2 = np.array(lp._smooth(numc.tolist(), denc.tolist(), pr_arr.tolist(), K, kx, kz))
        mine2 = shrink(smooth_np(numc, wx, wz), smooth_np(denc, wx, wz), pr_arr, K)
        worst = max(worst, float(np.abs(ref2 - mine2).max()))
    assert worst < 1e-9, f"numpy smoother differs from lp._smooth by {worst:.3e}"
    print(f"smoother verified against lp._smooth on {yr} ('FF','R','R'): "
          f"max |diff| = {worst:.2e}", flush=True)


# ═════════════════════════════════════════════════════════════════════════
#  LOSSES
# ═════════════════════════════════════════════════════════════════════════
def logloss(npos, ntot, p):
    """Summed negative log-likelihood; p may carry leading config axes."""
    p = np.clip(p, P_CLIP, 1 - P_CLIP)
    ll = -(npos * np.log(p) + (ntot - npos) * np.log1p(-p))
    return ll.sum(axis=tuple(range(p.ndim - npos.ndim, p.ndim)))


def mse_sum(n, s1, s2, p):
    v = s2 - 2 * p * s1 + n * p * p
    return v.sum(axis=tuple(range(p.ndim - n.ndim, p.ndim)))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sig(x):
    return 1 / (1 + np.exp(-x))


def align(acc_test, keys_train):
    """Test-season arrays aligned to the train key order; unseen keys are
    zero (dropped from the loss) and counted."""
    idx = {k: i for i, k in enumerate(acc_test['keys'])}
    nk = len(keys_train)
    out = {}
    for name in ('sw_d', 'sw_n', 'wh_n', 'fl_n', 'bip_n', 'bip_s1', 'bip_s2'):
        src = acc_test[name]
        dst = np.zeros((nk,) + src.shape[1:])
        for i, k in enumerate(keys_train):
            if k in idx:
                dst[i] = src[idx[k]]
        out[name] = dst
    missing = [k for k in acc_test['keys'] if k not in set(keys_train)]
    out['dropped_pitches'] = int(sum(acc_test['sw_d'][idx[k]].sum() for k in missing))
    out['cs_d'] = acc_test['cs_d']; out['cs_n'] = acc_test['cs_n']
    return out


# ═════════════════════════════════════════════════════════════════════════
#  SWEEP
# ═════════════════════════════════════════════════════════════════════════
def count_multipliers(num4, den4):
    """League per-count multiplier as build_surfaces computes it: count rate
    over overall rate, from the unsmoothed totals."""
    num_c = num4.sum(axis=(0, 2, 3)); den_c = den4.sum(axis=(0, 2, 3))
    overall = num_c.sum() / den_c.sum() if den_c.sum() else 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        m = np.where((den_c > 0) & (overall > 0), (num_c / den_c) / overall, 1.0)
    return m


def sweep_pair_family(train, tests, wx, wz, num4, den4, kc_grid, kn_grid,
                      test_num, test_den):
    """Two-level composition (collapsed K, then per-count K toward collapsed
    x multiplier). Returns losses[(Kc, Kn)] -> array over tests."""
    sn_c = smooth_np(num4, wx, wz); sd_c = smooth_np(den4, wx, wz)
    sn_coll = sn_c.sum(1); sd_coll = sd_c.sum(1)
    tot_n = num4.sum(axis=(1, 2, 3)); tot_d = den4.sum(axis=(1, 2, 3))
    prior = (tot_n / np.maximum(tot_d, 1))[:, None, None]
    mult = count_multipliers(num4, den4)
    Kn = np.array(kn_grid)[:, None, None, None, None]
    out = {}
    for Kc in kc_grid:
        coll = shrink(sn_coll, sd_coll, prior, Kc)
        prior_c = np.minimum(1.0, coll[:, None] * mult[None, :, None, None])
        P = shrink(sn_c[None], sd_c[None], prior_c[None], Kn)   # (nKn, nk, NC, NX, NZ)
        L = np.stack([logloss(tn, td, P) for tn, td in zip(test_num, test_den)], axis=1)
        for a, Knv in enumerate(kn_grid):
            out[(Kc, Knv)] = L[a]
    return out


def sweep_cs(wx, wz, csn, csd, k_grid, test_csn, test_csd):
    """Base surface per hand + count transform fitted on train."""
    base_n = smooth_np(csn.sum(1), wx, wz); base_d = smooth_np(csd.sum(1), wx, wz)
    prior = (csn.sum(axis=(1, 2, 3)) / np.maximum(csd.sum(axis=(1, 2, 3)), 1))[:, None, None]
    tk_n = csd.sum(axis=(2, 3)); obs = csn.sum(axis=(2, 3))          # (2, NC)
    out, out_nt = {}, {}
    for K in k_grid:
        b = shrink(base_n, base_d, prior, K)                          # (2, NX, NZ)
        pred = (csd * b[:, None]).sum(axis=(2, 3))
        ok = (tk_n >= MIN_CT_TAKES) & (obs > 0) & (obs < tk_n) & (pred > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            delta = np.where(ok, _logit(obs / np.maximum(tk_n, 1))
                             - _logit(pred / np.maximum(tk_n, 1)), 0.0)
        P = np.where(delta[:, :, None, None] == 0.0, b[:, None],
                     _sig(_logit(b)[:, None] + delta[:, :, None, None]))
        out[(K,)] = np.array([logloss(tn, td, P) for tn, td in zip(test_csn, test_csd)])
        Pnt = np.broadcast_to(b[:, None], P.shape)
        out_nt[(K,)] = np.array([logloss(tn, td, Pnt) for tn, td in zip(test_csn, test_csd)])
    return out, out_nt


def cmd_sweep(args):
    cache = pickle.load(open(COUNTS_PKL, 'rb'))
    seasons = sorted(cache)
    if args.seasons:
        seasons = [int(s) for s in args.seasons.split(',')]
    verify_smoother(cache)
    bx_grid = [float(x) for x in args.bx.split(',')] if args.bx else BX_GRID
    bz_grid = [float(x) for x in args.bz.split(',')] if args.bz else BZ_GRID
    pairs = [(t, s) for t in seasons for s in seasons if s != t]
    print(f"seasons {seasons}: {len(pairs)} ordered pairs; "
          f"bx {bx_grid}; bz {bz_grid}", flush=True)
    grids = {n: kgrid(n) for n in SHIPPED if n.startswith('K_')}
    res = {s: defaultdict(dict) for s in SURFACES}
    res['cs_notransform'] = defaultdict(dict)
    N = {}                     # normalisers per (surface, pair)
    dropped = {}
    t0 = time.time()
    for tr in seasons:
        A = cache[tr]
        tests = [s for s in seasons if s != tr]
        T = {s: align(cache[s], A['keys']) for s in tests}
        for s in tests:
            dropped[(tr, s)] = T[s]['dropped_pitches']
            N[('whiff', tr, s)] = float(T[s]['sw_n'].sum())
            N[('swing', tr, s)] = float(T[s]['sw_d'].sum())
            N[('foul', tr, s)] = float(T[s]['sw_n'].sum())
            N[('contact', tr, s)] = float(T[s]['bip_n'].sum())
            N[('cs', tr, s)] = float(T[s]['cs_d'].sum())
        for bx in bx_grid:
            wx = k1d_np(bx / lp.BIN_X_IN)
            for bz in bz_grid:
                wz = k1d_np(bz / lp.BIN_Z)
                bw = (bx, bz)
                # whiff: whn / swn per count
                L = sweep_pair_family(tr, tests, wx, wz, A['wh_n'], A['sw_n'],
                                      grids['K_WHIFF'], grids['K_WH_COUNT'],
                                      [T[s]['wh_n'] for s in tests],
                                      [T[s]['sw_n'] for s in tests])
                for kk, arr in L.items():
                    for s, v in zip(tests, arr):
                        res['whiff'][bw].setdefault(kk, {})[(tr, s)] = float(v)
                # swing: swn / swd per count
                L = sweep_pair_family(tr, tests, wx, wz, A['sw_n'], A['sw_d'],
                                      grids['K_SWING_COLL'], grids['K_SWING_COUNT'],
                                      [T[s]['sw_n'] for s in tests],
                                      [T[s]['sw_d'] for s in tests])
                for kk, arr in L.items():
                    for s, v in zip(tests, arr):
                        res['swing'][bw].setdefault(kk, {})[(tr, s)] = float(v)
                # foul: fln / swn (collapsed)
                sn = smooth_np(A['fl_n'], wx, wz); sd = smooth_np(A['sw_n'].sum(1), wx, wz)
                prior = (A['fl_n'].sum(axis=(1, 2)) / np.maximum(A['sw_n'].sum(axis=(1, 2, 3)), 1))[:, None, None]
                Kf = np.array(grids['K_FOUL'])[:, None, None, None]
                P = shrink(sn[None], sd[None], prior[None], Kf)
                for s in tests:
                    Ls = logloss(T[s]['fl_n'], T[s]['sw_n'].sum(1), P)
                    for a, K in enumerate(grids['K_FOUL']):
                        res['foul'][bw].setdefault((K,), {})[(tr, s)] = float(Ls[a])
                # contact: bip_s1 / bip_n (MSE)
                sn = smooth_np(A['bip_s1'], wx, wz); sd = smooth_np(A['bip_n'], wx, wz)
                prior = (A['bip_s1'].sum(axis=(1, 2)) / np.maximum(A['bip_n'].sum(axis=(1, 2)), 1))[:, None, None]
                Kx = np.array(grids['K_XWCON'])[:, None, None, None]
                P = shrink(sn[None], sd[None], prior[None], Kx)
                for s in tests:
                    Ls = mse_sum(T[s]['bip_n'], T[s]['bip_s1'], T[s]['bip_s2'], P)
                    for a, K in enumerate(grids['K_XWCON']):
                        res['contact'][bw].setdefault((K,), {})[(tr, s)] = float(Ls[a])
                # called strike
                Lc, Lnt = sweep_cs(wx, wz, A['cs_n'], A['cs_d'], grids['K_CS'],
                                   [T[s]['cs_n'] for s in tests],
                                   [T[s]['cs_d'] for s in tests])
                for kk, arr in Lc.items():
                    for s, v in zip(tests, arr):
                        res['cs'][bw].setdefault(kk, {})[(tr, s)] = float(v)
                for kk, arr in Lnt.items():
                    for s, v in zip(tests, arr):
                        res['cs_notransform'][bw].setdefault(kk, {})[(tr, s)] = float(v)
            print(f"  train {tr} bx {bx:>5.1f} done ({time.time() - t0:.0f}s)", flush=True)
    out = {'seasons': seasons, 'pairs': pairs, 'bx_grid': bx_grid, 'bz_grid': bz_grid,
           'kgrids': grids, 'N': N, 'dropped': dropped, 'res': {k: dict(v) for k, v in res.items()},
           'meta': {yr: cache[yr]['meta'] | {'n_base': cache[yr]['n_base'], 'keys': len(cache[yr]['keys'])}
                    for yr in seasons}}
    tmp = SWEEP_PKL + '.tmp'
    pickle.dump(out, open(tmp, 'wb'))
    os.replace(tmp, SWEEP_PKL)
    print(f"sweep done in {time.time() - t0:.0f}s -> {SWEEP_PKL}", flush=True)


# ═════════════════════════════════════════════════════════════════════════
#  REPORT
# ═════════════════════════════════════════════════════════════════════════
def kfmt(K):
    if isinstance(K, str):   # already formatted (classify() receives kfmt'd values)
        return K
    return 'inf' if K >= K_INF else (str(int(K)) if float(K).is_integer() else str(K))


def cfg_name(bw, kk):
    return f"bx{bw[0]:g}/bz{bw[1]:g}/K{'-'.join(kfmt(k) for k in kk)}"


def paired_stats(a, b, pairs):
    """a - b over the pairs: mean, naive SE (n pairs), cluster SE by TEST
    season (n test seasons), wins of a (lower loss)."""
    d = np.array([a[p] - b[p] for p in pairs])
    se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float('nan')
    by_test = defaultdict(list)
    for p, x in zip(pairs, d):
        by_test[p[1]].append(x)
    cl = np.array([np.mean(v) for v in by_test.values()])
    se_cl = cl.std(ddof=1) / math.sqrt(len(cl)) if len(cl) > 1 else float('nan')
    return {'mean': float(d.mean()), 'se': float(se), 'se_cluster': float(se_cl),
            't': float(d.mean() / se) if se > 0 else float('nan'),
            'wins': int((d < 0).sum()), 'n': len(d)}


def analyse(sw):
    pairs = [tuple(p) for p in sw['pairs']]
    npairs = len(pairs)
    report = {'pairs': npairs, 'seasons': sw['seasons'], 'surfaces': {}}
    common = {}                                  # (bw) -> {surface: excess SE units}
    for surf, (kind, knames) in SURFACES.items():
        R = sw['res'][surf]
        # per-pair loss normalised per event (nats or squared std-xwOBA units)
        norm = {p: sw['N'][(surf,) + p] for p in pairs}
        table = {}                               # (bw, kk) -> {pair: loss/N}
        for bw, dd in R.items():
            for kk, per in dd.items():
                table[(bw, kk)] = {p: per[p] / norm[p] for p in pairs}
        means = {c: float(np.mean([v[p] for p in pairs])) for c, v in table.items()}
        argmin = min(means, key=means.get)
        ship_bw = (SHIPPED['bx'], SHIPPED['bz'])
        ship_kk = tuple(SHIPPED[n] for n in knames)
        shipped = (ship_bw, ship_kk)
        st_ship = paired_stats(table[shipped], table[argmin], pairs)
        # flat region: configs within 1 naive SE of the argmin
        flat = []
        for c in table:
            st = paired_stats(table[c], table[argmin], pairs)
            if st['mean'] <= st['se']:
                flat.append((cfg_name(*c), round(st['mean'], 7), round(st['se'], 7)))
        # bandwidth ranges within the flat region
        flat_bx = sorted({float(n.split('/')[0][2:]) for n, _, _ in flat})
        flat_bz = sorted({float(n.split('/')[1][2:]) for n, _, _ in flat})
        # marginal curves at the shipped operating point + profiled bandwidths
        curves = {}
        for i, n in enumerate(knames):
            cur = []
            for K in sw['kgrids'][n]:
                kk = tuple(K if j == i else ship_kk[j] for j in range(len(knames)))
                c = (ship_bw, kk)
                st = paired_stats(table[c], table[shipped], pairs)
                cur.append({'value': kfmt(K), 'loss': means[c], 'd_vs_shipped': st['mean'],
                            'se': st['se'], 'se_cluster': st['se_cluster'], 't': st['t'],
                            'wins_vs_shipped': st['wins']})
            curves[n] = cur
        for axis, grid in (('bx', sw['bx_grid']), ('bz', sw['bz_grid'])):
            fixed, prof = [], []
            for v in grid:
                bw = (v, ship_bw[1]) if axis == 'bx' else (ship_bw[0], v)
                c = (bw, ship_kk)
                st = paired_stats(table[c], table[shipped], pairs)
                fixed.append({'value': v, 'loss': means[c], 'd_vs_shipped': st['mean'],
                              'se': st['se'], 't': st['t'], 'wins_vs_shipped': st['wins']})
                best = min((cc for cc in table if cc[0] == bw), key=means.get)
                st = paired_stats(table[best], table[shipped], pairs)
                prof.append({'value': v, 'best_K': '-'.join(kfmt(k) for k in best[1]),
                             'loss': means[best], 'd_vs_shipped': st['mean'],
                             'se': st['se'], 't': st['t'], 'wins_vs_shipped': st['wins']})
            curves[axis + '_fixedK'] = fixed
            curves[axis + '_profiled'] = prof
        # joint bandwidth surface: best K per (bx,bz)
        bw_best = {}
        for bw in R:
            best = min((cc for cc in table if cc[0] == bw), key=means.get)
            st = paired_stats(table[best], table[argmin], pairs)
            bw_best[bw] = {'best_K': '-'.join(kfmt(k) for k in best[1]), 'loss': means[best],
                           'excess_vs_argmin': st['mean'], 'se': st['se'],
                           'excess_se_units': (st['mean'] / st['se']) if st['se'] > 0 else 0.0}
            common.setdefault(bw, {})[surf] = bw_best[bw]['excess_se_units']
        # claim classes at the shipped operating point
        classes = {}
        for n in knames:
            classes[n] = classify(curves[n], kfmt(SHIPPED[n]), is_K=True)
        classes['bx'] = classify(curves['bx_profiled'], SHIPPED['bx'], is_K=False)
        classes['bz'] = classify(curves['bz_profiled'], SHIPPED['bz'], is_K=False)
        # reference: prior-only loss (K = inf everywhere) at shipped bw
        inf_kk = tuple(K_INF for _ in knames)
        ref_inf = means.get((ship_bw, inf_kk))
        report['surfaces'][surf] = {
            'loss_kind': kind, 'constants': list(knames),
            'argmin': cfg_name(*argmin), 'argmin_loss': means[argmin],
            'shipped': cfg_name(*shipped), 'shipped_loss': means[shipped],
            'shipped_minus_argmin': st_ship, 'argmin_wins_vs_shipped': npairs - st_ship['wins'],
            'prior_only_loss_at_shipped_bw': ref_inf,
            'flat_region_n': len(flat), 'flat_bx': flat_bx, 'flat_bz': flat_bz,
            'flat_region': flat, 'curves': curves,
            'bw_profile': {cfg_name(bw, ()): v for bw, v in bw_best.items()},
            'classes': classes,
        }
    # common bandwidth: max excess (SE units) across surfaces per (bx,bz)
    rows = []
    for bw, d in common.items():
        rows.append({'bx': bw[0], 'bz': bw[1], 'max_excess_se': max(d.values()),
                     'sum_excess_se': sum(d.values()), 'per_surface': d})
    rows.sort(key=lambda r: r['max_excess_se'])
    report['common_bandwidth'] = rows
    # cs no-transform reference
    R = sw['res']['cs_notransform']
    ship = (SHIPPED['bx'], SHIPPED['bz'])
    norm = {p: sw['N'][('cs',) + p] for p in pairs}
    report['cs_notransform_at_shipped'] = {
        kfmt(kk[0]): float(np.mean([R[ship][kk][p] / norm[p] for p in pairs]))
        for kk in R[ship]}
    report['dropped_pitches'] = {f"{a}->{b}": v for (a, b), v in sw['dropped'].items()}
    report['meta'] = sw['meta']
    return report


def classify(curve, shipped_value, is_K):
    """Claim class of the shipped value on a 1-D loss curve."""
    vals = [c['value'] for c in curve]
    key = kfmt(shipped_value) if is_K else shipped_value
    idx = vals.index(key)
    losses = [c['loss'] for c in curve]
    amin = int(np.argmin(losses))
    finite = [i for i, v in enumerate(vals) if not (isinstance(v, str) and v == 'inf')]
    at_edge = amin == finite[0] or amin == finite[-1] or amin not in finite
    d = losses[idx] - losses[amin]
    # SE of shipped vs argmin: reuse d_vs_shipped SE of the argmin entry
    se = curve[amin]['se'] if amin != idx else 0.0
    if amin == idx:
        cls = 'INTERIOR optimum' if not at_edge else 'EDGE (argmin at grid edge)'
    elif d <= se:
        cls = 'SHOWN FLAT (within 1 SE of argmin)'
    else:
        cls = 'WORSE than argmin'
    if at_edge and amin != idx:
        cls += ' — curve argmin at grid EDGE'
        if is_K and vals[amin] == '0':
            cls += ' (K=0 is the natural boundary: no shrinkage)'
    return {'class': cls, 'argmin_value': vals[amin], 'delta_vs_argmin': d,
            'se': se, 't': (d / se) if se > 0 else 0.0,
            'argmin_wins_vs_shipped': curve[amin]['wins_vs_shipped'] if amin != idx else None}


def fmt_curve(cur, keyname):
    lines = [f"| {keyname} | mean loss | d vs shipped | SE | t | wins vs shipped |",
             "|---|---|---|---|---|---|"]
    for c in cur:
        extra = f" (K {c['best_K']})" if 'best_K' in c else ''
        lines.append(f"| {c['value']}{extra} | {c['loss']:.6f} | {c['d_vs_shipped']:+.6f} | "
                     f"{c['se']:.6f} | {c['t']:+.2f} | {c['wins_vs_shipped']}/30 |")
    return '\n'.join(lines)


def cmd_report():
    sw = pickle.load(open(SWEEP_PKL, 'rb'))
    rep = analyse(sw)

    def _j(o):
        if isinstance(o, dict):
            return {str(k): _j(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_j(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    json.dump(_j(rep), open(OUT_JSON, 'w'), indent=1)
    md = [f"# Loc+ cell-fit sweep (2026-09-02)\n",
          f"Seasons {rep['seasons']}, {rep['pairs']} ordered train->test pairs. "
          "Loss per event: log loss (nats) for probability grids, MSE in standardised "
          "xwOBA units for the contact grid. SE = paired SE across the 30 pairs; "
          "se_cluster = SE across the 6 test seasons (pairs share seasons).\n"]
    md.append("## Season inputs\n\n| season | rows | baseline | keys | dates | bunt BIP note |\n|---|---|---|---|---|---|")
    for yr, m in rep['meta'].items():
        note = (f"excluded {m.get('bunt_bip_excluded')}" if m.get('bunt_bip_detectable')
                else f"NOT detectable, {m.get('bunt_bip_in_source')} in source")
        md.append(f"| {yr} | {m['rows']} | {m['n_base']} | {m['keys']} | {m['dates']} | {note} |")
    md.append("")
    for surf, r in rep['surfaces'].items():
        md.append(f"## {surf} ({r['loss_kind']}; constants {', '.join(r['constants'])})\n")
        s = r['shipped_minus_argmin']
        md.append(f"- argmin: `{r['argmin']}` loss {r['argmin_loss']:.6f}")
        md.append(f"- shipped: `{r['shipped']}` loss {r['shipped_loss']:.6f} "
                  f"(shipped - argmin {s['mean']:+.6f}, SE {s['se']:.6f}, cluster SE {s['se_cluster']:.6f}, "
                  f"t {s['t']:+.2f}; argmin wins {r['argmin_wins_vs_shipped']}/30)")
        md.append(f"- prior-only (K=inf) loss at shipped bandwidth: {r['prior_only_loss_at_shipped_bw']:.6f}")
        md.append(f"- flat region (within 1 SE of argmin): {r['flat_region_n']} configs; "
                  f"bx in {r['flat_bx']}, bz in {r['flat_bz']}")
        md.append("- claim classes at the shipped operating point:")
        for n, c in r['classes'].items():
            md.append(f"  - {n}: {c['class']}; curve argmin {c['argmin_value']}, "
                      f"delta {c['delta_vs_argmin']:+.6f}, t {c['t']:+.2f}")
        for n in r['constants']:
            md.append(f"\n### {n} marginal (others at shipped)\n\n" + fmt_curve(r['curves'][n], n))
        md.append("\n### bx (bz shipped): K fixed at shipped\n\n" + fmt_curve(r['curves']['bx_fixedK'], 'bx'))
        md.append("\n### bx (bz shipped): K profiled\n\n" + fmt_curve(r['curves']['bx_profiled'], 'bx'))
        md.append("\n### bz (bx shipped): K fixed at shipped\n\n" + fmt_curve(r['curves']['bz_fixedK'], 'bz'))
        md.append("\n### bz (bx shipped): K profiled\n\n" + fmt_curve(r['curves']['bz_profiled'], 'bz'))
        md.append("\n### joint (bx, bz) with best K, excess vs argmin in SE units\n")
        md.append("| bx | bz | best K | loss | excess | SE units |\n|---|---|---|---|---|---|")
        for name, v in sorted(r['bw_profile'].items(), key=lambda kv: kv[1]['loss']):
            bx, bz = name.split('/')[0][2:], name.split('/')[1][2:]
            md.append(f"| {bx} | {bz} | {v['best_K']} | {v['loss']:.6f} | "
                      f"{v['excess_vs_argmin']:+.6f} | {v['excess_se_units']:.2f} |")
        md.append("")
    md.append("## Common bandwidth (max excess over surfaces, SE units, best K per surface)\n")
    md.append("| bx | bz | max excess SE | sum | " + ' | '.join(SURFACES) + " |\n|---|---|---|---|" + '---|' * len(SURFACES))
    for row in rep['common_bandwidth'][:15]:
        md.append(f"| {row['bx']:g} | {row['bz']:g} | {row['max_excess_se']:.2f} | {row['sum_excess_se']:.2f} | "
                  + ' | '.join(f"{row['per_surface'][s]:.2f}" for s in SURFACES) + " |")
    ship = next(r for r in rep['common_bandwidth'] if r['bx'] == SHIPPED['bx'] and r['bz'] == SHIPPED['bz'])
    md.append(f"\nShipped (4.5, 0.22): max excess {ship['max_excess_se']:.2f} SE, "
              + ', '.join(f"{s} {ship['per_surface'][s]:.2f}" for s in SURFACES))
    md.append("\n## CS count transform reference (shipped bandwidth, loss with transform OFF)\n")
    md.append("| K_CS | loss (no transform) |\n|---|---|")
    for k, v in rep['cs_notransform_at_shipped'].items():
        md.append(f"| {k} | {v:.6f} |")
    md.append(f"\nTest pitches dropped for keys unseen in train: {rep['dropped_pitches']}")
    open(OUT_MD, 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))
    print(f"\nwrote {OUT_JSON} and {OUT_MD}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['accumulate', 'sweep', 'report'])
    ap.add_argument('season', nargs='?', type=int)
    ap.add_argument('--seasons', default=None)
    ap.add_argument('--bx', default=None)
    ap.add_argument('--bz', default=None)
    a = ap.parse_args()
    if a.cmd == 'accumulate':
        if a.season is None:
            raise SystemExit('accumulate needs a season')
        cmd_accumulate(a.season)
    elif a.cmd == 'sweep':
        cmd_sweep(a)
    else:
        cmd_report()


if __name__ == '__main__':
    main()
