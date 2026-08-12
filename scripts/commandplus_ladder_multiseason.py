"""Command+ multi-season battery: scorer ladder + volume-bias replication.

Two questions, one harness (both need per-season scoring, so they share it).

(1) LADDER — does the cell/GMM machinery earn its keep?  Four nested scorers,
    each run through the same objectives the metric was validated on:
      GLOBAL    mean distance to the pitcher's single global centroid.
                No cells, no GMM, no MIN_CELL.  The "one-line dispersion
                stat" null: if this ties the production scorer, everything
                downstream is decoration.
      CELLMEAN  production cells, K forced to 1 (distance to the cell mean).
                Isolates what the CELLS buy over the global centroid.
      PROD      production: K=1..3 by BIC, K capped by cell size (30/60).
                Isolates what the MIXTURE buys over one target per cell.
      KFREE     K=1..6 by BIC, no sample caps.  Tests whether the caps are
                leaving signal on the table or preventing overfit.

(2) VOLUME — does the low-volume flattery found in 2026
    (scripts/commandplus_volume_test.py: -0.105" at N=400) replicate on
    independent seasons?  Per the multi-season standard, a bias curve fitted
    to one season is not a constant.

METHODOLOGY mirrors the validated battery (commit 5518ec4,
scripts/commandplus_battery.py, since deleted):
  - split-half = ALTERNATING GAME DATES (not random pitches), targets fit per
    half independently, MIN_HALF=150, MIN_FULL=300
  - per-season replicates, NEVER pooled
  - pitcher key = (name, throws); MLB only; position players (any EP) dropped

Usage: python3 scripts/commandplus_ladder_multiseason.py
"""
import gc
import math
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_commandplus import MIN_CELL, count_group, is_eligible
from pipeline_utils import safe_float

# ── RETIRED v1 TARGET FITTER, vendored ──────────────────────────────────
# The 1-3 component GMM below WAS pipeline_commandplus.fit_targets until this
# script's own result retired it (see the PROD/KFREE rows).  It is kept here,
# and only here, so the ladder that killed it stays runnable — production is
# now K=1 and carries none of this.
REG_COVAR = 1e-3
EM_TOL = 1e-4
EM_MAX_ITER = 200


def _mean2(pts):
    n = len(pts)
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)


def _cov2(pts, mx, my):
    n = len(pts)
    sxx = syy = sxy = 0.0
    for x, y in pts:
        dx = x - mx; dy = y - my
        sxx += dx * dx; syy += dy * dy; sxy += dx * dy
    return (sxx / n + REG_COVAR, syy / n + REG_COVAR, sxy / n)


def _logpdf2(x, y, mx, my, cxx, cyy, cxy):
    det = cxx * cyy - cxy * cxy
    if det <= 1e-12:
        det = 1e-12
    dx = x - mx; dy = y - my
    q = (cyy * dx * dx - 2.0 * cxy * dx * dy + cxx * dy * dy) / det
    return -0.918938533204673 * 2 - 0.5 * math.log(det) - 0.5 * q


def _farthest_point_init(pts, k):
    seeds = [_mean2(pts)]
    while len(seeds) < k:
        best_d, best_p = -1.0, None
        for x, y in pts:
            d = min((x - sx) ** 2 + (y - sy) ** 2 for sx, sy in seeds)
            if d > best_d:
                best_d, best_p = d, (x, y)
        seeds.append(best_p)
    return seeds


def _em_fit(pts, k):
    n = len(pts)
    means = _farthest_point_init(pts, k)
    covs = [_cov2(pts, mx, my) for mx, my in means]
    weights = [1.0 / k] * k
    prev_ll = None
    resp = [[0.0] * k for _ in range(n)]
    for _it in range(EM_MAX_ITER):
        ll = 0.0
        for i, (x, y) in enumerate(pts):
            row = resp[i]
            mx_l = None
            for j in range(k):
                m = means[j]; c = covs[j]
                lp = (math.log(weights[j] + 1e-300)
                      + _logpdf2(x, y, m[0], m[1], c[0], c[1], c[2]))
                row[j] = lp
                if mx_l is None or lp > mx_l:
                    mx_l = lp
            s = 0.0
            for j in range(k):
                row[j] = math.exp(row[j] - mx_l)
                s += row[j]
            for j in range(k):
                row[j] /= s
            ll += mx_l + math.log(s)
        if prev_ll is not None and abs(ll - prev_ll) < EM_TOL * n:
            prev_ll = ll
            break
        prev_ll = ll
        for j in range(k):
            nj = sum(resp[i][j] for i in range(n))
            if nj < 1e-8:
                worst_i = max(range(n), key=lambda i: -max(resp[i]))
                means[j] = pts[worst_i]
                covs[j] = _cov2(pts, means[j][0], means[j][1])
                weights[j] = 1.0 / n
                continue
            mx = sum(resp[i][j] * pts[i][0] for i in range(n)) / nj
            my = sum(resp[i][j] * pts[i][1] for i in range(n)) / nj
            sxx = syy = sxy = 0.0
            for i in range(n):
                r = resp[i][j]
                dx = pts[i][0] - mx; dy = pts[i][1] - my
                sxx += r * dx * dx; syy += r * dy * dy; sxy += r * dx * dy
            means[j] = (mx, my)
            covs[j] = (sxx / nj + REG_COVAR, syy / nj + REG_COVAR, sxy / nj)
            weights[j] = nj / n
    return means, weights, prev_ll


def _bic(loglik, k, n):
    p = 6 * k - 1
    return -2.0 * loglik + p * math.log(n)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = {2021: 'data/_statcast2021_cache.pkl', 2022: 'data/_statcast2022_cache.pkl',
         2023: 'data/_statcast2023_cache.pkl', 2024: 'data/_statcast2024_cache.pkl',
         2025: 'data/_statcast2025_full_cache.pkl'}
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
MIN_FULL, MIN_HALF = 300, 150

SCORERS = ['GLOBAL', 'CELLMEAN', 'PROD', 'KFREE']
PROD_CAPS = ((60, 3), (30, 2), (0, 1))
FREE_KMAX = 6

N_GRID = [400, 600, 900, 1300]
VOL_SEEDS = 5
VOL_MIN_FULL = 1400


# ═══════════════════════════════ scorers ═══════════════════════════════
def _targets(pts, mode):
    """Target set for one cell under a given scorer."""
    n = len(pts)
    if mode == 'CELLMEAN':
        return [(sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)]
    kmax = (next(k for thr, k in PROD_CAPS if n >= thr) if mode == 'PROD'
            else min(FREE_KMAX, max(1, n // 10)))
    best, best_b = None, None
    for k in range(1, kmax + 1):
        means, _w, ll = _em_fit(pts, k)
        b = _bic(ll, k, n)
        if best_b is None or b < best_b:
            best, best_b = means, b
    return best


def score(pitches, mode):
    """pitches = [(pt, bats, cgroup, x_in, z_in), ...] -> (mean miss, n)."""
    if mode == 'GLOBAL':
        n = len(pitches)
        if n < MIN_CELL:
            return None, 0
        mx = sum(p[3] for p in pitches) / n
        mz = sum(p[4] for p in pitches) / n
        return sum(math.hypot(p[3] - mx, p[4] - mz) for p in pitches) / n, n
    cells = defaultdict(list)
    for p in pitches:
        cells[(p[0], p[1], p[2])].append((p[3], p[4]))
    total, n_tot = 0.0, 0
    for pts in cells.values():
        if len(pts) < MIN_CELL:
            continue
        tg = _targets(pts, mode)
        for x, z in pts:
            total += min(math.hypot(x - tx, z - tz) for tx, tz in tg)
            n_tot += 1
    return (total / n_tot, n_tot) if n_tot else (None, 0)


# ═══════════════════════════════ workers ═══════════════════════════════
def ladder_job(arg):
    key, pitches = arg
    out = {}
    for mode in SCORERS:
        full, nf = score(pitches, mode)
        if full is not None and nf >= MIN_FULL:
            out[(mode, 'full')] = full
        for h in (0, 1):
            sub = [p for p in pitches if p[5] == h]
            m, nh = score(sub, mode)
            if m is not None and nh >= MIN_HALF:
                out[(mode, f'h{h}')] = m
    return key, out


def volume_job(arg):
    key, pitches = arg
    full, _ = score(pitches, 'PROD')
    if full is None:
        return None
    out = {}
    for N in N_GRID:
        if N >= len(pitches):
            continue
        vals = []
        for s in range(VOL_SEEDS):
            rng = random.Random(hash((key, N, s)) & 0xFFFFFFFF)
            m, _n = score(rng.sample(pitches, N), 'PROD')
            if m is not None:
                vals.append(m)
        if vals:
            out[N] = sum(vals) / len(vals) - full
    return key, full, out


# ═══════════════════════════════ loading ═══════════════════════════════
def load_season(year):
    """-> (by_pitcher, bb_rate, zone).  by_pitcher[(name, throws)] = list of
    (pt, bats, cgroup, x_in, z_in, date_parity); zone = (bot_in, top_in), the
    season's mean called-zone extent, carried so downstream target-geometry
    work does not have to re-read the cache."""
    zt = zb = 0.0
    nz = 0
    if year == 2026:
        import pandas as pd
        D = pd.read_pickle(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'))
        ep = {(p.get('Pitcher'), p.get('PTeam')) for p in D if p.get('Pitch Type') == 'EP'}
        rows, bb, pa = [], defaultdict(int), defaultdict(int)
        for p in D:
            if p.get('_source', 'MLB') != 'MLB' or (p.get('Pitcher'), p.get('PTeam')) in ep:
                continue
            k = (p.get('Pitcher'), p.get('Throws'))
            _t, _b = safe_float(p.get('SzTop')), safe_float(p.get('SzBot'))
            if _t and _b:
                zt += _t; zb += _b; nz += 1
            ev = p.get('Event')
            if ev:
                pa[k] += 1
                if ev == 'Walk':
                    bb[k] += 1
            if is_eligible(p):
                rows.append((k, p['Pitch Type'], p['Bats'], count_group(p['Count']),
                             safe_float(p['PlateX']) * 12.0,
                             safe_float(p['PlateZ']) * 12.0, str(p['Game Date'])[:10]))
        del D
    else:
        df = pickle.load(open(os.path.join(ROOT, CACHE[year]), 'rb'))
        cols = ['pitch_type', 'plate_x', 'plate_z', 'balls', 'strikes', 'description',
                'stand', 'p_throws', 'player_name', 'game_date', 'events',
                'sz_top', 'sz_bot']
        sub = df[cols]
        del df
        gc.collect()
        from pipeline_commandplus import EXCLUDE_DESC, EXCLUDE_PT, HANDS
        # statcast description -> the pipeline's Description vocabulary, only
        # far enough to apply the same exclusions (HBP / bunts / pitchouts).
        DROP_DESC = {'hit_by_pitch', 'foul_bunt', 'missed_bunt', 'bunt_foul_tip',
                     'pitchout', 'swinging_pitchout', 'foul_pitchout'}
        rows, bb, pa = [], defaultdict(int), defaultdict(int)
        for r in sub.itertuples(index=False):
            k = (r.player_name, r.p_throws)
            try:
                _t, _b = float(r.sz_top), float(r.sz_bot)
                if _t == _t and _b == _b:
                    zt += _t; zb += _b; nz += 1
            except (TypeError, ValueError):
                pass
            ev = r.events
            if isinstance(ev, str) and ev:
                pa[k] += 1
                if ev == 'walk':
                    bb[k] += 1
            if r.description in DROP_DESC or ev == 'intent_walk':
                continue
            pt = r.pitch_type
            if not isinstance(pt, str) or pt in EXCLUDE_PT:
                continue
            if r.stand not in HANDS or r.p_throws not in HANDS:
                continue
            # pandas NA breaks safe_float's `val == ''` test; guarded cast,
            # same reason locplus_constants_multiseason.adapt() uses one.
            try:
                x, z = float(r.plate_x), float(r.plate_z)
            except (TypeError, ValueError):
                continue
            if x != x or z != z:
                continue
            try:
                b, s = int(r.balls), int(r.strikes)
            except (TypeError, ValueError):
                continue
            cg = count_group(f'{b}-{s}')
            if cg is None:
                continue
            rows.append((k, pt, r.stand, cg, x * 12.0, z * 12.0,
                         str(r.game_date)[:10]))
        del sub
    gc.collect()

    dates = sorted({r[6] for r in rows})
    par = {d: i % 2 for i, d in enumerate(dates)}
    by_p = defaultdict(list)
    for k, pt, st, cg, x, z, d in rows:
        by_p[k].append((pt, st, cg, x, z, par[d]))
    del rows
    gc.collect()
    bb_rate = {k: bb[k] / pa[k] for k in pa if pa[k] >= 100}
    zone = (12.0 * zb / nz, 12.0 * zt / nz) if nz else (18.6, 40.8)
    return dict(by_p), bb_rate, zone


# ═══════════════════════════════ stats ═══════════════════════════════
def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if sx <= 0 or sy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)


def paired(d1, d2):
    ks = [k for k in d1 if k in d2]
    return [d1[k] for k in ks], [d2[k] for k in ks], len(ks)


def fmt(v, w=8, p=3):
    return f'{v:>{w}.{p}f}' if v is not None else f'{"--":>{w}}'


# ═══════════════════════════════ main ═══════════════════════════════
def main():
    full = {m: {} for m in SCORERS}
    half = {m: {} for m in SCORERS}
    bbr, vol, sig, pool_note = {}, {}, {}, {}

    for y in SEASONS:
        t0 = time.time()
        by_p, bb_rate, _zone = load_season(y)
        bbr[y] = bb_rate
        jobs = [(k, v) for k, v in by_p.items() if len(v) >= MIN_FULL]

        with Pool() as pool:
            res = pool.map(ladder_job, jobs, chunksize=4)

        # COMMON POOL.  GLOBAL scores every pitch; the cell scorers drop
        # sub-MIN_CELL cells, so each scorer clears the MIN_FULL/MIN_HALF
        # gate on a different set of pitchers.  Comparing them on their own
        # pools would confound scorer quality with pool composition, so every
        # objective runs on the pitchers where ALL FOUR produce a value.
        c_full = set.intersection(*[{k for k, o in res if (m, 'full') in o}
                                    for m in SCORERS])
        c_half = set.intersection(*[{k for k, o in res
                                     if (m, 'h0') in o and (m, 'h1') in o}
                                    for m in SCORERS])
        for m in SCORERS:
            full[m][y] = {k: o[(m, 'full')] for k, o in res
                          if k in c_full and (m, 'full') in o}
            half[m][y] = {k: (o[(m, 'h0')], o[(m, 'h1')]) for k, o in res
                          if k in c_half and (m, 'h0') in o and (m, 'h1') in o}
        pool_note[y] = (len(jobs), len(c_full), len(c_half))

        p = list(full['PROD'][y].values())
        mu = sum(p) / len(p)
        sig[y] = math.sqrt(sum((x - mu) ** 2 for x in p) / len(p))

        vjobs = [(k, v) for k, v in by_p.items() if len(v) >= VOL_MIN_FULL]
        with Pool() as pool:
            vres = [r for r in pool.map(volume_job, vjobs, chunksize=2) if r]
        vol[y] = vres

        print(f'{y}: {len(jobs)} candidates -> common pool {len(c_full)} full / '
              f'{len(c_half)} half, {len(vjobs)} in volume cohort, '
              f'sigma {sig[y]:.3f}" ({time.time() - t0:.0f}s)', flush=True)
        del by_p
        gc.collect()

    hdr = ''.join(f'{y:>8d}' for y in SEASONS)

    print('\n' + '=' * 78)
    print('LADDER 1/3 — SPLIT-HALF RELIABILITY (alternating game dates)')
    print('=' * 78)
    print(f'{"scorer":<10}{hdr}{"mean":>8}')
    for m in SCORERS:
        rs = []
        line = f'{m:<10}'
        for y in SEASONS:
            a, b, n = paired({k: v[0] for k, v in half[m][y].items()},
                             {k: v[1] for k, v in half[m][y].items()})
            r = pearson(a, b)
            line += fmt(r)
            if r is not None:
                rs.append(r)
        print(line + fmt(sum(rs) / len(rs) if rs else None))

    print('\n' + '=' * 78)
    print('LADDER 2/3 — INTER-SEASON PERSISTENCE (year N -> N+1)')
    print('=' * 78)
    pairs = list(zip(SEASONS, SEASONS[1:]))
    print(f'{"scorer":<10}' + ''.join(f'{f"{a % 100}->{b % 100}":>8}' for a, b in pairs)
          + f'{"mean":>8}{"clean4":>8}')
    for m in SCORERS:
        rs, line, clean = [], f'{m:<10}', []
        for a, b in pairs:
            xs, ys, n = paired(full[m][a], full[m][b])
            r = pearson(xs, ys)
            line += fmt(r)
            if r is not None:
                rs.append(r)
                if b != 2026:
                    clean.append(r)
        print(line + fmt(sum(rs) / len(rs) if rs else None)
              + fmt(sum(clean) / len(clean) if clean else None))

    print('\n' + '=' * 78)
    print('LADDER 3/3 — WALK-RATE CORRELATION (same season, and N -> BB% N+1)')
    print('=' * 78)
    print(f'{"scorer":<10}{"":>2}{hdr}{"mean":>8}')
    for m in SCORERS:
        rs, line = [], f'{m:<10}{"":>2}'
        for y in SEASONS:
            xs, ys, n = paired(full[m][y], bbr[y])
            r = pearson(xs, ys)
            line += fmt(r)
            if r is not None:
                rs.append(r)
        print(line + fmt(sum(rs) / len(rs) if rs else None) + '   same-season')
    for m in SCORERS:
        rs, line = [], f'{m:<10}{"":>2}'
        for a, b in pairs:
            xs, ys, n = paired(full[m][a], bbr[b])
            r = pearson(xs, ys)
            line += fmt(r)
            if r is not None:
                rs.append(r)
        print(line + f'{"":>8}' + fmt(sum(rs) / len(rs) if rs else None)
              + '   next-season')

    print('\n' + '=' * 78)
    print('VOLUME BIAS REPLICATION  (delta vs full sample, PROD scorer)')
    print('=' * 78)
    print(f'{"season":<8}{"n":>5}{"sigma":>8}' +
          ''.join(f'{f"N={N}":>18}' for N in N_GRID))
    print(f'{"":<8}{"":>5}{"":>8}' + ''.join(f'{"inches / Cmd+ pts":>18}' for _ in N_GRID))
    for y in SEASONS:
        line = f'{y:<8}{len(vol[y]):>5}{sig[y]:>8.3f}'
        for N in N_GRID:
            ds = [o[N] for _k, _f, o in vol[y] if N in o]
            if ds:
                d = sum(ds) / len(ds)
                line += f'{d:>+9.3f}" {-d * 10 / sig[y]:>+7.2f}'
            else:
                line += f'{"--":>18}'
        print(line)


if __name__ == '__main__':
    main()
