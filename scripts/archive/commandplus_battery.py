"""commandplus_battery.py — Command+ SEP sweep + multi-season validation + Kirby benchmark.

THE STANDARD (feedback_multiseason_defensibility): every season rebuilt
self-contained, never pooled; the headline objective is INTER-SEASON
persistence — the property that motivated the metric (target: xCTRL's
published 0.65; the bar to clear: Loc+'s ~0.4).

STAGE A — per season (2021-2025 raw caches via the adapter, 2026 cache):
  Command+ scored at every SEP in the sweep grid (GMM fit once per cell,
  merge applied per SEP — see score_pitches_multi), full season + odd/even
  halves; plus per-pitcher Loc+ raw, FF velo, and actual xRV allowed.

STAGE B — objectives per (sep, variant in {mean, median}):
  rel     split-half r within each season
  persist year-pair r (21-22 ... 24-25). The 25-26 pair is reported
          SEPARATELY: 2026 is retagged and partial, so it is a caveat pair.
  indep   corr vs Loc+ raw and FF velo per season
  partial partial r(miss_N, xRV_{N+1} | loc_N) — does command add anything
          beyond location value in predicting next-season run prevention?

STAGE C — Kirby Index benchmark, 2021-2025 (raw caches carry vx0/vy0/vz0):
  release angles back-extrapolated from the y0=50ft state to the release
  point (Nathan kinematics); per (pitcher, type) >= 50 pitches: mean of
  SD(VRA), SD(HRA); pitcher score = usage-weighted across types (LOWER =
  more repeatable). Same reliability + persistence battery, head-to-head.

SEP GRID includes 0 (no merge guard at all): if 0 wins, the circularity
guard is unnecessary; if the curve is monotone to 16, the grid extends.

Usage: python3 scripts/commandplus_battery.py
"""
import os, sys, math, pickle, gc, time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np
import pipeline_locplus as lp
from pipeline_sdplus import make_rv_xrv
from locplus_constants_multiseason import adapt
from commandplus_v1 import score_pitches_multi, aggregate

LG, SCALE = 0.3169, 1.2393
SEPS = [0.0, 4.0, 8.0, 12.0, 16.0]
MIN_FULL, MIN_HALF = 300, 150
CACHE = {2021: 'data/_statcast2021_cache.pkl', 2022: 'data/_statcast2022_cache.pkl',
         2023: 'data/_statcast2023_cache.pkl', 2024: 'data/_statcast2024_cache.pkl',
         2025: 'data/_statcast2025_full_cache.pkl'}
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n; my = sum(ys) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if sx <= 0 or sy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)


def partial(r_xy, r_xz, r_zy):
    d = (1 - r_xz ** 2) * (1 - r_zy ** 2)
    return (r_xy - r_xz * r_zy) / math.sqrt(d) if d > 0 else None


def season_pitches(year):
    if year == 2026:
        D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
        ep = {(p.get('Pitcher'), p.get('PTeam')) for p in D if p.get('Pitch Type') == 'EP'}
        return [p for p in D if p.get('_source', 'MLB') == 'MLB'
                and (p.get('Pitcher'), p.get('PTeam')) not in ep]
    return adapt(os.path.join(ROOT, CACHE[year]))


def process_season(year):
    t0 = time.time()
    pitches = season_pitches(year)
    dates = sorted({p.get('Game Date') for p in pitches if p.get('Game Date')})
    par = {d: i % 2 for i, d in enumerate(dates)}

    res = {}
    multi = score_pitches_multi(pitches, SEPS)
    res['full'] = {s: aggregate(multi[s][0], MIN_FULL) for s in SEPS}
    for h in (0, 1):
        sub = [p for p in pitches if par.get(p.get('Game Date')) == h]
        mh = score_pitches_multi(sub, SEPS)
        res[f'half{h}'] = {s: aggregate(mh[s][0], MIN_HALF) for s in SEPS}
        del mh
        gc.collect()

    base = [p for p in pitches if lp.is_eligible_baseline(p)]
    S = lp.build_surfaces(base, LG, SCALE)
    acc = defaultdict(list)
    for p in base:
        v = lp.score_pitch(p, S)
        if v is not None:
            acc[(p.get('Pitcher'), p.get('Throws'))].append(v)
    res['loc'] = {k: sum(v) / len(v) for k, v in acc.items() if len(v) >= MIN_FULL}

    velo = defaultdict(list)
    for p in pitches:
        if p.get('Pitch Type') == 'FF':
            v = lp.safe_float(p.get('Velocity'))
            if v is not None:
                velo[(p.get('Pitcher'), p.get('Throws'))].append(v)
    res['velo'] = {k: sum(v) / len(v) for k, v in velo.items() if len(v) >= 50}

    rv_fn = make_rv_xrv(LG, SCALE)
    acc = defaultdict(list)
    for p in pitches:
        v = rv_fn(p)
        if v is not None:
            acc[(p.get('Pitcher'), p.get('Throws'))].append(v)
    res['xrv'] = {k: sum(v) / len(v) for k, v in acc.items() if len(v) >= MIN_FULL}

    print(f"  {year}: {len(pitches)} pitches, {len(res['full'][8.0])} pitchers "
          f">= {MIN_FULL} at sep=8 ({time.time()-t0:.0f}s)", flush=True)
    del pitches, multi, base, S
    gc.collect()
    return res


def kirby_season(year):
    """Release-angle repeatability from the raw cache kinematics."""
    import pandas as pd
    df = pickle.load(open(os.path.join(ROOT, CACHE[year]), 'rb'))
    cols = ['player_name', 'p_throws', 'pitch_type', 'game_date',
            'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'release_extension']
    d = df[cols].dropna().copy()
    y_rel = 60.5 - d['release_extension'].astype(float)
    vy0 = d['vy0'].astype(float)
    # back-extrapolate the y0=50ft state to release (Nathan kinematics)
    vy_rel = -np.sqrt(vy0 ** 2 + 2 * d['ay'].astype(float) * (y_rel - 50.0))
    t_back = (vy0 - vy_rel) / d['ay'].astype(float)
    vx_rel = d['vx0'].astype(float) - d['ax'].astype(float) * t_back
    vz_rel = d['vz0'].astype(float) - d['az'].astype(float) * t_back
    d['vra'] = np.degrees(np.arctan2(vz_rel, -vy_rel))
    d['hra'] = np.degrees(np.arctan2(vx_rel, -vy_rel))
    d['half'] = (pd.factorize(d['game_date'].astype(str).str[:10], sort=True)[0] % 2)

    def scores(frame, min_pt=50, min_total=300):
        g = frame.groupby(['player_name', 'p_throws', 'pitch_type']).agg(
            sv=('vra', 'std'), sh=('hra', 'std'), n=('vra', 'size')).reset_index()
        g = g[g['n'] >= min_pt]
        g['k'] = (g['sv'] + g['sh']) / 2.0
        out = {}
        for (pit, th), sub in g.groupby(['player_name', 'p_throws']):
            n = sub['n'].sum()
            if n >= min_total:
                out[(pit, th)] = float((sub['k'] * sub['n']).sum() / n)
        return out

    full = scores(d)
    h0 = scores(d[d['half'] == 0], min_total=150)
    h1 = scores(d[d['half'] == 1], min_total=150)
    del df, d
    gc.collect()
    return {'full': full, 'h0': h0, 'h1': h1}


def main():
    data = {}
    print("STAGE A — per-season Command+ scoring")
    for y in SEASONS:
        data[y] = process_season(y)

    print()
    print("STAGE B1 — split-half reliability by (sep, variant)")
    print(f"{'sep':>5s} {'var':>7s} " + "".join(f"{y:>8d}" for y in SEASONS) + f"{'mean':>8s}")
    print('-' * 70)
    for s in SEPS:
        for vi, vn in ((0, 'mean'), (1, 'median')):
            rels = []
            row = f"{s:>5.0f} {vn:>7s} "
            for y in SEASONS:
                a, b = data[y]['half0'][s], data[y]['half1'][s]
                keys = [k for k in a if k in b]
                r = pearson([a[k][vi] for k in keys], [b[k][vi] for k in keys])
                rels.append(r)
                row += f"{r:>8.3f}"
            row += f"{sum(rels)/len(rels):>8.3f}"
            print(row, flush=True)

    print()
    print("STAGE B2 — INTER-SEASON PERSISTENCE (headline) by (sep, variant)")
    pairs = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
    print(f"{'sep':>5s} {'var':>7s} " + "".join(f"{a}-{str(b)[2:]:>3s}" for a, b in pairs).rjust(32)
          + f"{'mean':>8s} {'25-26*':>8s}")
    print('-' * 78)
    for s in SEPS:
        for vi, vn in ((0, 'mean'), (1, 'median')):
            row = f"{s:>5.0f} {vn:>7s} "
            ps = []
            for a, b in pairs:
                fa, fb = data[a]['full'][s], data[b]['full'][s]
                keys = [k for k in fa if k in fb]
                r = pearson([fa[k][vi] for k in keys], [fb[k][vi] for k in keys])
                ps.append(r)
                row += f"{r:>8.3f}"
            fa, fb = data[2025]['full'][s], data[2026]['full'][s]
            keys = [k for k in fa if k in fb]
            r2526 = pearson([fa[k][vi] for k in keys], [fb[k][vi] for k in keys])
            row += f"{sum(ps)/len(ps):>8.3f}{r2526:>8.3f}"
            print(row, flush=True)
    print("(* 25-26 is a caveat pair: 2026 is retagged and partial)")

    print()
    print("STAGE B3 — independence & partial predictive value (sep=8, mean; "
          "re-check at the winning sep)")
    s = 8.0
    print(f"{'year':>6s} {'r(loc)':>8s} {'r(velo)':>8s}")
    for y in SEASONS:
        f = data[y]['full'][s]
        kk = [k for k in f if k in data[y]['loc']]
        rl = pearson([f[k][0] for k in kk], [data[y]['loc'][k] for k in kk])
        kv = [k for k in f if k in data[y]['velo']]
        rv = pearson([f[k][0] for k in kv], [data[y]['velo'][k] for k in kv])
        print(f"{y:>6d} {rl:>+8.3f} {rv:>+8.3f}")
    print()
    print("partial r(miss_N -> xRV_{N+1} | loc_N)  [positive = bigger misses,")
    print("more runs allowed next year, beyond what Loc+ already knew]")
    for a, b in pairs:
        f = data[a]['full'][s]
        kk = [k for k in f if k in data[a]['loc'] and k in data[b]['xrv']]
        if len(kk) < 40:
            continue
        miss = [f[k][0] for k in kk]
        locv = [data[a]['loc'][k] for k in kk]
        nxt = [data[b]['xrv'][k] for k in kk]
        r_my = pearson(miss, nxt); r_ml = pearson(miss, locv); r_ly = pearson(locv, nxt)
        print(f"  {a}->{b}: partial {partial(r_my, r_ml, r_ly):+.3f} "
              f"(raw {r_my:+.3f}, loc alone {r_ly:+.3f}, n={len(kk)})")

    print()
    print("STAGE C — Kirby Index benchmark (release-angle repeatability)")
    kb = {}
    for y in range(2021, 2026):
        kb[y] = kirby_season(y)
        print(f"  {y}: {len(kb[y]['full'])} pitchers", flush=True)
    rels, pers = [], []
    for y in range(2021, 2026):
        a, b = kb[y]['h0'], kb[y]['h1']
        keys = [k for k in a if k in b]
        rels.append(pearson([a[k] for k in keys], [b[k] for k in keys]))
    for a, b in pairs:
        fa, fb = kb[a]['full'], kb[b]['full']
        keys = [k for k in fa if k in fb]
        pers.append(pearson([fa[k] for k in keys], [fb[k] for k in keys]))
    print(f"  Kirby split-half by season: " + ", ".join(f"{r:.3f}" for r in rels))
    print(f"  Kirby year-pair persistence: " + ", ".join(f"{r:.3f}" for r in pers)
          + f"   mean {sum(pers)/len(pers):.3f}")
    # overlap with the miss-based metric, same season
    for y in (2024, 2025):
        f = data[y]['full'][8.0]
        kk = [k for k in f if k in kb[y]['full']]
        r = pearson([f[k][0] for k in kk], [kb[y]['full'][k] for k in kk])
        print(f"  corr(mean-miss, Kirby) {y}: {r:+.3f} (n={len(kk)})")

    print()
    print("READ: pick (sep, variant) on persistence primarily, reliability")
    print("second; sep must be interior or flat, else extend the grid. If")
    print("Kirby decisively beats the miss metric on persistence, the")
    print("kinematics sidecar becomes worth its cost.")


if __name__ == '__main__':
    main()
