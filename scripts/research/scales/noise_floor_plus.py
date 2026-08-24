#!/usr/bin/env python3
"""noise_floor_plus.py — split-half noise floors for the six non-Stuff+
"plus" metrics: Loc+, Command+, BB+, SD+, CT+, Hitter+ (2026-08-23).

Question: how many points of each metric is measurement noise, so a reader
knows the smallest quotable difference (the Stuff+ answer was ~1-1.5 pts at
qualified samples; this draws the same line for the rest).

Method: per player, split the 2026 season by alternating GAME DATES
(odd/even in date order), recompute the DISPLAYED metric on each half with
the production scorers against SEASON-FIXED anchors and tables:
  * BB+/SD+/CT+/Hitter+: pipeline.window_pool.score_window_against_season —
    the shipped card-window path (values from the half, anchors from the
    season's metadata).
  * Loc+: pipeline.locplus surfaces built on the full season baseline, half
    raw means normalized against the full-season pool with the production
    n_prior shrink.
  * Command+: pipeline.commandplus cells built on the full season, half
    misses scored against them, 200 - 100*miss/lg_miss with the SEASON
    league miss.
Then: split-half r over players; full-season reliability by Spearman-Brown
R = 2r/(1+r); noise SE of the displayed full-season value =
SD_observed * sqrt(1 - R); minimal detectable difference between two
players ~ 1.96 * SE * sqrt(2).

Caveat stated up front: alternating dates removes slow drift but the two
halves still share the season's park/opponent mix; and R is a pool
statistic, so the SE is an average, larger for low-PA players.

Usage: python3 scripts/research/scales/noise_floor_plus.py [--min-pa 200]
"""
import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from pipeline.window_pool import score_window_against_season, AAA_TEAMS  # noqa: E402
import pipeline.locplus as lp                                            # noqa: E402
import pipeline.commandplus as cp                                        # noqa: E402

DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(DATA, '_noise_floor_plus.json')
HITTER_METRICS = ['bbPlus', 'sdPlus', 'ctPlus', 'hitterPlus']
MIN_HALF_PITCH = 150      # pitcher-side floor per half


def pear(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float('nan'), 0
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def halves_by_date(pitches):
    dates = sorted({p.get('Game Date') for p in pitches if p.get('Game Date')})
    odd = {d for i, d in enumerate(dates) if i % 2}
    a = [p for p in pitches if p.get('Game Date') not in odd]
    b = [p for p in pitches if p.get('Game Date') in odd]
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-pa', type=int, default=200)
    a = ap.parse_args()

    print('loading season cache + shipped artifacts ...', flush=True)
    allp = pickle.load(open(os.path.join(DATA, 'all_pitches_rs_cache.pkl'), 'rb'))
    mlb = [p for p in allp if p.get('_source') == 'MLB']
    metadata = json.load(open(os.path.join(DATA, 'metadata_rs.json')))
    season_rows = json.load(open(os.path.join(DATA, 'hitter_leaderboard_rs.json')))
    plb = json.load(open(os.path.join(DATA, 'pitcher_leaderboard_rs.json')))

    results = {}

    # ── hitter side ──────────────────────────────────────────────────────
    by_hitter = defaultdict(list)
    for p in mlb:
        if p.get('Batter') and p.get('BTeam') and p['BTeam'] not in AAA_TEAMS:
            by_hitter[(p['Batter'], p['BTeam'])].append(p)
    pa_by_row = {(r['hitter'], r['team']): r.get('pa') or 0 for r in season_rows}
    keys = [k for k, v in by_hitter.items() if (pa_by_row.get(k) or 0) >= a.min_pa]
    print(f'hitters with >= {a.min_pa} PA: {len(keys)}', flush=True)
    H = {m: ([], []) for m in HITTER_METRICS}
    for i, k in enumerate(keys):
        h0, h1 = halves_by_date(by_hitter[k])
        try:
            r0 = score_window_against_season(k, h0, mlb, season_rows,
                                             metadata, verbose=False)
            r1 = score_window_against_season(k, h1, mlb, season_rows,
                                             metadata, verbose=False)
        except Exception as e:      # surface, don't hide — but keep the sweep
            print(f'  SKIP {k}: {type(e).__name__} {e}', flush=True)
            continue
        for m in HITTER_METRICS:
            H[m][0].append(r0.get(m) if r0.get(m) is not None else np.nan)
            H[m][1].append(r1.get(m) if r1.get(m) is not None else np.nan)
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(keys)} hitters', flush=True)
    for m in HITTER_METRICS:
        r, n = pear(*H[m])
        # observed full-season SD from the shipped rows at the same PA floor
        sd = float(np.nanstd([row.get(m) for row in season_rows
                              if (row.get('pa') or 0) >= a.min_pa
                              and row.get('team') not in AAA_TEAMS
                              and row.get(m) is not None], ddof=1))
        results[m] = dict(split_r=r, n=n, sd_obs=sd)

    # ── Loc+ ─────────────────────────────────────────────────────────────
    print('Loc+: building season surfaces ...', flush=True)
    base = [p for p in mlb if lp.is_eligible_baseline(p)]
    G = metadata.get('gutsConstants') or {}
    LG, SC = G.get('lgWOBA', 0.3169), G.get('wOBAScale', 1.2393)
    S = lp.build_surfaces(base, LG, SC)
    per = defaultdict(lambda: ([], []))
    by_pitcher = defaultdict(list)
    for p in base:
        by_pitcher[p.get('Pitcher')].append(p)
    season_raw = {}
    for pit, plist in by_pitcher.items():
        vals = [(p.get('Game Date'), lp.score_pitch(p, S)) for p in plist]
        vals = [(d, v) for d, v in vals if v is not None]
        if len(vals) < 2 * MIN_HALF_PITCH:
            continue
        dates = sorted({d for d, _ in vals})
        odd = {d for i, d in enumerate(dates) if i % 2}
        h0 = [v for d, v in vals if d not in odd]
        h1 = [v for d, v in vals if d in odd]
        if len(h0) >= MIN_HALF_PITCH and len(h1) >= MIN_HALF_PITCH:
            per[pit] = (h0, h1)
        season_raw[pit] = (float(np.mean([v for _, v in vals])), len(vals))
    # season pool anchors + production shrink (overall Loc+ path)
    mu = float(np.mean([m for m, n in season_raw.values()]))
    sg = float(np.std([m for m, n in season_raw.values()]))
    # N_PRIOR_OVERALL is 0 (no reliability shrink on the overall number)
    def locplus(vals):
        return 100.0 - 10.0 * (float(np.mean(vals)) - mu) / sg
    a0 = [locplus(h0) for h0, h1 in per.values()]
    a1 = [locplus(h1) for h0, h1 in per.values()]
    r, n = pear(a0, a1)
    sd = float(np.nanstd([row.get('locPlus') for row in plb
                          if (row.get('count') or 0) >= 2 * MIN_HALF_PITCH
                          and row.get('team') not in AAA_TEAMS
                          and row.get('locPlus') is not None], ddof=1))
    results['locPlus'] = dict(split_r=r, n=n, sd_obs=sd)

    # ── Command+ ─────────────────────────────────────────────────────────
    # Cells (K=1 targets) are fit within the pitcher's own pitches, so the
    # displayed half-season value rebuilds cells per half — that is what
    # score_misses does given the half. League miss comes from the SEASON.
    print('Command+ ...', flush=True)
    by_p = defaultdict(list)
    for p in mlb:
        if cp.is_eligible(p):
            by_p[p.get('Pitcher')].append(p)
    season_miss = cp.score_misses({k: v for k, v in by_p.items()})
    pool = [v['raw_miss'] for v in season_miss.values()
            if v['n_pitches'] >= 2 * MIN_HALF_PITCH]
    lg_miss = float(np.mean(pool))
    c0, c1 = [], []
    for pit, plist in by_p.items():
        h0, h1 = halves_by_date(plist)
        m = cp.score_misses({0: h0, 1: h1})
        if 0 in m and 1 in m and m[0]['n_pitches'] >= MIN_HALF_PITCH \
                and m[1]['n_pitches'] >= MIN_HALF_PITCH:
            c0.append(200.0 - 100.0 * m[0]['raw_miss'] / lg_miss)
            c1.append(200.0 - 100.0 * m[1]['raw_miss'] / lg_miss)
    r, n = pear(c0, c1)
    sd = float(np.nanstd([row.get('commandPlus') for row in plb
                          if (row.get('count') or 0) >= 2 * MIN_HALF_PITCH
                          and row.get('team') not in AAA_TEAMS
                          and row.get('commandPlus') is not None], ddof=1))
    results['commandPlus'] = dict(split_r=r, n=n, sd_obs=sd)

    for m, v in results.items():
        R = 2 * v['split_r'] / (1 + v['split_r'])
        v['rel_full'] = R
        v['se_pts'] = v['sd_obs'] * float(np.sqrt(max(0.0, 1 - R)))
        v['mdd_pts'] = 1.96 * v['se_pts'] * float(np.sqrt(2))
    json.dump(results, open(OUT, 'w'), indent=1)
    print('\nmetric        split_r    n   sd_obs  rel_full  SE(pts)  min-detectable-diff')
    for m, v in results.items():
        print(f"{m:12s} {v['split_r']:8.3f} {v['n']:4d} {v['sd_obs']:7.2f} "
              f"{v['rel_full']:8.3f} {v['se_pts']:7.2f}  {v['mdd_pts']:.1f}")


if __name__ == '__main__':
    main()
