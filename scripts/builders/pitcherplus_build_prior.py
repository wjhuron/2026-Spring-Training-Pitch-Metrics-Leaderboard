#!/usr/bin/env python3
"""pitcherplus_build_prior.py — freeze the prior-season Pitcher+ asset.

Pitcher+ Proj blends 70% current season / 30% prior season. Production has
no prior-season leaderboard (pitcher_leaderboard_rs.json is current-season
only), so the prior ships as a static asset built here from the research
tables and regenerated ONCE A YEAR after the season ends.

  data/pitcher_plus_prior.json
    {"season": 2025, "formula": "...", "values": {"<mlbId>": <Pitcher+>}}

Pitcher+ for the prior season is computed with the SAME frozen formula and
the same shrinkage constants as production (pipeline_pitcherplus), scored
against that season's own league baseline — so a 2025 Pitcher+ of 120 means
the same thing as a 2026 Pitcher+ of 120.

KNOWN CAVEAT (accepted by Wally 2026-07-24): the components here come from
PUBLIC Savant data — public pitch tags, and Stuff+ from the leave-one-
season-out models in pitcherplus_stuff_loso.py rather than the production
v11 bundle. Pitch tags feed Stuff+ and Loc+, so roughly 26% of the
projection's weight rests on data that is not Wally's retagging. This is
the same compromise Stuff+ v11 already makes by training on public 2021-24
tags. Rebuilding the prior through the production sheets path would remove
it, at much greater cost.

Usage: python3 scripts/builders/pitcherplus_build_prior.py [--season 2025]
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'research', 'stuff'))  # pitcherplus_search moved in 2026-08 reorg

import pitcherplus_search as ps                     # noqa: E402
from pipeline.pitcherplus import (COMPONENTS, QUAL_N,   # noqa: E402
                                  SCALE_K, MIN_POOL)

OUT = os.path.join(ROOT, 'data', 'pitcher_plus_prior.json')

# research-table column -> production component key
SRC = {'stuffScore': 'stuffRaw', 'locPlus': 'locRaw', 'kPct': 'kPct',
       'izWhiffPct': 'izWhiffPct', 'xRv100': 'xrv100', 'gbPct': 'gbPct'}


def main(season):
    t = pickle.load(open(ps.TABLES_PKL, 'rb'))
    t = ps.merge_external(t)
    for c in [x for x in t.columns if x not in ps.META]:
        t[c] = pd.to_numeric(t[c], errors='coerce')
    d = t[(t['half'] == 'full') & (t['season'] == season)
          & (t['n'] >= QUAL_N)].copy()
    if len(d) < MIN_POOL:
        sys.exit(f'{season}: only {len(d)} qualified pitchers')

    raw = np.zeros(len(d))
    for key, w, k in COMPONENTS:
        col = SRC[key]
        v = d[col].to_numpy(float)
        n = d['n'].to_numpy(float)
        ok = np.isfinite(v)
        mu = float(np.average(v[ok], weights=n[ok]))
        sd = float(np.std(v[ok], ddof=1))
        z = np.where(ok, (v - mu) / sd, 0.0)
        raw += w * z * (n / (n + k))
        print(f'  {key:12s} mu {mu:9.4f}  sd {sd:8.4f}  '
              f'coverage {ok.mean():.1%}')

    pplus = 100.0 + SCALE_K * (raw - raw.mean()) / raw.std(ddof=1)
    vals = {}
    for pid, v in zip(d['pid'].to_numpy(), pplus):
        try:
            vals[str(int(pid))] = round(float(v), 1)
        except (TypeError, ValueError):
            continue
    out = {
        'season': int(season),
        'note': ('Prior-season Pitcher+ for the 70/30 projection blend. '
                 'Built from public Savant components + LOSO Stuff+ (see '
                 'scripts/builders/pitcherplus_build_prior.py) — NOT the retagged '
                 'production pipeline. Regenerate annually.'),
        'nPitchers': len(vals),
        'values': vals,
    }
    with open(OUT, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'\nsaved {OUT}: {len(vals)} pitchers, season {season}')
    print(f'  Pitcher+ range {pplus.min():.1f} - {pplus.max():.1f}, '
          f'mean {pplus.mean():.1f}, sd {pplus.std(ddof=1):.2f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--season', type=int, default=2025)
    main(ap.parse_args().season)
