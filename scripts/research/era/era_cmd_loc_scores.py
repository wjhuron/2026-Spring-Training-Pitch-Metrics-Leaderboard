"""era_cmd_loc_scores.py — per-pitcher-season Loc+ and Command+ raw scores,
2021-2026, for the ERA-estimator screen.

Loc+: shipped config surfaces built IN-SEASON on the full season
(pipeline_locplus via the locplus_constants_multiseason adapter), each
pitcher's raw score = mean per-pitch location value. The h1 scope scores
first-half pitches on the full-season surfaces; the league surface is
pitcher-independent, so the h1 leak is negligible (noted in the report).

Command+: shipped production model (pipeline_commandplus.score_misses —
K=1 cell means + thin-cell cascade), targets fit within scope, so h1
targets come from h1 pitches only (self-contained, no leak).

2026 comes from the sheet pickle for Command+ only; Loc+ 2026 is already
in the battery (per-pitch sheet Loc+ means).

Keys are MLB ids resolved from data/_era_targets.json names (lowercased,
with a multi-word-surname variant). Ambiguous or unmatched names dropped.

Output: data/_era_internal_cmdloc.json
  {season: {pid: {loc_full, loc_h1, loc_n_full, loc_n_h1,
                  cmd_full, cmd_h1, cmd_n_full, cmd_n_h1}}}
"""
import gc
import json
import os
import pickle
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import pipeline.locplus as lp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'locplus'))  # locplus_constants_multiseason moved in 2026-08 reorg
import locplus_constants_multiseason as base
from pipeline.commandplus import score_misses
from pipeline.utils import _fullname_to_lastfirst

OUT = os.path.join(ROOT, 'data', '_era_internal_cmdloc.json')
TARGETS = json.load(open(os.path.join(ROOT, 'data', '_era_targets.json')))

SEASONS = [(2021, 'data/_statcast2021_cache.pkl'),
           (2022, 'data/_statcast2022_cache.pkl'),
           (2023, 'data/_statcast2023_cache.pkl'),
           (2024, 'data/_statcast2024_cache.pkl'),
           (2025, 'data/_statcast2025_full_cache.pkl')]

MIN_N = 50   # floor to emit a score at all; the screen applies its own gates


def name_map(season):
    m, ambig = {}, set()
    for pid, rec in TARGETS[str(season)]['pitchers'].items():
        full = (rec['name'] or '').strip()
        variants = {_fullname_to_lastfirst(full).lower()}
        parts = full.split()
        if len(parts) >= 3:
            variants.add((' '.join(parts[-2:]) + ', '
                          + ' '.join(parts[:-2])).lower())
        for lf in variants:
            if lf in m and m[lf] != int(pid):
                ambig.add(lf)
            m[lf] = int(pid)
    for lf in ambig:
        del m[lf]
    return m


def loc_scores(pitches, S):
    """(Pitcher, Throws) -> (mean location value, n) for one scope."""
    byp = defaultdict(list)
    for p in pitches:
        byp[(p.get('Pitcher'), p.get('Throws'))].append(p)
    out = {}
    for k, ps in byp.items():
        v = [s for s in (lp.score_pitch(p, S) for p in ps) if s is not None]
        if len(v) >= MIN_N:
            out[k] = (sum(v) / len(v), len(v))
    return out


def cmd_scores(pitches):
    byp = defaultdict(list)
    for p in pitches:
        byp[(p.get('Pitcher'), p.get('Throws'))].append(p)
    res = score_misses(byp)
    return {k: (r['raw_miss'], r['n_pitches']) for k, r in res.items()
            if r['n_pitches'] >= MIN_N}


def emit(season, loc_f, loc_h, cmd_f, cmd_h):
    nm = name_map(season)
    rec = defaultdict(dict)
    keys = set(loc_f) | set(loc_h) | set(cmd_f) | set(cmd_h)
    unmatched = 0
    for k in keys:
        pid = nm.get((k[0] or '').lower())
        if pid is None:
            unmatched += 1
            continue
        r = rec[str(pid)]
        for tag, src in (('loc_full', loc_f), ('loc_h1', loc_h),
                         ('cmd_full', cmd_f), ('cmd_h1', cmd_h)):
            if k in src:
                r[tag], r[tag.replace('full', 'n_full')
                          .replace('h1', 'n_h1')] = src[k]
    print(f'  {season}: {len(rec)} pitchers mapped, {unmatched} unmatched',
          flush=True)
    return dict(rec)


def main():
    result = {}
    for season, path in SEASONS:
        print(f'{season}: adapting {path}', flush=True)
        pitches = base.adapt(os.path.join(ROOT, path))
        asg = TARGETS[str(season)]['asg']
        h1 = [p for p in pitches if p.get('Game Date')
              and p['Game Date'] <= asg]
        b = [p for p in pitches if lp.is_eligible_baseline(p)]
        S = lp.build_surfaces(b, base.LG, base.SCALE)
        loc_f = loc_scores(pitches, S)
        loc_h = loc_scores(h1, S)
        cmd_f = cmd_scores(pitches)
        cmd_h = cmd_scores(h1)
        result[str(season)] = emit(season, loc_f, loc_h, cmd_f, cmd_h)
        del pitches, h1, b, S
        gc.collect()

    # 2026: Command+ only, from the sheet pickle
    print('2026: sheet pickle (Command+ only)', flush=True)
    raw = pickle.load(open(os.path.join(ROOT, 'data',
                                        'all_pitches_rs_cache.pkl'), 'rb'))
    asg = TARGETS['2026']['asg']
    mlb = [p for p in raw if p.get('_source') == 'MLB']
    del raw
    gc.collect()
    h1 = [p for p in mlb if p.get('Game Date')
          and str(p['Game Date'])[:10] <= asg]
    cmd_f = cmd_scores(mlb)
    cmd_h = cmd_scores(h1)
    result['2026'] = emit(2026, {}, {}, cmd_f, cmd_h)
    with open(OUT, 'w') as f:
        json.dump(result, f)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
