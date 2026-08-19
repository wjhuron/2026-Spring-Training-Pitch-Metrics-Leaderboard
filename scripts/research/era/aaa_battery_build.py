"""aaa_battery_build.py — the Triple-A side of the ERA battery, 2023-2026.

The MLB side already exists: data/_era_battery.json (channels),
_era_internal_cmdloc.json (Loc+/Command+) and _era_xrv100.json (xRV/100),
all keyed by MLB player id. This script produces the same quantities for
Triple-A, so a pitcher who appears at both levels in one season can be
compared to himself and the AAA-to-MLB channel offsets can be measured.

WHY IT REUSES THE MLB HARNESS UNCHANGED. aaa_pitch_consolidate.py emits
season caches shaped exactly like data/_statcastYYYY_cache.pkl — same 43
columns, same order, same nullable dtypes — so era_battery_build's own
adapt_statcast() and process_season() run on Triple-A with no fork. The
channel math is therefore identical by construction rather than by review.

KEYS ARE THE NUMERIC MLB ID, not a name. The Savant `pitcher` column
carries the same player id in the minors feed as in the majors, so the
AAA-to-MLB join needs no name matching at all. The existing Loc+ and xRV
passes key on `player_name` and resolve through _era_targets.json, which
holds MLB pitchers only and would drop every AAA-only arm; adapt_loc()
below is the same adapter with the id carried through.

TRANSLATION FRAMING, the one the repo already uses for Stuff+, Loc+ and
xRVOE at Rochester: the Loc+ SURFACES are fit on that season's MLB pitches
and Triple-A is SCORED against them. AAA never shapes a baseline. Scoring
AAA against AAA-fit surfaces would measure a pitcher against his own
league and destroy exactly the comparison this corpus exists to make.

RUNEXP CURRENCY IS CORRECTED HERE. Statcast's delta_run_exp is built on
each league's own run-expectancy matrix, so the identical event carries a
larger magnitude in Triple-A. compute_runexp_scale needs both leagues in
one frame, which is why aaa_pitch_consolidate deliberately left it alone.
The per-(Description, Count) factors are estimated from non-BIP cells, and
the factor divides the raw value, matching stuff_plus/train_stuff.py.
adapt_statcast reads RunExp from delta_pitcher_run_exp and adapt_loc from
delta_run_exp, which are the same quantity with opposite signs; flipping
both sides of a ratio leaves it unchanged, so ONE scale serves both.

MEMORY. This machine has 8 GB and one adapted season is roughly 700,000
pitch dicts, so the stages never hold two full lists at once: the MLB list
is loaded, reduced to surfaces plus slim currency records, and freed before
the AAA list is loaded.

Output: data/_aaa_battery.json
  {season: {pid: {'battery': {...}, 'loc': {v, n}, 'xrv': {v, n}}}}

    python3 scripts/research/era/aaa_battery_build.py
    python3 scripts/research/era/aaa_battery_build.py --seasons 2024
"""
import argparse
import gc
import json
import os
import pickle
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research', 'locplus'))

import pandas as pd

import pipeline.locplus as lp
from pipeline.sdplus import make_rv_xrv
from pipeline.utils import compute_runexp_scale, runexp_factor, safe_float
import locplus_constants_multiseason as base
from era_battery_build import adapt_statcast, process_season

OUT = os.path.join(ROOT, 'data', '_aaa_battery.json')
AAA_PATH = os.path.join(ROOT, 'data', '_aaa_statcast{y}_cache.pkl')
MLB_PATH = {
    2023: 'data/_statcast2023_cache.pkl',
    2024: 'data/_statcast2024_cache.pkl',
    2025: 'data/_statcast2025_full_cache.pkl',
    2026: 'data/all_pitches_rs_cache.pkl',      # sheet pickle, MLB rows
}
SEASONS = (2023, 2024, 2025, 2026)
MIN_N = 50          # matches the MLB passes; the fit applies its own gates


def adapt_loc(path, source):
    """Statcast DataFrame -> pipeline_locplus-shaped dicts, WITH the numeric
    MLB id. Mirrors locplus_constants_multiseason.adapt, which drops the id
    and keys on player_name; an AAA-only arm has no _era_targets.json row,
    so a name key would silently discard him.

    `source` matters: lp.is_eligible_baseline() admits only _source == 'MLB',
    which is the mechanism that keeps Triple-A out of the surfaces it is
    scored against. Omitting it built the surfaces on an empty list and
    every Loc+ came back None.

    RunExp is NEGATED, matching locplus_constants_multiseason.adapt and
    era_battery_build.adapt_statcast: Statcast delta_run_exp is
    offense-perspective and the pipeline expects pitcher-perspective. Leaving
    it un-negated made the MLB and AAA currency samples disagree in sign, so
    compute_runexp_scale rejected every cell on its sign guard and returned a
    negative global factor.
    """
    df = pickle.load(open(path, 'rb'))
    cols = ['pitcher', 'pitch_type', 'plate_x', 'plate_z', 'sz_top', 'sz_bot',
            'balls', 'strikes', 'description', 'delta_run_exp',
            'estimated_woba_using_speedangle', 'stand', 'p_throws',
            'player_name', 'game_date', 'release_speed']
    sub = df[cols]
    out = []
    for r in sub.itertuples(index=False):
        d = base.DMAP.get(r.description)
        if d is None or r.pitch_type is None:
            continue
        try:
            b, s = int(r.balls), int(r.strikes)
        except (TypeError, ValueError):
            continue
        if not (0 <= b <= 3 and 0 <= s <= 2):
            continue
        try:
            pid = int(r.pitcher)
        except (TypeError, ValueError):
            continue
        try:
            re = float(r.delta_run_exp)
            if re != re:
                re = None
        except (TypeError, ValueError):
            re = None
        out.append({
            'Pitcher': pid,
            'Pitch Type': _s(r.pitch_type), 'Bats': _s(r.stand),
            'Throws': _s(r.p_throws), 'Count': f'{b}-{s}',
            'Description': d,
            'PlateX': _fl(r.plate_x), 'PlateZ': _fl(r.plate_z),
            'SzTop': _fl(r.sz_top), 'SzBot': _fl(r.sz_bot),
            'RunExp': (None if re is None else -re),
            'xwOBA': _fl(r.estimated_woba_using_speedangle),
            'Game Date': str(r.game_date)[:10],
            'Velocity': _fl(r.release_speed),
            '_source': source, 'Event': None, 'BBType': None,
        })
    del df, sub
    gc.collect()
    return out


def _s(v):
    return v if isinstance(v, str) else None


def _fl(v):
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def adapt_loc_sheet(path):
    """The 2026 MLB side comes from the sheet pickle, which is already
    pipeline-shaped. Only the MLB rows may shape a surface."""
    raw = pickle.load(open(path, 'rb'))
    return [p for p in raw if p.get('_source') == 'MLB']


def slim(pitches, source):
    """The four fields compute_runexp_scale reads, and nothing else."""
    return [{'_source': source, 'Description': p.get('Description'),
             'Count': p.get('Count'), 'RunExp': p.get('RunExp')}
            for p in pitches]


def apply_currency(pitches, scale_for_source, label):
    """Divide RunExp by its (Description, Count) factor, in place."""
    if not scale_for_source:
        print(f'  {label}: NO currency scale — RunExp left MiLB-denominated',
              flush=True)
        return 0
    n = 0
    for p in pitches:
        v = safe_float(p.get('RunExp'))
        if v is None:
            continue
        f = runexp_factor(scale_for_source, p.get('Description'),
                          p.get('Count'))
        if f:
            p['RunExp'] = v / f
            n += 1
    print(f'  {label}: {n} pitches rescaled to MLB run currency', flush=True)
    return n


def loc_scores(pitches, S):
    """pid -> (mean location value, n). Scored on the MLB surfaces."""
    byp = defaultdict(list)
    for p in pitches:
        byp[p['Pitcher']].append(p)
    out = {}
    for pid, ps in byp.items():
        v = [s for s in (lp.score_pitch(p, S) for p in ps) if s is not None]
        if len(v) >= MIN_N:
            out[pid] = (sum(v) / len(v), len(v))
    return out


def xrv_scores(pitches, rv_fn):
    """pid -> (xRV/100 batter-positive, n)."""
    acc = defaultdict(lambda: [0.0, 0])
    for p in pitches:
        v = rv_fn(p)
        if v is None:
            continue
        acc[p['Pitcher']][0] += v
        acc[p['Pitcher']][1] += 1
    return {pid: (100.0 * s / n, n) for pid, (s, n) in acc.items()
            if n >= MIN_N}


def season_pass(season):
    aaa_path = AAA_PATH.format(y=season)
    if not os.path.exists(aaa_path):
        print(f'{season}: {aaa_path} absent — run '
              f'scripts/builders/aaa_pitch_consolidate.py first', flush=True)
        return None
    mlb_path = os.path.join(ROOT, MLB_PATH[season])
    if not os.path.exists(mlb_path):
        print(f'{season}: MLB source {mlb_path} absent — skipped', flush=True)
        return None

    # ── MLB stage: surfaces + slim currency records, then free ──────────
    print(f'{season}: MLB side from {MLB_PATH[season]}', flush=True)
    mlb_loc = (adapt_loc_sheet(mlb_path) if season == 2026
               else adapt_loc(mlb_path, 'MLB'))
    baseline = [p for p in mlb_loc if lp.is_eligible_baseline(p)]
    if len(baseline) < 100_000:
        # Empty or near-empty surfaces score every pitch None and the pass
        # still exits 0. Refuse instead: an MLB season is ~700k pitches.
        sys.exit(f'ABORT: {season} Loc+ baseline is {len(baseline)} pitches '
                 f'from {len(mlb_loc)} adapted. is_eligible_baseline needs '
                 f"_source == 'MLB'; check the adapter.")
    S = lp.build_surfaces(baseline, base.LG, base.SCALE)
    mlb_slim = slim(mlb_loc, 'MLB')
    print(f'  {len(mlb_loc)} MLB pitches, {len(baseline)} baseline '
          f'-> Loc+ surfaces built', flush=True)
    del baseline
    del mlb_loc
    gc.collect()

    # ── AAA stage ───────────────────────────────────────────────────────
    print(f'{season}: AAA side', flush=True)
    aaa_bat = adapt_statcast(aaa_path)
    scale = compute_runexp_scale(mlb_slim + slim(aaa_bat, 'AAA'))
    del mlb_slim
    gc.collect()
    sc = (scale or {}).get('AAA')
    if not sc:
        sys.exit(f'ABORT: {season} produced no AAA currency scale. '
                 f'RunExp would stay MiLB-denominated.')
    print(f"  currency: global {sc['global']:.4f}, "
          f"{len(sc['cell'])} cell factors, {len(sc['desc'])} desc",
          flush=True)
    if sc['global'] <= 0 or not sc['cell']:
        # A negative global factor or an empty cell table means the two
        # leagues' RunExp samples disagreed in SIGN, which is an adapter bug,
        # not a run environment. runexp_factor would then reject everything
        # and leave the values uncorrected while the pass still succeeded.
        sys.exit(f"ABORT: {season} currency scale is degenerate "
                 f"(global {sc['global']:.4f}, {len(sc['cell'])} cells). "
                 f'The MLB and AAA RunExp samples must share a sign '
                 f'convention; both adapters negate.')
    apply_currency(aaa_bat, sc, f'{season} battery')
    battery = process_season(season, aaa_bat)
    del aaa_bat
    gc.collect()

    aaa_loc = adapt_loc(aaa_path, 'AAA')
    apply_currency(aaa_loc, sc, f'{season} loc/xrv')
    loc = loc_scores(aaa_loc, S)
    xrv = xrv_scores(aaa_loc, make_rv_xrv(base.LG, base.SCALE))
    if not loc:
        sys.exit(f'ABORT: {season} scored 0 Loc+ from {len(aaa_loc)} '
                 f'AAA pitches against non-empty surfaces.')
    print(f'  {len(battery)} pitchers in the battery | {len(loc)} Loc+ | '
          f'{len(xrv)} xRV', flush=True)
    del aaa_loc, S
    gc.collect()

    out = {}
    for pid, rec in battery.items():
        e = {'battery': rec['full']}
        if int(pid) in loc:
            e['loc'] = {'v': loc[int(pid)][0], 'n': loc[int(pid)][1]}
        if int(pid) in xrv:
            e['xrv'] = {'v': xrv[int(pid)][0], 'n': xrv[int(pid)][1]}
        out[pid] = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seasons', nargs='*', type=int, default=list(SEASONS))
    a = ap.parse_args()
    result = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            result = json.load(f)
    for season in a.seasons:
        rec = season_pass(season)
        if rec is not None:
            result[str(season)] = rec
            # Write after every season: the pass is slow and an interrupted
            # run should not throw away the seasons already done.
            with open(OUT + '.tmp', 'w') as f:
                json.dump(result, f)
            os.replace(OUT + '.tmp', OUT)
            print(f'  wrote {OUT} ({len(result)} seasons)\n', flush=True)
    print('seasons on disk:', sorted(result))


if __name__ == '__main__':
    main()
