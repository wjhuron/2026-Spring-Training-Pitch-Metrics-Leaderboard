"""aaa_pitch_consolidate.py — fold the raw Triple-A pull into season caches.

Reads the gzipped per-window CSVs that aaa_pitch_pull.py wrote under
data/_aaa_raw/<season>/ and emits one DataFrame per season shaped EXACTLY
like data/_statcastYYYY_cache.pkl, so the existing ERA research harness
(scripts/research/era/era_battery_build.py and the Loc+/Stuff+/xRV passes)
runs on Triple-A with no column mapping.

The shapes already agree: the MLB cache carries 43 columns and the minors
search CSV carries 119, a strict superset. This script keeps the 43 and
drops the rest, so an AAA frame and an MLB frame are interchangeable to
every downstream consumer.

    game_type is filtered to 'R', matching the MLB caches.
    Rows are deduplicated on (game_pk, at_bat_number, pitch_number) because
    30-day windows are inclusive at both ends and a subdivided window can
    overlap its parent.

RunExp currency is NOT corrected here. delta_run_exp arrives
MiLB-denominated and the correction is per (Description, Count) via
pipeline.utils.compute_runexp_scale, which needs both leagues in one frame.
The scoring step applies it, exactly as stuff_plus/train_stuff.py does for
Rochester today. Correcting it here would hide that from the reader.

Output: data/_aaa_statcast<season>_cache.pkl   (gitignored, *.pkl)

    python3 scripts/builders/aaa_pitch_consolidate.py
    python3 scripts/builders/aaa_pitch_consolidate.py --seasons 2024
"""
import argparse
import glob
import gzip
import io
import os
import pickle
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, 'data', '_aaa_raw')

# The 43 columns data/_statcastYYYY_cache.pkl carries, in its order.
MLB_COLS = [
    'game_date', 'game_pk', 'player_name', 'pitcher', 'batter',
    'p_throws', 'stand', 'pitch_type', 'release_speed',
    'release_pos_x', 'release_pos_z', 'release_extension',
    'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z', 'plate_x',
    'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'arm_angle',
    'description', 'events', 'type', 'bb_type', 'launch_speed',
    'launch_angle', 'hc_x', 'hc_y', 'hit_distance_sc',
    'estimated_woba_using_speedangle', 'delta_run_exp',
    'delta_pitcher_run_exp', 'game_type', 'balls', 'strikes',
    'outs_when_up', 'sz_top', 'sz_bot']
DEDUPE = ['game_pk', 'at_bat_number', 'pitch_number']
KEEP = MLB_COLS + DEDUPE[1:]      # the two dedupe keys not already in MLB_COLS

# The MLB cache uses pandas NULLABLE dtypes; read_csv gives numpy float64,
# where a missing integer silently becomes a float and pd.NA becomes NaN.
# The frames are only interchangeable if the dtypes match, so cast to this
# frozen map rather than to whatever the CSV happened to parse as. Taken
# from data/_statcast2024_cache.pkl, verified column by column.
DTYPES = {
    'Int64': ['game_pk', 'pitcher', 'batter', 'balls', 'strikes',
              'outs_when_up', 'release_spin_rate', 'spin_axis',
              'launch_angle', 'hit_distance_sc'],
    'Float64': ['release_speed', 'release_pos_x', 'release_pos_z',
                'release_extension', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
                'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'arm_angle',
                'launch_speed', 'hc_x', 'hc_y',
                'estimated_woba_using_speedangle', 'delta_run_exp',
                'delta_pitcher_run_exp', 'sz_top', 'sz_bot'],
    'object': ['bb_type', 'description', 'events', 'game_date', 'game_type',
               'p_throws', 'pitch_type', 'player_name', 'stand', 'type'],
}


def load_season(season):
    files = sorted(glob.glob(os.path.join(RAW, str(season), '*.csv.gz')))
    if not files:
        return None, 0, 0
    frames = []
    empty = 0
    for p in files:
        with gzip.open(p, 'rt', encoding='utf-8-sig') as f:
            txt = f.read()
        if not txt.strip():
            empty += 1
            continue
        d = pd.read_csv(io.StringIO(txt), low_memory=False)
        keep = [c for c in KEEP if c in d.columns]
        if len(keep) < len(KEEP):
            sys.exit(f'ABORT: {os.path.basename(p)} is missing '
                     f'{sorted(set(KEEP) - set(keep))}. Schema changed.')
        # Subset before the concat, not after: it drops 119 columns to 46,
        # which cuts peak memory about 3x on a 30-club season and stops
        # pandas warning about all-NA columns it would have to dtype-guess.
        frames.append(d[keep])
    if not frames:
        return None, len(files), empty
    df = pd.concat(frames, ignore_index=True)
    return df, len(files), empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seasons', nargs='*', type=int,
                    default=[int(os.path.basename(d)) for d in
                             sorted(glob.glob(os.path.join(RAW, '*')))
                             if os.path.basename(d).isdigit()])
    a = ap.parse_args()
    if not a.seasons:
        sys.exit(f'ABORT: no season directories under {RAW}. '
                 f'Run scripts/builders/aaa_pitch_pull.py first.')
    for season in a.seasons:
        df, nfiles, nempty = load_season(season)
        if df is None:
            print(f'{season}: {nfiles} window files, all empty — skipped')
            continue
        raw = len(df)
        missing = [c for c in MLB_COLS if c not in df.columns]
        if missing:
            # A silently short frame would score against a partial channel
            # set and look fine, so refuse instead.
            sys.exit(f'ABORT: {season} is missing MLB-cache columns '
                     f'{missing}. The minors CSV schema changed.')
        have = [c for c in DEDUPE if c in df.columns]
        if len(have) == len(DEDUPE):
            df = df.drop_duplicates(subset=DEDUPE)
        else:
            sys.exit(f'ABORT: {season} lacks {set(DEDUPE)-set(have)}; '
                     f'cannot deduplicate overlapping windows safely.')
        deduped = len(df)
        df = df[df['game_type'] == 'R']
        df = df[MLB_COLS].copy()
        mapped = [c for v in DTYPES.values() for c in v]
        unmapped = [c for c in MLB_COLS if c not in mapped]
        if unmapped:
            sys.exit(f'ABORT: DTYPES does not cover {unmapped}. Add them '
                     f'rather than letting read_csv guess.')
        for dtype, cols in DTYPES.items():
            for c in cols:
                if dtype == 'object':
                    df[c] = df[c].astype(object)
                elif dtype == 'Int64':
                    # Savant serves these whole; round before the cast so a
                    # float artefact does not raise.
                    df[c] = pd.to_numeric(df[c], errors='coerce') \
                              .round().astype('Int64')
                else:
                    df[c] = pd.to_numeric(df[c], errors='coerce') \
                              .astype('Float64')
        df['game_date'] = df['game_date'].astype(str)
        out = os.path.join(ROOT, 'data', f'_aaa_statcast{season}_cache.pkl')
        with open(out + '.tmp', 'wb') as f:
            pickle.dump(df, f, protocol=4)
        os.replace(out + '.tmp', out)
        arm = 100.0 * df['arm_angle'].notna().mean()
        xw = 100.0 * df['estimated_woba_using_speedangle'].notna().mean()
        re_ = 100.0 * df['delta_run_exp'].notna().mean()
        print(f'{season}: {nfiles} windows ({nempty} empty) | {raw} rows -> '
              f'{deduped} deduped -> {len(df)} regular season | '
              f'{df["pitcher"].nunique()} pitchers | arm {arm:.1f}% '
              f'xwOBA {xw:.1f}% RunExp {re_:.1f}% | wrote {out}', flush=True)


if __name__ == '__main__':
    main()
