#!/usr/bin/env python3
"""
Compare raw VAA/HAA versus location-normalized nVAA/nHAA as predictors of
whiff rate and xwOBAcon, both as pitcher-level summary stats (Test 1) and
as location-stratified bucketing tools (Test 3).

Test 1: for each (pitch type, batter hand), compute Pearson and Spearman
  correlations of {VAA, nVAA, velocity, IVB} vs. {whiff%, xwOBAcon} across
  pitchers with >=25 pitches at that cell. Velo-partial correlations too.
  Same analysis for HAA vs. nHAA using {HAA, nHAA, velocity, HB}.

Test 3: for each (pitch type, batter hand, outcome), bucket pitchers into
  quintiles by nVAA (and separately by VAA), then smooth per-quintile
  whiff/xwOBAcon fields over PlateX/PlateZ with a Gaussian filter. Display
  each quintile as a ratio vs. the league-in-cell rate. If nVAA bucketing
  separates optimal zones more sharply than VAA bucketing, nVAA is the
  more actionable location signal.

Usage:
  python3 scripts/test_approach_angles.py              # run both tests
  python3 scripts/test_approach_angles.py --skip-test3 # Test 1 only
  python3 scripts/test_approach_angles.py --refetch    # refetch from Sheets

Outputs land in scripts/outputs/.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats, ndimage
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
CACHE_DIR = ROOT / 'scripts' / 'cache'  # caches stay local to the repo
OUTPUT_DIR = Path.home() / 'Downloads'
HEATMAP_DIR = OUTPUT_DIR / 'approach_angle_heatmaps'

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

PITCH_CACHE = CACHE_DIR / 'pitches_2026.json'

# ── Config ───────────────────────────────────────────────────────────────
VAA_PITCH_TYPES = ['FF', 'SI', 'CU', 'FS', 'CH']
HAA_PITCH_TYPES = ['FF', 'SI', 'SL', 'ST', 'CU']
MIN_COUNT = 25

# Heatmap zone and grid
PX_RANGE = (-1.5, 1.5)
PZ_RANGE = (0.5, 4.5)
GRID_SIZE = 30
SMOOTH_SIGMA = 1.5
# gaussian_filter preserves total count, not peak, so a single observation
# contributes ~1/(2*pi*sigma^2) ~= 0.07 to its peak cell. 0.5 is roughly
# "at least ~7 observations in the effective smoothed neighborhood" — strict
# enough to silence noise in sparse outer cells, lenient enough to keep xwOBAcon
# (which has ~10x fewer observations than whiff) populated.
MIN_SMOOTH_DENOM = 0.5

# Outcome definitions (matches pipeline_utils.SWING_DESCRIPTIONS)
SWING_DESCRIPTIONS = {'Swinging Strike', 'Foul', 'In Play'}
WHIFF_DESCRIPTIONS = {'Swinging Strike'}
BIP_DESCRIPTION = 'In Play'


# ── Data loading ─────────────────────────────────────────────────────────

def load_pitcher_stats():
    """pitcher + pitch-type level stats from pitch_leaderboard_rs.json."""
    path = DATA_DIR / 'pitch_leaderboard_rs.json'
    with path.open() as f:
        return json.load(f)


def aggregate_micro_by_hand():
    """Sum pitchMicro across dates per (pitcher, team, pitchType, batterHand).

    Returns dict: (pitcher_name, team, pitchType, batterHand) -> counts.
    """
    with (DATA_DIR / 'micro_data_rs.json').open() as f:
        micro = json.load(f)
    lookups = micro['lookups']
    pitchers = lookups['pitchers']
    teams = lookups['teams']
    pitch_types = lookups['pitchTypes']

    cols = micro['pitchCols']
    iPitcher = cols.index('pitcherIdx')
    iTeam = cols.index('teamIdx')
    iPT = cols.index('pitchTypeIdx')
    iBH = cols.index('batterHand')
    iN = cols.index('n')
    iSw = cols.index('sw')
    iWh = cols.index('wh')
    iBip = cols.index('bip')

    agg = defaultdict(lambda: {'n': 0, 'sw': 0, 'wh': 0, 'bip': 0})
    for r in micro['pitchMicro']:
        key = (pitchers[r[iPitcher]], teams[r[iTeam]], pitch_types[r[iPT]], r[iBH])
        agg[key]['n'] += r[iN]
        agg[key]['sw'] += r[iSw]
        agg[key]['wh'] += r[iWh]
        agg[key]['bip'] += r[iBip]
    return dict(agg)


def build_pitcher_level_df():
    """Join pitch leaderboard with per-hand micro aggregates into a long df
    keyed by (pitcher, team, pitchType, batterHand).
    """
    leaderboard = load_pitcher_stats()
    hand_agg = aggregate_micro_by_hand()

    rows = []
    for r in leaderboard:
        pt = r.get('pitchType')
        for bh in ('L', 'R'):
            key = (r['pitcher'], r['team'], pt, bh)
            if key not in hand_agg:
                continue
            m = hand_agg[key]
            if m['sw'] == 0:
                continue
            xwobacon_col = 'xwOBAcon_vsL' if bh == 'L' else 'xwOBAcon_vsR'
            rows.append({
                'pitcher': r['pitcher'],
                'team': r['team'],
                'pitchType': pt,
                'throws': r.get('throws'),
                'batterHand': bh,
                'count': m['n'],
                'nSwings': m['sw'],
                'vaa': r.get('vaa'),
                'nVAA': r.get('nVAA'),
                'haa': r.get('haa'),
                'nHAA': r.get('nHAA'),
                'velocity': r.get('velocity'),
                'indVertBrk': r.get('indVertBrk'),
                'horzBrk': r.get('horzBrk'),
                'whiffPct': m['wh'] / m['sw'],
                'xwOBAcon': r.get(xwobacon_col),
                'nBip': m['bip'],
            })
    return pd.DataFrame(rows)


# ── Sheets fetch (lazy, only when Test 3 runs) ──────────────────────────

def fetch_pitches_from_sheets():
    sys.path.insert(0, str(ROOT))
    from pipeline_fetch import read_all_pitches_from_sheets

    print("  reading the six division workbooks (huronalytics)...")
    pitches = read_all_pitches_from_sheets()
    print(f"    {len(pitches)} pitches")
    return pitches


def _safe_float(v):
    if v is None or v == '':
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def load_or_fetch_pitches(refetch=False):
    """Load cached MLB pitches, or fetch from Sheets and cache."""
    if PITCH_CACHE.exists() and not refetch:
        print(f"Loading cached pitches from {PITCH_CACHE}")
        with PITCH_CACHE.open() as f:
            return json.load(f)

    print("No pitch cache. Fetching from Sheets (a few minutes).")
    raw = fetch_pitches_from_sheets()

    keep_fields = (
        'Pitcher', 'PTeam', 'Throws', 'Bats', 'Pitch Type',
        'Description', 'Event', 'Game Date',
    )
    numeric_fields = ('PlateX', 'PlateZ', 'xwOBA', 'VAA', 'HAA')

    kept = []
    for p in raw:
        if p.get('_source') != 'MLB':
            continue
        slim = {k: p.get(k) for k in keep_fields}
        for nf in numeric_fields:
            slim[nf] = _safe_float(p.get(nf))
        kept.append(slim)

    print(f"Kept {len(kept)} MLB pitches (filtered from {len(raw)} raw).")
    with PITCH_CACHE.open('w') as f:
        json.dump(kept, f)
    print(f"Cached to {PITCH_CACHE}")
    return kept


# ── Test 1: pitcher-level correlations ──────────────────────────────────

def partial_corr_velo(x, y, velo):
    """Pearson correlation of x and y after residualizing both on velocity."""
    df = pd.DataFrame({'x': x, 'y': y, 'v': velo}).dropna()
    if len(df) < 5:
        return np.nan
    bx = np.polyfit(df['v'], df['x'], 1)
    by = np.polyfit(df['v'], df['y'], 1)
    rx = df['x'] - np.polyval(bx, df['v'])
    ry = df['y'] - np.polyval(by, df['v'])
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    r, _ = stats.pearsonr(rx, ry)
    return r


def run_test_1(df):
    print()
    print("=" * 78)
    print("TEST 1: pitcher-level correlations (whiff%, xwOBAcon)")
    print("=" * 78)

    analyses = [
        ('Vertical',   VAA_PITCH_TYPES, ['vaa', 'nVAA', 'velocity', 'indVertBrk']),
        ('Horizontal', HAA_PITCH_TYPES, ['haa', 'nHAA', 'velocity', 'horzBrk']),
    ]

    results = []
    for analysis_name, pitch_types, predictors in analyses:
        for pt in pitch_types:
            for bh in ('L', 'R'):
                cell = df[
                    (df.pitchType == pt)
                    & (df.batterHand == bh)
                    & (df['count'] >= MIN_COUNT)
                ]
                for outcome in ('whiffPct', 'xwOBAcon'):
                    sub = cell[[*predictors, outcome]].dropna()
                    if len(sub) < 10:
                        continue
                    row = {
                        'analysis': analysis_name,
                        'pitchType': pt,
                        'batterHand': bh,
                        'outcome': outcome,
                        'n_pitchers': len(sub),
                    }
                    velo_series = sub['velocity'] if 'velocity' in sub else None
                    for pred in predictors:
                        r_p, _ = stats.pearsonr(sub[pred], sub[outcome])
                        r_s, _ = stats.spearmanr(sub[pred], sub[outcome])
                        row[f'{pred}_pearson'] = r_p
                        row[f'{pred}_spearman'] = r_s
                        if pred != 'velocity' and velo_series is not None:
                            row[f'{pred}_partial_velo'] = partial_corr_velo(
                                sub[pred], sub[outcome], velo_series
                            )
                    results.append(row)

    out_df = pd.DataFrame(results)
    out_path = OUTPUT_DIR / 'test1_correlations.csv'
    out_df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")

    # Concise printed summary
    for analysis_name, pitch_types, predictors in analyses:
        print(f"\n--- {analysis_name} ---")
        sub = out_df[out_df.analysis == analysis_name]
        for outcome in ('whiffPct', 'xwOBAcon'):
            print(f"\n  {outcome} (Pearson r):")
            header = '    pitch   n  '
            for pred in predictors:
                header += f'{pred:>12s}'
            print(header)
            for pt in pitch_types:
                for bh in ('L', 'R'):
                    row = sub[
                        (sub.pitchType == pt) & (sub.batterHand == bh) & (sub.outcome == outcome)
                    ]
                    if row.empty:
                        continue
                    r = row.iloc[0]
                    line = f'    {pt:3s} v{bh}  {int(r.n_pitchers):3d}'
                    for pred in predictors:
                        val = r[f'{pred}_pearson']
                        line += f'{val:+11.3f} '
                    print(line)
    return out_df


# ── Test 3: location-stratified heatmaps ────────────────────────────────

def prepare_pitches_df(pitches_raw):
    """Turn raw pitch dicts into a DataFrame with outcome indicators."""
    pdf = pd.DataFrame(pitches_raw)
    pdf = pdf.rename(columns={
        'Pitcher': 'pitcher',
        'PTeam': 'team',
        'Throws': 'throws',
        'Bats': 'batterHand',
        'Pitch Type': 'pitchType',
    })
    pdf = pdf.dropna(subset=['PlateX', 'PlateZ'])
    pdf['_is_swing'] = pdf['Description'].isin(SWING_DESCRIPTIONS).astype(float)
    pdf['_is_whiff'] = pdf['Description'].isin(WHIFF_DESCRIPTIONS).astype(float)
    pdf['_is_bip'] = (pdf['Description'] == BIP_DESCRIPTION).astype(float)
    # xwOBA numerator only counts when BIP and xwOBA is not null
    valid_bip = (pdf['_is_bip'] == 1) & pdf['xwOBA'].notna()
    pdf['_xwoba_num'] = np.where(valid_bip, pdf['xwOBA'].fillna(0), 0.0)
    pdf['_xwoba_denom'] = np.where(valid_bip, 1.0, 0.0)
    return pdf


def compute_smoothed_rate(pdf, num_col, denom_col):
    x = pdf['PlateX'].values
    z = pdf['PlateZ'].values
    num_w = pdf[num_col].values.astype(float)
    den_w = pdf[denom_col].values.astype(float)

    num_h, xedges, zedges = np.histogram2d(
        x, z, bins=GRID_SIZE, range=[PX_RANGE, PZ_RANGE], weights=num_w
    )
    den_h, _, _ = np.histogram2d(
        x, z, bins=GRID_SIZE, range=[PX_RANGE, PZ_RANGE], weights=den_w
    )
    num_s = ndimage.gaussian_filter(num_h, sigma=SMOOTH_SIGMA)
    den_s = ndimage.gaussian_filter(den_h, sigma=SMOOTH_SIGMA)
    rate = np.where(den_s > MIN_SMOOTH_DENOM, num_s / den_s, np.nan)
    return rate, xedges, zedges


def plot_quintile_heatmap(pdf_qual, pdf_league, outcome_metric, primary_col, alt_col,
                          pitch_type, batter_hand, out_path):
    """2x2 panel fig: Q1/Q5 for primary_col (top) and alt_col (bottom), each vs. league rate."""
    if outcome_metric == 'whiff':
        num, den = '_is_whiff', '_is_swing'
        title = 'Whiff rate'
    else:
        num, den = '_xwoba_num', '_xwoba_denom'
        title = 'xwOBAcon'

    league_rate, xedges, zedges = compute_smoothed_rate(pdf_league, num, den)
    extent = [xedges[0], xedges[-1], zedges[0], zedges[-1]]
    norm = mcolors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=1.5)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)

    for row_idx, (bucket_name, q_col) in enumerate([(primary_col, 'q_primary'),
                                                     (alt_col, 'q_alt')]):
        quints = sorted(pdf_qual[q_col].dropna().unique())
        if len(quints) < 2:
            continue
        display = [quints[0], quints[-1]]
        for col_idx, q in enumerate(display):
            qdf = pdf_qual[pdf_qual[q_col] == q]
            q_rate, _, _ = compute_smoothed_rate(qdf, num, den)
            lift = np.where(
                (league_rate > 0) & ~np.isnan(league_rate) & ~np.isnan(q_rate),
                q_rate / league_rate,
                np.nan,
            )
            ax = axes[row_idx, col_idx]
            im = ax.imshow(
                lift.T, extent=extent, origin='lower', aspect='auto',
                cmap='RdBu_r', norm=norm,
            )
            ax.add_patch(plt.Rectangle(
                (-0.83, 1.5), 1.66, 2.0, fill=False, edgecolor='black', linewidth=1.5
            ))
            ax.set_title(
                f'Q{int(q)} of {bucket_name} '
                f'(pitchers={qdf["pitcher"].nunique()}, pitches={len(qdf)})',
                fontsize=11,
            )
            if col_idx == 0:
                ax.set_ylabel('Plate Z (ft)')
            if row_idx == 1:
                ax.set_xlabel('Plate X (ft)')

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
    cbar.set_label(f'{title} lift vs. league in-cell rate')
    fig.suptitle(
        f'{title}: {pitch_type} vs {batter_hand}HH '
        f'(top = {primary_col} quintiles, bottom = {alt_col} quintiles)',
        fontsize=13,
    )
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def run_test_3(df, pitches_raw):
    print()
    print("=" * 78)
    print("TEST 3: location-stratified heatmaps by nVAA/VAA quintile")
    print("=" * 78)

    pdf = prepare_pitches_df(pitches_raw)
    print(f"  prepared {len(pdf)} pitches")

    analyses = [
        ('Vertical',   VAA_PITCH_TYPES, 'nVAA', 'vaa'),
        ('Horizontal', HAA_PITCH_TYPES, 'nHAA', 'haa'),
    ]

    for analysis_name, pitch_types, primary_col, alt_col in analyses:
        for pt in pitch_types:
            for bh in ('L', 'R'):
                qual = df[
                    (df.pitchType == pt)
                    & (df.batterHand == bh)
                    & (df['count'] >= MIN_COUNT)
                ].dropna(subset=[primary_col, alt_col])
                if len(qual) < 10:
                    continue

                # Assign quintile labels (1..5) by each bucketing column.
                try:
                    q_primary = pd.qcut(qual[primary_col], q=5, labels=False, duplicates='drop') + 1
                    q_alt = pd.qcut(qual[alt_col], q=5, labels=False, duplicates='drop') + 1
                except ValueError:
                    continue
                qual = qual.assign(q_primary=q_primary, q_alt=q_alt)

                # League baseline: all pitches of this (pitchType, batterHand).
                pdf_league = pdf[(pdf.pitchType == pt) & (pdf.batterHand == bh)]
                if len(pdf_league) < 200:
                    continue

                # Qualifying pool for quintile rendering.
                keep = qual[['pitcher', 'team', 'pitchType', 'q_primary', 'q_alt']]
                pdf_qual = pdf_league.merge(
                    keep, on=['pitcher', 'team', 'pitchType'], how='inner'
                )
                if len(pdf_qual) < 100:
                    continue

                for outcome in ('whiff', 'xwobacon'):
                    out_path = HEATMAP_DIR / f'{analysis_name.lower()}_{pt}_vs{bh}_{outcome}.png'
                    plot_quintile_heatmap(
                        pdf_qual, pdf_league, outcome,
                        primary_col, alt_col, pt, bh, out_path,
                    )
                    print(f"  wrote {out_path.name}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-test3', action='store_true',
                        help='Run Test 1 only (no Sheets fetch)')
    parser.add_argument('--refetch', action='store_true',
                        help='Refetch pitches from Sheets even if cached')
    args = parser.parse_args()

    df = build_pitcher_level_df()
    print(f"Loaded {len(df)} (pitcher, pitchType, batterHand) rows "
          f"across {df.pitcher.nunique()} pitchers.")

    run_test_1(df)

    if args.skip_test3:
        print("\n(--skip-test3) done.")
        return

    pitches = load_or_fetch_pitches(refetch=args.refetch)
    run_test_3(df, pitches)

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
