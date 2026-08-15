#!/usr/bin/env python3
"""pitchtype_gate_measure.py — split-half reliability vs sample size for every
per-pitch-type metric the season cards color (2026-07-30, Wally's ask).

The site's 25-pitch outcome gate (MIN_PITCH_TYPE_OUTCOME) was an inherited
convention, never validated against a reliability criterion; only Loc+ (per-
group, 70-122) and now GB% (25 BIP) have measured gates. This measures the
rest at the RENDERED unit: (pitcher, pitch type, season).

Method (mirrors the GB% measurement): random split of the unit's pitches into
halves (5 seeds), metric per half, bin units by total sample, half-half r per
bin, Spearman-Brown to full sample. The 0.5 crossing is the reliability-based
gate candidate. Seasons 2023-2025 run as independent replicates.

Metrics and their sample denominations:
  pitches: Whiff%, Chase%, Zone%, Z-Whiff%, CSW%, 2K-Whiff%,
           RV/100 (delta_run_exp), xRV/100 proxy (see below)
  BIP:     xwOBAcon (companion to the measured GB% gate)

xRV/100 proxy: per-pitch delta_run_exp, with BIP pitches' values replaced by
the league-wide mean delta_run_exp of their xwOBA vigintile — the same
luck-neutralization idea as the site's xRunValue (only BIP outcomes are
expectation-replaced; count transitions keep their actual values).

Usage: python3 scripts/research/hitter/pitchtype_gate_measure.py
"""
import sys
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'misc'))  # leaderboard_metric_battery moved in 2026-08 reorg
from leaderboard_metric_battery import load_season  # noqa: E402

SEASONS = [2023, 2024, 2025]
SEEDS = 5
PITCH_BINS = [(10, 25), (25, 50), (50, 100), (100, 150), (150, 250),
              (250, 400), (400, 700), (700, 1200)]
BIP_BINS = [(10, 20), (20, 30), (30, 45), (45, 60), (60, 80), (80, 110),
            (110, 150), (150, 220), (220, 400)]
MIN_UNITS = 40   # bins with fewer (pitcher, pitch type) units are skipped


def add_xrv_proxy(df):
    """delta_run_exp with BIP outcomes replaced by their xwOBA-vigintile
    league mean — luck-neutralized in the same spirit as the site's xRV."""
    rv = df['delta_run_exp'].astype(float)
    bip = df['bip'] & df['estimated_woba_using_speedangle'].notna() & rv.notna()
    out = rv.copy()
    if bip.sum() > 1000:
        xw = df.loc[bip, 'estimated_woba_using_speedangle']
        binned = pd.qcut(xw, 20, duplicates='drop')
        exp = rv[bip].groupby(binned, observed=True).transform('mean')
        out[bip] = exp
    df['xrv_proxy'] = out
    return df


def half_metrics(h):
    """Per-(pitcher, pitch_type, half) metric values + sample counts."""
    g = h.groupby(['pitcher', 'pitch_type', 'half'])
    sw = g['swing'].sum()
    wh = g['whiff'].sum()
    iz = g['iz'].sum()
    izsw = g.apply(lambda x: (x['iz'] & x['swing']).sum())
    izwh = g.apply(lambda x: (x['iz'] & x['whiff']).sum())
    two = g.apply(lambda x: ((x['strikes'] == 2) & x['swing']).sum())
    twowh = g.apply(lambda x: ((x['strikes'] == 2) & x['whiff']).sum())
    return pd.DataFrame({
        'n': g.size(),
        'nbip': g['bip'].sum(),
        'whiffPct': wh / sw.replace(0, np.nan),
        'chasePct': g['chase_sw'].sum() / g['ooz'].sum().replace(0, np.nan),
        'zonePct': iz / g.size(),
        'izWhiffPct': izwh / izsw.replace(0, np.nan),
        'cswPct': g['csw'].sum() / g.size(),
        'twoKWhiffPct': twowh / two.replace(0, np.nan),
        'rv100': g['delta_run_exp'].mean() * 100,
        'xrv100': g['xrv_proxy'].mean() * 100,
        'xwobacon': g.apply(lambda x: x.loc[x['bip'],
                            'estimated_woba_using_speedangle'].mean()),
    })


PITCH_METRICS = ['whiffPct', 'chasePct', 'zonePct', 'izWhiffPct', 'cswPct',
                 'twoKWhiffPct', 'rv100', 'xrv100']
BIP_METRICS = ['xwobacon']


def main():
    results = {}
    for year in SEASONS:
        df = load_season(year)
        df['strikes'] = pd.to_numeric(df['strikes'], errors='coerce')
        df = add_xrv_proxy(df)
        per_seed = []
        for seed in range(SEEDS):
            rng = np.random.default_rng(1000 * year + seed)
            df['half'] = rng.integers(0, 2, len(df))
            hm = half_metrics(df).reset_index()
            wide = hm.pivot_table(index=['pitcher', 'pitch_type'],
                                  columns='half',
                                  values=['n', 'nbip'] + PITCH_METRICS + BIP_METRICS)
            wide['tot_n'] = wide[('n', 0)] + wide[('n', 1)]
            wide['tot_bip'] = wide[('nbip', 0)] + wide[('nbip', 1)]
            per_seed.append(wide)
        rows = []
        for metric in PITCH_METRICS + BIP_METRICS:
            bins = BIP_BINS if metric in BIP_METRICS else PITCH_BINS
            sizecol = 'tot_bip' if metric in BIP_METRICS else 'tot_n'
            for lo, hi in bins:
                rs = []
                for wide in per_seed:
                    m = wide[(wide[sizecol] >= lo) & (wide[sizecol] < hi)]
                    a, b = m[(metric, 0)], m[(metric, 1)]
                    ok = a.notna() & b.notna()
                    if ok.sum() < MIN_UNITS:
                        continue
                    rs.append(a[ok].corr(b[ok]))
                if not rs:
                    continue
                r = float(np.mean(rs))
                rows.append({'metric': metric, 'lo': lo, 'hi': hi,
                             'r': r, 'R_full': 2 * r / (1 + r)})
        results[year] = pd.DataFrame(rows)
        print(f'\n===== {year} =====')
        for metric in PITCH_METRICS + BIP_METRICS:
            t = results[year][results[year]['metric'] == metric]
            if t.empty:
                continue
            unit = 'BIP' if metric in BIP_METRICS else 'pitches'
            parts = [f"{int(x['lo'])}-{int(x['hi'])}:{x['R_full']:.2f}"
                     for _, x in t.iterrows()]
            # bracket the 0.5 crossing
            above = t[t['R_full'] >= 0.5]
            cross = (f"crosses in {int(above.iloc[0]['lo'])}-{int(above.iloc[0]['hi'])}"
                     if len(above) and above.iloc[0]['lo'] > t.iloc[0]['lo']
                     else ('below 0.5 everywhere' if not len(above)
                           else f"already >=0.5 at {int(t.iloc[0]['lo'])}+"))
            print(f"  {metric:14s} ({unit}): {' '.join(parts)}   [{cross}]")


if __name__ == '__main__':
    main()
