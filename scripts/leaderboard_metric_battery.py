#!/usr/bin/env python3
"""leaderboard_metric_battery.py — predictiveness/utility tests for every
historically-computable leaderboard metric, 2021-2026.

For the leaderboard-cleanup decision (2026-07-19). Per metric:
  reliability   split-half r within season (odd/even game split per player),
                Spearman-Brown corrected to full season
  stabilize_n   pitches/PA where reliability crosses 0.5 (from split-half r)
  yoy_r         same metric year N vs year N+1 (within player)
  pred_r        metric year N -> TARGET year N+1
                (pitchers: PA-level xwOBA against; hitters: PA-level xwOBA)
  curr_r        metric vs same-season target (concurrent validity)
Plus a same-tab redundancy correlation matrix (2025 full season).

Data: data/_statcast{2021..2025}_cache.pkl + _statcast2026_full.pkl
(Savant tags for 21-25 — tag-sensitive metrics carry a caveat flag in the
report; 2026 is the retagged-adjacent check).

Usage:
  python3 scripts/leaderboard_metric_battery.py --build     (~5-10 min)
  python3 scripts/leaderboard_metric_battery.py --analyze
Outputs: <scratch>/battery_tables.pkl, battery_results_{pitcher,hitter,arsenal}.csv,
battery_redundancy_{pitcher,hitter}.csv
"""
import argparse
import math
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline_utils import BALL_RADIUS_FT, ZONE_HALF_WIDTH  # noqa: E402

SCRATCH = os.environ.get(
    'BATTERY_OUT',
    '/private/tmp/claude-501/-Users-wallyhuron-Huronalytics/'
    '05c10437-7098-4dec-bf5b-10288eedf936/scratchpad')
# 2026 excluded: _statcast2026_full.pkl is a 22-col supplement cache (no
# events/ids). The 2026 retagged check runs separately on the site's own
# pitch cache (grade atoms, bat tracking) — see the cleanup plan.
SEASONS = [2021, 2022, 2023, 2024, 2025]

# FanGraphs wOBA weights (uBB, HBP, 1B, 2B, 3B, HR) — exactness irrelevant
# for correlation tests; 2026 reuses 2025.
W = {
    2021: (0.692, 0.722, 0.879, 1.242, 1.568, 2.007),
    2022: (0.689, 0.720, 0.884, 1.261, 1.601, 2.072),
    2023: (0.696, 0.726, 0.883, 1.244, 1.569, 2.004),
    2024: (0.689, 0.720, 0.882, 1.254, 1.590, 2.050),
    2025: (0.693, 0.723, 0.882, 1.250, 1.580, 2.030),
    2026: (0.693, 0.723, 0.882, 1.250, 1.580, 2.030),
}

WHIFF = {'swinging_strike', 'swinging_strike_blocked', 'foul_tip'}
SWING = WHIFF | {'foul', 'hit_into_play'}          # bunts excluded (site rule)
BUNTS = {'foul_bunt', 'missed_bunt', 'bunt_foul_tip'}
BALLS = {'ball', 'blocked_ball', 'intent_ball', 'pitchout', 'hit_by_pitch'}
NON_PA_EV = {
    'caught_stealing_2b', 'caught_stealing_3b', 'caught_stealing_home',
    'pickoff_1b', 'pickoff_2b', 'pickoff_3b', 'pickoff_caught_stealing_2b',
    'pickoff_caught_stealing_3b', 'pickoff_caught_stealing_home',
    'stolen_base_2b', 'stolen_base_3b', 'stolen_base_home',
    'wild_pitch', 'passed_ball', 'game_advisory', 'ejection', 'other_advance'}
K_EV = {'strikeout', 'strikeout_double_play'}
BB_EV = {'walk', 'intent_walk'}
HIT_EV = {'single', 'double', 'triple', 'home_run'}
SF_EV = {'sac_fly', 'sac_fly_double_play'}
SH_EV = {'sac_bunt', 'sac_bunt_double_play'}

LA_BINS = [-999, -10, 0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 999]


def load_season(year):
    fn = ('_statcast2026_full.pkl' if year == 2026
          else f'_statcast{year}_cache.pkl')
    df = pickle.load(open(os.path.join(ROOT, 'data', fn), 'rb'))
    if 'game_type' in df.columns:
        df = df[df['game_type'] == 'R']
    df = df.copy()
    for c in ('release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z',
              'release_extension', 'arm_angle', 'release_pos_x',
              'release_pos_z', 'plate_x', 'plate_z', 'sz_top', 'sz_bot',
              'launch_speed', 'launch_angle', 'hc_x', 'hc_y',
              'hit_distance_sc', 'estimated_woba_using_speedangle',
              'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
              'delta_run_exp', 'delta_pitcher_run_exp',
              'balls', 'strikes'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    d = df['description'].fillna('')
    ev = df['events'].where(df['events'].astype(str).str.len() > 0)
    df['is_bunt_desc'] = d.isin(BUNTS)
    df['swing'] = d.isin(SWING)
    df['whiff'] = d.isin(WHIFF)
    df['csw'] = d.isin(WHIFF | {'called_strike'})
    df['strike_any'] = ~d.isin(BALLS)                    # bunt fouls count
    df['iz'] = ((df['plate_x'].abs() <= ZONE_HALF_WIDTH)
                & (df['plate_z'] >= df['sz_bot'] - BALL_RADIUS_FT)
                & (df['plate_z'] <= df['sz_top'] + BALL_RADIUS_FT))
    df['ooz'] = ~df['iz'] & df['plate_x'].notna() & df['plate_z'].notna()
    df['chase_sw'] = df['ooz'] & df['swing']
    bb = df['bb_type'].fillna('')
    df['bip'] = (d == 'hit_into_play') & bb.ne('') & ~ev.isin(SH_EV).fillna(False)
    df['gb'] = df['bip'] & (bb == 'ground_ball')
    df['ld'] = df['bip'] & (bb == 'line_drive')
    df['fb'] = df['bip'] & (bb == 'fly_ball')
    df['pu'] = df['bip'] & (bb == 'popup')
    df['pa_end'] = ev.notna() & ~ev.isin(NON_PA_EV)
    df['k'] = ev.isin(K_EV).fillna(False)
    df['bbw'] = ev.isin(BB_EV).fillna(False)
    df['hbp'] = (ev == 'hit_by_pitch').fillna(False)
    df['hit'] = ev.isin(HIT_EV).fillna(False)
    df['hr'] = (ev == 'home_run').fillna(False)
    df['sf'] = ev.isin(SF_EV).fillna(False)
    df['sh'] = ev.isin(SH_EV).fillna(False)
    df['single'] = (ev == 'single').fillna(False)
    df['double'] = (ev == 'double').fillna(False)
    df['triple'] = (ev == 'triple').fillna(False)
    wbb, whb, w1, w2, w3, whr = W[year]
    df['woba_num'] = (df['bbw'] * wbb + df['hbp'] * whb + df['single'] * w1
                      + df['double'] * w2 + df['triple'] * w3 + df['hr'] * whr)
    df['woba_den'] = (df['pa_end'] & ~df['sh']).astype(float)
    # PA-level xwOBA (the target family): BIP -> est_woba; K -> 0; BB/HBP -> weight
    xw = df['estimated_woba_using_speedangle']
    df['paxw_num'] = np.where(
        df['pa_end'] & df['bip'] & xw.notna(), xw,
        np.where(df['k'], 0.0,
                 np.where(df['bbw'], wbb, np.where(df['hbp'], whb, np.nan))))
    df['paxw_den'] = (df['pa_end'] & (df['bip'] & xw.notna()
                      | df['k'] | df['bbw'] | df['hbp'])).astype(float)
    df.loc[df['paxw_den'] == 0, 'paxw_num'] = np.nan
    # run value, pitcher perspective (site convention)
    if 'delta_pitcher_run_exp' in df.columns:
        df['rv'] = df['delta_pitcher_run_exp'].where(
            df['delta_pitcher_run_exp'].notna(), -df['delta_run_exp'])
    else:
        df['rv'] = -df['delta_run_exp']
    # movement / physics
    df['ivb'] = df['pfx_z'] * 12.0
    df['hb'] = df['pfx_x'] * 12.0
    yf = 17.0 / 12.0
    vyf = -np.sqrt(np.maximum(df['vy0'] ** 2 + 2 * df['ay'] * (yf - 50.0), 0))
    t = (vyf - df['vy0']) / df['ay']
    df['vaa'] = -np.degrees(np.arctan((df['vz0'] + df['az'] * t) / vyf))
    df['haa'] = -np.degrees(np.arctan((df['vx0'] + df['ax'] * t) / vyf))
    # barrel approximation (official flag not in cache — caveat in report)
    evla_ok = df['launch_speed'].notna() & df['launch_angle'].notna()
    evv, la = df['launch_speed'], df['launch_angle']
    lo = np.maximum(26 - (evv - 98), 8)
    hi = np.minimum(30 + (evv - 98) * 2, 50)
    df['barrel'] = evla_ok & df['bip'] & (evv >= 98) & (la >= lo) & (la <= hi)
    df['hardhit'] = evla_ok & df['bip'] & (evv >= 95)
    # spray
    dx = df['hc_x'] - 125.42
    dy = 198.27 - df['hc_y']
    ang = np.degrees(np.arctan2(dx, dy)).where(dy > 0)
    side = np.where(df['stand'] == 'L', -1.0, 1.0)      # pull = negative x RHB
    rel = ang * -side                                   # >0 toward pull side
    df['pull'] = df['bip'] & (rel > 15)
    df['middle'] = df['bip'] & (rel >= -15) & (rel <= 15)
    df['oppo'] = df['bip'] & (rel < -15)
    df['airpull'] = df['pull'] & (df['ld'] | df['fb'])
    df['la_bin'] = pd.cut(df['launch_angle'], LA_BINS, right=False, labels=False)
    for c in ('bip', 'gb', 'ld', 'fb', 'pu', 'pa_end', 'barrel', 'hardhit',
              'pull', 'middle', 'oppo', 'airpull'):
        df[c] = df[c].fillna(False).astype(bool)
    df['spray3'] = np.select(
        [df['pull'].to_numpy(), df['middle'].to_numpy(), df['oppo'].to_numpy()],
        ['pull', 'middle', 'oppo'], default=None)
    # counts
    df['c00'] = (df['balls'] == 0) & (df['strikes'] == 0)
    df['fps'] = df['c00'] & df['strike_any'] & ~d.isin({'hit_by_pitch'})
    df['c11'] = (df['balls'] == 1) & (df['strikes'] == 1)
    df['win11'] = df['c11'] & df['strike_any']
    df['s2'] = df['strikes'] == 2
    df['ea'] = df['pa_end'] & ((df['balls'] + df['strikes']) <= 2)
    # BIP wOBA value (for xwOBAsp league tables)
    df['bip_woba'] = np.where(df['bip'],
                              np.where(df['hit'], df['woba_num'], 0.0), np.nan)
    return df


def add_half(df, pid_col):
    g = (df[[pid_col, 'game_pk', 'game_date']]
         .drop_duplicates([pid_col, 'game_pk'])
         .sort_values([pid_col, 'game_date', 'game_pk']))
    g['gidx'] = g.groupby(pid_col).cumcount()
    g['half'] = np.where(g['gidx'] % 2 == 0, 'A', 'B')
    return df.merge(g[[pid_col, 'game_pk', 'half']], on=[pid_col, 'game_pk'])


def spray_tables(df):
    b = df[df['bip'] & df['la_bin'].notna() & pd.notna(df['spray3'])
           & df['bip_woba'].notna()]
    zone = (b.groupby(['stand', 'spray3', 'la_bin'])['bip_woba']
            .agg(['mean', 'size']))
    zone = zone[zone['size'] >= 20]['mean']
    marg = (b.groupby(['stand', 'la_bin'])['bip_woba'].agg(['mean', 'size']))
    marg = marg[marg['size'] >= 20]['mean']
    return zone, marg


def rate_aggs(g):
    n = g['swing'].size
    sw = g['swing'].sum()
    ooz = g['ooz'].sum()
    izn = g['iz'].sum()
    iz_sw = (g['iz'] & g['swing']).sum()
    s2sw = (g['s2'] & g['swing']).sum()
    pa = g['pa_end'].sum()
    ab = pa - g['bbw'].sum() - g['hbp'].sum() - g['sf'].sum() - g['sh'].sum()
    bipn = g['bip'].sum()
    fbpu = g['fb'].sum() + g['pu'].sum()
    h = g['hit'].sum()
    hr = g['hr'].sum()
    k = g['k'].sum()
    evs = g.loc[g['bip'], 'launch_speed'].dropna()
    ev50 = evs[evs >= evs.median()].mean() if len(evs) >= 10 else np.nan
    c00 = g['c00'].sum()
    c11 = g['c11'].sum()
    babip_den = ab - k - hr + g['sf'].sum()
    out = {
        'n': n, 'pa': pa, 'nbip': bipn,
        'kPct': k / pa if pa else np.nan,
        'bbPct': g['bbw'].sum() / pa if pa else np.nan,
        'kbbPct': (k - g['bbw'].sum()) / pa if pa else np.nan,
        'woba': g['woba_num'].sum() / g['woba_den'].sum()
                if g['woba_den'].sum() else np.nan,
        'paxw': g['paxw_num'].mean(),
        'xwobacon': g.loc[g['bip'],
                          'estimated_woba_using_speedangle'].mean(),
        'babip': (h - hr) / babip_den if babip_den > 0 else np.nan,
        'avg': h / ab if ab else np.nan,
        'obp': (h + g['bbw'].sum() + g['hbp'].sum())
               / (ab + g['bbw'].sum() + g['hbp'].sum() + g['sf'].sum())
               if pa else np.nan,
        'slg': (g['single'].sum() + 2 * g['double'].sum()
                + 3 * g['triple'].sum() + 4 * hr) / ab if ab else np.nan,
        'iso': (g['double'].sum() + 2 * g['triple'].sum() + 3 * hr) / ab
               if ab else np.nan,
        'rv100': g['rv'].sum() / n * 100 if n else np.nan,
        'swingPct': sw / n if n else np.nan,
        'whiffPct': g['whiff'].sum() / sw if sw else np.nan,
        'cswPct': g['csw'].sum() / n if n else np.nan,
        'chasePct': g['chase_sw'].sum() / ooz if ooz else np.nan,
        'izPct': izn / n if n else np.nan,
        'izSwingPct': iz_sw / izn if izn else np.nan,
        'izWhiffPct': (g['iz'] & g['whiff']).sum() / iz_sw if iz_sw else np.nan,
        'izContactPct': 1 - (g['iz'] & g['whiff']).sum() / iz_sw
                        if iz_sw else np.nan,
        'contactPct': 1 - g['whiff'].sum() / sw if sw else np.nan,
        'twoStrikeWhiffPct': (g['s2'] & g['whiff']).sum() / s2sw
                             if s2sw else np.nan,
        'strikePct': g['strike_any'].sum() / n if n else np.nan,
        'fpsPct': g['fps'].sum() / c00 if c00 else np.nan,
        'fpSwingPct': (g['c00'] & g['swing']).sum() / c00 if c00 else np.nan,
        'oneOneWinPct': g['win11'].sum() / c11 if c11 else np.nan,
        'earlyActionPct': g['ea'].sum() / pa if pa else np.nan,
        'avgEV': evs.mean() if len(evs) else np.nan,
        'maxEV': evs.max() if len(evs) else np.nan,
        'ev50': ev50,
        'medLA': g.loc[g['bip'], 'launch_angle'].median(),
        'hardHitPct': g['hardhit'].sum() / bipn if bipn else np.nan,
        'barrelPct': g['barrel'].sum() / bipn if bipn else np.nan,
        'gbPct': g['gb'].sum() / bipn if bipn else np.nan,
        'ldPct': g['ld'].sum() / bipn if bipn else np.nan,
        'fbPct': g['fb'].sum() / bipn if bipn else np.nan,
        'puPct': g['pu'].sum() / bipn if bipn else np.nan,
        'hrFbPct': hr / fbpu if fbpu else np.nan,
        'pullPct': g['pull'].sum() / bipn if bipn else np.nan,
        'middlePct': g['middle'].sum() / bipn if bipn else np.nan,
        'oppoPct': g['oppo'].sum() / bipn if bipn else np.nan,
        'airPullPct': g['airpull'].sum() / bipn if bipn else np.nan,
        'avgFbDist': g.loc[g['fb'], 'hit_distance_sc'].mean(),
        'avgHrDist': g.loc[g['hr'], 'hit_distance_sc'].mean(),
    }
    out['izSwChase'] = (out['izSwingPct'] - out['chasePct']
                        if not (math.isnan(out['izSwingPct'])
                                or math.isnan(out['chasePct'])) else np.nan)
    out['bbToK'] = (g['bbw'].sum() / k) if k else np.nan
    out['ops'] = (out['obp'] + out['slg']
                  if not (math.isnan(out['obp']) or math.isnan(out['slg']))
                  else np.nan)
    return pd.Series(out)


def sp_aggs(g, zone, marg):
    """xwOBAsp + sprayVal per group from precomputed league tables."""
    b = g[g['bip'] & g['la_bin'].notna() & pd.notna(g['spray3'])]
    if not len(b):
        return pd.Series({'xwobasp': np.nan, 'sprayVal': np.nan})
    zv = [zone.get((st, s3, lb), np.nan)
          for st, s3, lb in zip(b['stand'], b['spray3'], b['la_bin'])]
    mv = [marg.get((st, lb), np.nan)
          for st, lb in zip(b['stand'], b['la_bin'])]
    zv, mv = np.array(zv, float), np.array(mv, float)
    ok = ~np.isnan(zv) & ~np.isnan(mv)
    return pd.Series({
        'xwobasp': np.nanmean(zv) if np.any(~np.isnan(zv)) else np.nan,
        'sprayVal': (zv[ok] - mv[ok]).mean() if ok.sum() else np.nan})


ARS_METRICS = ['release_speed', 'release_spin_rate', 'ivb', 'hb',
               'release_extension', 'arm_angle', 'release_pos_z',
               'release_pos_x', 'vaa', 'haa']


def ars_aggs(g):
    n = g['swing'].size
    sw = g['swing'].sum()
    ooz = g['ooz'].sum()
    bipn = g['bip'].sum()
    out = {'n': n}
    for m in ARS_METRICS:
        out[m] = g[m].mean()
    out['maxVelo'] = g['release_speed'].max()
    out['whiffPct'] = g['whiff'].sum() / sw if sw else np.nan
    out['cswPct'] = g['csw'].sum() / n if n else np.nan
    out['chasePct'] = g['chase_sw'].sum() / ooz if ooz else np.nan
    out['rv100'] = g['rv'].sum() / n * 100 if n else np.nan
    out['paxw'] = g['paxw_num'].mean()
    out['xwobacon'] = g.loc[g['bip'], 'estimated_woba_using_speedangle'].mean()
    out['barrelPct'] = g['barrel'].sum() / bipn if bipn else np.nan
    return pd.Series(out)


def build():
    pit_all, bat_all, ars_all = [], [], []
    for year in SEASONS:
        print(f'── {year}')
        df = load_season(year)
        zone, marg = spray_tables(df)
        for side, pid in (('pitcher', 'pitcher'), ('batter', 'batter')):
            dfx = add_half(df, pid)
            for half_label, sub in (('full', dfx), ('A', dfx[dfx['half'] == 'A']),
                                    ('B', dfx[dfx['half'] == 'B'])):
                r = sub.groupby(pid, group_keys=False).apply(rate_aggs)
                sp = sub.groupby(pid, group_keys=False).apply(
                    sp_aggs, zone=zone, marg=marg)
                r = r.join(sp)
                r['season'] = year
                r['half'] = half_label
                r.index.name = 'pid'
                (pit_all if side == 'pitcher' else bat_all).append(r.reset_index())
        dfa = add_half(df[df['pitch_type'].notna()], 'pitcher')
        for half_label, sub in (('full', dfa), ('A', dfa[dfa['half'] == 'A']),
                                ('B', dfa[dfa['half'] == 'B'])):
            r = sub.groupby(['pitcher', 'pitch_type'],
                            group_keys=False).apply(ars_aggs)
            r['season'] = year
            r['half'] = half_label
            ars_all.append(r.reset_index())
        print(f'   pitchers {len(pit_all[-3])}, batters {len(bat_all[-3])}, '
              f'arsenal {len(ars_all[-3])}')
    out = {'pitcher': pd.concat(pit_all, ignore_index=True),
           'batter': pd.concat(bat_all, ignore_index=True),
           'arsenal': pd.concat(ars_all, ignore_index=True)}
    with open(os.path.join(SCRATCH, 'battery_tables.pkl'), 'wb') as f:
        pickle.dump(out, f)
    print('saved battery_tables.pkl')


# ── analysis ─────────────────────────────────────────────────────────────
PIT_MIN_FULL, PIT_MIN_HALF = 800, 300      # pitches
BAT_MIN_FULL, BAT_MIN_HALF = 1200, 450     # pitches seen (~300/120 PA)
ARS_MIN_FULL, ARS_MIN_HALF = 300, 120

PIT_METRICS = ['kPct', 'bbPct', 'kbbPct', 'woba', 'paxw', 'xwobacon', 'babip',
               'rv100', 'whiffPct', 'cswPct', 'chasePct', 'izPct',
               'izWhiffPct', 'twoStrikeWhiffPct', 'strikePct', 'fpsPct',
               'oneOneWinPct', 'earlyActionPct', 'avgEV', 'maxEV', 'ev50',
               'hardHitPct', 'barrelPct', 'gbPct', 'ldPct', 'fbPct', 'puPct',
               'hrFbPct', 'xwobasp', 'medLA']
BAT_METRICS = ['avg', 'obp', 'slg', 'ops', 'iso', 'bbPct', 'kPct', 'bbToK',
               'babip', 'woba', 'paxw', 'xwobacon', 'avgEV', 'ev50', 'maxEV',
               'medLA', 'hardHitPct', 'barrelPct', 'gbPct', 'ldPct', 'fbPct',
               'puPct', 'hrFbPct', 'pullPct', 'middlePct', 'oppoPct',
               'airPullPct', 'avgFbDist', 'avgHrDist', 'xwobasp', 'sprayVal',
               'swingPct', 'izSwingPct', 'chasePct', 'izSwChase',
               'fpSwingPct', 'whiffPct', 'twoStrikeWhiffPct', 'contactPct',
               'izContactPct']
ARS_COLS = ARS_METRICS + ['maxVelo', 'whiffPct', 'cswPct', 'chasePct',
                          'rv100', 'paxw', 'xwobacon', 'barrelPct']


def pear(a, b):
    a = pd.to_numeric(pd.Series(a), errors='coerce').to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(b), errors='coerce').to_numpy(dtype=float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 30:
        return np.nan, int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def analyze():
    with open(os.path.join(SCRATCH, 'battery_tables.pkl'), 'rb') as f:
        T = pickle.load(f)

    def run(table, metrics, keys, min_full, min_half, target):
        t = T[table]
        full = t[(t['half'] == 'full') & (t['n'] >= min_full)]
        A = t[(t['half'] == 'A') & (t['n'] >= min_half)]
        B = t[(t['half'] == 'B') & (t['n'] >= min_half)]
        ab = A.merge(B, on=keys + ['season'], suffixes=('_a', '_b'))
        pairs = full.merge(
            full.assign(season=full['season'] - 1),
            on=keys + ['season'], suffixes=('', '_n1'))
        rows = []
        for m in metrics:
            sh, n_sh = pear(ab[m + '_a'].values, ab[m + '_b'].values)
            sb = 2 * sh / (1 + sh) if sh == sh else np.nan
            half_n = float(np.nanmean(np.minimum(ab['n_a'], ab['n_b']))) \
                if len(ab) else np.nan
            stab = (half_n * (1 - sh) / sh if sh and sh > 0 else np.nan)
            yoy, n_yoy = pear(pairs[m].values, pairs[m + '_n1'].values)
            pred, _ = pear(pairs[m].values, pairs[target + '_n1'].values)
            curr, _ = pear(full[m].values, full[target].values)
            rows.append({'metric': m, 'split_half_r': sh, 'full_season_r': sb,
                         'stabilize_n': stab, 'yoy_r': yoy, 'pred_r': pred,
                         'curr_r': curr, 'n_splits': n_sh, 'n_pairs': n_yoy})
        res = pd.DataFrame(rows)
        res.to_csv(os.path.join(SCRATCH, f'battery_results_{table}.csv'),
                   index=False)
        # redundancy matrix (2025 full)
        f25 = full[full['season'] == 2025]
        corr = f25[metrics].apply(pd.to_numeric, errors='coerce').corr()
        corr.to_csv(os.path.join(SCRATCH, f'battery_redundancy_{table}.csv'))
        return res

    rp = run('pitcher', PIT_METRICS, ['pid'], PIT_MIN_FULL, PIT_MIN_HALF,
             'paxw')
    rb = run('batter', BAT_METRICS, ['pid'], BAT_MIN_FULL, BAT_MIN_HALF,
             'paxw')
    ra = run('arsenal', ARS_COLS, ['pitcher', 'pitch_type'],
             ARS_MIN_FULL, ARS_MIN_HALF, 'paxw')
    pd.set_option('display.width', 200)
    for name, res in (('PITCHER', rp), ('BATTER', rb), ('ARSENAL', ra)):
        print(f'\n════ {name} (target: next-season PA-xwOBA) ════')
        print(res.round(3).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--analyze', action='store_true')
    args = ap.parse_args()
    if args.build:
        build()
    if args.analyze:
        analyze()
