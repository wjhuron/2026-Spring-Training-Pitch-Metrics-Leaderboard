"""battrack_prior_battery.py — is bat tracking worth anything to Hitter+
as a SMALL-SAMPLE PRIOR (the kin_eff pattern)?

WHAT IS ALREADY SETTLED. The 4th-component question is closed: all seven
bat-tracking candidates failed the pre-registered 0.15 partial gate at
full-season samples (hitter_battrack_screen.py, 2026-08-14; best
fast_rate +0.117). This battery tests the mechanism that screen did NOT
cover: bat speed is knowable within dozens of swings while BB+'s
ingredients need ~50+ BIP, so bat tracking could inform the SHRINKAGE
TARGET for a callup's first weeks, then wash out as real BIPs accumulate.

PRE-REGISTERED PROTOCOL (2026-08-21, before any result was seen):
  Universe   MLB hitter-seasons 2024 / 2025 / 2026 with >= 100 qualified
             tracked swings (data/_bt_seasons.json) and >= 150 non-bunt
             BIP; 2024-25 from the statcast caches (batter side), 2026
             from the sheet pickle (names bridged to MLBAM ids via the
             bat-tracking leaderboard).
  Estimand   BB+ raw on the REMAINDER of the season after his first n
             BIPs (n in {20, 40, 80}, chronological — the callup
             simulation), remainder >= 100 BIP. BB+ raw recipe = the
             shipped one: 0.30 * con100 (shrunk n0=200 toward 100) +
             0.70 * ev95_100 (unshrunk), league-scaled per season.
  Predictors bs   = avg_sweetspot_speed_mph      (registered)
             fast = avg_is_sweetspot_speed_high  (registered)
             squp = squared_up_per_swing         (registered)
             attack_angle                        (exploratory, report only)
  T1 (information, knob-free): partial r of each registered predictor
     with the remainder target, controlling the first-n shipped estimate.
     The stabilizer signature: positive partials at n=20, fading by n=80.
  T2 (estimator, no hand knobs): leave-season-out OLS —
     remainder ~ E0(first-n)            vs
     remainder ~ E0(first-n) + bs + fast + squp,
     coefficients fit on the OTHER seasons, RMSE evaluated held-out.
  DECISION   adoption-grade only if T1 registered partials are positive
             in >= 2/3 seasons at n=20 AND T2 held-out RMSE improves in
             >= 2/3 seasons at n in {20, 40}. If passed, the production
             mechanism (prior target + its n0) is designed and swept in a
             follow-up with constants measured, never estimated. If
             failed, bat tracking stays out of Hitter+ entirely and the
             result is recorded next to the 4th-atom rejection.

Usage: PYTHONHASHSEED=0 python3 scripts/research/hitter/battrack_prior_battery.py
Output: data/_battrack_prior_results.json + printed tables.
"""
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

D = lambda n: json.load(open(os.path.join(ROOT, 'data', n)))
SEASONS = (2024, 2025, 2026)
N_LEVELS = (20, 40, 80)
MIN_SWINGS = 100
MIN_BIP = 150
MIN_REMAINDER = 100
N0_CON = 200
W_EV = 0.70
BUNT_BB = {'bunt_grounder', 'bunt_popup', 'bunt_line_drive'}
SH_EV = {'sac_bunt', 'sac_bunt_double_play'}


def bip_frame_cache(year):
    p = os.path.join(ROOT, 'data', f'_statcast{year}_cache.pkl')
    df = pickle.load(open(p, 'rb'))
    if 'game_type' in df.columns:
        df = df[df['game_type'] == 'R']
    ev = df['events'].where(df['events'].astype(str).str.len() > 0)
    bip = ((df['description'] == 'hit_into_play')
           & df['bb_type'].fillna('').ne('')
           & ~df['bb_type'].isin(BUNT_BB)
           & ~ev.isin(SH_EV).fillna(False))
    out = pd.DataFrame({
        'hid': df.loc[bip, 'batter'].astype(int).astype(str),
        'date': df.loc[bip, 'game_date'].astype(str),
        'xw': pd.to_numeric(
            df.loc[bip, 'estimated_woba_using_speedangle'], errors='coerce'),
        'ev': pd.to_numeric(df.loc[bip, 'launch_speed'], errors='coerce'),
    })
    return out.dropna(subset=['ev']).sort_values('date', kind='stable')


def bip_frame_sheet():
    raw = pd.read_pickle(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'))
    df = pd.DataFrame(raw)
    df = df[(df['PTeam'] != 'ROC') & (df['BTeam'] != 'ROC')]
    bb = df['BBType'].fillna('').astype(str)
    bip = ((df['Description'] == 'In Play') & bb.ne('')
           & ~bb.isin(BUNT_BB))
    # name -> MLBAM id bridge from the bat-tracking leaderboard pull
    bt = D('_bt_seasons.json')
    name_to_id = {}
    for y in ('2026', '2025', '2024'):
        for pid, row in bt.get(y, {}).items():
            nm = row.get('name')
            if nm:
                name_to_id.setdefault(nm, pid)
    out = pd.DataFrame({
        'name': df.loc[bip, 'Batter'].astype(str),
        'date': df.loc[bip, 'Game Date'].astype(str),
        'xw': pd.to_numeric(df.loc[bip, 'xwOBA'], errors='coerce'),
        'ev': pd.to_numeric(df.loc[bip, 'ExitVelo'], errors='coerce'),
    })
    out['hid'] = out['name'].map(name_to_id)
    return (out.dropna(subset=['ev', 'hid'])
            .sort_values('date', kind='stable'))


def bb_raw(xw, ev, lg_con, lg_ev95, shrink_con=True):
    """The shipped BB+ raw recipe on an arbitrary BIP sample."""
    xw = xw[np.isfinite(xw)]
    con100 = 100.0 * (xw.mean() / lg_con) if len(xw) else np.nan
    if shrink_con and np.isfinite(con100):
        con100 = (con100 * len(xw) + N0_CON * 100.0) / (len(xw) + N0_CON)
    ev95 = np.percentile(ev, 95) if len(ev) else np.nan
    ev100 = 100.0 * ev95 / lg_ev95
    return (1 - W_EV) * con100 + W_EV * ev100


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 25:
        return None, int(m.sum())
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


def partial_r(x, y, ctrl):
    x, y, c = (np.asarray(v, float) for v in (x, y, ctrl))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    if m.sum() < 25:
        return None
    rx = x[m] - np.poly1d(np.polyfit(c[m], x[m], 1))(c[m])
    ry = y[m] - np.poly1d(np.polyfit(c[m], y[m], 1))(c[m])
    r, _ = pearson(rx, ry)
    return r


def main():
    bt = D('_bt_seasons.json')
    tables = {}
    for y in SEASONS:
        bips = bip_frame_sheet() if y == 2026 else bip_frame_cache(y)
        bty = bt.get(str(y), {})
        lg_con = float(bips['xw'].mean())
        recs = []
        for hid, g in bips.groupby('hid'):
            if len(g) < MIN_BIP:
                continue
            row = bty.get(str(hid))
            if not row or (row.get('swings_qualified') or 0) < MIN_SWINGS:
                continue
            recs.append({'hid': hid, 'g': g, 'bs': row['avg_sweetspot_speed_mph'],
                         'fast': row['avg_is_sweetspot_speed_high'],
                         'squp': row['squared_up_per_swing'],
                         'aa': row.get('attack_angle')})
        if len(recs) < 40:
            print(f'  {y}: pool {len(recs)} too thin — skipped')
            continue
        # league ev95 = BIP-weighted mean of full-season hitter ev95s
        lg_ev95 = float(np.average(
            [np.percentile(r['g']['ev'], 95) for r in recs],
            weights=[len(r['g']) for r in recs]))
        rows = []
        for r in recs:
            g = r['g']
            rec = {'hid': r['hid'], 'bs': r['bs'], 'fast': r['fast'],
                   'squp': r['squp'], 'aa': r['aa']}
            ok = False
            for n in N_LEVELS:
                if len(g) - n < MIN_REMAINDER:
                    continue
                first, rest = g.iloc[:n], g.iloc[n:]
                rec[f'e0_{n}'] = bb_raw(first['xw'].values, first['ev'].values,
                                        lg_con, lg_ev95)
                rec[f'tgt_{n}'] = bb_raw(rest['xw'].values, rest['ev'].values,
                                         lg_con, lg_ev95, shrink_con=False)
                ok = True
            if ok:
                rows.append(rec)
        tables[y] = pd.DataFrame(rows)
        print(f'  {y}: {len(rows)} hitter-seasons '
              f'(lg_con {lg_con:.3f}, lg_ev95 {lg_ev95:.1f})')

    print('\n── T1: partial r with remainder target | first-n estimate ──')
    t1 = defaultdict(dict)
    for y, t in tables.items():
        for n in N_LEVELS:
            if f'e0_{n}' not in t:
                continue
            parts = {p: partial_r(t[p], t[f'tgt_{n}'], t[f'e0_{n}'])
                     for p in ('bs', 'fast', 'squp', 'aa')}
            t1[y][n] = parts
            e0r, nn = pearson(t[f'e0_{n}'], t[f'tgt_{n}'])
            print(f'  {y} n={n:<3} (pool {nn}, E0 r={e0r:+.3f}): '
                  + '  '.join(f'{p}={v:+.3f}' if v is not None else f'{p}=—'
                              for p, v in parts.items()))

    print('\n── T2: leave-season-out OLS, held-out RMSE ──')
    t2 = defaultdict(dict)
    for y in tables:
        others = [tables[o] for o in tables if o != y]
        if not others:
            continue
        tr = pd.concat(others, ignore_index=True)
        te = tables[y]
        for n in N_LEVELS:
            cols = [f'e0_{n}', 'bs', 'fast', 'squp', f'tgt_{n}']
            trn = tr.dropna(subset=[c for c in cols if c in tr])
            ten = te.dropna(subset=[c for c in cols if c in te])
            if len(trn) < 60 or len(ten) < 25:
                continue

            def fit_pred(feat):
                X = np.column_stack([np.ones(len(trn))]
                                    + [np.asarray(trn[f], float) for f in feat])
                yv_tr = np.asarray(trn[f'tgt_{n}'], float)
                b, *_ = np.linalg.lstsq(X, yv_tr, rcond=None)
                Xt = np.column_stack([np.ones(len(ten))]
                                     + [np.asarray(ten[f], float) for f in feat])
                return Xt @ b

            base = fit_pred([f'e0_{n}'])
            full = fit_pred([f'e0_{n}', 'bs', 'fast', 'squp'])
            yv = ten[f'tgt_{n}'].values
            rmse0 = float(np.sqrt(np.mean((yv - base) ** 2)))
            rmse1 = float(np.sqrt(np.mean((yv - full) ** 2)))
            t2[y][n] = {'rmse_base': rmse0, 'rmse_bat': rmse1,
                        'n_test': len(ten)}
            print(f'  held-out {y} n={n:<3}: base {rmse0:.3f} -> '
                  f'+bat {rmse1:.3f}  '
                  f'({"improves" if rmse1 < rmse0 else "worse"}, '
                  f'test n={len(ten)})')

    # decision per pre-registration
    t1_pass = sum(1 for y in t1 if 20 in t1[y]
                  and all((t1[y][20][p] or 0) > 0
                          for p in ('bs', 'fast', 'squp')))
    t2_cells = [(y, n) for y in t2 for n in (20, 40) if n in t2[y]]
    t2_wins = sum(1 for y, n in t2_cells
                  if t2[y][n]['rmse_bat'] < t2[y][n]['rmse_base'])
    print(f'\n  T1: all three registered partials positive at n=20 in '
          f'{t1_pass}/{len(t1)} seasons')
    print(f'  T2: held-out RMSE improves in {t2_wins}/{len(t2_cells)} '
          f'season x n cells')
    out = {'t1': {str(y): {str(n): t1[y][n] for n in t1[y]} for y in t1},
           't2': {str(y): {str(n): t2[y][n] for n in t2[y]} for y in t2},
           'protocol': 'module docstring'}
    with open(os.path.join(ROOT, 'data',
                           '_battrack_prior_results.json'), 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print('  saved data/_battrack_prior_results.json')


if __name__ == '__main__':
    main()
