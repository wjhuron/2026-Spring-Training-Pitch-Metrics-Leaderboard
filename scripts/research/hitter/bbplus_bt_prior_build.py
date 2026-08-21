"""bbplus_bt_prior_build.py — production design + constants for the BB+
bat-tracking prior (follow-up the battery's pass mandates).

Everything here runs in PRODUCTION currency (pipeline/compute.py +
process_data definitions):
  batSpeed     mean per-swing BatSpeed at bs >= 50; nCompSwings = count
  squaredUpPct EV >= 0.80 * (0.212 * release velo + 1.23 * bat speed)
               among blast-eligible (bs >= 50, EV + velo present)
  BB+ raw      0.60 * shrink(conPlus, nBip, 130 -> tgt)
             + 0.40 * shrink(ev_c,    nBip,   0 -> tgt)
               conPlus = 100*mean(xwOBA on BIP)/lg, evPlus = 100*p95(EV)/lg,
               ev_c = 100 + (evPlus-100)*BETA, BETA = SD ratio per season
               pool (production measures it live; frozen 4.205 fallback)
               shipped tgt = 100 everywhere; the prior REPLACES tgt.

CAUSAL SIMULATION (fixes the battery's one look-ahead): at each n in
{20, 40, 80} first BIPs, the bat window is only swings on/before the nth
BIP's date — what the pipeline would actually know that morning.
Target: unshrunk raw composite on the remaining BIPs (>= 100).

FIT: prior = a + b*batSpeed + c*squaredUpPct on FULL-season unshrunk raw,
per season; applied leave-season-out; standardized-coefficient stability
reported. blastPct tested as a third predictor (report only).

DESIGNS + SWEEPS (grids pre-registered; interior optimum or stated-flat
required per the tuning rule):
  prior_eff = (nsw*prior + S0*100) / (nsw + S0)     S0 in {0,10,25,50,100}
  D1 composite blend: (nBip*E0 + N0BT*prior_eff) / (nBip + N0BT)
                                             N0BT in {0,5,10,20,40,80,160}
  D2 target substitution: shipped raw with tgt = prior_eff in BOTH
     channels, N0_EV re-swept in {0,5,10,20,40} (N0_CON held 130).
Objective: held-out RMSE vs the remainder target, per season x n cell.
Full-season invariance gate: winner applied at full season must keep
corr(new, shipped) >= 0.995 and mean |delta| small — the prior must wash
out as data accumulates.

Usage: PYTHONHASHSEED=0 python3 scripts/research/hitter/bbplus_bt_prior_build.py
Output: data/_bbplus_bt_prior_build.json + printed tables.
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
SEASONS = tuple(int(x) for x in os.environ.get('BT_SEASONS', '2024,2025,2026').split(','))
N_LEVELS = (20, 40, 80)
MIN_BIP, MIN_REMAINDER, MIN_SWINGS_FULL = 150, 100, 100
W_CON, W_EV, N0_CON, N0_EV = 0.60, 0.40, 130, 0
BETA_FROZEN, BETA_BAND = 4.205, (3.0, 6.0)
S0_GRID = (0, 10, 25, 50, 100)
N0BT_GRID = (0, 5, 10, 20, 40, 80, 160)
N0EV_GRID = (0, 5, 10, 20, 40)
BUNT_BB = {'bunt_grounder', 'bunt_popup', 'bunt_line_drive'}
SH_EV = {'sac_bunt', 'sac_bunt_double_play'}
SWING_DESC_SAVANT = {'swinging_strike', 'swinging_strike_blocked', 'foul',
                     'foul_tip', 'hit_into_play'}


def load_events_cache(year):
    """(swings, bips) per hitter from the statcast cache + battrack pull."""
    cache = pickle.load(open(
        os.path.join(ROOT, 'data', f'_statcast{year}_cache.pkl'), 'rb'))
    if 'game_type' in cache.columns:
        cache = cache[cache['game_type'] == 'R']
    ev_col = cache['events'].where(cache['events'].astype(str).str.len() > 0)
    bip = ((cache['description'] == 'hit_into_play')
           & cache['bb_type'].fillna('').ne('')
           & ~cache['bb_type'].isin(BUNT_BB)
           & ~ev_col.isin(SH_EV).fillna(False))
    bips = pd.DataFrame({
        'hid': cache.loc[bip, 'batter'].astype(int).astype(str),
        'date': cache.loc[bip, 'game_date'].astype(str),
        'xw': pd.to_numeric(cache.loc[bip, 'estimated_woba_using_speedangle'],
                            errors='coerce'),
        'ev': pd.to_numeric(cache.loc[bip, 'launch_speed'], errors='coerce'),
    }).dropna(subset=['ev']).sort_values('date', kind='stable')

    bt = pd.read_pickle(os.path.join(ROOT, 'data',
                                     f'_battrack_pitch_{year}.pkl'))
    bs = pd.to_numeric(bt['bat_speed'], errors='coerce')
    m = bs.notna() & (bs >= 50)
    swings = pd.DataFrame({
        'hid': bt.loc[m, 'batter'].astype(int).astype(str),
        'date': bt.loc[m, 'game_date'].astype(str),
        'bs': bs[m],
        'ev': pd.to_numeric(bt.loc[m, 'launch_speed'], errors='coerce'),
        'velo': pd.to_numeric(bt.loc[m, 'release_speed'], errors='coerce'),
    }).sort_values('date', kind='stable')
    return swings, bips


def load_events_sheet():
    raw = pd.read_pickle(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'))
    df = pd.DataFrame(raw)
    df = df[(df['PTeam'] != 'ROC') & (df['BTeam'] != 'ROC')]
    bt = D('_bt_seasons.json')
    name_to_id = {}
    for y in ('2026', '2025', '2024'):
        for pid, row in bt.get(y, {}).items():
            if row.get('name'):
                name_to_id.setdefault(row['name'], pid)
    df['hid'] = df['Batter'].astype(str).map(name_to_id)
    df = df.dropna(subset=['hid'])
    bb = df['BBType'].fillna('').astype(str)
    bip = (df['Description'] == 'In Play') & bb.ne('') & ~bb.isin(BUNT_BB)
    bips = pd.DataFrame({
        'hid': df.loc[bip, 'hid'],
        'date': df.loc[bip, 'Game Date'].astype(str),
        'xw': pd.to_numeric(df.loc[bip, 'xwOBA'], errors='coerce'),
        'ev': pd.to_numeric(df.loc[bip, 'ExitVelo'], errors='coerce'),
    }).dropna(subset=['ev']).sort_values('date', kind='stable')
    bs = pd.to_numeric(df['BatSpeed'], errors='coerce')
    m = bs.notna() & (bs >= 50)
    swings = pd.DataFrame({
        'hid': df.loc[m, 'hid'],
        'date': df.loc[m, 'Game Date'].astype(str),
        'bs': bs[m],
        'ev': pd.to_numeric(df.loc[m, 'ExitVelo'], errors='coerce'),
        'velo': pd.to_numeric(df.loc[m, 'Velocity'], errors='coerce'),
    }).sort_values('date', kind='stable')
    return swings, bips


def bat_metrics(sw):
    """Production batSpeed / squaredUpPct / blastPct on a swing window."""
    n = len(sw)
    if n == 0:
        return None
    bs_mean = float(sw['bs'].mean())
    elig = sw.dropna(subset=['ev', 'velo'])
    if len(elig):
        max_ev = 0.212 * elig['velo'] + 1.23 * elig['bs']
        squ = elig['ev'] >= 0.80 * max_ev
        squp = float(squ.mean())
        blast = float((squ & (elig['bs'] >= 75)).mean())
    else:
        squp = blast = None
    return {'bs': bs_mean, 'nsw': n, 'squp': squp, 'blast': blast}


def raw_parts(xw, ev, lg_con, lg_ev95, beta):
    con = 100.0 * np.nanmean(xw) / lg_con if len(xw) else np.nan
    ev95 = np.percentile(ev, 95) if len(ev) else np.nan
    ev_c = 100.0 + (100.0 * ev95 / lg_ev95 - 100.0) * beta
    return con, ev_c


def shipped_raw(con, ev_c, nbip, tgt_con=100.0, tgt_ev=100.0, n0_ev=N0_EV):
    c = (con * nbip + N0_CON * tgt_con) / (nbip + N0_CON)
    e = (ev_c * nbip + n0_ev * tgt_ev) / (nbip + n0_ev) if n0_ev else ev_c
    return W_CON * c + W_EV * e


def main():
    per_season = {}
    for y in SEASONS:
        swings, bips = (load_events_sheet() if y == 2026
                        else load_events_cache(y))
        # pool: full-season league values + BETA (production: SD ratio)
        rows = []
        for hid, g in bips.groupby('hid'):
            if len(g) < MIN_BIP:
                continue
            sw = swings[swings['hid'] == hid]
            if len(sw) < MIN_SWINGS_FULL:
                continue
            rows.append((hid, g, sw))
        lg_con = float(np.nanmean(
            np.concatenate([g['xw'].values for _, g, _ in rows])))
        ev95s = np.array([np.percentile(g['ev'], 95) for _, g, _ in rows])
        wts = np.array([len(g) for _, g, _ in rows], float)
        lg_ev95 = float(np.average(ev95s, weights=wts))
        cons = np.array([100 * np.nanmean(g['xw']) / lg_con
                         for _, g, _ in rows])
        evps = 100.0 * ev95s / lg_ev95
        beta = float(np.std(cons) / np.std(evps))
        if not (BETA_BAND[0] <= beta <= BETA_BAND[1]):
            beta = BETA_FROZEN
        recs = []
        for hid, g, sw in rows:
            bm_full = bat_metrics(sw)
            if bm_full is None or bm_full['squp'] is None:
                continue
            con_f, evc_f = raw_parts(g['xw'].values, g['ev'].values,
                                     lg_con, lg_ev95, beta)
            rec = {'hid': hid, 'nbip_full': len(g),
                   'bs_full': bm_full['bs'], 'squp_full': bm_full['squp'],
                   'blast_full': bm_full['blast'], 'nsw_full': bm_full['nsw'],
                   'con_full': con_f, 'evc_full': evc_f,
                   'raw_full': W_CON * con_f + W_EV * evc_f}
            for n in N_LEVELS:
                if len(g) - n < MIN_REMAINDER:
                    continue
                first, rest = g.iloc[:n], g.iloc[n:]
                cutdate = first['date'].iloc[-1]
                win = sw[sw['date'] <= cutdate]
                bm = bat_metrics(win)
                con_n, evc_n = raw_parts(first['xw'].values,
                                         first['ev'].values,
                                         lg_con, lg_ev95, beta)
                con_r, evc_r = raw_parts(rest['xw'].values, rest['ev'].values,
                                         lg_con, lg_ev95, beta)
                rec[f'con_{n}'], rec[f'evc_{n}'] = con_n, evc_n
                rec[f'tgt_{n}'] = W_CON * con_r + W_EV * evc_r
                rec[f'bs_{n}'] = bm['bs'] if bm else np.nan
                rec[f'squp_{n}'] = (bm['squp'] if bm and bm['squp'] is not None
                                    else np.nan)
                rec[f'nsw_{n}'] = bm['nsw'] if bm else 0
            recs.append(rec)
        per_season[y] = pd.DataFrame(recs)
        print(f'  {y}: {len(recs)} hitter-seasons, beta {beta:.2f}, '
              f'lg_con {lg_con:.3f}, lg_ev95 {lg_ev95:.1f}')

    # ── coefficient fits, per season + stability ──
    print('\n── prior fit: raw_full ~ bs_full + squp_full (per season) ──')
    coefs = {}
    for y, t in per_season.items():
        X = np.column_stack([np.ones(len(t)), t['bs_full'], t['squp_full']])
        b, *_ = np.linalg.lstsq(X, t['raw_full'].values, rcond=None)
        coefs[y] = b
        sd = t[['bs_full', 'squp_full']].std()
        r2 = 1 - np.var(t['raw_full'] - X @ b) / np.var(t['raw_full'])
        print(f'  {y}: a={b[0]:+.1f} b_bs={b[1]:+.3f} b_squp={b[2]:+.2f} '
              f'(std betas {b[1]*sd.iloc[0]:+.2f}/{b[2]*sd.iloc[1]:+.2f}) '
              f'R2={r2:.3f}')

    # Standardized-form prior (cross-currency robust): the 2024-25 bat
    # metrics come from the Savant pull, 2026's from the sheets, and the
    # two instruments carry a level shift (the recal diagnostic blew up on
    # 2026 with raw-coefficient transfer — same family as the known
    # extension gap between Hawk-Eye installs). Train seasons contribute
    # STANDARDIZED betas; the evaluated season supplies its own pool mean
    # and SD, which production computes live exactly like BETA.
    std_coefs = {}
    for y, t in per_season.items():
        mb, sb = t['bs_full'].mean(), t['bs_full'].std()
        mq, sq = t['squp_full'].mean(), t['squp_full'].std()
        mr = t['raw_full'].mean()
        b = coefs[y]
        std_coefs[y] = (float(b[1] * sb), float(b[2] * sq))
        per_season[y].attrs['pool'] = (mb, sb, mq, sq, mr)

    def prior_of(t, train_years):
        bb = np.mean([std_coefs[yy][0] for yy in train_years])
        bq = np.mean([std_coefs[yy][1] for yy in train_years])
        y_eval = t.attrs.get('_year')
        mb, sb, mq, sq, mr = per_season[y_eval].attrs['pool']
        return (mr + bb * (t['bs_col'] - mb) / sb
                + bq * (t['squp_col'] - mq) / sq)

    # ── design sweep ──
    print('\n── sweep: held-out RMSE vs remainder (rows = config) ──')
    results = defaultdict(dict)
    for y, t in per_season.items():
        train = [yy for yy in per_season if yy != y]
        for n in N_LEVELS:
            need = [f'con_{n}', f'evc_{n}', f'tgt_{n}', f'bs_{n}',
                    f'squp_{n}', f'nsw_{n}']
            tt = t.dropna(subset=[c for c in need if c in t]).copy()
            if len(tt) < 40:
                continue
            tt.attrs['_year'] = y
            tt['bs_col'], tt['squp_col'] = tt[f'bs_{n}'], tt[f'squp_{n}']
            prior = prior_of(tt, train)
            e0 = shipped_raw(tt[f'con_{n}'], tt[f'evc_{n}'], n)
            yv = tt[f'tgt_{n}'].values
            rmse0 = float(np.sqrt(np.mean((yv - e0) ** 2)))
            results[(y, n)]['base'] = rmse0
            for s0 in S0_GRID:
                pe = ((tt[f'nsw_{n}'] * prior + s0 * 100.0)
                      / (tt[f'nsw_{n}'] + s0))
                for n0bt in N0BT_GRID:
                    est = (n * e0 + n0bt * pe) / (n + n0bt)
                    results[(y, n)][f'D1_s{s0}_k{n0bt}'] = float(
                        np.sqrt(np.mean((yv - est) ** 2)))
                for n0ev in N0EV_GRID:
                    est = shipped_raw(tt[f'con_{n}'], tt[f'evc_{n}'], n,
                                      tgt_con=pe, tgt_ev=pe, n0_ev=n0ev)
                    results[(y, n)][f'D2_s{s0}_e{n0ev}'] = float(
                        np.sqrt(np.mean((yv - est) ** 2)))

    # D0_recal diagnostic: the OLS ceiling in production currency —
    # fit tgt ~ e0 + prior_eff(s0=50) on the OTHER seasons' same-n rows,
    # evaluate held-out. Separates "prior weak in real time" from "blend
    # form too constrained". Diagnostic only, never a ship candidate
    # (production does not recalibrate).
    cell_rows = {}
    for y, t in per_season.items():
        for n in N_LEVELS:
            need = [f'con_{n}', f'evc_{n}', f'tgt_{n}', f'bs_{n}',
                    f'squp_{n}', f'nsw_{n}']
            tt = t.dropna(subset=[c for c in need if c in t]).copy()
            if len(tt) < 40:
                continue
            tt.attrs['_year'] = y
            tt['bs_col'], tt['squp_col'] = tt[f'bs_{n}'], tt[f'squp_{n}']
            train = [yy for yy in per_season if yy != y]
            prior = prior_of(tt, train)
            pe = ((tt[f'nsw_{n}'] * prior + 50 * 100.0)
                  / (tt[f'nsw_{n}'] + 50))
            cell_rows[(y, n)] = pd.DataFrame({
                'e0': shipped_raw(tt[f'con_{n}'], tt[f'evc_{n}'], n),
                'pe': np.asarray(pe, float),
                'tgt': tt[f'tgt_{n}'].values})
    for (y, n), te in cell_rows.items():
        tr = pd.concat([cell_rows[(yy, nn)] for (yy, nn) in cell_rows
                        if yy != y and nn == n], ignore_index=True)
        if len(tr) < 60:
            continue
        for feats, tag in ((['e0'], 'recal_e0'), (['e0', 'pe'], 'recal_bat')):
            X = np.column_stack([np.ones(len(tr))]
                                + [np.asarray(tr[f], float) for f in feats])
            b, *_ = np.linalg.lstsq(X, tr['tgt'].values, rcond=None)
            Xt = np.column_stack([np.ones(len(te))]
                                 + [np.asarray(te[f], float) for f in feats])
            results[(y, n)][tag] = float(np.sqrt(np.mean(
                (te['tgt'].values - Xt @ b) ** 2)))
    print('  recal ceiling (diagnostic): '
          + '  '.join(f'{y}/{n}: e0 {results[(y, n)].get("recal_e0", float("nan")):.2f}'
                      f' -> +bat {results[(y, n)].get("recal_bat", float("nan")):.2f}'
                      for (y, n) in sorted(cell_rows)))

    # summarize: mean rank across cells; per-config wins vs base
    cells = sorted(results)
    configs = sorted({k for c in cells for k in results[c] if k != 'base'})
    summary = []
    for cfg in configs:
        deltas = [results[c][cfg] - results[c]['base'] for c in cells
                  if cfg in results[c]]
        wins = sum(1 for c in cells if cfg in results[c]
                   and results[c][cfg] < results[c]['base'])
        summary.append((float(np.mean(deltas)), wins, cfg))
    summary.sort()
    print(f'  {len(cells)} cells (season x n). Top 12 configs by mean '
          f'RMSE delta vs shipped baseline (negative = better):')
    for d, w, cfg in summary[:12]:
        print(f'   {cfg:<16} mean d={d:+.3f}  wins {w}/{len(cells)}')
    print('  baseline RMSE by cell: '
          + '  '.join(f'{y}/{n}={results[(y, n)]["base"]:.2f}'
                      for y, n in cells))

    # ── full-season invariance for the top config ──
    best_d1 = next(c for _, _, c in summary if c.startswith('D1'))
    best_d2 = next(c for _, _, c in summary if c.startswith('D2'))
    finalists = [best_d1, best_d2]
    best_cfg = best_d1  # n0_con re-check anchor
    # N0_CON re-check around the changed structure (shrinkage-audit
    # discipline): the shipped 130 was measured against a flat 100 target;
    # an informative target can move it. Swept at the winning (s0, n0_ev).
    if best_cfg.startswith('D2'):
        _, s_part, e_part = best_cfg.split('_')
        s0_w, e_w = int(s_part[1:]), int(e_part[1:])
        print(f'\n── N0_CON re-check at {best_cfg} ──')
        global N0_CON
        n0con_saved = N0_CON
        for n0c in (80, 130, 200, 300):
            N0_CON = n0c
            ds = []
            for y, t in per_season.items():
                train = [yy for yy in per_season if yy != y]
                for n in N_LEVELS:
                    need = [f'con_{n}', f'evc_{n}', f'tgt_{n}', f'bs_{n}',
                            f'squp_{n}', f'nsw_{n}']
                    tt = t.dropna(subset=[c for c in need if c in t]).copy()
                    if len(tt) < 40:
                        continue
                    tt.attrs['_year'] = y
                    tt['bs_col'] = tt[f'bs_{n}']
                    tt['squp_col'] = tt[f'squp_{n}']
                    prior = prior_of(tt, train)
                    pe = ((tt[f'nsw_{n}'] * prior + s0_w * 100.0)
                          / (tt[f'nsw_{n}'] + s0_w))
                    est = shipped_raw(tt[f'con_{n}'], tt[f'evc_{n}'], n,
                                      tgt_con=pe, tgt_ev=pe, n0_ev=e_w)
                    yv = tt[f'tgt_{n}'].values
                    base = results[(y, n)]['base']
                    ds.append(float(np.sqrt(np.mean((yv - est) ** 2)))
                              - base)
            print(f'  n0_con {n0c:>3}: mean d={np.mean(ds):+.3f}')
        N0_CON = n0con_saved
    inv = {}
    for cfg in finalists:
        print(f'\n── full-season invariance: {cfg} ──')
        inv[cfg] = {}
        for y, t in per_season.items():
            train = [yy for yy in per_season if yy != y]
            tt = t.copy()
            tt.attrs['_year'] = y
            tt['bs_col'], tt['squp_col'] = tt['bs_full'], tt['squp_full']
            prior = prior_of(tt, train)
            nb = tt['nbip_full'].values
            shipped_full = shipped_raw(tt['con_full'], tt['evc_full'], nb)
            parts = cfg.split('_')
            s0 = int(parts[1][1:])
            pe = (tt['nsw_full'] * prior + s0 * 100.0) / (tt['nsw_full'] + s0)
            if parts[0] == 'D1':
                k = int(parts[2][1:])
                new_v = (nb * shipped_full + k * pe) / (nb + k)
            else:
                n0ev = int(parts[2][1:])
                new_v = shipped_raw(tt['con_full'], tt['evc_full'], nb,
                                    tgt_con=pe, tgt_ev=pe, n0_ev=n0ev)
            new_v = np.asarray(new_v, float)
            sf = np.asarray(shipped_full, float)
            r = float(np.corrcoef(sf, new_v)[0, 1])
            inv[cfg][y] = {'corr': r,
                           'mean_abs': float(np.mean(np.abs(new_v - sf))),
                           'max_abs': float(np.max(np.abs(new_v - sf)))}
            print(f'  {y}: corr {r:.4f}, mean |d| '
                  f'{inv[cfg][y]["mean_abs"]:.2f}, max '
                  f'{inv[cfg][y]["max_abs"]:.2f} raw pts')
    frozen = {'std_beta_bs': float(np.mean([std_coefs[y][0] for y in std_coefs])),
              'std_beta_squp': float(np.mean([std_coefs[y][1] for y in std_coefs])),
              'per_season_std': {str(y): std_coefs[y] for y in std_coefs}}
    print(f"\n  frozen std betas: bs {frozen['std_beta_bs']:.2f}, "
          f"squp {frozen['std_beta_squp']:.2f} (pool-anchored form)")

    out = {'coefs': {str(y): list(map(float, coefs[y])) for y in coefs},
           'cells': {f'{y}_{n}': results[(y, n)] for y, n in cells},
           'top': [(d, w, c) for d, w, c in summary[:20]],
           'invariance': inv, 'frozen': frozen}
    with open(os.path.join(ROOT, 'data', '_bbplus_bt_prior_build.json'),
              'w') as f:
        json.dump(out, f, indent=1, default=float)
    print('  saved data/_bbplus_bt_prior_build.json')


if __name__ == '__main__':
    main()
