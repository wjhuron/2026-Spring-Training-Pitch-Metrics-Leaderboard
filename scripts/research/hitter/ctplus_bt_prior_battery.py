"""ctplus_bt_prior_battery.py — does bat tracking stabilize CT+ at small
samples, the way it does BB+? (The BB+ prior shipped e9a9080a; this asks
whether the mechanism generalizes to the contact component.)

CIRCULARITY GUARD (the design difference from the BB+ battery): CT+
measures contact frequency vs expectation, and squared-up PER SWING
contains the contact rate itself — it is a transformation of the outcome,
not a kinetics measurement. Registered predictors are therefore:
  bs    mean BatSpeed on competitive swings (>= 50 mph) — pure kinetics
  squpc squared-up per blast-eligible contact (the PRODUCTION
        squaredUpPct currency) — quality GIVEN contact; conditions on
        the outcome but does not contain its rate
Squared-up per swing is registered OUT, stated here so nobody re-adds it.

CURRENCY: everything runs in raw_ct (the actual/expected contact lift
ratio, pool-independent — compute_hitter_ct against the season's cell
tables). ctPlus itself is pool-normalized over the passed dict, so it
cannot be used with per-cut fake keys; raw_ct can.

PRE-REGISTERED PROTOCOL (2026-08-21, before any result was seen):
  Universe   MLB hitter-seasons 2024/2025/2026: >= 400 CT-eligible
             swings full-season, >= 100 tracked competitive swings.
             2026 = sheet pickle (native vocabulary, names bridged to
             MLBAM ids); 2024-25 = statcast caches through
             statcast_hitter_adapter + the battrack pitch pull for the
             kinetics windows.
  Simulation first-n CT-eligible swings, n in {40, 80, 160},
             chronological; bat window = swings on/before the nth
             swing's date (causal); remainder >= 200 swings.
  E0         production-style estimate on first-n: raw_ct shrunk toward
             the season pool mean at HITTER_PRIOR_N = 66.
  Target     remainder raw_ct, unshrunk.
  T1         partial r of each registered predictor with the target,
             controlling E0. Gate: CONSISTENT SIGN across seasons at
             n=40 (the BB+ battery's wording lesson: gate on sign
             consistency, not on an assumed direction).
  T2         leave-season-out OLS (tgt ~ E0 vs tgt ~ E0 + bs + squpc),
             held-out RMSE. Gate: improves in >= 2/3 of the n in
             {40, 80} cells.
  DECISION   both gates -> design + measure the production prior
             (separate follow-up, constants swept). Either gate fails ->
             CT+ stays as shipped and the result is recorded.

Usage: PYTHONHASHSEED=0 python3 scripts/research/hitter/ctplus_bt_prior_battery.py
Output: data/_ctplus_bt_prior_results.json + printed tables.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from statcast_hitter_adapter import season_dicts, GUTS  # noqa: E402
from pipeline.contact import (  # noqa: E402
    is_ct_eligible, build_bip_count_offsets, build_contact_cell_weights,
    zone_level_contact_means, shrink_contact_cells, compute_hitter_ct,
    HITTER_PRIOR_N)
from pipeline.sdplus import make_rv_xrv  # noqa: E402

D = lambda n: json.load(open(os.path.join(ROOT, 'data', n)))
SEASONS = (2024, 2025, 2026)
N_LEVELS = (40, 80, 160)
MIN_SWINGS_FULL, MIN_REMAINDER, MIN_TRACKED = 400, 200, 100


def name_bridge():
    bt = D('_bt_seasons.json')
    m = {}
    for y in ('2026', '2025', '2024'):
        for pid, row in bt.get(y, {}).items():
            if row.get('name'):
                m.setdefault(row['name'], pid)
    return m


def sheet_pitches_2026():
    raw = pd.read_pickle(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'))
    bridge = name_bridge()
    out = []
    for p in raw:
        if p.get('PTeam') == 'ROC' or p.get('BTeam') == 'ROC':
            continue
        hid = bridge.get(str(p.get('Batter')))
        if hid is None or not p.get('Game Date'):
            continue
        q = dict(p)
        q['Batter'] = str(hid)
        q['BTeam'] = 'X'
        out.append(q)
    return out


def kin_swings(year):
    """(hid, date, bs, ev, velo) rows for bat windows."""
    if year == 2026:
        raw = pd.read_pickle(os.path.join(ROOT, 'data',
                                          'all_pitches_rs_cache.pkl'))
        bridge = name_bridge()
        df = pd.DataFrame(raw)
        df = df[(df['PTeam'] != 'ROC') & (df['BTeam'] != 'ROC')]
        df['hid'] = df['Batter'].astype(str).map(bridge)
        df = df.dropna(subset=['hid'])
        bs = pd.to_numeric(df['BatSpeed'], errors='coerce')
        m = bs.notna() & (bs >= 50)
        return pd.DataFrame({
            'hid': df.loc[m, 'hid'], 'date': df.loc[m, 'Game Date'].astype(str),
            'bs': bs[m],
            'ev': pd.to_numeric(df.loc[m, 'ExitVelo'], errors='coerce'),
            'velo': pd.to_numeric(df.loc[m, 'Velocity'], errors='coerce'),
        }).sort_values('date', kind='stable')
    bt = pd.read_pickle(os.path.join(ROOT, 'data',
                                     f'_battrack_pitch_{year}.pkl'))
    bs = pd.to_numeric(bt['bat_speed'], errors='coerce')
    m = bs.notna() & (bs >= 50)
    return pd.DataFrame({
        'hid': bt.loc[m, 'batter'].astype(int).astype(str),
        'date': bt.loc[m, 'game_date'].astype(str),
        'bs': bs[m],
        'ev': pd.to_numeric(bt.loc[m, 'launch_speed'], errors='coerce'),
        'velo': pd.to_numeric(bt.loc[m, 'release_speed'], errors='coerce'),
    }).sort_values('date', kind='stable')


def bat_metrics(sw):
    """Production currency: bs mean; squared-up per blast-eligible."""
    if not len(sw):
        return None
    elig = sw.dropna(subset=['ev', 'velo'])
    squpc = None
    if len(elig):
        max_ev = 0.212 * elig['velo'] + 1.23 * elig['bs']
        squpc = float((elig['ev'] >= 0.80 * max_ev).mean())
    return {'bs': float(sw['bs'].mean()), 'nsw': len(sw), 'squpc': squpc}


def main():
    per_season = {}
    for y in SEASONS:
        if y == 2026:
            pitches = sheet_pitches_2026()
            md = D('metadata_rs.json')
            G = md.get('gutsConstants') or {}
            lgw, ws = G.get('lgWOBA', 0.313), G.get('wOBAScale', 1.232)
        else:
            pitches = season_dicts(y)
            lgw, ws = GUTS[y]
        kin = kin_swings(y)
        kin_by = dict(tuple(kin.groupby('hid')))
        # season cell tables, built ONCE
        swings = [p for p in pitches if is_ct_eligible(p)]
        offsets = build_bip_count_offsets(swings, lgw, ws)
        rv_fn = make_rv_xrv(lgw, ws, offsets)
        cells = shrink_contact_cells(build_contact_cell_weights(swings, rv_fn),
                                     zone_level_contact_means(swings, rv_fn))
        # per-hitter chronological eligible swings
        by_h = defaultdict(list)
        for p in swings:
            by_h[p['Batter']].append(p)
        for h in by_h:
            by_h[h].sort(key=lambda p: p.get('Game Date') or '')
        cuts = {}
        meta = {}
        for h, ps in by_h.items():
            if len(ps) < MIN_SWINGS_FULL:
                continue
            ksw = kin_by.get(h)
            if ksw is None or len(ksw) < MIN_TRACKED:
                continue
            cuts[(h + '|full', 'X')] = ps
            meta[h] = {'n_full': len(ps)}
            for n in N_LEVELS:
                if len(ps) - n < MIN_REMAINDER:
                    continue
                cuts[(f'{h}|f{n}', 'X')] = ps[:n]
                cuts[(f'{h}|r{n}', 'X')] = ps[n:]
                meta[h][f'cut_{n}'] = ps[n - 1].get('Game Date')
        raw_ct = compute_hitter_ct(cuts, cells)
        recs = []
        for h, mt in meta.items():
            full = raw_ct.get((h + '|full', 'X'))
            ksw = kin_by[h]
            bm_full = bat_metrics(ksw)
            if not full or bm_full is None or bm_full['squpc'] is None:
                continue
            rec = {'hid': h, 'raw_full': full['raw_ct'],
                   'bs_full': bm_full['bs'], 'squpc_full': bm_full['squpc']}
            for n in N_LEVELS:
                f = raw_ct.get((f'{h}|f{n}', 'X'))
                r = raw_ct.get((f'{h}|r{n}', 'X'))
                cutdate = mt.get(f'cut_{n}')
                if not f or not r or not cutdate:
                    continue
                win = ksw[ksw['date'] <= cutdate]
                bm = bat_metrics(win)
                rec[f'raw_{n}'] = f['raw_ct']
                rec[f'tgt_{n}'] = r['raw_ct']
                rec[f'bs_{n}'] = bm['bs'] if bm else np.nan
                rec[f'squpc_{n}'] = (bm['squpc'] if bm
                                     and bm['squpc'] is not None else np.nan)
            recs.append(rec)
        per_season[y] = pd.DataFrame(recs)
        lg_raw = float(per_season[y]['raw_full'].mean())
        per_season[y].attrs['lg_raw'] = lg_raw
        print(f'  {y}: {len(recs)} hitter-seasons, lg raw_ct {lg_raw:.4f}')

    def pearson(x, y_):
        x, y_ = np.asarray(x, float), np.asarray(y_, float)
        m = np.isfinite(x) & np.isfinite(y_)
        if m.sum() < 25:
            return None
        return float(np.corrcoef(x[m], y_[m])[0, 1])

    def partial(x, y_, c):
        x, y_, c = (np.asarray(v, float) for v in (x, y_, c))
        m = np.isfinite(x) & np.isfinite(y_) & np.isfinite(c)
        if m.sum() < 25:
            return None
        rx = x[m] - np.poly1d(np.polyfit(c[m], x[m], 1))(c[m])
        ry = y_[m] - np.poly1d(np.polyfit(c[m], y_[m], 1))(c[m])
        return pearson(rx, ry)

    print('\n── T1: partial r with remainder raw_ct | E0(first-n) ──')
    for y, t in per_season.items():
        lg = t.attrs['lg_raw']
        for n in N_LEVELS:
            if f'raw_{n}' not in t:
                continue
            e0 = (n * t[f'raw_{n}'] + HITTER_PRIOR_N * lg) \
                / (n + HITTER_PRIOR_N)
            r0 = pearson(e0, t[f'tgt_{n}'])
            pb = partial(t[f'bs_{n}'], t[f'tgt_{n}'], e0)
            pq = partial(t[f'squpc_{n}'], t[f'tgt_{n}'], e0)
            print(f'  {y} n={n:<4} E0 r={r0:+.3f}  bs={pb:+.3f}  '
                  f'squpc={pq:+.3f}')

    print('\n── T2: leave-season-out OLS, held-out RMSE ──')
    wins = 0
    cells_n = 0
    out_t2 = {}
    for y in per_season:
        others = [per_season[o] for o in per_season if o != y]
        tr_all = pd.concat(others, ignore_index=True)
        te = per_season[y]
        lg_tr = float(np.mean([o.attrs['lg_raw'] for o in others]))
        lg_te = te.attrs['lg_raw']
        for n in N_LEVELS:
            cols = [f'raw_{n}', f'bs_{n}', f'squpc_{n}', f'tgt_{n}']
            trn = tr_all.dropna(subset=[c for c in cols if c in tr_all])
            ten = te.dropna(subset=[c for c in cols if c in te])
            if len(trn) < 60 or len(ten) < 25:
                continue
            e0_tr = (n * trn[f'raw_{n}'] + HITTER_PRIOR_N * lg_tr) \
                / (n + HITTER_PRIOR_N)
            e0_te = (n * ten[f'raw_{n}'] + HITTER_PRIOR_N * lg_te) \
                / (n + HITTER_PRIOR_N)

            def fit_pred(feats_tr, feats_te):
                X = np.column_stack([np.ones(len(trn))]
                                    + [np.asarray(f, float) for f in feats_tr])
                b, *_ = np.linalg.lstsq(X, np.asarray(trn[f'tgt_{n}'], float),
                                        rcond=None)
                Xt = np.column_stack([np.ones(len(ten))]
                                     + [np.asarray(f, float)
                                        for f in feats_te])
                return Xt @ b
            yv = np.asarray(ten[f'tgt_{n}'], float)
            base = fit_pred([e0_tr], [e0_te])
            full = fit_pred([e0_tr, trn[f'bs_{n}'], trn[f'squpc_{n}']],
                            [e0_te, ten[f'bs_{n}'], ten[f'squpc_{n}']])
            squ = fit_pred([e0_tr, trn[f'squpc_{n}']],
                           [e0_te, ten[f'squpc_{n}']])
            r0 = float(np.sqrt(np.mean((yv - base) ** 2)))
            r1 = float(np.sqrt(np.mean((yv - full) ** 2)))
            r2 = float(np.sqrt(np.mean((yv - squ) ** 2)))
            out_t2[f'{y}_{n}'] = {'base': r0, 'bat': r1, 'squ_only': r2,
                                  'n': len(ten)}
            tag = 'improves' if r1 < r0 else 'worse'
            print(f'  held-out {y} n={n:<4} base {r0:.4f} -> +bat {r1:.4f} '
                  f'({tag})  squpc-only {r2:.4f}')
            if n in (40, 80):
                cells_n += 1
                wins += (r1 < r0)
    print(f'\n  T2 decision cells (n in 40/80): improves {wins}/{cells_n}')
    with open(os.path.join(ROOT, 'data',
                           '_ctplus_bt_prior_results.json'), 'w') as f:
        json.dump({'t2': out_t2, 'protocol': 'module docstring'}, f,
                  indent=1, default=float)
    print('  saved data/_ctplus_bt_prior_results.json')


if __name__ == '__main__':
    main()
