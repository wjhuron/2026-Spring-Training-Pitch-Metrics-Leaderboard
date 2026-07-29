"""pitching_plus_experiment.py — Pitching+ weight-sweep experiment (2026-07-07).

Question: Wally wants a Pitching+ = f(Stuff+, Loc+) composite. What weighting
of the two components best predicts FUTURE run prevention, and how do the
components behave (correlation, reliability, role splits)?

Method:
  - per-pitch STUFF value: production OOF stuff_raw (pitcher-grouped 8-fold,
    2025 prior in every fold, current 12-feature BASE_FEATS) — runs/pitch,
    higher = better.
  - per-pitch LOCATION value: -score_pitch() from the production Loc+
    surfaces (pipeline_locplus, built on 2026 MLB baseline) — runs/pitch,
    higher = better. NOTE: surfaces are full-season league aggregates, so the
    early->late eval carries mild leakage through the league surface only
    (same as production convention).
  - unit level: (pitcher, throws) overall AND (pitcher, throws, pitch_type).
    Components are z-scored on the qualified early-period pool, composite
    z_comp(w) = w*z_stuff + (1-w)*z_loc, w in 0..1.
  - metrics: pred (early z_comp -> late mean target_xrv), pred_rv (-> late
    mean actual -RunExp), split-half reliability (odd/even dates), and
    descriptive (full-period z vs full-period target). Role split: starters
    (>=15 pitches per appearance-date median... simplified: >=40 pitches per
    game-date avg) vs relievers.

Usage: STUFF_RETUNE_CACHE=<dir> python3 scripts/pitching_plus_experiment.py
"""
import os, sys, json, pickle, time, warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import stuff_plus_v11.train_stuff_v11 as T
import pipeline_locplus as L

PKL26 = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
CACHE_DIR = os.environ.get('STUFF_RETUNE_CACHE', '/tmp')
CACHE25 = os.path.join(CACHE_DIR, 'stuff_retune_df25.pkl')
EARLY_CUT = '2026-05-01'
SPLIT_MIN_OVERALL = 200   # pitches per period, pitcher level
SPLIT_MIN_PT = 50         # per period, per pitch type (stuff-harness parity)


def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 5:
        return None
    xs, ys = xs[m], ys[m]
    if xs.std() == 0 or ys.std() == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def build_mask(pitches):
    """Replicate T.build_df's row filter (assert alignment afterwards)."""
    mask = np.zeros(len(pitches), bool)
    for i, p in enumerate(pitches):
        pt, thr, bats = p.get('Pitch Type'), p.get('Throws'), p.get('Bats')
        if pt not in T.SUPPORTED or thr not in ('L', 'R') or bats not in ('L', 'R'):
            continue
        v = T.sf(p.get('Velocity')); iv = T.sf(p.get('xIndVrtBrk'))
        hb = T.sf(p.get('xHorzBrk')); vaa = T.sf(p.get('VAA'))
        ext = T.sf(p.get('Extension')); rz = T.sf(p.get('RelPosZ'))
        rx = T.sf(p.get('RelPosX'))
        if None in (v, iv, hb, vaa, ext, rz, rx):
            continue
        mask[i] = True
    return mask


def main():
    t0 = time.time()
    print('loading pickle ...', flush=True)
    D = pickle.load(open(PKL26, 'rb'))
    p26 = [p for p in D if p.get('_source', 'MLB') == 'MLB']
    del D

    # ── Loc+ surfaces (production path) + per-pitch location value ──
    guts = json.load(open(os.path.join(ROOT, 'data', 'metadata_rs.json'))).get(
        'gutsConstants') or {}
    lg_woba, woba_scale = guts.get('lgWOBA'), guts.get('wOBAScale')
    baseline = [p for p in p26 if L.is_eligible_baseline(p)]
    print(f'  building Loc+ surfaces on {len(baseline)} baseline pitches ...',
          flush=True)
    S = L.build_surfaces(baseline, lg_woba, woba_scale)
    locval = np.full(len(p26), np.nan)
    n_loc = 0
    for i, p in enumerate(p26):
        if not L._is_scorable(p):
            continue
        v = L.score_pitch(p, S)
        if v is not None:
            locval[i] = -v   # pitcher perspective: higher = better location
            n_loc += 1
    print(f'  loc value scored on {n_loc}/{len(p26)} pitches '
          f'({time.time()-t0:.0f}s)', flush=True)

    # actual outcome value per pitch (pitcher perspective)
    rv_actual = np.array([(-r if (r := T.sf(p.get('RunExp'))) is not None
                           else np.nan) for p in p26])

    # ── stuff df + alignment ──
    df26 = T.build_df(p26)
    mask = build_mask(p26)
    assert mask.sum() == len(df26), (mask.sum(), len(df26))
    df26['loc_raw'] = locval[mask]
    df26['rv_actual'] = rv_actual[mask]
    # sanity: velocities align
    vcheck = np.array([T.sf(p.get('Velocity')) for p, m in zip(p26, mask) if m])
    assert np.allclose(vcheck[:1000], df26['velocity'].values[:1000])
    df26 = df26[df26['target_xrv'].notna()].reset_index(drop=True)
    date_order = {d: i for i, d in enumerate(sorted(df26['date'].dropna().unique()))}
    df26['half'] = df26['date'].map(date_order).fillna(0).astype(int) % 2
    df26['period'] = np.where(df26['date'] < EARLY_CUT, 'early', 'late')
    print(f'  {len(df26)} scored rows; loc coverage '
          f'{df26.loc_raw.notna().mean()*100:.1f}%', flush=True)

    # ── production OOF stuff_raw (12-feature set) ──
    df25 = pd.read_pickle(CACHE25)
    X26 = T.design(df26); X25 = T.design(df25).reindex(columns=X26.columns,
                                                       fill_value=0)
    y26, y25 = df26['target_xrv'].values, df25['target_xrv'].values
    params = T._params_for(X26)
    oof = np.full(len(df26), np.nan)
    for tr, te in GroupKFold(n_splits=8).split(X26, y26, df26['pitcher'].values):
        m = xgb.XGBRegressor(**params)
        m.fit(pd.concat([X26.iloc[tr], X25], ignore_index=True),
              np.concatenate([y26[tr], y25]))
        oof[te] = m.predict(X26.iloc[te])
    df26['stuff_raw'] = -oof
    print(f'  OOF stuff done ({time.time()-t0:.0f}s)', flush=True)

    # starter/reliever tag: mean pitches per game date
    ppg = df26.groupby(['pitcher', 'throws', 'date']).size().groupby(
        ['pitcher', 'throws']).mean()
    role = (ppg >= 40).rename('is_sp')   # >=40 pitches/outing ~ starter

    out_rows = []

    def sweep(unit_keys, split_min, label, role_filter=None):
        d = df26[df26['loc_raw'].notna()].copy()
        if role_filter is not None:
            d = d.merge(role.reset_index(), on=['pitcher', 'throws'])
            d = d[d['is_sp'] == role_filter]
        g = d.groupby(unit_keys + ['period'])
        agg = g.agg(stuff=('stuff_raw', 'mean'), loc=('loc_raw', 'mean'),
                    xrv=('target_xrv', 'mean'), rva=('rv_actual', 'mean'),
                    n=('stuff_raw', 'size')).reset_index()
        wide = agg.pivot_table(index=unit_keys, columns='period',
                               values=['stuff', 'loc', 'xrv', 'rva', 'n'],
                               aggfunc='first')
        ok = ((wide[('n', 'early')].fillna(0) >= split_min) &
              (wide[('n', 'late')].fillna(0) >= split_min))
        wide = wide[ok]
        if len(wide) < 20:
            print(f'  [{label}] only {len(wide)} units, skipping', flush=True)
            return
        zs = {}
        for comp in ('stuff', 'loc'):
            e = wide[(comp, 'early')]
            zs[comp] = (e - e.mean()) / e.std()
        # split-half reliability inputs (odd/even dates, full season)
        gh = d.groupby(unit_keys + ['half'])
        ha = gh.agg(stuff=('stuff_raw', 'mean'), loc=('loc_raw', 'mean'),
                    n=('stuff_raw', 'size')).reset_index()
        hw = ha.pivot_table(index=unit_keys, columns='half',
                            values=['stuff', 'loc', 'n'], aggfunc='first')
        hok = ((hw[('n', 0)].fillna(0) >= split_min // 2) &
               (hw[('n', 1)].fillna(0) >= split_min // 2))
        hw = hw[hok]
        hz = {}
        for comp in ('stuff', 'loc'):
            for h in (0, 1):
                col = hw[(comp, h)]
                hz[(comp, h)] = (col - col.mean()) / col.std()
        # component facts
        r_sl = pearson(zs['stuff'], zs['loc'])
        print(f'\n  [{label}] units={len(wide)}  '
              f'corr(stuff, loc) early = {r_sl:+.3f}', flush=True)
        print(f'  {"w_stuff":>8s} {"pred_xrv":>9s} {"pred_rv":>8s} {"reliab":>7s}')
        neg_late_xrv = -wide[('xrv', 'late')]   # lower xrv allowed = better
        neg_late_rva = wide[('rva', 'late')]    # rv_actual already pitcher-persp
        for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
                  0.8, 0.9, 1.0]:
            comp_e = w * zs['stuff'] + (1 - w) * zs['loc']
            pr_x = pearson(comp_e, -neg_late_xrv)
            pr_x = None if pr_x is None else -pr_x
            pr_r = pearson(comp_e, neg_late_rva)
            c0 = w * hz[('stuff', 0)] + (1 - w) * hz[('loc', 0)]
            c1 = w * hz[('stuff', 1)] + (1 - w) * hz[('loc', 1)]
            rel = pearson(c0, c1)
            out_rows.append(dict(scope=label, w=w, pred_xrv=pr_x, pred_rv=pr_r,
                                 reliab=rel, n_units=len(wide)))
            print(f'  {w:8.2f} {pr_x:9.3f} {pr_r:8.3f} {rel:7.3f}', flush=True)

    sweep(['pitcher', 'throws'], SPLIT_MIN_OVERALL, 'pitcher overall')
    sweep(['pitcher', 'throws'], SPLIT_MIN_OVERALL, 'starters', role_filter=True)
    sweep(['pitcher', 'throws'], SPLIT_MIN_OVERALL, 'relievers', role_filter=False)
    sweep(['pitcher', 'throws', 'pitch_type'], SPLIT_MIN_PT, 'per pitch type')

    pd.DataFrame(out_rows).to_csv(
        os.path.join(ROOT, 'scripts', 'pitching_plus_sweep.csv'), index=False)
    print('\nsaved -> scripts/pitching_plus_sweep.csv', flush=True)


if __name__ == '__main__':
    main()
