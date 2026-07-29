"""multi_season_retrain.py — does 2021-24 volume improve Stuff+? (+ arm-angle A/B)

Season-blocked protocol (same as production): prior seasons ALWAYS join every
fold's training set; 2026 is pitcher-grouped 8-fold OOF. Each season's BIP target
is built with that season's wOBA Guts. 2025 tags harmonized to 2026 (production
behavior); 2021-24 are public tags (agnostic model => labels never enter the
model, only the fastball anchor, so no harmonization needed).

Variants:
  BASELINE_25         prior = 2025 only            (current production)
  FULL_21_25          prior = 2021,22,23,24,25     (the volume test)
  FULL_noarm          FULL_21_25 minus arm_angle   (does arm_angle earn its place?)

Metrics on 2026: split-half reliability, pred_xRV (early->late), descriptive.

Usage: python3 scripts/multi_season_retrain.py
"""
import os, sys, time, pickle, warnings
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import stuff_plus_v11.train_stuff_v11 as T
import scripts.build_historical_training_set as H

PKL26 = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
GUTS = dict(H.GUTS); GUTS[2025] = (T.PRIOR_LG_WOBA, T.PRIOR_WOBA_SCALE)


def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 5:
        return None
    xs, ys = xs[m], ys[m]
    return float(np.corrcoef(xs, ys)[0, 1]) if xs.std() and ys.std() else None


def build_with_guts(pitches, year):
    lg, sc = GUTS[year]
    _lg, _sc = T.LG_WOBA, T.WOBA_SCALE
    T.LG_WOBA, T.WOBA_SCALE = lg, sc
    df = T.build_df(pitches)
    T.LG_WOBA, T.WOBA_SCALE = _lg, _sc
    return df[df.target_xrv.notna()].reset_index(drop=True)


def evaluate(prior_df, df26, feats):
    def design(d):
        return pd.concat([d[feats].reset_index(drop=True),
                          d[['platoon_same']].reset_index(drop=True)], axis=1)
    X26 = design(df26); y26 = df26.target_xrv.values; g = df26.pitcher.values
    Xp = design(prior_df).reindex(columns=X26.columns, fill_value=0); yp = prior_df.target_xrv.values
    p = T._params_for(X26)
    oof = np.full(len(df26), np.nan)
    for tr, te in GroupKFold(n_splits=8).split(X26, y26, g):
        Xtr = pd.concat([X26.iloc[tr], Xp], ignore_index=True)
        ytr = np.concatenate([y26[tr], yp])
        m = xgb.XGBRegressor(**p); m.fit(Xtr, ytr); oof[te] = m.predict(X26.iloc[te])
    d = df26.copy(); d['stuff'] = -oof
    a0, a1, est, late = [], [], {}, {}
    for key, grp in d.groupby(['pitcher', 'throws', 'pitch_type']):
        e, l = grp[grp.period == 'early'], grp[grp.period == 'late']
        if len(e) >= 50 and len(l) >= 50:
            late[key] = l.target_xrv.mean(); est[key] = e.stuff.mean()
        h0, h1 = grp[grp.half == 0], grp[grp.half == 1]
        if len(h0) >= 40 and len(h1) >= 40:
            a0.append(h0.stuff.mean()); a1.append(h1.stuff.mean())
    ks = list(est)
    dx = [gp.stuff.mean() for _, gp in d.groupby(['pitcher', 'throws', 'pitch_type']) if len(gp) >= 100]
    dy = [gp.target_xrv.mean() for _, gp in d.groupby(['pitcher', 'throws', 'pitch_type']) if len(gp) >= 100]
    return pearson(a0, a1), -pearson([est[k] for k in ks], [late[k] for k in ks]), -pearson(dx, dy), len(ks)


def main():
    t0 = time.time()
    p26 = [p for p in pickle.load(open(PKL26, 'rb')) if p.get('_source') == 'MLB']
    priors = {}
    for yr in (2021, 2022, 2023, 2024, 2025):
        pk = pickle.load(open(os.path.join(ROOT, 'data', f'_pitches{yr}_training.pkl'), 'rb'))
        if yr == 2025:
            T._harmonize_tags(pk, p26)
        priors[yr] = pk
        print(f"loaded {yr}: {len(pk)}", flush=True)

    df26 = build_with_guts(p26, 2026) if 2026 in GUTS else None
    # 2026 uses live constants (T.LG_WOBA already = 2026 Guts)
    df26 = T.build_df(p26); df26 = df26[df26.target_xrv.notna()].reset_index(drop=True)
    order = {d: i for i, d in enumerate(sorted(df26['date'].dropna().unique()))}
    df26['half'] = df26['date'].map(order).fillna(0).astype(int) % 2
    df26['period'] = np.where(df26['date'] < '2026-05-01', 'early', 'late')
    print(f"2026 OOF rows: {len(df26)}", flush=True)

    dfs = {yr: build_with_guts(priors[yr], yr) for yr in priors}
    for yr in dfs:
        print(f"  built {yr}: {len(dfs[yr])} rows", flush=True)

    prior_25 = dfs[2025]
    prior_full = pd.concat([dfs[yr] for yr in (2021, 2022, 2023, 2024, 2025)], ignore_index=True)
    print(f"\nprior sizes: 2025-only={len(prior_25)}  full-6season={len(prior_full)}\n", flush=True)

    print(f"{'variant':16s} {'reliab':>7s} {'pred_xRV':>8s} {'desc':>7s}   n_pred", flush=True)
    runs = [
        ('BASELINE_25', prior_25, list(T.BASE_FEATS)),
        ('FULL_21_25', prior_full, list(T.BASE_FEATS)),
        ('FULL_noarm', prior_full, [f for f in T.BASE_FEATS if f != 'arm_angle']),
    ]
    res = {}
    for name, prior, feats in runs:
        # noarm: drop arm_angle from df26 design too (evaluate uses feats for both)
        rel, pred, desc, npred = evaluate(prior, df26, feats)
        res[name] = (rel, pred, desc)
        print(f"{name:16s} {rel:7.3f} {pred:8.3f} {desc:7.3f}   {npred}", flush=True)

    b = res['BASELINE_25']
    print(f"\nvs BASELINE_25 (current production):")
    for name in ('FULL_21_25', 'FULL_noarm'):
        r = res[name]
        print(f"  {name:16s} d_reliab {r[0]-b[0]:+.3f}  d_pred {r[1]-b[1]:+.3f}  d_desc {r[2]-b[2]:+.3f}")
    fa, fn = res['FULL_21_25'], res['FULL_noarm']
    print(f"\narm_angle earns-its-place (FULL_21_25 minus FULL_noarm):"
          f"  d_reliab {fa[0]-fn[0]:+.3f}  d_pred {fa[1]-fn[1]:+.3f}  d_desc {fa[2]-fn[2]:+.3f}")
    print(f"\n[{time.time()-t0:.0f}s]")


if __name__ == '__main__':
    main()
