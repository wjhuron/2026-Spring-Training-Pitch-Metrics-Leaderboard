"""stuff_hp_retune.py — Stuff+ v11 hyperparameter re-tune at the ~1.08M-row scale.

TUNED (depth 4 / 800 trees / lr .025 / mcw 10) was tuned when the training set
was ~375k rows (2026-only). With the season-blocked 2025 pickle the training
set is ~1.08M rows, which typically supports deeper/bigger models. This harness
re-screens a small grid and also tests recency down-weighting of the 2025 rows.

Protocol (exact production season-blocked scheme, same as
scripts/agnostic_stuff_experiment.py):
  - 2025 pickle tag-harmonized to 2026 labels (T._harmonize_tags)
  - 2025 joins EVERY fold's training set (targets on 2025 Guts constants)
  - 2026 scored pitcher-grouped K-fold OOF (4-fold screening, 8-fold final)
  - monotone velocity constraint on every config

Metrics on 2026 OOF stuff, unit = (pitcher, throws, pitch_type):
  - reliab : split-half reliability, odd/even calendar dates, >=40 per half
  - pred   : early (<2026-05-01) OOF stuff -> late-period mean target, >=50/period
  - desc   : same-period OOF stuff vs mean target, >=100 pitches

Decision rule (8-fold run): adopt a new config only if
  pred >= current + 0.005  AND  reliab >= current - 0.010; ties -> current.

Usage:
  python3 scripts/stuff_hp_retune.py --folds 4 --configs current,d5_n800_lr025,d6_n800_lr025,d4_n1400_lr015,d5_n1400_lr015,d6_n1400_lr015
  python3 scripts/stuff_hp_retune.py --folds 8 --configs current,<top2>
  python3 scripts/stuff_hp_retune.py --folds 4 --configs current --w25 0.5,0.75,1.25   # recency task
"""
import os, sys, time, pickle, argparse, warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import stuff_plus_v11.train_stuff_v11 as T

PKL26 = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
PKL25 = os.path.join(ROOT, 'data', '_pitches2025_training.pkl')
CACHE_DIR = os.environ.get('STUFF_RETUNE_CACHE', '/tmp')
CACHE25 = os.path.join(CACHE_DIR, 'stuff_retune_df25.pkl')
CACHE26 = os.path.join(CACHE_DIR, 'stuff_retune_df26.pkl')


def config_registry():
    """Named hyperparameter configs. Every config = TUNED with overrides.
    'current' is production TUNED exactly (d4 / 800 / .025 / mcw10)."""
    reg = {'current': {}}
    for d in (4, 5, 6):
        for n, lr, tag in ((800, 0.025, 'n800_lr025'), (1400, 0.015, 'n1400_lr015')):
            for mcw in (10, 40):
                name = f'd{d}_{tag}' + ('' if mcw == 10 else f'_mcw{mcw}')
                reg[name] = dict(max_depth=d, n_estimators=n, learning_rate=lr,
                                 min_child_weight=mcw)
    return reg


def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 5:
        return None
    xs, ys = xs[m], ys[m]
    if xs.std() == 0 or ys.std() == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def load_dfs(use_cache=True):
    if use_cache and os.path.exists(CACHE25) and os.path.exists(CACHE26):
        print('loading cached feature dfs ...', flush=True)
        return pd.read_pickle(CACHE25), pd.read_pickle(CACHE26)
    print('loading pickles (slow path) ...', flush=True)
    D26 = pickle.load(open(PKL26, 'rb'))
    p26 = [p for p in D26 if p.get('_source', 'MLB') == 'MLB']
    p25 = pickle.load(open(PKL25, 'rb'))
    T._harmonize_tags(p25, p26)                      # production convention
    df26 = T.build_df(p26)
    del D26, p26
    _lg, _sc = T.LG_WOBA, T.WOBA_SCALE
    T.LG_WOBA, T.WOBA_SCALE = T.PRIOR_LG_WOBA, T.PRIOR_WOBA_SCALE
    df25 = T.build_df(p25)
    T.LG_WOBA, T.WOBA_SCALE = _lg, _sc
    del p25
    df26 = df26[df26['target_xrv'].notna()].reset_index(drop=True)
    df25 = df25[df25['target_xrv'].notna()].reset_index(drop=True)
    date_order = {dt: i for i, dt in enumerate(sorted(df26['date'].dropna().unique()))}
    df26['half'] = df26['date'].map(date_order).fillna(0).astype(int) % 2
    df26['period'] = np.where(df26['date'] < '2026-05-01', 'early', 'late')
    df25.to_pickle(CACHE25); df26.to_pickle(CACHE26)
    print(f'  cached -> {CACHE25}, {CACHE26}', flush=True)
    return df25, df26


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=4)
    ap.add_argument('--configs', default='current')
    ap.add_argument('--w25', default='1.0',
                    help='comma list of 2025 sample weights; each (config, w25) pair runs')
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()

    reg = config_registry()
    names = [c.strip() for c in args.configs.split(',') if c.strip()]
    weights = [float(w) for w in args.w25.split(',')]
    for n in names:
        if n not in reg:
            sys.exit(f'unknown config {n!r}; known: {sorted(reg)}')

    df25, df26 = load_dfs(use_cache=not args.no_cache)
    print(f'training rows: 2025={len(df25)}, 2026={len(df26)}, '
          f'total={len(df25) + len(df26)}', flush=True)

    y26, y25 = df26['target_xrv'].values, df25['target_xrv'].values
    g26 = df26['pitcher'].values
    X26 = T.design(df26)
    X25 = T.design(df25).reindex(columns=X26.columns, fill_value=0)
    folds = list(GroupKFold(n_splits=args.folds).split(X26, y26, g26))

    # late-period targets (config-independent)
    late = {}
    for key, grp in df26.groupby(['pitcher', 'throws', 'pitch_type']):
        e, l = grp[grp.period == 'early'], grp[grp.period == 'late']
        if len(e) >= 50 and len(l) >= 50:
            late[key] = l['target_xrv'].dropna().mean()

    runs = [(nm, w) for nm in names for w in weights]
    print(f'\n{args.folds}-fold OOF, {len(runs)} run(s)\n', flush=True)
    header = (f"{'config':24s} {'w25':>5s} {'reliab':>7s} {'pred_xRV':>8s} "
              f"{'desc':>7s} {'n_rel':>6s} {'n_pred':>6s} {'mins':>6s}")
    print(header, flush=True)
    results = []
    for nm, w25 in runs:
        t0 = time.time()
        params = T._params_for(X26)
        params.update(reg[nm])
        oof = np.full(len(df26), np.nan)
        for tr, te in folds:
            Xtr = pd.concat([X26.iloc[tr], X25], ignore_index=True)
            ytr = np.concatenate([y26[tr], y25])
            sw = None
            if w25 != 1.0:
                sw = np.concatenate([np.ones(len(tr)), np.full(len(y25), w25)])
            m = xgb.XGBRegressor(**params)
            m.fit(Xtr, ytr, sample_weight=sw)
            oof[te] = m.predict(X26.iloc[te])
        d = df26[['pitcher', 'throws', 'pitch_type', 'half', 'period', 'target_xrv']].copy()
        d['stuff'] = -oof

        a0, a1, est, desc_x, desc_y = [], [], {}, [], []
        for key, grp in d.groupby(['pitcher', 'throws', 'pitch_type']):
            h0, h1 = grp[grp.half == 0], grp[grp.half == 1]
            if len(h0) >= 40 and len(h1) >= 40:
                a0.append(h0['stuff'].mean()); a1.append(h1['stuff'].mean())
            if key in late:
                est[key] = grp[grp.period == 'early']['stuff'].mean()
            if len(grp) >= 100:
                desc_x.append(grp['stuff'].mean()); desc_y.append(grp['target_xrv'].mean())
        ks = list(est)
        rel = pearson(a0, a1)
        pr = -pearson([est[k] for k in ks], [late[k] for k in ks])
        dsc = -pearson(desc_x, desc_y)
        mins = (time.time() - t0) / 60
        results.append((nm, w25, rel, pr, dsc))
        print(f'{nm:24s} {w25:5.2f} {rel:7.3f} {pr:8.3f} {dsc:7.3f} '
              f'{len(a0):6d} {len(ks):6d} {mins:6.1f}', flush=True)

    print('\nsummary (sorted by pred_xRV):')
    print(header.rsplit(' ', 1)[0])
    for nm, w25, rel, pr, dsc in sorted(results, key=lambda r: -r[3]):
        print(f'{nm:24s} {w25:5.2f} {rel:7.3f} {pr:8.3f} {dsc:7.3f}')


if __name__ == '__main__':
    main()
