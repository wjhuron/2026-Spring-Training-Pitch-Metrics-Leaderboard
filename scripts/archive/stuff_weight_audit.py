"""stuff_weight_audit.py — Stuff+ v11 feature-weighting audit (2026-07-07).

Question: is each of the 14 model inputs (13 BASE_FEATS + platoon_same)
properly valued by the pooled XGBoost model? Three lenses:

  A. learned reliance  : gain / total_gain / cover importance + SHAP
                         (pred_contribs) share on the full production model
  B. permutation       : shuffle one feature at OOF scoring time (fold models
                         held fixed) -> degradation of reliab / pred / desc
  C. drop-one retrain  : remove the feature, retrain the exact production
                         season-blocked protocol -> does pred_xRV drop?

Protocol matches scripts/stuff_hp_retune.py exactly (production TUNED params,
monotone velocity, 2025 harmonized pickle in every fold's training set,
pitcher-grouped K-fold OOF on 2026). 4-fold screening by default.

Metrics on 2026 OOF stuff, unit = (pitcher, throws, pitch_type):
  - reliab : split-half reliability, odd/even calendar dates, >=40 per half
  - pred   : early (<2026-05-01) OOF stuff -> late-period mean target, >=50/period
  - desc   : same-period OOF stuff vs mean target, >=100 pitches

Usage:
  python3 scripts/stuff_weight_audit.py --phase a
  python3 scripts/stuff_weight_audit.py --phase b --folds 4
  python3 scripts/stuff_weight_audit.py --phase c --folds 4
  python3 scripts/stuff_weight_audit.py --phase c --folds 8 --feats <shortlist>
"""
import os, sys, json, time, argparse, warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import stuff_plus_v11.train_stuff_v11 as T
import stuff_hp_retune as R

OUT_DIR = os.environ.get('STUFF_AUDIT_OUT', os.path.join(ROOT, 'scripts'))
ALL_FEATS = list(T.BASE_FEATS) + ['platoon_same']


def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 5:
        return None
    xs, ys = xs[m], ys[m]
    if xs.std() == 0 or ys.std() == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def unit_metrics(df26, stuff, late):
    """(reliab, pred, desc) on OOF stuff scores, production conventions."""
    d = df26[['pitcher', 'throws', 'pitch_type', 'half', 'period', 'target_xrv']].copy()
    d['stuff'] = stuff
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
    return rel, pr, dsc


def late_targets(df26):
    late = {}
    for key, grp in df26.groupby(['pitcher', 'throws', 'pitch_type']):
        e, l = grp[grp.period == 'early'], grp[grp.period == 'late']
        if len(e) >= 50 and len(l) >= 50:
            late[key] = l['target_xrv'].dropna().mean()
    return late


def design(df, feats):
    """Agnostic production design restricted to `feats` (platoon_same is a
    plain member of the feature list here, not auto-appended)."""
    return df[feats].reset_index(drop=True).copy()


def fit_oof(df26, df25, feats, n_splits, keep_models=False):
    X26 = design(df26, feats)
    X25 = design(df25, feats)
    y26, y25 = df26['target_xrv'].values, df25['target_xrv'].values
    g26 = df26['pitcher'].values
    params = dict(T.TUNED)
    params['monotone_constraints'] = tuple(
        -1 if c == T.MONO_FEAT else 0 for c in X26.columns)
    oof = np.full(len(df26), np.nan)
    models, tests = [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(X26, y26, g26):
        Xtr = pd.concat([X26.iloc[tr], X25], ignore_index=True)
        ytr = np.concatenate([y26[tr], y25])
        m = xgb.XGBRegressor(**params)
        m.fit(Xtr, ytr)
        oof[te] = m.predict(X26.iloc[te])
        if keep_models:
            models.append(m); tests.append(te)
    return -oof, X26, models, tests


# ── Phase A: learned reliance ────────────────────────────────────────────
def phase_a(df25, df26, shap_sample=150_000):
    feats = ALL_FEATS
    X26, X25 = design(df26, feats), design(df25, feats)
    X_all = pd.concat([X26, X25], ignore_index=True)
    y_all = np.concatenate([df26['target_xrv'].values, df25['target_xrv'].values])
    params = dict(T.TUNED)
    params['monotone_constraints'] = tuple(
        -1 if c == T.MONO_FEAT else 0 for c in X_all.columns)
    t0 = time.time()
    model = xgb.XGBRegressor(**params)
    model.fit(X_all, y_all)
    print(f'  full-data fit: {(time.time() - t0) / 60:.1f} min', flush=True)

    booster = model.get_booster()
    imp = {}
    for kind in ('gain', 'total_gain', 'weight', 'cover'):
        sc = booster.get_score(importance_type=kind)
        imp[kind] = {f: sc.get(f, 0.0) for f in feats}

    rng = np.random.RandomState(20260707)
    idx = rng.choice(len(X26), min(shap_sample, len(X26)), replace=False)
    Xs = X26.iloc[idx]
    t0 = time.time()
    dm = xgb.DMatrix(Xs)
    contribs = booster.predict(dm, pred_contribs=True)  # (n, n_feat+1)
    print(f'  SHAP on {len(Xs)} 2026 pitches: {(time.time() - t0) / 60:.1f} min',
          flush=True)
    mean_abs = np.abs(contribs[:, :-1]).mean(axis=0)
    shap_share = mean_abs / mean_abs.sum()
    # convert to Stuff+ points: raw sd -> 10 pts per between-pitcher SD is
    # downstream; report raw target units AND share
    shap_tbl = pd.DataFrame({
        'feature': feats,
        'shap_mean_abs': mean_abs,
        'shap_share_pct': 100 * shap_share,
    })

    # per-pitch-type SHAP shares (which inputs drive which pitch types)
    pt_rows = []
    pts = df26['pitch_type'].values[idx]
    for pt in sorted(set(pts)):
        m = pts == pt
        if m.sum() < 2000:
            continue
        ma = np.abs(contribs[m, :-1]).mean(axis=0)
        row = {'pitch_type': pt, 'n': int(m.sum())}
        row.update({f: 100 * ma[i] / ma.sum() for i, f in enumerate(feats)})
        pt_rows.append(row)

    corr = X26.corr().round(3)

    out = {
        'importance': imp,
        'shap': shap_tbl.to_dict('records'),
        'shap_by_pitch_type': pt_rows,
        'corr': corr.to_dict(),
    }
    path = os.path.join(OUT_DIR, 'stuff_audit_phase_a.json')
    json.dump(out, open(path, 'w'), indent=1, default=float)

    print('\n  feature reliance (full production model):')
    tg = imp['total_gain']; tgs = sum(tg.values()) or 1.0
    print(f"  {'feature':14s} {'shap_share':>10s} {'totgain%':>9s} {'splits':>7s}")
    for _, r in shap_tbl.sort_values('shap_share_pct', ascending=False).iterrows():
        f = r['feature']
        print(f"  {f:14s} {r['shap_share_pct']:9.1f}% {100 * tg[f] / tgs:8.1f}% "
              f"{imp['weight'][f]:7.0f}")
    print(f'\n  high feature correlations (|r| >= 0.4):')
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            c = corr.loc[a, b]
            if abs(c) >= 0.4:
                print(f'    {a:12s} x {b:12s} r = {c:+.2f}')
    print(f'  saved -> {path}')


# ── Phase B: permutation at scoring time ────────────────────────────────
def phase_b(df25, df26, n_splits):
    feats = ALL_FEATS
    late = late_targets(df26)
    t0 = time.time()
    base_oof, X26, models, tests = fit_oof(df26, df25, feats, n_splits,
                                           keep_models=True)
    print(f'  baseline OOF ({n_splits}-fold): {(time.time() - t0) / 60:.1f} min',
          flush=True)
    base = unit_metrics(df26, base_oof, late)
    base_pr = pearson(base_oof, -df26['target_xrv'].values)
    print(f"  {'config':16s} {'reliab':>7s} {'pred':>7s} {'desc':>7s} {'pitch_r':>8s}")
    print(f"  {'baseline':16s} {base[0]:7.3f} {base[1]:7.3f} {base[2]:7.3f} "
          f"{base_pr:8.4f}", flush=True)

    rng = np.random.RandomState(20260707)
    rows = [dict(feature='baseline', reliab=base[0], pred=base[1], desc=base[2],
                 pitch_r=base_pr)]
    for f in feats:
        Xp = X26.copy()
        Xp[f] = Xp[f].values[rng.permutation(len(Xp))]
        oof = np.full(len(df26), np.nan)
        for m, te in zip(models, tests):
            oof[te] = m.predict(Xp.iloc[te])
        oof = -oof
        rel, pr, dsc = unit_metrics(df26, oof, late)
        pr_pitch = pearson(oof, -df26['target_xrv'].values)
        rows.append(dict(feature=f, reliab=rel, pred=pr, desc=dsc, pitch_r=pr_pitch))
        print(f'  perm {f:11s} {rel:7.3f} {pr:7.3f} {dsc:7.3f} {pr_pitch:8.4f}',
              flush=True)
    out = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, 'stuff_audit_phase_b.csv')
    out.to_csv(path, index=False)
    print(f'  saved -> {path}')


# ── Phase C: drop-one retrains ───────────────────────────────────────────
def phase_c(df25, df26, n_splits, only=None):
    late = late_targets(df26)
    configs = [('baseline', ALL_FEATS)]
    for f in (only or ALL_FEATS):
        configs.append((f'drop_{f}', [x for x in ALL_FEATS if x != f]))
    rows = []
    print(f"  {'config':20s} {'reliab':>7s} {'pred':>7s} {'desc':>7s} {'mins':>6s}",
          flush=True)
    for name, feats in configs:
        t0 = time.time()
        oof, _, _, _ = fit_oof(df26, df25, feats, n_splits)
        rel, pr, dsc = unit_metrics(df26, oof, late)
        mins = (time.time() - t0) / 60
        rows.append(dict(config=name, reliab=rel, pred=pr, desc=dsc, mins=mins))
        print(f'  {name:20s} {rel:7.3f} {pr:7.3f} {dsc:7.3f} {mins:6.1f}',
              flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(OUT_DIR, f'stuff_audit_phase_c_{n_splits}fold.csv'),
            index=False)
    print('  done.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['a', 'b', 'c', 'd'])
    ap.add_argument('--folds', type=int, default=4)
    ap.add_argument('--feats', default=None,
                    help='comma list: restrict phase c drops to these features')
    args = ap.parse_args()

    df25, df26 = R.load_dfs(use_cache=True)
    print(f'rows: 2025={len(df25)}, 2026={len(df26)}', flush=True)

    if args.phase == 'a':
        phase_a(df25, df26)
    elif args.phase == 'b':
        phase_b(df25, df26, args.folds)
    elif args.phase == 'd':
        phase_d(df25, df26, args.folds)
    else:
        only = [f.strip() for f in args.feats.split(',')] if args.feats else None
        phase_c(df25, df26, args.folds, only)




# ── Phase D: 8-fold confirmation of drop candidates ──────────────────────
def fit_oof_seed(df26, df25, feats, n_splits, seed):
    X26 = design(df26, feats)
    X25 = design(df25, feats)
    y26, y25 = df26['target_xrv'].values, df25['target_xrv'].values
    g26 = df26['pitcher'].values
    params = dict(T.TUNED)
    params['random_state'] = seed
    params['monotone_constraints'] = tuple(
        -1 if c == T.MONO_FEAT else 0 for c in X26.columns)
    oof = np.full(len(df26), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X26, y26, g26):
        Xtr = pd.concat([X26.iloc[tr], X25], ignore_index=True)
        ytr = np.concatenate([y26[tr], y25])
        m = xgb.XGBRegressor(**params)
        m.fit(Xtr, ytr)
        oof[te] = m.predict(X26.iloc[te])
    return -oof


def per_type_desc(df26, stuff):
    """Descriptive r by pitch type (>=100-pitch units)."""
    d = df26[['pitcher', 'throws', 'pitch_type', 'target_xrv']].copy()
    d['stuff'] = stuff
    out = {}
    for pt, sub in d.groupby('pitch_type'):
        xs, ys = [], []
        for key, grp in sub.groupby(['pitcher', 'throws']):
            if len(grp) >= 100:
                xs.append(grp['stuff'].mean()); ys.append(grp['target_xrv'].mean())
        r = pearson(xs, ys)
        out[pt] = (None if r is None else -r, len(xs))
    return out


def phase_d(df25, df26, n_splits=8):
    late = late_targets(df26)
    NOARM = [f for f in ALL_FEATS if f != 'arm_angle']
    configs = [
        ('baseline', ALL_FEATS, (0, 1, 2)),
        ('drop_rel_z', [f for f in ALL_FEATS if f != 'rel_z'], (0, 1, 2)),
        ('drop_rz_hbd_spin', [f for f in ALL_FEATS
                              if f not in ('rel_z', 'hb_diff', 'spin_rate')], (0, 1, 2)),
        ('drop_hb_diff', [f for f in ALL_FEATS if f != 'hb_diff'], (0,)),
        ('drop_spin_rate', [f for f in ALL_FEATS if f != 'spin_rate'], (0,)),
        ('drop_platoon', [f for f in ALL_FEATS if f != 'platoon_same'], (0,)),
        ('noarm_base', NOARM, (0,)),
        ('noarm_no_rel_z', [f for f in NOARM if f != 'rel_z'], (0,)),
    ]
    rows = []
    keep_ptd = {}
    print(f"  {'config':20s} {'seed':>4s} {'reliab':>7s} {'pred':>7s} {'desc':>7s} {'mins':>6s}",
          flush=True)
    for name, feats, seeds in configs:
        for seed in seeds:
            t0 = time.time()
            oof = fit_oof_seed(df26, df25, feats, n_splits, seed)
            rel, pr, dsc = unit_metrics(df26, oof, late)
            mins = (time.time() - t0) / 60
            rows.append(dict(config=name, seed=seed, reliab=rel, pred=pr,
                             desc=dsc, mins=mins))
            print(f'  {name:20s} {seed:4d} {rel:7.3f} {pr:7.3f} {dsc:7.3f} {mins:6.1f}',
                  flush=True)
            if seed == 0 and name in ('baseline', 'drop_rel_z', 'drop_rz_hbd_spin',
                                      'drop_spin_rate', 'drop_hb_diff'):
                keep_ptd[name] = per_type_desc(df26, oof)
            pd.DataFrame(rows).to_csv(
                os.path.join(OUT_DIR, 'stuff_audit_phase_d.csv'), index=False)
    print('\n  per-pitch-type descriptive r (seed 0):')
    pts = sorted({pt for d_ in keep_ptd.values() for pt in d_})
    hdr = '  ' + 'config'.ljust(20) + ''.join(f'{pt:>8s}' for pt in pts)
    print(hdr)
    for name, d_ in keep_ptd.items():
        line = '  ' + name.ljust(20)
        for pt in pts:
            r, n = d_.get(pt, (None, 0))
            line += ('    None' if r is None else f'{r:8.3f}')
        print(line)
    line = '  ' + '(n units)'.ljust(20)
    for pt in pts:
        line += f"{keep_ptd['baseline'].get(pt, (None, 0))[1]:8d}"
    print(line)
    print('  saved -> stuff_audit_phase_d.csv')


if __name__ == '__main__':
    main()
