"""stuff_improve_experiments.py — A/B tests from the 2026-07-05 scouting report.

Runs on the exact production protocol (2025 in every fold's training set, 2026
pitcher-grouped 8-fold OOF, agnostic features, monotone velocity, 2025 tags
harmonized to 2026, 2025 targets on 2025 Guts). Metrics on 2026: split-half
reliability, pred_xRV (early->late future prediction), descriptive.

Variants:
  BASE               current production build_df + BASE_FEATS
  UNEXP_signed       + ivb_resid, hb_resid  (movement minus release-point kNN expectation)
  UNEXP_mag          + move_unexp           (Euclidean magnitude of that residual)
  HANDSPLIT_ANCHOR   fastball differentials anchored per BATTER HANDEDNESS
                     (RHP: FF-vs-LHH / SI-vs-RHH each anchor their own side)

Decision rule (same as fb_anchor): adopt a change unless it drops pred_xRV by
>0.010 with no reliability gain.

Usage: python3 scripts/stuff_improve_experiments.py
"""
import os, sys, math, pickle, warnings, time
from collections import defaultdict
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import stuff_plus_v11.train_stuff_v11 as T

PKL26 = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
PKL25 = os.path.join(ROOT, 'data', '_pitches2025_training.pkl')
LG25, SCALE25 = T.PRIOR_LG_WOBA, T.PRIOR_WOBA_SCALE


def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 5:
        return None
    xs, ys = xs[m], ys[m]
    return float(np.corrcoef(xs, ys)[0, 1]) if xs.std() and ys.std() else None


# ----------------------------------------------------------------------------
# TEST 2: handedness-split fastball anchor. Copy of build_df with the primary
# fastball keyed by (pitcher, throws, BATS); falls back to the pooled
# (pitcher, throws) anchor when a platoon cell has no true fastball.
# ----------------------------------------------------------------------------
def build_df_handsplit(pitches):
    fb = defaultdict(lambda: defaultdict(lambda: {'v': 0.0, 'iv': 0.0, 'hb': 0.0, 'vaa': 0.0, 'n': 0, 'n_vaa': 0}))
    fb_all = defaultdict(lambda: defaultdict(lambda: {'v': 0.0, 'iv': 0.0, 'hb': 0.0, 'vaa': 0.0, 'n': 0, 'n_vaa': 0}))
    for p in pitches:
        pt, thr, bats = p.get('Pitch Type'), p.get('Throws'), p.get('Bats')
        if pt not in T.FB_TYPES or thr not in ('L', 'R') or bats not in ('L', 'R'):
            continue
        v, iv, hb, vaa = T.sf(p.get('Velocity')), T.sf(p.get('IndVertBrk')), T.sf(p.get('HorzBrk')), T.sf(p.get('VAA'))
        if None in (v, iv, hb):
            continue
        s = 1.0 if thr == 'R' else -1.0
        for tgt, key in ((fb, (p.get('Pitcher'), thr, bats)), (fb_all, (p.get('Pitcher'), thr))):
            a = tgt[key][pt]
            a['v'] += v; a['iv'] += iv; a['hb'] += hb * s; a['n'] += 1
            if vaa is not None:
                a['vaa'] += vaa; a['n_vaa'] += 1

    def _pick(bt, name):
        if name in T.FC_ANCHOR_PITCHERS and 'FC' in bt:
            cand = {'FC': bt['FC']}
        else:
            cand = {pt: d for pt, d in bt.items() if pt in ('FF', 'SI')} or bt
        sel = max(cand, key=lambda pt: cand[pt]['n'])
        b = cand[sel]; n = b['n']
        return {'v': b['v'] / n, 'iv': b['iv'] / n, 'hb': b['hb'] / n,
                'vaa': (b['vaa'] / b['n_vaa']) if b['n_vaa'] else None}

    primary = {k: _pick(bt, k[0]) for k, bt in fb.items()}
    primary_all = {k: _pick(bt, k[0]) for k, bt in fb_all.items()}

    # arm-angle placeholder (same as production)
    arm_pt = defaultdict(lambda: [0.0, 0]); arm_all = defaultdict(lambda: [0.0, 0])
    for p in pitches:
        aa = T.sf(p.get('ArmAngle'))
        if aa is None:
            continue
        arm_pt[(p.get('Pitcher'), p.get('Pitch Type'))][0] += aa
        arm_pt[(p.get('Pitcher'), p.get('Pitch Type'))][1] += 1
        arm_all[p.get('Pitcher')][0] += aa; arm_all[p.get('Pitcher')][1] += 1

    def _arm(pit, pt0):
        a = arm_pt.get((pit, pt0))
        if a and a[1]:
            return a[0] / a[1]
        a = arm_all.get(pit)
        return a[0] / a[1] if (a and a[1]) else None

    rows = []
    for p in pitches:
        pt, thr, bats = p.get('Pitch Type'), p.get('Throws'), p.get('Bats')
        if pt not in T.SUPPORTED or thr not in ('L', 'R') or bats not in ('L', 'R'):
            continue
        v, spin = T.sf(p.get('Velocity')), T.sf(p.get('Spin Rate'))
        iv, hb_raw = T.sf(p.get('xIndVrtBrk')), T.sf(p.get('xHorzBrk'))
        vaa, ext = T.sf(p.get('VAA')), T.sf(p.get('Extension'))
        arm, rel_z = T.sf(p.get('ArmAngle')), T.sf(p.get('RelPosZ'))
        rel_x_raw = T.sf(p.get('RelPosX'))
        if arm is None:
            arm = _arm(p.get('Pitcher'), pt)
        if None in (v, iv, hb_raw, vaa, ext, rel_z, rel_x_raw):
            continue
        s = 1.0 if thr == 'R' else -1.0
        hb = hb_raw * s
        rel_x = rel_x_raw * s
        fbref = primary.get((p.get('Pitcher'), thr, bats)) or primary_all.get((p.get('Pitcher'), thr))
        if fbref:
            velo_diff = v - fbref['v']; ivb_diff = iv - fbref['iv']
            hb_diff = hb - fbref['hb']
            vaa_diff = (vaa - fbref['vaa']) if fbref['vaa'] is not None else None
        else:
            velo_diff = ivb_diff = hb_diff = vaa_diff = None
        desc = p.get('Description', '')
        xw, re = T.sf(p.get('xwOBA')), T.sf(p.get('RunExp'))
        if desc == 'In Play' and xw is not None:
            target = (xw - T.LG_WOBA) / T.WOBA_SCALE
        elif re is not None:
            target = -re
        else:
            target = None
        rows.append({
            'pitcher': p.get('Pitcher'), 'team': p.get('PTeam'), 'throws': thr,
            'date': p.get('Game Date'), 'pitch_type': pt,
            'platoon_same': 1 if bats == thr else 0,
            'velocity': v, 'ivb': iv, 'hb': hb, 'velo_diff': velo_diff,
            'ivb_diff': ivb_diff, 'hb_diff': hb_diff, 'spin_rate': spin,
            'extension': ext, 'arm_angle': arm, 'rel_z': rel_z, 'rel_x': rel_x,
            'vaa': vaa, 'vaa_diff': vaa_diff, 'target_xrv': target,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# TEST 1: unexpected movement. Per pitch type, predict (ivb, hb) from
# (rel_x, rel_z, arm_angle) via kNN(10) on a subsample, residual = actual-pred.
# Pure physics transform (no target) -> no leakage; compute once on the pooled df.
# ----------------------------------------------------------------------------
def add_move_resid(df, ref_frac=0.05, k=10, seed=0):
    df = df.copy()
    df['ivb_resid'] = np.nan; df['hb_resid'] = np.nan; df['move_unexp'] = np.nan
    rng = np.random.RandomState(seed)
    for pt, idx in df.groupby('pitch_type').groups.items():
        idx = np.array(idx)
        sub = df.loc[idx]
        Xr = sub[['rel_x', 'rel_z', 'arm_angle']].values
        yv = sub[['ivb', 'hb']].values
        ok = np.isfinite(Xr).all(1) & np.isfinite(yv).all(1)
        if ok.sum() < 50:
            continue
        idx_ok = idx[ok]; Xr_ok = Xr[ok]; yv_ok = yv[ok]
        n = len(idx_ok)
        m = min(n, max(2000, int(n * ref_frac)))
        sel = rng.choice(n, m, replace=False)
        knn = KNeighborsRegressor(n_neighbors=k).fit(Xr_ok[sel], yv_ok[sel])
        pred = knn.predict(Xr_ok)
        resid = yv_ok - pred
        df.loc[idx_ok, 'ivb_resid'] = resid[:, 0]
        df.loc[idx_ok, 'hb_resid'] = resid[:, 1]
        df.loc[idx_ok, 'move_unexp'] = np.sqrt(resid[:, 0] ** 2 + resid[:, 1] ** 2)
    for c in ('ivb_resid', 'hb_resid', 'move_unexp'):
        df[c] = df[c].fillna(0.0)
    return df


def prep(p26, p25, builder=T.build_df):
    df26 = builder(p26)
    _lg, _sc = T.LG_WOBA, T.WOBA_SCALE
    T.LG_WOBA, T.WOBA_SCALE = LG25, SCALE25
    df25 = builder(p25)
    T.LG_WOBA, T.WOBA_SCALE = _lg, _sc
    df26 = df26[df26.target_xrv.notna()].reset_index(drop=True)
    df25 = df25[df25.target_xrv.notna()].reset_index(drop=True)
    order = {d: i for i, d in enumerate(sorted(df26['date'].dropna().unique()))}
    df26['half'] = df26['date'].map(order).fillna(0).astype(int) % 2
    df26['period'] = np.where(df26['date'] < '2026-05-01', 'early', 'late')
    return df26, df25


def evaluate(df26, df25, feats):
    def design(d):
        return pd.concat([d[feats].reset_index(drop=True),
                          d[['platoon_same']].reset_index(drop=True)], axis=1)
    X26 = design(df26); y26 = df26.target_xrv.values; g = df26.pitcher.values
    X25 = design(df25).reindex(columns=X26.columns, fill_value=0); y25 = df25.target_xrv.values
    p = T._params_for(X26)
    oof = np.full(len(df26), np.nan)
    for tr, te in GroupKFold(n_splits=8).split(X26, y26, g):
        Xtr = pd.concat([X26.iloc[tr], X25], ignore_index=True)
        ytr = np.concatenate([y26[tr], y25])
        m = xgb.XGBRegressor(**p); m.fit(Xtr, ytr); oof[te] = m.predict(X26.iloc[te])
    d = df26.copy(); d['stuff'] = -oof
    late, a0, a1, est = {}, [], [], {}
    for key, grp in d.groupby(['pitcher', 'throws', 'pitch_type']):
        e, l = grp[grp.period == 'early'], grp[grp.period == 'late']
        if len(e) >= 50 and len(l) >= 50:
            late[key] = l.target_xrv.dropna().mean()
        h0, h1 = grp[grp.half == 0], grp[grp.half == 1]
        if len(h0) >= 40 and len(h1) >= 40:
            a0.append(h0.stuff.mean()); a1.append(h1.stuff.mean())
        if key in late:
            est[key] = grp[grp.period == 'early'].stuff.mean()
    ks = list(est)
    dx = [gp.stuff.mean() for _, gp in d.groupby(['pitcher', 'throws', 'pitch_type']) if len(gp) >= 100]
    dy = [gp.target_xrv.mean() for _, gp in d.groupby(['pitcher', 'throws', 'pitch_type']) if len(gp) >= 100]
    return (pearson(a0, a1), -pearson([est[k] for k in ks], [late[k] for k in ks]),
            -pearson(dx, dy), len(a0), len(ks))


def main():
    t0 = time.time()
    p26 = [p for p in pickle.load(open(PKL26, 'rb')) if p.get('_source') == 'MLB']
    p25 = pickle.load(open(PKL25, 'rb'))
    T._harmonize_tags(p25, p26)
    print(f"2026={len(p26)}  2025={len(p25)}\n", flush=True)

    # base dfs (production anchor) + residual augmentation
    df26, df25 = prep(p26, p25)
    df26 = add_move_resid(df26); df25 = add_move_resid(df25)
    # handsplit dfs (different anchor)
    hs26, hs25 = prep(p26, p25, builder=build_df_handsplit)

    BASE = list(T.BASE_FEATS)
    variants = [
        ('BASE', df26, df25, BASE),
        ('UNEXP_signed', df26, df25, BASE + ['ivb_resid', 'hb_resid']),
        ('UNEXP_mag', df26, df25, BASE + ['move_unexp']),
        ('HANDSPLIT_ANCHOR', hs26, hs25, BASE),
    ]
    print(f"{'variant':20s} {'reliab':>7s} {'pred_xRV':>8s} {'desc':>7s}   n_rel/n_pred", flush=True)
    res = {}
    for name, a26, a25, feats in variants:
        rel, pred, desc, nrel, npred = evaluate(a26, a25, feats)
        res[name] = (rel, pred, desc)
        print(f"{name:20s} {rel:7.3f} {pred:8.3f} {desc:7.3f}   {nrel}/{npred}", flush=True)

    b = res['BASE']
    print(f"\n{'variant':20s} {'d_reliab':>8s} {'d_pred':>8s} {'d_desc':>8s}   verdict")
    for name in ('UNEXP_signed', 'UNEXP_mag', 'HANDSPLIT_ANCHOR'):
        r = res[name]
        dp = r[1] - b[1]
        verdict = 'ADOPT' if (dp > -0.010 and (dp > 0 or r[0] >= b[0])) else 'HOLD'
        if dp > 0.005 and r[1] > b[1]:
            verdict = 'ADOPT (gain)'
        print(f"{name:20s} {r[0]-b[0]:+8.3f} {dp:+8.3f} {r[2]-b[2]:+8.3f}   {verdict}")
    print(f"\n[{time.time()-t0:.0f}s]")


if __name__ == '__main__':
    main()
