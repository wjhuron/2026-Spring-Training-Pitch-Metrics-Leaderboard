"""squared_up_explore.py — feasibility of a squared-up-allowed stuff target (#3).

Squared-up (MLB): a batted ball reaches >=80% of the max exit velo the collision
could have produced given bat speed + pitch speed. maxEV = 1.23*bat_speed +
0.2116*pitch_speed; squared_up = EV/maxEV >= 0.80. (Calibrated: 64.6% of BIP, ~
MLB's published rate.) "Squared-up-allowed" = the pitcher/pitch analogue: how
often the pitch, when put in play, yields flush contact. LOWER is better stuff.

This does NOT ship anything. It answers four questions the scouting report needs:
  Q1 RELIABLE?     is squared-up-allowed a stable pitcher x pitch-type skill?
  Q2 PREDICTABLE?  can the current physics feature set predict it (OOF AUC)?
  Q3 LOCATION?     how much is physics vs where the pitch was located?
  Q4 ORTHOGONAL?   does it add signal beyond the current xwOBAcon BIP target,
                   or is it redundant with what Stuff+ already learns?

2026 MLB only (bat tracking is 2024+; 2026 pickle carries it on swings).
Usage: python3 scripts/squared_up_explore.py
"""
import os, sys, pickle, warnings
from collections import defaultdict
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import stuff_plus_v11.train_stuff_v11 as T
sf = T.sf


def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 5:
        return None
    xs, ys = xs[m], ys[m]
    return float(np.corrcoef(xs, ys)[0, 1]) if xs.std() and ys.std() else None


def squared_up(ev, bs, rs):
    if None in (ev, bs, rs):
        return None
    maxev = 1.23 * bs + 0.2116 * rs
    return 1 if maxev > 0 and ev / maxev >= 0.80 else 0


def build_labeled(pitches):
    """Production build_df anchor + per-BIP squared-up label + plate location."""
    fb = defaultdict(lambda: defaultdict(lambda: {'v': 0.0, 'iv': 0.0, 'hb': 0.0, 'vaa': 0.0, 'n': 0, 'n_vaa': 0}))
    for p in pitches:
        pt, thr = p.get('Pitch Type'), p.get('Throws')
        if pt not in T.FB_TYPES or thr not in ('L', 'R'):
            continue
        v, iv, hb, vaa = sf(p.get('Velocity')), sf(p.get('IndVertBrk')), sf(p.get('HorzBrk')), sf(p.get('VAA'))
        if None in (v, iv, hb):
            continue
        s = 1.0 if thr == 'R' else -1.0
        a = fb[(p.get('Pitcher'), thr)][pt]
        a['v'] += v; a['iv'] += iv; a['hb'] += hb * s; a['n'] += 1
        if vaa is not None:
            a['vaa'] += vaa; a['n_vaa'] += 1
    primary = {}
    for k, bt in fb.items():
        if k[0] in T.FC_ANCHOR_PITCHERS and 'FC' in bt:
            cand = {'FC': bt['FC']}
        else:
            cand = {pt: d for pt, d in bt.items() if pt in ('FF', 'SI')} or bt
        sel = max(cand, key=lambda pt: cand[pt]['n'])
        b = cand[sel]; n = b['n']
        primary[k] = {'v': b['v'] / n, 'iv': b['iv'] / n, 'hb': b['hb'] / n,
                      'vaa': (b['vaa'] / b['n_vaa']) if b['n_vaa'] else None}

    rows = []
    for p in pitches:
        pt, thr, bats = p.get('Pitch Type'), p.get('Throws'), p.get('Bats')
        if pt not in T.SUPPORTED or thr not in ('L', 'R') or bats not in ('L', 'R'):
            continue
        v, spin = sf(p.get('Velocity')), sf(p.get('Spin Rate'))
        iv, hb_raw = sf(p.get('xIndVrtBrk')), sf(p.get('xHorzBrk'))
        vaa, ext = sf(p.get('VAA')), sf(p.get('Extension'))
        arm, rel_z = sf(p.get('ArmAngle')), sf(p.get('RelPosZ'))
        rel_x_raw = sf(p.get('RelPosX'))
        if None in (v, iv, hb_raw, vaa, ext, rel_z, rel_x_raw, arm):
            continue
        s = 1.0 if thr == 'R' else -1.0
        hb = hb_raw * s; rel_x = rel_x_raw * s
        fbref = primary.get((p.get('Pitcher'), thr))
        if fbref:
            velo_diff = v - fbref['v']; ivb_diff = iv - fbref['iv']; hb_diff = hb - fbref['hb']
            vaa_diff = (vaa - fbref['vaa']) if fbref['vaa'] is not None else None
        else:
            velo_diff = ivb_diff = hb_diff = vaa_diff = None
        is_bip = p.get('Description') == 'In Play'
        ev, bs = sf(p.get('ExitVelo')), sf(p.get('BatSpeed'))
        su = squared_up(ev, bs, v) if is_bip else None
        xw = sf(p.get('xwOBA'))
        xwoba_con = ((xw - T.LG_WOBA) / T.WOBA_SCALE) if (is_bip and xw is not None) else None
        px, pz = sf(p.get('PlateX')), sf(p.get('PlateZ'))
        rows.append({
            'pitcher': p.get('Pitcher'), 'throws': thr, 'pitch_type': pt,
            'date': p.get('Game Date'), 'platoon_same': 1 if bats == thr else 0,
            'velocity': v, 'ivb': iv, 'hb': hb, 'velo_diff': velo_diff,
            'ivb_diff': ivb_diff, 'hb_diff': hb_diff, 'spin_rate': spin,
            'extension': ext, 'arm_angle': arm, 'rel_z': rel_z, 'rel_x': rel_x,
            'vaa': vaa, 'vaa_diff': vaa_diff,
            'is_bip': is_bip, 'squared_up': su, 'xwoba_con': xwoba_con,
            'plate_x': px, 'plate_z': pz,
        })
    df = pd.DataFrame(rows)
    return df[df[T.BASE_FEATS].notna().all(1)].reset_index(drop=True)


def main():
    mlb = [p for p in pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
           if p.get('_source') == 'MLB']
    df = build_labeled(mlb)
    bip = df[df.is_bip & df.squared_up.notna()].reset_index(drop=True)
    print(f"BIP with squared-up label: {len(bip)}   league squared-up-allowed rate: {bip.squared_up.mean():.1%}\n")

    # -- Q1 RELIABLE? split-half of raw squared-up-allowed by pitcher x type --
    order = {d: i for i, d in enumerate(sorted(bip['date'].dropna().unique()))}
    bip['half'] = bip['date'].map(order).fillna(0).astype(int) % 2
    a0, a1, minN = [], [], 40
    for _, g in bip.groupby(['pitcher', 'throws', 'pitch_type']):
        h0, h1 = g[g.half == 0], g[g.half == 1]
        if len(h0) >= minN and len(h1) >= minN:
            a0.append(h0.squared_up.mean()); a1.append(h1.squared_up.mean())
    print(f"Q1 RELIABLE?  raw squared-up-allowed split-half r = {pearson(a0,a1):.3f}  (n={len(a0)} cells, >={minN}/half)")
    print(f"              (compare: your xRV Stuff+ split-half reliability ~0.86)\n")

    # -- Q2/Q3 PREDICTABLE + LOCATION? OOF AUC physics-only vs +location --
    y = bip.squared_up.values.astype(int); g = bip.pitcher.values
    params = dict(max_depth=6, n_estimators=400, learning_rate=0.03, min_child_weight=20,
                  subsample=0.8, colsample_bytree=0.8, n_jobs=-1, tree_method='hist',
                  objective='binary:logistic', eval_metric='auc')
    for label, feats in [('physics only (BASE_FEATS)', T.BASE_FEATS + ['platoon_same']),
                         ('physics + plate location', T.BASE_FEATS + ['platoon_same', 'plate_x', 'plate_z'])]:
        X = bip[feats]
        oof = np.full(len(bip), np.nan)
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            m = xgb.XGBClassifier(**params); m.fit(X.iloc[tr], y[tr]); oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        auc = roc_auc_score(y, oof)
        bip[f'pred_{label[:4]}'] = oof
        tag = 'Q2 PREDICTABLE?' if 'only' in label else 'Q3 LOCATION?  '
        print(f"{tag}  {label:28s} OOF AUC = {auc:.3f}")
    print("              (AUC 0.5 = physics says nothing; gap to +location = the location share)\n")

    # model-score reliability: does the physics prediction of squared-up persist?
    b0, b1 = [], []
    for _, gg in bip.groupby(['pitcher', 'throws', 'pitch_type']):
        h0, h1 = gg[gg.half == 0], gg[gg.half == 1]
        if len(h0) >= minN and len(h1) >= minN:
            b0.append(h0['pred_phys'].mean()); b1.append(h1['pred_phys'].mean())
    print(f"              physics-model squared-up score split-half r = {pearson(b0,b1):.3f}\n")

    # -- Q4 ORTHOGONAL? squared-up-allowed vs xwOBAcon-allowed at cell level --
    cx, cy = [], []
    for _, g2 in bip.groupby(['pitcher', 'throws', 'pitch_type']):
        if len(g2) >= 60 and g2.xwoba_con.notna().sum() >= 60:
            cx.append(g2.squared_up.mean()); cy.append(g2.xwoba_con.mean())
    r = pearson(cx, cy)
    print(f"Q4 ORTHOGONAL?  squared-up-allowed vs xwOBAcon-allowed (cell means, n={len(cx)}): r = {r:.3f}")
    print(f"                r^2 = {r*r:.2f} shared. 1 - r^2 = {1-r*r:.2f} of squared-up is NEW vs your BIP target.")


if __name__ == '__main__':
    main()
