"""accel_test.py — does raw acceleration (ax/az) add signal over IVB/HB? (report item A3)

tjStuff+ v3.0 swapped normalized movement (IVB/HB) for raw accelerations (ax/az),
arguing accelerations retain velocity-dependent flight info that IVB/HB normalize
away. Cleanest test of that claim on our data: add ax/az as EXTRA features next to
IVB/HB and see if the trees find incremental signal. If yes -> the full swap is
worth pursuing; if not, the claim doesn't hold here.

2025-only (the season whose raw Statcast — with ax/ay/az — is cached locally in
_statcast2025_cache.pkl). Re-joins the cache to the retagged 2025 training pickle
by the same velo+plate fingerprint, attaches ax/az, then runs an 8-fold
pitcher-grouped OOF with split-half reliability + early/late pred, BASE vs
BASE+accel. A within-2025 A/B; absolute numbers won't match the 2025+2026
production protocol, only the BASE-vs-variant delta matters.

Usage: python3 scripts/accel_test.py
"""
import os, sys, pickle, warnings
from collections import defaultdict
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import stuff_plus_v11.train_stuff_v11 as T
import scripts.build_2025_training_set as B
sf = T.sf

PKL25 = os.path.join(ROOT, 'data', '_pitches2025_training.pkl')
CACHE = os.path.join(ROOT, 'data', '_statcast2025_cache.pkl')


def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 5:
        return None
    xs, ys = xs[m], ys[m]
    return float(np.corrcoef(xs, ys)[0, 1]) if xs.std() and ys.std() else None


def attach_accel(pitches, df):
    """Re-run the velo+plate fingerprint match, storing ax/az on each pitch."""
    pub = defaultdict(list)
    for row in df.itertuples(index=False):
        pub[(row.game_date, row.player_name)].append(row)
    matched = 0
    for key, mine in B._group(pitches).items():
        cands = pub.get(key, [])
        used = [False] * len(cands)
        for p in mine:
            v, px, pz = sf(p.get('Velocity')), sf(p.get('PlateX')), sf(p.get('PlateZ'))
            if v is None:
                continue
            best_i, best_d = None, 1e9
            for i, c in enumerate(cands):
                if used[i]:
                    continue
                cv, cx, cz = sf(c.release_speed), sf(c.plate_x), sf(c.plate_z)
                if cv is None or abs(cv - v) > 0.25:
                    continue
                d = abs(cv - v) * 2.0
                if px is not None and cx is not None:
                    d += abs(cx - px)
                if pz is not None and cz is not None:
                    d += abs(cz - pz)
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is None or best_d > 0.5:
                continue
            used[best_i] = True
            c = cands[best_i]
            p['ax'] = sf(c.ax); p['az'] = sf(c.az)
            matched += 1
    print(f"accel join: {matched}/{len(pitches)} matched ({matched/len(pitches):.1%})")


def build_accel(pitches):
    """Self-contained copy of production build_df + ax_norm/az columns (guarantees
    row alignment; the arm-angle placeholder path makes a re-walk fragile)."""
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

    arm_pt = defaultdict(lambda: [0.0, 0]); arm_all = defaultdict(lambda: [0.0, 0])
    for p in pitches:
        aa = sf(p.get('ArmAngle'))
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
        v, spin = sf(p.get('Velocity')), sf(p.get('Spin Rate'))
        iv, hb_raw = sf(p.get('xIndVrtBrk')), sf(p.get('xHorzBrk'))
        vaa, ext = sf(p.get('VAA')), sf(p.get('Extension'))
        arm, rel_z = sf(p.get('ArmAngle')), sf(p.get('RelPosZ'))
        rel_x_raw = sf(p.get('RelPosX'))
        if arm is None:
            arm = _arm(p.get('Pitcher'), pt)
        if None in (v, iv, hb_raw, vaa, ext, rel_z, rel_x_raw):
            continue
        s = 1.0 if thr == 'R' else -1.0
        hb = hb_raw * s; rel_x = rel_x_raw * s
        fbref = primary.get((p.get('Pitcher'), thr))
        if fbref:
            velo_diff = v - fbref['v']; ivb_diff = iv - fbref['iv']; hb_diff = hb - fbref['hb']
            vaa_diff = (vaa - fbref['vaa']) if fbref['vaa'] is not None else None
        else:
            velo_diff = ivb_diff = hb_diff = vaa_diff = None
        xw, re = sf(p.get('xwOBA')), sf(p.get('RunExp'))
        if p.get('Description') == 'In Play' and xw is not None:
            target = (xw - T.LG_WOBA) / T.WOBA_SCALE
        elif re is not None:
            target = -re
        else:
            target = None
        ax, az = sf(p.get('ax')), sf(p.get('az'))
        rows.append({
            'pitcher': p.get('Pitcher'), 'throws': thr, 'pitch_type': pt, 'date': p.get('Game Date'),
            'platoon_same': 1 if bats == thr else 0,
            'velocity': v, 'ivb': iv, 'hb': hb, 'velo_diff': velo_diff, 'ivb_diff': ivb_diff,
            'hb_diff': hb_diff, 'spin_rate': spin, 'extension': ext, 'arm_angle': arm,
            'rel_z': rel_z, 'rel_x': rel_x, 'vaa': vaa, 'vaa_diff': vaa_diff, 'target_xrv': target,
            'ax_norm': (ax * s if ax is not None else np.nan), 'az': (az if az is not None else np.nan),
        })
    return pd.DataFrame(rows)


def evaluate(df, feats):
    d = df[df.target_xrv.notna() & df[feats].notna().all(1)].reset_index(drop=True)
    X = pd.concat([d[feats], d[['platoon_same']]], axis=1)
    y = d.target_xrv.values; g = d.pitcher.values
    p = T._params_for(X)
    oof = np.full(len(d), np.nan)
    for tr, te in GroupKFold(n_splits=8).split(X, y, g):
        m = xgb.XGBRegressor(**p); m.fit(X.iloc[tr], y[tr]); oof[te] = m.predict(X.iloc[te])
    d = d.copy(); d['stuff'] = -oof
    dates = sorted(d['date'].dropna().unique())
    mid = dates[len(dates) // 2]
    d['period'] = np.where(d['date'] < mid, 'early', 'late')
    order = {dd: i for i, dd in enumerate(dates)}
    d['half'] = d['date'].map(order).fillna(0).astype(int) % 2
    a0, a1, est, late = [], [], {}, {}
    for key, grp in d.groupby(['pitcher', 'throws', 'pitch_type']):
        e, l = grp[grp.period == 'early'], grp[grp.period == 'late']
        if len(e) >= 50 and len(l) >= 50:
            late[key] = l.target_xrv.mean(); est[key] = e.stuff.mean()
        h0, h1 = grp[grp.half == 0], grp[grp.half == 1]
        if len(h0) >= 40 and len(h1) >= 40:
            a0.append(h0.stuff.mean()); a1.append(h1.stuff.mean())
    ks = list(est)
    return pearson(a0, a1), -pearson([est[k] for k in ks], [late[k] for k in ks]), len(a0), len(ks), len(d)


def main():
    p25 = pickle.load(open(PKL25, 'rb'))
    df = pickle.load(open(CACHE, 'rb'))
    attach_accel(p25, df)
    adf = build_accel(p25)
    cov = adf[['ax_norm', 'az']].notna().all(1).mean()
    print(f"rows={len(adf)}  ax/az coverage={cov:.1%}\n")
    print(f"{'variant':22s} {'reliab':>7s} {'pred':>7s}   n_rel/n_pred/N")
    base = evaluate(adf, list(T.BASE_FEATS))
    accel = evaluate(adf, list(T.BASE_FEATS) + ['ax_norm', 'az'])
    for name, r in [('BASE (ivb/hb)', base), ('BASE + ax/az', accel)]:
        print(f"{name:22s} {r[0]:7.3f} {r[1]:7.3f}   {r[2]}/{r[3]}/{r[4]}")
    print(f"\ndelta reliab {accel[0]-base[0]:+.3f}   delta pred {accel[1]-base[1]:+.3f}")
    print("ADOPT-worthy if accel adds pred with no reliability loss; else IVB/HB already capture it.")


if __name__ == '__main__':
    main()
