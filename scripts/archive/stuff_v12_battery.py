"""stuff_v12_battery.py — Stuff+ v12 candidate battery (2026-07-14).

Variants, each under the production protocol (2025 prior joins every fold's
training set; 2026 scored pitcher-grouped 8-fold OOF):

  BASE     current BASE_FEATS
  AXIS     + axis_dev, axis_dev_abs   (OTilt-RTilt circular diff, deg,
                                       hand-signed — the SSW proxy; Wally's
                                       own tilt conventions on both seasons)
  HAA      + haa, haa_diff            (hand-signed horizontal approach angle
                                       + differential vs primary fastball)
  EVELO    + eff_delta                (extension-driven perceived-velo delta:
                                       velo*(54.1/(60.5-ext) - 1))
  ALL3     + all of the above
  VCAL     BASE + per-fold velo recalibration (train-fold linear resid~velo
           applied to test preds — the honest fix for the velo-tail lean)
  D7 / D8  BASE at max_depth 7 / 8    (capacity retune probe)

Metrics on 2026: split-half reliability (odd/even dates, >=40/half),
pred future xRV (early->late, >=50 each), descriptive (>=100), plus the
drain diagnostics: unit-level corr(residual, velo) and corr(residual,
axis_dev) — did the new feature absorb what xRVOE was catching?

2025 prior is rebuilt from the RAW statcast cache so the axis/haa/eff
features exist there too (public tags; fine — the model is agnostic).

Usage: python3 scripts/stuff_v12_battery.py
"""
import os, sys, math, json, pickle, time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import build_2025_training_set as B25
import build_historical_training_set as H
from pipeline_utils import break_tilt_to_minutes

# ── stuff module (exec, pre-main) with pid passthrough already in prod ──
src = open(os.path.join(ROOT, 'stuff_plus_v11', 'train_stuff_v11.py')).read()
T = {'__name__': '_stuff_mod',
     '__file__': os.path.join(ROOT, 'stuff_plus_v11', 'train_stuff_v11.py')}
exec(compile(src.split('def main()')[0], 'train_stuff_v11.py', 'exec'), T)

GUTS25 = (0.3131, 1.2317)


def circ_dev_deg(otilt_min, rtilt_min):
    """Signed circular difference OTilt-RTilt in degrees, wrapped [-180,180)."""
    if otilt_min is None or rtilt_min is None:
        return None
    d = (otilt_min - rtilt_min) % 720
    if d >= 360:
        d -= 720
    return d * 0.5


def spin_axis_to_min(deg):
    """Savant spin_axis degrees -> tilt minutes (Wally's RTilt convention:
    180 deg = 12:00; hours = deg/30 - 6)."""
    if deg is None:
        return None
    h = (deg / 30.0 - 6.0) % 12.0
    return h * 60.0


def move_to_min(ivb, hb):
    """Movement direction -> tilt minutes (Wally's OTilt convention:
    atan2(HB, IVB) clockwise from 12:00)."""
    if ivb is None or hb is None or (ivb == 0 and hb == 0):
        return None
    deg = math.degrees(math.atan2(hb, ivb)) % 360.0
    return deg * 2.0


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


DESC_MAP = {
    'hit_into_play': 'In Play', 'hit_into_play_no_out': 'In Play',
    'hit_into_play_score': 'In Play',
    'swinging_strike': 'Swinging Strike', 'swinging_strike_blocked': 'Swinging Strike',
    'foul_tip': 'Swinging Strike', 'foul': 'Foul', 'called_strike': 'Called Strike',
    'ball': 'Ball', 'blocked_ball': 'Ball', 'automatic_ball': 'Ball',
}


def build_2025():
    df = pickle.load(open(os.path.join(ROOT, 'data', '_statcast2025_cache.pkl'), 'rb'))
    if 'game_type' in df.columns:
        df = df[df['game_type'] == 'R']
    out = []
    for r in df.itertuples(index=False):
        desc = DESC_MAP.get(r.description)
        pt = r.pitch_type if isinstance(r.pitch_type, str) else None
        if desc is None or not pt or pt == 'EP':
            continue
        pfx_x, pfx_z = _f(r.pfx_x), _f(r.pfx_z)
        rpx, rpz = _f(r.release_pos_x), _f(r.release_pos_z)
        runexp = _f(getattr(r, 'delta_pitcher_run_exp', None))
        if runexp is None:
            dre = _f(r.delta_run_exp)
            runexp = -dre if dre is not None else None
        is_bip = desc == 'In Play'
        name = r.player_name if isinstance(r.player_name, str) else None
        if not name:
            continue
        ivb = round(pfx_z * 12, 1) if pfx_z is not None else None
        hb = round(-pfx_x * 12, 1) if pfx_x is not None else None
        out.append({
            'Pitcher': name, 'PTeam': 'X',
            'Game Date': str(r.game_date)[:10], 'Pitch Type': pt,
            'Bats': r.stand if isinstance(r.stand, str) else None,
            'Throws': r.p_throws if isinstance(r.p_throws, str) else None,
            'Velocity': _f(r.release_speed), 'Spin Rate': _f(r.release_spin_rate),
            'IndVertBrk': ivb, 'HorzBrk': hb, 'xIndVrtBrk': ivb, 'xHorzBrk': hb,
            'Extension': _f(r.release_extension), 'ArmAngle': _f(r.arm_angle),
            'RelPosX': round(H.RX_A * rpx + H.RX_B, 3) if rpx is not None else None,
            'RelPosZ': round(H.RZ_A * rpz + H.RZ_B, 3) if rpz is not None else None,
            'VAA': B25.vaa_of(_f(r.vy0), _f(r.vz0), _f(r.ay), _f(r.az)),
            'HAA': B25.vaa_of(_f(r.vy0), _f(r.vx0), _f(r.ay), _f(r.ax)),
            '_rt_min': spin_axis_to_min(_f(r.spin_axis)),
            '_ot_min': move_to_min(ivb, hb),
            'Description': desc, 'RunExp': runexp,
            'xwOBA': _f(r.estimated_woba_using_speedangle) if is_bip else None,
            '_source': 'MLB',
        })
    return out


def load_2026():
    D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    ep = {(p.get('Pitcher'), p.get('PTeam')) for p in D if p.get('Pitch Type') == 'EP'}
    out = []
    for p in D:
        if p.get('_source', 'MLB') != 'MLB':
            continue
        if (p.get('Pitcher'), p.get('PTeam')) in ep:
            continue
        p = dict(p)
        p['_rt_min'] = break_tilt_to_minutes(p.get('RTilt'))
        p['_ot_min'] = break_tilt_to_minutes(p.get('OTilt'))
        out.append(p)
    return out


def add_features(pitches, d):
    """Attach v12 candidate features to a built df (row order = build order
    for pitches passing build_df's filters — join on pid).

    axis_dev: hand-signed circular OTilt-RTilt (deg).
    haa: hand-signed HAA; haa_diff vs the pitcher's primary fastball.
    eff_delta: extension-driven perceived-velo delta."""
    by_pid = {}
    for i, p in enumerate(pitches):
        pid = p.get('PitchID') or f'x_{i}'
        p['PitchID'] = pid
        s = 1.0 if p.get('Throws') == 'R' else -1.0
        dev = circ_dev_deg(p.get('_ot_min'), p.get('_rt_min'))
        haa = _f(p.get('HAA'))
        by_pid[pid] = (s * dev if dev is not None else None,
                       abs(dev) if dev is not None else None,
                       s * haa if haa is not None else None)
    d['axis_dev'] = d['pid'].map(lambda k: by_pid.get(k, (None,)*3)[0])
    d['axis_dev_abs'] = d['pid'].map(lambda k: by_pid.get(k, (None,)*3)[1])
    d['haa'] = d['pid'].map(lambda k: by_pid.get(k, (None,)*3)[2])
    d['eff_delta'] = d['velocity'] * (54.1 / (60.5 - d['extension']) - 1.0)
    # haa_diff vs primary fastball (same convention as ivb_diff etc.)
    fb_haa = (d[d['pitch_type'].isin(['FF', 'SI'])]
              .groupby('pitcher')['haa'].mean())
    d['haa_diff'] = d['haa'] - d['pitcher'].map(fb_haa)
    return d


print('building data ...', flush=True)
p26 = load_2026()
p25 = build_2025()
for _i, _p in enumerate(p25):      # pids must exist BEFORE build_df so the
    _p['PitchID'] = f'p25_{_i}'    # feature join has a key on the prior too
T['LG_WOBA'], T['WOBA_SCALE'] = json.load(open(os.path.join(ROOT, 'data', 'metadata_rs.json')))['gutsConstants']['lgWOBA'], None
# build dfs with proper guts
g26 = json.load(open(os.path.join(ROOT, 'data', 'metadata_rs.json')))['gutsConstants']
T['LG_WOBA'], T['WOBA_SCALE'] = g26['lgWOBA'], g26['wOBAScale']
d26 = T['build_df'](p26); d26 = d26[d26['target_xrv'].notna()].reset_index(drop=True)
T['LG_WOBA'], T['WOBA_SCALE'] = GUTS25
d25 = T['build_df'](p25); d25 = d25[d25['target_xrv'].notna()].reset_index(drop=True)
d26 = add_features(p26, d26)
d25 = add_features(p25, d25)
print(f'2026: {len(d26)} rows, 2025 prior: {len(d25)} rows', flush=True)
print(f'axis_dev coverage: 2026 {d26.axis_dev.notna().mean()*100:.0f}%, 2025 {d25.axis_dev.notna().mean()*100:.0f}%')
print(f'haa means (sanity, hand-signed): 2026 {d26.haa.mean():+.2f}, 2025 {d25.haa.mean():+.2f}', flush=True)

BASE = list(T['BASE_FEATS'])
VARIANTS = [
    ('BASE',  BASE,                                   6, False),
    ('AXIS',  BASE + ['axis_dev', 'axis_dev_abs'],    6, False),
    ('HAA',   BASE + ['haa', 'haa_diff'],             6, False),
    ('EVELO', BASE + ['eff_delta'],                   6, False),
    ('ALL3',  BASE + ['axis_dev', 'axis_dev_abs', 'haa', 'haa_diff', 'eff_delta'], 6, False),
    ('VCAL',  BASE,                                   6, True),
    ('D7',    BASE,                                   7, False),
    ('D8',    BASE,                                   8, False),
]

dates = sorted(d26['date'].dropna().unique())
order = {dt: i for i, dt in enumerate(dates)}
d26['half'] = d26['date'].map(order).fillna(0).astype(int) % 2
d26['period'] = np.where(d26['date'] < '2026-05-01', 'early', 'late')
groups = d26['pitcher'].values
y26 = d26['target_xrv'].values


def design(d, feats):
    X = d[feats].reset_index(drop=True)
    return pd.concat([X, d[['platoon_same']].reset_index(drop=True)], axis=1)


def run_variant(name, feats, depth, vcal):
    t0 = time.time()
    params = dict(T['TUNED']); params['max_depth'] = depth
    mono = ','.join('-1' if f == 'velocity' else '0' for f in feats + ['platoon_same'])
    params['monotone_constraints'] = '(' + mono + ')'
    X = design(d26, feats)
    Xp = design(d25, feats)
    yp = d25['target_xrv'].values
    oof = np.full(len(d26), np.nan)
    for tr, te in GroupKFold(n_splits=8).split(X, y26, groups):
        Xtr = pd.concat([X.iloc[tr], Xp], ignore_index=True)
        ytr = np.concatenate([y26[tr], yp])
        m = xgb.XGBRegressor(**params)
        m.fit(Xtr, ytr)
        pred_te = m.predict(X.iloc[te])
        if vcal:
            pred_tr = m.predict(X.iloc[tr])
            resid_tr = y26[tr] - pred_tr
            v = d26['velocity'].values
            msk = ~np.isnan(v[tr])
            b = np.polyfit(v[tr][msk], resid_tr[msk], 1)
            adj = np.polyval(b, v[te])
            pred_te = pred_te + np.where(np.isnan(adj), 0, adj)
        oof[te] = pred_te
    d = d26.copy()
    d['stuff'] = -oof
    # metrics
    a0, a1, est, late = [], [], {}, {}
    for key, grp in d.groupby(['pitcher', 'throws', 'pitch_type']):
        e, l = grp[grp.period == 'early'], grp[grp.period == 'late']
        if len(e) >= 50 and len(l) >= 50:
            late[key] = l.target_xrv.mean(); est[key] = e.stuff.mean()
        h0, h1 = grp[grp.half == 0], grp[grp.half == 1]
        if len(h0) >= 40 and len(h1) >= 40:
            a0.append(h0.stuff.mean()); a1.append(h1.stuff.mean())
    ks = list(est)
    def pear(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return float(np.corrcoef(a, b)[0, 1])
    rel = pear(a0, a1)
    pred = -pear([est[k] for k in ks], [late[k] for k in ks])
    dx, dy = [], []
    for _, gp in d.groupby(['pitcher', 'throws', 'pitch_type']):
        if len(gp) >= 100:
            dx.append(gp.stuff.mean()); dy.append(gp.target_xrv.mean())
    desc = -pear(dx, dy)
    # drain diagnostics: unit residual vs velo / axis_dev
    d['resid'] = d['target_xrv'] - (-d['stuff'])
    g = d.groupby(['pitcher', 'throws', 'pitch_type']).agg(
        resid=('resid', 'mean'), n=('resid', 'size'),
        velo=('velocity', 'mean'), adev=('axis_dev', 'mean'))
    gq = g[g['n'] >= 100]
    rv = pear(gq['resid'], gq['velo'])
    ga = gq[gq['adev'].notna()]
    ra = pear(ga['resid'], ga['adev']) if len(ga) > 50 else float('nan')
    print(f'{name:6s} reliab {rel:.4f}  pred {pred:.4f}  desc {desc:.4f}  '
          f'resid~velo {rv:+.3f}  resid~axis {ra:+.3f}  [{time.time()-t0:.0f}s]', flush=True)
    return dict(name=name, rel=rel, pred=pred, desc=desc, rv=rv, ra=ra)


print(f"\n{'variant':6s} {'reliab':>7s} {'pred':>7s} {'desc':>7s}", flush=True)
res = [run_variant(*v) for v in VARIANTS]
