"""xrvoe_crossyear.py — does xRVOE persist ACROSS SEASONS? (2021-2026)

The Tier-1 check (scripts/xrvoe_feasibility.py) showed within-season
persistence (+0.22 half-to-half) and incremental prediction (+0.167 partial).
This answers the durability question: is beating your stuff+location
expectation a TRAIT that carries year to year, or season-local context?

Protocol (leakage-controlled):
  - Seasons 2021-2025 built from the public Statcast caches (physics mapped
    with build_historical_training_set's calibration; location/count fields
    carried alongside); 2026 from the pipeline cache (MLB, EP excluded).
  - For each adjacent pair (Y, Y+1), the stuff expectation comes from an
    XGBoost model trained on ALL OTHER SEASONS ONLY (leave-two-years-out) —
    the production bundle can't be used here because 2021-25 are now in its
    training set.
  - Loc ExpRV from that season's own surfaces + Guts (league constants).
  - Stacking (xRV ~ stuff_pred + loc_exprv) fit per season; per-pitch
    residual = xRVOE; units = (pitcher, pitch_type), n >= 200 both years.
  - Report per-pair and pooled: persistence r, and incremental prediction of
    year-(Y+1) actual unit xRV beyond year-Y expectation.

Caveats: 2021-25 movement is raw (no density adjustment) while 2026 is
adjusted — a small feature-convention mismatch tolerated for a residual
study; public pitch tags drift across seasons, so units join on exact
(pitcher, pitch_type) and drifted tags fall out of the sample.

Usage: python3 scripts/xrvoe_crossyear.py
"""
import os, sys, json, pickle, math
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import build_2025_training_set as B25
import build_historical_training_set as H
import pipeline_locplus as L

GUTS = dict(H.GUTS)
GUTS[2025] = (0.3131, 1.2317)
_md = json.load(open(os.path.join(ROOT, 'data', 'metadata_rs.json')))
_g26 = _md.get('gutsConstants') or {}
GUTS[2026] = (_g26.get('lgWOBA', 0.3172), _g26.get('wOBAScale', 1.2343))

DESC_MAP = {
    'hit_into_play': 'In Play', 'hit_into_play_no_out': 'In Play',
    'hit_into_play_score': 'In Play',
    'swinging_strike': 'Swinging Strike', 'swinging_strike_blocked': 'Swinging Strike',
    'foul_tip': 'Swinging Strike', 'foul': 'Foul',
    'called_strike': 'Called Strike',
    'ball': 'Ball', 'blocked_ball': 'Ball', 'automatic_ball': 'Ball',
}


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def build_season(year):
    """Full dicts from the raw statcast cache: stuff features + loc fields."""
    df = pickle.load(open(os.path.join(ROOT, 'data', f'_statcast{year}_cache.pkl'), 'rb'))
    if 'game_type' in df.columns:
        df = df[df['game_type'] == 'R']
    out = []
    for idx, r in enumerate(df.itertuples(index=False)):
        desc = DESC_MAP.get(r.description)
        if desc is None:
            continue
        pt = r.pitch_type if isinstance(r.pitch_type, str) else None
        if not pt or pt == 'EP':
            continue
        pfx_x, pfx_z = _f(r.pfx_x), _f(r.pfx_z)
        rpx, rpz = _f(r.release_pos_x), _f(r.release_pos_z)
        runexp = _f(getattr(r, 'delta_pitcher_run_exp', None))
        if runexp is None:
            dre = _f(r.delta_run_exp)
            runexp = -dre if dre is not None else None
        is_bip = desc == 'In Play'
        b, s = _f(r.balls), _f(r.strikes)
        name = r.player_name if isinstance(r.player_name, str) else None
        if not name:
            continue
        out.append({
            'PitchID': f'{year}_{idx}',
            'Pitcher': name, 'PTeam': 'X',
            'Game Date': str(r.game_date)[:10],
            'Pitch Type': pt,
            'Bats': r.stand if isinstance(r.stand, str) else None,
            'Throws': r.p_throws if isinstance(r.p_throws, str) else None,
            'Velocity': _f(r.release_speed), 'Spin Rate': _f(r.release_spin_rate),
            'IndVertBrk': round(pfx_z * 12, 1) if pfx_z is not None else None,
            'HorzBrk': round(-pfx_x * 12, 1) if pfx_x is not None else None,
            'xIndVrtBrk': round(pfx_z * 12, 1) if pfx_z is not None else None,
            'xHorzBrk': round(-pfx_x * 12, 1) if pfx_x is not None else None,
            'Extension': _f(r.release_extension), 'ArmAngle': _f(r.arm_angle),
            'RelPosX': round(H.RX_A * rpx + H.RX_B, 3) if rpx is not None else None,
            'RelPosZ': round(H.RZ_A * rpz + H.RZ_B, 3) if rpz is not None else None,
            'VAA': B25.vaa_of(_f(r.vy0), _f(r.vz0), _f(r.ay), _f(r.az)),
            'Description': desc,
            'RunExp': runexp,
            'xwOBA': _f(r.estimated_woba_using_speedangle) if is_bip else None,
            'PlateX': _f(r.plate_x), 'PlateZ': _f(r.plate_z),
            'SzTop': _f(r.sz_top), 'SzBot': _f(r.sz_bot),
            'Count': (f'{int(b)}-{int(s)}' if b is not None and s is not None else None),
            '_source': 'MLB',
        })
    return out


def load_2026():
    D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    ep = {(p.get('Pitcher'), p.get('PTeam')) for p in D if p.get('Pitch Type') == 'EP'}
    return [p for p in D if p.get('_source', 'MLB') == 'MLB'
            and (p.get('Pitcher'), p.get('PTeam')) not in ep]


# ── stuff-module functions with pid passthrough (in-memory exec patch) ──
src = open(os.path.join(ROOT, 'stuff_plus_v11', 'train_stuff_v11.py')).read()
NEEDLE = "            'ivb_diff': ivb_diff, 'hb_diff': hb_diff, 'spin_rate': spin,"
assert NEEDLE in src
src = src.replace(NEEDLE, NEEDLE + "\n            'pid': p.get('PitchID'),")
T = {'__name__': '_stuff_mod',
     '__file__': os.path.join(ROOT, 'stuff_plus_v11', 'train_stuff_v11.py')}
exec(compile(src.split('def main()')[0], 'train_stuff_v11.py', 'exec'), T)

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
season_df = {}
season_pitches = {}
for y in SEASONS:
    pitches = load_2026() if y == 2026 else build_season(y)
    season_pitches[y] = pitches
    T['LG_WOBA'], T['WOBA_SCALE'] = GUTS[y]
    d = T['build_df'](pitches)
    d = d[d['target_xrv'].notna()].reset_index(drop=True)
    season_df[y] = d
    print(f'{y}: {len(pitches)} pitches, {len(d)} stuff rows', flush=True)

FEATS = None
PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025), (2025, 2026)]

# ── loc ExpRV per season ──
loc_map = {}
for y in SEASONS:
    baseline = [p for p in season_pitches[y] if L.is_eligible_baseline(p)]
    S = L.build_surfaces(baseline, GUTS[y][0], GUTS[y][1])
    m = {}
    for p in baseline:
        v = L.score_pitch(p, S)
        if v is not None and p.get('PitchID'):
            m[p['PitchID']] = v
    loc_map[y] = m
    print(f'{y}: loc scored {len(m)}', flush=True)

# ── per-pair: leave-two-out stuff model, stack per season, residual units ──
def design(d):
    return T['design'](d)

results = []
pooled_a, pooled_b = [], []
pooled_exp_a, pooled_act_b = [], []
for ya, yb in PAIRS:
    train_years = [y for y in SEASONS if y not in (ya, yb)]
    Xtr = pd.concat([design(season_df[y]) for y in train_years], ignore_index=True)
    ytr = np.concatenate([season_df[y]['target_xrv'].values for y in train_years])
    model = xgb.XGBRegressor(**T['TUNED'])
    model.fit(Xtr, ytr)

    unit = {}
    for y in (ya, yb):
        d = season_df[y].copy()
        d['stuff_pred'] = model.predict(design(d).reindex(columns=Xtr.columns, fill_value=0))
        d['loc_exprv'] = d['pid'].map(loc_map[y])
        d = d[d['loc_exprv'].notna()]
        yv = d['target_xrv'].values
        A = np.column_stack([np.ones(len(d)), d['stuff_pred'].values, d['loc_exprv'].values])
        beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
        d['expect'] = A @ beta
        d['resid'] = yv - d['expect']
        g = d.groupby(['pitcher', 'pitch_type']).agg(
            resid=('resid', 'mean'), expect=('expect', 'mean'),
            act=('target_xrv', 'mean'), n=('resid', 'size'))
        unit[y] = g[g['n'] >= 200]
    common = unit[ya].index.intersection(unit[yb].index)
    if len(common) < 20:
        print(f'{ya}->{yb}: only {len(common)} units — skipped')
        continue
    ra = unit[ya].loc[common, 'resid'].values
    rb = unit[yb].loc[common, 'resid'].values
    r = np.corrcoef(ra, rb)[0, 1]
    results.append((ya, yb, r, len(common)))
    pooled_a.extend(ra); pooled_b.extend(rb)
    pooled_exp_a.extend(unit[ya].loc[common, 'expect'].values)
    pooled_act_b.extend(unit[yb].loc[common, 'act'].values)
    print(f'{ya}->{yb}: persistence r = {r:+.3f}  (units {len(common)})', flush=True)

pa, pb = np.array(pooled_a), np.array(pooled_b)
ea, ab_ = np.array(pooled_exp_a), np.array(pooled_act_b)
print(f'\nPOOLED cross-year persistence: r = {np.corrcoef(pa, pb)[0,1]:+.3f} (n={len(pa)})')
r_exp = np.corrcoef(ea, ab_)[0, 1]
A2 = np.column_stack([np.ones(len(ea)), ea, pa])
bb, *_ = np.linalg.lstsq(A2, ab_, rcond=None)
r_both = np.corrcoef(A2 @ bb, ab_)[0, 1]
res_partial = np.corrcoef(pa, ab_ - np.polyval(np.polyfit(ea, ab_, 1), ea))[0, 1]
print(f'year-Y expectation -> year-Y+1 actual xRV: r = {r_exp:+.3f}')
print(f'+ year-Y xRVOE (stacked):                  r = {r_both:+.3f}')
print(f'partial corr of xRVOE with next-year actual: {res_partial:+.3f}')
