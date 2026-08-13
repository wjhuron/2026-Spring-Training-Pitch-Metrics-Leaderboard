"""build_historical_training_set.py — reconstruct 2021-24 training pitches from
public Statcast (no retagged sheets exist for those years).

Stuff+ is pitch-type AGNOSTIC, so public pitch tags are fine (they feed only the
fastball anchor + display grouping, never the model inputs). Mapping public ->
Wally's dict schema was calibrated against 2026 (where both exist), n=14091:
  HorzBrk    = -1.0 * pfx_x*12         (sheets flip Statcast horizontal sign)
  IndVertBrk = +1.0 * pfx_z*12         (identical)
  ArmAngle   = arm_angle               (identical; Savant backfills it to 2021)
  RelPosX    = 0.9102*release_pos_x - 0.0011   (r=0.999; release projected to true release dist)
  RelPosZ    = 0.9086*release_pos_z + 0.4276   (r=0.981)
  VAA        = vaa_of(vy0,vz0,ay,az)   (same physics build_2025 uses)
Density-adjusted xIndVrtBrk/xHorzBrk come from add_weather (e=1.05), same as 2025.

Per-season wOBA Guts (lgwOBA, wOBAScale) center the BIP target in run units so it
matches the -RunExp non-BIP branch. VALUES ARE BEST-KNOWN FanGraphs Guts — verify
before shipping; a per-season affine error only shifts that season's BIP centering.

Usage:
  python3 scripts/build_historical_training_set.py --sanity   # 2024, no weather, checks only
  python3 scripts/build_historical_training_set.py            # full build all seasons + weather
"""
import os, sys, pickle, warnings
import numpy as np
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import scripts.build_2025_training_set as B
import stuff_plus.train_stuff as T
sf = B.sf

# FanGraphs Guts (lgwOBA, wOBAScale), confirmed by Wally 2026-07-05 from the
# FanGraphs Guts table. (Current pickles were built with 2024=1.239; corrected to
# 1.242 here — a 0.24% BIP-target rescale, immaterial, no rebuild needed.)
GUTS = {
    2021: (0.314, 1.209), 2022: (0.310, 1.259),
    2023: (0.318, 1.204), 2024: (0.310, 1.242),
}
RX_A, RX_B = 0.9102, -0.0011
RZ_A, RZ_B = 0.9086, 0.4276
INPLAY_DESC = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}


def build_season_dicts(year):
    df = pickle.load(open(os.path.join(ROOT, 'data', f'_statcast{year}_cache.pkl'), 'rb'))
    cols = {c: i for i, c in enumerate(df.columns)}
    has_dpre = 'delta_pitcher_run_exp' in cols
    out = []
    for r in df.itertuples(index=False):
        pfx_x, pfx_z = sf(r.pfx_x), sf(r.pfx_z)
        rpx, rpz = sf(r.release_pos_x), sf(r.release_pos_z)
        typ = getattr(r, 'type', None)
        desc = r.description
        is_bip = (typ == 'X') or (desc in INPLAY_DESC)
        if has_dpre and sf(r.delta_pitcher_run_exp) is not None:
            runexp = sf(r.delta_pitcher_run_exp)
        else:
            dre = sf(r.delta_run_exp)
            runexp = -dre if dre is not None else None
        out.append({
            'Pitcher': r.player_name, 'Throws': r.p_throws, 'Bats': r.stand,
            'Pitch Type': r.pitch_type, 'Game Date': r.game_date,
            'Velocity': sf(r.release_speed), 'Spin Rate': sf(r.release_spin_rate),
            'IndVertBrk': round(pfx_z * 12, 1) if pfx_z is not None else None,
            'HorzBrk': round(-pfx_x * 12, 1) if pfx_x is not None else None,
            # raw defaults; add_weather overwrites with density-adjusted (e=1.05)
            'xIndVrtBrk': round(pfx_z * 12, 1) if pfx_z is not None else None,
            'xHorzBrk': round(-pfx_x * 12, 1) if pfx_x is not None else None,
            'Extension': sf(r.release_extension), 'ArmAngle': sf(r.arm_angle),
            'RelPosX': round(RX_A * rpx + RX_B, 3) if rpx is not None else None,
            'RelPosZ': round(RZ_A * rpz + RZ_B, 3) if rpz is not None else None,
            'PlateX': sf(r.plate_x), 'PlateZ': sf(r.plate_z),
            'VAA': B.vaa_of(sf(r.vy0), sf(r.vz0), sf(r.ay), sf(r.az)),
            'Description': 'In Play' if is_bip else desc,
            'RunExp': runexp,
            'xwOBA': sf(r.estimated_woba_using_speedangle) if is_bip else None,
            '_game_pk': int(r.game_pk) if sf(r.game_pk) is not None else None,
            '_source': f'MLB{year}',
        })
    return out


def sanity():
    yr = 2024
    pitches = build_season_dicts(yr)
    print(f"{yr}: {len(pitches)} dicts")
    lg, sc = GUTS[yr]
    _lg, _sc = T.LG_WOBA, T.WOBA_SCALE
    T.LG_WOBA, T.WOBA_SCALE = lg, sc
    df = T.build_df(pitches)
    T.LG_WOBA, T.WOBA_SCALE = _lg, _sc
    # compare to 2025 retagged build for consistency
    p25 = pickle.load(open(os.path.join(ROOT, 'data', '_pitches2025_training.pkl'), 'rb'))
    T.LG_WOBA, T.WOBA_SCALE = T.PRIOR_LG_WOBA, T.PRIOR_WOBA_SCALE
    d25 = T.build_df(p25)
    T.LG_WOBA, T.WOBA_SCALE = _lg, _sc
    print(f"  build_df rows: {yr}={len(df)}  2025={len(d25)}")
    print(f"  target coverage: {df.target_xrv.notna().mean():.1%}\n")
    print(f"  {'feature':12s} {'2024 mean/sd':>18s} {'2025 mean/sd':>18s}")
    for c in ['velocity', 'ivb', 'hb', 'velo_diff', 'ivb_diff', 'hb_diff', 'spin_rate',
              'extension', 'arm_angle', 'rel_z', 'rel_x', 'vaa', 'vaa_diff', 'target_xrv']:
        a, b = df[c].dropna(), d25[c].dropna()
        flag = ''
        if abs(a.mean() - b.mean()) > 0.5 * (abs(b.std()) + 1e-9) and c != 'target_xrv':
            flag = '  <-- CHECK'
        print(f"  {c:12s} {a.mean():8.2f}/{a.std():6.2f}   {b.mean():8.2f}/{b.std():6.2f}{flag}")
    # hb sign sanity: RHP sinkers should have + normalized hb (arm-side), like 2025
    for lbl, d in [('2024', df), ('2025', d25)]:
        si = d[(d.pitch_type == 'SI') & (d.throws == 'R')]
        print(f"  {lbl} RHP SI normalized hb mean = {si.hb.mean():+.2f} (n={len(si)})  [expect strongly +]")


def full():
    allp = []
    for yr in (2021, 2022, 2023, 2024):
        pitches = build_season_dicts(yr)
        pitches = [p for p in pitches if p.get('_game_pk')]
        print(f"{yr}: {len(pitches)} pitches, fetching weather ...", flush=True)
        pitches = B.add_weather(pitches)
        out = os.path.join(ROOT, 'data', f'_pitches{yr}_training.pkl')
        pickle.dump(pitches, open(out, 'wb'))
        nbip = sum(1 for p in pitches if p['Description'] == 'In Play')
        nxw = sum(1 for p in pitches if p['Description'] == 'In Play' and p.get('xwOBA') is not None)
        nre = sum(1 for p in pitches if p.get('RunExp') is not None)
        print(f"  saved -> {out}  VAA {sum(1 for p in pitches if p.get('VAA') is not None)/len(pitches):.1%}"
              f"  RunExp {nre/len(pitches):.1%}  xwOBA/BIP {nxw/max(nbip,1):.1%}", flush=True)


if __name__ == '__main__':
    if '--sanity' in sys.argv:
        sanity()
    else:
        full()
