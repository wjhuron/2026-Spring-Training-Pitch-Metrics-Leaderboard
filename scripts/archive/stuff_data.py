"""stuff_data.py — engineer the Stuff+ training dataset from the pitch pickle.

Builds one row per MLB pitch with: physical stuff features (velo, movement,
spin, release, arm angle), weather-adjustment deltas (raw minus density-
adjusted movement — NOT expected-movement residuals; axis_dev is the SSW
proxy), fastball-relative diffs, several candidate TARGETS, outcome flags,
and split/half tags for the validation harness. Caches to scratch for fast
experiment iteration.

All movement/release/HAA features are handedness-normalized so "arm side" is a
consistent sign for L and R pitchers.
"""
import pickle, math, os, collections
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
# Override with STUFF_DF_OUT; default keeps the artifact out of the repo.
OUT = os.environ.get('STUFF_DF_OUT',
                     os.path.join(ROOT, 'data', '_stuff_df_scratch.pkl'))
LG_WOBA, WOBA_SCALE = 0.3169, 1.2393

SUPPORTED = {'FF', 'SI', 'SL', 'CH', 'ST', 'FC', 'CU', 'FS', 'SV', 'KC'}
FB_TYPES = {'FF', 'SI', 'FC'}   # candidates for the "primary fastball" reference

def sf(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def get_count(c):
    if not isinstance(c, str) or '-' not in c: return (None, None)
    try:
        b, s = c.split('-', 1); return int(b), int(s)
    except (TypeError, ValueError): return (None, None)

def tilt_deg(s):
    """Clock-face tilt 'HH:MM' -> degrees (12:00=0, 3:00=90, ...)."""
    if not isinstance(s, str) or ':' not in s: return None
    try:
        h, m = s.split(':'); return (int(h) % 12) * 30 + int(m) * 0.5
    except (TypeError, ValueError): return None

print("loading ...", flush=True)
D = pickle.load(open(PKL, 'rb'))
mlb = [p for p in D if p.get('_source') == 'MLB']
print(f"  {len(mlb)} MLB pitches", flush=True)

# ── pass 1: per-pitcher primary fastball reference (handedness-normalized) ──
fb_acc = collections.defaultdict(lambda: collections.defaultdict(
    lambda: {'v': 0.0, 'iv': 0.0, 'hb': 0.0, 'vaa': 0.0, 'n': 0}))
for p in mlb:
    pt = p.get('Pitch Type'); thr = p.get('Throws')
    if pt not in FB_TYPES or thr not in ('L', 'R'): continue
    v = sf(p.get('Velocity')); iv = sf(p.get('IndVertBrk')); hb = sf(p.get('HorzBrk'))
    vaa = sf(p.get('VAA'))
    if None in (v, iv, hb): continue
    sgn = 1.0 if thr == 'R' else -1.0
    a = fb_acc[(p.get('Pitcher'), thr)][pt]
    a['v'] += v; a['iv'] += iv; a['hb'] += hb * sgn
    a['vaa'] += (vaa if vaa is not None else 0.0); a['n'] += 1

primary_fb = {}
for key, bytype in fb_acc.items():
    best = max(bytype.values(), key=lambda d: d['n'])  # most-thrown FB
    n = best['n']
    primary_fb[key] = {'v': best['v']/n, 'iv': best['iv']/n,
                       'hb': best['hb']/n, 'vaa': best['vaa']/n}

mean_ext = np.mean([sf(p.get('Extension')) for p in mlb
                    if sf(p.get('Extension')) is not None])

# ── pass 2: build rows ──
rows = []
for p in mlb:
    pt = p.get('Pitch Type'); thr = p.get('Throws'); bats = p.get('Bats')
    if pt not in SUPPORTED or thr not in ('L', 'R') or bats not in ('L', 'R'):
        continue
    v = sf(p.get('Velocity')); spin = sf(p.get('Spin Rate'))
    iv = sf(p.get('IndVertBrk')); hb_raw = sf(p.get('HorzBrk'))
    xiv = sf(p.get('xIndVrtBrk')); xhb = sf(p.get('xHorzBrk'))
    vaa = sf(p.get('VAA')); haa_raw = sf(p.get('HAA'))
    ext = sf(p.get('Extension')); arm = sf(p.get('ArmAngle'))
    rel_z = sf(p.get('RelPosZ')); rel_x_raw = sf(p.get('RelPosX'))
    if None in (v, iv, hb_raw, vaa, ext, rel_z, rel_x_raw):
        continue
    sgn = 1.0 if thr == 'R' else -1.0
    hb = hb_raw * sgn
    rel_x = rel_x_raw * sgn
    haa = (haa_raw * sgn) if haa_raw is not None else None
    # NOT SSW residuals: xIndVrtBrk/xHorzBrk in the pickle are the WEATHER-
    # ADJUSTED movement (raw × density factor), not MVN-expected movement.
    # These deltas are therefore just the density correction itself (≈0
    # outside altitude parks). Kept under honest names for density-related
    # experiments; any past experiment that used them as "SSW" tested noise.
    ivb_wx_delta = (iv - xiv) if (xiv is not None) else None
    hb_wx_delta = ((hb_raw - xhb) * sgn) if (xhb is not None) else None
    # plate location (for location-neutralization tests); handedness-normalized x
    px = sf(p.get('PlateX')); pz = sf(p.get('PlateZ'))
    szt = sf(p.get('SzTop')); szb = sf(p.get('SzBot'))
    plate_x = (px * sgn) if px is not None else None
    plate_z_norm = ((pz - szb) / (szt - szb)) if (pz is not None and szt and szb and szt > szb) else None
    # spin-axis deviation: observed movement tilt vs spin-based tilt (the
    # actual SSW proxy in this dataset)
    ot = tilt_deg(p.get('OTilt')); rt = tilt_deg(p.get('RTilt'))
    if ot is not None and rt is not None:
        d = ((ot - rt + 180) % 360) - 180   # wrap to [-180, 180]
        axis_dev = d * sgn
    else:
        axis_dev = None
    perceived = v * (60.5 - mean_ext) / (60.5 - ext) if ext < 60.0 else v
    total_mov = math.hypot(iv, hb)
    mov_angle = math.degrees(math.atan2(hb, iv))
    spin_per_mph = spin / v if (spin and v) else None

    fb = primary_fb.get((p.get('Pitcher'), thr))
    if fb:
        velo_diff = v - fb['v']; ivb_diff = iv - fb['iv']
        hb_diff = hb - fb['hb']; vaa_diff = (vaa - fb['vaa'])
    else:
        velo_diff = ivb_diff = hb_diff = vaa_diff = None

    desc = p.get('Description', '')
    is_swing = desc in ('Swinging Strike', 'Foul', 'In Play')
    is_whiff = desc == 'Swinging Strike'
    is_bip = desc == 'In Play'
    bb = (p.get('BBType') or '').lower()
    is_gb = 1 if bb in ('ground_ball', 'groundball', 'gb') else 0
    xw = sf(p.get('xwOBA'))
    re = sf(p.get('RunExp'))
    b, s = get_count(p.get('Count'))

    # luck-neutral hitter-perspective xRV target (lower = better stuff)
    if is_bip and xw is not None:
        target_xrv = (xw - LG_WOBA) / WOBA_SCALE
    elif re is not None:
        target_xrv = -re
    else:
        target_xrv = None

    rows.append({
        'pitcher': p.get('Pitcher'), 'team': p.get('PTeam'), 'throws': thr,
        'stands': bats, 'pitch_type': pt, 'date': p.get('Game Date'),
        'platoon_same': 1 if bats == thr else 0,
        'balls': b, 'strikes': s,
        # features
        'velocity': v, 'perceived_velo': perceived, 'spin_rate': spin,
        'ivb': iv, 'hb': hb, 'ivb_wx_delta': ivb_wx_delta, 'hb_wx_delta': hb_wx_delta,
        'vaa': vaa, 'haa': haa, 'extension': ext, 'arm_angle': arm,
        'rel_z': rel_z, 'rel_x': rel_x,
        'plate_x': plate_x, 'plate_z_norm': plate_z_norm, 'axis_dev': axis_dev,
        'velo_diff': velo_diff, 'ivb_diff': ivb_diff, 'hb_diff': hb_diff, 'vaa_diff': vaa_diff,
        'total_mov': total_mov, 'mov_angle': mov_angle, 'spin_per_mph': spin_per_mph,
        # outcomes / targets
        'is_swing': int(is_swing), 'is_whiff': int(is_whiff), 'is_bip': int(is_bip),
        'is_gb': is_gb, 'xwoba': xw, 'run_value': re, 'target_xrv': target_xrv,
    })

df = pd.DataFrame(rows)
# half tags for reliability: alternate by calendar date (global), so each
# pitcher's appearances split roughly evenly between the two halves.
date_order = {d: i for i, d in enumerate(sorted(df['date'].dropna().unique()))}
df['half'] = df['date'].map(date_order).fillna(0).astype(int) % 2
df['period'] = np.where(df['date'] < '2026-05-01', 'early', 'late')

df.to_pickle(OUT)
print(f"\nsaved {len(df)} rows -> {OUT}")
print("by pitch type:\n", df['pitch_type'].value_counts().to_string())
print("\ntarget_xrv: mean %.4f std %.4f  (non-null %d)" % (
    df['target_xrv'].mean(), df['target_xrv'].std(), df['target_xrv'].notna().sum()))
print("feature null rates (key):")
for c in ['ivb_wx_delta', 'hb_wx_delta', 'arm_angle', 'velo_diff', 'spin_per_mph', 'haa']:
    print(f"  {c:12s} {df[c].isna().mean():.1%}")
print("whiff/swing rate: %.1f%%   bip rate: %.1f%%   gb/bip: %.1f%%" % (
    100*df['is_whiff'].sum()/df['is_swing'].sum(),
    100*df['is_bip'].mean(),
    100*df.loc[df['is_bip']==1,'is_gb'].mean()))
