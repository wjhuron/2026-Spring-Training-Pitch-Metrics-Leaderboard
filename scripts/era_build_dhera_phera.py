"""era_build_dhera_phera.py — final builds of dhERA and phERA.

dhERA (descriptive, deserved): how well has he pitched this season, with
defense, sequencing, and ballpark luck stripped. Anchored on xwOBA
against; the design search below settles whether a second deserved
channel (FIP / xFIP / K-BB / xwOBAcon / HR9) earns a place, and the
FIP-vs-xFIP decomposition prices how much of any FIP gain is HR/FB luck
(which "deserved" should exclude by philosophy).

phERA (predictive, going forward): one unified formula for both horizons,
xwOBA skeleton + Pitching+ channel. The weight is settled by sweeping
BOTH the next-season and rest-of-season replicate sets and requiring an
overlapping plateau; the cost vs the horizon-specialized variants from
phase 2 is documented, not hidden.

Both metrics display on ERA scale:
  metric = lgERA(season) + b * comp
with b frozen from 2021-2025 replicates and validated on held-out data.

Sections:
  1. dhERA second-channel search (DESC replicates, fixed-w sweeps)
  2. HR-luck decomposition of the FIP gain
  3. dhERA calibration (per-season slope stability, 2026 holdout)
  4. phERA unified weight sweep on NEXT and ROS
  5. phERA calibration (ROS slope = in-season display; NEXT slope noted)
  6. 2026 values -> ~/Downloads/dhERA_phERA_2026.csv

Usage: python3 scripts/era_build_dhera_phera.py
Output: console, data/_era_dhera_phera_build.json, CSV deliverable.
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from era_estimator_screen import (feature_rows, targets_for, pearson,
                                  SEASONS)
from era_combo_preview import zscore_within, CANDS

GATE = 60
TARGETS = json.load(open(os.path.join(ROOT, 'data', '_era_targets.json')))


FEATS = sorted(set(CANDS) | {'hr9_off'})
POOL_OUTS = 90          # 30 IP: the production z-pool definition
POOL_OUTS_HALF = 45     # 15 IP per half for the ROS scope


def pooled_z(season, scope, min_outs):
    """PRODUCTION CONVENTION: z-scores computed over the pool of pitchers
    with >= min_outs in that scope. Returns {pid: {feat: z}} for pool
    members only."""
    targ = targets_for(season, scope if scope != 'full' else 'full')
    rows = {pid: r for pid, r in feature_rows(
                season, 'full' if scope == 'full' else 'h1').items()
            if targ.get(pid, {}).get('outs', 0) >= min_outs}
    return zscore_within(rows, FEATS), targ


def desc_reps(gate=GATE):
    """Units z-scored in the production pool; unit list gated at `gate`
    for evaluation (gate >= pool floor)."""
    reps = []
    for season in SEASONS:
        fr, targ = pooled_z(season, 'full', POOL_OUTS)
        units = [(fr[pid], targ[pid]['era']) for pid in fr
                 if pid in targ and targ[pid]['outs'] >= gate * 3]
        reps.append((str(season), units))
    return reps


def next_reps(gate=GATE):
    reps = []
    for season in SEASONS[:-1]:
        fr, tc = pooled_z(season, 'full', POOL_OUTS)
        tn = targets_for(season + 1, 'full')
        units = [(fr[pid], tn[pid]['era']) for pid in fr
                 if pid in tn and tc[pid]['outs'] >= gate * 3
                 and tn[pid]['outs'] >= gate * 3]
        reps.append((f'{season}->{season + 1}', units))
    return reps


def ros_reps(gate=GATE):
    reps = []
    for season in SEASONS:
        fr, t1 = pooled_z(season, 'h1', POOL_OUTS_HALF)
        t2 = targets_for(season, 'h2')
        hg = max(gate * 3 // 2, POOL_OUTS_HALF)
        units = [(fr[pid], t2[pid]['era']) for pid in fr
                 if pid in t2 and t1[pid]['outs'] >= hg
                 and t2[pid]['outs'] >= hg]
        reps.append((f'{season}h', units))
    return reps


def sweep(reps, f1, f2, sign2=+1, steps=21):
    """comp = (1-w)*z(f1) + sign2*w*z(f2). Returns {w: (mean, per)}."""
    out = {}
    for wi in range(steps):
        w = wi / (steps - 1)
        per = []
        for label, units in reps:
            xs, ys = [], []
            for x, y in units:
                if f1 in x and f2 in x:
                    xs.append((1 - w) * x[f1] + sign2 * w * x[f2])
                    ys.append(y)
            r = pearson(xs, ys)
            if r is not None:
                per.append((label, r))
        if not per:
            return None
        out[round(w, 2)] = (sum(r for _, r in per) / len(per), per)
    return out


def show_sweep(name, sw, base_label):
    best_w = max(sw, key=lambda w: abs(sw[w][0]))
    print(f'  {name}: base(w=0) {sw[0.0][0]:+.4f} -> best w={best_w} '
          f'mean {sw[best_w][0]:+.4f}')
    wins = sum(1 for (l1, r1), (l0, r0) in
               zip(sw[best_w][1], sw[0.0][1]) if abs(r1) > abs(r0))
    print(f'    wins vs {base_label} alone: {wins}/{len(sw[best_w][1])}  '
          f'per-rep at best w: '
          + ' '.join(f'{r:+.3f}' for _, r in sw[best_w][1]))
    return best_w


def comp_value(x, f1, f2, w, sign2):
    return (1 - w) * x[f1] + sign2 * w * x[f2]


def calibrate(reps, f1, f2, w, sign2):
    """Held-one-out slope stability + errors. Fit a,b on other reps."""
    rows = []
    for i, (label, test_units) in enumerate(reps):
        train = [u for j, (_, us) in enumerate(reps) if j != i for u in us]
        xs = [comp_value(x, f1, f2, w, sign2) for x, _ in train
              if f1 in x and f2 in x]
        ys = [y for x, y in train if f1 in x and f2 in x]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((v - mx) ** 2 for v in xs)
        b = sum((v - mx) * (y - my) for v, y in zip(xs, ys)) / sxx
        a = my - b * mx
        px = [comp_value(x, f1, f2, w, sign2) for x, _ in test_units
              if f1 in x and f2 in x]
        py = [y for x, y in test_units if f1 in x and f2 in x]
        preds = [a + b * v for v in px]
        rmse = math.sqrt(sum((p - y) ** 2
                             for p, y in zip(preds, py)) / len(py))
        rows.append({'fold': label, 'a': round(a, 3), 'b': round(b, 3),
                     'rmse': round(rmse, 3)})
        print(f'    fold {label}: a {a:+.3f}  b {b:+.3f}  rmse {rmse:.3f}')
    bs = [r['b'] for r in rows]
    print(f'    slope b: mean {sum(bs) / len(bs):+.3f}  '
          f'range [{min(bs):+.3f}, {max(bs):+.3f}]')
    return rows, sum(bs) / len(bs)


def league_era(season, scope='full'):
    er = outs = 0
    for pid, rec in TARGETS[str(season)]['pitchers'].items():
        src = rec if scope == 'full' else rec[scope]
        er += src['er']
        outs += src['outs']
    return er * 27.0 / outs


def main():
    out = {}
    reps_d = desc_reps()

    # ── 1. dhERA second-channel search ────────────────────────────────
    print('=== 1. dhERA: second channel on top of z(xwOBA), DESC ===')
    sweeps = {}
    for f2, sign2 in [('fip_core', +1), ('xfip_core', +1),
                      ('kbb_off', -1), ('xwobacon', +1), ('hr9_off', +1)]:
        sw = sweep(reps_d, 'xwoba', f2, sign2)
        sweeps[f2] = sw
        show_sweep(f'xwoba + {f2}', sw, 'xwoba')
    out['dhera_sweeps'] = {f: {str(w): v[0] for w, v in sw.items()}
                           for f, sw in sweeps.items()}

    # ── 2. HR-luck decomposition ─────────────────────────────────────
    print('\n=== 2. HR-luck decomposition ===')
    best_fip = max(sweeps['fip_core'], key=lambda w: sweeps['fip_core'][w][0])
    best_xfip = max(sweeps['xfip_core'],
                    key=lambda w: sweeps['xfip_core'][w][0])
    g_fip = sweeps['fip_core'][best_fip][0] - sweeps['fip_core'][0.0][0]
    g_xfip = sweeps['xfip_core'][best_xfip][0] - sweeps['xfip_core'][0.0][0]
    print(f'  gain from +FIP  (HR included): {g_fip:+.4f} at w={best_fip}')
    print(f'  gain from +xFIP (HR neutralized): {g_xfip:+.4f} '
          f'at w={best_xfip}')
    print(f'  -> share of the FIP gain that is the HR channel: '
          f'{1 - g_xfip / g_fip:.0%} (deserved excludes it)')
    out['hr_decomp'] = {'gain_fip': g_fip, 'gain_xfip': g_xfip}

    # dhERA config decision happens here in code so the constants are
    # explicit: keep the xFIP channel only if its best w cleared a
    # majority of replicates AND the gain is not noise (>= 0.01).
    sw_x = sweeps['xfip_core']
    wins_x = sum(1 for (l1, r1), (l0, r0) in
                 zip(sw_x[best_xfip][1], sw_x[0.0][1]) if abs(r1) > abs(r0))
    adopt = wins_x >= (len(sw_x[best_xfip][1]) // 2 + 1) and g_xfip >= 0.01
    DH_W = best_xfip if adopt else 0.0
    print(f'  dhERA config: comp = {1 - DH_W:.2f}*z(xwOBA)'
          + (f' + {DH_W:.2f}*z(xFIP)' if DH_W else ' (xwOBA alone)'))
    out['dhera_config'] = {'w_xfip': DH_W, 'adopted': adopt}

    # ── 3. dhERA calibration ─────────────────────────────────────────
    # Fit at the DISPLAY population (30+ IP), per the tune-at-production-
    # sample-size rule: the 60 IP pool is starters-only and its slope
    # extrapolates nonsense for 4-sigma relievers. Selection/ranking used
    # the 60 gate; the display mapping is fit where it will be used.
    print('\n=== 3. dhERA calibration (30+ IP display population) ===')
    reps_d30 = desc_reps(gate=30)
    dh_rows, dh_b = calibrate(reps_d30, 'xwoba', 'xfip_core', DH_W, +1)
    out['dhera_calib'] = {'rows': dh_rows, 'b': dh_b}

    # ── 4. phERA unified weight sweep ────────────────────────────────
    print('\n=== 4. phERA: (1-w)*z(xwOBA) - w*z(Pitching+) ===')
    reps_next = next_reps()
    reps_ros = ros_reps()
    print('  NEXT horizon:')
    sw_n = sweep(reps_next, 'xwoba', 'pitchingplus_z', -1)
    for w in (0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.7, 1.0):
        print(f'    w={w:.2f}  mean {sw_n[w][0]:+.4f}')
    print('  ROS horizon:')
    sw_r = sweep(reps_ros, 'xwoba', 'pitchingplus_z', -1)
    for w in (0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.7, 1.0):
        print(f'    w={w:.2f}  mean {sw_r[w][0]:+.4f}')
    bn = max(sw_n, key=lambda w: abs(sw_n[w][0]))
    br = max(sw_r, key=lambda w: abs(sw_r[w][0]))
    print(f'  argmax NEXT w={bn}, ROS w={br}')
    out['phera_sweep_next'] = {str(w): v[0] for w, v in sw_n.items()}
    out['phera_sweep_ros'] = {str(w): v[0] for w, v in sw_r.items()}
    # unified w = midpoint of overlapping plateau, chosen in code:
    # plateau = ws within 0.003 of each horizon's max
    pn = {w for w in sw_n if abs(sw_n[w][0]) >= abs(sw_n[bn][0]) - 0.003}
    pr = {w for w in sw_r if abs(sw_r[w][0]) >= abs(sw_r[br][0]) - 0.003}
    overlap = sorted(pn & pr)
    PH_W = overlap[len(overlap) // 2] if overlap else round((bn + br) / 2, 2)
    print(f'  NEXT plateau {sorted(pn)}')
    print(f'  ROS plateau {sorted(pr)}')
    print(f'  unified w = {PH_W} (overlap midpoint)')
    print(f'  cost vs specialized: NEXT {abs(sw_n[bn][0]) - abs(sw_n[PH_W][0]):.4f}, '
          f'ROS {abs(sw_r[br][0]) - abs(sw_r[PH_W][0]):.4f}')
    out['phera_config'] = {'w': PH_W, 'next_plateau': sorted(pn),
                           'ros_plateau': sorted(pr)}

    # ── 5. phERA calibration ─────────────────────────────────────────
    print('\n=== 5. phERA calibration (30+ IP display population) ===')
    reps_ros30 = ros_reps(gate=30)
    reps_next30 = next_reps(gate=30)
    print('  vs rest-of-season ERA (the in-season display slope):')
    ph_rows_r, ph_b_ros = calibrate(reps_ros30, 'xwoba', 'pitchingplus_z',
                                    PH_W, -1)
    print('  vs next-season ERA (offseason interpretation):')
    ph_rows_n, ph_b_next = calibrate(reps_next30, 'xwoba',
                                     'pitchingplus_z', PH_W, -1)
    out['phera_calib'] = {'ros': ph_rows_r, 'b_ros': ph_b_ros,
                          'next': ph_rows_n, 'b_next': ph_b_next}

    # ── 6. 2026 values ───────────────────────────────────────────────
    print('\n=== 6. 2026 values ===')
    feats = ['xwoba', 'xfip_core', 'pitchingplus_z']
    fr_raw = feature_rows(2026, 'full')
    targ = targets_for(2026, 'full')
    # z-pool: the 30+ IP display population, matching the calibration
    pool = {pid: r for pid, r in fr_raw.items()
            if targ.get(pid, {}).get('outs', 0) >= 90}
    stats = {}
    for f in feats:
        v = [r[f] for r in pool.values() if f in r]
        m = sum(v) / len(v)
        s = math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
        stats[f] = (m, s)
    lg = league_era(2026)
    print(f'  2026 league ERA {lg:.2f}; z-pool n={len(pool)}')
    rows_out = []
    for pid, r in fr_raw.items():
        t = targ.get(pid)
        if t is None or t['outs'] < 90:      # 30 IP floor
            continue
        z = {}
        for f in feats:
            if f in r:
                m, s = stats[f]
                z[f] = (r[f] - m) / s
        if 'xwoba' not in z:
            continue
        dh = lg + dh_b * ((1 - DH_W) * z['xwoba']
                          + DH_W * z.get('xfip_core', 0.0))
        ph = None
        if 'pitchingplus_z' in z:
            ph = lg + ph_b_ros * ((1 - PH_W) * z['xwoba']
                                  - PH_W * z['pitchingplus_z'])
        name = TARGETS['2026']['pitchers'][pid]['name']
        era = t['era']
        rows_out.append({'mlbId': pid, 'name': name,
                         'ip': round(t['outs'] / 3.0, 1),
                         'qualified': 1 if t['outs'] >= 180 else 0,
                         'era': round(era, 2),
                         'dhERA': round(dh, 2),
                         'phERA': None if ph is None else round(ph, 2)})
    rows_out.sort(key=lambda r: r['dhERA'])
    csv_path = os.path.expanduser('~/Downloads/dhERA_phERA_2026.csv')
    with open(csv_path, 'w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=['mlbId', 'name', 'ip',
                                            'qualified', 'era', 'dhERA',
                                            'phERA'])
        wtr.writeheader()
        wtr.writerows(rows_out)
    print(f'  wrote {csv_path} ({len(rows_out)} pitchers, 30+ IP)')
    print('  best 12 dhERA (60+ IP):')
    for r in [r for r in rows_out if r['qualified']][:12]:
        print(f"    {r['name']:<24} IP {r['ip']:>6} ERA {r['era']:>5} "
              f"dhERA {r['dhERA']:>5} phERA {r['phERA']}")
    print('  worst 5 dhERA (60+ IP):')
    for r in [r for r in rows_out if r['qualified']][-5:]:
        print(f"    {r['name']:<24} IP {r['ip']:>6} ERA {r['era']:>5} "
              f"dhERA {r['dhERA']:>5} phERA {r['phERA']}")

    with open(os.path.join(ROOT, 'data', '_era_dhera_phera_build.json'),
              'w') as f:
        json.dump(out, f)
    print('\nwrote data/_era_dhera_phera_build.json')


if __name__ == '__main__':
    main()
