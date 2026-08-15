"""era_shrinkage_sweep.py — measure the shrinkage constants for the
dhERA/phERA input channels, per season, both split directions.

Model: rate_shrunk = (num + n0 * m) / (den + n0), m = the shrink target.
Objective: predict the OTHER half's raw rate, weighted by that half's
denominator (its precision). Reported per season (replicates), argmax
must be interior and agree across seasons, per the tuning standard.

Channels and denominators:
  xwOBA      xw_num / xw_den   (PA-ish)   <- the dhERA channel
  K%         k / pa
  BB%        bb / pa
  xwOBAcon   xwc_sum / xwc_n   (BIP)
  Stuff+/Loc+/Pitching+ (2026 only; the sheet carries per-pitch values,
  prior seasons' internal scores were not persisted per-pitch) — compared
  against the Pitcher+ shipped ks (stuff 42, loc 215 pitches).

Shrink-target variants for xwOBA: league mean vs role mean (starter =
gs/g >= 0.5 from the official line). If role wins consistently, the
display shrink target is role-specific.

Usage: python3 scripts/era_shrinkage_sweep.py
Output: console + data/_era_shrinkage.json
"""
import json
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = json.load(open(os.path.join(ROOT, 'data', '_era_parity_halves.json')))
TARGETS = json.load(open(os.path.join(ROOT, 'data', '_era_targets.json')))

GRID = [0, 5, 10, 15, 20, 30, 40, 55, 70, 90, 110, 140, 170, 200, 250,
        300, 400, 500, 700, 1000, 1500]
MIN_DEN = 20     # both halves need at least this much denominator


def is_starter(season, pid):
    rec = TARGETS[str(season)]['pitchers'].get(pid)
    if not rec or not rec.get('g'):
        return False
    return rec['gs'] / rec['g'] >= 0.5


def channel_units(season, num_k, den_k):
    """[(pid, num_a, den_a, num_b, den_b)] with both halves present."""
    out = []
    for pid, halves in H[str(season)].items():
        a, b = halves.get('a', {}), halves.get('b', {})
        if a.get(den_k, 0) >= MIN_DEN and b.get(den_k, 0) >= MIN_DEN:
            out.append((pid, a.get(num_k, 0.0), a[den_k],
                        b.get(num_k, 0.0), b[den_k]))
    return out


def sweep_channel(season, num_k, den_k, role_target=False):
    units = channel_units(season, num_k, den_k)
    if len(units) < 60:
        return None
    # shrink targets from THIS season's pool (den-weighted)
    def mean_of(pred):
        n = sum(u[1] + u[3] for u in units if pred(u[0]))
        d = sum(u[2] + u[4] for u in units if pred(u[0]))
        return n / d if d else 0.0
    lg = mean_of(lambda pid: True)
    m_sp = mean_of(lambda pid: is_starter(season, pid))
    m_rp = mean_of(lambda pid: not is_starter(season, pid))

    def target(pid):
        if not role_target:
            return lg
        return m_sp if is_starter(season, pid) else m_rp

    curve = {}
    for n0 in GRID:
        sse = wsum = 0.0
        for pid, na, da, nb, db in units:
            m = target(pid)
            # both directions
            for (n1, d1, n2, d2) in ((na, da, nb, db), (nb, db, na, da)):
                pred = (n1 + n0 * m) / (d1 + n0)
                act = n2 / d2
                sse += d2 * (pred - act) ** 2
                wsum += d2
        curve[n0] = math.sqrt(sse / wsum)
    best = min(curve, key=curve.get)
    return {'curve': curve, 'best': best, 'n_units': len(units),
            'lg': lg, 'm_sp': m_sp, 'm_rp': m_rp}


def run(num_k, den_k, label, seasons, role_target=False):
    print(f'\n=== {label}'
          + (' (role-specific target)' if role_target else '') + ' ===')
    bests = []
    out = {}
    for season in seasons:
        r = sweep_channel(season, num_k, den_k, role_target)
        if r is None:
            continue
        gi = GRID.index(r['best'])
        interior = 0 < gi < len(GRID) - 1
        lo = GRID[max(gi - 1, 0)]
        hi = GRID[min(gi + 1, len(GRID) - 1)]
        print(f"  {season}: n0* = {r['best']:>4} "
              f"(rmse {r['curve'][r['best']]:.4f}; neighbors "
              f"{lo}:{r['curve'][lo]:.4f} {hi}:{r['curve'][hi]:.4f}) "
              f"n={r['n_units']}"
              + ('' if interior else '  EDGE'))
        bests.append(r['best'])
        out[season] = r
    if bests:
        print(f'  argmax across replicates: {sorted(bests)}')
    return out


def main():
    res = {}
    seasons = [s for s in ('2021', '2022', '2023', '2024', '2025', '2026')
               if s in H]
    res['xwoba_lg'] = run('xw_num', 'xw_den', 'xwOBA (PA), league target',
                          seasons)
    res['xwoba_role'] = run('xw_num', 'xw_den', 'xwOBA (PA)', seasons,
                            role_target=True)
    res['k_pct'] = run('k', 'pa', 'K% (PA), league target', seasons)
    res['bb_pct'] = run('bb', 'pa', 'BB% (PA), league target', seasons)
    res['xwobacon'] = run('xwc_sum', 'xwc_n', 'xwOBAcon (BIP), league',
                          seasons)
    res['stuff26'] = run('st_sum', 'st_n', 'Stuff+ (pitches, 2026 only)',
                         ['2026'])
    res['loc26'] = run('lo_sum', 'lo_n', 'Loc+ (pitches, 2026 only)',
                       ['2026'])
    res['pplus26'] = run('pp_sum', 'pp_n', 'Pitching+ (pitches, 2026)',
                         ['2026'])
    ser = {k: ({str(s): {'best': v[s]['best'],
                         'curve': {str(n): c for n, c in
                                   v[s]['curve'].items()}}
                for s in v} if v else {}) for k, v in res.items()}
    with open(os.path.join(ROOT, 'data', '_era_shrinkage.json'), 'w') as f:
        json.dump(ser, f)
    print('\nwrote data/_era_shrinkage.json')


if __name__ == '__main__':
    main()
