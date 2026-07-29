"""data_diff_savant.py — diff your stored tracking columns against current Savant.

Read-only. Focuses on the feed-sourced columns that backfill_supplement never
re-pulls, so a post-scrape Statcast revision leaves them stale: batted-ball
tracking (ExitVelo, LaunchAngle, Distance) and pitch velocity, plus xwOBA (which
backfill fills-if-empty but never overwrites). Matches by (game_pk, at_bat,
pitch_number) parsed from PitchID. Reports per-column mismatch counts + examples.

Usage: python3 scripts/data_diff_savant.py
"""
import os, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sf(x):
    try:
        v = float(x)
        return v if v == v else None   # drop NaN
    except (TypeError, ValueError):
        return None


def main():
    allp = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    sc = pickle.load(open(os.path.join(ROOT, 'data', '_statcast2026_diff.pkl'), 'rb'))
    look = {}
    for r in sc.itertuples(index=False):
        try:
            look[(int(r.game_pk), int(r.at_bat_number), int(r.pitch_number))] = r
        except Exception:
            continue
    print(f"pickle pitches: {len(allp)}   savant rows: {len(look)}\n")

    # (label, pickle_col, savant_attr, tol, bip_only)
    CHECKS = [
        ('Velocity',   'Velocity',   'release_speed', 0.2, False),
        ('ExitVelo',   'ExitVelo',   'launch_speed',  0.5, True),
        ('LaunchAngle', 'LaunchAngle', 'launch_angle', 1.5, True),
        ('Distance',   'Distance',   'hit_distance_sc', 5.0, True),
        ('xwOBA',      'xwOBA',      'estimated_woba_using_speedangle', 0.03, True),
        ('SpinRate',   'Spin Rate',  'release_spin_rate', 25, False),
        ('Extension',  'Extension',  'release_extension', 0.15, False),
    ]
    stats = {c[0]: {'n': 0, 'mismatch': 0, 'ex': []} for c in CHECKS}
    matched = 0
    fixed_pas = {'824448_043', '824527_053', '824525_045', '824280_051', '824600_048',
                 '823056_069', '823545_025', '823700_063', '822725_065', '822968_070'}
    for p in allp:
        pid = p.get('PitchID', '')
        parts = pid.split('_')
        if len(parts) != 3 or p.get('_source') != 'MLB':
            continue
        if '_'.join(parts[:2]) in fixed_pas:
            continue   # pickle stale on the 10 we just fixed in the sheet
        try:
            k = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            continue
        sr = look.get(k)
        if sr is None:
            continue
        matched += 1
        for label, pcol, sattr, tol, bip in CHECKS:
            pv = sf(p.get(pcol)); sv = sf(getattr(sr, sattr, None))
            if pv is None or sv is None:
                continue
            stats[label]['n'] += 1
            if abs(pv - sv) > tol:
                stats[label]['mismatch'] += 1
                if len(stats[label]['ex']) < 6:
                    stats[label]['ex'].append(f"{pid}: yours={pv} savant={sv}")

    print(f"matched pitches: {matched}\n")
    print(f"{'column':12s} {'compared':>9s} {'mismatch':>9s} {'rate':>7s}")
    for label, _, _, _, _ in CHECKS:
        s = stats[label]
        rate = (s['mismatch'] / s['n'] * 100) if s['n'] else 0
        print(f"{label:12s} {s['n']:9d} {s['mismatch']:9d} {rate:6.2f}%")
    print()
    for label, _, _, _, _ in CHECKS:
        if stats[label]['ex']:
            print(f"  {label} examples:")
            for e in stats[label]['ex']:
                print(f"    {e}")


if __name__ == '__main__':
    main()
