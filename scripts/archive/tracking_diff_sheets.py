"""tracking_diff_sheets.py — tracking-field discrepancies: CURRENT sheets vs
freshly-pulled Savant, with auto-ball-aware pitch numbering. READ-ONLY.

Supersedes the earlier data_diff_savant.py run, which (a) read the stale pickle
and (b) matched on raw pitch_number, so the auto-ball offset made it compare the
wrong pitches in ~230 PAs. This reads the live sheets (corrected PitchIDs) and
renumbers Savant to drop automatic events, so every comparison is like-for-like.

Compares Velocity, Spin Rate, ExitVelo, LaunchAngle, Distance, Extension.

Usage: python3 scripts/tracking_diff_sheets.py
"""
import os, sys, time, warnings, socket
warnings.filterwarnings('ignore')
socket.setdefaulttimeout(90)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import gspread
import backfill_supplement as B
from pybaseball import statcast
from collections import defaultdict

FIELDS = [  # (sheet col, savant attr, buckets)
    ('Velocity',    'release_speed',     [0.2, 0.5, 1.0]),
    ('Spin Rate',   'release_spin_rate', [25, 75, 150]),
    ('ExitVelo',    'launch_speed',      [0.5, 2.0, 5.0]),
    ('LaunchAngle', 'launch_angle',      [1.5, 3.0, 6.0]),
    ('Distance',    'hit_distance_sc',   [5, 15, 40]),
    ('Extension',   'release_extension', [0.1, 0.25, 0.5]),
]


def sf(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def main():
    print("pulling current Savant ...", flush=True)
    df = statcast(start_dt='2026-03-15', end_dt='2026-07-06', verbose=False)
    bypa = defaultdict(list)
    for r in df[['game_pk', 'at_bat_number', 'pitch_number', 'description',
                 'release_speed', 'release_spin_rate', 'launch_speed', 'launch_angle',
                 'hit_distance_sc', 'release_extension']].itertuples(index=False):
        try:
            bypa[(int(r.game_pk), int(r.at_bat_number))].append((int(r.pitch_number), r))
        except Exception:
            continue
    aligned = {}
    for (pk, ab), evs in bypa.items():
        evs.sort(key=lambda t: t[0])
        feed = 0
        for pn, r in evs:
            if 'automatic' in str(r.description or '').lower():
                continue
            feed += 1
            aligned[(pk, ab, feed)] = r
    print(f"aligned savant pitches: {len(aligned)}", flush=True)

    gc = gspread.service_account()
    stats = {f[0]: {'n': 0, 'b': [0, 0, 0], 'ex': []} for f in FIELDS}
    matched = 0
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for ws in sh.worksheets():
            if ws.title.upper() not in B.ALL_TRACKED_TEAMS or ws.title.upper() in ('ROC', 'AAA', 'FCL'):
                continue  # MLB tabs only (Savant has no per-pitch minor-league feed here)
            time.sleep(0.7)
            vals = ws.get_all_values()
            if not vals or len(vals) < 2 or 'PitchID' not in vals[0]:
                continue
            ci = {n: j for j, n in enumerate(vals[0]) if n}
            for r in vals[1:]:
                pid = r[ci['PitchID']] if ci['PitchID'] < len(r) else ''
                parts = pid.split('_')
                if len(parts) != 3:
                    continue
                try:
                    key = (int(parts[0]), int(parts[1]), int(parts[2]))
                except ValueError:
                    continue
                sr = aligned.get(key)
                if sr is None:
                    continue
                matched += 1
                for col, attr, buckets in FIELDS:
                    pv = sf(r[ci[col]]) if col in ci and ci[col] < len(r) else None
                    sv = sf(getattr(sr, attr, None))
                    if pv is None or sv is None:
                        continue
                    stats[col]['n'] += 1
                    d = abs(pv - sv)
                    if d > buckets[0]:
                        for bi, t in enumerate(buckets):
                            if d > t:
                                stats[col]['b'][bi] += 1
                        if len(stats[col]['ex']) < 10:
                            stats[col]['ex'].append(f"{pid}: sheet={pv} savant={sv}")
            print(f"  [{label}/{ws.title}] done", flush=True)

    print(f"\nmatched pitches: {matched}\n")
    print(f"{'field':12s} {'compared':>9s}  {'thresholds (count > x)'}")
    for col, attr, buckets in FIELDS:
        s = stats[col]
        print(f"{col:12s} {s['n']:9d}  >{buckets[0]}: {s['b'][0]:5d}   >{buckets[1]}: {s['b'][1]:4d}   >{buckets[2]}: {s['b'][2]:4d}")
    for col, attr, buckets in FIELDS:
        print(f"\n{col} largest examples:")
        for e in sorted(stats[col]['ex'], key=lambda s: -abs(sf(s.split('savant=')[1]) - sf(s.split('sheet=')[1].split(' ')[0])))[:6]:
            print(f"   {e}")
    print("\nDONE", flush=True)


if __name__ == '__main__':
    main()
