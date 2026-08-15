"""drift_check_savant.py — do the Savant-sourced columns change between the
stored (scrape-time) value and what Savant serves NOW? READ-ONLY.

Statcast reprocesses tracking after games (we already saw it for PlateZ and for
SwingLength at Progressive Field). This measures, for a spread of dates across
the season, how often each Savant-sourced column has drifted, so we can decide
whether re-pulling any of them is worth it.

Columns: xBA, xSLG, xwOBA, RunExp, Barrel, ArmAngle (+ their Savant source).
"""
import os, sys, time, pickle, warnings
from collections import defaultdict
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
import gspread
import scrapers.backfill_supplement as B
from pybaseball import statcast

# sheet col -> (savant col, tolerance, decimals-for-display)
FIELDS = {
    'xBA':      ('estimated_ba_using_speedangle', 0.005),
    'xSLG':     ('estimated_slg_using_speedangle', 0.005),
    'xwOBA':    ('estimated_woba_using_speedangle', 0.005),
    'RunExp':   ('delta_pitcher_run_exp', 0.005),
    'Barrel':   ('launch_speed_angle', 0.5),
    'ArmAngle': ('arm_angle', 0.15),
}


def sf(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def main():
    meta = pickle.load(open(os.path.join(ROOT, 'scripts', '_feed_meta_cache.pkl'), 'rb'))
    # spread ~14 dates across the season
    dates = sorted({d for (d, _, _) in meta.values() if d})
    pick = [dates[int(i * (len(dates) - 1) / 13)] for i in range(14)]
    pick = sorted(set(pick))
    print("sample dates:", pick, flush=True)
    sample_games = {pk for pk, (d, _, _) in meta.items() if d in pick}
    print(f"sample games: {len(sample_games)}", flush=True)

    # current Savant, keyed by PitchID (auto-ball renumbered to feed numbers)
    cur = {}
    for d in pick:
        try:
            df = statcast(start_dt=d, end_dt=d, verbose=False)
        except Exception as e:
            print(f"  {d}: pull failed {e!r}"); continue
        bypa = defaultdict(list)
        cols = ['game_pk', 'at_bat_number', 'pitch_number', 'description'] + [v[0] for v in FIELDS.values()]
        cols = [c for c in cols if c in df.columns]
        for r in df[cols].itertuples(index=False):
            try:
                bypa[(int(r.game_pk), int(r.at_bat_number))].append(r)
            except Exception:
                continue
        for (pk, ab), evs in bypa.items():
            evs.sort(key=lambda r: int(r.pitch_number))
            fn = 0
            for r in evs:
                if 'automatic' in str(getattr(r, 'description', '') or '').lower():
                    continue
                fn += 1
                cur[f"{pk}_{ab:03d}_{fn:02d}"] = r
        print(f"  {d}: {len(df)} pitches", flush=True)
    print(f"current Savant pitches keyed: {len(cur)}", flush=True)

    stats = {c: {'n': 0, 'chg': 0, 'absum': 0.0, 'ex': []} for c in FIELDS}
    gc = gspread.service_account()
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for ws in sh.worksheets():
            t = ws.title.upper()
            if t not in B.ALL_TRACKED_TEAMS or t in ('ROC', 'AAA', 'FCL'):
                continue
            time.sleep(0.4)
            vals = ws.get_all_values()
            if not vals or 'PitchID' not in vals[0]:
                continue
            ci = {n: j for j, n in enumerate(vals[0]) if n}
            pc = ci['PitchID']
            for r in vals[1:]:
                pid = r[pc] if pc < len(r) else ''
                g = pid.split('_')[0]
                if not g.isdigit() or int(g) not in sample_games:
                    continue
                sr = cur.get(pid)
                if sr is None:
                    continue
                for col, (scol, tol) in FIELDS.items():
                    if col not in ci:
                        continue
                    stored = sf(r[ci[col]]) if ci[col] < len(r) else None
                    now = sf(getattr(sr, scol, None))
                    if stored is None or now is None:
                        continue
                    s = stats[col]
                    s['n'] += 1
                    d = abs(stored - now)
                    if d > tol:
                        s['chg'] += 1; s['absum'] += d
                        if len(s['ex']) < 6:
                            s['ex'].append(f"{pid}: {stored}->{now}")
    print(f"\n{'column':10s} {'compared':>9s} {'changed':>8s} {'%chg':>6s} {'mean|chg|(when changed)'}")
    for col in FIELDS:
        s = stats[col]
        pct = 100 * s['chg'] / s['n'] if s['n'] else 0
        mean = s['absum'] / s['chg'] if s['chg'] else 0
        print(f"{col:10s} {s['n']:9d} {s['chg']:8d} {pct:5.1f}% {mean:.4f}")
    for col in FIELDS:
        if stats[col]['ex']:
            print(f"\n{col} examples: {stats[col]['ex']}")


if __name__ == '__main__':
    main()
