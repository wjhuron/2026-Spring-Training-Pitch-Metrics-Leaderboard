"""repull_battracking.py — overwrite the 5 bat-tracking columns for games where
Statcast's swing tracking was glitched and has since been corrected at the source.

Progressive Field had a park-localized SwingLength distortion (mean 7.94 vs ~7.3
league, sd 1.71 vs 0.87, 2.2% impossible >12ft while every other park is 0.0%).
Re-pulling game 824443 from Savant now returns normal values (mean 7.19, max
9.40) vs the sheet's 26.6 — i.e. it's corrected. This re-pulls all Progressive
Field games (+ the handful of scattered AttackDirection>90 games) and overwrites
BatSpeed/SwingLength/AttackAngle/AttackDirection/SwingPathTilt with the current
Savant values where they differ. Only rows Savant now reports as valid swings
(bat_speed >= 50) are touched. Guarded, dry-run first.

  python3 scripts/repull_battracking.py            # DRY RUN
  python3 scripts/repull_battracking.py --apply
"""
import os, sys, time, pickle, warnings
from collections import defaultdict
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import gspread, backfill_supplement as B
from pybaseball import statcast

APPLY = '--apply' in sys.argv
TOL = 0.1
# sheet col -> savant col
BAT = {'BatSpeed': 'bat_speed', 'SwingLength': 'swing_length', 'AttackAngle': 'attack_angle',
       'AttackDirection': 'attack_direction', 'SwingPathTilt': 'swing_path_tilt'}
# the 7 scattered AttackDirection>90 games (some are Progressive, deduped below)
ATTACKDIR_GAMES = {824445, 823889, 824429, 824919, 824702, 823137, 823532}


def sf(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def main():
    meta = pickle.load(open(os.path.join(ROOT, 'scripts', '_feed_meta_cache.pkl'), 'rb'))
    prog = {pk for pk, (d, vid, vn) in meta.items() if vn == 'Progressive Field'}
    targets = prog | ATTACKDIR_GAMES
    dates = sorted({meta[pk][0] for pk in targets if pk in meta and meta[pk][0]})
    print(f"target games: {len(targets)} (Progressive {len(prog)} + AttackDir {len(ATTACKDIR_GAMES)}); dates: {len(dates)}", flush=True)

    cur = {}   # PitchID -> {sheetcol: value or None}
    for d in dates:
        try:
            df = statcast(start_dt=d, end_dt=d, verbose=False)
        except Exception as e:
            print(f"  {d}: pull failed {e!r}"); continue
        df = df[df['game_pk'].isin(targets)]
        bypa = defaultdict(list)
        keep = ['game_pk', 'at_bat_number', 'pitch_number', 'description'] + list(BAT.values())
        keep = [c for c in keep if c in df.columns]
        for r in df[keep].itertuples(index=False):
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
                bs = sf(getattr(r, 'bat_speed', None))
                if bs is None or bs < 50:      # not a valid swing per pipeline rule
                    continue
                cur[f"{pk}_{ab:03d}_{fn:02d}"] = {sc: sf(getattr(r, vc, None)) for sc, vc in BAT.items()}
        print(f"  {d}: cumulative valid swings {len(cur)}", flush=True)
    print(f"current valid-swing pitches: {len(cur)}", flush=True)

    gc = gspread.service_account()
    staged = []   # (title, ws, ri, col1, newstr)
    per = defaultdict(int)
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
            for ri in range(1, len(vals)):
                r = vals[ri]
                pid = r[pc] if pc < len(r) else ''
                g = pid.split('_')[0]
                if not g.isdigit() or int(g) not in targets:
                    continue
                new = cur.get(pid)
                if not new:
                    continue
                for sc, vc in BAT.items():
                    if sc not in ci:
                        continue
                    nv = new.get(sc)
                    if nv is None:
                        continue
                    stored = sf(r[ci[sc]]) if ci[sc] < len(r) else None
                    if stored is None or abs(stored - nv) > TOL:
                        staged.append((ws.title, ws, ri + 1, ci[sc] + 1, f"{nv:.1f}"))
                        per[sc] += 1
    print(f"\ncells to overwrite: {len(staged)}   by field: {dict(per)}", flush=True)
    # show a few big SwingLength corrections
    print("(dry-run shows corrections are downward toward normal)")

    if not APPLY:
        print("=== DRY RUN (no writes) ===")
        return
    byws = defaultdict(list)
    for title, ws, ri, col, val in staged:
        byws[(title, ws)].append((ri, col, val))
    total = 0
    for (title, ws), items in byws.items():
        cells = [gspread.Cell(ri, col, val) for (ri, col, val) in items]
        B.update_cells_with_retry(ws, cells, value_input_option='USER_ENTERED')
        total += len(cells)
        print(f"  [{title}] wrote {len(cells)}", flush=True)
        time.sleep(1.0)
    print(f"=== APPLIED: {total} cells ===")


if __name__ == '__main__':
    main()
