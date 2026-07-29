"""repair_spin.py — sync Spin Rate to the current (reprocessed) feed value,
INCLUDING recovering blanks (deletions Statcast has since fixed).

Wally's workflow was to delete clearly-wrong spins; but the feed often has the
corrected value now, so recovering beats deleting. This flags every pitch where
the feed value differs from what's stored (blank OR a value) AND the feed value
is plausible for THAT pitcher's pitch type — the plausibility gate is what keeps
us from re-filling a blank with the same bad value that was deleted, or writing a
still-garbage feed reading.

Plausible = within BAND of the pitcher's per-pitch-type median feed spin (needs
>= MIN_SAMPLES), else a general sanity range. Categories:
  blank_recovered  — cell was blank, feed now has a good value
  corrupt_repaired — stored was implausible (e.g. 675-rpm slider), feed is good
  drift_updated    — stored plausible but the feed reprocessed to a new value

Guarded, dry-run first.
  python3 scripts/repair_spin.py            # DRY RUN
  python3 scripts/repair_spin.py --apply
"""
import os, sys, time, statistics as st, warnings, concurrent.futures as cf
from collections import defaultdict
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import requests, gspread
import backfill_supplement as B

APPLY = '--apply' in sys.argv
BAND = 500          # feed must be within this of the pitcher's pitch-type median
MIN_SAMPLES = 5     # ...if we have at least this many feed samples for the group
MIN_DELTA = 25      # ignore sub-this differences on non-blank cells (rounding)
FEED = "https://statsapi.mlb.com/api/v1.1/game/{}/feed/live"
SKIP_TYPES = {'KN', 'EP'}   # legitimately low/odd spin — use a wide sanity range


def sf(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def feed_spin(pk):
    try:
        plays = requests.get(FEED.format(pk), timeout=60).json()['liveData']['plays']['allPlays']
    except Exception:
        return None
    out = {}
    for play in plays:
        ab = play.get('atBatIndex', 0) + 1
        for ev in play.get('playEvents', []):
            if not ev.get('isPitch', False):
                continue
            pn = ev.get('pitchNumber') or 0
            sr = ev.get('pitchData', {}).get('breaks', {}).get('spinRate')
            if sr is not None:
                out[f"{pk}_{ab:03d}_{pn:02d}"] = float(sr)
    return out


def main():
    gc = gspread.service_account()
    tabs = []; pks = set()
    print("reading sheets ...", flush=True)
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for ws in sh.worksheets():
            t = ws.title.upper()
            if t not in B.ALL_TRACKED_TEAMS or t in ('ROC', 'AAA', 'FCL'):
                continue
            time.sleep(0.4)
            vals = ws.get_all_values()
            if not vals or 'PitchID' not in vals[0] or 'Spin Rate' not in vals[0]:
                continue
            ci = {n: j for j, n in enumerate(vals[0]) if n}
            tabs.append((label, ws.title, ws, vals, ci))
            pc = ci['PitchID']
            for r in vals[1:]:
                p = (r[pc] if pc < len(r) else '').split('_')
                if len(p) == 3 and p[0].isdigit():
                    pks.add(int(p[0]))
    print(f"MLB tabs: {len(tabs)}   games: {len(pks)}", flush=True)

    print("pulling feeds ...", flush=True)
    feed = {}; failed = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(feed_spin, pk): pk for pk in pks}
        done = 0
        for fu in cf.as_completed(futs):
            res = fu.result()
            if res is None: failed.append(futs[fu])
            else: feed.update(res)
            done += 1
            if done % 300 == 0: print(f"  {done}/{len(pks)}", flush=True)
    print(f"feed spins: {len(feed)}   failed games: {len(failed)} {failed[:6]}", flush=True)

    # per (pitcher, pitch type) median feed spin — robust baseline
    groups = defaultdict(list)
    rows = []   # (title, ws, ri, col, pid, stored, pitcher, ptype, feedv)
    for label, title, ws, vals, ci in tabs:
        pc, sc = ci['PitchID'], ci['Spin Rate']
        ptc, pit = ci.get('Pitch Type'), ci.get('Pitcher')
        for ri in range(1, len(vals)):
            r = vals[ri]
            pid = r[pc] if pc < len(r) else ''
            fv = feed.get(pid)
            if fv is None:
                continue
            stored = sf(r[sc]) if sc < len(r) else None
            pitcher = r[pit] if pit is not None and pit < len(r) else ''
            ptype = r[ptc] if ptc is not None and ptc < len(r) else ''
            groups[(pitcher, ptype)].append(fv)
            rows.append((title, ws, ri + 1, sc + 1, pid, stored, pitcher, ptype, fv))
    median = {k: st.median(v) for k, v in groups.items() if len(v) >= MIN_SAMPLES}

    def plausible(pitcher, ptype, fv):
        m = median.get((pitcher, ptype))
        if m is not None:
            return abs(fv - m) <= BAND
        lo = 500 if ptype in SKIP_TYPES else 1100
        return lo <= fv <= 3800

    staged = []   # (title, ws, ri, col, pid, stored, feedv, pitcher, ptype, category)
    for title, ws, ri, col, pid, stored, pitcher, ptype, fv in rows:
        if stored is not None and abs(stored - fv) <= MIN_DELTA:
            continue
        if not plausible(pitcher, ptype, fv):
            continue
        if stored is None:
            cat = 'blank_recovered'
        elif (ptype not in SKIP_TYPES and stored < 1000) or abs(stored - fv) > 800:
            cat = 'corrupt_repaired'
        else:
            cat = 'drift_updated'
        staged.append((title, ws, ri, col, pid, stored, fv, pitcher, ptype, cat))

    bycat = defaultdict(int)
    for s in staged:
        bycat[s[9]] += 1
    print(f"\n=== spin cells to update: {len(staged)} ===")
    for k in ('corrupt_repaired', 'blank_recovered', 'drift_updated'):
        print(f"  {k}: {bycat[k]}")
    print("\n  corrupt_repaired examples (largest):")
    for s in sorted([s for s in staged if s[9] == 'corrupt_repaired'], key=lambda z: -abs((z[5] or 0) - z[6]))[:12]:
        print(f"    {s[4]} [{s[0]}] {s[7]} {s[8]}: sheet={s[5]:.0f} -> feed={s[6]:.0f}")
    print("  blank_recovered examples:")
    for s in [s for s in staged if s[9] == 'blank_recovered'][:12]:
        print(f"    {s[4]} [{s[0]}] {s[7]} {s[8]}: blank -> feed={s[6]:.0f}")

    if not APPLY:
        print("\n=== DRY RUN (no writes) ===")
        return
    byws = defaultdict(list)
    for title, ws, ri, col, pid, stored, fv, pitcher, ptype, cat in staged:
        byws[(title, ws)].append((ri, col, fv))
    total = 0
    for (title, ws), items in byws.items():
        cells = [gspread.Cell(ri, col, str(int(round(fv)))) for (ri, col, fv) in items]
        B.update_cells_with_retry(ws, cells, value_input_option='USER_ENTERED')
        total += len(cells)
        print(f"  [{title}] wrote {len(cells)}", flush=True)
        time.sleep(1.0)
    print(f"=== APPLIED: {total} cells ===")


if __name__ == '__main__':
    main()
