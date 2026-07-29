"""apply_spin_rtilt_curation.py — apply Wally's curated spin/RTilt decisions from
"Stuff - Sheet9.csv". Decision -> action:
  APPLY                              -> Spin=feed, RTilt=feed
  SKIP-spin, APPLY-RTilt             -> RTilt=feed only
  Only Update Spin                   -> Spin=feed only
  Keep Spin, Delete ... RTilt        -> blank the RTilt cell
Overrides (per Wally): the 7 flagged spin outliers stay spin-deleted (demoted to
SKIP-spin/APPLY-RTilt), and King 824102_075_03 promoted to APPLY.
Feed values come straight from the CSV. Guarded (only writes a cell that still
holds the CSV's sheet_ value) + dry-run.

  python3 scripts/apply_spin_rtilt_curation.py            # DRY RUN
  python3 scripts/apply_spin_rtilt_curation.py --apply
"""
import os, sys, csv, time, warnings
from collections import defaultdict
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import gspread
import backfill_supplement as B

APPLY = '--apply' in sys.argv
CSVP = "/Users/wallyhuron/Downloads/Stuff - Sheet9.csv"
SEVEN = {'824605_052_06', '824044_056_02', '824779_002_01', '825090_052_03',
         '823536_022_01', '824203_078_01', '824044_072_01'}   # keep spin deleted
KING = '824102_075_03'                                        # promote to APPLY
DELETE_DEC = "Keep Spin, Delete current RTilt value and don't add new one"


def sf(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def actions(dec, feed_spin, feed_tilt):
    """returns (spin_write, tilt_write) where each is (do?, value). value '' = blank."""
    spin = (False, None); tilt = (False, None)
    if dec == 'APPLY':
        if feed_spin != '': spin = (True, feed_spin)
        if feed_tilt != '': tilt = (True, feed_tilt)
    elif dec == 'SKIP-spin, APPLY-RTilt':
        if feed_tilt != '': tilt = (True, feed_tilt)
    elif dec == 'Only Update Spin':
        if feed_spin != '': spin = (True, feed_spin)
    elif dec == DELETE_DEC:
        tilt = (True, '')          # blank
    return spin, tilt


def main():
    recs = {}
    for r in csv.DictReader(open(CSVP)):
        pid = r['PitchID']
        dec = r['recommendation'].strip()
        if pid in SEVEN: dec = 'SKIP-spin, APPLY-RTilt'
        if pid == KING: dec = 'APPLY'
        recs[pid] = dict(dec=dec, fs=r['feed_Spin'].strip(), ft=r['feed_RTilt'].strip(),
                         ss=r['sheet_Spin'].strip(), st=r['sheet_RTilt'].strip())
    print(f"curated rows: {len(recs)}")

    gc = gspread.service_account()
    staged = []   # (title, ws, row1, col1, newval, kind)
    guard_skips = []
    seen = set()
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
            pc = ci['PitchID']; sc = ci.get('Spin Rate'); tc = ci.get('RTilt')
            for ri in range(1, len(vals)):
                r = vals[ri]
                pid = r[pc] if pc < len(r) else ''
                if pid not in recs:
                    continue
                seen.add(pid)
                rec = recs[pid]
                (dospin, spinv), (dotilt, tiltv) = actions(rec['dec'], rec['fs'], rec['ft'])
                # SPIN
                if dospin and sc is not None:
                    cur = r[sc] if sc < len(r) else ''
                    exp = rec['ss']
                    ok = (cur.strip() == '' and exp == '') or (sf(cur) is not None and sf(exp) is not None and abs(sf(cur) - sf(exp)) < 0.5)
                    if ok:
                        if str(cur).strip() != str(spinv).strip():
                            staged.append((ws.title, ws, ri + 1, sc + 1, str(int(float(spinv))), 'spin'))
                    else:
                        guard_skips.append(f"{pid} spin (cur={cur!r} expected={exp!r})")
                # RTILT
                if dotilt and tc is not None:
                    cur = r[tc] if tc < len(r) else ''
                    exp = rec['st']
                    if cur.strip() == exp:
                        if cur.strip() != tiltv:
                            staged.append((ws.title, ws, ri + 1, tc + 1, tiltv, 'tilt' if tiltv else 'tilt-blank'))
                    else:
                        guard_skips.append(f"{pid} tilt (cur={cur!r} expected={exp!r})")

    missing = set(recs) - seen
    kinds = defaultdict(int)
    for s in staged:
        kinds[s[5]] += 1
    print(f"\ncells to write: {len(staged)}   {dict(kinds)}")
    print(f"guard skips (cell no longer holds expected value): {len(guard_skips)}")
    for g in guard_skips[:12]:
        print("   ", g)
    if missing:
        print(f"PitchIDs not found in sheets: {len(missing)} {list(missing)[:6]}")

    if not APPLY:
        print("\n=== DRY RUN (no writes) ===")
        return
    byws = defaultdict(list)
    for title, ws, r1, c1, val, kind in staged:
        byws[(title, ws)].append((r1, c1, val))
    total = 0
    for (title, ws), items in byws.items():
        cells = [gspread.Cell(r1, c1, val) for (r1, c1, val) in items]
        B.update_cells_with_retry(ws, cells, value_input_option='USER_ENTERED')
        total += len(cells)
        print(f"  [{title}] wrote {len(cells)}", flush=True)
        time.sleep(1.0)
    print(f"\n=== APPLIED: {total} cells ===")


if __name__ == '__main__':
    main()
