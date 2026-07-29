"""fix_attribution.py — write the confirmed per-pitch attribution corrections.

From attribution_mismatches_2026.csv:
  - BATTER errors: set Batter (+ Bats) to the feed-correct value for every pitch of
    the PA (whole-PA mislabels / lineup shifts / pre-PA pinch-hit).
  - HAND (bats) errors: set Bats to the feed's actual side (switch hitters who
    batted their natural side).
Pitcher errors are skipped (824702 handled manually). Guarded: only overwrites a
cell that still holds the exact wrong value from the audit.

  python3 scripts/fix_attribution.py            # DRY RUN
  python3 scripts/fix_attribution.py --apply
"""
import os, sys, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gspread
import backfill_supplement as B

APPLY = '--apply' in sys.argv
CSVPATH = os.path.expanduser('~/Downloads/attribution_mismatches_2026.csv')


def main():
    # per PitchID: what to set + the expected current (wrong) value to guard on
    fixes = {}  # pid -> {'Batter': (wrong,right)|None, 'Bats': (wrong,right)|None}
    for r in csv.DictReader(open(CSVPATH)):
        if 'PITCHER' in r['issue'] and 'BATTER' not in r['issue']:
            continue  # pitcher-only (824702) handled manually
        pid = r['PitchID']
        f = fixes.setdefault(pid, {'Batter': None, 'Bats': None})
        if 'BATTER' in r['issue']:
            f['Batter'] = (r['sheet_batter'], r['feed_batter'])
            f['Bats'] = (r['sheet_bats'], r['feed_bats'])
        elif 'HAND' in r['issue'] and r['sheet_bats'] != r['feed_bats']:
            f['Bats'] = (r['sheet_bats'], r['feed_bats'])
    print(f"pitches to fix: {len(fixes)}")

    gc = gspread.service_account()
    done = 0
    skipped = []
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for wi, ws in enumerate(sh.worksheets()):
            if ws.title.upper() not in B.ALL_TRACKED_TEAMS:
                continue
            if wi:
                time.sleep(0.5)
            vals = ws.get_all_values()
            if not vals or 'PitchID' not in vals[0]:
                continue
            ci = {n: j for j, n in enumerate(vals[0]) if n}
            pc = ci['PitchID']
            cells = []
            for ri in range(1, len(vals)):
                pid = str(vals[ri][pc]) if pc < len(vals[ri]) else ''
                if pid not in fixes:
                    continue
                f = fixes[pid]
                for col in ('Batter', 'Bats'):
                    if f[col] is None or col not in ci:
                        continue
                    wrong, right = f[col]
                    cur = vals[ri][ci[col]] if ci[col] < len(vals[ri]) else ''
                    if cur == right:
                        continue  # already correct
                    if cur != wrong:
                        skipped.append(f"{pid}.{col}: cur='{cur}' expected-wrong='{wrong}'")
                        continue
                    cells.append(gspread.Cell(ri + 1, ci[col] + 1, right))
                    print(f"  {pid} {col}: '{wrong}' -> '{right}'")
            if cells and APPLY:
                B.update_cells_with_retry(ws, cells, value_input_option='USER_ENTERED')
                time.sleep(1.0)
            done += len(cells)

    print(f"\n=== {'APPLIED' if APPLY else 'DRY RUN'}: {done} cells ===")
    if skipped:
        print(f"skipped (current value unexpected) {len(skipped)}:")
        for s in skipped[:20]:
            print(f"   {s}")


if __name__ == '__main__':
    main()
