"""fix_outcomes.py — apply the OUTCOME_MISMATCH corrections (go with the feed).

For each pitch where the pickle disagrees with the current official feed on the
result, update Description and/or Count to the feed value. Surgical + guarded:
a cell is written ONLY if it still holds the exact wrong value. PAs that contain
an In-Play description change are EXCLUDED (those imply hit-data/Event changes
that need manual review) and printed for follow-up.

  python3 scripts/fix_outcomes.py            # DRY RUN
  python3 scripts/fix_outcomes.py --apply    # write to Sheets
"""
import os, sys, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gspread
import backfill_supplement as B

APPLY = '--apply' in sys.argv
CSV = os.path.expanduser('~/Downloads/count_review_2026.csv')


def main():
    rows = [r for r in csv.DictReader(open(CSV)) if r['category'] == 'OUTCOME_MISMATCH']
    # exclude any PA that has an In-Play description change (manual review)
    manual_pa = set()
    for r in rows:
        if r['correct_desc'] and 'In Play' in (r['stored_desc'], r['correct_desc']) \
                and r['stored_desc'] != r['correct_desc']:
            manual_pa.add('_'.join(r['PitchID'].split('_')[:2]))
    fixes = {}
    for r in rows:
        if '_'.join(r['PitchID'].split('_')[:2]) in manual_pa:
            continue
        d_change = r['correct_desc'] and r['stored_desc'] != r['correct_desc']
        c_change = r['stored_count'] != r['correct_count'] and r['correct_count']
        if d_change or c_change:
            fixes[r['PitchID']] = {'sd': r['stored_desc'], 'cd': r['correct_desc'] if d_change else None,
                                   'sc': r['stored_count'], 'cc': r['correct_count'] if c_change else None}
    print(f"{'APPLY' if APPLY else 'DRY RUN'}: {len(fixes)} outcome pitches to fix "
          f"({len(manual_pa)} PAs held for manual review)\n", flush=True)

    gc = gspread.service_account()
    found = set(); desc_fx = cnt_fx = already = mism = 0
    notes = []
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for i, ws in enumerate(sh.worksheets()):
            if i:
                time.sleep(1.2)
            data = B.read_sheet_with_retry(ws)
            if not data or len(data) < 2:
                continue
            ci = {n: j for j, n in enumerate(data[0]) if n}
            if 'PitchID' not in ci or 'Count' not in ci or 'Description' not in ci:
                continue
            pc, cc, dc = ci['PitchID'], ci['Count'], ci['Description']
            cells = []
            for li in range(1, len(data)):
                row = data[li]
                pid = row[pc] if pc < len(row) else ''
                if pid not in fixes:
                    continue
                found.add(pid); fx = fixes[pid]
                if fx['cd']:
                    cur = row[dc] if dc < len(row) else ''
                    if cur == fx['cd']:
                        already += 1
                    elif cur == fx['sd']:
                        cells.append(gspread.Cell(li + 1, dc + 1, fx['cd'])); desc_fx += 1
                    else:
                        mism += 1; notes.append(f"{pid} desc sheet='{cur}' expected '{fx['sd']}' -> skip")
                if fx['cc']:
                    cur = row[cc] if cc < len(row) else ''
                    if cur == fx['cc']:
                        already += 1
                    elif cur == fx['sc']:
                        cells.append(gspread.Cell(li + 1, cc + 1, fx['cc'])); cnt_fx += 1
                    else:
                        mism += 1; notes.append(f"{pid} count sheet='{cur}' expected '{fx['sc']}' -> skip")
            if cells:
                print(f"  [{label}/{ws.title}] {len(cells)} cell(s) {'WRITING' if APPLY else 'would fix'}", flush=True)
                if APPLY:
                    B.update_cells_with_retry(ws, cells, value_input_option='RAW')

    print(f"\n=== {'APPLIED' if APPLY else 'DRY RUN'} SUMMARY ===")
    print(f"  Description cells: {desc_fx}")
    print(f"  Count cells:       {cnt_fx}")
    print(f"  already correct:   {already}")
    print(f"  unexpected (skip): {mism}")
    print(f"  not found:         {len(fixes) - len(found)}")
    for n in notes[:20]:
        print(f"    {n}")
    if manual_pa:
        print(f"\n  HELD FOR MANUAL REVIEW (In-Play change): {len(manual_pa)} PAs")
        for pa in sorted(manual_pa):
            ex = [r for r in rows if '_'.join(r['PitchID'].split('_')[:2]) == pa and r['correct_desc']
                  and 'In Play' in (r['stored_desc'], r['correct_desc'])]
            for r in ex:
                print(f"    {r['PitchID']}  {r['stored_desc']} -> {r['correct_desc']}")
    if not APPLY:
        print("\n(dry run only — re-run with --apply)")


if __name__ == '__main__':
    main()
