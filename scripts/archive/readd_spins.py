"""readd_spins.py — write the curated Savant spins back into Spin Rate.

Reads the PitchID -> spin map (parsed from Wally's reviewed
deleted_spin_review_2026.numbers) and writes each spin into that pitch's Spin
Rate cell. Written as a real NUMBER via USER_ENTERED so it sorts with the
originals. Guarded: only a currently-BLANK Spin Rate cell is filled; any cell
that already holds a value is left alone and reported.

  python3 scripts/readd_spins.py            # DRY RUN
  python3 scripts/readd_spins.py --apply
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gspread
import backfill_supplement as B

APPLY = '--apply' in sys.argv
MAP_PATH = ('/private/tmp/claude-501/-Users-wallyhuron-Huronalytics/'
            'c95d9ba9-9386-4e80-a03e-e4563719adbb/scratchpad/spin_readd.json')


def col_letter(n):
    s = ''
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def main():
    spin = json.load(open(MAP_PATH))
    print(f"spins to re-add: {len(spin)}\n")
    gc = gspread.service_account()
    written = 0
    skip_nonblank = []
    found = set()
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for wi, ws in enumerate(sh.worksheets()):
            if wi:
                time.sleep(0.6)
            vals = ws.get_all_values()
            if not vals or len(vals) < 2:
                continue
            ci = {n: j for j, n in enumerate(vals[0]) if n}
            if 'PitchID' not in ci or 'Spin Rate' not in ci:
                continue
            pc = ci['PitchID']; sc = ci['Spin Rate']
            data = []
            for ri in range(1, len(vals)):
                pid = vals[ri][pc] if pc < len(vals[ri]) else ''
                if pid not in spin:
                    continue
                found.add(pid)
                cur = vals[ri][sc] if sc < len(vals[ri]) else ''
                if cur not in ('', None):
                    skip_nonblank.append((pid, cur))
                    continue
                data.append({'range': f"{col_letter(sc + 1)}{ri + 1}", 'values': [[spin[pid]]]})
            if data:
                print(f"[{label}/{ws.title}] {len(data)} spins to write")
                written += len(data)
                if APPLY:
                    ws.batch_update(data, value_input_option='USER_ENTERED')
                    time.sleep(1.2)

    unmatched = [p for p in spin if p not in found]
    print(f"\n=== {'APPLIED' if APPLY else 'DRY RUN'} ===")
    print(f"  would write: {written}" if not APPLY else f"  written: {written}")
    print(f"  skipped (Spin Rate not blank — left alone): {len(skip_nonblank)}")
    for pid, cur in skip_nonblank[:10]:
        print(f"     {pid}: has '{cur}'")
    print(f"  PitchIDs not found in any tab: {len(unmatched)}  {unmatched[:6]}")


if __name__ == '__main__':
    main()
