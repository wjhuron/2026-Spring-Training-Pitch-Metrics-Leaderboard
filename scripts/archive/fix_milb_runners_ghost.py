"""fix_milb_runners_ghost.py — correct ROC/AAA Runners for the extra-innings
ghost runner that the initial backfill missed.

backfill_milb_feed.py originally reset the base state to empty each half-inning,
which dropped the automatic runner on 2B that starts every extra (10th+) inning.
feed_gamestate is now fixed; this re-derives ROC/AAA Runners with the corrected
logic and OVERWRITES cells that disagree (the feed is the only base-state source
for MiLB). MLB is untouched — its Runners comes from Savant, which already had
the ghost runner. Guarded, dry-run first.

  python3 scripts/fix_milb_runners_ghost.py            # DRY RUN
  python3 scripts/fix_milb_runners_ghost.py --apply
"""
import os, sys, time, warnings, concurrent.futures as cf
from collections import defaultdict
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import gspread, backfill_supplement as B
import backfill_milb_feed as MF

APPLY = '--apply' in sys.argv
MILB_BOOK = 'NLE2026'
MILB_TABS = ('ROC', 'AAA')


def main():
    gc = gspread.service_account()
    sh = gc.open_by_key(B.SPREADSHEET_IDS[MILB_BOOK])
    tabs = []; pks = set()
    for name in MILB_TABS:
        ws = sh.worksheet(name)
        vals = ws.get_all_values()
        ci = {n: j for j, n in enumerate(vals[0]) if n}
        if 'PitchID' not in ci or 'Runners' not in ci:
            continue
        tabs.append((name, ws, vals, ci))
        pc = ci['PitchID']
        for r in vals[1:]:
            p = (r[pc] if pc < len(r) else '').split('_')
            if len(p) == 3 and p[0].isdigit():
                pks.add(int(p[0]))
    print(f"MiLB games: {len(pks)}", flush=True)

    feed = {}
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(MF.feed_gamestate, pk): pk for pk in pks}
        for fu in cf.as_completed(futs):
            res = fu.result()
            if res:
                for pid, (o, run) in res.items():
                    feed[pid] = run

    staged = []
    for name, ws, vals, ci in tabs:
        pc, rc = ci['PitchID'], ci['Runners']
        for ri in range(1, len(vals)):
            r = vals[ri]
            pid = r[pc] if pc < len(r) else ''
            fv = feed.get(pid)
            if fv is None:
                continue
            cur = str(r[rc]).strip() if rc < len(r) else ''
            if cur != '' and cur != fv:
                staged.append((name, ws, ri + 1, rc + 1, cur, fv))

    print(f"\nRunners cells to correct: {len(staged)}")
    patt = defaultdict(int)
    for _, _, _, _, cur, fv in staged:
        patt[f"{cur!r}->{fv!r}"] += 1
    for k, v in sorted(patt.items(), key=lambda x: -x[1]):
        print(f"   {k}: {v}")

    if not APPLY:
        print("=== DRY RUN (no writes) ===")
        return
    byws = defaultdict(list)
    for name, ws, ri, col, cur, fv in staged:
        byws[(name, ws)].append((ri, col, fv))
    total = 0
    for (name, ws), items in byws.items():
        cells = [gspread.Cell(ri, col, fv) for (ri, col, fv) in items]
        B.update_cells_with_retry(ws, cells, value_input_option='USER_ENTERED')
        total += len(cells)
        print(f"  [{name}] wrote {len(cells)}", flush=True)
        time.sleep(1.0)
    print(f"=== APPLIED: {total} cells ===")


if __name__ == '__main__':
    main()
