"""audit_runners.py — audit the Runners column across all 32 tabs (30 MLB +
ROC + AAA) against the feed's per-pitch base state. READ-ONLY.

MLB Runners comes from the Savant supplement (on_1b/2b/3b per pitch); ROC/AAA was
just backfilled from the feed. This re-derives per-pitch base state from the feed
(pre-PA state carried across plays + mid-PA runner movements — the logic verified
157/157 on 822757 incl. a steal) and flags any stored value that disagrees,
highlighting PAs where the base state changed mid-PA (steal/pickoff/etc.).
"""
import os, sys, time, warnings, concurrent.futures as cf
from collections import defaultdict
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
import gspread
import scrapers.backfill_supplement as B
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'builders'))
import backfill_milb_feed as MF   # reuse feed_gamestate (returns {PitchID:(outs,runners)})


def main():
    gc = gspread.service_account()
    stored = {}   # PitchID -> stored runners
    pks = set()
    tabs_read = []
    print("reading sheets ...", flush=True)
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for ws in sh.worksheets():
            t = ws.title.upper()
            if t not in B.ALL_TRACKED_TEAMS and t not in ('ROC', 'AAA'):
                continue
            if t == 'FCL':
                continue
            time.sleep(0.4)
            vals = ws.get_all_values()
            if not vals or 'PitchID' not in vals[0] or 'Runners' not in vals[0]:
                continue
            ci = {n: j for j, n in enumerate(vals[0]) if n}
            pc, rc = ci['PitchID'], ci['Runners']
            tabs_read.append(t)
            for r in vals[1:]:
                pid = r[pc] if pc < len(r) else ''
                p = pid.split('_')
                if len(p) != 3 or not p[0].isdigit():
                    continue
                rv = r[rc] if rc < len(r) else ''
                if str(rv).strip() == '':
                    continue
                stored[pid] = str(rv).strip()
                pks.add(int(p[0]))
    print(f"tabs: {len(tabs_read)}   pitches with stored Runners: {len(stored)}   games: {len(pks)}", flush=True)

    print("pulling feeds ...", flush=True)
    feed = {}; failed = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(MF.feed_gamestate, pk): pk for pk in pks}
        done = 0
        for fu in cf.as_completed(futs):
            res = fu.result()
            if res is None:
                failed.append(futs[fu])
            else:
                for pid, (o, run) in res.items():
                    feed[pid] = run
            done += 1
            if done % 300 == 0:
                print(f"  {done}/{len(pks)}", flush=True)
    print(f"feed pitches: {len(feed)}   failed games: {len(failed)} {failed[:6]}", flush=True)

    # which PAs have a mid-PA base change (feed runners varies across the PA's pitches)?
    pa_vals = defaultdict(set)
    for pid, run in feed.items():
        g, pa, _ = pid.split('_')
        pa_vals[(g, pa)].add(run)
    midpa = {k for k, s in pa_vals.items() if len(s) > 1}
    print(f"PAs with a mid-PA base change in the feed: {len(midpa)}", flush=True)

    mism = []; midpa_mism = []
    for pid, sv in stored.items():
        fv = feed.get(pid)
        if fv is None:
            continue
        if sv != fv:
            g, pa, _ = pid.split('_')
            is_mid = (g, pa) in midpa
            mism.append((pid, sv, fv, is_mid))
            if is_mid:
                midpa_mism.append((pid, sv, fv))

    print(f"\n=== compared: {sum(1 for p in stored if p in feed)}   mismatches: {len(mism)} ===")
    print(f"  of which in a mid-PA-change PA: {len(midpa_mism)}")
    # split MLB vs MiLB (MiLB game_pks < ~830000 heuristic won't hold; use tab set instead)
    print("\n  mid-PA-change mismatches (stored did not reflect the base change):")
    for pid, sv, fv in midpa_mism[:30]:
        print(f"    {pid}: stored={sv!r}  feed={fv!r}")
    other = [m for m in mism if not m[3]]
    print(f"\n  other mismatches (not mid-PA): {len(other)}")
    for pid, sv, fv, _ in other[:20]:
        print(f"    {pid}: stored={sv!r}  feed={fv!r}")


if __name__ == '__main__':
    main()
