"""missing_pa_check.py — find whole PAs present in the feed but absent from the
sheets (and sheet PAs absent from the feed). READ-ONLY.

Complements audit_full.py, whose count comparison only fires when the sheet has
at least one pitch of the PA. Reads ONLY the PitchID column per tab (cheap),
fetches every game's play-by-play, and reports:
  - MISSING_PA: feed has pitches, sheet has none
  - PHANTOM_PA: sheet has pitches, feed has none

Usage: python3 scripts/audits/missing_pa_check.py
"""
import os, sys, time, requests
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import gspread
import scrapers.backfill_supplement as B
from concurrent.futures import ThreadPoolExecutor

PBP = "https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay"


def fetch(pk):
    try:
        return pk, requests.get(PBP.format(pk=pk), timeout=30).json()
    except Exception:
        return pk, None


def main():
    gc = gspread.service_account()
    sheet_pas = defaultdict(int)     # (pk, ab) -> pitch count
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for ws in sh.worksheets():
            time.sleep(0.5)
            if ws.title.upper() == 'NEW':
                continue   # Wally's scratch tab: per-pitcher copies, not pipeline data
            try:
                header = ws.row_values(1)
            except Exception:
                continue
            if 'PitchID' not in header:
                continue
            col = header.index('PitchID') + 1
            for pid in ws.col_values(col)[1:]:
                parts = str(pid).split('_')
                if len(parts) == 3 and parts[0].isdigit():
                    sheet_pas[(parts[0], int(parts[1]))] += 1
    pks = sorted({pk for pk, ab in sheet_pas})
    print(f"sheet PAs: {len(sheet_pas)} across {len(pks)} games; fetching feeds ...", flush=True)

    feeds = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        for i, (pk, js) in enumerate(ex.map(fetch, pks)):
            feeds[pk] = js
            if (i + 1) % 400 == 0:
                print(f"  {i + 1}/{len(pks)}", flush=True)

    missing, phantom, mismatch = [], [], []
    feed_pas = set()
    for pk, js in feeds.items():
        if not js:
            continue
        for play in js.get('allPlays', []):
            ab = play.get('atBatIndex', 0) + 1
            fn = sum(1 for e in play.get('playEvents', []) if e.get('isPitch'))
            feed_pas.add((pk, ab))
            pn = sheet_pas.get((pk, ab), 0)
            m = play.get('matchup', {})
            who = (f"({m.get('batter',{}).get('fullName')} vs "
                   f"{m.get('pitcher',{}).get('fullName')}, "
                   f"{play.get('result',{}).get('event')})")
            if fn > 0 and pn == 0:
                missing.append(f"{pk}_{ab:03d}: feed={fn} pitches {who}")
            elif fn > 0 and pn != fn:
                mismatch.append(f"{pk}_{ab:03d}: sheet={pn} feed={fn} {who}")
    for (pk, ab), n in sheet_pas.items():
        if feeds.get(pk) and (pk, ab) not in feed_pas:
            phantom.append(f"{pk}_{ab:03d}: sheet has {n} pitches, feed has no such PA")

    print(f"\n=== MISSING whole PAs (feed yes, sheet no): {len(missing)} ===")
    for m in missing[:60]:
        print(f"  {m}")
    if len(missing) > 60:
        print(f"  ... and {len(missing) - 60} more")
    print(f"\n=== COUNT MISMATCH PAs (both have pitches, counts differ): {len(mismatch)} ===")
    for m in mismatch[:40]:
        print(f"  {m}")
    print(f"\n=== PHANTOM PAs (sheet yes, feed no): {len(phantom)} ===")
    for p in phantom[:20]:
        print(f"  {p}")


if __name__ == '__main__':
    main()
