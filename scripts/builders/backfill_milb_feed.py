"""backfill_milb_feed.py — fill the feed-recoverable columns for the ROC/AAA tabs.

Statcast is thin in the minors, so most supplement columns (ArmAngle, xBA/xSLG/
xwOBA, RunExp, and all bat-tracking) are simply not published by Baseball Savant
for MiLB and can't be recovered. But two game-state columns come from the MLB
Stats API feed, which DOES cover Triple-A:

  * Outs    — the Triple-A feed omits about.outs (the MLB path's source), but the
              value is in each pitch's playEvents[].count.outs. Verified identical
              to the stored MLB Outs (822757: 157/157).
  * Runners — reconstructed per pitch by tracking base occupancy across plays
              (postOnFirst/Second/Third for the pre-PA state) and applying mid-PA
              runner movements (steals). Verified identical to stored MLB Runners
              (822757: 157/157, incl. a mid-PA steal). Format matches the sheet:
              bases occupied joined by '+', '0' when empty (e.g. '1+3', '0').

Fill-only (never overwrites an existing value), guarded, dry-run first. Runs
locally with the write-capable account. Both ROC and AAA tabs are covered.

  python3 scripts/builders/backfill_milb_feed.py            # DRY RUN
  python3 scripts/builders/backfill_milb_feed.py --apply
"""
import os, sys, time, warnings, concurrent.futures as cf
from collections import defaultdict
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
import requests, gspread
import scrapers.backfill_supplement as B

FEED = "https://statsapi.mlb.com/api/v1.1/game/{}/feed/live"
MILB_BOOK = 'NLE2026'          # the book carrying ROC/AAA/FCL
MILB_TABS = ('ROC', 'AAA')
COLS = ('Outs', 'Runners')     # the feed-recoverable columns


def _fmt_runners(state):
    p = [lbl for lbl, b in (('1', '1B'), ('2', '2B'), ('3', '3B')) if state[b]]
    return '+'.join(p) if p else '0'


def feed_gamestate(pk):
    """{PitchID -> (outs, runners)} for one game, matching MLB semantics."""
    try:
        plays = requests.get(FEED.format(pk), timeout=60).json()['liveData']['plays']['allPlays']
    except Exception:
        return None
    out = {}
    prev_half = None
    base = {'1B': False, '2B': False, '3B': False}
    for play in plays:
        inning = play.get('about', {}).get('inning')
        key = (inning, play.get('about', {}).get('halfInning'))
        if key != prev_half:                       # new half-inning
            # extra innings (10th+) start with the automatic runner ("ghost"/Manfred
            # runner) on 2B — MLB and MiLB regular season both use it.
            base = {'1B': False, '2B': bool(inning and inning >= 10), '3B': False}
            prev_half = key
        ab = play.get('atBatIndex', 0) + 1
        # runner movements grouped by the playEvent index where they occur
        mv = defaultdict(list)
        for r in play.get('runners', []):
            idx = (r.get('details', {}) or {}).get('playIndex')
            if idx is not None:
                m = r.get('movement', {}) or {}
                mv[idx].append((m.get('start'), m.get('end')))
        cur = dict(base)
        for i, ev in enumerate(play.get('playEvents', [])):
            if ev.get('isPitch', False):
                pn = ev.get('pitchNumber') or 0
                o = (ev.get('count', {}) or {}).get('outs')
                out[f"{pk}_{ab:03d}_{pn:02d}"] = (o, _fmt_runners(cur))
            for st, en in mv.get(i, []):           # apply mid-PA movement after the pitch it follows
                if st in ('1B', '2B', '3B'): cur[st] = False
                if en in ('1B', '2B', '3B'): cur[en] = True
        mt = play.get('matchup', {})               # authoritative hand-off to next play
        base = {'1B': bool(mt.get('postOnFirst')),
                '2B': bool(mt.get('postOnSecond')),
                '3B': bool(mt.get('postOnThird'))}
    return out


def run(gc=None, apply=False, log=print):
    """Backfill Outs+Runners into blank ROC/AAA cells. Returns cells written."""
    gc = gc or gspread.service_account()
    sh = gc.open_by_key(B.SPREADSHEET_IDS[MILB_BOOK])
    tabs = []; pks = set()
    for name in MILB_TABS:
        try:
            ws = sh.worksheet(name)
        except gspread.exceptions.WorksheetNotFound:
            log(f"[{name}] tab not found — skipping"); continue
        vals = ws.get_all_values()
        ci = {n: j for j, n in enumerate(vals[0]) if n}
        if 'PitchID' not in ci or not any(c in ci for c in COLS):
            log(f"[{name}] missing PitchID/{COLS} — skipping"); continue
        tabs.append((name, ws, vals, ci))
        pc = ci['PitchID']
        for r in vals[1:]:
            p = (r[pc] if pc < len(r) else '').split('_')
            if len(p) == 3 and p[0].isdigit():
                pks.add(int(p[0]))
    log(f"MiLB tabs: {[t[0] for t in tabs]}   distinct games: {len(pks)}")
    if not tabs:
        return 0

    log("pulling feeds ...")
    feed = {}; failed = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(feed_gamestate, pk): pk for pk in pks}
        for fu in cf.as_completed(futs):
            res = fu.result()
            if res is None: failed.append(futs[fu])
            else: feed.update(res)
    log(f"feed pitches: {len(feed)}   failed games: {len(failed)} {failed[:6]}")

    staged = []   # (name, ws, ri, col1, value)
    for name, ws, vals, ci in tabs:
        pc = ci['PitchID']
        counts = {c: [0, 0] for c in COLS}   # col -> [blank, filled]
        for ri in range(1, len(vals)):
            r = vals[ri]
            pid = r[pc] if pc < len(r) else ''
            gs = feed.get(pid)
            for k, col in enumerate(COLS):
                if col not in ci:
                    continue
                cur = r[ci[col]] if ci[col] < len(r) else ''
                if str(cur).strip() != '':
                    counts[col][1] += 1; continue     # never overwrite
                counts[col][0] += 1
                if gs is None or gs[k] is None:
                    continue
                staged.append((name, ws, ri + 1, ci[col] + 1, str(gs[k])))
        log(f"  [{name}] " + "  ".join(f"{c}: blank={counts[c][0]} filled={counts[c][1]}" for c in COLS if c in ci))

    percol = defaultdict(int)
    for _, _, _, col1, _ in staged:
        percol[col1] += 1
    log(f"\ncells to fill: {len(staged)}   (by column index: {dict(percol)})")

    if not apply:
        log("=== DRY RUN (no writes) ===")
        return 0
    byws = defaultdict(list)
    for name, ws, ri, col, val in staged:
        byws[(name, ws)].append((ri, col, val))
    total = 0
    for (name, ws), items in byws.items():
        cells = [gspread.Cell(ri, col, val) for (ri, col, val) in items]
        B.update_cells_with_retry(ws, cells, value_input_option='USER_ENTERED')
        total += len(cells)
        log(f"  [{name}] wrote {len(cells)}")
        time.sleep(1.0)
    log(f"=== APPLIED: {total} cells ===")
    return total


if __name__ == '__main__':
    run(apply='--apply' in sys.argv)
