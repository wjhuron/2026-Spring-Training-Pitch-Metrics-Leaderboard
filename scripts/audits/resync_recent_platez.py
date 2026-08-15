"""resync_recent_platez.py — keep recent games' PlateZ current as Statcast
reprocesses them.

Statcast revises plate_z after games (the early-2026 vertical recalibration is
the big example: March games ran ~1" high, corrected weeks later — see
memory reference_statcast_platez_recalibration). A game scraped soon after it
happens holds the pre-correction value until reprocessing lands (~weeks). This
job re-pulls the last N days of games, compares each pitch's stored PlateZ to
the current feed, and overwrites only the ones that changed (guarded). PlateX is
left alone (no systematic reprocessing bias). Games older than the window are
already settled, so they're never touched.

Writes to the six division books, so it runs LOCALLY with the write-capable
huronalytics account (same as backfill_supplement / sheets_append), NOT in the
read-only CI workflow.

  python3 scripts/audits/resync_recent_platez.py                 # DRY RUN, last 35 days
  python3 scripts/audits/resync_recent_platez.py --days 21 --apply
"""
import os, sys, json, time, argparse, warnings, concurrent.futures as cf
from collections import defaultdict
from datetime import date, timedelta
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
import requests, gspread
import scrapers.backfill_supplement as B

THRESH = 0.02   # sync genuine revisions above this (below = rounding/no-op)
CAP = 0.20      # never touch >0.2 here (that magnitude is lag corruption, handled separately)
SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={}&endDate={}"
FEED = "https://statsapi.mlb.com/api/v1.1/game/{}/feed/live"


def gspread_client():
    """Write-capable. Local: default huronalytics SA file. CI/env: full-scope key."""
    sa = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
    if sa:
        from google.oauth2.service_account import Credentials
        creds = Credentials.from_service_account_info(
            json.loads(sa), scopes=['https://www.googleapis.com/auth/spreadsheets'])
        return gspread.authorize(creds)
    return gspread.service_account()


def sf(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def recent_game_pks(days):
    end = date.today(); start = end - timedelta(days=days)
    data = requests.get(SCHEDULE.format(start, end), timeout=60).json()
    pks = set()
    for d in data.get('dates', []):
        for g in d.get('games', []):
            # final ('F') or official ('O') games only — the tracked data
            if (g.get('status', {}) or {}).get('codedGameState') in ('F', 'O'):
                pks.add(int(g['gamePk']))
    return pks


def feed_platez(pk):
    """{PitchID -> feed plate_z (round3)} replicating Pitcher2026 numbering."""
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
            pz = ev.get('pitchData', {}).get('coordinates', {}).get('pZ')
            out[f"{pk}_{ab:03d}_{pn:02d}"] = round(pz, 3) if pz is not None else None
    return out


def resync(gc, days=35, apply=False, log=print):
    """Re-sync recent games' PlateZ to the current feed using an existing
    (write-capable) gspread client. Returns the number of cells written (0 on a
    dry run or when nothing changed). Safe to call from backfill_supplement."""
    pks = recent_game_pks(days)
    log(f"recent games (last {days}d, final): {len(pks)}")
    if not pks:
        log("no games in window — nothing to do."); return 0

    log("pulling feeds ...")
    feed = {}; failed = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(feed_platez, pk): pk for pk in pks}
        for fu in cf.as_completed(futs):
            res = fu.result()
            if res is None: failed.append(futs[fu])
            else: feed.update(res)
    log(f"feed pitches: {len(feed)}   failed games: {len(failed)} {failed[:6]}")

    staged = []   # (title, ws, ri, cz, sheet_pz, feed_pz)
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for ws in sh.worksheets():
            t = ws.title.upper()
            if t not in B.ALL_TRACKED_TEAMS or t in ('ROC', 'AAA', 'FCL'):
                continue
            time.sleep(0.5)
            vals = ws.get_all_values()
            if not vals or 'PitchID' not in vals[0]:
                continue
            ci = {n: j for j, n in enumerate(vals[0]) if n}
            if 'PlateZ' not in ci:
                continue
            pc = ci['PitchID']; zc = ci['PlateZ']
            for ri in range(1, len(vals)):
                r = vals[ri]
                pid = r[pc] if pc < len(r) else ''
                p = pid.split('_')
                if len(p) != 3 or not p[0].isdigit() or int(p[0]) not in pks:
                    continue
                fz = feed.get(pid)
                if fz is None:
                    continue
                sz = sf(r[zc]) if zc < len(r) else None
                if sz is None:
                    continue
                if THRESH < abs(sz - fz) <= CAP:
                    staged.append((ws.title, ws, ri + 1, zc + 1, sz, fz))

    bytab = defaultdict(int)
    for s in staged:
        bytab[s[0]] += 1
    log(f"PlateZ cells to sync: {len(staged)}" + (f"   {dict(bytab)}" if staged else ""))

    if not apply:
        log("=== DRY RUN (no writes) — pass --apply to write ===")
        return 0
    byws = defaultdict(list)
    for title, ws, ri, cz, sz, fz in staged:
        byws[(title, ws)].append((ri, cz, fz))
    total = 0
    for (title, ws), items in byws.items():
        cells = [gspread.Cell(ri, cz, f"{fz:.3f}") for (ri, cz, fz) in items]
        B.update_cells_with_retry(ws, cells, value_input_option='USER_ENTERED')
        total += len(cells)
        log(f"  [{title}] wrote {len(cells)}")
        time.sleep(1.0)
    log(f"=== APPLIED: {total} PlateZ cells across {len(byws)} tabs ===")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=35)
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    resync(gspread_client(), days=args.days, apply=args.apply)


if __name__ == '__main__':
    main()
