"""missing_pa_breakdown.py — full breakdown of every whole PA missing from the
sheets, grouped by game with per-club attribution. READ-ONLY.

For each game with missing PAs: date, matchup, and per pitching club how many
PAs are tracked vs missing, plus the missing PAs' pitchers. Shows whether the
misses are one-sided (opponent halves untracked = likely by design) or include
the tracked org's own arms (= real gaps).

Usage: python3 scripts/missing_pa_breakdown.py
"""
import os, sys, time, requests
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import gspread
import backfill_supplement as B
from concurrent.futures import ThreadPoolExecutor


def fetch(pk):
    try:
        return pk, requests.get(f"https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay", timeout=30).json()
    except Exception:
        return pk, None


def main():
    gc = gspread.service_account()
    sheet_pas = defaultdict(int)
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for ws in sh.worksheets():
            time.sleep(0.5)
            if ws.title.upper() == 'NEW':
                continue
            try:
                header = ws.row_values(1)
            except Exception:
                continue
            if 'PitchID' not in header:
                continue
            for pid in ws.col_values(header.index('PitchID') + 1)[1:]:
                parts = str(pid).split('_')
                if len(parts) == 3 and parts[0].isdigit():
                    sheet_pas[(parts[0], int(parts[1]))] += 1
    pks = sorted({pk for pk, ab in sheet_pas})
    print(f"fetching {len(pks)} feeds ...", flush=True)
    feeds = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        for pk, js in ex.map(fetch, pks):
            feeds[pk] = js

    # collect missing per game
    per_game = defaultdict(list)
    for pk, js in feeds.items():
        if not js:
            continue
        for play in js.get('allPlays', []):
            ab = play.get('atBatIndex', 0) + 1
            fn = sum(1 for e in play.get('playEvents', []) if e.get('isPitch'))
            if fn > 0 and sheet_pas.get((pk, ab), 0) == 0:
                per_game[pk].append((ab, play))

    print(f"\ngames with missing PAs: {len(per_game)}\n")
    for pk in sorted(per_game):
        js = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live", timeout=30).json()
        gd = js.get('gameData', {})
        date = gd.get('datetime', {}).get('officialDate', '?')
        home = gd.get('teams', {}).get('home', {}).get('name', '?')
        away = gd.get('teams', {}).get('away', {}).get('name', '?')
        gtype = gd.get('game', {}).get('type', '?')
        # tally tracked/missing per pitching side
        missing_abs = {ab for ab, _ in per_game[pk]}
        side_stat = {'home': [0, 0], 'away': [0, 0]}   # [tracked, missing]
        pbp = feeds[pk]
        for play in pbp.get('allPlays', []):
            ab = play.get('atBatIndex', 0) + 1
            fn = sum(1 for e in play.get('playEvents', []) if e.get('isPitch'))
            if fn == 0:
                continue
            side = 'home' if play.get('about', {}).get('halfInning') == 'top' else 'away'
            side_stat[side][1 if ab in missing_abs else 0] += 1
        print(f"{'='*84}")
        print(f"game {pk}  {date}  {away} @ {home}   (type={gtype})")
        print(f"  pitching side coverage: HOME({home.split()[-1]}): {side_stat['home'][0]} tracked / "
              f"{side_stat['home'][1]} missing   AWAY({away.split()[-1]}): {side_stat['away'][0]} tracked / "
              f"{side_stat['away'][1]} missing")
        bypitcher = defaultdict(list)
        for ab, play in sorted(per_game[pk]):
            bypitcher[play.get('matchup', {}).get('pitcher', {}).get('fullName')].append(ab)
        for p, abs_ in bypitcher.items():
            print(f"    missing: {p}  ({len(abs_)} PAs: {abs_})")


if __name__ == '__main__':
    main()
