"""Live catch probability watcher.

Polls Baseball Savant's player-services/range endpoint for every outfielder
in a game and prints each newly tracked OF opportunity with the official
catch probability (catch_rate), distance needed, opportunity time, and
back/wall flags. Standalone; does not touch the Pitcher2026 pipeline.

Usage:
  python3 scripts/live_catch_prob.py --game_pk 823598            # poll live
  python3 scripts/live_catch_prob.py --game_pk 823598 --once     # one pass
  python3 scripts/live_catch_prob.py --game_pk 823598 --all      # include 0.99 routine plays

Notes:
  - catch_rate is the official Statcast catch probability, quantized to
    0.05 steps and capped at 0.99. stars: 5 = 0-25%, 4 = 26-50%, 3 = 51-75%,
    2 = 76-90%, 1 = 91-95%.
  - The endpoint 403s python-urllib's default User-Agent; a browser UA fixes it.
  - Latency vs. the live broadcast is not guaranteed; this prints an arrival
    timestamp per play so latency can be measured against game time.
"""

import argparse
import json
import time
import urllib.request
from datetime import datetime

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
OF_POS = {"LF", "CF", "RF", "OF"}


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=20) as f:
        return json.load(f)


def game_outfielders(game_pk):
    """All players on both rosters whose listed position is OF (starters + bench)."""
    box = get_json(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore")
    season = None
    ofs = {}
    for side in ("home", "away"):
        team = box["teams"][side]
        for key, p in team["players"].items():
            pos = p.get("position", {}).get("abbreviation")
            all_pos = {q.get("abbreviation") for q in p.get("allPositions", [])}
            if pos in OF_POS or all_pos & OF_POS:
                ofs[p["person"]["id"]] = p["person"]["fullName"]
    sched = get_json(
        f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    )
    season = sched["gameData"]["game"]["season"]
    return ofs, season


def fetch_range(player_id, season):
    url = (
        "https://baseballsavant.mlb.com/player-services/range"
        f"?playerId={player_id}&season={season}"
    )
    try:
        return get_json(url)
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game_pk", type=int, required=True)
    ap.add_argument("--interval", type=int, default=60, help="poll seconds")
    ap.add_argument("--once", action="store_true", help="single pass, no loop")
    ap.add_argument("--all", action="store_true", help="show routine (0.99) plays too")
    args = ap.parse_args()

    ofs, season = game_outfielders(args.game_pk)
    print(f"Watching game {args.game_pk} ({season}): "
          f"{len(ofs)} outfielders: {', '.join(sorted(ofs.values()))}")

    seen = set()
    first_pass = True
    while True:
        for pid, name in ofs.items():
            for r in fetch_range(pid, season):
                if r.get("game_pk") != args.game_pk:
                    continue
                play_id = r["play_id"]
                if play_id in seen:
                    continue
                seen.add(play_id)
                cr = float(r["catch_rate"])
                if cr >= 0.99 and not args.all:
                    continue
                stamp = datetime.now().strftime("%H:%M:%S")
                flags = []
                if int(r.get("back", 0)):
                    flags.append("BACK")
                if int(r.get("wall", 0)):
                    flags.append("WALL")
                note = " ".join(flags)
                out = "OUT" if int(r.get("out", 0)) else "no catch"
                # first pass just backfills already-recorded plays quietly
                prefix = "[backfill]" if first_pass else f"[{stamp}]"
                print(f"{prefix} {name}: catch prob {cr:.2f} "
                      f"({r['stars']}-star), {r['distance']} ft in "
                      f"{r['opportunity_time']}s {note} -> {out}")
        if args.once:
            break
        first_pass = False
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
