"""Pull a specific pitch from a live/recent game and print the matrix inputs.

Finds every pitch matching a batter + date (+ optional count / inning / half),
straight from MLB's Gameday feed, and prints the exact situation to type into
the ABS challenge matrix tool: count, inning, half, outs, runners, score
(away-home), and the pitch's plate_x / plate_z. Also gives the Savant clip.

Usage:
    python3 scripts/abs/abs_pull_pitch.py --player "James Wood" --date 2026-07-22 \
        --count 3-1 --inning 1 --half top
    python3 scripts/abs/abs_pull_pitch.py --player "Aaron Judge"   # all his pitches today

"count" is the PRE-pitch count (the count the pitch was thrown on).
"""

import argparse
import unicodedata
from datetime import date

import requests

SCHED = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={d}"
FEED = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"
VIDEO = "https://baseballsavant.mlb.com/sporty-videos?playId={pid}"


def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", required=True)
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--count", help="pre-pitch count, e.g. 3-1")
    ap.add_argument("--inning", type=int)
    ap.add_argument("--half", choices=["top", "bottom"])
    args = ap.parse_args()

    want_bs = None
    if args.count:
        b, s = args.count.split("-")
        want_bs = (int(b), int(s))
    target = norm(args.player)

    session = requests.Session()
    sched = session.get(SCHED.format(d=args.date), timeout=30).json()
    pks = [g["gamePk"] for dt in sched.get("dates", []) for g in dt["games"]]
    if not pks:
        print(f"no MLB games on {args.date}")
        return

    hits = []
    for pk in pks:
        try:
            feed = session.get(FEED.format(pk=pk), timeout=30).json()
        except Exception:
            continue
        gd, live = feed["gameData"], feed["liveData"]
        away = gd["teams"]["away"]["abbreviation"]
        home = gd["teams"]["home"]["abbreviation"]
        score = {"away": 0, "home": 0}
        prev_half, bases = None, {}
        for play in live["plays"]["allPlays"]:
            about = play["about"]
            half = "top" if about["halfInning"] == "top" else "bottom"
            hk = (about["inning"], half)
            if hk != prev_half:
                bases, prev_half = {}, hk
            bat_side = "away" if half == "top" else "home"
            batter = play["matchup"]["batter"]["fullName"]
            match_batter = target in norm(batter)
            b = s = 0
            for ev in play.get("playEvents", []):
                if ev.get("isPitch"):
                    pd = ev.get("pitchData", {})
                    coords = pd.get("coordinates", {})
                    if (match_batter and coords.get("pX") is not None
                            and (want_bs is None or (b, s) == want_bs)
                            and (args.inning is None or about["inning"] == args.inning)
                            and (args.half is None or half == args.half)):
                        det = ev["details"]
                        hits.append({
                            "batter": batter, "pitcher": play["matchup"]["pitcher"]["fullName"],
                            "inning": about["inning"], "half": half,
                            "balls": b, "strikes": s, "outs": ev["count"]["outs"],
                            "bases": "".join("1" if x in bases else "0" for x in ("1B", "2B", "3B")),
                            "away": score["away"], "home": score["home"],
                            "awayAbbr": away, "homeAbbr": home,
                            "px": coords["pX"], "pz": coords["pZ"],
                            "ptype": (det.get("type") or {}).get("description"),
                            "call": det["call"]["description"],
                            "playId": ev.get("playId"),
                        })
                    # advance count with this pitch's result
                    c = ev.get("count", {})
                    b, s = c.get("balls", b), c.get("strikes", s)
            # apply runner movement + runs after the PA
            runs = len({e["details"]["runner"]["id"] for e in play.get("runners", [])
                        if e.get("movement", {}).get("end") == "score"})
            score[bat_side] += runs
            for e in sorted((e for e in play.get("runners", []) if e.get("movement")),
                            key=lambda e: e["details"].get("playIndex", 0)):
                mv = e["movement"]
                rid = e["details"]["runner"]["id"]
                for bb in list(bases):
                    if bases[bb] == rid:
                        del bases[bb]
                if mv.get("end") in ("1B", "2B", "3B") and not mv.get("isOut"):
                    bases[mv["end"]] = rid

    if not hits:
        print(f"no matching pitch for {args.player} on {args.date}"
              + (f", count {args.count}" if args.count else "")
              + (f", inning {args.inning}" if args.inning else "")
              + (f", {args.half}" if args.half else ""))
        return

    print(f"\n{len(hits)} matching pitch(es):\n")
    for h in hits:
        on = [b for b, x in zip(("1B", "2B", "3B"), h["bases"]) if x == "1"]
        print(f"  {h['batter']} vs {h['pitcher']} - {h['ptype']}, called {h['call']}")
        print(f"  --- type this into the matrix ---")
        print(f"    Count:    {h['balls']}-{h['strikes']}")
        print(f"    Inning:   {h['inning']} ({h['half']})")
        print(f"    Outs:     {h['outs']}")
        print(f"    Runners:  {', '.join(on) if on else 'bases empty'}")
        print(f"    Score:    Away {h['away']} ({h['awayAbbr']}) - Home {h['home']} ({h['homeAbbr']})")
        print(f"    Hitter:   {h['batter']}  (loads exact zone)")
        print(f"    plate_x:  {h['px']:.2f}    plate_z: {h['pz']:.2f}")
        if h["playId"]:
            print(f"    video:    {VIDEO.format(pid=h['playId'])}")
        print()


if __name__ == "__main__":
    main()
