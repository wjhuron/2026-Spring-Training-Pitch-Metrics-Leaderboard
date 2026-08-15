"""Depth layer: value every affiliated player outside the core universe.

The core universe is Cot's 40-man/contracted players plus Board-graded
prospects. This sweep covers everyone else in affiliated ball (waived
veterans on minor deals, unranked farmhands): full-season rosters of all
affiliates via statsapi, deduped against the universe, then valued with the
same machinery as the historical trade corpus:

  - MLB history -> Marcel projection + debut-based control (value_mlb_at
    from tradevalue_snapshot, evaluated at today's date)
  - no MLB history -> unranked-minor-leaguer floor

Output: data/tradevalue_depth.json. The engine merges these as engine
"depth" rows so the site's search and team lists cover literally everyone.
"""

import json
from datetime import date
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
import tradevalue_snapshot as snap

BASE = Path("/Users/wallyhuron/Huronalytics")
DATA = BASE / "data"
OUT_PATH = DATA / "tradevalue_depth.json"
SEASON = 2026
SPORT_IDS = "11,12,13,14,16"


def main():
    universe = json.loads((DATA / "tradevalue_universe.json").read_text())
    known = {str(p["mlbam"]) for p in universe["mlb"] if p["mlbam"]}
    known |= {str(p["mlbam"]) for p in universe["prospects"] if p["mlbam"]}

    mlb_teams = snap.fetch_json(
        "https://statsapi.mlb.com/api/v1/teams?sportId=1")["teams"]
    abbrev = {t["id"]: t["abbreviation"] for t in mlb_teams}

    affiliates = snap.fetch_json(
        f"https://statsapi.mlb.com/api/v1/teams?sportIds={SPORT_IDS}")["teams"]
    players = {}  # mlbam -> {name, org, level, pos}
    for t in affiliates:
        parent = t.get("parentOrgId")
        if parent not in abbrev:
            continue
        try:
            roster = snap.fetch_json(
                f"https://statsapi.mlb.com/api/v1/teams/{t['id']}/roster"
                "?rosterType=fullSeason").get("roster", [])
        except Exception as e:
            print(f"roster FAILED {t['name']}: {e}")
            continue
        level = (t.get("sport") or {}).get("name", "")
        for entry in roster:
            pid = str(entry["person"]["id"])
            if pid in known or pid in players:
                continue
            players[pid] = {
                "mlbam": int(pid),
                "name": entry["person"]["fullName"],
                "team": abbrev[parent],
                "level": level,
                "pos": (entry.get("position") or {}).get("abbreviation", ""),
            }
    print(f"depth candidates: {len(players)} across {len(affiliates)} affiliates")

    warhist = snap.load_warhist()
    people = snap.load_people(sorted(players))
    today = date.today().isoformat()

    n_mlb_path = 0
    out = []
    for pid, rec in players.items():
        val = snap.value_mlb_at(int(pid), today, SEASON, people, warhist)
        if val is not None:
            n_mlb_path += 1
            # a minor-deal depth player carries no salary obligation an
            # acquirer must eat: floor at the unranked minimum
            rec.update({"value": max(val["value"], snap.FILLER_VALUE),
                        "warProj": val["warProj"],
                        "controlLeft": val["controlLeft"], "path": "mlbHistory"})
        else:
            rec.update({"value": snap.FILLER_VALUE, "path": "unranked"})
        out.append(rec)
    out.sort(key=lambda r: r["value"], reverse=True)

    OUT_PATH.write_text(json.dumps({
        "generated": date.today().isoformat(),
        "players": out,
    }, indent=1))
    print(f"Wrote {len(out)} depth players -> {OUT_PATH} "
          f"({n_mlb_path} valued off MLB history, rest at the floor)")
    print("top 10 by value:")
    for r in out[:10]:
        print(f'  {r["value"]/1e6:6.1f}M  {r["name"]:24} {r["team"]:4} '
              f'{r["level"]} ({r["path"]})')


if __name__ == "__main__":
    main()
