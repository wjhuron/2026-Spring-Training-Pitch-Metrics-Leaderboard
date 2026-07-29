"""Backfill the MLB trade corpus for the trade value market layer.

Pulls every typeCode=TR transaction from the MLB Stats API (sportId=1),
2017 through the present, and groups rows into trades: the API emits one row
per player with person.id / fromTeam / toTeam and a shared description, so
(date, description) reconstructs the deal and toTeam groups its sides.

Output: data/tradevalue_trades.json
  [{date, season, deadline, description, teams: [teamId,...],
    players: [{mlbam, name, fromTeamId, toTeamId}],
    flags: [cash|ptbnl|intlBonus|draftPick]}]
"""

import json
import re
import time
import urllib.request
from datetime import date
from pathlib import Path

BASE = Path("/Users/wallyhuron/Huronalytics")
OUT_PATH = BASE / "data" / "tradevalue_trades.json"
START_YEAR = 2017
UA = {"User-Agent": "huronalytics-tradevalue/1.0"}

FLAG_PATTERNS = {
    "cash": r"\bcash\b",
    "ptbnl": r"player to be named",
    "intlBonus": r"international",
    "draftPick": r"(?:Competitive Balance|draft pick|Round [AB])",
}


def fetch(url, tries=3):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def pull_rows():
    rows = []
    today = date.today()
    for year in range(START_YEAR, today.year + 1):
        for half in ((f"{year}-01-01", f"{year}-06-30"),
                     (f"{year}-07-01", f"{year}-12-31")):
            url = ("https://statsapi.mlb.com/api/v1/transactions"
                   f"?startDate={half[0]}&endDate={half[1]}&sportId=1")
            d = fetch(url)
            got = [t for t in d.get("transactions", []) if t.get("typeCode") == "TR"]
            rows.extend(got)
            print(f"{half[0]}..{half[1]}: {len(got)} trade rows")
    return rows


def main():
    rows = pull_rows()
    trades = {}
    for t in rows:
        desc = t.get("description") or ""
        key = (t.get("date"), desc)
        tr = trades.setdefault(key, {
            "date": t.get("date"),
            "season": int(t["date"][:4]) if t.get("date") else None,
            "description": desc,
            "players": [],
            "flags": sorted(
                f for f, pat in FLAG_PATTERNS.items()
                if re.search(pat, desc, re.I)
            ),
        })
        person = t.get("person") or {}
        if person.get("id") and not any(
            p["mlbam"] == person["id"] for p in tr["players"]
        ):
            tr["players"].append({
                "mlbam": person["id"],
                "name": person.get("fullName"),
                "fromTeamId": (t.get("fromTeam") or {}).get("id"),
                "toTeamId": (t.get("toTeam") or {}).get("id"),
            })

    out = []
    deadline_windows = {}  # season -> (start, end) of deadline month proxy
    for (dt, desc), tr in sorted(trades.items()):
        teams = sorted({p["toTeamId"] for p in tr["players"] if p["toTeamId"]})
        tr["teams"] = teams
        month_day = tr["date"][5:] if tr["date"] else ""
        tr["deadline"] = "06-15" <= month_day <= "07-31"
        out.append(tr)

    OUT_PATH.write_text(json.dumps(out, indent=1))
    n_deadline = sum(1 for t in out if t["deadline"])
    n_multi = sum(1 for t in out if len(t["players"]) > 1)
    seasons = {}
    for t in out:
        seasons[t["season"]] = seasons.get(t["season"], 0) + 1
    print(f"\nWrote {len(out)} trades ({n_multi} multi-player, "
          f"{n_deadline} in deadline window) -> {OUT_PATH}")
    print("per season:", dict(sorted(seasons.items())))


if __name__ == "__main__":
    main()
