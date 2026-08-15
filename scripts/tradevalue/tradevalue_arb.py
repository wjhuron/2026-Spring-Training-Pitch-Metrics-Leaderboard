"""Scrape MLB Trade Rumors arbitration salary projections (Swartz model).

MLBTR publishes the full arb-eligible class each October as
"Projected Arbitration Salaries For {season}". Page structure:
  <span style="text-decoration: underline;">Angels (10)</span>
  <ul><li><strong><a href="{bbref-url}">Taylor Ward</a></strong> (5.164): $13.7MM</li>...

We keep the bbref id from the player link; it feeds the ID map.
Output: data/tradevalue_arb.json.
"""

import json
import re
import urllib.request
from datetime import date
from pathlib import Path

BASE = Path("/Users/wallyhuron/Huronalytics")
OUT_PATH = BASE / "data" / "tradevalue_arb.json"
SEASON = 2026
# published Oct 2025 for the 2026 season; update the year segments annually
ARB_URL = (
    "https://www.mlbtraderumors.com/2025/10/"
    "projected-arbitration-salaries-for-2026.html"
)
TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams?sportId=1"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def club_abbrevs():
    """MLBTR team headers use the club name ('Angels') -> abbreviation."""
    data = json.loads(fetch(TEAMS_URL))
    clubs = {t["teamName"]: t["abbreviation"] for t in data["teams"]}
    clubs["Diamondbacks"] = clubs.get("D-backs", "AZ")  # MLBTR uses the long name
    return clubs


def parse_salary(s):
    m = re.match(r"\$([\d.]+)\s*(MM|M|K)", s.strip())
    if not m:
        return None
    v = float(m.group(1))
    return v * (1_000_000 if m.group(2) in ("MM", "M") else 1_000)


def main():
    page = fetch(ARB_URL)
    clubs = club_abbrevs()
    # Team headers are "{Club} (N)" but the markup varies (underlined span for
    # most, bare text for some), so match on the club names themselves.
    club_alt = "|".join(
        re.escape(c) for c in sorted(clubs, key=len, reverse=True)
    )
    tokens = re.findall(
        r'>\s*(' + club_alt + r')\s*\(\d+\)\s*<'
        r'|<li><strong><a href="https://www\.baseball-reference\.com/players/'
        r'[a-z]/([a-z0-9.]+)\.shtml[^"]*"[^>]*>([^<]+)</a></strong>'
        r'\s*\(([\d.]+)\):\s*(\$[\d.]+\s*(?:MM|M|K))',
        page,
    )
    players, team, unmatched_teams = [], None, set()
    for club, bbref, name, mls, salary in tokens:
        if club:
            team = clubs.get(club.strip())
            if team is None:
                unmatched_teams.add(club.strip())
        else:
            players.append({
                "name": name.strip(),
                "bbrefId": bbref,
                "team": team,
                "mls": mls,
                "projSalary": parse_salary(salary),
            })
    if unmatched_teams:
        print(f"WARNING: unmatched team headers: {sorted(unmatched_teams)}")
    missing = [p["name"] for p in players if p["team"] is None or p["projSalary"] is None]
    if missing:
        print(f"WARNING: {len(missing)} entries missing team/salary: {missing[:10]}")
    out = {
        "fetched": date.today().isoformat(),
        "season": SEASON,
        "source": ARB_URL,
        "players": players,
    }
    OUT_PATH.write_text(json.dumps(out, indent=1))
    teams_seen = len({p["team"] for p in players if p["team"]})
    print(f"Wrote {len(players)} arb projections across {teams_seen} teams -> {OUT_PATH}")


if __name__ == "__main__":
    main()
