"""Pull the full Washington Nationals org: rosters, bios, and year-by-year stats.

Builds one JSON blob per player covering:
  - bio (birth date, position, bats/throws, height/weight, draft, debut)
  - 2024/2025/2026 year-by-year hitting and pitching splits by team/level
  - career MLB totals (rookie-eligibility screen: 130 AB / 50 IP)

Usage: python3 scripts/tools/prospect_org_pull.py --out <dir>
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://statsapi.mlb.com/api/v1"
SEASON = 2026
AFFILIATES = {
    120: ("Washington Nationals", "MLB"),
    534: ("Rochester Red Wings", "AAA"),
    547: ("Harrisburg Senators", "AA"),
    426: ("Wilmington Blue Rocks", "High-A"),
    436: ("Fredericksburg Nationals", "Single-A"),
    466: ("FCL Nationals", "FCL"),
    1270: ("DSL Nationals", "DSL"),
}
EXCLUDE_NAMES = {"Christian Franklin"}
# Deadline acquisitions that must appear (sanity check after the pull)
MUST_HAVE = [
    "Will Dion", "Josh Hartle", "Kendeglys Virguez", "Nick Mitchell",
    "Joe Glassey", "Yovanny Cruz", "Jack Cebert", "Ben Grable",
]


def get(url, retries=3):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))


def pull_rosters():
    players = {}
    for team_id, (team_name, level) in AFFILIATES.items():
        for roster_type in ("fullRoster",):
            url = f"{BASE}/teams/{team_id}/roster?rosterType={roster_type}&season={SEASON}"
            data = get(url)
            for entry in data.get("roster", []):
                p = entry["person"]
                pid = p["id"]
                rec = players.setdefault(pid, {
                    "id": pid,
                    "name": p["fullName"],
                    "rosterTeams": [],
                })
                rec["rosterTeams"].append({
                    "team": team_name,
                    "level": level,
                    "status": entry.get("status", {}).get("description", ""),
                    "position": entry.get("position", {}).get("abbreviation", ""),
                })
            time.sleep(0.3)
    return players


def hydrate_people(players):
    ids = sorted(players)
    for chunk_start in range(0, len(ids), 40):
        chunk = ids[chunk_start:chunk_start + 40]
        idstr = ",".join(map(str, chunk))
        url = (f"{BASE}/people?personIds={idstr}"
               f"&hydrate=draft,currentTeam")
        data = get(url)
        for person in data.get("people", []):
            rec = players[person["id"]]
            rec["bio"] = {
                "birthDate": person.get("birthDate"),
                "currentAge": person.get("currentAge"),
                "height": person.get("height"),
                "weight": person.get("weight"),
                "primaryPosition": person.get("primaryPosition", {}).get("abbreviation"),
                "batSide": person.get("batSide", {}).get("code"),
                "pitchHand": person.get("pitchHand", {}).get("code"),
                "mlbDebutDate": person.get("mlbDebutDate"),
                "birthCountry": person.get("birthCountry"),
                "currentTeam": (person.get("currentTeam") or {}).get("name"),
            }
            drafts = person.get("drafts") or []
            if drafts:
                d = drafts[-1]
                rec["draft"] = {
                    "year": d.get("year"),
                    "round": d.get("pickRound"),
                    "pick": d.get("pickNumber"),
                    "school": (d.get("school") or {}).get("name"),
                }
        time.sleep(0.4)


SPORT_IDS = [1, 11, 12, 13, 14, 16]


def fetch_seasons(rec):
    """yearByYear stats: the API requires one call per (sportId, group)."""
    pos = (rec.get("bio") or {}).get("primaryPosition") or ""
    if pos == "P":
        groups = ["pitching"]
    elif pos in ("TWP", "", None):
        groups = ["hitting", "pitching"]
    else:
        groups = ["hitting"]
    seasons = []
    for sid in SPORT_IDS:
        for group in groups:
            url = (f"{BASE}/people/{rec['id']}/stats?stats=yearByYear"
                   f"&group={group}&sportId={sid}")
            try:
                data = get(url)
            except Exception:
                continue
            for statblock in data.get("stats", []):
                for split in statblock.get("splits", []):
                    team = split.get("team", {})
                    league = split.get("league", {})
                    seasons.append({
                        "season": split.get("season"),
                        "group": group,
                        "team": team.get("name"),
                        "teamId": team.get("id"),
                        "sportId": sid,
                        "league": league.get("name"),
                        "stat": split.get("stat", {}),
                    })
    rec["seasons"] = seasons
    return rec["id"]


def fetch_all_seasons(players):
    with ThreadPoolExecutor(max_workers=8) as ex:
        done = 0
        for _ in ex.map(fetch_seasons, players.values()):
            done += 1
            if done % 50 == 0:
                print(f"  stats fetched for {done}/{len(players)}")


def mlb_career_totals(rec):
    ab = 0
    outs = 0.0
    for s in rec.get("seasons", []):
        if s.get("sportId") != 1:
            continue
        st = s["stat"]
        if s["group"] == "hitting":
            ab += int(st.get("atBats") or 0)
        elif s["group"] == "pitching":
            ip = st.get("inningsPitched") or "0.0"
            whole, _, frac = str(ip).partition(".")
            outs += int(whole) * 3 + int(frac or 0)
    return ab, outs / 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    players = pull_rosters()
    print(f"rostered players: {len(players)}")
    hydrate_people(players)
    fetch_all_seasons(players)

    out = []
    for pid, rec in players.items():
        if rec["name"] in EXCLUDE_NAMES:
            continue
        ab, ip = mlb_career_totals(rec)
        rec["mlbCareerAB"] = ab
        rec["mlbCareerIP"] = round(ip, 1)
        rec["rookieEligibleByTotals"] = ab < 130 and ip < 50
        out.append(rec)

    names = {r["name"] for r in out}
    for want in MUST_HAVE:
        flag = "OK" if want in names else "MISSING"
        print(f"  acquisition check: {want}: {flag}")

    path = f"{args.out}/wsh_org_2026.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {len(out)} players -> {path}")


if __name__ == "__main__":
    main()
