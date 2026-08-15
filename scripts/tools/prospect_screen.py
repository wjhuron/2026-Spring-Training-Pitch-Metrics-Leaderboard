"""Flatten the org pull into a screening CSV: one row per player-season-level.

Usage: python3 scripts/tools/prospect_screen.py --src <dir>
Writes <dir>/screen_hitters.csv and <dir>/screen_pitchers.csv plus a
players.csv bio index.
"""
import argparse
import csv
import json

LEVEL_BY_SPORT = {1: "MLB", 11: "AAA", 12: "AA", 13: "A+", 14: "A", 16: "Rk", 17: "WIN", 22: "COL"}


def ip_to_float(ip):
    whole, _, frac = str(ip or "0.0").partition(".")
    return int(whole) + int(frac or 0) / 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    args = ap.parse_args()
    players = json.load(open(f"{args.src}/wsh_org_2026.json"))

    bio_rows, hit_rows, pit_rows = [], [], []
    for p in players:
        bio = p.get("bio", {})
        draft = p.get("draft", {})
        levels_2026 = sorted({t["level"] for t in p["rosterTeams"]})
        bio_rows.append({
            "id": p["id"], "name": p["name"], "pos": bio.get("primaryPosition"),
            "age": bio.get("currentAge"), "birthDate": bio.get("birthDate"),
            "bats": bio.get("batSide"), "throws": bio.get("pitchHand"),
            "height": bio.get("height"), "weight": bio.get("weight"),
            "country": bio.get("birthCountry"),
            "draftYear": draft.get("year"), "draftRound": draft.get("round"),
            "draftPick": draft.get("pick"), "school": draft.get("school"),
            "mlbDebut": bio.get("mlbDebutDate"),
            "mlbAB": p["mlbCareerAB"], "mlbIP": p["mlbCareerIP"],
            "rookieEligible": p["rookieEligibleByTotals"],
            "rosterLevels": "/".join(levels_2026),
            "currentTeam": bio.get("currentTeam"),
        })
        for s in p.get("seasons", []):
            if s["season"] not in ("2024", "2025", "2026"):
                continue
            lvl = LEVEL_BY_SPORT.get(s.get("sportId"), s.get("sport"))
            st = s["stat"]
            base = {"id": p["id"], "name": p["name"], "age": bio.get("currentAge"),
                    "season": s["season"], "level": lvl, "team": s.get("team")}
            if s["group"] == "hitting":
                pa = int(st.get("plateAppearances") or 0)
                if pa == 0:
                    continue
                so = int(st.get("strikeOuts") or 0)
                bb = int(st.get("baseOnBalls") or 0)
                hit_rows.append({**base,
                    "PA": pa, "AVG": st.get("avg"), "OBP": st.get("obp"),
                    "SLG": st.get("slg"), "OPS": st.get("ops"),
                    "HR": st.get("homeRuns"), "SB": st.get("stolenBases"),
                    "CS": st.get("caughtStealing"),
                    "BBpct": round(100 * bb / pa, 1), "Kpct": round(100 * so / pa, 1),
                    "BABIP": st.get("babip"), "doubles": st.get("doubles"),
                    "triples": st.get("triples"), "HBP": st.get("hitByPitch"),
                })
            elif s["group"] == "pitching":
                bf = int(st.get("battersFaced") or 0)
                if bf == 0:
                    continue
                so = int(st.get("strikeOuts") or 0)
                bb = int(st.get("baseOnBalls") or 0)
                ip = ip_to_float(st.get("inningsPitched"))
                pit_rows.append({**base,
                    "G": st.get("gamesPlayed"), "GS": st.get("gamesStarted"),
                    "IP": round(ip, 1), "BF": bf, "ERA": st.get("era"),
                    "Kpct": round(100 * so / bf, 1), "BBpct": round(100 * bb / bf, 1),
                    "K9": st.get("strikeoutsPer9Inn"), "BB9": st.get("walksPer9Inn"),
                    "HR": st.get("homeRuns"), "H9": st.get("hitsPer9Inn"),
                    "WHIP": st.get("whip"), "AVGagainst": st.get("avg"),
                    "GOAO": st.get("groundOutsToAirouts"), "SV": st.get("saves"),
                    "HBP": st.get("hitByPitch"), "WP": st.get("wildPitches"),
                })

    for fname, rows in (("players.csv", bio_rows), ("screen_hitters.csv", hit_rows),
                        ("screen_pitchers.csv", pit_rows)):
        if not rows:
            continue
        with open(f"{args.src}/{fname}", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"{fname}: {len(rows)} rows")


if __name__ == "__main__":
    main()
