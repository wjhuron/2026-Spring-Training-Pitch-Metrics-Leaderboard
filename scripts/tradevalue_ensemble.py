"""Build ensemble RoS projections from multiple FanGraphs system exports.

Consensus projections beat any single system (the most replicated result in
forecasting); this averages every system present per player with EQUAL
weights (the standard no-tuning choice) and records the cross-system spread.

Input: data/proj_raw/ros_{bat|pit}_*.csv (raw FG exports, one per system).
Output: data/tradevalue_fg_{bat|pit}.csv in the exact shape the universe
builder reads (Name/Team/G/GS/WAR/PlayerId/MLBAMID), plus nSys and warSd
columns (per-player system count and projection disagreement - the latter
feeds uncertainty display later).

2027/2028 full-season exports are stored at data/proj_raw/full_*.csv for a
future validated test of per-player decay paths; they are NOT wired into
values (the pooled 21%/yr decay is the fitted, validated path).
"""

import csv
import glob
import statistics
import sys
from pathlib import Path

BASE = Path("/Users/wallyhuron/Huronalytics")
RAW = BASE / "data" / "proj_raw"
OUT = {"bat": BASE / "data" / "tradevalue_fg_bat.csv",
       "pit": BASE / "data" / "tradevalue_fg_pit.csv"}


def build(kind):
    files = sorted(glob.glob(str(RAW / f"ros_{kind}_*.csv")))
    if not files:
        sys.exit(f"no raw files for {kind} in {RAW}")
    players = {}  # mlbam -> {name, team, fgId, wars[], gs[], g[]}
    for f in files:
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            m = (r.get("MLBAMID") or "").strip()
            if not m:
                continue
            p = players.setdefault(m, {
                "name": r["Name"], "team": r.get("Team", ""),
                "fgId": r.get("PlayerId", ""), "wars": [], "g": [], "gs": [],
            })
            if r.get("WAR"):
                p["wars"].append(float(r["WAR"]))
            if kind == "pit":
                if r.get("G"):
                    p["g"].append(float(r["G"]))
                if r.get("GS"):
                    p["gs"].append(float(r["GS"]))

    cols = ["Name", "Team", "G", "GS", "WAR", "PlayerId", "MLBAMID",
            "nSys", "warSd"]
    n_out = 0
    with open(OUT[kind], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for m, p in players.items():
            if not p["wars"]:
                continue
            war = statistics.mean(p["wars"])
            sd = statistics.pstdev(p["wars"]) if len(p["wars"]) > 1 else ""
            g = statistics.mean(p["g"]) if p["g"] else ""
            gs = statistics.mean(p["gs"]) if p["gs"] else ""
            w.writerow([p["name"], p["team"], g, gs, round(war, 5),
                        p["fgId"], m, len(p["wars"]),
                        round(sd, 5) if sd != "" else ""])
            n_out += 1
    counts = {}
    for p in players.values():
        counts[len(p["wars"])] = counts.get(len(p["wars"]), 0) + 1
    print(f"{kind}: {len(files)} systems -> {n_out} players "
          f"(by system count: {dict(sorted(counts.items()))})")


if __name__ == "__main__":
    build("bat")
    build("pit")
