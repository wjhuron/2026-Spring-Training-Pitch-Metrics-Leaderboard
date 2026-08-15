"""Pull historical PRESEASON Steamer projections from the FanGraphs API.

The public projections API serves archived preseason finals under
year-suffixed type tokens: type=steamer_{year} (underscore required;
'steamer2019' 500s, which caused an earlier false dead-end). Current-season
tokens ('steamer', 'rsteamer') are a different, rolling payload.

Anti-bot note: Cloudflare challenges browser-shaped UAs on this API but
passes a plain tool UA (same pattern as the Anubis block on legacy BP).

Output: data/steamer_preseason/steamer_{year}_{bat|pit}.csv with
mlbam, fgid, name, war, pa (bat) / ip, gs, g (pit). WAR is the full-season
preseason projection - the at-trade-time talent estimate for corpus trades.
"""

import csv
import json
import time
import urllib.request
from pathlib import Path

BASE = Path("/Users/wallyhuron/Huronalytics")
OUT_DIR = BASE / "data" / "steamer_preseason"
YEARS = range(2017, 2027)
UA = {"User-Agent": "huronalytics-tradevalue/1.0", "Accept": "application/json"}


def fetch(url, tries=3):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))


SYSTEMS = ("steamer", "zips")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    for system in SYSTEMS:
        for year in YEARS:
            for stats in ("bat", "pit"):
                pull(system, year, stats)


def pull(system, year, stats):
            out_path = OUT_DIR / f"{system}_{year}_{stats}.csv"
            if out_path.exists():
                print(f"{system} {year} {stats}: exists, skipping")
                return
            url = ("https://www.fangraphs.com/api/projections"
                   f"?type={system}_{year}&stats={stats}"
                   "&pos=all&team=0&players=0&lg=all")
            try:
                rows = fetch(url)
            except Exception as e:
                print(f"{system} {year} {stats}: FAILED ({e})")
                return
            if not isinstance(rows, list) or not rows:
                print(f"{system} {year} {stats}: empty/absent")
                return
            n_id = 0
            with open(out_path, "w", newline="") as f:
                if stats == "bat":
                    w = csv.writer(f)
                    w.writerow(["mlbam", "fgid", "name", "war", "pa"])
                    for r in rows:
                        mlbam = r.get("xMLBAMID") or ""
                        n_id += bool(mlbam)
                        w.writerow([mlbam, r.get("playerid"),
                                    r.get("PlayerName"),
                                    r.get("WAR"), r.get("PA")])
                else:
                    w = csv.writer(f)
                    w.writerow(["mlbam", "fgid", "name", "war", "ip",
                                "gs", "g"])
                    for r in rows:
                        mlbam = r.get("xMLBAMID") or ""
                        n_id += bool(mlbam)
                        w.writerow([mlbam, r.get("playerid"),
                                    r.get("PlayerName"),
                                    r.get("WAR"), r.get("IP"),
                                    r.get("GS"), r.get("G")])
            print(f"{system} {year} {stats}: {len(rows)} rows ({n_id} with mlbam) "
                  f"-> {out_path.name}")
            time.sleep(2)


if __name__ == "__main__":
    main()
