"""Historical season WAR + salary per player from Baseball-Reference war_daily.

Downloads war_daily_bat.txt and war_daily_pitch.txt (free, ~30MB combined),
trims to what the trade-corpus valuation needs, and writes
data/tradevalue_warhist.csv with:
  mlbam, season, age, warBat, warPit, salary

Seasons kept: 2013+ (three-year lookback for 2017 trades). The current
season's row is season-to-date. bWAR here vs fWAR in the live engine is an
accepted inconsistency: the market layer only compares packages valued the
same way as each other.
"""

import csv
import io
import urllib.request
from pathlib import Path

BASE = Path("/Users/wallyhuron/Huronalytics")
OUT_PATH = BASE / "data" / "tradevalue_warhist.csv"
MIN_SEASON = 2013
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
URLS = {
    "bat": "https://www.baseball-reference.com/data/war_daily_bat.txt",
    "pit": "https://www.baseball-reference.com/data/war_daily_pitch.txt",
}


def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read().decode("utf-8", errors="replace")


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    # (mlbam, season) -> {age, warBat, warPit, salary}
    table = {}
    for kind, url in URLS.items():
        text = fetch(url)
        n = 0
        for row in csv.DictReader(io.StringIO(text)):
            year = row.get("year_ID")
            mlbam = row.get("mlb_ID")
            if not (year and year.isdigit() and int(year) >= MIN_SEASON):
                continue
            if not (mlbam and mlbam.isdigit()):
                continue
            key = (int(mlbam), int(year))
            rec = table.setdefault(key, {
                "age": None, "warBat": None, "warPit": None, "salary": None,
            })
            war = to_float(row.get("WAR"))
            if war is not None:  # sum stints
                field = "warBat" if kind == "bat" else "warPit"
                rec[field] = (rec[field] or 0.0) + war
            if rec["age"] is None:
                rec["age"] = row.get("age") if (row.get("age") or "").isdigit() else None
            sal = to_float(row.get("salary"))
            if sal is not None and sal > 0 and rec["salary"] is None:
                rec["salary"] = int(sal)
            n += 1
        print(f"{kind}: {n} rows kept")

    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mlbam", "season", "age", "warBat", "warPit", "salary"])
        for (mlbam, season), rec in sorted(table.items()):
            w.writerow([mlbam, season, rec["age"],
                        rec["warBat"], rec["warPit"], rec["salary"]])
    print(f"Wrote {len(table)} player-seasons -> {OUT_PATH}")


if __name__ == "__main__":
    main()
