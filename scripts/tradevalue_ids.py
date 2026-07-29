"""Build the cross-source player ID map from the Chadwick Bureau register.

The register (github.com/chadwickbureau/register) splits its people table into
16 hex-suffixed CSVs. We keep every row holding an MLBAM id plus either a FanGraphs/bbref id or
recent pro activity (pro_played_last >= 2020) — the latter covers current
minor leaguers who have MLBAM ids but no FG/bbref keys yet.

Output: data/tradevalue_idmap.csv with
  mlbam, fangraphs, bbref, retro, last, first, birthYear
"""

import csv
import io
import urllib.request
from pathlib import Path

BASE = Path("/Users/wallyhuron/Huronalytics")
OUT_PATH = BASE / "data" / "tradevalue_idmap.csv"
REGISTER = (
    "https://raw.githubusercontent.com/chadwickbureau/register/master/data/"
    "people-{}.csv"
)
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
KEEP = ["key_mlbam", "key_fangraphs", "key_bbref", "key_retro",
        "name_last", "name_first", "birth_year", "pro_played_last"]
OUT_COLS = ["mlbam", "fangraphs", "bbref", "retro", "last", "first",
            "birthYear", "proLast"]


def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8", errors="replace")


def main():
    rows_out = []
    for suffix in "0123456789abcdef":
        text = fetch(REGISTER.format(suffix))
        reader = csv.DictReader(io.StringIO(text))
        kept = 0
        for row in reader:
            if not row.get("key_mlbam"):
                continue
            pro_last = row.get("pro_played_last", "")
            recent = pro_last.isdigit() and int(pro_last) >= 2020
            if not (row.get("key_fangraphs") or row.get("key_bbref") or recent):
                continue
            rows_out.append([row.get(k, "") for k in KEEP])
            kept += 1
        print(f"people-{suffix}: kept {kept}")
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(OUT_COLS)
        w.writerows(rows_out)
    print(f"Wrote {len(rows_out)} id rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
