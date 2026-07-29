"""Pull historical FanGraphs Board lists (FV grades) for at-trade-time values.

The public JSON endpoint is
  fangraphs.com/api/prospects/board/prospects-list?draft={year}{variant}&pos=all&type=0
with variant 'prospect' (preseason list) and 'updated' (in-season list).
No auth needed; the once-suspected Cloudflare block was a wrong API path.

Snapshot rule for trades: date before June 1 -> that year's preseason list;
June 1 onward (deadline + offseason) -> that year's updated list.

Output: data/tradevalue_board_hist.json
  {"2021updated": [{fgId, minorMasterId, name, fv, eta, risk, pos, age,
                    rank, org, birthDate}], ...}
"""

import json
import time
import urllib.request
from pathlib import Path

BASE = Path("/Users/wallyhuron/Huronalytics")
OUT_PATH = BASE / "data" / "tradevalue_board_hist.json"
YEARS = range(2017, 2027)
VARIANTS = ("prospect", "updated")
UA = {"User-Agent": "huronalytics-tradevalue/1.0"}


def fetch(url, tries=3):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def trim(row):
    return {
        "fgId": row.get("PlayerId"),
        "minorMasterId": row.get("minorMasterId"),
        "name": row.get("playerName"),
        "fv": str(row.get("cFV") or "").strip(),
        "eta": row.get("cETA"),
        "risk": row.get("cRisk"),
        "pos": row.get("Position"),
        "age": row.get("Age"),
        "rank": row.get("Ovr_Rank"),
        "org": row.get("Team"),
        "birthDate": row.get("BirthDate"),
    }


def main():
    out = {}
    for year in YEARS:
        for variant in VARIANTS:
            key = f"{year}{variant}"
            url = ("https://www.fangraphs.com/api/prospects/board/prospects-list"
                   f"?draft={key}&pos=all&type=0")
            try:
                rows = fetch(url)
            except Exception as e:
                print(f"{key}: FAILED ({e})")
                continue
            if not isinstance(rows, list) or not rows:
                print(f"{key}: empty/absent")
                continue
            out[key] = [trim(r) for r in rows]
            fvs = sum(1 for r in out[key] if r["fv"])
            print(f"{key}: {len(rows)} prospects ({fvs} with FV)")
            time.sleep(1)
    OUT_PATH.write_text(json.dumps(out, indent=1))
    print(f"\nWrote {len(out)} board lists -> {OUT_PATH}")


if __name__ == "__main__":
    main()
