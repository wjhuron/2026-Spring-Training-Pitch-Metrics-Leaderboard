"""Historical MLB injured-list stints from statsapi transactions.

Monthly pulls of /api/v1/transactions (typeCode SC status changes), keeping
only MAJOR-league IL moves: '10-day/15-day/60-day injured list' plus the
pre-2019 'disabled list' wording. MiLB ILs (7-day) are excluded. Episodes
are per-player (placement -> activation); an open episode has end=None.

Output: data/tradevalue_il_hist.json
  {"episodes": {mlbam: [{"on": date, "off": date|null, "kind": "10|15|60"}]},
   "range": [start, end]}
"""

import json
import re
import time
import urllib.request
from datetime import date
from pathlib import Path

BASE = Path("/Users/wallyhuron/Huronalytics")
OUT = BASE / "data" / "tradevalue_il_hist.json"
UA = {"User-Agent": "huronalytics-tradevalue/1.0"}
START = date(2016, 3, 1)
END = date(2026, 7, 31)

PLACE_RE = re.compile(
    r"placed .* on the (10|15|60)-day (injured|disabled) list", re.I)
ACT_RE = re.compile(
    r"(activated|reinstated) .* from the (10|15|60)-day "
    r"(injured|disabled) list", re.I)


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


def month_starts():
    y, m = START.year, START.month
    while (y, m) <= (END.year, END.month):
        start = date(y, m, 1)
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        end = min(END, date(ny, nm, 1))
        yield start, end
        y, m = ny, nm


def main():
    place, act = [], []
    for start, end in month_starts():
        url = ("https://statsapi.mlb.com/api/v1/transactions"
               f"?startDate={start}&endDate={end}")
        try:
            txs = fetch(url).get("transactions", [])
        except Exception as e:
            print(f"{start}: FAILED ({e})")
            continue
        n = 0
        for t in txs:
            if t.get("typeCode") != "SC":
                continue
            desc = t.get("description") or ""
            pid = (t.get("person") or {}).get("id")
            d = t.get("date")
            if not (pid and d):
                continue
            mp = PLACE_RE.search(desc)
            ma = ACT_RE.search(desc)
            if mp:
                place.append((pid, d, mp.group(1)))
                n += 1
            elif ma:
                act.append((pid, d))
                n += 1
        print(f"{start}: {len(txs)} txs, {n} MLB IL moves")
        time.sleep(0.4)

    # assemble episodes per player
    place.sort(key=lambda x: (x[0], x[1]))
    act.sort()
    act_by_pid = {}
    for pid, d in act:
        act_by_pid.setdefault(pid, []).append(d)
    episodes = {}
    for pid, d_on, kind in place:
        offs = [x for x in act_by_pid.get(pid, []) if x > d_on]
        d_off = min(offs) if offs else None
        episodes.setdefault(str(pid), []).append(
            {"on": d_on, "off": d_off, "kind": kind})
    n_ep = sum(len(v) for v in episodes.values())
    OUT.write_text(json.dumps(
        {"episodes": episodes, "range": [str(START), str(END)]}))
    print(f"\n{n_ep} IL episodes for {len(episodes)} players -> {OUT}")


if __name__ == "__main__":
    main()
