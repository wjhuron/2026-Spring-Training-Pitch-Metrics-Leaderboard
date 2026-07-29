"""Diagnostic: does the arb ladder (15/35/50/75) misprice by role? (2026-07-29)

Ran against 573 arb player-seasons 2016-2025 (warhist actual salaries vs
ladder x Marcel-market prediction). Result, recorded here as the receipt for
SKIPPING an empirical arb model:

  role arbYr    n   median actual/pred
  POS      1   51   0.85
  POS      2  124   0.64
  POS      3  158   0.58
  RP       1   11   1.76   <- the counting-stat (saves) effect, tiny n
  RP       2   34   0.95
  RP       3   38   0.96
  SP       1   23   0.93
  SP       2   60   0.55
  SP       3   74   0.69

Reading: the across-the-board sub-1.0 ratios for POS/SP are mostly a
currency artifact (historical salaries vs 2026 $/WAR rates), not ladder
error. The real signal is RELATIVE: relievers' arb-1 awards run ~2x the
POS/SP level, worth roughly $2-4M per closer over a full arb run after
decay/discount - below the $5M materiality bar, on a small archetype, and a
clean fix needs era-adjusted $/WAR. Verdict: not worth building; revisit
only if the model starts pricing individual closers' arb years.
"""

import sys
import json
import statistics
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tradevalue_snapshot as snap
from tradevalue_engine import CONFIG, rate_for


def main():
    warhist = snap.load_warhist()
    people = json.loads(snap.PEOPLE_CACHE.read_text())
    rows = []
    for pid, per in people.items():
        debut = per.get("debut")
        if not debut:
            continue
        dy = int(debut[:4])
        hist = warhist.get(int(pid), {})
        for s in range(2016, 2026):
            if s == 2020:
                continue
            service = s - dy
            if service not in (3, 4, 5):
                continue
            rec = hist.get(s)
            if not rec or not rec.get("salary") or rec["salary"] < 700_000:
                continue
            birth = per.get("birthDate")
            age = s - int(birth[:4]) if birth else None
            war_proj = snap.marcel(hist, s, age)
            if war_proj <= 0.5:
                continue
            arb_year = service - 2
            pred = max(780_000,
                       CONFIG["arbLadder"][arb_year] * rate_for(war_proj) * war_proj)
            g, gs = rec.get("gPit", 0), rec.get("gsPit", 0)
            if per.get("pitcher"):
                role = "RP" if (g >= 8 and gs / g < 0.3) else "SP"
            else:
                role = "POS"
            rows.append((role, arb_year, rec["salary"], pred))
    by = defaultdict(list)
    for role, k, actual, pred in rows:
        by[(role, k)].append(actual / pred)
    print(f"sample: {len(rows)}")
    for (role, k), ratios in sorted(by.items()):
        print(f"{role:4} arb{k}  n={len(ratios):4}  "
              f"median actual/pred={statistics.median(ratios):.2f}")


if __name__ == "__main__":
    main()
