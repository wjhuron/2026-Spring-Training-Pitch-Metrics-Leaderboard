#!/usr/bin/env python3
"""Regenerate data/abs_hitter_zones_2026.json from the challenge dataset.

Reconstruction (2026-07-29) of the generator behind the file added data-only
in commit 9292540: per hitter, the MODAL (szTop, szBot) pair across all their
records in data/abs_challenges_2026.json — the feed's fixed height-based ABS
zone is constant per batter all game, so the mode across games is the
hitter's zone; n = total records for that hitter.

Consumed by js/abs.js (zone overlay) and scripts/abs_matrix_page.py.

Usage:
    python3 scripts/abs_hitter_zones.py            # dry run: compare vs shipped file
    python3 scripts/abs_hitter_zones.py --apply    # overwrite the shipped file
"""
import json
import os
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHALLENGES = os.path.join(REPO_ROOT, "data", "abs_challenges_2026.json")
OUT = os.path.join(REPO_ROOT, "data", "abs_hitter_zones_2026.json")


def build():
    with open(CHALLENGES) as f:
        records = json.load(f)["records"]

    per_hitter = defaultdict(Counter)
    names = {}
    for r in records:
        bid, top, bot = r.get("batterId"), r.get("szTop"), r.get("szBot")
        if bid is None or top is None or bot is None:
            continue
        per_hitter[bid][(top, bot)] += 1
        names[bid] = r.get("batter")

    zones = {}
    for bid, counts in per_hitter.items():
        (top, bot), _ = counts.most_common(1)[0]
        zones[str(bid)] = {"name": names[bid], "szTop": top, "szBot": bot,
                           "n": sum(counts.values())}
    return {"meta": {"source": "feed ABS zones, modal per hitter",
                     "hitters": len(zones)},
            "zones": zones}


def main():
    apply_mode = "--apply" in sys.argv
    new = build()

    if os.path.exists(OUT):
        with open(OUT) as f:
            old = json.load(f)
        old_z, new_z = old.get("zones", {}), new["zones"]
        shared = set(old_z) & set(new_z)
        same = sum(1 for k in shared
                   if old_z[k]["szTop"] == new_z[k]["szTop"]
                   and old_z[k]["szBot"] == new_z[k]["szBot"])
        print(f"shipped: {len(old_z)} hitters | rebuilt: {len(new_z)} hitters")
        print(f"of {len(shared)} shared: {same} identical zones, "
              f"{len(shared) - same} changed, "
              f"{len(new_z) - len(shared)} new hitters")

    if apply_mode:
        with open(OUT, "w") as f:
            json.dump(new, f, separators=(",", ":"))
        print(f"wrote {OUT}: {new['meta']['hitters']} hitters")
    else:
        print("dry run — pass --apply to overwrite")


if __name__ == "__main__":
    main()
