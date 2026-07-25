"""Robustness checks before acting on round-2 findings.

Round 2 suggested (a) the per-player sigma scale might help and (b) a lower
SKILL_PREF might predict better. Both came from a single split on ~43 players,
where the standard error of a correlation is roughly 0.15 -- big enough to
manufacture either result. This re-tests both properly:

  * sigma scale: split each player into THIRDS, fit the scale on the first
    third only, and score the second and third independently. That removes the
    cross-contamination in the round-2 test, where each half's score was
    computed with a scale derived from the other half.

  * SKILL_PREF: swept across several pool definitions and both split types,
    so a peak has to survive specification changes to count.

Usage: python3 scripts/abs_research_round3.py
"""

import json
import os
import random
import statistics
from collections import defaultdict

from scipy.stats import pearsonr

from abs_research_round2 import balanced, build, fit_sigma, pool

PREF = 0.434


def main():
    per, o = build()
    lsig = o["perceptionPooled"]["fld"]["sigma"]

    # ---------------------------------------------------------------- sigma
    print("=== 1. Per-player sigma scale, contamination-free (thirds) ===")
    print("    fit the scale on third 1 only; correlate scores on thirds 2 and 3")
    pl = pool(per, "fld", 0.05, 2.5, 300)
    print(f"    catchers with >=300 decisions: {len(pl)}")
    rows = {"no scale": ([], []), "scale (fit on third 1)": ([], [])}
    for evs in pl.values():
        k = len(evs) // 3
        t1, t2, t3 = evs[:k], evs[k:2 * k], evs[2 * k:]
        sc = lsig / fit_sigma(t1, lsig)
        for label, s in (("no scale", 1.0), ("scale (fit on third 1)", sc)):
            a, b = balanced(t2, o, scale=s), balanced(t3, o, scale=s)
            if a is not None and b is not None:
                rows[label][0].append(a)
                rows[label][1].append(b)
    for label, (xs, ys) in rows.items():
        if len(xs) >= 8:
            r, p = pearsonr(xs, ys)
            print(f"  {label:<26} r={r:+.3f} (p={p:.3f}) n={len(xs)}")
    print("  -> the scale only earns its place if it clearly beats 'no scale' here")

    # ------------------------------------------------------------ SKILL_PREF
    print("\n=== 2. SKILL_PREF: does a peak survive changing the specification? ===")
    prefs = (0.30, 0.35, 0.38, 0.40, 0.434, 0.47, 0.50)
    specs = [("g>=.05 m<=2.5 min240", 0.05, 2.5, 240),
             ("g>=.05 m<=2.5 min300", 0.05, 2.5, 300),
             ("g>=.03 m<=2.5 min240", 0.03, 2.5, 240),
             ("g>=.05 m<=3.0 min240", 0.05, 3.0, 240),
             ("g>=.05 m<=2.0 min200", 0.05, 2.0, 200)]
    print("    spec                    " + "".join(f"{p:>8.3f}" for p in prefs))
    tally = defaultdict(list)
    for name, gm, mm, md in specs:
        p2 = pool(per, "fld", gm, mm, md)
        line = f"    {name:<22}"
        for pref in prefs:
            xs, ys = [], []
            for evs in p2.values():
                h = len(evs) // 2
                a, b = balanced(evs[:h], o, pref=pref), balanced(evs[h:], o, pref=pref)
                if a is not None and b is not None:
                    xs.append(a); ys.append(b)
            r = pearsonr(xs, ys)[0] if len(xs) >= 8 else float("nan")
            tally[pref].append(r)
            line += f"{r:>8.3f}"
        print(line + f"   (n={len(p2)})")
    print("    " + "-" * 22 + "-" * (8 * len(prefs)))
    print("    mean across specs     " + "".join(f"{statistics.mean(tally[p]):>8.3f}" for p in prefs))
    print("    worst spec            " + "".join(f"{min(tally[p]):>8.3f}" for p in prefs))

    # ------------------------------------------------- random-split stability
    print("\n=== 3. Same sweep on RANDOM splits (context-shared upper bound) ===")
    rng = random.Random(7)
    p2 = pool(per, "fld", 0.05, 2.5, 240)
    line = "    random split          "
    for pref in prefs:
        xs, ys = [], []
        for evs in p2.values():
            sh = rng.sample(evs, len(evs))
            h = len(sh) // 2
            a, b = balanced(sh[:h], o, pref=pref), balanced(sh[h:], o, pref=pref)
            if a is not None and b is not None:
                xs.append(a); ys.append(b)
        line += f"{pearsonr(xs,ys)[0]:>8.3f}"
    print("    spec                    " + "".join(f"{p:>8.3f}" for p in prefs))
    print(line)


if __name__ == "__main__":
    main()
