"""IL discount fit: does being on the injured list at midseason predict
realized-WAR shortfall beyond the pooled decay?

The engine displays IL status but applies no discount (the original
decision: no INVENTED medical discounts). This fits one from realized
outcomes, two-stage-doctrine style (value constants never fit on trade
balance):

    pred(next season) = war1 * (1 - lambda) * (1 - d * onIL)

where war1 is the preseason Steamer/ZiPS blend, lambda the shipped 0.21,
and onIL = on a 10/15/60-day MLB IL on July 31 of season s. A second knob
splits by severity (60-day vs 10/15-day). Sample: all projected player-
seasons 2017-2025 (2020 skipped), NOT just traded players.

PRE-REGISTERED GATES: d must bracket an interior optimum (or the curve be
shown flat), and the fitted d must beat d=0 in a majority of held-out
seasons. Report-only: nothing here touches the engine.
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import tradevalue_snapshot as snap
from tradevalue_engine import CONFIG

SEASONS = [y for y in range(2017, 2025) if y != 2020]  # target = s+1 realized
LAM = CONFIG["riskDecay"]["POS"]
GRID_D = [round(0.05 * i, 2) for i in range(0, 13)]  # 0.00 .. 0.60


def load_il():
    raw = json.loads((snap.DATA / "tradevalue_il_hist.json").read_text())
    return {int(k): v for k, v in raw["episodes"].items()}


def on_il(episodes, ref):
    """(onIL, kind) at ref date. Episodes with no recorded activation
    (offseason roll-offs, releases) are capped at Nov 30 of the placement
    year - otherwise one un-closed stint reads as on-IL forever."""
    for ep in episodes:
        off = ep["off"] or (ep["on"][:4] + "-11-30")
        if ep["on"] <= ref < off:
            return True, ep["kind"]
    return False, None


def build_sample():
    ctx = snap.load_context()
    steamer = snap.load_steamer()
    il = load_il()
    rows = []
    for mlbam, hist in ctx["warhist"].items():
        eps = il.get(mlbam, [])
        for s in SEASONS:
            if s not in hist:
                continue
            war1 = steamer.get((s, mlbam))
            if war1 is None:
                continue
            ref = f"{s}-07-31"
            flag, kind = on_il(eps, ref)
            realized = (hist.get(s + 1) or {}).get("war", 0.0)
            rows.append({"season": s, "war1": max(0.0, war1),
                         "onIL": flag, "kind": kind, "real": realized})
    n_il = sum(1 for r in rows if r["onIL"])
    n_60 = sum(1 for r in rows if r["kind"] == "60")
    print(f"sample: {len(rows)} player-seasons, {n_il} on IL at Jul 31 "
          f"({n_60} on the 60-day)")
    return rows


def mse(sub, d, d60=None):
    err = 0.0
    for r in sub:
        disc = 1.0
        if r["onIL"]:
            disc = 1 - (d60 if (d60 is not None and r["kind"] == "60") else d)
        pred = r["war1"] * (1 - LAM) * max(0.0, disc)
        err += (pred - r["real"]) ** 2
    return err / len(sub)


def main():
    rows = build_sample()
    il_rows = [r for r in rows if r["onIL"]]

    # descriptive: realized/projection ratio, IL vs healthy
    for grp, name in ((il_rows, "on IL Jul 31"),
                      ([r for r in rows if not r["onIL"]], "healthy"),
                      ([r for r in il_rows if r["kind"] == "60"], "60-day"),
                      ([r for r in il_rows if r["kind"] != "60"], "10/15-day")):
        p = sum(r["war1"] * (1 - LAM) for r in grp)
        a = sum(r["real"] for r in grp)
        print(f"  {name:12} n={len(grp):6}  realized/projected "
              f"{a / p if p else 0:.3f}")

    # single-knob sweep (fit on IL rows' error only would ignore the
    # healthy anchor; full-sample MSE is dominated by healthy rows whose
    # error is d-invariant, so sweep on IL rows - identical argmin, more
    # readable curve)
    print("\nsingle discount d (IL rows MSE):")
    curve = [(d, mse(il_rows, d)) for d in GRID_D]
    d_best, m_best = min(curve, key=lambda c: c[1])
    pos = GRID_D.index(d_best)
    status = "interior" if 0 < pos < len(GRID_D) - 1 else "GRID EDGE"
    for d, m in curve:
        print(f"  d={d:.2f}  mse={m:.4f}" + ("  <--" if d == d_best else ""))
    print(f"pooled optimum d={d_best} ({status})")

    # severity split
    print("\nseverity split (d10/15, d60):")
    best2, m2 = None, None
    for d1 in GRID_D:
        for d6 in GRID_D:
            m = mse(il_rows, d1, d6)
            if m2 is None or m < m2:
                best2, m2 = (d1, d6), m
    print(f"  best (d10/15={best2[0]}, d60={best2[1]})  mse={m2:.4f} "
          f"vs single {m_best:.4f}")

    # season-out validation vs d=0 (and split vs single)
    wins, wins2, tot = 0, 0, 0
    for s in SEASONS:
        train = [r for r in il_rows if r["season"] != s]
        held = [r for r in il_rows if r["season"] == s]
        if not held:
            continue
        d_c = min(GRID_D, key=lambda d: mse(train, d))
        b2 = min(((d1, d6) for d1 in GRID_D for d6 in GRID_D),
                 key=lambda p: mse(train, p[0], p[1]))
        tot += 1
        w = mse(held, d_c) < mse(held, 0.0)
        w2 = mse(held, b2[0], b2[1]) < mse(held, d_c)
        wins += w
        wins2 += w2
        print(f"  held-out {s}: fitted d={d_c:.2f}  "
              f"{'WIN' if w else 'lose'} vs d=0; "
              f"split ({b2[0]:.2f},{b2[1]:.2f}) "
              f"{'beats' if w2 else 'loses to'} single")
    print(f"\nfitted d beats d=0 in {wins}/{tot} held-out seasons")
    print(f"severity split beats single in {wins2}/{tot}")
    verdict = status == "interior" and wins > tot / 2
    print("VERDICT:", f"SHIP d={d_best}" if verdict else "NULL - no IL discount",
          "+ severity split" if (verdict and wins2 > tot / 2) else "")


if __name__ == "__main__":
    main()
