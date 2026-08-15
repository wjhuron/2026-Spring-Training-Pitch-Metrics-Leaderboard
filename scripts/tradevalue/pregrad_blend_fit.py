"""Pre-graduation blend fit: should a projection temper MLB-ready prospects?

The prospect path is performance-blind: an FV-50 catcher having a bad AAA
season carries the same value as one raking (Harry Ford case). But every
MLB-ready prospect has a real projection (Steamer projects minor leaguers
via MLEs, and the historical preseason files cover 2017+), and the
graduation-cliff blend already validated FV-vs-projection mixing one step
later. This fits the same form one step EARLIER, for prospects still on
the Board but close:

    predicted = w * FV_expected_WAR * hetero_mult + (1 - w) * proj_WAR

against realized bWAR over the 5 seasons from the list year (the same
window as the heterogeneity fit). Cohorts: 2017-2021 preseason Boards,
prospects with ETA <= list year + 1 and a Steamer preseason row.

PRE-REGISTERED GATES: the pooled optimum w must be interior (or the curve
shown flat), AND the LOSO blend must beat FV-only in >=4/5 held-out
cohorts AND proj-only in >=4/5. Report-only: nothing touches the engine.
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import tradevalue_snapshot as snap
from tradevalue_engine import CONFIG

COHORTS = range(2017, 2022)
HORIZON = 5                       # seasons from list year, matches hetero fit
LAM = CONFIG["riskDecay"]["POS"]
GRID_W = [round(0.05 * i, 2) for i in range(0, 21)]
FV_TABLE = CONFIG["fvTable"]
CONTROL_YEARS_FV = 6.0


def build_sample(eta_window):
    ctx = snap.load_context()
    warhist = ctx["warhist"]
    steamer = snap.load_steamer()
    import json
    by_fg = {}
    for r in csv.DictReader(open(snap.DATA / "tradevalue_idmap.csv")):
        if r["fangraphs"]:
            by_fg[r["fangraphs"]] = int(r["mlbam"])
    raw = json.loads((snap.DATA / "tradevalue_board_hist.json").read_text())

    rows = []
    n_no_proj = 0
    for year in COHORTS:
        for r in raw.get(f"{year}prospect", []):
            fv = str(r.get("fv") or "").strip()
            if fv not in FV_TABLE:
                continue
            try:
                eta = float(r.get("eta"))
            except (TypeError, ValueError):
                continue
            if not (year <= eta <= year + eta_window):
                continue
            mlbam = by_fg.get(str(r.get("fgId") or ""))
            if mlbam is None:
                continue
            st = steamer.get((year, mlbam))
            if st is None:
                n_no_proj += 1
                continue
            pitcher = (r.get("pos") or "").upper() in snap._PITCH_POS
            exp_war = FV_TABLE[fv][3 if pitcher else 2]
            fv_war = (exp_war * HORIZON / CONTROL_YEARS_FV
                      * snap.hetero_mult(r, year))
            decay_sum = sum((1 - LAM) ** (t - 1)
                            for t in range(1, HORIZON + 1))
            proj_war = max(0.0, st) * decay_sum
            hist = warhist.get(mlbam, {})
            realized = sum((hist.get(year + t) or {}).get("war", 0.0)
                           for t in range(0, HORIZON))
            rows.append({"cohort": year, "fv": fv, "name": r.get("name"),
                         "fvWar": fv_war, "proj": proj_war,
                         "real": realized})
    print(f"eta window +{eta_window}: {len(rows)} prospect-listings "
          f"({n_no_proj} near-ETA listings had no Steamer row)")
    return rows


def fit(rows, label):
    print(f"\n=== {label}: n={len(rows)} ===")
    by_c = {}
    for r in rows:
        by_c.setdefault(r["cohort"], []).append(r)
    print("cohort sizes:", {c: len(v) for c, v in sorted(by_c.items())})

    def mse(sub, w):
        e = [(w * r["fvWar"] + (1 - w) * r["proj"] - r["real"]) ** 2
             for r in sub]
        return float(np.mean(e))

    curve = [(w, mse(rows, w)) for w in GRID_W]
    w_best, m_best = min(curve, key=lambda x: x[1])
    pos = GRID_W.index(w_best)
    status = ("interior" if 0 < pos < len(GRID_W) - 1 else "GRID EDGE")
    print(f"pooled optimum w={w_best} ({status})")
    for w, m in curve[::2]:
        print(f"  w={w:.2f}  mse={m:.4f}" + ("  <--" if w == w_best else ""))

    wins_proj, wins_fv, tot = 0, 0, 0
    fold_ws = []
    for c, held in sorted(by_c.items()):
        train = [r for r in rows if r["cohort"] != c]
        w_c = min(GRID_W, key=lambda w: mse(train, w))
        fold_ws.append(w_c)
        m_blend = mse(held, w_c)
        m_proj = mse(held, 0.0)
        m_fv = mse(held, 1.0)
        tot += 1
        wins_proj += m_blend < m_proj
        wins_fv += m_blend < m_fv
        print(f"  held-out {c}: fitted w={w_c:.2f}  blend={m_blend:.4f}  "
              f"proj-only={m_proj:.4f}  fv-only={m_fv:.4f}")
    print(f"LOSO: blend beats proj-only {wins_proj}/{tot}, "
          f"fv-only {wins_fv}/{tot}")
    verdict = (status == "interior" and wins_proj >= 4 and wins_fv >= 4)
    print("GATES:", "PASS" if verdict else "fail",
          f"(interior={status == 'interior'}, vs-proj {wins_proj}/5, "
          f"vs-fv {wins_fv}/5, fold ws {fold_ws})")
    return w_best, verdict


def main():
    primary = build_sample(eta_window=1)
    w1, ok1 = fit(primary, "ETA <= list year + 1 (primary)")
    strict = build_sample(eta_window=0)
    w0, ok0 = fit(strict, "ETA <= list year (strict)")
    print(f"\nsummary: w(+1)={w1} pass={ok1}   w(0)={w0} pass={ok0}")


if __name__ == "__main__":
    main()
