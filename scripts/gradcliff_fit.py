"""Graduation-cliff fit: how much should final FV count after graduation?

The live engine values a graduated player purely off his projection; the
Board FV path vanishes the day he graduates (Connelly Early: FV prospect ->
0.84-WAR back-end starter overnight). This fits the blend
    predicted = w * FV_expected_WAR + (1 - w) * projection_WAR
against REALIZED post-graduation WAR (value-chart objective, per the
two-chart doctrine — trade balance cannot identify value constants).

Cohorts: MLB debuts 2017-2021 who carried a Board FV at debut (last list at
or before the debut season). Horizon: realized bWAR over the 4 seasons after
the debut season (complete through 2025). Validation: leave-one-cohort-out
across the 5 debut years; the blend weight must bracket an interior optimum
or the curve be shown flat. A second fit re-runs the exercise one season
after graduation to measure how fast the FV weight should decay.

Report-only: nothing here touches the engine.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import tradevalue_snapshot as snap
from tradevalue_engine import CONFIG

DEBUT_YEARS = range(2017, 2022)
HORIZON = 4
LAM = CONFIG["riskDecay"]["POS"]      # 0.21 multiplicative decay
GRID_W = [round(0.05 * i, 2) for i in range(0, 21)]
FV_TABLE = CONFIG["fvTable"]          # fv -> [h$, p$, expWarH, expWarP, ...]
CONTROL_YEARS_FV = 6.0                # table expWar covers the control window


def find_fv(mlbam, name, person, debut_year, boards):
    """Last Board FV at or before the debut season."""
    for key in (f"{debut_year}updated", f"{debut_year}prospect",
                f"{debut_year - 1}updated", f"{debut_year - 1}prospect"):
        b = boards.get(key)
        if not b:
            continue
        row = b["byMlbam"].get(mlbam)
        if row is None:
            nm = snap.norm_name(person.get("name") or name)
            row = (b["byName"].get((nm, (person.get("birthDate") or "")[:4]))
                   or b["byNameOnly"].get(nm))
        if row is not None and str(row.get("fv", "")).strip() in FV_TABLE:
            return str(row["fv"]).strip()
    return None


def build_cohorts(offset):
    """offset 0 = valued right after debut season; 1 = one season later."""
    ctx = snap.load_context()
    warhist, boards = ctx["warhist"], ctx["boards"]
    # make sure every candidate has a person record (debut date, position)
    candidates = [m for m, h in warhist.items()
                  if min(h) in range(2016, 2023)]
    people = snap.load_people(sorted(candidates))

    rows = []
    for mlbam, hist in warhist.items():
        person = people.get(str(mlbam), {})
        debut = person.get("debut")
        if not debut:
            continue
        d = int(debut[:4])
        if d not in DEBUT_YEARS:
            continue
        fv = find_fv(mlbam, person.get("name") or "", person, d, boards)
        if fv is None:
            continue
        eval_season = d + 1 + offset
        birth = person.get("birthDate")
        age = eval_season - int(birth[:4]) if birth else None
        proj1 = snap.marcel(hist, eval_season, age)
        horizon = HORIZON - offset
        decay_sum = sum((1 - LAM) ** (t - 1) for t in range(1, horizon + 1))
        proj_war = proj1 * decay_sum
        pitcher = bool(person.get("pitcher"))
        exp_war = FV_TABLE[fv][3 if pitcher else 2]
        fv_war = exp_war * horizon / CONTROL_YEARS_FV
        realized = sum((hist.get(eval_season + t) or {}).get("war", 0.0)
                       for t in range(0, horizon))
        rows.append({"mlbam": mlbam, "cohort": d, "fv": fv,
                     "pitcher": pitcher, "proj": proj_war,
                     "fvWar": fv_war, "real": realized})
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

    # pooled curve
    curve = [(w, mse(rows, w)) for w in GRID_W]
    w_best, m_best = min(curve, key=lambda x: x[1])
    pos = GRID_W.index(w_best)
    status = ("interior" if 0 < pos < len(GRID_W) - 1 else "GRID EDGE")
    print(f"pooled optimum w={w_best} ({status})")
    for w, m in curve[::2]:
        print(f"  w={w:.2f}  mse={m:.4f}" + ("  <--" if w == w_best else ""))

    # leave-one-cohort-out: fit w on other cohorts, eval held-out vs endpoints
    wins_proj, wins_fv, tot = 0, 0, 0
    for c, held in sorted(by_c.items()):
        train = [r for r in rows if r["cohort"] != c]
        w_c = min(GRID_W, key=lambda w: mse(train, w))
        m_blend = mse(held, w_c)
        m_proj = mse(held, 0.0)
        m_fv = mse(held, 1.0)
        tot += 1
        wins_proj += m_blend < m_proj
        wins_fv += m_blend < m_fv
        print(f"  held-out {c}: fitted w={w_c:.2f}  blend mse={m_blend:.4f}  "
              f"proj-only={m_proj:.4f}  fv-only={m_fv:.4f}")
    print(f"LOSO: blend beats proj-only {wins_proj}/{tot}, "
          f"fv-only {wins_fv}/{tot}")

    # scale check: unconstrained OLS
    A = np.column_stack([np.ones(len(rows)),
                         [r["proj"] for r in rows],
                         [r["fvWar"] for r in rows]])
    y = np.array([r["real"] for r in rows])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    print(f"unconstrained OLS: intercept={coef[0]:.2f} "
          f"proj={coef[1]:.2f} fv={coef[2]:.2f}")

    # tier split (descriptive)
    for tiers, name in ((("50", "55", "60", "65", "70"), "FV>=50"),
                        (("40", "40+", "45", "45+", "35+"), "FV<50")):
        sub = [r for r in rows if r["fv"] in tiers]
        if len(sub) >= 40:
            w_t = min(GRID_W, key=lambda w: mse(sub, w))
            print(f"  {name} (n={len(sub)}): optimum w={w_t:.2f}")
    return w_best


def main():
    fresh = build_cohorts(offset=0)
    w0 = fit(fresh, "fresh graduates (valued season after debut)")
    later = build_cohorts(offset=1)
    w1 = fit(later, "one season post-graduation")
    print(f"\nsummary: w(fresh)={w0}  w(+1 season)={w1}")


if __name__ == "__main__":
    main()
