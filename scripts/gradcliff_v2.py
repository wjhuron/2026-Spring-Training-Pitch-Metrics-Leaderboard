"""Graduation blend v2 candidates: information-based decay (report-only).

The shipped ramp decays the FV weight on CALENDAR time. Time is a proxy for
accumulated MLB evidence — the Dylan Crews problem: a busted-looking grad
with 1,000+ bad PA keeps a high FV weight just because he graduated
recently. Candidates, all fit/validated leave-one-cohort-out on pooled
offsets 0-3 of the 2017-2021 graduate cohorts:

  RAMP    w(t) refit per offset (the shipped scheme's form, 4 params)
  SHRINK  w = k/(k + n), n = career MLB PA-equivalents at valuation (1 param)
  HYBRID  w = w0 * k/(k + n) (2 params)

Feature probes (same LOSO discipline): age at debut, FV trajectory (last
list's grade vs the one before). Nothing here touches the engine.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import tradevalue_snapshot as snap
import gradcliff_fit as gf

GRID_K = [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
GRID_W0 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
GRID_W = gf.GRID_W


def build_pooled():
    ctx = snap.load_context()
    warhist, boards = ctx["warhist"], ctx["boards"]
    candidates = [m for m, h in warhist.items() if min(h) in range(2016, 2023)]
    people = snap.load_people(sorted(candidates))
    rows = []
    for mlbam, hist in warhist.items():
        person = people.get(str(mlbam), {})
        debut = person.get("debut")
        if not debut:
            continue
        d = int(debut[:4])
        if d not in gf.DEBUT_YEARS:
            continue
        fv = gf.find_fv(mlbam, person.get("name") or "", person, d, boards)
        if fv is None:
            continue
        # previous FV (trajectory): earliest of the two most recent grades
        prev_fv = None
        for back in (2, 3, 4):
            for key in (f"{d - back}updated", f"{d - back}prospect"):
                b = boards.get(key)
                if not b:
                    continue
                row = b["byMlbam"].get(mlbam)
                if row is None:
                    nm = snap.norm_name(person.get("name") or "")
                    row = (b["byName"].get(
                               (nm, (person.get("birthDate") or "")[:4]))
                           or b["byNameOnly"].get(nm))
                if row is not None and str(row.get("fv", "")).strip() in gf.FV_TABLE:
                    prev_fv = str(row["fv"]).strip()
                    break
            if prev_fv:
                break
        birth = person.get("birthDate")
        debut_age = d - int(birth[:4]) if birth else None
        pitcher = bool(person.get("pitcher"))
        exp_war = gf.FV_TABLE[fv][3 if pitcher else 2]
        for offset in range(0, 4):
            eval_season = d + 1 + offset
            horizon = gf.HORIZON - offset
            if horizon <= 0:
                continue
            age = eval_season - int(birth[:4]) if birth else None
            proj1 = snap.marcel(hist, eval_season, age)
            decay_sum = sum((1 - gf.LAM) ** (t - 1)
                            for t in range(1, horizon + 1))
            n_exp = sum((hist.get(s) or {}).get("pa", 0.0)
                        + 1.4 * (hist.get(s) or {}).get("ipouts", 0.0)
                        for s in hist if s < eval_season)
            realized = sum((hist.get(eval_season + t) or {}).get("war", 0.0)
                           for t in range(0, horizon))
            rows.append({
                "cohort": d, "t": offset, "fv": fv, "prevFv": prev_fv,
                "debutAge": debut_age, "pitcher": pitcher,
                "n": n_exp, "proj": proj1 * decay_sum,
                "fvWar": exp_war * horizon / gf.CONTROL_YEARS_FV,
                "real": realized,
            })
    return rows


def pred_err(rows, w_of):
    e = [(w_of(r) * r["fvWar"] + (1 - w_of(r)) * r["proj"] - r["real"]) ** 2
         for r in rows]
    return float(np.mean(e))


def fit_ramp(rows):
    ws = {}
    for t in range(4):
        sub = [r for r in rows if r["t"] == t]
        ws[t] = min(GRID_W, key=lambda w: pred_err(sub, lambda r, w=w: w))
    return ws


def fit_shrink(rows):
    return min(GRID_K,
               key=lambda k: pred_err(rows, lambda r: k / (k + r["n"])))


def fit_hybrid(rows):
    best, best_m = None, None
    for w0 in GRID_W0:
        for k in GRID_K:
            m = pred_err(rows, lambda r: w0 * k / (k + r["n"]))
            if best_m is None or m < best_m:
                best, best_m = (w0, k), m
    return best


def main():
    rows = build_pooled()
    cohorts = sorted({r["cohort"] for r in rows})
    print(f"pooled rows: {len(rows)} across offsets 0-3, cohorts {cohorts}")

    wins = {"SHRINK": 0, "HYBRID": 0}
    for c in cohorts:
        tr = [r for r in rows if r["cohort"] != c]
        he = [r for r in rows if r["cohort"] == c]
        ws = fit_ramp(tr)
        m_ramp = pred_err(he, lambda r: ws[r["t"]])
        k = fit_shrink(tr)
        m_shr = pred_err(he, lambda r: k / (k + r["n"]))
        w0, k2 = fit_hybrid(tr)
        m_hyb = pred_err(he, lambda r: w0 * k2 / (k2 + r["n"]))
        wins["SHRINK"] += m_shr < m_ramp
        wins["HYBRID"] += m_hyb < m_ramp
        print(f"  held-out {c}: ramp={m_ramp:.3f}  "
              f"shrink(k={k})={m_shr:.3f}  "
              f"hybrid(w0={w0},k={k2})={m_hyb:.3f}")
    print(f"vs time-ramp: SHRINK wins {wins['SHRINK']}/{len(cohorts)}, "
          f"HYBRID wins {wins['HYBRID']}/{len(cohorts)}")

    # pooled fits + curves for the winner's params
    k = fit_shrink(rows)
    w0, k2 = fit_hybrid(rows)
    print(f"\npooled: shrink k={k}; hybrid w0={w0}, k={k2}")
    print("shrink curve:")
    for kk in GRID_K:
        m = pred_err(rows, lambda r: kk / (kk + r["n"]))
        print(f"  k={kk:5}: mse={m:.4f}" + ("  <--" if kk == k else ""))

    # implied weights for reference profiles
    for n in (0, 150, 400, 800, 1500):
        print(f"  hybrid w at n={n:4} PA-eq: "
              f"{w0 * k2 / (k2 + n):.2f}")

    # feature probes under LOSO: does splitting beat the pooled winner?
    def probe(label, split_fn):
        w = 0
        for c in cohorts:
            tr = [r for r in rows if r["cohort"] != c]
            he = [r for r in rows if r["cohort"] == c]
            w0a, ka = fit_hybrid([r for r in tr if split_fn(r)])
            w0b, kb = fit_hybrid([r for r in tr if not split_fn(r)])
            w0p, kp = fit_hybrid(tr)
            e_split = (sum((w0a * ka / (ka + r["n"]) * r["fvWar"]
                            + (1 - w0a * ka / (ka + r["n"])) * r["proj"]
                            - r["real"]) ** 2 for r in he if split_fn(r))
                       + sum((w0b * kb / (kb + r["n"]) * r["fvWar"]
                              + (1 - w0b * kb / (kb + r["n"])) * r["proj"]
                              - r["real"]) ** 2
                             for r in he if not split_fn(r))) / len(he)
        # (single final-cohort print keeps output short; wins tallied)
            e_pool = pred_err(he, lambda r: w0p * kp / (kp + r["n"]))
            w += e_split < e_pool
        print(f"  {label}: split wins {w}/{len(cohorts)}")

    print("\nfeature probes (vs pooled hybrid):")
    probe("young debut (<=23)", lambda r: (r["debutAge"] or 25) <= 23)
    probe("FV was cut pre-debut",
          lambda r: r["prevFv"] is not None
          and gf.FV_TABLE[r["fv"]][2] < gf.FV_TABLE[r["prevFv"]][2])


if __name__ == "__main__":
    main()
