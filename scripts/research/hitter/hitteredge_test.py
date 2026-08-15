"""Hitter analog of the stuff blend: does contact quality forecast?

Mirror of stuffedge_blend_analysis.py for batters. Per pitch, from the
Statcast caches (batter perspective, positive = good for the batter):

  actual  = delta_run_exp                       (luck included)
  process = (xwOBA_contact - lgwOBA)/scale on balls in play,
            delta_run_exp on everything else     (contact-quality neutralized)

Question: does FIRST-HALF process add signal for SECOND-HALF actual
production beyond FH actual + three prior seasons of actual rates
(regression-weighted)? Evaluated on 2024 and 2025 as replicates (priors
2021-2023 / 2022-2024). No model training: the process metric is data.

PRE-REGISTERED (same bar as the pitcher blend): build the hitter adjustment
iff (a) partial corr of FH process with SH actual, controlling the full
baseline, is positive in BOTH seasons at the primary threshold (>=700 FH
and >=700 SH pitches), AND (b) cross-season transfer improves pitch-weighted
SH MSE in both directions. Report-only.

RESULT (2026-07-29): NULL, twice. Raw xwOBA process: partials 0.188/0.115
but transfer fails one direction and the coefficient halves across seasons
(0.36/0.18). With the prior process-gap decontamination in the baseline
(this file's current form): partials shrink to 0.095/0.076 and transfer
still fails (-0.3%/+0.8%) - the apparent edge was mostly the persistent
speed/spray skill that belongs in the baseline. No hitter adjustment is
built. Revisit with bat-tracking features once three full seasons exist
(2024-2026) to support a season-blocked design.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

# league wOBA / scale per season (same constants the stuff pipeline uses)
GUTS = {2021: (0.314, 1.209), 2022: (0.310, 1.259), 2023: (0.318, 1.204),
        2024: (0.310, 1.242), 2025: (0.3131, 1.2317)}
CACHE = "/Users/wallyhuron/Huronalytics/data/_statcast{year}_cache.pkl"
EVAL_YEARS = (2024, 2025)
SPLIT = "-07-01"
THR = 700
PRIOR_W = (5, 4, 3)
# prior_gap: the batter's own prior-season (process - actual) gap. For
# hitters that gap is partly persistent skill (speed/spray beat xwOBA every
# year), so it belongs in the BASELINE; without it the process residual is
# contaminated player-specifically and the pooled coefficient transfers
# badly (the first run's failure mode).
BASE = ["fh_actual", "prior_rate", "prior_cov", "prior_gap"]
FULL = BASE + ["fh_process"]
PITCHES_PER_SEASON = 2500   # full-time hitter, for WAR translation
RUNS_PER_WIN = 10.0


def season_frame(year):
    df = pickle.load(open(CACHE.format(year=year), "rb"))
    df = df[df["game_type"] == "R"] if "game_type" in df else df
    lg, sc = GUTS[year]
    actual = df["delta_run_exp"].astype(float)
    xw = df["estimated_woba_using_speedangle"].astype(float)
    process = np.where(xw.notna(), (xw - lg) / sc, actual)
    out = pd.DataFrame({
        "batter": df["batter"].values,
        "date": df["game_date"].astype(str).values,
        "actual": actual.values,
        "process": process,
    }).dropna(subset=["actual"])
    print(f"  {year}: {len(out)} pitches")
    return out


def main():
    season_rates = {}
    tables = {}
    for year in range(2021, 2026):
        df = season_frame(year)
        g = df.groupby("batter").agg(mean=("actual", "mean"),
                                     size=("actual", "size"),
                                     pmean=("process", "mean"))
        season_rates[year] = {b: (m, n, pm) for b, (m, n, pm)
                              in zip(g.index, g.values)}
        if year in EVAL_YEARS:
            fh = df["date"] < f"{year}{SPLIT}"
            g_fh = df[fh].groupby("batter").agg(
                fh_actual=("actual", "mean"), fh_process=("process", "mean"),
                fh_n=("actual", "size"))
            g_sh = df[~fh].groupby("batter").agg(
                sh_actual=("actual", "mean"), sh_n=("actual", "size"))
            t = g_fh.join(g_sh, how="inner").dropna()
            pri, cov, gap = [], [], []
            for b in t.index:
                acc, tw, gacc = 0.0, 0.0, 0.0
                for w, back in zip(PRIOR_W, (1, 2, 3)):
                    r = season_rates.get(year - back, {}).get(b)
                    if r is not None and r[1] >= 400:
                        acc += w * r[0]
                        gacc += w * (r[2] - r[0])
                        tw += w
                pri.append(acc / tw if tw else 0.0)
                gap.append(gacc / tw if tw else 0.0)
                cov.append(tw / sum(PRIOR_W))
            t["prior_rate"], t["prior_cov"], t["prior_gap"] = pri, cov, gap
            tables[year] = t[(t["fh_n"] >= THR) & (t["sh_n"] >= THR)]
        del df

    def wls(cols, frame):
        w = np.sqrt(frame["sh_n"].values)
        A = np.column_stack([np.ones(len(frame))] +
                            [frame[c].values for c in cols])
        coef, *_ = np.linalg.lstsq(A * w[:, None],
                                   frame["sh_actual"].values * w, rcond=None)
        return coef

    def wmse(cols, coef, frame):
        A = np.column_stack([np.ones(len(frame))] +
                            [frame[c].values for c in cols])
        return float(np.average((frame["sh_actual"].values - A @ coef) ** 2,
                                weights=frame["sh_n"].values))

    print("\n=== cross-season transfer ===")
    transfer_ok = True
    for fit_yr in EVAL_YEARS:
        eval_yr = EVAL_YEARS[1] if fit_yr == EVAL_YEARS[0] else EVAL_YEARS[0]
        f, e = tables[fit_yr], tables[eval_yr]
        m_base = wmse(BASE, wls(BASE, f), e)
        coef_full = wls(FULL, f)
        m_full = wmse(FULL, coef_full, e)
        print(f"fit {fit_yr} -> eval {eval_yr} (n={len(e)}): "
              f"baseline wMSE={m_base:.7f}  +process={m_full:.7f}  "
              f"improvement={100 * (1 - m_full / m_base):.1f}%  "
              f"process coef={coef_full[-1]:.3f}")
        if m_full >= m_base:
            transfer_ok = False

    print("\n=== partial correlation of process given the baseline ===")
    partials = {}
    for year, t in tables.items():
        def resid(a, cols):
            A = np.column_stack([np.ones(len(t))] +
                                [t[c].values for c in cols])
            c, *_ = np.linalg.lstsq(A, a, rcond=None)
            return a - A @ c
        r = float(np.corrcoef(resid(t["fh_process"].values, BASE),
                              resid(t["sh_actual"].values, BASE))[0, 1])
        partials[year] = r
        print(f"{year}: partial(process | FH actual + priors) = {r:.3f}  "
              f"(n={len(t)})")

    print("\n=== implied WAR adjustment sizes (coef from other season) ===")
    for year in EVAL_YEARS:
        other = EVAL_YEARS[1] if year == EVAL_YEARS[0] else EVAL_YEARS[0]
        coef = wls(FULL, tables[other])
        t = tables[year]
        A = np.column_stack([np.ones(len(t))] +
                            [t[c].values for c in BASE])
        c_r, *_ = np.linalg.lstsq(A, t["fh_process"].values, rcond=None)
        resid_p = t["fh_process"].values - A @ c_r
        adj_war = coef[-1] * resid_p * PITCHES_PER_SEASON / RUNS_PER_WIN
        q = np.percentile(np.abs(adj_war), [50, 90, 99])
        print(f"{year}: |WAR adj| median={q[0]:.2f}  p90={q[1]:.2f}  "
              f"p99={q[2]:.2f}")

    both_pos = all(r > 0 for r in partials.values())
    print("\n=== verdict (pre-registered) ===")
    print(f"(a) partial positive both seasons: {both_pos} "
          f"({ {y: round(r, 3) for y, r in partials.items()} })")
    print(f"(b) transfer improves both directions: {transfer_ok}")
    print("BUILD the hitter adjustment" if (both_pos and transfer_ok)
          else "NULL RESULT - do not build")


if __name__ == "__main__":
    main()
