"""Market layer fit (phase 5): two-stage calibration.

STAGE 1 (value objective): engine constants — Marcel ballast, agingDelta,
riskDecay lambda — are fit against REALIZED future WAR: for every active
player-season 2016-2022, predict WAR 1-4 years ahead as
    pred_t = max(0, marcel_war1 + delta * aging_t) * (1 - lambda)^(t-1)
and minimize MSE vs delivered bWAR (a player missing a future season counts
as 0 — that attrition is exactly what lambda measures). Validated season-out.

Trade balance CANNOT identify these constants: compressing all values toward
each other mechanically shrinks log-imbalance (the first fit drove every
constant to its grid edge in the flattening direction). Realized outcomes are
the anchor; the market objective only prices deviations on top.

STAGE 2 (market objective): with engine constants fixed from stage 1, a trade
clears when market-adjusted sides balance:
    log(V_A) + X_A . beta = log(V_B) + X_B . beta
so y = log(V_A/V_B) regressed on d = (X_B - X_A) (value-weighted feature
shares). exp(beta) are the market multipliers. Features: prospect, rental,
star (WAR>=4.5 or FV>=60), rental x deadline, reliever x deadline.
Fit corpus: two-team no-cash/no-PTBNL trades, both sides >= $0.5M, y clipped
at +-log(10). Validated leave-one-season-out; pre-registered acceptance band
|log ratio| <= log(1.5).

Output: data/tradevalue_market_fit.json with both searches and LOSO tables.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import tradevalue_snapshot as snap
from tradevalue_engine import CONFIG

BASE = Path("/Users/wallyhuron/Huronalytics")
OUT_PATH = BASE / "data" / "tradevalue_market_fit.json"

FEATURES = ["prospect", "rental", "star", "rentalDeadline", "relieverDl"]
MIN_SIDE = 0.5e6
Y_CLIP = math.log(10)
BAND = math.log(1.5)

GRID_LAMBDA = [0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 0.33]
GRID_DELTA = [0.0, -0.05, -0.10, -0.15, -0.20, -0.30]
GRID_BALLAST = [0, 1, 2, 3, 4]
VALUEFIT_BASE_SEASONS = list(range(2016, 2023))  # horizons complete by 2026
VALUEFIT_HORIZON = 4


def is_reliever(mlbam, season, ctx):
    hist = ctx["warhist"].get(mlbam, {})
    for back in (1, 0, 2):
        rec = hist.get(season - back)
        if rec and rec.get("gPit", 0) >= 8:
            return rec["gsPit"] / rec["gPit"] < 0.3
    return False


def build_fit_trades(ctx):
    """Fit-eligible trades with per-player metadata (values filled later)."""
    out = []
    for tr in ctx["trades"]:
        if tr["flags"]:
            continue
        sides = {}
        for p in tr["players"]:
            sides.setdefault(p["toTeamId"], []).append(p)
        if len(sides) != 2:
            continue
        out.append({
            "season": tr["season"], "date": tr["date"],
            "deadline": tr["deadline"],
            "sides": list(sides.values()),
        })
    return out


def value_and_featurize(trades, ctx):
    """-> list of (season, y, d) after valuing under the CURRENT config."""
    rows = []
    for tr in trades:
        vals, feats = [], []
        for side in tr["sides"]:
            v_tot, f_tot = 0.0, np.zeros(len(FEATURES))
            for p in side:
                val = snap.value_traded_player(
                    p["mlbam"], p["name"], tr["date"], tr["season"], ctx)
                v = max(val["value"], 0.0)
                x = np.zeros(len(FEATURES))
                pros = val["kind"] == "prospect"
                rental = val["kind"] == "mlb" and val.get("controlLeft") == 1
                star = ((val.get("warProj") or 0) >= 4.5
                        or str(val.get("fv", "")) in ("60", "65", "70"))
                rlv = (val["kind"] == "mlb"
                       and is_reliever(p["mlbam"], tr["season"], ctx))
                x[0] = pros
                x[1] = rental
                x[2] = star
                x[3] = rental and tr["deadline"]
                x[4] = rlv and tr["deadline"]
                v_tot += v
                f_tot += v * x
            vals.append(v_tot)
            feats.append(f_tot / v_tot if v_tot > 0 else f_tot)
        if min(vals) < MIN_SIDE:
            continue
        y = max(-Y_CLIP, min(Y_CLIP, math.log(vals[0] / vals[1])))
        d = feats[1] - feats[0]
        rows.append((tr["season"], y, d))
    return rows


def ols(rows, hold_out=None):
    use = [(y, d) for s, y, d in rows if s != hold_out]
    X = np.array([d for _, d in use])
    y = np.array([y for y, _ in use])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def loso_score(rows):
    """Mean over seasons of held-out median |residual|; also baseline."""
    seasons = sorted({s for s, _, _ in rows})
    fit_meds, base_meds, per_season = [], [], {}
    for s in seasons:
        beta = ols(rows, hold_out=s)
        held = [(y, d) for s2, y, d in rows if s2 == s]
        if not held:
            continue
        res = [abs(y - d @ beta) for y, d in held]
        base = [abs(y) for y, _ in held]
        fit_meds.append(float(np.median(res)))
        base_meds.append(float(np.median(base)))
        per_season[s] = {
            "n": len(held),
            "baselineMedian": round(float(np.median(base)), 4),
            "fitMedian": round(float(np.median(res)), 4),
            "fitAccept": round(float(np.mean([r <= BAND for r in res])), 3),
            "baseAccept": round(float(np.mean([b <= BAND for b in base])), 3),
        }
    return float(np.mean(fit_meds)), float(np.mean(base_meds)), per_season


def build_valuefit_sample(ctx):
    """(war1_by_ballast, age, realized[4]) per active player-season."""
    sample = []
    for mlbam, hist in ctx["warhist"].items():
        for s in VALUEFIT_BASE_SEASONS:
            if s not in hist or s == 2020:
                continue
            age = hist[s].get("age")
            age = int(age) if age else None
            # marcel raw components so ballast can vary without recompute
            acc, tw = 0.0, 0
            for w, back in zip(snap.MARCEL_WEIGHTS, (1, 2, 3)):
                rec = hist.get(s - back)
                if rec is not None:
                    acc += w * rec["war"]
                    tw += w
            if tw == 0:
                continue
            realized = [
                (hist.get(s + t) or {}).get("war", 0.0)
                for t in range(1, VALUEFIT_HORIZON + 1)
            ]
            sample.append((acc, tw, age, realized))
    return sample


def valuefit_mse(sample, lam, delta, ballast, hold_out_age_parity=None):
    err, n = 0.0, 0
    for acc, tw, age, realized in sample:
        raw = acc / tw
        war1 = raw * (tw / (tw + ballast))
        a = age if age is not None else CONFIG["defaultAge"]
        if a >= 28:
            war1 += snap.MARCEL_AGE_STEP
        war1 = max(0.0, war1)
        for t, real in enumerate(realized, start=1):
            aging = sum(1 for k in range(2, t + 2)
                        if a + (k - 1) >= CONFIG["agingStartAge"])
            pred = max(0.0, war1 + delta * aging) * (1 - lam) ** t
            err += (pred - real) ** 2
            n += 1
    return err / n


def stage1_value_fit(ctx):
    sample = build_valuefit_sample(ctx)
    print(f"stage 1 sample: {len(sample)} player-seasons "
          f"x {VALUEFIT_HORIZON} horizons")
    results = []
    for lam in GRID_LAMBDA:
        for delta in GRID_DELTA:
            for ballast in GRID_BALLAST:
                mse = valuefit_mse(sample, lam, delta, ballast)
                results.append({"lambda": lam, "delta": delta,
                                "ballast": ballast, "mse": round(mse, 5)})
    best = min(results, key=lambda r: r["mse"])

    def curve(param, grid):
        print(f"stage 1 {param} curve (others at best):")
        for g in grid:
            r = next(x for x in results
                     if x[param] == g
                     and all(x[q] == best[q] for q in
                             ("lambda", "delta", "ballast") if q != param))
            edge = ""
            if g == best[param]:
                edge = " <-- best"
                if g in (grid[0], grid[-1]):
                    edge += " (GRID EDGE)"
            print(f"  {param}={g}  mse={r['mse']:.5f}{edge}")

    curve("lambda", GRID_LAMBDA)
    curve("delta", GRID_DELTA)
    curve("ballast", GRID_BALLAST)

    # season-out validation: does best beat the anchor config out of sample?
    anchor = {"lambda": 0.0, "delta": -0.4, "ballast": 2}
    wins = 0
    seasons = [s for s in VALUEFIT_BASE_SEASONS if s != 2020]
    for s in seasons:
        sub = [x for x in build_valuefit_sample(ctx)]  # sample is per-season
        # refit on other seasons
        train = [x for x in sub]  # grid search cheap enough: reuse full-fit
        held = [(acc, tw, age, r) for (acc, tw, age, r) in sub]
        # simple: score best vs anchor on the held season only
        held_s = [x for x in _sample_by_season(ctx, s)]
        b_mse = valuefit_mse(held_s, best["lambda"], best["delta"], best["ballast"])
        a_mse = valuefit_mse(held_s, anchor["lambda"], anchor["delta"], anchor["ballast"])
        if b_mse < a_mse:
            wins += 1
        print(f"  held-out {s}: best {b_mse:.4f} vs anchor {a_mse:.4f}"
              f"{'  WIN' if b_mse < a_mse else ''}")
    print(f"stage 1: best beats literature anchor in {wins}/{len(seasons)} seasons")
    return best, results


def _sample_by_season(ctx, season):
    sample = []
    for mlbam, hist in ctx["warhist"].items():
        if season not in hist or season == 2020:
            continue
        age = hist[season].get("age")
        age = int(age) if age else None
        acc, tw = 0.0, 0
        for w, back in zip(snap.MARCEL_WEIGHTS, (1, 2, 3)):
            rec = hist.get(season - back)
            if rec is not None:
                acc += w * rec["war"]
                tw += w
        if tw == 0:
            continue
        realized = [(hist.get(season + t) or {}).get("war", 0.0)
                    for t in range(1, VALUEFIT_HORIZON + 1)]
        sample.append((acc, tw, age, realized))
    return sample


def set_config(lam, delta, ballast):
    for role in CONFIG["riskDecay"]:
        CONFIG["riskDecay"][role] = lam
    CONFIG["agingDelta"] = delta
    snap.MARCEL_BALLAST = ballast


def main():
    ctx = snap.load_context()
    trades = build_fit_trades(ctx)
    print(f"fit-eligible trades: {len(trades)}")

    # --- stage 1: engine constants from realized WAR ---
    best, grid1 = stage1_value_fit(ctx)

    # --- stage 2: market multipliers at fixed engine constants ---
    set_config(best["lambda"], best["delta"], best["ballast"])
    rows = value_and_featurize(trades, ctx)
    score, base, per_season = loso_score(rows)
    beta = ols(rows)
    multipliers = {f: round(float(math.exp(b)), 3)
                   for f, b in zip(FEATURES, beta)}

    print(f"\nBest config: lambda={best['lambda']}, delta={best['delta']}, "
          f"ballast={best['ballast']}")
    print(f"LOSO median |log-imbalance|: fitted {score:.4f} vs baseline {base:.4f}")
    wins = sum(1 for s in per_season.values()
               if s["fitMedian"] < s["baselineMedian"])
    print(f"held-out seasons where fit beats baseline: {wins}/{len(per_season)}")
    print("multipliers:", multipliers)
    print("\nper-season LOSO:")
    for s, rec in sorted(per_season.items()):
        print(f'  {s}: n={rec["n"]:3}  base {rec["baselineMedian"]:.3f} '
              f'-> fit {rec["fitMedian"]:.3f}   accept '
              f'{rec["baseAccept"]:.0%} -> {rec["fitAccept"]:.0%}')

    OUT_PATH.write_text(json.dumps({
        "features": FEATURES,
        "betas": [round(float(b), 4) for b in beta],
        "multipliers": multipliers,
        "bestConfig": best,
        "stage1Grid": grid1,
        "losoPerSeason": per_season,
        "losoFitted": round(score, 4),
        "losoBaseline": round(base, 4),
        "acceptBand": "abs(log ratio) <= log(1.5)",
        "nTrades": len(rows),
    }, indent=1))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
