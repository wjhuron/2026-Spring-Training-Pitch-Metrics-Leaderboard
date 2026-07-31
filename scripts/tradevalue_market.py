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

# Wide feature vocabulary; evaluation is staged (see main): the incumbent
# set F0 plus candidate groups A (reliever rental/controlled split),
# B (defensive-value share), C (catcher share). Pre-registered keep rule:
# a group stays iff LOSO mean-of-medians improves AND it wins >=6/10
# held-out seasons vs the set without it.
FEATURES = ["prospect", "rental", "star", "rentalDeadline", "relieverDl",
            "gradW", "rlvRentalDl", "rlvCtrlDl", "defFrac", "catcher",
            "onIL", "spDl", "starDl"]
F0 = ["prospect", "rental", "star", "rentalDeadline", "relieverDl", "gradW"]
# RETIRED candidates (tested on the pick-expanded corpus 2026-07-31,
# role-based groups ordered first per the working hypothesis):
#   E spDeadline  4/10 reject - rental (1.97) + star already carry
#                 deadline-SP economics; no residual SP effect
#   F starDeadline 4/10 reject - year-round star discount suffices
#   B defFrac     4/10 reject; 1 pass in 5 corpus versions = noise.
#                 Position-player defensive share is DEAD as a market
#                 feature (the hypothesis was right).
#   D onIL        2/10 reject - injury is intrinsic (CONFIG ilDiscount),
#                 not market behavior
# Surviving structure is role-based + catcher: the reliever deadline
# split (strongest effect, 2.9x/2.2x) and the catcher discount (passed
# 3 of 5 corpus versions, 7/10 on the current one).
GROUPS = [("A relieverSplit", ["rlvRentalDl", "rlvCtrlDl"], ["relieverDl"]),
          ("C catcher", ["catcher"], [])]
MIN_SIDE = 0.5e6
Y_CLIP = math.log(10)
BAND = math.log(1.5)

GRID_LAMBDA = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27,
               0.30, 0.33]
GRID_DELTA = [0.0, -0.05, -0.10, -0.15, -0.20, -0.30]
GRID_BALLAST = [0, 1, 2, 3, 4]
VALUEFIT_BASE_SEASONS = list(range(2016, 2023))  # horizons complete by 2026
VALUEFIT_HORIZON = 4
# a-priori talent tiers for the bucketed-lambda fit (conventional cuts)
WAR_BUCKETS = [(0.0, 2.0), (2.0, 4.0), (4.0, 99.0)]
GRID_LAMBDA_FINE = [round(0.03 * i, 2) for i in range(0, 12)]


def is_reliever(mlbam, season, ctx):
    hist = ctx["warhist"].get(mlbam, {})
    for back in (1, 0, 2):
        rec = hist.get(season - back)
        if rec and rec.get("gPit", 0) >= 8:
            return rec["gsPit"] / rec["gPit"] < 0.3
    return False


_IL = None


def on_il_at(mlbam, trade_date):
    """On a 10/15/60-day MLB IL on the trade date (tradevalue_ilhist.py)."""
    global _IL
    if _IL is None:
        import json as _j
        path = BASE / "data" / "tradevalue_il_hist.json"
        _IL = ({int(k): v for k, v in
                _j.loads(path.read_text())["episodes"].items()}
               if path.exists() else {})
    for ep in _IL.get(mlbam, []):
        # un-closed stints cap at Nov 30 of the placement year (offseason
        # roll-offs never post an activation transaction)
        off = ep["off"] or (ep["on"][:4] + "-11-30")
        if ep["on"] <= trade_date < off:
            return True
    return False


# Tradeable comp-round picks in otherwise-clean trades, hand-mapped from
# descriptions (only 7 pick trades exist in the corpus; 2 others stay
# excluded: Priester has a PTBNL, Donovan has a 3-team-id anomaly).
# Values from data/tradevalue_pick_curve.json (isotonic median of eephus
# implied pick surplus, Clemens-2026 dollars). The Waters and Baker
# trades are pick-ONLY sides - structurally unfittable before this.
PICK_TRADES = {
    ("2022-07-11", 144): 12.3e6,   # ATL gets CBA #35 for Waters package
    ("2024-12-14", 139): 12.35e6,  # TBR gets CBA-A (~#33) in Springs deal
    ("2025-01-07", 119): 12.0e6,   # LAD gets CIN comp-A (#40) for Lux
    ("2025-07-10", 110): 12.3e6,   # BAL gets #37 for Bryan Baker
    ("2025-12-19", 139): 12.35e6,  # TBR gets 2026 CBA #33 in Baz deal
}


def build_fit_trades(ctx):
    """Fit-eligible trades with per-player metadata (values filled later)."""
    out = []
    for tr in ctx["trades"]:
        pick_sides = {tid: v for (d, tid), v in PICK_TRADES.items()
                      if d == tr["date"]}
        if tr["flags"] and not (tr["flags"] == ["draftPick"] and pick_sides):
            continue
        sides = {}
        for p in tr["players"]:
            sides.setdefault(p["toTeamId"], []).append(p)
        for tid in pick_sides:
            sides.setdefault(tid, [])
        if len(sides) != 2:
            continue
        out.append({
            "season": tr["season"], "date": tr["date"],
            "deadline": tr["deadline"],
            "sides": [(tid, ps) for tid, ps in sides.items()],
            "pickValues": pick_sides,
        })
    return out


def def_frac(mlbam, season, ctx):
    """Defensive share of a position player's trailing run value, [-1, 1].

    Trailing = trade season + prior season pooled; gated at 300 PA so
    part-timers don't produce wild ratios.
    """
    hist = ctx["warhist"].get(mlbam, {})
    rb = rd = pa = 0.0
    for s in (season, season - 1):
        rec = hist.get(s)
        if rec:
            rb += rec.get("runsBat", 0.0)
            rd += rec.get("runsDef", 0.0)
            pa += rec.get("pa", 0.0)
    if pa < 300:
        return 0.0
    denom = abs(rb) + abs(rd)
    return rd / denom if denom > 1.0 else 0.0


def value_and_featurize(trades, ctx):
    """-> list of (season, y, d) after valuing under the CURRENT config."""
    people = ctx["people"]
    rows = []
    for tr in trades:
        vals, feats = [], []
        for tid, side in tr["sides"]:
            # comp-round picks: face value from the fitted curve, no
            # market-feature flags (they are neither MLB nor Board rows)
            v_tot = tr.get("pickValues", {}).get(tid, 0.0)
            f_tot = np.zeros(len(FEATURES))
            for p in side:
                val = snap.value_traded_player(
                    p["mlbam"], p["name"], tr["date"], tr["season"], ctx)
                v = max(val["value"], 0.0)
                x = np.zeros(len(FEATURES))
                pros = val["kind"] == "prospect"
                is_mlb = val["kind"] == "mlb"
                rental = is_mlb and val.get("controlLeft") == 1
                star = ((val.get("warProj") or 0) >= 4.5
                        or str(val.get("fv", "")) in ("60", "65", "70"))
                rlv = is_mlb and is_reliever(p["mlbam"], tr["season"], ctx)
                person = people.get(str(p["mlbam"]), {})
                pos = person.get("pos")
                x[0] = pros
                x[1] = rental
                x[2] = star
                x[3] = rental and tr["deadline"]
                x[4] = rlv and tr["deadline"]
                x[5] = val.get("gradBlendW", 0.0)
                x[6] = rlv and tr["deadline"] and rental
                x[7] = rlv and tr["deadline"] and not rental
                x[8] = (def_frac(p["mlbam"], tr["season"], ctx)
                        if is_mlb and not person.get("pitcher") else 0.0)
                x[9] = pos == "C"
                x[10] = is_mlb and on_il_at(p["mlbam"], tr["date"])
                sp = is_mlb and person.get("pitcher") and not rlv
                x[11] = sp and tr["deadline"]
                x[12] = star and tr["deadline"]
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


def _cols(featset):
    return [FEATURES.index(f) for f in featset]


def ols(rows, hold_out=None, featset=None):
    idx = _cols(featset or FEATURES)
    use = [(y, d) for s, y, d in rows if s != hold_out]
    X = np.array([d[idx] for _, d in use])
    y = np.array([y for y, _ in use])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def loso_score(rows, featset=None):
    """Mean over seasons of held-out median |residual|; also baseline."""
    idx = _cols(featset or FEATURES)
    seasons = sorted({s for s, _, _ in rows})
    fit_meds, base_meds, per_season = [], [], {}
    for s in seasons:
        beta = ols(rows, hold_out=s, featset=featset)
        held = [(y, d) for s2, y, d in rows if s2 == s]
        if not held:
            continue
        res = [abs(y - d[idx] @ beta) for y, d in held]
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
    """(steamer_war1, marcel_acc, marcel_tw, age, realized[4]) per active
    player-season. war1 source mirrors production: preseason Steamer where
    available (2017+), Marcel components as fallback (ballast varies in the
    grid without recompute)."""
    steamer = snap.load_steamer()
    sample = []
    for mlbam, hist in ctx["warhist"].items():
        for s in VALUEFIT_BASE_SEASONS:
            if s not in hist or s == 2020:
                continue
            age = hist[s].get("age")
            age = int(age) if age else None
            st = steamer.get((s, mlbam))
            acc, tw = 0.0, 0
            for w, back in zip(snap.MARCEL_WEIGHTS, (1, 2, 3)):
                rec = hist.get(s - back)
                if rec is not None:
                    acc += w * rec["war"]
                    tw += w
            if st is None and tw == 0:
                continue
            realized = [
                (hist.get(s + t) or {}).get("war", 0.0)
                for t in range(1, VALUEFIT_HORIZON + 1)
            ]
            sample.append((st, acc, tw, age, realized))
    return sample


def valuefit_mse(sample, lam, delta, ballast, hold_out_age_parity=None):
    err, n = 0.0, 0
    for st, acc, tw, age, realized in sample:
        a = age if age is not None else CONFIG["defaultAge"]
        if st is not None:
            war1 = st
        else:
            raw = acc / tw
            war1 = raw * (tw / (tw + ballast))
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
        # score best vs anchor on the held season only
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
    steamer = snap.load_steamer()
    sample = []
    for mlbam, hist in ctx["warhist"].items():
        if season not in hist or season == 2020:
            continue
        age = hist[season].get("age")
        age = int(age) if age else None
        st = steamer.get((season, mlbam))
        acc, tw = 0.0, 0
        for w, back in zip(snap.MARCEL_WEIGHTS, (1, 2, 3)):
            rec = hist.get(season - back)
            if rec is not None:
                acc += w * rec["war"]
                tw += w
        if st is None and tw == 0:
            continue
        realized = [(hist.get(season + t) or {}).get("war", 0.0)
                    for t in range(1, VALUEFIT_HORIZON + 1)]
        sample.append((st, acc, tw, age, realized))
    return sample


def stage1b_bucketed_lambda(ctx, best):
    """Refit lambda per war1 talent bucket (delta/ballast fixed from stage 1).

    The pooled lambda is dominated by ordinary players; whether elite players
    decay at the same PROPORTIONAL rate is an empirical question this answers.
    """
    sample = build_valuefit_sample(ctx)
    delta, ballast = best["delta"], best["ballast"]

    def war1_of(st, acc, tw, age):
        if st is not None:
            return max(0.0, st)
        raw = acc / tw
        w = raw * (tw / (tw + ballast))
        a = age if age is not None else CONFIG["defaultAge"]
        if a >= 28:
            w += snap.MARCEL_AGE_STEP
        return max(0.0, w)

    fitted = []
    for lo, hi in WAR_BUCKETS:
        rows = [x for x in sample if lo <= war1_of(x[0], x[1], x[2], x[3]) < hi]
        curve = [(lam, valuefit_mse(rows, lam, delta, ballast))
                 for lam in GRID_LAMBDA_FINE]
        lam_best, mse_best = min(curve, key=lambda c: c[1])
        pos = GRID_LAMBDA_FINE.index(lam_best)
        status = ("interior" if 0 < pos < len(GRID_LAMBDA_FINE) - 1
                  else "GRID EDGE")
        print(f"stage 1b bucket {lo}-{hi} WAR: n={len(rows)}, "
              f"lambda={lam_best} ({status})")
        for lam, m in curve:
            print(f"    lambda={lam:.2f}  mse={m:.5f}"
                  + ("  <--" if lam == lam_best else ""))
        # season-out check vs the pooled lambda
        wins, tot = 0, 0
        for s in [x for x in VALUEFIT_BASE_SEASONS if x != 2020]:
            held = [x for x in _sample_by_season(ctx, s)
                    if lo <= war1_of(x[0], x[1], x[2], x[3]) < hi]
            if not held:
                continue
            tot += 1
            if (valuefit_mse(held, lam_best, delta, ballast)
                    < valuefit_mse(held, best["lambda"], delta, ballast)):
                wins += 1
        print(f"    beats pooled lambda in {wins}/{tot} held-out seasons")
        fitted.append({"lo": lo, "hi": hi, "lambda": lam_best,
                       "n": len(rows), "status": status,
                       "seasonWins": f"{wins}/{tot}"})
    return fitted


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
    buckets = stage1b_bucketed_lambda(ctx, best)

    # --- stage 2: market multipliers at fixed engine constants ---
    # Bucketed-lambda verdict: REJECTED for shipping, re-checked on the
    # Steamer-basis sample 2026-07-30. Tier fits land within 0.03 of the
    # pooled 0.21 and beat it out of sample in 0/6, 3/6, 5/6 held-out
    # seasons - the majority of buckets favor pooled. The 4+ WAR bucket
    # (lambda 0.18, 5/6) is a MONITOR item, not shipped: one bucket of
    # three at n=221, and the effect is ~4% on long-control stars.
    set_config(best["lambda"], best["delta"], best["ballast"])
    rows = value_and_featurize(trades, ctx)

    # staged feature-group evaluation (pre-registered keep rule in header)
    def season_medians(featset):
        _, _, per = loso_score(rows, featset)
        return {s: rec["fitMedian"] for s, rec in per.items()}

    current = list(F0)
    print("\n=== staged feature-group evaluation ===")
    base_meds = season_medians(current)
    print(f"incumbent {current}: LOSO mean-of-medians "
          f"{np.mean(list(base_meds.values())):.4f}")
    kept_groups = []
    for gname, adds, drops in GROUPS:
        cand = [f for f in current if f not in drops] + adds
        cand_meds = season_medians(cand)
        wins = sum(cand_meds[s] < base_meds[s] for s in cand_meds)
        cur_m = float(np.mean(list(base_meds.values())))
        cand_m = float(np.mean(list(cand_meds.values())))
        keep = cand_m < cur_m and wins >= 6
        print(f"  {gname}: {cand_m:.4f} vs {cur_m:.4f}, wins {wins}/10 "
              f"-> {'KEEP' if keep else 'reject'}")
        if keep:
            current, base_meds = cand, cand_meds
            kept_groups.append(gname)

    featset = current
    score, base, per_season = loso_score(rows, featset)
    beta = ols(rows, featset=featset)
    multipliers = {f: round(float(math.exp(b)), 3)
                   for f, b in zip(featset, beta)}
    print(f"final feature set: {featset} (kept: {kept_groups or 'none'})")

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
        "features": featset,
        "betas": [round(float(b), 4) for b in beta],
        "multipliers": multipliers,
        "bestConfig": best,
        "lambdaByWarBucket": buckets,
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
