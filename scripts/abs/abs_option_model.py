"""ABS option model: what is a held challenge worth for the rest of the game?

Consumes data/abs_challenges_2026.json (every near-zone take with full state)
and data/abs_value_tables_2026.json (flip pricing). Produces C(k, T): the
marginal value, in leveraged runs, of holding your k-th challenge with T
half-innings of regulation remaining, plus the perception model that turns a
pitch location into a challenge-success confidence.

Model:
  1. Margin m per take, from the wronged side's perspective: m > 0 means a
     challenge would succeed (ruling boundary = ball-edge rule, center within
     1.4495in of the zone at the plate-midpoint plane).
  2. Perception: deciders observe x ~ N(m, sigma). Sigma is fit per side
     (fielding = catcher eyes, batting = batter eyes) by probit MLE on
     observed challenge decisions: P(challenge | m) = Phi((m - x*) / sigma).
     Rulings are deterministic, so failed challenges exist ONLY because of
     perception noise; sigma is identified by how challenge rates ramp with
     true margin.
  3. Confidence: p(x) = P(m > 0 | x) by Bayes against the empirical margin
     prior. This is "how sure the decider can be" after seeing x.
  4. DP: V(k, T) = A(C_k) + (1 - Pf(C_k)) V(k, T-1) + Pf(C_k) V(k-1, T-1),
     linearized per half-inning over the empirical opportunity stream, where
     the optimal policy challenges iff p(x) g >= (1 - p(x)) C_k, each record's
     gain g priced by the value engine at its true game state (so blowout
     opportunities contribute ~nothing, exactly as they should).
  Successful challenges are retained: only failures decrement k.

Decision rule output: challenge iff confidence >= p* = C(k,T) / (g + C(k,T)).

Output: data/abs_option_model_2026.json.

Usage: python3 scripts/abs/abs_option_model.py
"""

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import date

import abs_value_engine as ve

RULING_THR_IN = 1.4495       # ball-edge boundary at the midpoint plane
M_RANGE = 6.0                # margin grid half-width, inches
M_BIN = 0.1
REG_HALVES = 18              # regulation half-innings per game
EPS_COST = 0.01              # nuisance friction: never challenge for less than
                             # this expected edge (stops degenerate challenge-
                             # everything behavior when a challenge is free)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = os.path.join(REPO_ROOT, "data", "abs_challenges_2026.json")
TABLES = os.path.join(REPO_ROOT, "data", "abs_value_tables_2026.json")
DEFAULT_OUT = os.path.join(REPO_ROOT, "data", "abs_option_model_2026.json")


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_pdf(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


# ------------------------------------------------------------- opportunities

def load_opportunities(dataset_path, tables):
    """One opportunity per take, for the side wronged by the ORIGINAL call.

    Returns list of dicts: side ('bat'/'fld'), m (inches, >0 = challenge
    succeeds), g (leveraged-run gain if flipped), challenged, hadChallenge
    (wronged team had one remaining), T (half-innings of regulation left,
    current one included).
    """
    with open(dataset_path) as f:
        data = json.load(f)
    opps, side_mismatch = [], 0
    for r in data["records"]:
        if r["distMidIn"] is None or r.get("posPitcher"):
            continue
        if r["originalCall"] == "strike":
            side = "bat"                      # batting team wants a ball
            m = r["distMidIn"] - RULING_THR_IN
            wronged = r["batSide"]            # 'away'/'home' of batting team
        else:
            side = "fld"                      # fielding team wants a strike
            m = RULING_THR_IN - r["distMidIn"]
            wronged = "home" if r["batSide"] == "away" else "away"
        rem = r["remAway"] if wronged == "away" else r["remHome"]
        challenged = r["challenge"] is not None
        if challenged and r["challenge"].get("side") not in (wronged, None):
            side_mismatch += 1
            challenged = False
        v = ve.value_of_flip(r["balls"], r["strikes"], r["bases"], r["outs"],
                             r["inning"], r["half"], r["homeScore"] - r["awayScore"],
                             tables)
        inning = min(r["inning"], 9)
        T = 2 * (9 - inning) + (2 if r["half"] == "top" else 1)
        reg = (edge_region(r["pXmid"], r["pZmid"], r["szTop"], r["szBot"])
               if r.get("pXmid") is not None else "side")
        opps.append({"side": side, "m": m, "g": v["leveragedRuns"], "dre": v["dRE"],
                     "challenged": challenged, "hadChallenge": rem > 0,
                     "T": T, "playId": r["playId"],
                     "reg": reg, "cls": count_class(r["balls"], r["strikes"])})
    if side_mismatch:
        print(f"note: {side_mismatch} challenges by non-wronged side ignored in fit")
    return opps, data["meta"]["games"]


def count_half_innings(games):
    """Total half-innings ~= games x league average (17.7 with 51% skipped
    bottom 9ths and extras roughly cancelling)."""
    return games * 17.7


# ---------------------------------------------------------- perception model

REGIONS = ("side", "top", "bottom")
CLASSES = ("full", "b3", "s2", "other")


def count_class(balls, strikes):
    """Stakes class of the count: full, 3-ball, 2-strike, other. Challenge
    thresholds differ sharply by class (terminal calls get challenged far more
    readily), so pooling them biases the perception fit."""
    if balls == 3 and strikes == 2:
        return "full"
    if balls == 3:
        return "b3"
    if strikes == 2:
        return "s2"
    return "other"


def edge_region(px_mid, pz_mid, sz_top, sz_bot):
    """Which zone edge governs this pitch: side, top, or bottom. The binding
    edge is the one with the smallest clearance (works inside and outside).
    Low pitches are the hardest to judge; side pitches the easiest."""
    x = abs(px_mid) * 12.0
    z = pz_mid * 12.0
    dx = 8.5 - x
    dt = sz_top * 12.0 - z
    db = z - sz_bot * 12.0
    m = min(dx, dt, db)
    if m == dx:
        return "side"
    return "top" if m == dt else "bottom"


def fit_probit_rc(opps, side):
    """Per zone region: shared sigma, challenge threshold x* per count class.

    P(challenge | m, region, class) = Phi((m - x*[region,class]) / sigma[region])
    Classes partition the data within a region, so each x* maximizes
    independently given sigma.
    """
    out = {}
    for reg in REGIONS:
        hist = {c: defaultdict(lambda: [0, 0]) for c in CLASSES}
        for o in opps:
            if o["side"] != side or not o["hadChallenge"] or o["reg"] != reg:
                continue
            b = round(max(-M_RANGE, min(M_RANGE, o["m"])) / 0.05) * 0.05
            h = hist[o["cls"]][b]
            h[1] += 1
            h[0] += o["challenged"]
        best = None
        for sigma in [0.2 + 0.1 * i for i in range(49)]:
            ll_tot, xs = 0.0, {}
            for c in CLASSES:
                best_c = None
                for x_star in [0.0 + 0.1 * j for j in range(80)]:
                    ll = 0.0
                    for m, (ch, n) in hist[c].items():
                        p = min(max(phi((m - x_star) / sigma), 1e-9), 1 - 1e-9)
                        ll += ch * math.log(p) + (n - ch) * math.log(1.0 - p)
                    if best_c is None or ll > best_c[0]:
                        best_c = (ll, x_star)
                ll_tot += best_c[0]
                xs[c] = best_c[1]
            if best is None or ll_tot > best[0]:
                best = (ll_tot, sigma, xs)
        out[reg] = {"sigma": best[1], "xStar": best[2]}
    return out


def fit_probit(opps, side):
    """MLE grid fit of P(challenge | m) = Phi((m - x*) / sigma) for one side.

    Uses only takes where the wronged team had a challenge available.
    """
    hist = defaultdict(lambda: [0, 0])       # m bin -> [challenged, total]
    for o in opps:
        if o["side"] != side or not o["hadChallenge"]:
            continue
        b = round(max(-M_RANGE, min(M_RANGE, o["m"])) / 0.05) * 0.05
        hist[b][1] += 1
        hist[b][0] += o["challenged"]
    best = None
    for sigma in [0.2 + 0.02 * i for i in range(240)]:
        for x_star in [0.0 + 0.02 * i for i in range(350)]:
            ll = 0.0
            for m, (c, n) in hist.items():
                p = min(max(phi((m - x_star) / sigma), 1e-9), 1 - 1e-9)
                ll += c * math.log(p) + (n - c) * math.log(1.0 - p)
            if best is None or ll > best[0]:
                best = (ll, sigma, x_star)
    return {"sigma": best[1], "xStar": best[2], "logLik": best[0]}


def margin_prior(opps, side, reg=None):
    """Empirical density of true margins for one side (0.1in bins), optionally
    restricted to one zone region."""
    hist = defaultdict(int)
    n = 0
    for o in opps:
        if o["side"] != side or (reg is not None and o["reg"] != reg):
            continue
        b = round(max(-M_RANGE, min(M_RANGE, o["m"])) / M_BIN) * M_BIN
        hist[b] += 1
        n += 1
    return {b: c / n for b, c in hist.items()}


def posterior_grid(prior, sigma):
    """p(x) = P(m > 0 | perceived x) on a grid; monotone increasing in x."""
    xs = [round(-M_RANGE + 0.05 * i, 3) for i in range(int(2 * M_RANGE / 0.05) + 1)]
    grid = []
    for x in xs:
        num = den = 0.0
        for m, w in prior.items():
            lik = norm_pdf((x - m) / sigma)
            den += w * lik
            if m > 0:
                num += w * lik
        grid.append((x, num / den if den > 0 else (1.0 if x > 0 else 0.0)))
    # enforce monotonicity against tail-bin noise
    for i in range(1, len(grid)):
        if grid[i][1] < grid[i - 1][1]:
            grid[i] = (grid[i][0], grid[i - 1][1])
    return grid


def interp_grid(grid, x):
    """Linear interpolation on a [[x, p], ...] grid with uniform steps."""
    if x <= grid[0][0]:
        return grid[0][1]
    if x >= grid[-1][0]:
        return grid[-1][1]
    step = grid[1][0] - grid[0][0]
    i = int((x - grid[0][0]) / step)
    x0, p0 = grid[i]
    x1, p1 = grid[min(i + 1, len(grid) - 1)]
    return p0 + (p1 - p0) * (x - x0) / max(x1 - x0, 1e-9)


def look_grids(post, sigma, x_star):
    """Expected-confidence curves vs the TRUE margin m.

    pLook(m) = E[p(x) | x ~ N(m, sigma)]: what an attentive decider's
    confidence averages out to when the pitch is truly at m (used for grading
    unchallenged pitches - can this wrong call be identified at all?).

    pSel(m) = E[p(x) | x ~ N(m, sigma), x >= x*]: the same conditioned on the
    decider having seen enough to pull the trigger (league threshold x*). This
    is the right confidence for grading challenges that were actually made -
    evaluating the raw posterior at x = m ignores that selection and grades
    real challenges far too harshly.
    """
    p_at_xstar = interp_grid(post, x_star)
    ms = [round(-M_RANGE + 0.1 * i, 2) for i in range(int(2 * M_RANGE / 0.1) + 1)]
    p_look, p_sel = [], []
    for m in ms:
        num = den = num_s = den_s = 0.0
        for x, p in post:
            w = norm_pdf((x - m) / sigma)
            num += w * p
            den += w
            if x >= x_star:
                num_s += w * p
                den_s += w
        p_look.append([m, num / den if den > 0 else p_at_xstar])
        p_sel.append([m, num_s / den_s if den_s > 1e-12 else p_at_xstar])
    return p_look, p_sel


def x_threshold(post, c_over_g):
    """Smallest perceived x with p/(1-p) >= C/g; None if unreachable."""
    if c_over_g <= 0:
        return -M_RANGE
    lo, hi = 0, len(post) - 1
    if post[hi][1] / max(1 - post[hi][1], 1e-12) < c_over_g:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        p = post[mid][1]
        if p / max(1 - p, 1e-12) >= c_over_g:
            hi = mid
        else:
            lo = mid + 1
    return post[lo][0]


# ------------------------------------------------- score-conditioned DP for C

DIFF_RANGE = 12


def clamp_d(d):
    return max(-DIFF_RANGE, min(DIFF_RANGE, d))


def inning_from_T(T):
    return 9 - (T - (2 if T % 2 == 0 else 1)) // 2


def leverage_at(tables, T, d):
    """Average run-leverage multiplier for a team at score diff d (their runs
    minus opponent's) with T regulation half-innings left. Averages top/bottom
    and home/away orientations - the DP doesn't track which team bats when."""
    i = inning_from_T(T)
    g = (tables["G"][(i, "top", clamp_d(-d))] + tables["G"][(i, "bottom", clamp_d(d))]) / 2.0
    return g / tables["gAvg"]


def stream_curves(opps, halves_total, perception, posts):
    """One-dimensional reductions of the opportunity stream.

    Gains factor as dre x L (unleveraged value x state leverage), so the
    optimal policy at challenge cost C depends only on u = C / L. Returns
    per-team-half curves on a log-u grid: gainTilde(u) (unleveraged expected
    gain), failP(u), attempts(u). The DP then re-leverages: A = L x gainTilde.
    """
    us = [10.0 ** (-4.0 + 6.0 * i / 79.0) for i in range(80)]
    # cache x_threshold per (side, region) on a log-q grid (q = u / dre)
    qs = [10.0 ** (-5.0 + 9.0 * i / 199.0) for i in range(200)]
    xc_cache = {(side, reg): [x_threshold(posts[side][reg], q) for q in qs]
                for side in posts for reg in posts[side]}

    def xc_for(side, reg, q):
        import math as _m
        idx = int(round((_m.log10(max(q, 1e-5)) + 5.0) / 9.0 * 199.0))
        return xc_cache[(side, reg)][max(0, min(199, idx))]

    per_half = 1.0 / (2.0 * halves_total)
    gain = [0.0] * 80
    fail = [0.0] * 80
    att = [0.0] * 80
    for o in opps:
        dre = o["dre"]
        if dre <= 1e-6:
            continue
        sig = perception[o["side"]][o["reg"]]["sigma"]
        m = o["m"]
        win = m > 0
        for j, u in enumerate(us):
            xc = xc_for(o["side"], o["reg"], u / dre)
            if xc is None:
                continue
            r = phi((m - xc) / sig)
            att[j] += r
            if win:
                gain[j] += r * dre
            else:
                fail[j] += r
    return {"us": us,
            "gain": [v * per_half for v in gain],
            "fail": [v * per_half for v in fail],
            "att": [v * per_half for v in att]}


def curve_at(curve, key, u):
    """Log-linear interpolation of a stream curve at cost/leverage ratio u."""
    import math as _m
    us = curve["us"]
    lo, hi = us[0], us[-1]
    u = max(lo, min(hi, u))
    pos = (_m.log10(u) - _m.log10(lo)) / (_m.log10(hi) - _m.log10(lo)) * 79.0
    i = min(int(pos), 78)
    f = pos - i
    ys = curve[key]
    return ys[i] + (ys[i + 1] - ys[i]) * f


def build_dp_scored(curves, tables):
    """V(k, T, d) and C(k, T, d) over score diff d, with d evolving by the
    empirical fresh-half run distribution (random batting side)."""
    F = tables["runDist"][("000", 0)]
    diffs = list(range(-DIFF_RANGE, DIFF_RANGE + 1))
    V = {k: {0: {d: 0.0 for d in diffs}} for k in (0, 1, 2)}
    stats0 = {}
    for T in range(1, REG_HALVES + 1):
        for k in (0, 1, 2):
            V[k][T] = {}
        for d in diffs:
            V[0][T][d] = 0.0
            L = max(leverage_at(tables, T, d), 1e-6)
            for k in (1, 2):
                cost = max(V[k][T - 1][d] - V[k - 1][T - 1][d], EPS_COST)
                u = cost / L
                a = L * curve_at(curves, "gain", u)
                pf = curve_at(curves, "fail", u)
                nk = nk1 = 0.0
                for r, p in F.items():
                    for sgn in (1, -1):
                        d2 = clamp_d(d + sgn * r)
                        nk += 0.5 * p * V[k][T - 1][d2]
                        nk1 += 0.5 * p * V[k - 1][T - 1][d2]
                V[k][T][d] = a + (1.0 - pf) * nk + pf * nk1
                if d == 0:
                    stats0[(k, T)] = {"attempts": curve_at(curves, "att", u), "failP": pf}
    C = {k: {T: {d: V[k][T][d] - V[k - 1][T][d] for d in diffs}
             for T in range(REG_HALVES + 1)} for k in (1, 2)}
    return V, C, stats0


def build_dp(opps, halves_total, perception, posts):
    """V(k, T) and C(k, T) = V(k,T) - V(k-1,T), k in {1,2}, T in 0..18."""
    per_half = 1.0 / (2.0 * halves_total)    # each take is one team's opp

    def half_stats(cost):
        """(expected gain, fail prob, attempts) per team-half at challenge cost."""
        gain = fail = att = 0.0
        for o in opps:
            g = o["g"]
            if g <= 0 and o["m"] <= 0:
                continue
            sig = perception[o["side"]]["sigma"]
            xc = x_threshold(posts[o["side"]], cost / g if g > 0 else float("inf"))
            if xc is None:
                continue
            r = phi((o["m"] - xc) / sig)
            att += r
            if o["m"] > 0:
                gain += r * g
            else:
                fail += r
        return gain * per_half, fail * per_half, att * per_half

    V = {0: [0.0] * (REG_HALVES + 1), 1: [0.0] * (REG_HALVES + 1),
         2: [0.0] * (REG_HALVES + 1)}
    stats_log = {}
    for T in range(1, REG_HALVES + 1):
        for k in (1, 2):
            cost = max(V[k][T - 1] - V[k - 1][T - 1], EPS_COST)
            a, pf, att = half_stats(cost)
            V[k][T] = a + (1.0 - pf) * V[k][T - 1] + pf * V[k - 1][T - 1]
            stats_log[(k, T)] = {"cost": cost, "gain": a, "failP": pf, "attempts": att}
    C = {k: [V[k][T] - V[k - 1][T] for T in range(REG_HALVES + 1)] for k in (1, 2)}
    return V, C, stats_log


# ------------------------------------------------------------------- output

def main():
    ap = argparse.ArgumentParser(description="Build ABS option-value model")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--tables", default=TABLES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    tables = ve.tables_from_json(args.tables)
    opps, n_games = load_opportunities(args.dataset, tables)
    halves = count_half_innings(n_games)
    print(f"{len(opps)} opportunities from {n_games} games")

    perception, posts, priors = {}, {}, {}
    pooled = {}
    looks, sels = {}, {}
    for side in ("fld", "bat"):
        pooled[side] = fit_probit(opps, side)
        perception[side] = fit_probit_rc(opps, side)
        priors[side], posts[side], looks[side], sels[side] = {}, {}, {}, {}
        for reg in REGIONS:
            pr = perception[side][reg]
            priors[side][reg] = margin_prior(opps, side, reg)
            posts[side][reg] = posterior_grid(priors[side][reg], pr["sigma"])
            sels[side][reg] = {}
            for cls in CLASSES:
                lk, sl = look_grids(posts[side][reg], pr["sigma"], pr["xStar"][cls])
                sels[side][reg][cls] = sl
                if cls == "other":
                    looks[side][reg] = lk
            print(f"perception[{side}|{reg}]: sigma={pr['sigma']:.2f}in x* "
                  + " ".join(f"{c}={pr['xStar'][c]:.1f}" for c in CLASSES))
        chal = [o for o in opps if o["side"] == side and o["challenged"]]
        pred = sum(interp_grid(sels[side][o["reg"]][o["cls"]], o["m"]) for o in chal) / len(chal)
        actual = sum(1 for o in chal if o["m"] > 0) / len(chal)
        print(f"self-check [{side}]: mean selection-conditioned confidence on "
              f"actual challenges {pred:.3f} vs observed success {actual:.3f}")

    curves = stream_curves(opps, halves, perception, posts)
    V, Cg, stats0 = build_dp_scored(curves, tables)
    C = {k: [Cg[k][T][0] for T in range(REG_HALVES + 1)] for k in (1, 2)}

    out = {
        "meta": {"generated": date.today().isoformat(), "games": n_games,
                 "opportunities": len(opps), "rulingThrIn": RULING_THR_IN,
                 "regHalves": REG_HALVES},
        "perception": perception,
        "perceptionPooled": pooled,
        "marginPrior": {f"{s}|{r}": {f"{b:.1f}": round(w, 6)
                                     for b, w in sorted(priors[s][r].items())}
                        for s in priors for r in priors[s]},
        "posterior": {f"{s}|{r}": [[x, round(p, 5)] for x, p in posts[s][r]]
                      for s in posts for r in posts[s]},
        "pLook": {f"{s}|{r}": [[m, round(p, 5)] for m, p in looks[s][r]]
                  for s in looks for r in looks[s]},
        "pSel": {f"{s}|{r}|{c}": [[m, round(p, 5)] for m, p in sels[s][r][c]]
                 for s in sels for r in sels[s] for c in sels[s][r]},
        "C": {str(k): [round(c, 5) for c in C[k]] for k in C},
        "Cgrid": {f"{k}|{T}|{d}": round(Cg[k][T][d], 5)
                  for k in (1, 2) for T in range(REG_HALVES + 1)
                  for d in range(-DIFF_RANGE, DIFF_RANGE + 1)},
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")

    # ------------------------------------------------------------ validation
    print("\nC(k, T, d=0) leveraged runs (T = regulation half-innings remaining):")
    print("  T:      " + " ".join(f"{T:5d}" for T in (18, 14, 10, 6, 4, 2, 1)))
    for k in (1, 2):
        row = " ".join(f"{C[k][T]:.3f}" for T in (18, 14, 10, 6, 4, 2, 1))
        print(f"  C(k={k}): {row}")
    print("\nC(1, T, d) across score margins (blowout conditioning):")
    print("  d:      " + " ".join(f"{d:5d}" for d in (0, 2, 4, 6, 8, -4, -8)))
    for T in (14, 6):
        row = " ".join(f"{Cg[1][T][d]:.3f}" for d in (0, 2, 4, 6, 8, -4, -8))
        print(f"  T={T:2d}:   {row}")
    # forward-simulate the k chain at d=0 for honest usage numbers
    pk = {2: 1.0, 1: 0.0, 0: 0.0}
    att_game = fail_game = 0.0
    for T in range(REG_HALVES, 0, -1):
        nxt = {2: 0.0, 1: 0.0, 0: pk[0]}
        for k in (1, 2):
            s = stats0[(k, T)]
            att_game += pk[k] * s["attempts"]
            fail_game += pk[k] * s["failP"]
            nxt[k] += pk[k] * (1.0 - s["failP"])
            nxt[k - 1] += pk[k] * s["failP"]
        pk = nxt
    succ = att_game - fail_game
    print(f"\noptimal-policy usage/team-game (close game): {att_game:.2f} attempts, "
          f"{100 * succ / att_game:.0f}% success (league: ~2.1 attempts, ~61%)")

    print("\nbreak-even confidence p* = C/(g+C):")
    demos = [("0-0 pitch, neutral, game start (k=2)", 0.078, C[2][18]),
             ("0-0 pitch, neutral, game start (k=1)", 0.078, C[1][18]),
             ("3-2 loaded, bottom 9 tie (k=1)", 5.02, Cg[1][1][0]),
             ("2-2 pitch, tie 8th inning (k=1)", 0.140, Cg[1][4][0]),
             ("2-2 pitch, 8th inning down 8 (k=1)", 0.011, Cg[1][4][-8])]
    for label, g, c in demos:
        print(f"  {label}: p* = {c / (g + c):.2f}")


if __name__ == "__main__":
    main()
