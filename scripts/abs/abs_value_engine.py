"""ABS value engine: prices a flipped ball/strike call in leveraged runs.

Builds, from 2026 MLB feed data (pure Python, no deps beyond requests):
  1. RE24: base-out run expectancy (innings 1-8, completed halves only).
  2. Through-count run values RV(b,s): expected whole-PA run impact given the
     PA reached count (b,s). A flipped non-terminal call is worth
     RV(after-strike) - RV(after-ball).
  3. Terminal call values by base-out: ball four = forced-advance walk value
     from RE24 (runs force in with bases loaded); strike three = out value
     (inning-ending at 2 outs).
  4. Win-probability dynamic program from empirical half-inning run
     distributions (ghost-runner distribution for extras, walk-off logic in
     the 9th and later), yielding G(inning, half, diff) = the batting team's
     win-probability value of one run. LI = G / league-average G.

Challenge currency (Wally's call, 2026-07-20): run expectancy x leverage
index, not raw WP. value_of_flip() returns delta-RE (batting team gains when
a strike becomes a ball), LI, and their product in leveraged runs.

Known simplification: the continuation branch of a terminal flip uses the
league-average RV for the resulting count, not a base-out-conditioned RV
(e.g. RV(3-2) with bases loaded is really higher than league RV(3-2)); the
terminal branch itself is exact. Second-order; revisit if it matters.

Output: data/abs_value_tables_2026.json (small; consumed by the option-value
model and the challenge matrix).

Usage:
    python3 scripts/abs/abs_value_engine.py                  # build + validate
    python3 scripts/abs/abs_value_engine.py --cache <path>   # reuse event cache
"""

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import requests

import abs_challenges as ac

MAX_K = 13          # cap on runs counted in a half-inning distribution
DIFF_CAP = 12       # score-diff clamp for WP / G tables
EXTRA_INNING = 10   # innings >= this use ghost-runner distributions
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(REPO_ROOT, "data", "abs_value_tables_2026.json")


# ---------------------------------------------------------------- extraction

def extract_game(feed):
    """Compact per-play event list: state at play start, runs, counts passed."""
    live = feed["liveData"]
    plays_out = []
    score = {"away": 0, "home": 0}
    prev_half = None
    bases = {}
    outs = 0
    for play in live["plays"]["allPlays"]:
        about = play["about"]
        half_key = (about["inning"], about["halfInning"])
        if half_key != prev_half:
            bases, outs, prev_half = {}, 0, half_key
        bat = "away" if about["halfInning"] == "top" else "home"
        runners = play.get("runners", [])
        runs = len({e["details"]["runner"]["id"] for e in runners
                    if e.get("movement", {}).get("end") == "score"})
        events = play.get("playEvents", [])
        n_pitch = sum(1 for e in events if e.get("isPitch"))
        # counts this PA sat at awaiting a pitch: 0-0 plus every event's
        # post count except the PA-ending event's
        counts = {(0, 0)}
        for ev in events[:-1]:
            c = ev.get("count")
            if c is not None:
                counts.add((min(c.get("balls", 0), 3), min(c.get("strikes", 0), 2)))
        post_outs = play.get("count", {}).get("outs", outs)
        plays_out.append({
            "inning": about["inning"],
            "half": about["halfInning"],
            "outs": outs,
            "outsAfter": post_outs,
            "bases": "".join("1" if b in bases else "0" for b in ("1B", "2B", "3B")),
            "runs": runs,
            "diff": score["home"] - score["away"],
            "counts": sorted(counts) if n_pitch else [],
        })
        movements = sorted((e for e in runners if e.get("movement")),
                           key=lambda e: e["details"].get("playIndex", 0))
        i = 0
        while i < len(movements):
            j = i
            pi = movements[i]["details"].get("playIndex", 0)
            while j < len(movements) and movements[j]["details"].get("playIndex", 0) == pi:
                j += 1
            ac.apply_movements(bases, movements[i:j])
            i = j
        score[bat] += runs
        outs = post_outs
    return {"plays": plays_out}


def fetch_all_games(start, end, workers=6):
    session = requests.Session()
    pks = ac.get_final_game_pks(session, start, end)
    games = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(session.get, ac.FEED_URL.format(pk=pk), timeout=30): pk
                   for pk in pks}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                games.append(extract_game(fut.result().json()))
            except Exception as e:
                print(f"  FAIL {futures[fut]}: {e}")
            if i % 300 == 0 or i == len(pks):
                print(f"  {i}/{len(pks)} games extracted")
    return games


# ------------------------------------------------------------------- tables

def group_halves(game):
    halves = defaultdict(list)
    for p in game["plays"]:
        halves[(p["inning"], p["half"])].append(p)
    return halves


def build_state_tables(games):
    """RE24 + P(runs remaining in half | base, out), innings 1-8 complete halves."""
    re_sum, re_n = defaultdict(float), defaultdict(int)
    hist = defaultdict(lambda: defaultdict(int))
    for g in games:
        for (inning, _half), plist in group_halves(g).items():
            if inning > 8 or plist[-1]["outsAfter"] != 3:
                continue
            total = sum(p["runs"] for p in plist)
            cum = 0
            for p in plist:
                remaining = min(total - cum, MAX_K)
                key = (p["bases"], p["outs"])
                re_sum[key] += total - cum
                re_n[key] += 1
                hist[key][remaining] += 1
                cum += p["runs"]
    re24 = {k: re_sum[k] / re_n[k] for k in re_n}
    dists = {k: {r: n / sum(h.values()) for r, n in h.items()} for k, h in hist.items()}
    return re24, dists, dict(re_n)


RV288_SHRINK_K = 200


def build_count_values(games, re24):
    """RV(b,s) = mean whole-PA run impact (dRE24 + runs) over PAs through (b,s).

    Also returns RV288: the same value conditioned on (count, bases, outs),
    shrunk toward the league count value with k=200 pseudo-observations. This
    matters most for 3-ball / 2-strike counts with runners on, where the
    looming walk/strikeout makes the count worth more than league average.
    """
    rv_sum, rv_n = defaultdict(float), defaultdict(int)
    rv288_sum, rv288_n = defaultdict(float), defaultdict(int)
    for g in games:
        for (inning, _half), plist in group_halves(g).items():
            if inning > 8 or plist[-1]["outsAfter"] != 3:
                continue
            for i, p in enumerate(plist):
                if not p["counts"]:
                    continue
                start = re24[(p["bases"], p["outs"])]
                if i + 1 < len(plist):
                    nxt = plist[i + 1]
                    end = re24[(nxt["bases"], nxt["outs"])]
                else:
                    end = 0.0
                rv = end - start + p["runs"]
                for b, s in p["counts"]:
                    rv_sum[(b, s)] += rv
                    rv_n[(b, s)] += 1
                    key288 = (b, s, p["bases"], p["outs"])
                    rv288_sum[key288] += rv
                    rv288_n[key288] += 1
    rv = {c: rv_sum[c] / rv_n[c] for c in rv_n}
    rv288 = {}
    for (b, s) in rv:
        for bases in ("000", "100", "010", "001", "110", "101", "011", "111"):
            for out in range(3):
                key = (b, s, bases, out)
                n = rv288_n.get(key, 0)
                mean = rv288_sum.get(key, 0.0) / n if n else 0.0
                rv288[key] = (n * mean + RV288_SHRINK_K * rv[(b, s)]) / (n + RV288_SHRINK_K)
    return rv, dict(rv_n), rv288


def walk_bases(bases):
    """Base state and forced runs after a walk (forced advances only)."""
    first, second, third = (bases[0] == "1", bases[1] == "1", bases[2] == "1")
    runs = 1 if (first and second and third) else 0
    new_third = third if not (first and second) else True
    new_second = second if not first else True
    return "1" + ("1" if new_second else "0") + ("1" if new_third else "0"), runs


def terminal_values(re24):
    bb, k = {}, {}
    for bases in ("000", "100", "010", "001", "110", "101", "011", "111"):
        for out in range(3):
            cur = re24[(bases, out)]
            nb, runs = walk_bases(bases)
            bb[(bases, out)] = runs + re24[(nb, out)] - cur
            k[(bases, out)] = (re24[(bases, out + 1)] if out < 2 else 0.0) - cur
    return bb, k


# --------------------------------------------------------- win probability DP

def clamp(d):
    return max(-DIFF_CAP, min(DIFF_CAP, d))


def build_wp(dist_fresh, dist_ghost):
    """W[(inning, half)][diff] = P(home wins) at the START of that half.

    Regulation innings 1-9 by backward recursion; extras via a generic
    ghost-runner inning solved as W_extra = 0.5 at tie (identical teams) with
    explicit walk-off handling for bottom halves of the 9th and later.
    """
    F = dist_fresh
    Fg = dist_ghost
    w_extra_tie = 0.5  # symmetric identical-team fixed point

    def bot_final(d, dist):
        """Home batting in a potentially game-ending bottom half, diff d<=0."""
        win = sum(p for k, p in dist.items() if k > -d)
        tie = dist.get(-d, 0.0)
        return win + tie * w_extra_tie

    diffs = range(-DIFF_CAP, DIFF_CAP + 1)
    W = {}
    # generic extra inning
    w_bot_x = {d: (1.0 if d > 0 else bot_final(d, Fg)) for d in diffs}
    w_top_x = {d: sum(p * w_bot_x[clamp(d - k)] for k, p in Fg.items()) for d in diffs}
    W[(EXTRA_INNING, "bottom")] = w_bot_x
    W[(EXTRA_INNING, "top")] = w_top_x

    w_bot9 = {d: (1.0 if d > 0 else bot_final(d, F)) for d in diffs}
    W[(9, "bottom")] = w_bot9
    W[(9, "top")] = {d: sum(p * w_bot9[clamp(d - k)] for k, p in F.items()) for d in diffs}

    nxt = W[(9, "top")]
    for inning in range(8, 0, -1):
        w_bot = {d: sum(p * nxt[clamp(d + k)] for k, p in F.items()) for d in diffs}
        w_top = {d: sum(p * w_bot[clamp(d - k)] for k, p in F.items()) for d in diffs}
        W[(inning, "bottom")] = w_bot
        W[(inning, "top")] = w_top
        nxt = w_top
    return W, w_extra_tie


def build_g(W, w_extra_tie):
    """G[(inning, half, diff)] = batting team's win-prob value of +1 run this
    half, evaluated through the end-of-half continuation."""
    diffs = range(-DIFF_CAP, DIFF_CAP + 1)

    def cont_after_top(inning, d):    # away team's perspective
        if inning >= 9:
            key = (min(inning, EXTRA_INNING), "bottom") if d <= 0 else None
            home = 1.0 if d > 0 else W[key][clamp(d)]
        else:
            home = W[(inning, "bottom")][clamp(d)]
        return 1.0 - home

    def cont_after_bot(inning, d):    # home team's perspective
        if inning >= 9:
            return 1.0 if d > 0 else (0.0 if d < 0 else w_extra_tie)
        return W[(inning + 1, "top")][clamp(d)]

    G = {}
    for inning in list(range(1, 10)) + [EXTRA_INNING]:
        for d in diffs:
            G[(inning, "top", d)] = cont_after_top(inning, d - 1) - cont_after_top(inning, d)
            G[(inning, "bottom", d)] = cont_after_bot(inning, d + 1) - cont_after_bot(inning, d)
    return G


def g_average(games, G):
    """Occupancy-weighted league-average G (the LI denominator)."""
    total, n = 0.0, 0
    for g in games:
        for p in g["plays"]:
            if not p["counts"]:
                continue
            inning = min(p["inning"], EXTRA_INNING)
            total += G[(inning, p["half"], clamp(p["diff"]))]
            n += 1
    return total / n


# ---------------------------------------------------------------- assembly

def cont_bat(inning, half, d, tables):
    """Batting team's win prob once the current half ends at home-away diff d."""
    W = tables["W"]
    d = clamp(d)
    if half == "top":
        if inning >= 9:
            home = 1.0 if d > 0 else W[(min(inning, EXTRA_INNING), "bottom")][d]
        else:
            home = W[(inning, "bottom")][d]
        return 1.0 - home
    if inning >= 9:
        return 1.0 if d > 0 else (0.0 if d < 0 else tables["wExtraTie"])
    return W[(inning + 1, "top")][d]


def wp_mid_batting(inning, half, diff, bases, out, tables):
    """Batting team's win prob mid-half, from the empirical distribution of
    runs scored in the remainder of the half from (bases, out). Walk-off
    truncation is handled by cont_bat (any winning total counts as a win)."""
    if out >= 3:
        return cont_bat(inning, half, diff, tables)
    total = 0.0
    for k, p in tables["runDist"][(bases, out)].items():
        d2 = diff - k if half == "top" else diff + k
        total += p * cont_bat(inning, half, d2, tables)
    return total


def run_slope(inning, half, diff, bases, out, tables):
    """Win-probability value of one more run to the BATTING team, right here.

    A local derivative of wp_mid_batting at this exact base-out-score state.
    This is what makes the two branches of a flip commensurable: terminal
    outcomes are priced as exact WP transitions, so a continuation outcome
    priced as runs must be converted with the same local WP curve rather than
    with G (which is a half-end, fresh-inning average and disagrees with the
    exact calculation whenever WP is curved -- i.e. whenever a team is ahead).
    """
    sgn = -1 if half == "top" else 1
    base = wp_mid_batting(inning, half, diff, bases, out, tables)
    up = wp_mid_batting(inning, half, clamp(diff + sgn), bases, out, tables)
    return up - base, base


def value_of_flip(b, s, bases, out, inning, half, diff, tables):
    """Price a called-strike vs called-ball flip at a given state.

    Both branches are valued in win probability off the same local curve:
    terminal outcomes (ball four / strike three) as exact WP transitions, and
    continuation outcomes as their base-out-conditioned count value (RV288)
    times the local WP-per-run slope at this state. Result is expressed in
    leveraged runs (delta-WP / gAvg).

    Returns dict with leveragedRuns (batting-team gain if the call is a BALL;
    >= 0), li (local run leverage), and dRE (the run-equivalent).
    diff is home minus away score at the time of the pitch.
    """
    rv288 = tables["countRV288"]
    slope, wp_base = run_slope(inning, half, diff, bases, out, tables)

    if b == 3:   # ball four: exact WP after the forced walk
        nb, runs = walk_bases(bases)
        d2 = diff - runs if half == "top" else diff + runs
        ball_wp = wp_mid_batting(inning, half, d2, nb, out, tables) - wp_base
    else:
        ball_wp = rv288[(b + 1, s, bases, out)] * slope

    if s == 2:   # strike three: exact WP after the out
        strike_wp = wp_mid_batting(inning, half, diff, bases, out + 1, tables) - wp_base
    else:
        strike_wp = rv288[(b, s + 1, bases, out)] * slope

    li = slope / tables["gAvg"]
    leveraged = (ball_wp - strike_wp) / tables["gAvg"]
    return {"dRE": leveraged / li if li > 1e-9 else 0.0, "li": li,
            "leveragedRuns": leveraged}


def build_all(games):
    re24, dists, re_n = build_state_tables(games)
    count_rv, rv_n, rv288 = build_count_values(games, re24)
    bb, k = terminal_values(re24)
    W, w_extra_tie = build_wp(dists[("000", 0)], dists[("010", 0)])
    G = build_g(W, w_extra_tie)
    g_avg = g_average(games, G)
    return {"re24": re24, "re24N": re_n, "countRV": count_rv, "countRVN": rv_n,
            "countRV288": rv288, "runDist": dists, "bbValue": bb, "kValue": k,
            "W": W, "wExtraTie": w_extra_tie, "G": G, "gAvg": g_avg}


def tables_to_json(t):
    return {
        "meta": {"generated": date.today().isoformat(), "diffCap": DIFF_CAP,
                 "extraInning": EXTRA_INNING, "gAvg": t["gAvg"]},
        "re24": {f"{b}|{o}": round(v, 4) for (b, o), v in t["re24"].items()},
        "re24N": {f"{b}|{o}": n for (b, o), n in t["re24N"].items()},
        "countRV": {f"{b}-{s}": round(v, 4) for (b, s), v in t["countRV"].items()},
        "countRVN": {f"{b}-{s}": n for (b, s), n in t["countRVN"].items()},
        "bbValue": {f"{b}|{o}": round(v, 4) for (b, o), v in t["bbValue"].items()},
        "kValue": {f"{b}|{o}": round(v, 4) for (b, o), v in t["kValue"].items()},
        "G": {f"{i}|{h}|{d}": round(v, 6) for (i, h, d), v in t["G"].items()},
        "countRV288": {f"{b}-{s}|{ba}|{o}": round(v, 4)
                       for (b, s, ba, o), v in t["countRV288"].items()},
        "runDist": {f"{ba}|{o}": {str(k): round(p, 6) for k, p in d.items()}
                    for (ba, o), d in t["runDist"].items()},
        "W": {f"{i}|{h}|{d}": round(v, 6)
              for (i, h), row in t["W"].items() for d, v in row.items()},
        "wExtraTie": t["wExtraTie"],
    }


def tables_from_json(path):
    with open(path) as f:
        j = json.load(f)
    t = {"gAvg": j["meta"]["gAvg"]}
    t["re24"] = {(k.split("|")[0], int(k.split("|")[1])): v for k, v in j["re24"].items()}
    t["countRV"] = {(int(k.split("-")[0]), int(k.split("-")[1])): v
                    for k, v in j["countRV"].items()}
    t["bbValue"] = {(k.split("|")[0], int(k.split("|")[1])): v for k, v in j["bbValue"].items()}
    t["kValue"] = {(k.split("|")[0], int(k.split("|")[1])): v for k, v in j["kValue"].items()}
    t["G"] = {}
    for k, v in j["G"].items():
        i, h, d = k.split("|")
        t["G"][(int(i), h, int(d))] = v
    t["countRV288"] = {}
    for k, v in j["countRV288"].items():
        cnt, ba, o = k.split("|")
        b, s = cnt.split("-")
        t["countRV288"][(int(b), int(s), ba, int(o))] = v
    t["runDist"] = {}
    for k, d in j["runDist"].items():
        ba, o = k.split("|")
        t["runDist"][(ba, int(o))] = {int(r): p for r, p in d.items()}
    t["W"] = defaultdict(dict)
    for k, v in j["W"].items():
        i, h, d = k.split("|")
        t["W"][(int(i), h)][int(d)] = v
    t["W"] = dict(t["W"])
    t["wExtraTie"] = j["wExtraTie"]
    return t


def main():
    ap = argparse.ArgumentParser(description="Build ABS value tables")
    ap.add_argument("--start", default="2026-03-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--cache", default="", help="path to event cache json (reused if exists)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    if args.cache and os.path.exists(args.cache):
        with open(args.cache) as f:
            games = json.load(f)
        # JSON round-trip turns count tuples into lists
        for g in games:
            for p in g["plays"]:
                p["counts"] = [tuple(c) for c in p["counts"]]
        print(f"loaded {len(games)} games from cache")
    else:
        games = fetch_all_games(args.start, args.end, args.workers)
        if args.cache:
            with open(args.cache, "w") as f:
                json.dump(games, f, separators=(",", ":"))
    t = build_all(games)
    with open(args.out, "w") as f:
        json.dump(tables_to_json(t), f, indent=1)
    print(f"wrote {args.out}")

    # ------------------------------------------------------------- validation
    print("\nRE24 (bases | 0/1/2 outs):")
    for bases in ("000", "100", "010", "001", "110", "101", "011", "111"):
        row = " ".join(f"{t['re24'][(bases, o)]:.3f}" for o in range(3))
        print(f"  {bases}: {row}")
    print("\nCount RV (through-count, runs vs PA start):")
    for b in range(4):
        row = " ".join(f"{b}-{s}:{t['countRV'][(b, s)]:+.3f}" for s in range(3))
        print(f"  {row}")
    print(f"\ngAvg (league mean win value of a run): {t['gAvg']:.4f}")
    for label, st in [("1st inning top, tie", (1, "top", 0)),
                      ("9th bottom, tie", (9, "bottom", 0)),
                      ("9th bottom, down 4", (9, "bottom", -4)),
                      ("7th top, up 1 (home persp)", (7, "top", 1))]:
        print(f"  LI {label}: {t['G'][st] / t['gAvg']:.2f}")

    print("\nScenario demos (dRE = batting team's gain if call is ball):")
    demos = [
        ("0-0, empty, 0 out, 1st top, tie", (0, 0, "000", 0, 1, "top", 0)),
        ("3-2, loaded, 2 out, 9th bottom, tie", (3, 2, "111", 2, 9, "bottom", 0)),
        ("1-1, R2, 1 out, 5th top, +2 home", (1, 1, "010", 1, 5, "top", 2)),
    ]
    for label, (b, s, bases, out, inn, half, diff) in demos:
        v = value_of_flip(b, s, bases, out, inn, half, diff, t)
        print(f"  {label}: dRE={v['dRE']:.3f} LI={v['li']:.2f} "
              f"leveraged={v['leveragedRuns']:.3f}")


if __name__ == "__main__":
    main()
