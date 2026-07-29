"""Bake-off for the ABS skill and value metrics.

Every candidate is scored on the same players with the same decision stream,
on four criteria that actually matter for a public leaderboard:

  1. chronological split-half r  -- does the first half of a player's season
     predict the second? This is the real "is it skill" test.
  2. random split-half r         -- an upper bound (shares game context, so it
     flatters); the gap to chronological is drift + context leakage.
  3. corr with aggressiveness    -- how much of the metric is just "how often
     do you challenge?" Near zero is the goal.
  4. corr with opportunity count -- a talent rate should not reward playing time.

Also reports Spearman-Brown full-season reliability and how many players are
rankable, plus a value-metric section (raw vs winsorized) and a sensitivity
check on the leverage basis used by the option-value DP.

Usage: python3 scripts/abs_metric_bakeoff.py
"""

import json
import math
import os
import random
import statistics
from collections import defaultdict

from scipy.stats import pearsonr, spearmanr

import abs_value_engine as ve
from abs_option_model import count_class, edge_region, phi

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREF = 0.434
G_MIN, M_MAX = 0.05, 2.5
MIN_CH = 3          # challenges needed before a selection metric is defined


def interp(g, x):
    if x <= g[0][0]:
        return g[0][1]
    if x >= g[-1][0]:
        return g[-1][1]
    st = g[1][0] - g[0][0]
    i = int((x - g[0][0]) / st)
    a, b = g[i], g[min(i + 1, len(g) - 1)]
    return a[1] + (b[1] - a[1]) * (x - a[0]) / max(b[0] - a[0], 1e-9)


def load():
    d = json.load(open(os.path.join(REPO, "data", "abs_challenges_2026.json")))
    o = json.load(open(os.path.join(REPO, "data", "abs_option_model_2026.json")))
    t = ve.tables_from_json(os.path.join(REPO, "data", "abs_value_tables_2026.json"))
    thr = o["meta"]["rulingThrIn"]
    Cg = {}
    for k, v in o["Cgrid"].items():
        a, b, c = k.split("|")
        Cg[(int(a), int(b), int(c))] = v
    per = defaultdict(list)
    for r in d["records"]:
        if r["distMidIn"] is None or r.get("posPitcher"):
            continue
        if r["originalCall"] == "strike":
            side, m, wr, oid = "bat", r["distMidIn"] - thr, r["batSide"], r["batterId"]
        else:
            side, m = "fld", thr - r["distMidIn"]
            wr = "home" if r["batSide"] == "away" else "away"
            oid = r["catcherId"]
        if oid is None:
            continue
        rem = r["remAway"] if wr == "away" else r["remHome"]
        if rem <= 0:
            continue
        dt = (r["awayScore"] - r["homeScore"]) if wr == "away" else (r["homeScore"] - r["awayScore"])
        v = ve.value_of_flip(r["balls"], r["strikes"], r["bases"], r["outs"],
                             r["inning"], r["half"], r["homeScore"] - r["awayScore"], t)
        g = v["leveragedRuns"]
        if g < G_MIN or abs(m) > M_MAX:
            continue
        reg = edge_region(r["pXmid"], r["pZmid"], r["szTop"], r["szBot"])
        cls = count_class(r["balls"], r["strikes"])
        T = 2 * (9 - min(r["inning"], 9)) + (2 if r["half"] == "top" else 1)
        C = Cg[(max(1, min(2, rem)), T, max(-12, min(12, dt)))]
        ch = r["challenge"]
        oc = (ch is not None and ch.get("side") == wr
              and ((side == "bat" and ch["role"] == "batter")
                   or (side == "fld" and ch["role"] == "fielder")))
        p = interp(o["pSel"][f"{side}|{reg}|{cls}"], m) if oc else interp(o["pLook"][f"{side}|{reg}"], m)
        per[(side, oid)].append({"date": r["date"], "p": p, "oc": oc, "m": m,
                                 "g": g, "C": C})
    return per


# ----------------------------------------------------------------- candidates

def m_current(e):
    return statistics.mean((1 if ((x["p"] >= PREF) == x["oc"]) else -1) * abs(x["p"] - PREF) for x in e)


def m_balanced(e):
    """Same correctness rule, but the two classes get equal weight, so being
    right on the abundant low-confidence pitches can't carry the score."""
    hi = [x for x in e if x["p"] >= PREF]
    lo = [x for x in e if x["p"] < PREF]
    if not hi or not lo:
        return None
    f = lambda s: statistics.mean((1 if ((x["p"] >= PREF) == x["oc"]) else -1) * abs(x["p"] - PREF) for x in s)
    return 0.5 * f(hi) + 0.5 * f(lo)


def m_disc(e):
    """Mean confidence of the pitches you chose to challenge, minus your own
    opportunity baseline. Pure selection, independent of how often you go."""
    ch = [x["p"] for x in e if x["oc"]]
    if len(ch) < MIN_CH:
        return None
    return statistics.mean(ch) - statistics.mean(x["p"] for x in e)


def m_disc_margin(e):
    """Same idea in inches rather than through the posterior."""
    ch = [x["m"] for x in e if x["oc"]]
    if len(ch) < MIN_CH:
        return None
    return statistics.mean(ch) - statistics.mean(x["m"] for x in e)


def m_disc_z(e):
    """Discrimination standardized by the spread of opportunities the player
    actually faced, so an easy or hard slate doesn't inflate it."""
    ch = [x["p"] for x in e if x["oc"]]
    if len(ch) < MIN_CH:
        return None
    allp = [x["p"] for x in e]
    sd = statistics.pstdev(allp)
    if sd < 1e-6:
        return None
    return (statistics.mean(ch) - statistics.mean(allp)) / sd


def m_auc(e):
    ch = [x["p"] for x in e if x["oc"]]
    no = [x["p"] for x in e if not x["oc"]]
    if len(ch) < MIN_CH or len(no) < MIN_CH:
        return None
    w = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in ch for b in no)
    return w / (len(ch) * len(no))


def m_sigma(e):
    """Fitted perception noise (inches), sign-flipped so higher = sharper.
    A direct read on eyesight rather than on decision policy."""
    n_ch = sum(1 for x in e if x["oc"])
    if n_ch < MIN_CH:
        return None
    bins = defaultdict(lambda: [0, 0])
    for x in e:
        b = round(max(-6.0, min(6.0, x["m"])) / 0.25) * 0.25
        bins[b][1] += 1
        bins[b][0] += x["oc"]
    best = None
    for sig in [0.4 + 0.2 * i for i in range(24)]:
        lo, hi = -6.0, 10.0
        for _ in range(40):
            mid = (lo + hi) / 2
            pred = sum(n * phi((mm - mid) / sig) for mm, (_c, n) in bins.items())
            if pred > n_ch:
                lo = mid
            else:
                hi = mid
        xs = (lo + hi) / 2
        ll = 0.0
        for mm, (c, n) in bins.items():
            pp = min(max(phi((mm - xs) / sig), 1e-9), 1 - 1e-9)
            ll += c * math.log(pp) + (n - c) * math.log(1 - pp)
        if best is None or ll > best[0]:
            best = (ll, sig)
    return -best[1]


CANDS = [("current (fixed bar)", m_current), ("balanced classes", m_balanced),
         ("discrimination", m_disc), ("discrimination (z)", m_disc_z),
         ("discrimination (inches)", m_disc_margin), ("AUC (rank only)", m_auc),
         ("perception sigma", m_sigma)]


def evaluate(per, side, min_dec, label):
    rng = random.Random(11)
    pool = {k: sorted(v, key=lambda x: x["date"]) for k, v in per.items()
            if k[0] == side and len(v) >= min_dec}
    print(f"\n{label}  ({len(pool)} players with >= {min_dec} decisions)")
    print(f"  {'metric':<26}{'n':>4}{'chrono r':>10}{'(p)':>9}{'rand r':>9}"
          f"{'SB rel':>8}{'vs aggr':>9}{'vs vol':>8}")
    for name, fn in CANDS:
        xs, ys, rx, ry, full, aggr, vol = [], [], [], [], [], [], []
        for k, e in pool.items():
            h = len(e) // 2
            a, b = fn(e[:h]), fn(e[h:])
            sh = rng.sample(e, len(e))
            ra, rb = fn(sh[:h]), fn(sh[h:])
            f = fn(e)
            if None in (a, b, f):
                continue
            xs.append(a); ys.append(b); full.append(f)
            if ra is not None and rb is not None:
                rx.append(ra); ry.append(rb)
            aggr.append(100.0 * sum(1 for x in e if x["oc"]) / len(e))
            vol.append(len(e))
        if len(xs) < 8:
            print(f"  {name:<26}{len(xs):>4}   too few rankable")
            continue
        cr, cp = pearsonr(xs, ys)
        rr = pearsonr(rx, ry)[0] if len(rx) >= 8 else float("nan")
        sb = 2 * cr / (1 + cr) if cr > -1 else float("nan")
        ar = pearsonr(full, aggr)[0]
        vr = pearsonr(full, vol)[0]
        print(f"  {name:<26}{len(xs):>4}{cr:>10.3f}{cp:>9.3f}{rr:>9.3f}"
              f"{sb:>8.2f}{ar:>9.3f}{vr:>8.3f}")


def value_section(per):
    print("\n\n=== VALUE metric: is it stabilizable at all? ===")
    print("  (value per consequential decision, raw vs winsorized at a percentile)")
    for side, label, min_dec in (("fld", "CATCHERS", 240), ("bat", "HITTERS", 80)):
        pool = {k: sorted(v, key=lambda x: x["date"]) for k, v in per.items()
                if k[0] == side and len(v) >= min_dec}
        allg = sorted(x["g"] for e in pool.values() for x in e)
        print(f"\n  {label} ({len(pool)} players)  gain p50={statistics.median(allg):.3f} "
              f"p95={allg[int(.95*len(allg))]:.3f} p99={allg[int(.99*len(allg))]:.3f} "
              f"max={allg[-1]:.2f}")
        for cap_lbl, cap in (("raw", None), ("cap p95", allg[int(.95 * len(allg))]),
                             ("cap p90", allg[int(.90 * len(allg))]),
                             ("cap p75", allg[int(.75 * len(allg))])):
            def val(e):
                tot = 0.0
                for x in e:
                    g = min(x["g"], cap) if cap else x["g"]
                    c = x["C"]
                    if x["oc"]:
                        tot += x["p"] * g - (1 - x["p"]) * c
                    else:
                        ev = x["p"] * g - (1 - x["p"]) * c
                        tot += -ev if ev > 0 else 0.0
                return tot / len(e)
            xs, ys = [], []
            for e in pool.values():
                h = len(e) // 2
                xs.append(val(e[:h])); ys.append(val(e[h:]))
            if len(xs) >= 8:
                r, p = pearsonr(xs, ys)
                print(f"    {cap_lbl:<9} chrono split-half r={r:+.3f} (p={p:.3f})  "
                      f"SB rel={2*r/(1+r):.2f}")


def dp_leverage_sensitivity():
    print("\n\n=== DP leverage basis: how much does the residual G-vs-slope gap matter? ===")
    t = ve.tables_from_json(os.path.join(REPO, "data", "abs_value_tables_2026.json"))
    d = json.load(open(os.path.join(REPO, "data", "abs_challenges_2026.json")))
    rng = random.Random(5)
    samp = rng.sample([r for r in d["records"] if not r.get("posPitcher")], 6000)
    ratios = []
    for r in samp:
        diff = r["homeScore"] - r["awayScore"]
        G = t["G"][(min(r["inning"], 10), r["half"], max(-12, min(12, diff)))]
        slope, _ = ve.run_slope(r["inning"], r["half"], diff, r["bases"], r["outs"], t)
        if G > 1e-9:
            ratios.append(slope / G)
    ratios.sort()
    print(f"  slope/G over {len(ratios)} real states:")
    print(f"    p10={ratios[len(ratios)//10]:.2f}  p50={ratios[len(ratios)//2]:.2f}  "
          f"p90={ratios[9*len(ratios)//10]:.2f}  mean={statistics.mean(ratios):.2f}")
    print("  C(k,T,d) is built on G, gains on slope. A ratio far from 1 means the")
    print("  option-value table is priced on a slightly different leverage scale.")


def main():
    per = load()
    n = sum(len(v) for v in per.values())
    print(f"{n} consequential decisions, {len(per)} player-roles")
    print("\n=== SKILL candidates ===")
    print("chrono r = predictive (the real test) | rand r = upper bound")
    print("vs aggr  = contamination by how often they challenge (want ~0)")
    print("vs vol   = contamination by playing time (want ~0)")
    evaluate(per, "fld", 240, "CATCHERS")
    evaluate(per, "bat", 80, "HITTERS")
    value_section(per)
    dp_leverage_sensitivity()


if __name__ == "__main__":
    main()
