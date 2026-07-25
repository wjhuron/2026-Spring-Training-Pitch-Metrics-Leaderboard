"""Round-2 research on the ABS grading model.

Three questions the first bake-off left open:

  1. Does the per-player perception scale (readSigma) carry real information,
     or is it circular? Production fits a player's sigma on the same challenges
     it then grades. Tested three ways: no scale, scale fit IN-sample (what
     production does), and scale fit on the player's OTHER half (honest).

  2. Is SKILL_PREF = 0.434 the right reference bar? Swept against
     chronological split-half prediction.

  3. Are CONS_G_MIN / CONS_M_MAX still right after the value engine changed?
     They were grid-searched under the old (inconsistent-branch) pricing.

Usage: python3 scripts/abs_research_round2.py
"""

import json
import math
import os
import statistics
from collections import defaultdict

from scipy.stats import pearsonr

import abs_value_engine as ve
from abs_option_model import count_class, edge_region, phi

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREF = 0.434


def interp(g, x):
    if x <= g[0][0]:
        return g[0][1]
    if x >= g[-1][0]:
        return g[-1][1]
    st = g[1][0] - g[0][0]
    i = int((x - g[0][0]) / st)
    a, b = g[i], g[min(i + 1, len(g) - 1)]
    return a[1] + (b[1] - a[1]) * (x - a[0]) / max(b[0] - a[0], 1e-9)


def build():
    """Every candidate decision with generous bounds, so thresholds can be
    swept afterwards without recomputing the expensive WP pricing."""
    d = json.load(open(os.path.join(REPO, "data", "abs_challenges_2026.json")))
    o = json.load(open(os.path.join(REPO, "data", "abs_option_model_2026.json")))
    t = ve.tables_from_json(os.path.join(REPO, "data", "abs_value_tables_2026.json"))
    thr = o["meta"]["rulingThrIn"]
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
        if rem <= 0 or abs(m) > 4.5:
            continue
        v = ve.value_of_flip(r["balls"], r["strikes"], r["bases"], r["outs"],
                             r["inning"], r["half"], r["homeScore"] - r["awayScore"], t)
        reg = edge_region(r["pXmid"], r["pZmid"], r["szTop"], r["szBot"])
        cls = count_class(r["balls"], r["strikes"])
        ch = r["challenge"]
        oc = (ch is not None and ch.get("side") == wr
              and ((side == "bat" and ch["role"] == "batter")
                   or (side == "fld" and ch["role"] == "fielder")))
        per[(side, oid)].append({"date": r["date"], "m": m, "g": v["leveragedRuns"],
                                 "oc": oc, "reg": reg, "cls": cls, "side": side})
    return per, o


def conf(o, e, scale=1.0):
    key_sel = f"{e['side']}|{e['reg']}|{e['cls']}"
    key_look = f"{e['side']}|{e['reg']}"
    m = e["m"] * scale
    return interp(o["pSel"][key_sel], m) if e["oc"] else interp(o["pLook"][key_look], m)


def balanced(evs, o, pref=PREF, scale=1.0):
    hi, lo = [], []
    for e in evs:
        p = conf(o, e, scale)
        s = (1.0 if ((p >= pref) == e["oc"]) else -1.0) * abs(p - pref)
        (hi if p >= pref else lo).append(s)
    if not hi or not lo:
        return None
    return 0.5 * statistics.mean(hi) + 0.5 * statistics.mean(lo)


def fit_sigma(evs, league_sigma, n0=40):
    n_ch = sum(1 for e in evs if e["oc"])
    if n_ch < 1:
        return league_sigma
    bins = defaultdict(lambda: [0, 0])
    for e in evs:
        b = round(max(-6.0, min(6.0, e["m"])) / 0.25) * 0.25
        bins[b][1] += 1
        bins[b][0] += e["oc"]
    best = None
    for sig in [0.4 + 0.2 * i for i in range(24)]:
        lo_, hi_ = -6.0, 10.0
        for _ in range(30):
            mid = (lo_ + hi_) / 2
            pred = sum(n * phi((mm - mid) / sig) for mm, (_c, n) in bins.items())
            if pred > n_ch:
                lo_ = mid
            else:
                hi_ = mid
        xs = (lo_ + hi_) / 2
        ll = 0.0
        for mm, (c, n) in bins.items():
            pp = min(max(phi((mm - xs) / sig), 1e-9), 1 - 1e-9)
            ll += c * math.log(pp) + (n - c) * math.log(1 - pp)
        if best is None or ll > best[0]:
            best = (ll, sig)
    return (n_ch * best[1] + n0 * league_sigma) / (n_ch + n0)


def pool(per, side, g_min, m_max, min_dec):
    out = {}
    for k, evs in per.items():
        if k[0] != side:
            continue
        f = sorted([e for e in evs if e["g"] >= g_min and abs(e["m"]) <= m_max],
                   key=lambda x: x["date"])
        if len(f) >= min_dec:
            out[k] = f
    return out


def chrono_r(pl, scorer):
    xs, ys = [], []
    for evs in pl.values():
        h = len(evs) // 2
        a, b = scorer(evs[:h]), scorer(evs[h:])
        if a is None or b is None:
            continue
        xs.append(a); ys.append(b)
    if len(xs) < 8:
        return None, len(xs)
    return pearsonr(xs, ys)[0], len(xs)


def main():
    per, o = build()
    lsig = o["perceptionPooled"]["fld"]["sigma"]
    print(f"{sum(len(v) for v in per.values())} candidate decisions "
          f"({len(per)} player-roles); league catcher sigma {lsig}")

    # ---------------- 1. is the per-player sigma scale real or circular?
    print("\n=== 1. Per-player perception scale: real signal or circularity? ===")
    pl = pool(per, "fld", 0.05, 2.5, 240)
    print(f"    catchers with >=240 decisions: {len(pl)}")
    r_none, n = chrono_r(pl, lambda e: balanced(e, o))
    print(f"  no player scale (league curves)      r={r_none:+.3f}  n={n}")

    xs, ys = [], []
    for evs in pl.values():                      # in-sample: what production does
        h = len(evs) // 2
        a = balanced(evs[:h], o, scale=lsig / fit_sigma(evs[:h], lsig))
        b = balanced(evs[h:], o, scale=lsig / fit_sigma(evs[h:], lsig))
        if a is not None and b is not None:
            xs.append(a); ys.append(b)
    print(f"  scale fit IN-sample (production)     r={pearsonr(xs,ys)[0]:+.3f}  n={len(xs)}")

    xs, ys = [], []
    for evs in pl.values():                      # honest: fit on the other half
        h = len(evs) // 2
        s1 = lsig / fit_sigma(evs[h:], lsig)
        s2 = lsig / fit_sigma(evs[:h], lsig)
        a = balanced(evs[:h], o, scale=s1)
        b = balanced(evs[h:], o, scale=s2)
        if a is not None and b is not None:
            xs.append(a); ys.append(b)
    print(f"  scale fit OUT-of-sample (honest)     r={pearsonr(xs,ys)[0]:+.3f}  n={len(xs)}")
    print("  -> if honest ~= none, the scale is adding nothing but circularity")

    # ---------------- 2. is 0.434 the right reference bar?
    print("\n=== 2. SKILL_PREF reference sweep (catchers, balanced metric) ===")
    print("    pref   chrono r    n")
    for pref in (0.25, 0.30, 0.35, 0.40, 0.434, 0.45, 0.50, 0.55, 0.60):
        r, n = chrono_r(pl, lambda e, p=pref: balanced(e, o, pref=p))
        mark = "  <- current" if abs(pref - PREF) < 1e-6 else ""
        print(f"    {pref:.3f}  {r:+.3f}    {n}{mark}")

    # ---------------- 3. consequential thresholds under the new pricing
    print("\n=== 3. Consequential-decision thresholds under the NEW value engine ===")
    print("    (chronological split-half r of the balanced skill metric)")
    print("    g_min \\ m_max     1.5      2.0      2.5      3.0      3.5")
    for g_min in (0.02, 0.035, 0.05, 0.07, 0.10):
        row = f"    {g_min:.3f}          "
        for m_max in (1.5, 2.0, 2.5, 3.0, 3.5):
            p2 = pool(per, "fld", g_min, m_max, 240)
            if len(p2) < 12:
                row += "   n/a  "
                continue
            r, n = chrono_r(p2, lambda e: balanced(e, o))
            row += f" {r:+.3f}({len(p2):>2})" if r is not None else "   n/a  "
        print(row)
    print("    current setting: g_min=0.05, m_max=2.5")


if __name__ == "__main__":
    main()
