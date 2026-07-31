"""Within-FV prospect heterogeneity fit (report-only).

Every prospect with the same FV currently gets the same value. This asks
whether the Board's OTHER fields carry realized-outcome signal WITHIN an FV
tier: ETA proximity, age (for level), the risk label, position (SS/C), and
overall-rank position inside the tier.

Sample: 2017-2021 preseason Board lists (5 cohorts). Target: discounted
realized bWAR over the 5 seasons from the list year (7% net, matching the
FV table's discounting; never-reached-MLB = 0). Design: demean target and
features within (FV tier x hitter/pitcher x cohort) cells - the FV table
already owns the cell means; only within-cell structure is up for grabs.

Validation: leave-one-cohort-out. PRE-REGISTERED: a feature ships iff its
coefficient sign is stable in >=4/5 LOSO folds AND the surviving feature
set beats the cells-only baseline (MSE) in >=4/5 held-out cohorts.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import tradevalue_snapshot as snap

COHORTS = range(2017, 2022)
HORIZON = 5
NET_DISC = 0.07
RISK_ORD = {"Low": 0, "Medium": 1, "Med": 1, "High": 2, "Extreme": 3}
FEATS = ["etaGap", "ageGap", "riskOrd", "posSS", "posC", "rankGap", "hasRank"]


def build_sample():
    ctx = snap.load_context()
    warhist, boards = ctx["warhist"], ctx["boards"]
    import csv
    by_fg, by_name = {}, {}
    for r in csv.DictReader(open(snap.DATA / "tradevalue_idmap.csv")):
        if r["fangraphs"]:
            by_fg[r["fangraphs"]] = int(r["mlbam"])
        key = (snap.norm_name(r["first"] + " " + r["last"]), r["birthYear"])
        by_name.setdefault(key, []).append(int(r["mlbam"]))

    import json
    raw = json.loads((snap.DATA / "tradevalue_board_hist.json").read_text())
    rows = []
    n_unmatched = 0
    for year in COHORTS:
        for r in raw.get(f"{year}prospect", []):
            fv = str(r.get("fv") or "").strip()
            if fv not in snap.CONFIG["fvTable"] if hasattr(snap, "CONFIG") else True:
                pass
            if fv not in ("70", "65", "60", "55", "50", "45+", "45",
                          "40+", "40", "35+"):
                continue
            mlbam = by_fg.get(str(r.get("fgId") or ""))
            if mlbam is None:
                key = (snap.norm_name(r["name"] or ""),
                       (r.get("birthDate") or "")[:4])
                cands = by_name.get(key, [])
                mlbam = cands[0] if len(cands) == 1 else None
            realized = 0.0
            if mlbam is not None and mlbam in warhist:
                hist = warhist[mlbam]
                realized = sum(hist.get(year + t, {}).get("war", 0.0)
                               / (1 + NET_DISC) ** t
                               for t in range(0, HORIZON))
            elif mlbam is None:
                n_unmatched += 1  # never-pro or match failure -> realized 0
            pos = (r.get("pos") or "").upper()
            pitcher = pos in ("SP", "RP", "SIRP", "MIRP", "RHP", "LHP", "TWP")
            def num(v):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
            eta = num(r.get("eta"))
            if eta is not None and not (year <= eta <= year + 8):
                eta = None  # garbage ETA cells exist in old lists
            age = num(r.get("age"))
            if age is not None and not (15 <= age <= 35):
                age = None
            rows.append({
                "cohort": year, "fv": fv, "pitcher": pitcher,
                "eta": eta, "age": age,
                "risk": RISK_ORD.get(str(r.get("risk") or "").strip().capitalize(),
                                     1),
                "posSS": 1.0 * (pos == "SS"), "posC": 1.0 * (pos == "C"),
                "rank": num(r.get("rank")), "realized": realized,
            })
    print(f"sample: {len(rows)} prospect-listings across {len(list(COHORTS))} "
          f"cohorts ({n_unmatched} with no id match -> realized 0)")
    return rows


def featurize(rows):
    """Cell-demeaned features and target; cell = (fv, pitcher, cohort)."""
    from collections import defaultdict
    cells = defaultdict(list)
    for r in rows:
        cells[(r["fv"], r["pitcher"], r["cohort"])].append(r)
    out = []
    for cell, members in cells.items():
        if len(members) < 8:
            continue
        cell_mean_real = np.mean([m["realized"] for m in members])
        if cell_mean_real < 0.5:
            continue  # relative target undefined on near-zero cells
        etas = [m["eta"] for m in members if m["eta"]]
        ages = [m["age"] for m in members if m["age"]]
        ranks = [m["rank"] for m in members if m["rank"]]
        risks = [m["risk"] for m in members]
        mean_eta = np.mean(etas) if etas else None
        mean_age = np.mean(ages) if ages else None
        mean_rank = np.mean(ranks) if ranks else None
        mean_risk = np.mean(risks)
        mean_real = np.mean([m["realized"] for m in members])
        for m in members:
            out.append({
                "cohort": m["cohort"], "cell": cell,
                "etaGap": (m["eta"] - mean_eta) if (m["eta"] and mean_eta) else 0.0,
                "ageGap": (m["age"] - mean_age) if (m["age"] and mean_age) else 0.0,
                "riskOrd": m["risk"] - mean_risk,
                "posSS": m["posSS"] - np.mean([x["posSS"] for x in members]),
                "posC": m["posC"] - np.mean([x["posC"] for x in members]),
                "rankGap": ((m["rank"] - mean_rank) / 25.0
                            if (m["rank"] and mean_rank) else 0.0),
                "hasRank": 1.0 * bool(m["rank"])
                           - np.mean([1.0 * bool(x["rank"]) for x in members]),
                # RELATIVE target: fraction of the cell mean (the shipped
                # multiplier form; additive-WAR fits distort small cells)
                "y": (m["realized"] - mean_real) / cell_mean_real,
            })
    print(f"featurized: {len(out)} rows in cells of >=8")
    return out


def main():
    rows = build_sample()
    data = featurize(rows)
    cohorts = sorted({d["cohort"] for d in data})

    def mats(sub, feats):
        X = np.array([[d[f] for f in feats] for d in sub])
        y = np.array([d["y"] for d in sub])
        return X, y

    # per-fold coefficient signs
    print("\ncoefficient sign stability (LOSO folds):")
    signs = {f: [] for f in FEATS}
    for c in cohorts:
        tr = [d for d in data if d["cohort"] != c]
        X, y = mats(tr, FEATS)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        for f, b in zip(FEATS, beta):
            signs[f].append(np.sign(b))
    stable = []
    for f in FEATS:
        s = signs[f]
        dom = max(s.count(1), s.count(-1))
        sgn = "+" if s.count(1) >= s.count(-1) else "-"
        ok = dom >= 4
        print(f"  {f:8} signs {[int(x) for x in s]}  -> "
              f"{'STABLE ' + sgn if ok else 'unstable'}")
        if ok:
            stable.append(f)
    print(f"sign-stable features: {stable}")

    # does the stable set beat cells-only, held-out?
    wins = 0
    for c in cohorts:
        tr = [d for d in data if d["cohort"] != c]
        he = [d for d in data if d["cohort"] == c]
        base_mse = float(np.mean([d["y"] ** 2 for d in he]))
        if stable:
            X, y = mats(tr, stable)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            Xh, yh = mats(he, stable)
            fit_mse = float(np.mean((yh - Xh @ beta) ** 2))
        else:
            fit_mse = base_mse
        w = fit_mse < base_mse
        wins += w
        print(f"  held-out {c}: cells-only {base_mse:.3f}  "
              f"+features {fit_mse:.3f}{'  WIN' if w else ''}")
    print(f"feature set beats cells-only in {wins}/{len(cohorts)} cohorts")

    # pooled coefficients for the record (WAR units per feature unit)
    if stable:
        X, y = mats(data, stable)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        print("\npooled coefficients (discounted WAR per unit):")
        for f, b in zip(stable, beta):
            print(f"  {f:8} {b:+.3f}")
    verdict = bool(stable) and wins >= 4
    print("\nVERDICT:", "fit multipliers from the stable set"
          if verdict else "NULL - FV grades already absorb these fields")

    if verdict:
        # emit the live artifact: pooled coefficients + per-cell realized
        # means (the fit's own denominator for relative adjustments) +
        # fitted feature ranges (extrapolation guard bounds)
        import json as _json
        from collections import defaultdict as _dd
        cell_real = _dd(list)
        for r in rows:
            cell_real[(r["fv"], r["pitcher"])].append(r["realized"])
        ranges = {f: [float(np.percentile([d[f] for d in data], 1)),
                      float(np.percentile([d[f] for d in data], 99))]
                  for f in stable}
        # FIT-ERA cell feature means, pooled across cohorts: the demeaning
        # basis the engine must use. Demeaning against the LIVE board's cell
        # means broke when FG's tier composition drifted (2026 FV50-hitter
        # mean ETA gap 2.64 yrs vs 1.3-1.5 in every fit cohort): near-ETA
        # prospects collected boosts beyond fitted support while the dollar
        # base still reflects fit-era composition. ETA is stored as an
        # offset from the list year (absolute ETAs don't transport).
        cell_feats = {}
        by_cell = _dd(list)
        for r in rows:
            by_cell[(r["fv"], r["pitcher"])].append(r)
        for (fv, pitcher), members in by_cell.items():
            if len(members) < 8:
                continue
            etas = [m["eta"] - m["cohort"] for m in members if m["eta"]]
            ages = [m["age"] for m in members if m["age"]]
            ranks = [m["rank"] for m in members if m["rank"]]
            cell_feats[f"{fv}|{'P' if pitcher else 'H'}"] = {
                "etaOffset": float(np.mean(etas)) if etas else None,
                "age": float(np.mean(ages)) if ages else None,
                "risk": float(np.mean([m["risk"] for m in members])),
                "rank": float(np.mean(ranks)) if ranks else None,
                "hasRank": float(np.mean([1.0 * bool(m["rank"])
                                          for m in members])),
                "posSS": float(np.mean([m["posSS"] for m in members])),
                "posC": float(np.mean([m["posC"] for m in members])),
                "n": len(members),
            }
        artifact = {
            "coefs": {f: float(b) for f, b in zip(stable, beta)},
            "form": "relative",   # m = max(0, 1 + sum(coef * feature))
            "featureRanges": ranges,
            "cellFeatureMeans": cell_feats,
            "fitNote": "relative-space refit: y = (realized - cellMean)/"
                       "cellMean, cells mean>=0.5; LOSO-validated 2017-2021",
        }
        out = Path(__file__).parent.parent / "data" / "tradevalue_prospect_adj.json"
        out.write_text(_json.dumps(artifact, indent=1))
        print(f"artifact -> {out}")


if __name__ == "__main__":
    main()
