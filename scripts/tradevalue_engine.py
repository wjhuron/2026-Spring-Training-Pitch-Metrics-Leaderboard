"""Intrinsic surplus value engine (phase 3 of the trade value model).

Reads data/tradevalue_universe.json and produces data/tradevalue_intrinsic.json:
one intrinsic surplus value (in $) per player, MLB engine and prospect engine.

MLB engine, per remaining control year t:
  W_t   = max(0, warBase + agingDelta * agingSeasons) * (1 - riskDecay_role)^(t-1)
  V_t   = rate(W_t) * W_t * (1 + winInflation)^(t-1)
(decay applies to WAR itself so declining players also slide down the $/WAR
tiers; fitted vs realized future bWAR in tradevalue_market.py stage 1)
  C_t   = signed salary | arbLadder_k * market value | league minimum
  S     = sum_t (V_t - C_t) / (1 + discountRate)^(t-1)
Year 1 is prorated by the fraction of the current season remaining.

Cot's year cells for deferred contracts already hold the Labor-Relations
present values spread across years (verified: Ohtani $28.2M/yr = $282.2M/10),
so signed-year salaries need no further deferral adjustment.

Prospect engine: Clemens July 2026 FV table (hitter/pitcher split), with
adjustment multipliers left at 1.0 until the market layer fits them.

CALIBRATION (phase 5, 2026-07-28, tradevalue_market.py):
- riskDecay 0.21 and agingDelta 0.0: decline is multiplicative — projected
  WAR decays 21%/yr. Fit vs realized bWAR 1-4 yrs ahead over 6,832 active
  player-seasons 2016-2022; lambda interior-bracketed, ballast interior at 2,
  delta at its natural boundary 0. Beats the literature anchor
  (lambda 0, delta -0.4, ballast 2) in 6/6 held-out seasons.
- market multipliers (data/tradevalue_market_fit.json) price deviations on
  top of intrinsic surplus, fit on 200+ two-sided trades 2017-2026 with
  engine constants FIXED (trade balance alone rewards value-flattening and
  cannot identify engine constants). LOSO: beats intrinsic-only in 8/10
  held-out seasons. Residuals stay wide (median ~0.9 log units): team
  context dominates single trades; treat marketValue bands as wide.
"""

import csv
import json
import re
import unicodedata
from datetime import date
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT_PATH = DATA / "tradevalue_intrinsic.json"
SEASON = 2026

CONFIG = {
    "leagueMin": 780_000,          # 2026 CBA minimum; held flat beyond 2026
    # $/WAR by the year's projected WAR tier (Clemens 2026 graduated scale)
    "dollarsPerWarTiers": [[1.0, 7_000_000], [2.0, 8_500_000], [None, 13_000_000]],
    "winInflation": 0.03,          # annual growth in the cost of a win
    "discountRate": 0.10,          # nominal; 10% - 3% inflation = FG's 7% net
    "agingDelta": 0.0,             # fitted: no linear term needed (see header)
    "agingStartAge": 27,           # inert while agingDelta is 0
    "arbLadder": {1: 0.15, 2: 0.35, 3: 0.50, 4: 0.75},  # FG 2026, fitted to awards
    # fitted 21%/yr multiplicative decline (pooled roles; role split untested)
    "riskDecay": {"SP": 0.21, "RP": 0.21, "C": 0.21, "POS": 0.21},
    # per-talent-tier decline overrides the pooled value when present:
    # [[upper WAR bound, lambda], ...] fitted in tradevalue_market stage 1b
    "riskDecayByWar": None,
    "horizonYears": 10,
    "defaultAge": 27,              # when Cot's age is junk/missing
    # projections are FG RoS exports: WAR covers only the REMAINING season,
    # so the full-season talent baseline = RoS WAR / season fraction remaining
    # (clamped: annualizing through a tiny remaining sliver is meaningless)
    "projectionIsRoS": True,
    "minAnnualizeFrac": 0.15,
    # Clemens July 2026 FV table: fv -> (hitter $, pitcher $, expWAR h, expWAR p,
    # star odds h, star odds p)
    "fvTable": {
        "70":  [195e6, 195e6, 27.5, 27.0, 0.875, 0.875],
        "65":  [95e6,  95e6,  13.5, 13.5, 0.400, 0.400],
        "60":  [82e6,  70e6,  12.5, 11.0, 0.330, 0.210],
        "55":  [55e6,  45e6,   8.0,  7.0, 0.175, 0.070],
        "50":  [45e6,  33.5e6, 7.0,  5.0, 0.135, 0.070],
        "45+": [18.5e6, 15e6,  3.2,  2.6, 0.060, 0.030],
        "45":  [14.5e6, 9.5e6, 2.5,  1.6, 0.035, 0.015],
        "40+": [8e6,   7e6,    1.2,  1.0, 0.018, 0.010],
        "40":  [5.5e6, 4e6,    0.75, 0.55, 0.008, 0.004],
        "35+": [2e6,   1.5e6,  0.3,  0.25, 0.004, 0.004],
    },
    # prospect adjustment multipliers: 1.0 until fitted (phase 5)
    "prospectMultipliers": {"eta": 1.0, "position": 1.0, "role": 1.0, "roster": 1.0},
    # graduation-cliff blend (gradcliff_fit.py): value = w x FV$ + (1-w) x
    # engine$, w by years since the player's last Board FV. Fit on realized
    # 4-yr WAR of 758 graduates 2017-2021, LOSO 5/5 vs projection-only,
    # confirmed on the untouched 2022 cohort; single weight (tier and
    # position splits tested, not earned). MSE-fit; MAE would say ~0.35-0.40
    # (right-skew: values are expectations, so the mean-targeting loss is
    # the principled one).
    "gradBlendRamp": {0: 0.60, 1: 0.45, 2: 0.30, 3: 0.10},
}

PROSPECT_PITCHER_POS = {"SP", "SIRP", "MIRP", "RP"}

# Map every internal abbrev style (statsapi or FG) to the SITE convention
# (the sheets' PTeam set: ARI/ATH/CWS/KCR/SDP/SFG/TBR/WSH...)
SITE_TEAM_ALIAS = {"AZ": "ARI", "KC": "KCR", "SD": "SDP", "SF": "SFG",
                   "TB": "TBR", "CHW": "CWS", "WSN": "WSH"}


def season_fraction_remaining(today=None):
    """Fraction of the current season's value still to be delivered."""
    today = today or date.today()
    start, end = date(SEASON, 3, 27), date(SEASON, 9, 27)
    if today <= start:
        return 1.0
    if today >= end:
        return 0.0
    return (end - today).days / (end - start).days


def role_of(pos):
    """Cot's pos string -> risk-decay role bucket."""
    toks = re.split(r"[-/ ]", pos.lower())
    if "rhp" in toks or "lhp" in toks:
        if "dh" in toks:  # two-way players are not relievers
            return "POS"
        return "SP" if "s" in toks else "RP"
    if "c" in toks:
        return "C"
    return "POS"


def rate_for(war):
    for cap, rate in CONFIG["dollarsPerWarTiers"]:
        if cap is None or war < cap:
            return rate
    return CONFIG["dollarsPerWarTiers"][-1][1]


OPTION_TYPES = {"m": "mutual", "cl": "club", "v": "vesting",
                "cond": "vesting", "cdl": "vesting", "p": "player",
                "player": "player"}


def parse_option_years(text):
    """Contract-string option markers -> {season: type}.

    Handles '+27 m opt', '+30 cl opt', '+28 v opt', '+26, 27 opts',
    '+28-29 opt', '+28 opt' (bare = club). Cot's grids often put the
    mutual-option year's salary in a plain cell, so this is the only
    reliable source of option TYPE.
    """
    out = {}
    for m in re.finditer(
        r"\+\s*((?:\d{2})(?:\s*[,\-]\s*\d{2})*)\s*"
        r"(m|cl|v|cond|cdl|player|p)?\.?\s*opts?(?![a-z])",
        text, re.I,
    ):
        years_part, type_key = m.group(1), (m.group(2) or "").lower()
        opt_type = OPTION_TYPES.get(type_key, "club")
        for tok in re.split(r"[,\-]", years_part):
            tok = tok.strip()
            if tok.isdigit():
                out[2000 + int(tok)] = opt_type
    return out


def parse_contract_span(text):
    """'11 yr/$288.7M (24-34)' -> (2024, 2034, 288.7e6); None where unknown."""
    m = re.search(r"\((\d{2})-(\d{2})\)", text)
    span = (2000 + int(m.group(1)), 2000 + int(m.group(2))) if m else None
    m2 = re.search(r"\$([\d.]+)\s*M", text)
    total = float(m2.group(1)) * 1e6 if m2 else None
    return span, total


def build_ladder(p):
    """Season-by-season control status out to the horizon.

    Returns list of dicts {season, status, salary, flags}. Cot's codes cover
    the visible 5-year grid; beyond it (and for rookies whose codes run out)
    control is derived from the contract span and MLS service time.
    """
    years = p["years"]
    mls_years = int(p["mls"].split(".")[0]) if p["mls"] and p["mls"][0].isdigit() else 0
    span, _ = parse_contract_span(p["contract"])
    option_years = parse_option_years(p["contract"])
    ladder, last_salary, flags = [], None, []
    ended = False
    for i in range(CONFIG["horizonYears"]):
        season = SEASON + i
        cell = years.get(str(season))
        opt_type = option_years.get(season)
        if opt_type in ("mutual", "vesting"):
            # mutual options are essentially never exercised by both sides;
            # vesting options are treated conservatively the same way, so
            # control ends here (Littell '1y+27 m opt' = expiring deal)
            flags.append(opt_type + "OptionEndsControl")
            break
        if cell:
            t = cell["type"]
            if t == "fa":
                break
            if t == "signed":
                last_salary = cell["salary"]
                if opt_type in ("club", "player"):
                    # grid holds the option salary as a plain number; keep
                    # the salary but treat the year as the option it is
                    ladder.append({"season": season,
                                   "status": "option" if opt_type == "club"
                                   else "playerOption",
                                   "salary": cell["salary"]})
                else:
                    ladder.append({"season": season, "status": "signed",
                                   "salary": cell["salary"]})
            elif t == "arb":
                ladder.append({"season": season, "status": "arb",
                               "arbYear": cell["arbYear"]})
            elif t == "option":
                ladder.append({"season": season,
                               "status": "playerOption"
                               if opt_type == "player" else "option",
                               "salary": last_salary})
                flags.append("optionYearEstimated")
            else:  # prearb
                ladder.append({"season": season, "status": "prearb"})
            continue
        # past the visible grid: signed extension, then service-time derivation
        if span and span[1] >= season and last_salary is not None:
            ladder.append({"season": season, "status": "signed",
                           "salary": last_salary})
            if "contractExtendedPastGrid" not in flags:
                flags.append("contractExtendedPastGrid")
            continue
        service = mls_years + i  # approx: one service year per season
        if service >= 6:
            break
        if service >= 3:
            arb_year = service - 2
            if arb_year > 4:
                break
            ladder.append({"season": season, "status": "arb", "arbYear": arb_year})
        else:
            ladder.append({"season": season, "status": "prearb"})
        if "controlDerivedFromMLS" not in flags:
            flags.append("controlDerivedFromMLS")
        ended = False
    return ladder, flags


def value_mlb(p, frac_y1):
    cfg = CONFIG
    war1 = (p["warBat"] or 0.0) + (p["warPit"] or 0.0)
    if cfg["projectionIsRoS"]:
        war1 = war1 / max(frac_y1, cfg["minAnnualizeFrac"])
    # validated Stuff+ projection adjustment (stuffedge_build.py):
    # season-blocked pooled coefficient x stuff-vs-results residual,
    # scaled to this pitcher's annualized workload
    war1 += p.get("stuffWarAdj", 0.0)
    age = p["age"] if isinstance(p["age"], int) and 15 <= p["age"] <= 55 else None
    if age is None:
        age = cfg["defaultAge"]
    role = role_of(p["pos"])
    lam = cfg["riskDecay"][role]
    if cfg.get("riskDecayByWar"):
        for hi, bucket_lam in cfg["riskDecayByWar"]:
            if war1 < hi:
                lam = bucket_lam
                break
    ladder, flags = build_ladder(p)
    if p["warBat"] is None and p["warPit"] is None:
        flags.append("noProjection")

    years_out, surplus = [], 0.0
    aging_seasons = 0
    for t, step in enumerate(ladder, start=1):
        season_age = age + (t - 1)
        if t > 1 and season_age >= cfg["agingStartAge"]:
            aging_seasons += 1
        war_t = (max(0.0, war1 + cfg["agingDelta"] * aging_seasons)
                 * (1 - lam) ** (t - 1))
        market = rate_for(war_t) * war_t * (1 + cfg["winInflation"]) ** (t - 1)
        value = market
        if step["status"] in ("signed", "playerOption"):
            cost = step["salary"] if step["salary"] is not None else market
        elif step["status"] == "arb":
            cost = max(cfg["leagueMin"],
                       cfg["arbLadder"][step["arbYear"]] * market)
        elif step["status"] == "option":
            cost = step["salary"] if step["salary"] is not None else market
        else:
            cost = cfg["leagueMin"]
        frac = frac_y1 if t == 1 else 1.0
        net = (value - cost) * frac
        # pre-arb and arb salaries are not guaranteed (non-tender/outright) and
        # club options can be declined, so those years never force a loss on
        # the club; only signed years can go negative. A player option is the
        # mirror image: the player stays only when it favors him, so the year
        # can only subtract value.
        if step["status"] in ("option", "arb", "prearb"):
            net = max(0.0, net)
        elif step["status"] == "playerOption":
            net = min(0.0, net)
        disc = net / (1 + cfg["discountRate"]) ** (t - 1)
        surplus += disc
        years_out.append({
            "season": step["season"], "status": step["status"],
            "war": round(war_t, 2), "value": round(value),
            "cost": round(cost), "net": round(disc),
        })
    return surplus, years_out, flags


# pre-graduation blend weight on the FV side, by tier (pregrad_blend_fit.py
# + tier-split check): FV>=50 w=0.50 (tier fit equals the pooled interior
# optimum), 45/45+ w=0.70 (interior, beats pooled 4/5 LOSO), 40+ and below
# absent = no blend (w=1.0 boundary, FV-only beats blending 4/5 — MLE
# projections carry no signal for org-guy tiers).
PREGRAD_W = {"70": 0.5, "65": 0.5, "60": 0.5, "55": 0.5, "50": 0.5,
             "45+": 0.7, "45": 0.7}


def value_predebut(war1, frac_y1, pitcher):
    """MLB-path value for a not-yet-debuted prospect off a real projection:
    7 control years from scratch (3 pre-arb + 4 arb), no salary owed, every
    year non-tenderable (floors at 0). Mirrors value_mlb's year math; the
    corpus twin is tradevalue_snapshot.value_predebut_at."""
    cfg = CONFIG
    lam = cfg["riskDecay"]["SP" if pitcher else "POS"]
    surplus = 0.0
    for t in range(1, 8):
        war_t = max(0.0, war1) * (1 - lam) ** (t - 1)
        market = rate_for(war_t) * war_t * (1 + cfg["winInflation"]) ** (t - 1)
        service_t = t - 1
        if service_t >= 3:
            cost = max(cfg["leagueMin"],
                       cfg["arbLadder"][min(4, service_t - 2)] * market)
        else:
            cost = cfg["leagueMin"]
        net = max(0.0, (market - cost) * (frac_y1 if t == 1 else 1.0))
        surplus += net / (1 + cfg["discountRate"]) ** (t - 1)
    return surplus


RISK_ORD = {"Low": 0, "Medium": 1, "Med": 1, "High": 2, "Extreme": 3}


def prospect_adjuster(prospects, season=None):
    """Within-FV multiplier from the fitted heterogeneity artifact.

    Features are demeaned against the FIT-ERA (2017-2021) cell feature
    means stored in the artifact — NOT the current board's cells. FG's tier
    composition drifted (2026 FV50-hitter mean ETA gap 2.64 yrs vs 1.3-1.5
    in every fit cohort), so live-cell demeaning handed near-ETA prospects
    boosts beyond fitted support while the FV dollar base still reflects
    fit-era composition (the Harry Ford / Culpepper-class inflation).
    ETA means are stored as offsets from the list year and rebased to the
    current season. Every feature with a fitted range is clipped to the
    1st-99th percentile support; cells absent from the fit get 1.0.
    """
    adj_path = DATA / "tradevalue_prospect_adj.json"
    if not adj_path.exists():
        return lambda p: 1.0
    art = json.loads(adj_path.read_text())
    coefs, ranges = art["coefs"], art["featureRanges"]
    cell_feats = art.get("cellFeatureMeans") or {}
    if season is None:
        season = date.today().year

    def clip(f, v):
        if f in ranges:
            lo, hi = ranges[f]
            return max(lo, min(hi, v))
        return v

    def adjust(p):
        fv = str(p.get("fv") or "").strip()
        is_p = p["pos"] in PROSPECT_PITCHER_POS
        mu = cell_feats.get(f"{fv}|{'P' if is_p else 'H'}")
        if mu is None:
            return 1.0
        feats = {}
        try:
            eta = float(p.get("eta"))
        except (TypeError, ValueError):
            eta = None
        feats["etaGap"] = (clip("etaGap", eta - (season + mu["etaOffset"]))
                           if (eta and mu["etaOffset"] is not None) else 0.0)
        try:
            age = float(p.get("age"))
        except (TypeError, ValueError):
            age = None
        feats["ageGap"] = (clip("ageGap", age - mu["age"])
                           if (age and mu["age"]) else 0.0)
        feats["riskOrd"] = clip("riskOrd",
                                RISK_ORD.get(str(p.get("risk") or "").capitalize(), 1)
                                - mu["risk"])
        feats["posSS"] = clip("posSS",
                              (1.0 if p["pos"] == "SS" else 0.0) - mu["posSS"])
        feats["posC"] = clip("posC",
                             (1.0 if p["pos"] == "C" else 0.0) - mu["posC"])
        rank = p.get("top100")
        feats["rankGap"] = (clip("rankGap", (rank - mu["rank"]) / 25.0)
                            if (rank and mu["rank"]) else 0.0)
        feats["hasRank"] = clip("hasRank", (1.0 if rank else 0.0) - mu["hasRank"])
        # relative form: coefficients are fractions of tier value per
        # feature unit; expectation floors at zero (structural)
        adj = sum(coefs[f] * feats.get(f, 0.0) for f in coefs)
        return max(0.0, 1.0 + adj)

    return adjust


def value_prospect(p):
    cfg = CONFIG
    row = cfg["fvTable"].get(p["fv"])
    if row is None:
        return None, None, None
    is_p = p["pos"] in PROSPECT_PITCHER_POS or p["pos"] == "TWP"
    # two-way prospects valued as hitters (the higher, safer path)
    if p["pos"] == "TWP":
        is_p = False
    value = row[1] if is_p else row[0]
    exp_war = row[3] if is_p else row[2]
    star = row[5] if is_p else row[4]
    for mult in cfg["prospectMultipliers"].values():
        value *= mult
    return value, exp_war, star


def market_multiplier(rec, fit, in_deadline):
    """Fitted market multipliers applied at player level (positive values).

    Interpretation caveat: multipliers were fit on players who actually
    traded. The rental/deadline/reliever premia are mechanism-backed; the
    star (0.42) and prospect (0.48) discounts describe what TRADED stars and
    prospects returned (distress sales, willing-to-deal orgs) and understate
    what an untradeable star would fetch. Rank by intrinsic surplus;
    marketValue answers "what have players like this returned in real
    trades", with wide bands.
    """
    if rec["surplus"] <= 0 or fit is None:
        return 1.0
    m = fit["multipliers"]
    is_prospect = rec["engine"] == "prospect"
    rental = rec["engine"] == "mlb" and rec.get("controlYears") == 1
    star = (rec.get("warY1", 0) or 0) >= 4.5 or rec.get("fv") in ("60", "65", "70")
    reliever = rec["engine"] == "mlb" and rec.get("role") == "RP"
    out = 1.0
    if is_prospect:
        out *= m["prospect"]
    if rental:
        out *= m["rental"]
    if star:
        out *= m["star"]
    if rental and in_deadline:
        out *= m["rentalDeadline"]
    if reliever and in_deadline:
        if "rlvRentalDl" in m:   # fitted rental/controlled reliever split
            out *= m["rlvRentalDl"] if rental else m["rlvCtrlDl"]
        elif "relieverDl" in m:
            out *= m["relieverDl"]
    if rec.get("gradBlend") and "gradW" in m:
        # exponent = the blend weight, matching the corpus featurization
        out *= m["gradW"] ** rec["gradBlend"]["w"]
    if "catcher" in m and (rec.get("role") == "C"
                           or (rec["engine"] == "prospect"
                               and rec.get("pos") == "C")):
        out *= m["catcher"]
    if "defFrac" in m and rec.get("defFrac"):
        # continuous feature: multiplier^feature = exp(beta x feature)
        out *= m["defFrac"] ** rec["defFrac"]
    return out


def main():
    u = json.load(open(DATA / "tradevalue_universe.json"))
    fit_path = DATA / "tradevalue_market_fit.json"
    market_fit = json.loads(fit_path.read_text()) if fit_path.exists() else None
    today = date.today()
    in_deadline = "06-15" <= today.isoformat()[5:] <= "07-31"
    frac = season_fraction_remaining()
    print(f"Season fraction remaining: {frac:.3f}")

    board_mlbam = {p["mlbam"] for p in u["prospects"] if p["mlbam"]}
    players = []

    def _nrm(s):
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return s.lower().replace(".", "").strip()

    def _keys(name):
        """exact-normalized plus lastname|first-3 (Matt/Matthew tolerant)."""
        full = _nrm(name)
        parts = full.split(",", 1)
        loose = (parts[0].strip() + "|" + parts[1].strip()[:3]
                 if len(parts) == 2 and parts[1].strip() else None)
        return full, loose

    # site-canonical hitter positions (max games played, MLB API) override
    # Cot's roster labels (Garcia Jr. is a 1B this season, not the 2b Cot's
    # still lists)
    pos_cache = {}
    pos_path = DATA / "hitter_position_cache.json"
    if pos_path.exists():
        pos_cache = {k: v["position"]
                     for k, v in json.loads(pos_path.read_text()).items()
                     if v.get("position")}

    # defensive-value share for the fitted market discount (trailing
    # 2025+2026 bbref run components, mirroring the corpus featurization)
    def_frac_map = {}
    import csv as _csv
    wh = {}
    for r in _csv.DictReader(open(DATA / "tradevalue_warhist.csv")):
        if int(r["season"]) >= 2025:
            rec = wh.setdefault(r["mlbam"], {"rb": 0.0, "rd": 0.0, "pa": 0.0})
            rec["rb"] += float(r["runsBat"]) if r["runsBat"] else 0.0
            rec["rd"] += float(r["runsDef"]) if r["runsDef"] else 0.0
            rec["pa"] += float(r["pa"]) if r["pa"] else 0.0
    for m_id, rec in wh.items():
        denom = abs(rec["rb"]) + abs(rec["rd"])
        if rec["pa"] >= 300 and denom > 1.0:
            def_frac_map[m_id] = rec["rd"] / denom

    stuff_adj, stuff_loose = {}, {}
    adj_path = DATA / "tradevalue_stuffadj.json"
    if adj_path.exists():
        elapsed = max(1.0 - frac, 0.2)
        for rec in json.loads(adj_path.read_text())["players"]:
            annual_pitches = rec["n"] / elapsed
            war_adj = rec["adjRunsPerPitchPitcherPositive"] * annual_pitches / 10.0
            full, loose = _keys(rec["name"])
            stuff_adj[full] = war_adj
            if loose:
                # loose key kept only while unambiguous
                stuff_loose[loose] = (None if loose in stuff_loose else war_adj)
        print(f"stuff adjustments loaded for {len(stuff_adj)} pitchers")

    mlb_by_mlbam = {}
    n_adj = 0
    for p in u["mlb"]:
        full, loose = _keys(p["name"])
        adj_val = stuff_adj.get(full)
        if adj_val is None and loose:
            adj_val = stuff_loose.get(loose) or 0.0
        p["stuffWarAdj"] = adj_val or 0.0
        if p["stuffWarAdj"]:
            n_adj += 1
        surplus, years_out, flags = value_mlb(p, frac)
        if p["stuffWarAdj"]:
            flags.append("stuffAdjusted")
        war_baseline = round((((p["warBat"] or 0) + (p["warPit"] or 0))
                              / max(frac, CONFIG["minAnnualizeFrac"])
                              if CONFIG["projectionIsRoS"]
                              else (p["warBat"] or 0) + (p["warPit"] or 0))
                             + p.get("stuffWarAdj", 0.0), 2)
        # role from projected usage when available (Cot's pos suffix
        # mislabels young starters tagged plain 'rhp' as relievers)
        if p.get("gPit"):
            role = "SP" if (p.get("gsPit") or 0) / p["gPit"] >= 0.3 else "RP"
        else:
            role = role_of(p["pos"])
        display_pos = p["pos"]
        if role == "POS" and p["mlbam"] and str(p["mlbam"]) in pos_cache:
            display_pos = pos_cache[str(p["mlbam"])]
        rec = {
            "defFrac": (def_frac_map.get(str(p["mlbam"]))
                        if role == "POS" or role == "C" else None),
            "name": p["name"], "team": p["team"], "pos": display_pos,
            "age": p["age"], "mls": p["mls"], "mlbam": p["mlbam"],
            "fgId": p["fgId"], "engine": "mlb", "role": role,
            "warY1": war_baseline,
            "ilStatus": p.get("ilStatus"),
            "contract": p["contract"],
            "surplus": surplus,
            "controlYears": len(years_out),
            "years": years_out, "flags": flags,
            "alsoProspect": bool(p["mlbam"] and p["mlbam"] in board_mlbam),
        }
        if p["mlbam"]:
            mlb_by_mlbam[p["mlbam"]] = rec
        if not rec["alsoProspect"]:
            players.append(rec)

    if stuff_adj:
        print(f"stuff adjustments applied to {n_adj} universe pitchers")

    # graduation blend: last Board FV within the ramp window pulls a
    # graduated player's value toward his FV-table value
    hist_path = DATA / "tradevalue_board_hist.json"
    last_fv = {}   # key (fgId or normName) -> (fv, listSeason), newest wins
    if hist_path.exists():
        raw = json.loads(hist_path.read_text())
        for season_key in sorted(raw, reverse=True):
            season = int(season_key[:4])
            if SEASON - season > max(CONFIG["gradBlendRamp"]):
                continue
            for r in raw[season_key]:
                fv = str(r.get("fv") or "").strip()
                if fv not in CONFIG["fvTable"]:
                    continue
                for key in (str(r.get("fgId") or ""),
                            _nrm(r.get("name") or "")):
                    if key and key not in last_fv:
                        last_fv[key] = (fv, season)
    n_blend = 0
    for r in players:
        if r["engine"] != "mlb":
            continue
        parts = r["name"].split(",", 1)
        flipped = (parts[1].strip() + " " + parts[0].strip()
                   if len(parts) == 2 else r["name"])
        hit = (last_fv.get(str(r.get("fgId") or "x"))
               or last_fv.get(_nrm(flipped)))
        if not hit:
            continue
        fv, season = hit
        t = SEASON - season
        w = CONFIG["gradBlendRamp"].get(t)
        if w is None:
            continue
        is_pitcher = r.get("role") in ("SP", "RP")
        fv_dollars = CONFIG["fvTable"][fv][1 if is_pitcher else 0]
        r["surplus"] = w * fv_dollars + (1 - w) * r["surplus"]
        r["gradBlend"] = {"fv": fv, "listSeason": season, "w": w}
        r["flags"].append("gradBlend")
        n_blend += 1
    print(f"graduation blend applied to {n_blend} recent graduates")

    # ensemble RoS WAR by mlbam, for pre-graduation blends of ready
    # prospects outside the Cot's universe (no mlb_rec to lean on)
    ens_proj = {}
    for kind in ("bat", "pit"):
        fg_path = DATA / f"tradevalue_fg_{kind}.csv"
        if not fg_path.exists():
            continue
        for r in csv.DictReader(open(fg_path, encoding="utf-8-sig")):
            m = r["MLBAMID"].strip()
            if m:
                ens_proj[m] = (ens_proj.get(m, 0.0)
                               + (float(r["WAR"]) if r["WAR"] else 0.0))

    adjust = prospect_adjuster(u["prospects"])
    n_hetero = 0
    skipped_fv = 0
    for p in u["prospects"]:
        value, exp_war, star = value_prospect(p)
        if value is None:
            skipped_fv += 1
            continue
        m_hetero = adjust(p)
        if abs(m_hetero - 1.0) > 1e-9:
            value *= m_hetero
            n_hetero += 1
        mlb_rec = mlb_by_mlbam.get(p["mlbam"]) if p["mlbam"] else None
        # pre-graduation blend (pregrad_blend_fit.py): MLB-ready prospects
        # are half grade, half projection — w=0.50 interior, LOSO 5/5 vs
        # both endpoints, 2017-2021 cohorts. The projection side hears
        # current-season performance the FV path is blind to (Harry Ford's
        # bad AAA year): the MLB-engine value when the player is in the
        # Cot's universe, else a pre-debut ladder off the ensemble RoS row
        # (the fit's Steamer sample covered non-40-man prospects too).
        pregrad = None
        try:
            eta_num = float(p.get("eta"))
        except (TypeError, ValueError):
            eta_num = None
        w_tier = PREGRAD_W.get(str(p.get("fv") or "").strip())
        if (w_tier is not None and eta_num is not None
                and eta_num <= SEASON + 1):
            proj_surplus = mlb_rec["surplus"] if mlb_rec is not None else None
            if proj_surplus is None and p["mlbam"] in ens_proj:
                war1 = (ens_proj[p["mlbam"]]
                        / max(frac, CONFIG["minAnnualizeFrac"]))
                proj_surplus = value_predebut(
                    war1, frac, p["pos"] in PROSPECT_PITCHER_POS)
            if proj_surplus is not None:
                pregrad = w_tier
                value = pregrad * value + (1 - pregrad) * proj_surplus
        players.append({
            "name": p["name"], "team": p["org"], "pos": p["pos"],
            "age": p["age"], "mlbam": p["mlbam"], "fgId": p["fgId"],
            "engine": "prospect",
            "fv": p["fv"], "risk": p["risk"], "eta": p["eta"],
            "top100": p["top100"], "level": p["level"],
            "surplus": value,
            "expWar": exp_war, "starOdds": star,
            "onFortyMan": p["onFortyMan"],
            # rookie-eligible 40-man players: FV value is the headline,
            # the MLB-engine number rides along as reference
            "mlbSurplus": mlb_rec["surplus"] if mlb_rec else None,
            "heteroMult": round(m_hetero, 3),
            "pregradW": pregrad,
            "flags": ((["heteroAdj"] if abs(m_hetero - 1.0) > 1e-9 else [])
                      + (["pregradBlend"] if pregrad is not None else [])),
        })
    if skipped_fv:
        print(f"WARNING: {skipped_fv} prospects with unknown FV skipped")
    n_pregrad = sum(1 for r in players
                    if r["engine"] == "prospect" and r.get("pregradW"))
    print(f"within-FV heterogeneity adjustment applied to {n_hetero} prospects")
    print(f"pre-graduation blend applied to {n_pregrad} MLB-ready prospects")

    for r in players:
        r["marketValue"] = r["surplus"] * market_multiplier(r, market_fit, in_deadline)
        # traded-star selection caveat: the star discount is fitted on stars
        # that actually moved (distress sales); untradeable stars would fetch
        # more, so the site marks these market values as lower bounds
        r["starCaveat"] = bool(
            r["surplus"] > 0
            and ((r.get("warY1") or 0) >= 4.5
                 or r.get("fv") in ("60", "65", "70")))
    # depth layer: everyone in affiliated ball outside the core universe
    depth_path = DATA / "tradevalue_depth.json"
    lvl_short = {"Triple-A": "AAA", "Double-A": "AA", "High-A": "A+",
                 "Single-A": "A", "Rookie": "Rk"}
    n_depth = 0
    if depth_path.exists():
        for d in json.loads(depth_path.read_text())["players"]:
            players.append({
                "name": d["name"], "team": d["team"], "pos": d.get("pos", ""),
                "mlbam": str(d["mlbam"]), "fgId": None, "engine": "depth",
                "level": lvl_short.get(d.get("level", ""), d.get("level", "")),
                "warY1": d.get("warProj"),
                "controlYears": d.get("controlLeft"),
                "surplus": d["value"],
                "marketValue": d["value"],
                "flags": [d.get("path", "")],
            })
            n_depth += 1
    players.sort(key=lambda r: r["surplus"], reverse=True)
    out = {
        "generated": date.today().isoformat(),
        "season": SEASON,
        "seasonFractionRemaining": round(frac, 4),
        "config": CONFIG,
        "calibration": "PRE-CALIBRATION: riskDecay unfitted (0.0); constants are "
                       "literature anchors pending the phase-5 sweep",
        "players": players,
    }
    OUT_PATH.write_text(json.dumps(out, indent=1))

    # site export: trimmed records for the trade-builder page
    site = []
    for r in players:
        nm = r["name"]
        if "," in nm:
            last, first = nm.split(",", 1)
            nm = f"{first.strip()} {last.strip()}"
        rec = {
            "n": nm, "t": SITE_TEAM_ALIAS.get(r["team"], r["team"]),
            "p": r["pos"], "e": r["engine"],
            "s": round(r["surplus"] / 1e6, 1),
            "m": round(r["marketValue"] / 1e6, 1),
        }
        if r.get("ilStatus"):
            rec["il"] = r["ilStatus"]
        if r.get("starCaveat"):
            rec["sc"] = 1
        if r["engine"] == "prospect":
            rec["fv"] = r["fv"]
            if r.get("eta"):
                rec["eta"] = r["eta"]
        elif r["engine"] == "depth":
            rec["lvl"] = r.get("level", "")
            if r.get("warY1"):
                rec["w"] = r["warY1"]
                rec["c"] = r["controlYears"]
        else:
            rec["w"] = r["warY1"]
            rec["c"] = r["controlYears"]
            if r.get("contract"):
                rec["k"] = r["contract"][:40]
        site.append(rec)
    site_payload = {
        "generated": out["generated"],
        "note": ("Intrinsic surplus + market-adjusted value, 2026 dollars. "
                 "Projections: 8-system FanGraphs rest-of-season consensus, annualized."),
        "players": site,
    }
    # Gzipped site export (2026-07-29, same pattern as the data embed):
    # trade.js fetches + inflates it via DecompressionStream. mtime=0 so an
    # unchanged payload gzips byte-identically (no spurious commits).
    import gzip
    gz_path = BASE / "data" / "tradevalue_data.json.gz"
    payload_bytes = json.dumps(site_payload, separators=(",", ":")).encode()
    gz_path.write_bytes(gzip.compress(payload_bytes, compresslevel=9, mtime=0))
    legacy_js = BASE / "js" / "tradevalue_data.js"
    if legacy_js.exists():
        legacy_js.unlink()
    print(f"Site export: data/tradevalue_data.json.gz ({len(site)} players, "
          f"{gz_path.stat().st_size/1e3:.0f} KB gz)")

    n_mlb = sum(1 for r in players if r["engine"] == "mlb")
    n_pro = sum(1 for r in players if r["engine"] == "prospect")
    print(f"Wrote {len(players)} players ({n_mlb} mlb + {n_pro} prospect "
          f"+ {n_depth} depth) -> {OUT_PATH}")

    print("\nTop 30 by intrinsic surplus (market value alongside):")
    for r in players[:30]:
        extra = (f'FV {r["fv"]}' if r["engine"] == "prospect"
                 else f'{r["warY1"]} WAR, {r["controlYears"]}y ctrl')
        print(f'  {r["surplus"]/1e6:7.1f}M  mkt {r["marketValue"]/1e6:7.1f}M  '
              f'{r["name"]:24} {r["team"]:4} {extra}')
    print("\nBottom 10 (worst contracts):")
    for r in players[-10:]:
        print(f'  {r["surplus"]/1e6:7.1f}M  {r["name"]:24} {r["team"]:4} '
              f'{r.get("contract", "")}')


if __name__ == "__main__":
    main()
