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

import json
import re
from datetime import date
from pathlib import Path

BASE = Path("/Users/wallyhuron/Huronalytics")
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
    "horizonYears": 10,
    "defaultAge": 27,              # when Cot's age is junk/missing
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
}

PROSPECT_PITCHER_POS = {"SP", "SIRP", "MIRP", "RP"}


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
        return "SP" if "s" in toks else "RP"
    if "c" in toks:
        return "C"
    return "POS"


def rate_for(war):
    for cap, rate in CONFIG["dollarsPerWarTiers"]:
        if cap is None or war < cap:
            return rate
    return CONFIG["dollarsPerWarTiers"][-1][1]


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
    ladder, last_salary, flags = [], None, []
    ended = False
    for i in range(CONFIG["horizonYears"]):
        season = SEASON + i
        cell = years.get(str(season))
        if cell:
            t = cell["type"]
            if t == "fa":
                break
            if t == "signed":
                last_salary = cell["salary"]
                ladder.append({"season": season, "status": "signed",
                               "salary": cell["salary"]})
            elif t == "arb":
                ladder.append({"season": season, "status": "arb",
                               "arbYear": cell["arbYear"]})
            elif t == "option":
                ladder.append({"season": season, "status": "option",
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
    age = p["age"] if isinstance(p["age"], int) and 15 <= p["age"] <= 55 else None
    if age is None:
        age = cfg["defaultAge"]
    role = role_of(p["pos"])
    lam = cfg["riskDecay"][role]
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
        if step["status"] == "signed":
            cost = step["salary"]
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
        # the club; only signed years can go negative
        if step["status"] in ("option", "arb", "prearb"):
            net = max(0.0, net)
        disc = net / (1 + cfg["discountRate"]) ** (t - 1)
        surplus += disc
        years_out.append({
            "season": step["season"], "status": step["status"],
            "war": round(war_t, 2), "value": round(value),
            "cost": round(cost), "net": round(disc),
        })
    return surplus, years_out, flags


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
    reliever = rec["engine"] == "mlb" and role_of(rec["pos"]) == "RP"
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
        out *= m["relieverDl"]
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

    mlb_by_mlbam = {}
    for p in u["mlb"]:
        surplus, years_out, flags = value_mlb(p, frac)
        rec = {
            "name": p["name"], "team": p["team"], "pos": p["pos"],
            "age": p["age"], "mls": p["mls"], "mlbam": p["mlbam"],
            "fgId": p["fgId"], "engine": "mlb",
            "warY1": round((p["warBat"] or 0) + (p["warPit"] or 0), 2),
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

    skipped_fv = 0
    for p in u["prospects"]:
        value, exp_war, star = value_prospect(p)
        if value is None:
            skipped_fv += 1
            continue
        mlb_rec = mlb_by_mlbam.get(p["mlbam"]) if p["mlbam"] else None
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
            "flags": [],
        })
    if skipped_fv:
        print(f"WARNING: {skipped_fv} prospects with unknown FV skipped")

    for r in players:
        r["marketValue"] = r["surplus"] * market_multiplier(r, market_fit, in_deadline)
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
            "n": nm, "t": r["team"], "p": r["pos"], "e": r["engine"],
            "s": round(r["surplus"] / 1e6, 1),
            "m": round(r["marketValue"] / 1e6, 1),
        }
        if r["engine"] == "prospect":
            rec["fv"] = r["fv"]
            if r.get("eta"):
                rec["eta"] = r["eta"]
        else:
            rec["w"] = r["warY1"]
            rec["c"] = r["controlYears"]
            if r.get("contract"):
                rec["k"] = r["contract"][:40]
        site.append(rec)
    site_payload = {
        "generated": out["generated"],
        "note": ("Intrinsic surplus + market-adjusted value, 2026 dollars. "
                 "Projections: ATC opening-day 2026."),
        "players": site,
    }
    (BASE / "js" / "tradevalue_data.js").write_text(
        "window.TRADE_VALUES = " + json.dumps(site_payload) + ";\n")
    print(f"Site export: js/tradevalue_data.js ({len(site)} players)")

    n_mlb = sum(1 for r in players if r["engine"] == "mlb")
    n_pro = len(players) - n_mlb
    print(f"Wrote {len(players)} players ({n_mlb} mlb + {n_pro} prospect) -> {OUT_PATH}")

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
