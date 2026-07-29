"""Value both sides of every historical trade at trade time (phase 4).

For each trade in data/tradevalue_trades.json, each traded player is valued
with information available on the trade date:

  Prospect path: the player appears on that season's Board list (before June 1
  the preseason list, otherwise the in-season updated list) -> Clemens FV
  table, matched by fgId through the idmap, else by (name, birth year).

  MLB path: Marcel-style projection from bbref season WAR (5/4/3 weights over
  the three prior seasons, shrunk by sum_w/(sum_w+4), -0.4 if age >= 28),
  control years from MLB debut (service proxy = season - debut year), actual
  bbref salary for the trade season, and the same value math/config as the
  live engine (tradevalue_engine.CONFIG).

Everything is valued in constant 2026 dollars on both sides — the market
layer compares packages against each other, so the currency just has to be
the same everywhere.

Outputs:
  data/tradevalue_people_cache.json  (mlbam -> name/birthDate/debut, cached)
  data/tradevalue_snapshots.json     per-trade side values
"""

import csv
import json
import time
import unicodedata
import urllib.request
from datetime import date
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from tradevalue_engine import CONFIG, rate_for  # single source for constants

BASE = Path("/Users/wallyhuron/Huronalytics")
DATA = BASE / "data"
OUT_PATH = DATA / "tradevalue_snapshots.json"
PEOPLE_CACHE = DATA / "tradevalue_people_cache.json"
UA = {"User-Agent": "huronalytics-tradevalue/1.0"}

MARCEL_WEIGHTS = (5, 4, 3)   # seasons S-1, S-2, S-3
# proj = raw * sum_w/(sum_w+ballast): ballast 2 gives ~0.86 compression for a
# full 3-season player, in line with Marcel's documented full-timer
# reliability (~0.83). Sweep candidate in phase 5.
MARCEL_BALLAST = 2
MARCEL_AGE_STEP = -0.4       # applied once when age >= 28
SHORT_SEASON_SCALE = {2020: 162 / 60}  # COVID season WAR to full-season scale
FILLER_VALUE = 300_000       # unranked minor leaguer floor (convention)
PROSPECT_PITCHER_POS = {"SP", "RP", "SIRP", "MIRP", "RHP", "LHP"}


def fetch_json(url, tries=3):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def norm_name(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().replace(".", "").replace("-", " ").split())


def load_people(mlbams):
    cache = json.loads(PEOPLE_CACHE.read_text()) if PEOPLE_CACHE.exists() else {}
    missing = [m for m in mlbams if str(m) not in cache]
    for i in range(0, len(missing), 100):
        chunk = missing[i:i + 100]
        d = fetch_json(
            "https://statsapi.mlb.com/api/v1/people?personIds="
            + ",".join(str(m) for m in chunk)
        )
        for p in d.get("people", []):
            cache[str(p["id"])] = {
                "name": p.get("fullName"),
                "birthDate": p.get("birthDate"),
                "debut": p.get("mlbDebutDate"),
                "pitcher": (p.get("primaryPosition") or {}).get("abbreviation")
                           in ("P", "TWP"),
            }
        print(f"people cache: {min(i + 100, len(missing))}/{len(missing)}")
    PEOPLE_CACHE.write_text(json.dumps(cache, indent=1))
    return cache


def load_warhist():
    hist = {}
    for r in csv.DictReader(open(DATA / "tradevalue_warhist.csv")):
        key = int(r["mlbam"])
        season = int(r["season"])
        war = ((float(r["warBat"]) if r["warBat"] else 0.0)
               + (float(r["warPit"]) if r["warPit"] else 0.0))
        hist.setdefault(key, {})[season] = {
            "war": war * SHORT_SEASON_SCALE.get(season, 1.0),
            "salary": int(r["salary"]) if r["salary"] else None,
            "gPit": float(r["gPit"]) if r.get("gPit") else 0.0,
            "gsPit": float(r["gsPit"]) if r.get("gsPit") else 0.0,
        }
    return hist


def load_boards(idmap_fg_to_mlbam):
    """boardKey -> {mlbam or (normName, birthYear): row}"""
    raw = json.loads((DATA / "tradevalue_board_hist.json").read_text())
    boards = {}
    for key, rows in raw.items():
        by_mlbam, by_name, name_only = {}, {}, {}
        for r in rows:
            fg = str(r["fgId"] or "")
            mlbam = idmap_fg_to_mlbam.get(fg)
            if mlbam:
                by_mlbam[mlbam] = r
            bd = (r.get("birthDate") or "")[:4]
            by_name[(norm_name(r["name"]), bd)] = r
            name_only.setdefault(norm_name(r["name"]), []).append(r)
        # name-only fallback for rows missing a birth date: unique names only
        by_name_only = {n: rs[0] for n, rs in name_only.items() if len(rs) == 1}
        boards[key] = {"byMlbam": by_mlbam, "byName": by_name,
                       "byNameOnly": by_name_only}
    return boards


def marcel(hist_for_player, season, age):
    total_w, acc = 0, 0.0
    for w, back in zip(MARCEL_WEIGHTS, (1, 2, 3)):
        rec = hist_for_player.get(season - back)
        if rec is not None:
            acc += w * rec["war"]
            total_w += w
    if total_w == 0:
        return 0.0
    proj = (acc / total_w) * (total_w / (total_w + MARCEL_BALLAST))
    if age is not None and age >= 28:
        proj += MARCEL_AGE_STEP
    return max(0.0, proj)


def season_fraction(trade_date, season):
    start, end = date(season, 3, 27), date(season, 9, 27)
    d = date.fromisoformat(trade_date)
    if d <= start:
        return 1.0
    if d >= end:
        return 0.0
    return (end - d).days / (end - start).days


def value_mlb_at(mlbam, trade_date, season, people, warhist):
    person = people.get(str(mlbam), {})
    hist = warhist.get(mlbam, {})
    birth = person.get("birthDate")
    age = season - int(birth[:4]) if birth else None
    debut = person.get("debut")
    debut_year = int(debut[:4]) if debut else None
    if debut_year is None or debut_year > season:
        return None  # no MLB track yet and not on a board -> unvaluable
    service = max(0, season - debut_year)
    # a mid-year debut leaves partial service in the debut season, so free
    # agency comes after debut_year + 6 (Soto: 5/2018 debut -> FA after 2024)
    control_left = max(1, min(7, debut_year + 6 - season + 1))
    war1 = marcel(hist, season, age)
    cur_salary = (hist.get(season) or {}).get("salary")
    frac = season_fraction(trade_date, season)

    cfg = CONFIG
    surplus = 0.0
    for t in range(1, control_left + 1):
        season_age = (age or cfg["defaultAge"]) + (t - 1)
        aging = sum(1 for k in range(2, t + 1)
                    if (age or cfg["defaultAge"]) + (k - 1) >= cfg["agingStartAge"])
        lam = cfg["riskDecay"]["POS" if not person.get("pitcher") else "SP"]
        war_t = (max(0.0, war1 + cfg["agingDelta"] * aging)
                 * (1 - lam) ** (t - 1))
        market = rate_for(war_t) * war_t * (1 + cfg["winInflation"]) ** (t - 1)
        service_t = service + (t - 1)
        if t == 1 and cur_salary:
            cost = cur_salary
        elif service_t >= 3:
            arb_year = min(4, service_t - 2)
            cost = max(cfg["leagueMin"], cfg["arbLadder"][arb_year] * market)
        else:
            cost = cfg["leagueMin"]
        year_frac = frac if t == 1 else 1.0
        net = (market - cost) * year_frac
        # the current season's salary is owed (negative allowed: salary-dump
        # component); future years have no contract data here and behave as
        # non-tenderable, so they floor at zero
        if not (t == 1 and cur_salary):
            net = max(0.0, net)
        surplus += net / (1 + cfg["discountRate"]) ** (t - 1)
    return {"kind": "mlb", "warProj": round(war1, 2), "controlLeft": control_left,
            "value": surplus}


def value_prospect_at(board_row):
    fv = str(board_row["fv"]).strip()
    row = CONFIG["fvTable"].get(fv)
    if row is None:
        return None
    is_p = (board_row.get("pos") or "").upper() in PROSPECT_PITCHER_POS
    return {"kind": "prospect", "fv": fv,
            "value": row[1] if is_p else row[0]}


def load_context():
    """Everything value_traded_player needs, loaded once."""
    trades = json.loads((DATA / "tradevalue_trades.json").read_text())
    idmap_fg = {}
    for r in csv.DictReader(open(DATA / "tradevalue_idmap.csv")):
        if r["fangraphs"]:
            idmap_fg[r["fangraphs"]] = int(r["mlbam"])
    warhist = load_warhist()
    boards = load_boards(idmap_fg)
    all_ids = sorted({p["mlbam"] for t in trades for p in t["players"]})
    people = load_people(all_ids)
    return {"trades": trades, "warhist": warhist, "boards": boards,
            "people": people}


def board_chain(trade_date, season):
    variant = "prospect" if trade_date[5:7] < "06" else "updated"
    return [f"{season}{variant}",
            f"{season}{'updated' if variant == 'prospect' else 'prospect'}",
            f"{season - 1}updated"]


def value_traded_player(mlbam, name, trade_date, season, ctx):
    people, warhist, boards = ctx["people"], ctx["warhist"], ctx["boards"]
    person = people.get(str(mlbam), {})
    debut = person.get("debut")
    debut_year = int(debut[:4]) if debut else None
    board_eligible = debut_year is None or season - debut_year <= 1
    if board_eligible:
        for bkey in board_chain(trade_date, season):
            board = boards.get(bkey)
            if not board:
                continue
            row = board["byMlbam"].get(mlbam)
            if row is None:
                nm = norm_name(person.get("name") or name)
                row = (board["byName"].get(
                           (nm, (person.get("birthDate") or "")[:4]))
                       or board["byNameOnly"].get(nm))
            if row is not None:
                val = value_prospect_at(row)
                if val is not None:
                    return val
    val = value_mlb_at(mlbam, trade_date, season, people, warhist)
    if val is None:
        val = {"kind": "filler", "value": FILLER_VALUE}
    return val


def main():
    ctx = load_context()
    trades = ctx["trades"]

    out, n_players, n_valued = [], 0, 0
    for tr in trades:
        season = tr["season"]
        sides = {}
        for p in tr["players"]:
            n_players += 1
            val = value_traded_player(p["mlbam"], p["name"], tr["date"],
                                      season, ctx)
            n_valued += 1
            sides.setdefault(p["toTeamId"], []).append({
                "mlbam": p["mlbam"], "name": p["name"], **val,
            })
        out.append({
            "date": tr["date"], "season": season, "deadline": tr["deadline"],
            "description": tr["description"], "flags": tr["flags"],
            "sides": [
                {"teamId": tid,
                 "value": sum(x.get("value", 0.0) for x in players),
                 "players": players}
                for tid, players in sides.items()
            ],
        })

    OUT_PATH.write_text(json.dumps(out, indent=1))
    n_two_sided = sum(1 for t in out
                      if len(t["sides"]) == 2
                      and all(any(x["kind"] != "unvalued" for x in s["players"])
                              for s in t["sides"]))
    print(f"\nWrote {len(out)} trade snapshots -> {OUT_PATH}")
    print(f"players valued: {n_valued}/{n_players} "
          f"({100 * n_valued / n_players:.1f}%)")
    print(f"trades with both sides holding valued players: {n_two_sided}")


if __name__ == "__main__":
    main()
