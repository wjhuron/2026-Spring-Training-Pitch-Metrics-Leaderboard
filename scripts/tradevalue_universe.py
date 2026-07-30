"""Assemble the trade value player universe: MLB engine + prospect engine.

Inputs (all already in data/):
  tradevalue_contracts.json  Cot's: 40-man contracts, MLS, control ladders
  tradevalue_arb.json        MLBTR arb projections (bbref ids)
  tradevalue_idmap.csv       Chadwick register mlbam/fangraphs/bbref map
  tradevalue_fg_bat.csv      FG RoS projections, hitters (MLBAMID)
  tradevalue_fg_pit.csv      FG RoS projections, pitchers (MLBAMID)
  tradevalue_fg_board.csv    THE BOARD (FV, Risk, ETA; FG ids, many 'sa'-prefixed)

Matching: Cot's names carry no ids, so they resolve through Chadwick on
normalized (last, first) with birth-year disambiguation from Cot's age
(age as of 7/1, so birthYear is season-age or season-age-1). Unmatched and
ambiguous names are printed in full: fix them here, never silently drop.

Output: data/tradevalue_universe.json
  { mlb: [...], prospects: [...] } — every Cot's player (engine 'mlb') and
  every Board player (engine 'prospect'), with attached projections, arb
  numbers, deferral PVs, and overlap flags.
"""

import csv
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

BASE = Path("/Users/wallyhuron/Huronalytics")
DATA = BASE / "data"
OUT_PATH = DATA / "tradevalue_universe.json"
SEASON = 2026

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def sane_age(age):
    """Cot's Age cells occasionally hold junk (e.g. 103); gate before using."""
    return age if isinstance(age, int) and 15 <= age <= 55 else None

# Cot's typos, nicknames, and spelling variants the register cannot resolve.
# Keyed by (team, exact Cot's name); if a player moves teams the loud
# unmatched report resurfaces and the entry gets updated.
MANUAL_IDS = {
    ("MIA", "Acosta, Max"): "691185",        # register: Maximo Acosta
    ("PIT", "Chander, Bubba"): "696149",     # Cot's typo: Bubba Chandler
    ("PIT", "Curtis, Kristian"): "694753",   # register: Khristian Curtis
    ("LAD", "Hernández, Kiké"): "571771",    # register: Enrique Hernandez
    ("TEX", "Baumier, Carter"): "691945",    # Cot's typo: Carter Baumler
    ("SD", "King, Michael"): "650633",       # register: Mike King
}

# FG org abbrevs (Board) -> statsapi abbrevs (Cot's)
FG_ORG_ALIAS = {"ARI": "AZ", "CHW": "CWS", "KCR": "KC", "SDP": "SD",
                "SFG": "SF", "TBR": "TB", "WSN": "WSH"}


def strip_accents(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def norm_last(s):
    s = strip_accents(s).lower().replace(".", "").replace("'", "").replace("-", " ")
    toks = [t for t in s.split() if t not in SUFFIXES]
    return " ".join(toks)


def norm_first(s):
    """'J.C.' -> 'jc'; 'José O.' -> 'jose'; 'Jung Hoo' -> 'jung'."""
    s = strip_accents(s).lower().replace("'", "").replace("-", " ")
    # collapse dotted initials before tokenizing
    s = re.sub(r"\b([a-z])\.\s*([a-z])\.?", r"\1\2", s)
    s = s.replace(".", " ")
    toks = [t for t in s.split() if t not in SUFFIXES]
    # drop a trailing single-letter middle initial
    if len(toks) > 1 and len(toks[-1]) == 1:
        toks = toks[:-1]
    return toks[0] if toks else ""


def split_cots_name(name):
    parts = name.split(",", 1)
    last = parts[0].strip()
    first = parts[1].strip() if len(parts) > 1 else ""
    return norm_last(last), norm_first(first)


def split_fg_name(name):
    parts = name.strip().rsplit(" ", 1)
    if len(parts) == 1:
        return norm_last(parts[0]), ""
    # rsplit puts suffixes in the "last" slot; norm_last strips them
    first, last = parts[0], parts[1]
    if norm_last(last) in ("", None) or last.lower().strip(".") in SUFFIXES:
        first, last = name.strip().rsplit(" ", 2)[0], name.strip().rsplit(" ", 2)[1]
    return norm_last(last), norm_first(first)


def fetch_json(url):
    req = urllib.request.Request(
        url, headers={"User-Agent": "huronalytics-tradevalue/1.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def affiliate_parents():
    """Every affiliated team id -> parent MLB org teamId (MLB teams map to self)."""
    d = fetch_json(
        "https://statsapi.mlb.com/api/v1/teams?sportIds=1,11,12,13,14,16"
    )
    return {
        t["id"]: t.get("parentOrgId", t["id"]) for t in d["teams"]
    }


def statsapi_resolve(cots_name, org_team_id, parents, age=None):
    """Fallback: statsapi people search, verified by parent-org membership.

    Handles register gaps and accent variants; returns an mlbam id string or
    None. Only called for the handful of names Chadwick cannot settle.
    """
    if "," in cots_name:
        last, first = [x.strip() for x in cots_name.split(",", 1)]
        query = f"{first} {last}"
    else:
        query = cots_name
    try:
        d = fetch_json(
            "https://statsapi.mlb.com/api/v1/people/search?names="
            + urllib.parse.quote(query) + "&hydrate=currentTeam"
        )
    except Exception:
        return None
    cands = []
    for p in d.get("people", []):
        if not p.get("active"):
            continue
        team = (p.get("currentTeam") or {}).get("id")
        if org_team_id and parents.get(team) != org_team_id:
            continue
        if age is not None and p.get("birthDate"):
            by = int(p["birthDate"][:4])
            if by not in (SEASON - age, SEASON - age - 1):
                continue
        cands.append(str(p["id"]))
    return cands[0] if len(set(cands)) == 1 else None


def load_idmap():
    by_name, by_bbref, by_fg = {}, {}, {}
    for r in csv.DictReader(open(DATA / "tradevalue_idmap.csv")):
        key = (norm_last(r["last"]), norm_first(r["first"]))
        by_name.setdefault(key, []).append(r)
        if r["bbref"]:
            by_bbref[r["bbref"]] = r
        if r["fangraphs"]:
            by_fg[r["fangraphs"]] = r
    return by_name, by_bbref, by_fg


def match_mlbam(by_name, last, first, age=None):
    """Resolve to a single mlbam id, or (None, reason)."""
    cands = by_name.get((last, first), [])
    if not cands:
        # initial-vs-name fallback: 'cj' matching 'c j' etc. is rare; try
        # first-initial match within same last name
        cands = [
            r for k, rows in by_name.items() if k[0] == last
            for r in rows if first and k[1] and k[1][0] == first[0]
        ] if last else []
        if len(cands) != 1:
            return None, "no-match"
    if len(cands) > 1 and age is not None:
        want = {SEASON - age, SEASON - age - 1}
        narrowed = [r for r in cands if r["birthYear"].isdigit()
                    and int(r["birthYear"]) in want]
        if narrowed:
            cands = narrowed
    if len(cands) > 1:
        # prefer the row that has a fangraphs id (register keeps dupes)
        with_fg = [r for r in cands if r["fangraphs"]]
        if len({r["mlbam"] for r in cands}) == 1:
            cands = cands[:1]
        elif len(with_fg) == 1:
            cands = with_fg
    if len(cands) != 1:
        return None, f"ambiguous({len(cands)})"
    return cands[0]["mlbam"], None


def main():
    cots = json.load(open(DATA / "tradevalue_contracts.json"))
    arb = json.load(open(DATA / "tradevalue_arb.json"))
    by_name, by_bbref, by_fg = load_idmap()

    # projections keyed by mlbam; two-way players appear in both files
    proj = {}
    for kind in ("bat", "pit"):
        for r in csv.DictReader(open(DATA / f"tradevalue_fg_{kind}.csv", encoding="utf-8-sig")):
            mlbam = r["MLBAMID"].strip()
            war = float(r["WAR"]) if r["WAR"] else 0.0
            p = proj.setdefault(mlbam, {
                "name": r["Name"], "fgId": r["PlayerId"],
                "warBat": None, "warPit": None, "gPit": None, "gsPit": None,
            })
            p["warBat" if kind == "bat" else "warPit"] = war
            if kind == "pit":
                p["gPit"] = float(r["G"]) if r.get("G") else None
                p["gsPit"] = float(r["GS"]) if r.get("GS") else None

    # arb projections keyed by mlbam through bbref
    arb_by_mlbam, arb_unmatched = {}, []
    for a in arb["players"]:
        row = by_bbref.get(a["bbrefId"])
        if row:
            arb_by_mlbam[row["mlbam"]] = a
        else:
            arb_unmatched.append(a["name"])

    # deferral PVs keyed by (team, normalized full name)
    defer = {}
    for d in cots["deferralPresentValues"]:
        last, first = split_fg_name(d["name"])
        defer.setdefault((d["team"], last, first), []).append(d)

    parents = affiliate_parents()
    # IL status per mlbam from 40-man rosters (D10/D15/D60 etc)
    il_status = {}
    team_ids_all = sorted({p["teamId"] for p in cots["players"]})
    for tid in team_ids_all:
        try:
            roster = fetch_json(
                f"https://statsapi.mlb.com/api/v1/teams/{tid}/roster"
                "?rosterType=40Man").get("roster", [])
        except Exception:
            continue
        for x in roster:
            code = (x.get("status") or {}).get("code", "")
            if code.startswith("D"):
                il_status[str(x["person"]["id"])] = code
    print(f"IL-listed players: {len(il_status)}")
    mlb, unmatched, ambiguous = [], [], []
    for p in cots["players"]:
        manual = MANUAL_IDS.get((p["team"], p["name"]))
        last, first = split_cots_name(p["name"])
        if manual:
            mlbam, err = manual, None
        else:
            age = sane_age(p["age"])
            mlbam, err = match_mlbam(by_name, last, first, age)
            if err:
                fallback = statsapi_resolve(p["name"], p["teamId"], parents, age)
                if fallback:
                    mlbam, err = fallback, None
        if err == "no-match":
            unmatched.append(f'{p["team"]} {p["name"]}')
        elif err:
            ambiguous.append(f'{p["team"]} {p["name"]} {err}')
        pr = proj.get(mlbam) if mlbam else None
        dv = defer.get((p["team"], last, first))
        mlb.append({
            **{k: p[k] for k in ("name", "team", "teamId", "pos", "age", "mls",
                                 "optsLeft", "optsTotal", "contract", "years",
                                 "deferred")},
            "mlbam": mlbam,
            "fgId": pr["fgId"] if pr else None,
            "warBat": pr["warBat"] if pr else None,
            "warPit": pr["warPit"] if pr else None,
            "gPit": pr.get("gPit") if pr else None,
            "gsPit": pr.get("gsPit") if pr else None,
            "arbProjSalary": arb_by_mlbam[mlbam]["projSalary"] if mlbam in arb_by_mlbam else None,
            "ilStatus": il_status.get(str(mlbam)) if mlbam else None,
            "deferralPVs": dv,
            "engine": "mlb",
        })

    # a recently traded player can appear on both teams' Cot's sheets;
    # keep the row matching statsapi's current parent org, drop the stale one
    from collections import Counter
    counts = Counter(p["mlbam"] for p in mlb if p["mlbam"])
    for dupe_id in [k for k, v in counts.items() if v > 1]:
        rows_d = [p for p in mlb if p["mlbam"] == dupe_id]
        try:
            person = fetch_json(
                f"https://statsapi.mlb.com/api/v1/people/{dupe_id}"
                "?hydrate=currentTeam"
            )["people"][0]
            team = (person.get("currentTeam") or {}).get("id")
            current_org = parents.get(team, team)
        except Exception:
            current_org = None
        keep = [p for p in rows_d if p["teamId"] == current_org]
        if len(keep) == 1:
            stale = [p for p in rows_d if p is not keep[0]]
            for p in stale:
                mlb.remove(p)
            print(f'dupe {rows_d[0]["name"]}: kept {keep[0]["team"]}, '
                  f'dropped {[p["team"] for p in stale]}')
        else:
            print(f'dupe UNRESOLVED, kept both: {[(p["team"], p["name"]) for p in rows_d]}')

    matched_mlbam = {p["mlbam"] for p in mlb if p["mlbam"]}
    team_ids = {p["team"]: p["teamId"] for p in mlb}
    prospects = []
    board_rows = list(csv.DictReader(open(DATA / "tradevalue_fg_board.csv", encoding="utf-8-sig")))
    for r in board_rows:
        fg_id = r["PlayerId"].strip()
        mlbam = None
        if fg_id and not fg_id.startswith("sa"):
            row = by_fg.get(fg_id)
            mlbam = row["mlbam"] if row else None
        if mlbam is None:
            last, first = split_fg_name(r["Name"])
            age = int(float(r["Age"])) if r["Age"] else None
            mlbam, err = match_mlbam(by_name, last, first, age)
            if err:
                org = FG_ORG_ALIAS.get(r["Org"], r["Org"])
                org_id = team_ids.get(org)
                mlbam = statsapi_resolve(r["Name"], org_id, parents, age)
        prospects.append({
            "name": r["Name"],
            "org": r["Org"],
            "pos": r["Pos"],
            "level": r["Current Level"],
            "eta": int(r["ETA"]) if r["ETA"].strip().isdigit() else None,
            "fv": r["FV"].strip(),
            "risk": r["Risk"].strip().capitalize() if r["Risk"].strip() else None,
            "age": float(r["Age"]) if r["Age"] else None,
            "top100": int(r["Top 100"]) if r["Top 100"].strip().isdigit() else None,
            "orgRank": int(r["Org Rk"]) if r["Org Rk"].strip().isdigit() else None,
            "fgId": fg_id or None,
            "mlbam": mlbam,
            "onFortyMan": bool(mlbam and mlbam in matched_mlbam),
            "hasProjection": bool(mlbam and mlbam in proj),
            "engine": "prospect",
        })

    out = {
        "fetched": date.today().isoformat(),
        "season": SEASON,
        "projectionSource": "FanGraphs THE BAT X RoS 2026 export",
        "mlb": mlb,
        "prospects": prospects,
    }
    OUT_PATH.write_text(json.dumps(out, indent=1))

    n_proj = sum(1 for p in mlb if p["warBat"] is not None or p["warPit"] is not None)
    n_arb = sum(1 for p in mlb if p["arbProjSalary"] is not None)
    n_pros_id = sum(1 for p in prospects if p["mlbam"])
    n_40 = sum(1 for p in prospects if p["onFortyMan"])
    print(f"MLB engine: {len(mlb)} players | {n_proj} with projection | {n_arb} with arb proj")
    print(f"Prospects: {len(prospects)} | {n_pros_id} resolved to mlbam | {n_40} on a 40-man")
    print(f"Arb rows without bbref match: {len(arb_unmatched)} {arb_unmatched[:5]}")
    if unmatched:
        print(f"\nUNMATCHED Cot's names ({len(unmatched)}):")
        for n in unmatched:
            print("  ", n)
    if ambiguous:
        print(f"\nAMBIGUOUS Cot's names ({len(ambiguous)}):")
        for n in ambiguous:
            print("  ", n)


if __name__ == "__main__":
    main()
