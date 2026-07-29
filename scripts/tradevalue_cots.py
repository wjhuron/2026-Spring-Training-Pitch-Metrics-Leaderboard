"""Pull contract + control-ladder data from Cot's Contracts for the trade value model.

Cot's (legacy.baseballprospectus.com/compensation/cots/) links each team's
current-season salary sheet as a public Google Sheet. The first payroll table
on the page holds the current-year sheets; each row pairs a team logo
(mlbstatic team-logos/{teamId}.svg) with the sheet link.

Each sheet carries, per player: position, age (as of 7/1), MLS service time,
minor-league options remaining, contract description, and year-by-year cells
that are either a salary (signed year) or a control-status code:
  A1..A4 = arbitration years (Cot's already resolves Super Two)
  FA     = first free-agent year (control ends)
  opt    = option year (club/player/vesting; type lives in the contract text)
  ''     = pre-arb renewable year (when it appears before any A-code/FA)

Output: data/tradevalue_contracts.json. Loud failures over silent skips:
any sheet that cannot be parsed raises at the end with a summary.
"""

import csv
import io
import json
import re
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

BASE = Path("/Users/wallyhuron/Huronalytics")
OUT_PATH = BASE / "data" / "tradevalue_contracts.json"
COTS_URL = "https://legacy.baseballprospectus.com/compensation/cots/"
TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams?sportId=1"
SEASON = 2026
# NB: legacy.baseballprospectus.com runs Anubis anti-bot, which challenges
# browser-like ("Mozilla") user agents; a plain tool UA passes through.
UA = {"User-Agent": "huronalytics-tradevalue/1.0"}

ARB_CODES = {"A1", "A2", "A3", "A4"}
# codes seen in year cells that are not salaries
STATUS_CODES = ARB_CODES | {"FA", "OPT", "MIN", "VOPT", "COPT", "POPT"}


def fetch(url, tries=3):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower().replace(".", "")).strip("-")


def team_slug_map():
    """Cot's team-page slug -> (teamId, abbreviation)."""
    data = json.loads(fetch(TEAMS_URL))
    slugs = {}
    for t in data["teams"]:
        slugs[slugify(t["name"])] = (t["id"], t["abbreviation"])
    # Cot's still uses the old slug for the Athletics
    if "athletics" in slugs:
        slugs["oakland-athletics"] = slugs["athletics"]
    return slugs


def current_sheets(page, slugs):
    """(teamId, abbrev, sheetId, pubStyle) from the first payroll table.

    Team identity comes from the team-page link in the same cell block as the
    sheet link (logos are a mix of mlbstatic SVGs and blogspot PNGs, so they
    are not reliable). The first table lists each team once with its
    current-season sheet; later tables repeat teams for past years, so we stop
    at the first repeated team.
    """
    tokens = re.findall(
        r'compensation/cots/[a-z-]+/([a-z-]+)/'
        r'|docs\.google\.com/spreadsheets/d/(e/)?([A-Za-z0-9_-]{20,})',
        page,
    )
    seen, current, pending = set(), [], None
    for slug, pub_style, sheet_id in tokens:
        if slug:
            pending = slugs.get(slug)  # non-team links (feed, themes...) -> None
        elif sheet_id and pending is not None:
            team_id, abbrev = pending
            if team_id in seen:
                break
            seen.add(team_id)
            # published sheets (/d/e/...) need a different CSV export URL
            current.append((team_id, abbrev, sheet_id, bool(pub_style)))
            pending = None
    return current


def parse_money(cell):
    """'$42,500,000' -> 42500000; '$42.500' -> 42500000 (millions); '($10.0)' negative."""
    s = cell.strip().replace("$", "").replace(",", "")
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]
    try:
        v = float(s)
    except ValueError:
        return None
    if abs(v) < 1000:  # future years are written in millions
        v *= 1_000_000
    return -v if neg else v


def classify_cell(cell):
    """One year cell -> dict describing that control year, or None if empty/past-FA."""
    s = cell.strip()
    if not s:
        return {"type": "blank"}
    code = s.upper().rstrip(".")
    if code in ARB_CODES:
        return {"type": "arb", "arbYear": int(code[1])}
    if code == "FA":
        return {"type": "fa"}
    if "OPT" in code:
        return {"type": "option", "raw": s}
    money = parse_money(s)
    if money is not None:
        return {"type": "signed", "salary": money}
    return {"type": "unknown", "raw": s}


def parse_sheet(sheet_id, team_id, abbrev, pub=False):
    if pub:
        url = f"https://docs.google.com/spreadsheets/d/e/{sheet_id}/pub?output=csv"
    else:
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    rows = list(csv.reader(io.StringIO(fetch(url))))
    # locate the Player header row and the year-label row beneath it
    header_i = next(
        (i for i, r in enumerate(rows) if r and r[0].strip() == "Player"), None
    )
    if header_i is None:
        raise ValueError(f"{abbrev}: no 'Player' header row")
    header = rows[header_i]
    label_row = rows[header_i + 1]

    def col_of(name):
        for j, c in enumerate(header):
            if c.strip().lower().startswith(name):
                return j
        return None

    c_pos = col_of("pos")
    c_age = col_of("age")
    c_mls = col_of("mls")
    c_opts = col_of("opts")
    c_contract = col_of("length")
    if None in (c_pos, c_mls, c_contract):
        raise ValueError(f"{abbrev}: missing expected columns in {header}")

    # year columns: contiguous run of 4-digit years in the label row after the
    # contract column; stops at the gap before the CBT block
    year_cols = []
    started = False
    for j in range(c_contract + 1, len(label_row)):
        m = re.fullmatch(r"(20\d\d)", label_row[j].strip())
        if m:
            year_cols.append((j, int(m.group(1))))
            started = True
        elif started:
            break
    if not year_cols or year_cols[0][1] != SEASON:
        raise ValueError(f"{abbrev}: bad year labels {label_row}")

    players, skipped = [], 0
    for r in rows[header_i + 2:]:
        if not r or not r[0].strip():
            continue
        name = r[0].strip()
        # section footers end the player block
        if re.match(
            r"(?i)pre-arb|estimated|projected|competitive|amount under|notes on",
            name,
        ):
            break
        pos = r[c_pos].strip() if c_pos < len(r) else ""
        if not pos:  # dead money / retained salary / adjustment rows
            skipped += 1
            continue
        years = {}
        for j, yr in year_cols:
            cell = classify_cell(r[j] if j < len(r) else "")
            if cell["type"] == "fa":
                years[str(yr)] = cell
                break  # nothing after control ends
            # blanks are pre-arb renewable years when real content follows
            years[str(yr)] = cell if cell["type"] != "blank" else {"type": "prearb"}
        # trailing blanks are empty grid past the contract, not control years
        for yr in sorted(years, reverse=True):
            if years[yr]["type"] == "prearb":
                del years[yr]
            else:
                break
        opts_raw = r[c_opts].strip() if c_opts is not None and c_opts < len(r) else ""
        m_opts = re.match(r"(\d)\s*/\s*(\d)", opts_raw)
        m_marker = re.match(r"^(.*?)([*#+^]+)$", name)
        if m_marker:
            name = m_marker.group(1).strip()
        players.append({
            "name": name,
            # '*' on Cot's = contract contains deferred money (PVs in sheet footer)
            "deferred": bool(m_marker and "*" in m_marker.group(2)),
            "team": abbrev,
            "teamId": team_id,
            "pos": pos,
            "age": (int(r[c_age].strip()) if c_age is not None and c_age < len(r)
                    and r[c_age].strip().isdigit() else None),
            "mls": r[c_mls].strip() if c_mls < len(r) else "",
            "optsLeft": int(m_opts.group(1)) if m_opts else None,
            "optsTotal": int(m_opts.group(2)) if m_opts else None,
            "contract": r[c_contract].strip() if c_contract < len(r) else "",
            "years": years,
        })
    # footer lines like "* Kyle Tucker Present Values: Labor Relations:
    # $221,769,006 (8.00% discount rate). Competitive Balance Tax: $228,783,778
    # (3.87%)." carry the official NPVs for every deferred contract
    # Formats vary by team ("{Name} Present Values: Labor Relations: $X (r%)."
    # vs "{Name} Labor Relations present value: $X (r%)."), including typos
    # ("Competitve"), so key off the "Labor Relations" phrase.
    deferrals = []
    for r in rows:
        text = " ".join(c.strip() for c in r if c.strip())
        if "Labor Relations" not in text and "LRD" not in text:
            continue
        m = re.match(r"\*?\s*(.+?)\s*(?:Labor Relations|LRD)", text)
        name = m.group(1).strip() if m else ""
        name = re.sub(r"(?i)[,.]?\s*present values?\b.*$", "", name).strip()
        # KC/BOS style: "Salvador Perez 2022" = which contract the deferral is on
        m_yr = re.match(r"^(.+?)\s+((?:19|20)\d\d)$", name)
        contract_year = None
        if m_yr:
            name, contract_year = m_yr.group(1).strip(), int(m_yr.group(2))
        junk = ("payroll", "note", "figure", "projected", "40-man")
        if not name or any(w in name.lower() for w in junk):
            continue
        lr = re.search(r"(?:Labor Relations|LRD)[^$]*?\$([\d,]+)\s*\(([\d.]+)%", text)
        cbt = re.search(
            r"(?:Balance Tax|CBT)[^$]*?\$([\d,]+)\s*\((?:([\d.]+)%|not discounted)",
            text,
        )
        deferrals.append({
            "name": name,
            "contractYear": contract_year,
            "team": abbrev,
            "laborPV": parse_money("$" + lr.group(1)) if lr else None,
            "laborRate": float(lr.group(2)) if lr else None,
            "cbtPV": parse_money("$" + cbt.group(1)) if cbt else None,
            "cbtRate": (float(cbt.group(2)) if cbt and cbt.group(2) else None),
        })
    return players, skipped, deferrals


def main():
    page = fetch(COTS_URL)
    sheets = current_sheets(page, team_slug_map())
    if len(sheets) != 30:
        print(f"WARNING: expected 30 current sheets, found {len(sheets)}")
    all_players, all_deferrals, failures = [], [], []
    for team_id, abbrev, sheet_id, pub in sheets:
        try:
            players, skipped, deferrals = parse_sheet(sheet_id, team_id, abbrev, pub)
            all_players.extend(players)
            all_deferrals.extend(deferrals)
            print(f"{abbrev}: {len(players)} players, {len(deferrals)} deferral PVs "
                  f"({skipped} dead-money rows skipped)")
        except Exception as e:  # collect, report all at end, then fail loud
            failures.append(f"{abbrev} ({sheet_id}): {e}")
            print(f"FAIL {abbrev}: {e}")
    unknown = [
        (p["team"], p["name"], y, c["raw"])
        for p in all_players for y, c in p["years"].items()
        if c["type"] == "unknown"
    ]
    if unknown:
        print(f"\n{len(unknown)} unknown year cells (first 20):")
        for u in unknown[:20]:
            print("  ", u)
    # every '*'-marked player should have a footer PV; surface the gaps
    pv_last = {(d["team"], d["name"].split()[-1].lower()) for d in all_deferrals}
    missing_pv = [
        f'{p["team"]} {p["name"]}' for p in all_players
        if p["deferred"]
        and (p["team"], p["name"].split(",")[0].split()[-1].lower()) not in pv_last
    ]
    if missing_pv:
        print(f"\n{len(missing_pv)} deferred-marked players with no footer PV:")
        for n in missing_pv:
            print("  ", n)
    out = {
        "fetched": date.today().isoformat(),
        "season": SEASON,
        "source": "Cot's Contracts (Baseball Prospectus)",
        "players": all_players,
        "deferralPresentValues": all_deferrals,
    }
    OUT_PATH.write_text(json.dumps(out, indent=1))
    print(f"\nWrote {len(all_players)} players -> {OUT_PATH}")
    if failures:
        sys.exit("Unparsed teams:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
