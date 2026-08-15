"""Pull ARCHIVED Cot's contract sheets (prior-season vintages).

The Cot's front page stacks payroll tables for several seasons (currently
2026/2025/2024), each linking every team's sheet for that vintage. This
parses all of them (season autodetected from each sheet's year labels) so
the trade corpus can price real contract obligations at trade time for
those seasons instead of the service-time approximation.

Output: data/tradevalue_contracts_hist.json  {season: [player rows]}
"""

import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tradevalue_cots import fetch, team_slug_map, parse_sheet, COTS_URL

BASE = Path("/Users/wallyhuron/Huronalytics")
OUT_PATH = BASE / "data" / "tradevalue_contracts_hist.json"


def all_sheets(page, slugs):
    """Every (abbrev, sheetId, pub) pair on the page, in order."""
    tokens = re.findall(
        r'compensation/cots/[a-z-]+/([a-z-]+)/'
        r'|docs\.google\.com/spreadsheets/d/(e/)?([A-Za-z0-9_-]{20,})',
        page,
    )
    pairs, pending = [], None
    for slug, pub_style, sheet_id in tokens:
        if slug:
            pending = slugs.get(slug)
        elif sheet_id and pending is not None:
            team_id, abbrev = pending
            pairs.append((team_id, abbrev, sheet_id, bool(pub_style)))
            pending = None
    return pairs


def main():
    page = fetch(COTS_URL)
    pairs = all_sheets(page, team_slug_map())
    print(f"{len(pairs)} team-sheet pairs on the page")
    by_season, failures = {}, []
    for team_id, abbrev, sheet_id, pub in pairs:
        try:
            players, skipped, _, season = parse_sheet(
                sheet_id, team_id, abbrev, pub, expected_season=None)
        except Exception as e:
            failures.append(f"{abbrev} {sheet_id}: {e}")
            continue
        by_season.setdefault(str(season), []).extend(players)
    for season, rows in sorted(by_season.items()):
        print(f"season {season}: {len(rows)} players")
    if failures:
        print(f"{len(failures)} sheets failed:")
        for f in failures:
            print("  ", f)
    OUT_PATH.write_text(json.dumps({
        "fetched": date.today().isoformat(),
        "seasons": by_season,
    }, indent=1))
    print(f"Wrote -> {OUT_PATH}")


if __name__ == "__main__":
    main()
