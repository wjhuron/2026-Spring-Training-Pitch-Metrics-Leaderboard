"""Retroactively apply Las Vegas Ballpark weather (density) adjustment to the
xIndVrtBrk / xHorzBrk columns for the Athletics' 2026-06-08 → 2026-06-14
homestand in Summerlin.

The MLB feed reports no elevation for venue 5355 (Las Vegas Ballpark), so the
download-time adjustment defaulted to a factor of 1.0 and the "x" movement
columns were left equal to the raw IndVertBrk / HorzBrk. This script recomputes
them in place:

    xIndVrtBrk = IndVertBrk * factor(game)
    xHorzBrk   = HorzBrk    * factor(game)

where factor(game) = compute_weather_adj_factor(compute_air_density(3010, temp))
uses the SAME pure-physics functions the pipeline uses, with the verified
ballpark elevation (~3010 ft, USGS) and each game's actual feed temperature.

Every pitch thrown in these games was in that thin air, so all three pitching
tabs are corrected: ATH (AL 2026) plus the visiting Brewers and Rockies
pitchers (MIL, COL in NL 2026). Rows are matched precisely by the game_pk
prefix of PitchID, so only these six games are touched.

Dry run by default (reports what would change). Pass --write to apply.
"""

import argparse
import importlib.util
import os
import sys

REPO = '/Users/wallyhuron/Huronalytics'

# Import the pipeline's exact physics + workbook constants
_spec = importlib.util.spec_from_file_location('p26', os.path.join(REPO, 'Pitcher2026.py'))
p26 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p26)

sys.path.insert(0, REPO)
import sheets_append as sa  # SHEETS_AL, SHEETS_NL, _get_client

# Las Vegas Ballpark ground elevation (USGS EPQS at the venue coordinates).
LV_ELEVATION_FT = 3010.0

# game_pk -> recorded feed temperature (F) for the six-game homestand.
GAME_TEMP_F = {
    '824998': 87,   # 2026-06-08 vs MIL
    '824999': 94,   # 2026-06-09 vs MIL
    '824996': 102,  # 2026-06-10 vs MIL
    '824997': 103,  # 2026-06-12 vs COL
    '824995': 102,  # 2026-06-13 vs COL
    '824994': 100,  # 2026-06-14 vs COL
}

# Precompute the density-adjustment factor per game (identical math to the pipeline).
GAME_FACTOR = {
    pk: p26.compute_weather_adj_factor(p26.compute_air_density(LV_ELEVATION_FT, t))
    for pk, t in GAME_TEMP_F.items()
}

# (workbook id, tab name) holding each set of pitchers from these games.
TABS = [
    (sa.SHEETS_AL, 'ATH'),  # Athletics pitchers
    (sa.SHEETS_NL, 'MIL'),  # Brewers pitchers (6/8-6/10)
    (sa.SHEETS_NL, 'COL'),  # Rockies pitchers (6/12-6/14)
]


def _f(s):
    """Parse a sheet string to float, or None if blank/invalid."""
    if s is None:
        return None
    s = str(s).strip()
    if s == '':
        return None
    try:
        return float(s)
    except ValueError:
        return None


def process_tab(gc, wb_id, tab, write):
    import gspread

    ws = gc.open_by_key(wb_id).worksheet(tab)
    values = ws.get_all_values()
    if not values:
        print(f"  [{tab}] empty tab; skipping")
        return

    header = values[0]
    try:
        col = {name: header.index(name) for name in
               ('PitchID', 'IndVertBrk', 'HorzBrk', 'xIndVrtBrk', 'xHorzBrk')}
    except ValueError as e:
        print(f"  [{tab}] missing expected column: {e}; skipping")
        return

    i_pid, i_ivb, i_hb, i_xivb, i_xhb = (
        col['PitchID'], col['IndVertBrk'], col['HorzBrk'],
        col['xIndVrtBrk'], col['xHorzBrk'])

    cells = []          # gspread.Cell objects to write
    matched = 0         # rows belonging to the six games
    updated = 0         # rows with a valid IVB/HB we recomputed
    already_eq = 0      # rows where current xIVB == IVB (premise sanity check)
    per_game = {}       # pk -> count
    samples = []

    for r in range(1, len(values)):
        row = values[r]
        pid = row[i_pid] if i_pid < len(row) else ''
        gpk = pid.split('_')[0] if pid else ''
        if gpk not in GAME_FACTOR:
            continue
        matched += 1
        per_game[gpk] = per_game.get(gpk, 0) + 1

        factor = GAME_FACTOR[gpk]
        ivb = _f(row[i_ivb] if i_ivb < len(row) else '')
        hb = _f(row[i_hb] if i_hb < len(row) else '')
        cur_xivb = _f(row[i_xivb] if i_xivb < len(row) else '')

        if ivb is None and hb is None:
            continue  # no movement data — nothing to adjust

        if ivb is not None and cur_xivb is not None and abs(cur_xivb - ivb) < 1e-9:
            already_eq += 1

        sheet_row = r + 1  # gspread is 1-indexed; header is row 1
        if ivb is not None:
            new_xivb = round(ivb * factor, 1)
            cells.append(gspread.Cell(sheet_row, i_xivb + 1, f"{new_xivb:.1f}"))
        if hb is not None:
            new_xhb = round(hb * factor, 1)
            cells.append(gspread.Cell(sheet_row, i_xhb + 1, f"{new_xhb:.1f}"))
        updated += 1

        if len(samples) < 3:
            samples.append(
                f"      row {sheet_row} (pk {gpk}, f={factor:.4f}): "
                f"IVB {ivb} -> xIVB {round(ivb*factor,1) if ivb is not None else '—'}, "
                f"HB {hb} -> xHB {round(hb*factor,1) if hb is not None else '—'}")

    print(f"  [{tab}] matched {matched} rows across "
          f"{len(per_game)} game(s): "
          + ", ".join(f"{pk}:{n}" for pk, n in sorted(per_game.items())))
    print(f"  [{tab}] {updated} rows to update, "
          f"{already_eq}/{matched} currently have xIVB == IVB (expected pre-fix)")
    for s in samples:
        print(s)

    if write and cells:
        ws.update_cells(cells, value_input_option='USER_ENTERED')
        print(f"  [{tab}] WROTE {len(cells)} cells "
              f"({updated} rows x xIVB+xHB)")
    elif not write:
        print(f"  [{tab}] dry run — no write ({len(cells)} cells would change)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--write', action='store_true',
                    help='Apply the changes (default: dry run)')
    args = ap.parse_args()

    print(f"Las Vegas Ballpark elevation: {LV_ELEVATION_FT:.0f} ft")
    print("Per-game adjustment factors:")
    for pk, t in GAME_TEMP_F.items():
        print(f"  {pk}: {t}F -> factor {GAME_FACTOR[pk]:.4f}")
    print(f"\nMode: {'WRITE' if args.write else 'DRY RUN'}\n")

    gc = sa._get_client()
    for wb_id, tab in TABS:
        process_tab(gc, wb_id, tab, args.write)

    if not args.write:
        print("\nDry run complete. Re-run with --write to apply.")
    else:
        print("\nDone.")


if __name__ == '__main__':
    main()
