"""restore_feedrev_pas.py — fix the 8 feed-revision MLB PAs found by the audit.

Same approach as restore_missing_pitch_pas.py (full PA re-scrape → delete old
rows → insert the correct sequence, carrying Wally's Pitch Type retags +
supplement fields by physics match), with two fixes learned since:

  1. Rows are written the sheets_append way: USER_ENTERED, text-forcing only
     the columns whose canonical type is text, NA→blank, then number-format
     paste from row 2 — so numerics sort with the originals (no RAW+str bug).
  2. Position-player pitchers' new pitches are tagged EP per Wally's convention;
     Jansen's pitches keep feed types.

  python3 scripts/restore_feedrev_pas.py           # DRY RUN
  python3 scripts/restore_feedrev_pas.py --apply
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gspread
import backfill_supplement as B
import sheets_append as SA
from Pitcher2026 import BaseballSavantFocusedDownloader

APPLY = '--apply' in sys.argv
CASES = [('824295', 80), ('824295', 81), ('824290', 78), ('824290', 79),
         ('823867', 72), ('825081', 67), ('823374', 78), ('823535', 75)]
# every unmatched (new) pitch in these PAs gets EP (position players pitching)
EP_PAS = {'824290_078', '824290_079', '823867_072', '825081_067',
          '823374_078', '823535_075'}
CARRY = ['Pitch Type', 'ArmAngle', 'BatSpeed', 'SwingLength', 'AttackAngle', 'AttackDirection',
         'SwingPathTilt', 'RunExp', 'xBA', 'xSLG', 'xwOBA', 'Barrel', 'Runners']
NA_STRINGS = {'<NA>', 'nan', 'NaN', 'NAN', 'None', 'NaT', '#N/A'}


def sf(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def key(velo, px, pz, ivb):
    return (round(sf(velo) if sf(velo) is not None else -9, 1),
            round(sf(px) if sf(px) is not None else -9, 2),
            round(sf(pz) if sf(pz) is not None else -9, 2),
            round(sf(ivb) if sf(ivb) is not None else -99, 1))


def main():
    dl = BaseballSavantFocusedDownloader()
    fresh = {}
    for pk, ab in CASES:
        if pk not in fresh:
            fresh[pk] = dl.download_game_data(int(pk))
    pa_fresh = {}
    for pk, ab in CASES:
        sub = fresh[pk][fresh[pk]['PitchID'].str.startswith(f"{pk}_{ab:03d}_")]
        pa_fresh[f"{pk}_{ab:03d}"] = [dict(r) for _, r in sub.iterrows()]
    prefixes = tuple(f"{pk}_{ab:03d}_" for pk, ab in CASES)

    gc = gspread.service_account()
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for wi, ws in enumerate(sh.worksheets()):
            if ws.title.upper() not in B.ALL_TRACKED_TEAMS:
                continue
            if wi:
                time.sleep(0.8)
            unf = ws.get_all_values(value_render_option='UNFORMATTED_VALUE')
            fmt = ws.get_all_values()
            if not fmt or len(fmt) < 2:
                continue
            header = fmt[0]
            ci = {n: j for j, n in enumerate(header) if n}
            if 'PitchID' not in ci:
                continue
            pcol = ci['PitchID']
            ncol = len(header)
            tab_pas = {}
            for li in range(1, len(fmt)):
                pid = str(fmt[li][pcol]) if pcol < len(fmt[li]) else ''
                if pid.startswith(prefixes):
                    tab_pas.setdefault('_'.join(pid.split('_')[:2]), []).append((li, fmt[li]))
            # which whole-missing PAs belong to this tab? (no existing rows to find,
            # so use the fresh scrape's PTeam vs tab name)
            for pa_key, rows_ in pa_fresh.items():
                if pa_key in tab_pas or not rows_:
                    continue
                if str(rows_[0].get('PTeam', '')).upper() == ws.title.upper():
                    tab_pas[pa_key] = []
            if not tab_pas:
                continue
            # canonical per-column typing from existing (non-affected) rows
            base_num = [False] * ncol
            seen = [False] * ncol
            affected_lis = {li for lst in tab_pas.values() for li, _ in lst}
            for li in range(1, len(unf)):
                if li in affected_lis:
                    continue
                row = unf[li]
                for c in range(min(ncol, len(row))):
                    if seen[c] or row[c] in ('', None):
                        continue
                    base_num[c] = isinstance(row[c], (int, float)) and not isinstance(row[c], bool)
                    seen[c] = True

            del_idx = []
            add_rows = []
            for pa_key, oldrows in tab_pas.items():
                omap = {}
                for li, row in oldrows:
                    def cell(c):
                        return row[ci[c]] if c in ci and ci[c] < len(row) else None
                    omap[key(cell('Velocity'), cell('PlateX'), cell('PlateZ'), cell('IndVertBrk'))] = row
                for fr in pa_fresh[pa_key]:
                    fk = key(fr.get('Velocity'), fr.get('PlateX'), fr.get('PlateZ'), fr.get('IndVertBrk'))
                    old = omap.get(fk)
                    outrow = {h: fr.get(h) for h in header}
                    if old is not None:
                        for f in CARRY:
                            if f in ci and ci[f] < len(old):
                                outrow[f] = old[ci[f]]
                    elif pa_key in EP_PAS:
                        outrow['Pitch Type'] = 'EP'
                    # typed cell values: NA→blank; text cols apostrophe-forced;
                    # numeric cols raw so USER_ENTERED parses them
                    vals = []
                    for c, h in enumerate(header):
                        v = outrow.get(h)
                        v = '' if v is None else str(v)
                        if v.strip() in NA_STRINGS:
                            v = ''
                        elif v != '' and not base_num[c]:
                            v = "'" + v
                        vals.append(v)
                    add_rows.append((pa_key, fr.get('PitchID'), vals, old is not None))
                del_idx += [li for li, _ in oldrows]

            print(f"\n[{label}/{ws.title}]  delete {len(del_idx)} old rows, add {len(add_rows)}:")
            for pa_key, pid, vals, matched in sorted(add_rows):
                d = vals[ci['Description']].lstrip("'")
                c = vals[ci['Count']].lstrip("'")
                pt = vals[ci['Pitch Type']].lstrip("'")
                ev = vals[ci['Event']].lstrip("'") if 'Event' in ci else ''
                print(f"   {pid}  {d:16s} ({c}) type={pt:3s} {('EVENT='+ev) if ev else '':20s}"
                      f"{'  <== NEW PITCH' if not matched else ''}")
            if APPLY:
                for li in sorted(del_idx, reverse=True):
                    ws.delete_rows(li + 1)
                start = len(ws.col_values(1)) + 1
                end = start + len(add_rows) - 1
                if end > ws.row_count:
                    ws.add_rows(end - ws.row_count + 100)
                rng = f"A{start}:{gspread.utils.rowcol_to_a1(1, ncol)[:-1]}{end}"
                ws.update(rng, [v for _, _, v, _ in add_rows], value_input_option='USER_ENTERED')
                SA._paste_number_formats_from_row(ws, 2, start, end, ncol)
                time.sleep(1.0)

    print(f"\n=== {'APPLIED' if APPLY else 'DRY RUN'} ===")
    if APPLY:
        print("Next: backfill DET,ATL,COL,CWS,PIT to fill the new pitches' supplement fields.")


if __name__ == '__main__':
    main()
