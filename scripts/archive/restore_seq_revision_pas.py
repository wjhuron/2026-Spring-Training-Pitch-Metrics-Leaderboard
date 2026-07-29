"""restore_seq_revision_pas.py — fix same-count feed SEQUENCE revisions.

These PAs have the right pitch COUNT but the feed re-sequenced them after the
scrape (leadoff pitch changed, two pitches transposed, or a mid-PA reorder), so
the sheet's per-pitch velo/type no longer matches the feed. Re-scrape each PA
from the current feed, then:
  - base fields (velo, movement, location, spin, RTilt/OTilt, ...) come from the
    fresh scrape = current feed truth.
  - Pitch Type retag is carried from the nearest old pitch by (velo, plate x/z),
    which survives small reprocessing + reordering; a revised pitch with no close
    old match keeps the raw feed type.
  - supplement fields (ArmAngle, xwOBA, RunExp, Barrel, bat tracking, Runners) are
    left blank and refilled by a follow-up backfill against the aligned Savant.
Rows written the sheets_append way (USER_ENTERED + text-force + number formats).

  python3 scripts/restore_seq_revision_pas.py           # DRY RUN
  python3 scripts/restore_seq_revision_pas.py --apply
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gspread
import backfill_supplement as B
import sheets_append as SA
from Pitcher2026 import BaseballSavantFocusedDownloader

APPLY = '--apply' in sys.argv
CASES = [('822739', 12), ('823138', 8), ('824214', 9), ('824215', 83), ('824458', 56)]
NA_STRINGS = {'<NA>', 'nan', 'NaN', 'NAN', 'None', 'NaT', '#N/A'}


def sf(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def match_nearest(fresh_rows, old_rows, ci):
    """Greedy nearest match fresh->old by velo (primary) + plate loc (secondary).
    Returns {fresh PitchID: old_row or None}."""
    avail = list(old_rows)
    res = {}
    # match closest-velo pairs first for stability
    order = sorted(fresh_rows, key=lambda fr: sf(fr.get('Velocity')) or 0)
    for fr in order:
        fv = sf(fr.get('Velocity')); fx = sf(fr.get('PlateX')); fz = sf(fr.get('PlateZ'))
        best, bestd = None, 1e9
        for orow in avail:
            ov = sf(orow[ci['Velocity']]); ox = sf(orow[ci['PlateX']]); oz = sf(orow[ci['PlateZ']])
            if fv is None or ov is None:
                continue
            d = abs(fv - ov) + 2.0 * (abs((fx or 0) - (ox or 0)) + abs((fz or 0) - (oz or 0)))
            if d < bestd:
                bestd, best = d, orow
        if best is not None and bestd < 3.0:
            res[fr['PitchID']] = best
            avail.remove(best)
        else:
            res[fr['PitchID']] = None
    return res


def col_letter(n):
    s = ''
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


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
            if not fmt or len(fmt) < 2 or 'PitchID' not in fmt[0]:
                continue
            header = fmt[0]
            ci = {n: j for j, n in enumerate(header) if n}
            pcol = ci['PitchID']; ncol = len(header)
            tab_pas = {}
            for li in range(1, len(fmt)):
                pid = str(fmt[li][pcol]) if pcol < len(fmt[li]) else ''
                if pid.startswith(prefixes):
                    tab_pas.setdefault('_'.join(pid.split('_')[:2]), []).append((li, fmt[li]))
            if not tab_pas:
                continue
            # canonical typing from non-affected rows
            base_num = [False] * ncol; seen = [False] * ncol
            aff = {li for lst in tab_pas.values() for li, _ in lst}
            for li in range(1, len(unf)):
                if li in aff:
                    continue
                for c in range(min(ncol, len(unf[li]))):
                    if seen[c] or unf[li][c] in ('', None):
                        continue
                    base_num[c] = isinstance(unf[li][c], (int, float)) and not isinstance(unf[li][c], bool)
                    seen[c] = True

            del_idx = []
            add_rows = []
            for pa_key, oldrows in tab_pas.items():
                orows = [row for _, row in oldrows]
                match = match_nearest(pa_fresh[pa_key], orows, ci)
                for fr in pa_fresh[pa_key]:
                    old = match.get(fr['PitchID'])
                    outrow = {h: fr.get(h) for h in header}
                    carried = False
                    if old is not None and 'Pitch Type' in ci:
                        outrow['Pitch Type'] = old[ci['Pitch Type']]
                        carried = True
                    vals = []
                    for c, h in enumerate(header):
                        v = outrow.get(h)
                        v = '' if v is None else str(v)
                        if v.strip() in NA_STRINGS:
                            v = ''
                        elif v != '' and not base_num[c]:
                            v = "'" + v
                        vals.append(v)
                    add_rows.append((pa_key, fr['PitchID'], vals, carried, sf(fr.get('Velocity')),
                                     old[ci['Velocity']] if old is not None else None))
                del_idx += [li for li, _ in oldrows]

            print(f"\n[{label}/{ws.title}] delete {len(del_idx)}, add {len(add_rows)}:")
            for pa_key, pid, vals, carried, fv, ov in sorted(add_rows):
                d = vals[ci['Description']].lstrip("'"); c = vals[ci['Count']].lstrip("'")
                pt = vals[ci['Pitch Type']].lstrip("'")
                tag = f"retag<-{ov}" if carried else "FEED TYPE (revised pitch)"
                print(f"   {pid}  velo={fv}  {d:16s} ({c}) type={pt:3s}  [{tag}]")
            if APPLY:
                for li in sorted(del_idx, reverse=True):
                    ws.delete_rows(li + 1)
                start = len(ws.col_values(1)) + 1
                end = start + len(add_rows) - 1
                if end > ws.row_count:
                    ws.add_rows(end - ws.row_count + 100)
                rng = f"A{start}:{col_letter(ncol)}{end}"
                ws.update(rng, [v for _, _, v, _, _, _ in add_rows], value_input_option='USER_ENTERED')
                SA._paste_number_formats_from_row(ws, 2, start, end, ncol)
                time.sleep(1.0)

    print(f"\n=== {'APPLIED' if APPLY else 'DRY RUN'} ===")
    if APPLY:
        print("Next: backfill WSH,SEA,HOU,CHC to refill supplement for the re-scraped pitches.")


if __name__ == '__main__':
    main()
