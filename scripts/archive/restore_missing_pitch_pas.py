"""restore_missing_pitch_pas.py — fully restore the 10 feed-revision PAs.

For each PA: re-scrape the game (current feed truth), delete the PA's existing Sheet
rows, and reinsert the correct sequence. Physics, Description, Count, Event, hit
data and PitchID come from the fresh feed; Pitch Type retags + Statcast supplement
fields (ArmAngle, xwOBA, xBA/xSLG, RunExp, Barrel, bat tracking, Runners) are
carried over from the old rows by matching each pitch on its physics. The one new
pitch per PA gets the raw feed type (overridden to SL for Holton) and its
supplement fields fill in on the next backfill_supplement run.

This reverts the 14 misaligned OUTCOME_MISMATCH edits AND inserts the missing pitch.

  python3 scripts/restore_missing_pitch_pas.py           # DRY RUN
  python3 scripts/restore_missing_pitch_pas.py --apply
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gspread
import backfill_supplement as B
from Pitcher2026 import BaseballSavantFocusedDownloader

APPLY = '--apply' in sys.argv
CASES = [('824448', 43), ('824527', 53), ('824525', 45), ('824280', 51), ('824600', 48),
         ('823056', 69), ('823545', 25), ('823700', 63), ('822725', 65), ('822968', 70)]
NEWTYPE_OVERRIDE = {'824527_053': 'SL'}   # Holton's cutter is retagged slider
CARRY = ['Pitch Type', 'ArmAngle', 'BatSpeed', 'SwingLength', 'AttackAngle', 'AttackDirection',
         'SwingPathTilt', 'RunExp', 'xBA', 'xSLG', 'xwOBA', 'Barrel', 'Runners']


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
    prefixes = {f"{pk}_{ab:03d}_" for pk, ab in CASES}

    gc = gspread.service_account()
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for i, ws in enumerate(sh.worksheets()):
            if i:
                time.sleep(1.0)
            data = B.read_sheet_with_retry(ws)
            if not data or len(data) < 2:
                continue
            header = data[0]
            ci = {n: j for j, n in enumerate(header) if n}
            if 'PitchID' not in ci:
                continue
            pcol = ci['PitchID']
            tab_pas = {}
            for li in range(1, len(data)):
                pid = data[li][pcol] if pcol < len(data[li]) else ''
                for pref in prefixes:
                    if pid.startswith(pref):
                        tab_pas.setdefault(pref[:-1], []).append((li, data[li]))
            if not tab_pas:
                continue
            del_idx = []
            add_rows = []   # (pa_key, pitchid, values, matched)
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
                    elif pa_key in NEWTYPE_OVERRIDE:
                        outrow['Pitch Type'] = NEWTYPE_OVERRIDE[pa_key]
                    vals = [('' if outrow.get(h) is None else str(outrow.get(h))) for h in header]
                    add_rows.append((pa_key, fr.get('PitchID'), vals, old is not None))
                del_idx += [li for li, _ in oldrows]

            print(f"\n[{label}/{ws.title}]  delete {len(del_idx)} old rows, add {len(add_rows)}:")
            for pa_key, pid, vals, matched in sorted(add_rows):
                d = vals[ci['Description']]; c = vals[ci['Count']]; pt = vals[ci['Pitch Type']]
                ev = vals[ci['Event']] if 'Event' in ci else ''
                print(f"   {pid}  {d:16s} ({c}) type={pt:3s} {('EVENT='+ev) if ev else '':18s}{'  <== NEW PITCH' if not matched else ''}")
            if APPLY:
                for li in sorted(del_idx, reverse=True):
                    ws.delete_rows(li + 1)
                ws.append_rows([v for _, _, v, _ in add_rows], value_input_option='RAW')

    print(f"\n=== {'APPLIED' if APPLY else 'DRY RUN'} ===")
    if APPLY:
        print("Next: run backfill_supplement.py to fill the 10 new pitches' Statcast fields.")
    else:
        print("dry run — re-run with --apply once the rows look right.")


if __name__ == '__main__':
    main()
