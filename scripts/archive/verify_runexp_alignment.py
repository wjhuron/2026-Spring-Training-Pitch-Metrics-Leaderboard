"""verify_runexp_alignment.py — prove the 549 previously-shifted pitches now
carry the CORRECT RunExp (and xwOBA) after the auto-ball fix + reconcile. READ-ONLY.

audit_full.py couldn't check RunExp (the cached Statcast pickle lacks
delta_pitcher_run_exp), so this goes to the source: for every team tab that
contains shifted pitches, download the team's Statcast CSV through the PATCHED
B.download_statcast (feed-aligned numbering) and compare the sheet's stored
RunExp/xwOBA strings against the freshly formatted Savant values.

Usage: python3 scripts/verify_runexp_alignment.py
"""
import os, sys, time, pickle, requests
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import gspread
import backfill_supplement as B


def main():
    shifted = pickle.load(open(os.path.join(ROOT, 'data', '_shifted_pitchids.pkl'), 'rb'))
    gc = gspread.service_account()
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
                            'Accept': 'text/csv'})
    ok = wrong = blank_both = noref = 0
    wrongs = []
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for ws in sh.worksheets():
            if ws.title.upper() not in B.ALL_TRACKED_TEAMS:
                continue
            time.sleep(0.5)
            header = ws.row_values(1)
            if 'PitchID' not in header or 'RunExp' not in header:
                continue
            pids = ws.col_values(header.index('PitchID') + 1)[1:]
            hits = [(i, p) for i, p in enumerate(pids) if p in shifted]
            if not hits:
                continue
            runexp = ws.col_values(header.index('RunExp') + 1)[1:]
            xwoba = ws.col_values(header.index('xwOBA') + 1)[1:] if 'xwOBA' in header else []
            time.sleep(2)
            lookup = B.download_statcast(ws.title, '2026-03-20', '2026-07-06', session)
            if not lookup:
                print(f"[{label}/{ws.title}] no statcast — skipped")
                continue
            t_ok = t_wrong = 0
            for i, pid in hits:
                parts = pid.split('_')
                key = (parts[0], str(int(parts[1])), str(int(parts[2])))
                sr = lookup.get(key, {})
                for col, series in (('RunExp', runexp), ('xwOBA', xwoba)):
                    sheet_v = series[i] if i < len(series) else ''
                    sav_v = sr.get(col, '')
                    if not sav_v and not sheet_v:
                        blank_both += 1
                    elif not sav_v:
                        noref += 1
                    elif str(sheet_v) == str(sav_v):
                        ok += 1; t_ok += 1
                    else:
                        wrong += 1; t_wrong += 1
                        wrongs.append(f"{pid} {col}: sheet='{sheet_v}' savant='{sav_v}'")
            print(f"[{label}/{ws.title}] {len(hits)} shifted pitches: {t_ok} match, {t_wrong} wrong", flush=True)

    print(f"\n=== RESULT ===")
    print(f"  values matching Savant exactly: {ok}")
    print(f"  wrong: {wrong}")
    print(f"  blank on both sides: {blank_both}   savant-has-no-value: {noref}")
    for w in wrongs[:20]:
        print(f"   {w}")


if __name__ == '__main__':
    main()
