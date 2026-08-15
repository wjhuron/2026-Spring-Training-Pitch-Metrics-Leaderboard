"""audit_all_columns.py — full data-quality profile of every column across the
30 MLB tabs (the six division books). READ-ONLY.

For each column it applies the right check:
  * CATEGORICAL -> full distinct-value set + counts, flagged against the allowed set
  * NUMERIC     -> blanks, non-numeric text, and out-of-physical-range values (+examples)
  * FORMAT      -> PitchID / Count / tilt / date pattern conformance
Plus global checks: PitchID uniqueness, and per-tab row counts.

  python3 scripts/audits/audit_all_columns.py
"""
import os, sys, re, time, argparse, warnings
from collections import defaultdict, Counter
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
import gspread
import scrapers.backfill_supplement as B

# --milb audits the ROC/AAA tabs instead of the 30 MLB tabs (Statcast is thinner
# in the minors, so blanks are expected — we still check formats/ranges/domains).
_ap = argparse.ArgumentParser()
_ap.add_argument('--milb', action='store_true')
MILB = _ap.parse_args().milb

# categorical -> allowed set (None = just list, don't flag)
CATEGORICAL = {
    'Throws': {'L', 'R'},
    'Bats': {'L', 'R'},
    'Outs': {'0', '1', '2'},
    'BBType': {'ground_ball', 'line_drive', 'fly_ball', 'popup', 'bunt', ''},
    'Barrel': {'', '0', '1', '2', '3', '4', '5', '6'},
    'Description': {'In Play', 'Swinging Strike', 'Ball', 'Called Strike', 'Hit By Pitch',
                    'Foul', 'Foul Bunt', 'Bunt Foul Tip', 'Missed Bunt', 'Intent Ball',
                    'Pitchout', 'Swinging Pitchout', 'Foul Pitchout'},
    'Pitch Type': None,   # retagged; list distinct
    'Event': None,        # many valid; list distinct
    'PTeam': None,
    'BTeam': None,
}
# numeric -> (lo, hi) physical bounds
NUMERIC = {
    'Velocity': (40, 106), 'Spin Rate': (0, 3800),
    'IndVertBrk': (-35, 35), 'HorzBrk': (-35, 35), 'xIndVrtBrk': (-35, 35), 'xHorzBrk': (-35, 35),
    'RelPosZ': (-1, 8), 'RelPosX': (-7, 7), 'Extension': (2, 9), 'ArmAngle': (-100, 180),
    'PlateZ': (-4, 8), 'PlateX': (-4, 4), 'SzTop': (2, 5), 'SzBot': (0.3, 3),
    'VAA': (-16, 3), 'HAA': (-9, 9),
    'ExitVelo': (5, 130), 'LaunchAngle': (-95, 95), 'Distance': (0, 560),
    'HC_X': (0, 260), 'HC_Y': (0, 260),
    'xBA': (0, 5), 'xSLG': (0, 6), 'xwOBA': (0, 5), 'RunExp': (-3, 3),
    'BatSpeed': (10, 110), 'SwingLength': (1, 15), 'AttackAngle': (-95, 95),
    'AttackDirection': (-95, 95), 'SwingPathTilt': (-95, 95),
}
FORMAT = {
    'PitchID': re.compile(r'^\d+_\d{3}_\d{2}$'),
    'Count': re.compile(r'^[0-3]-[0-2]$'),
    'RTilt': re.compile(r'^\d{1,2}:\d{2}$'),
    'OTilt': re.compile(r'^\d{1,2}:\d{2}$'),
    'Game Date': re.compile(r'^\d{4}-\d{2}-\d{2}$'),
}
TEAM_OK = set(B.ALL_TRACKED_TEAMS) | {'ROC', 'AAA', 'FCL', 'EP', 'AL', 'NL'}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    gc = gspread.service_account()
    cat = {c: Counter() for c in CATEGORICAL}
    num = {c: {'n': 0, 'blank': 0, 'text': [], 'oor': [], 'lo': 1e9, 'hi': -1e9} for c in NUMERIC}
    fmt = {c: {'bad': [], 'blank': 0} for c in FORMAT}
    pid_seen = {}
    dup_pids = []
    tab_rows = {}

    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for ws in sh.worksheets():
            t = ws.title.upper()
            if MILB:
                if t not in ('ROC', 'AAA'):
                    continue
            elif t not in B.ALL_TRACKED_TEAMS or t in ('ROC', 'AAA', 'FCL'):
                continue  # MLB tabs only
            time.sleep(0.5)
            vals = ws.get_all_values()
            if not vals or 'PitchID' not in vals[0]:
                continue
            hdr = vals[0]; ci = {n: j for j, n in enumerate(hdr) if n}
            tab_rows[t] = len(vals) - 1
            for r in vals[1:]:
                def g(col):
                    j = ci.get(col)
                    return r[j] if j is not None and j < len(r) else ''
                # categorical
                for c in CATEGORICAL:
                    if c in ci:
                        cat[c][str(g(c))] += 1
                # numeric
                for c, (lo, hi) in NUMERIC.items():
                    if c not in ci:
                        continue
                    v = g(c)
                    if v == '' or v is None:
                        num[c]['blank'] += 1; continue
                    fv = fnum(v)
                    if fv is None:
                        if len(num[c]['text']) < 6:
                            num[c]['text'].append(f"{g('PitchID')}={v!r}")
                        continue
                    num[c]['n'] += 1
                    num[c]['lo'] = min(num[c]['lo'], fv); num[c]['hi'] = max(num[c]['hi'], fv)
                    if not (lo <= fv <= hi) and len(num[c]['oor']) < 8:
                        num[c]['oor'].append(f"{g('PitchID')}={fv}")
                    elif not (lo <= fv <= hi):
                        num[c]['oor'].append(None)  # count only
                # format
                for c, rx in FORMAT.items():
                    if c not in ci:
                        continue
                    v = str(g(c))
                    if v == '':
                        fmt[c]['blank'] += 1
                    elif not rx.match(v):
                        if len(fmt[c]['bad']) < 8:
                            fmt[c]['bad'].append(f"{g('PitchID')} {c}={v!r}")
                # pid uniqueness
                pid = str(g('PitchID'))
                if pid:
                    if pid in pid_seen:
                        if len(dup_pids) < 12:
                            dup_pids.append(f"{pid} ({pid_seen[pid]} & {t})")
                    else:
                        pid_seen[pid] = t
            print(f"  [{label}/{t}] {tab_rows[t]} rows", flush=True)

    print(f"\n{'='*70}\nTOTAL rows: {sum(tab_rows.values())}   unique PitchIDs: {len(pid_seen)}   dup PitchIDs: {len(dup_pids)}")
    if dup_pids:
        print("  DUP examples:", dup_pids)

    print(f"\n{'='*70}\nCATEGORICAL columns (distinct values):")
    for c in CATEGORICAL:
        allowed = CATEGORICAL[c]
        items = sorted(cat[c].items(), key=lambda kv: -kv[1])
        if allowed is not None:
            bad = {k: v for k, v in cat[c].items() if k not in allowed}
            tag = f"  <<< UNEXPECTED: {bad}" if bad else "  (all allowed)"
            print(f"\n  {c}{tag}")
            for k, v in items:
                mark = '' if (k in allowed) else '  <<<'
                print(f"      {k!r:22s} {v:7d}{mark}")
        else:
            print(f"\n  {c}  ({len(items)} distinct)")
            for k, v in items[:40]:
                print(f"      {k!r:26s} {v:7d}")
            if len(items) > 40:
                print(f"      ... +{len(items)-40} more")

    print(f"\n{'='*70}\nNUMERIC columns (range + anomalies):")
    for c, (lo, hi) in NUMERIC.items():
        s = num[c]
        oor_n = len(s['oor'])
        flag = ''
        if s['text']: flag += f"  TEXT={len(s['text'])}"
        if oor_n: flag += f"  OOR={oor_n}"
        rng = f"[{s['lo']:.2f},{s['hi']:.2f}]" if s['n'] else "[--]"
        print(f"  {c:16s} n={s['n']:7d} blank={s['blank']:7d} range={rng}{flag}")
        if s['text']:
            print(f"      TEXT ex: {s['text'][:6]}")
        exs = [e for e in s['oor'] if e][:8]
        if exs:
            print(f"      OOR ex:  {exs}")

    print(f"\n{'='*70}\nFORMAT columns:")
    for c in FORMAT:
        s = fmt[c]
        print(f"  {c:12s} blank={s['blank']:6d}  bad_format={len(s['bad'])}")
        if s['bad']:
            print(f"      ex: {s['bad']}")


if __name__ == '__main__':
    main()
