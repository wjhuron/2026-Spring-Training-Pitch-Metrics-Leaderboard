"""audit_full.py — comprehensive post-fix verification audit. READ-ONLY.

Verifies every change from the 2026-07-05/06 data-quality session against ground
truth, plus a general integrity sweep of every sheet tab:

  1. PitchID integrity: format, duplicates (in-tab + cross-tab), per-PA
     contiguity (pitch numbers 1..N with no gaps).
  2. Sheet-vs-feed completeness: per-PA pitch counts for EVERY game in the
     sheets vs the current MLB Stats API feed (catches new feed revisions too).
  3. Restored-PA correctness: the 10 rebuilt PAs replayed against the feed
     (pre-pitch count + description category per pitch).
  4. Backfill alignment: the 549 previously-shifted pitches' RunExp/xwOBA vs
     the auto-ball-aligned Savant values; the 10 new pitches' supplement filled.
  5. Spin re-adds: all 1,142 present as numbers matching the curated file.
  6. Cell typing: numeric columns holding text values; NA-like strings anywhere.
  7. Freshness: max Game Date per tab.

Usage: python3 scripts/audits/audit_full.py
"""
import os, sys, re, json, time, pickle, datetime, requests
from collections import defaultdict, Counter
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import gspread
import scrapers.backfill_supplement as B

SCRATCH = ('/private/tmp/claude-501/-Users-wallyhuron-Huronalytics/'
           'c95d9ba9-9386-4e80-a03e-e4563719adbb/scratchpad')
RESTORED = ['824448_043', '824527_053', '824525_045', '824280_051', '824600_048',
            '823056_069', '823545_025', '823700_063', '822725_065', '822968_070']
PID_RE = re.compile(r'^\d{6}_\d{3}_\d{2}$')
NA_STRINGS = {'<NA>', 'nan', 'NaN', 'NAN', 'None', 'NaT', '#N/A', '#n/a'}
# columns whose values are legitimately text (supplement RAW-written, ids, labels)
TEXT_OK = set(B.SUPPLEMENT_MAP.keys()) | {
    'Runners', 'Count', 'Description', 'Event', 'PitchID', 'Pitcher', 'Batter',
    'PTeam', 'BTeam', 'Throws', 'Bats', 'Pitch Type', 'BBType', 'Inning'}
PBP = "https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay"
EPOCH = datetime.date(1899, 12, 30)


def sf(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def desc_cat(d):
    d = (d or '').lower()
    if 'in play' in d:
        return 'inplay'
    if 'hit by pitch' in d:
        return 'hbp'
    if 'foul tip' in d or 'foul bunt' in d:
        return 'strike'
    if 'foul' in d:
        return 'foul'
    if 'strike' in d or 'missed bunt' in d:
        return 'strike'
    if 'ball' in d or 'pitchout' in d:
        return 'ball'
    return '?'


def main():
    issues = defaultdict(list)

    # ---------- read every tab once ----------
    gc = gspread.service_account()
    tabs = {}          # (wb, tab) -> {'header':…, 'ci':…, 'rows':[…]}
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for wi, ws in enumerate(sh.worksheets()):
            time.sleep(0.8)
            try:
                vals = ws.get_all_values(value_render_option='UNFORMATTED_VALUE')
            except Exception as e:
                issues['READ_FAIL'].append(f"{label}/{ws.title}: {e}")
                continue
            if not vals or len(vals) < 2 or 'PitchID' not in vals[0]:
                continue
            ci = {n: j for j, n in enumerate(vals[0]) if n}
            tabs[(label, ws.title)] = {'header': vals[0], 'ci': ci, 'rows': vals[1:]}
    print(f"tabs with pitch data: {len(tabs)}", flush=True)

    # ---------- 1. PitchID integrity + 6. typing + 7. freshness ----------
    pid_owner = {}
    sheet_pa_count = defaultdict(int)
    pa_pitchnums = defaultdict(set)
    restored_rows = defaultdict(list)     # pa -> [(tabkey, row)]
    all_pids_by_tab = {}
    for tk, T in tabs.items():
        ci = T['ci']; pc = ci['PitchID']
        ncol = len(T['header'])
        # canonical numeric-ness per column (majority vote over non-empty)
        num_ct = [0] * ncol; str_ct = [0] * ncol
        maxdate = None
        pids = set()
        for row in T['rows']:
            pid = str(row[pc]) if pc < len(row) else ''
            if not pid:
                continue
            if not PID_RE.match(pid):
                issues['MALFORMED_PID'].append(f"{tk[0]}/{tk[1]}: '{pid}'")
                continue
            if pid in pids:
                issues['DUP_PID_IN_TAB'].append(f"{tk[0]}/{tk[1]}: {pid}")
            pids.add(pid)
            if pid in pid_owner and pid_owner[pid] != tk:
                issues['DUP_PID_CROSS_TAB'].append(f"{pid}: {pid_owner[pid]} + {tk}")
            pid_owner[pid] = tk
            pk, ab, pn = pid.split('_')
            sheet_pa_count[(pk, int(ab))] += 1
            pa_pitchnums[(pk, int(ab))].add(int(pn))
            pref = f"{pk}_{ab}"
            if pref in RESTORED:
                restored_rows[pref].append((tk, row))
            for c in range(min(ncol, len(row))):
                v = row[c]
                if v == '' or v is None:
                    continue
                if isinstance(v, str) and v.strip() in NA_STRINGS:
                    issues['NA_STRING'].append(f"{tk[0]}/{tk[1]} {pid} col='{T['header'][c]}' val='{v}'")
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    num_ct[c] += 1
                else:
                    str_ct[c] += 1
            if 'Game Date' in ci and ci['Game Date'] < len(row):
                gv = row[ci['Game Date']]
                if isinstance(gv, (int, float)):
                    maxdate = max(maxdate or 0, gv)
        all_pids_by_tab[tk] = pids
        # numeric columns holding stray text values
        for c in range(ncol):
            name = T['header'][c]
            if not name or name in TEXT_OK:
                continue
            if num_ct[c] >= 20 and str_ct[c] > 0 and num_ct[c] / (num_ct[c] + str_ct[c]) > 0.9:
                # find up to 2 examples
                exs = []
                for row in T['rows']:
                    if c < len(row) and isinstance(row[c], str) and row[c] not in ('',):
                        pid = row[pc] if pc < len(row) else '?'
                        exs.append(f"{pid}='{row[c]}'")
                        if len(exs) == 2:
                            break
                issues['TEXT_IN_NUMERIC_COL'].append(
                    f"{tk[0]}/{tk[1]} col='{name}': {str_ct[c]} text of {num_ct[c]+str_ct[c]} e.g. {exs}")
        if maxdate:
            d = EPOCH + datetime.timedelta(days=int(maxdate))
            T['maxdate'] = str(d)
    # per-PA contiguity
    for (pk, ab), nums in pa_pitchnums.items():
        if nums != set(range(1, max(nums) + 1)):
            issues['PA_GAP'].append(f"{pk}_{ab:03d}: pitch numbers {sorted(nums)}")

    print("integrity pass done", flush=True)

    # ---------- 2. sheet-vs-feed completeness (every game) ----------
    from concurrent.futures import ThreadPoolExecutor
    game_pks = sorted({pk for (pk, ab) in sheet_pa_count})
    print(f"fetching feeds for {len(game_pks)} games ...", flush=True)

    def fetch(pk):
        try:
            return pk, requests.get(PBP.format(pk=pk), timeout=30).json()
        except Exception:
            return pk, None

    feeds = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        for i, (pk, js) in enumerate(ex.map(fetch, game_pks)):
            feeds[pk] = js
            if (i + 1) % 400 == 0:
                print(f"  {i + 1}/{len(game_pks)}", flush=True)
    fetch_fail = [pk for pk, js in feeds.items() if not js]
    for pk in fetch_fail:
        issues['FEED_FETCH_FAIL'].append(pk)

    count_mismatch = []
    for pk, js in feeds.items():
        if not js:
            continue
        for play in js.get('allPlays', []):
            ab = play.get('atBatIndex', 0) + 1
            fn = sum(1 for e in play.get('playEvents', []) if e.get('isPitch'))
            pn = sheet_pa_count.get((pk, ab), 0)
            if fn and pn and pn != fn:
                count_mismatch.append(
                    f"{pk}_{ab:03d}: sheet={pn} feed={fn} "
                    f"({play.get('matchup',{}).get('batter',{}).get('fullName')} vs "
                    f"{play.get('matchup',{}).get('pitcher',{}).get('fullName')}, "
                    f"{play.get('result',{}).get('event')})")
    issues['PA_COUNT_VS_FEED'] = count_mismatch
    print("feed completeness pass done", flush=True)

    # ---------- 3. restored PAs vs feed replay ----------
    for pref in RESTORED:
        pk, abs_ = pref.split('_')
        ab = int(abs_)
        js = feeds.get(pk)
        if not js:
            issues['RESTORED_CHECK'].append(f"{pref}: feed unavailable")
            continue
        play = js['allPlays'][ab - 1]
        replay = []          # (pre_count, category)
        b = s = 0
        for e in play.get('playEvents', []):
            if e.get('isPitch'):
                d = e.get('details', {}).get('description', '')
                replay.append((f"{b}-{s}", desc_cat(d)))
                c = desc_cat(d)
                if c == 'ball':
                    b += 1
                elif c == 'strike':
                    s += 1
                elif c == 'foul':
                    s += 1 if s < 2 else 0
            else:
                ad = str(e.get('details', {}).get('description', '')).lower()
                if 'automatic ball' in ad:
                    b += 1
                elif 'automatic strike' in ad:
                    s += 1
        rows = sorted(restored_rows.get(pref, []),
                      key=lambda t: str(t[1][tabs[t[0]]['ci']['PitchID']]))
        if len(rows) != len(replay):
            issues['RESTORED_CHECK'].append(f"{pref}: sheet has {len(rows)} pitches, feed {len(replay)}")
            continue
        for i, (tk, row) in enumerate(rows):
            ci = tabs[tk]['ci']
            scount = str(row[ci['Count']]) if 'Count' in ci else '?'
            sdesc = str(row[ci['Description']]) if 'Description' in ci else '?'
            fcount, fcat = replay[i]
            if scount != fcount:
                issues['RESTORED_CHECK'].append(
                    f"{pref} pitch {i+1}: sheet count '{scount}' != feed '{fcount}'")
            if desc_cat(sdesc) != fcat and not (desc_cat(sdesc) == 'strike' and fcat == 'foul'):
                issues['RESTORED_CHECK'].append(
                    f"{pref} pitch {i+1}: sheet desc '{sdesc}' vs feed category '{fcat}'")
    print("restored-PA pass done", flush=True)

    # ---------- 4. backfill alignment on shifted pitches + new-pitch supplement ----------
    shifted = pickle.load(open(os.path.join(ROOT, 'data', '_shifted_pitchids.pkl'), 'rb'))
    sc = pickle.load(open(os.path.join(ROOT, 'data', '_statcast2026_diff.pkl'), 'rb'))
    bypa = defaultdict(list)
    for r in sc.itertuples(index=False):
        try:
            bypa[(int(r.game_pk), int(r.at_bat_number))].append((int(r.pitch_number), r))
        except Exception:
            continue
    aligned = {}
    for (pk, ab), evs in bypa.items():
        evs.sort(key=lambda t: t[0])
        feed = 0
        for pn, r in evs:
            if 'automatic' in str(r.description or '').lower():
                continue
            feed += 1
            aligned[f"{pk}_{ab:03d}_{feed:02d}"] = r
    dec_run = B.ROUND_DECIMALS.get('RunExp', 1)
    dec_xw = B.ROUND_DECIMALS.get('xwOBA', 1)
    n_ok = n_bad = n_blank = 0
    for pid in shifted:
        tk = pid_owner.get(pid)
        if tk is None:
            continue
        T = tabs[tk]; ci = T['ci']
        row = next((r for r in T['rows'] if str(r[ci['PitchID']]) == pid), None)
        sr = aligned.get(pid)
        if row is None or sr is None:
            continue
        sav_run = sf(getattr(sr, 'delta_pitcher_run_exp', None))
        sheet_run = sf(row[ci['RunExp']]) if 'RunExp' in ci and ci['RunExp'] < len(row) else None
        if sav_run is not None:
            if sheet_run is None:
                n_blank += 1
                issues['SHIFTED_STILL_BLANK'].append(f"{pid}: RunExp blank, savant {sav_run}")
            elif abs(sheet_run - round(sav_run, dec_run)) <= 10 ** -dec_run:
                n_ok += 1
            else:
                n_bad += 1
                issues['SHIFTED_WRONG'].append(f"{pid}: RunExp sheet={sheet_run} savant={round(sav_run, dec_run)}")
        # xwOBA on batted balls
        sav_xw = sf(getattr(sr, 'estimated_woba_using_speedangle', None))
        sheet_xw = sf(row[ci['xwOBA']]) if 'xwOBA' in ci and ci['xwOBA'] < len(row) else None
        if sav_xw is not None and sheet_xw is not None:
            if abs(sheet_xw - round(sav_xw, dec_xw)) > 10 ** -dec_xw:
                issues['SHIFTED_WRONG'].append(f"{pid}: xwOBA sheet={sheet_xw} savant={round(sav_xw, dec_xw)}")
        elif sav_xw is not None and sheet_xw is None:
            issues['SHIFTED_STILL_BLANK'].append(f"{pid}: xwOBA blank, savant {round(sav_xw, dec_xw)}")
    print(f"shifted-pitch alignment: {n_ok} RunExp correct, {n_bad} wrong, {n_blank} blank", flush=True)

    # new pitches' supplement filled?
    NEWP = {'823545_025_01', '822968_070_05', '824280_051_05', '823700_063_04',
            '824527_053_06', '824448_043_01', '822725_065_05', '824600_048_05',
            '824525_045_06', '823056_069_05'}
    for pid in sorted(NEWP):
        tk = pid_owner.get(pid)
        if tk is None:
            issues['NEW_PITCH'].append(f"{pid}: NOT FOUND in any tab")
            continue
        T = tabs[tk]; ci = T['ci']
        row = next((r for r in T['rows'] if str(r[ci['PitchID']]) == pid), None)
        sr = aligned.get(pid)
        aa = row[ci['ArmAngle']] if 'ArmAngle' in ci and ci['ArmAngle'] < len(row) else ''
        re_ = row[ci['RunExp']] if 'RunExp' in ci and ci['RunExp'] < len(row) else ''
        sav_aa = sf(getattr(sr, 'arm_angle', None)) if sr is not None else None
        if sav_aa is not None and (aa == '' or aa is None):
            issues['NEW_PITCH'].append(f"{pid}: ArmAngle still blank (savant has {sav_aa})")
        if sr is not None and sf(getattr(sr, 'delta_pitcher_run_exp', None)) is not None and (re_ == '' or re_ is None):
            issues['NEW_PITCH'].append(f"{pid}: RunExp still blank")

    # ---------- 5. spin re-adds ----------
    spin = json.load(open(os.path.join(SCRATCH, 'spin_readd.json')))
    sp_ok = 0
    for pid, want in spin.items():
        tk = pid_owner.get(pid)
        if tk is None:
            issues['SPIN_READD'].append(f"{pid}: not found in any tab")
            continue
        T = tabs[tk]; ci = T['ci']
        row = next((r for r in T['rows'] if str(r[ci['PitchID']]) == pid), None)
        v = row[ci['Spin Rate']] if 'Spin Rate' in ci and ci['Spin Rate'] < len(row) else ''
        if isinstance(v, (int, float)) and int(v) == int(want):
            sp_ok += 1
        else:
            issues['SPIN_READD'].append(f"{pid}: sheet={v!r} expected {want}")
    print(f"spin re-adds verified: {sp_ok}/{len(spin)}", flush=True)

    # ---------- report ----------
    out = os.path.join(SCRATCH, 'audit_full_report.txt')
    with open(out, 'w') as f:
        f.write("=== FRESHNESS (max Game Date per tab) ===\n")
        for tk, T in sorted(tabs.items()):
            f.write(f"  {tk[0]}/{tk[1]}: {T.get('maxdate', '?')}  ({len(T['rows'])} rows)\n")
        f.write("\n=== ISSUES ===\n")
        for k in sorted(issues):
            lst = issues[k]
            f.write(f"\n[{k}] {len(lst)}\n")
            for e in lst[:40]:
                f.write(f"   {e}\n")
            if len(lst) > 40:
                f.write(f"   ... and {len(lst) - 40} more\n")
    print(f"\nwrote {out}")
    print("\n=== SUMMARY ===")
    print(f"  tabs audited: {len(tabs)}   pitches: {len(pid_owner)}   games: {len(game_pks)}")
    for k in sorted(issues):
        print(f"  {k}: {len(issues[k])}")
    if not issues:
        print("  NO ISSUES FOUND")


if __name__ == '__main__':
    main()
