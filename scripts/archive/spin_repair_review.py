"""spin_repair_review.py — review CSV for the spin (+RTilt) repairs, with a
per-pitcher per-pitch-type standard-deviation verdict on BOTH the feed spin and
the feed release tilt. READ-ONLY. EP excluded entirely.

For every candidate repair it compares the feed value to that pitcher's own
distribution for that pitch type:
  spin  -> mean +/- SD (rpm)
  RTilt -> circular mean +/- SD (clock; tilt wraps at 12:00 so we use vector stats)
labelling each good (<=1 SD) / review (1-2 SD) / bad (>2 SD) / low_n. A combined
`recommendation` (APPLY / REVIEW / SKIP) folds the two together so you can sort
and blast through it. Writes ~/Downloads/spin_rtilt_repair_review.csv.
"""
import os, sys, csv, time, math, statistics as st, warnings, concurrent.futures as cf
from collections import defaultdict
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import requests, gspread
import backfill_supplement as B
import Pitcher2026

MIN_N = 8; MIN_DELTA = 25; REF_MIN = 700
FEED = "https://statsapi.mlb.com/api/v1.1/game/{}/feed/live"
_DL = Pitcher2026.BaseballSavantFocusedDownloader()


def sf(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def tilt_to_min(s):
    s = str(s).strip()
    if ':' not in s:
        return None
    try:
        h, m = s.split(':'); return (int(h) % 12) * 60 + int(m)   # 0..719
    except ValueError:
        return None


def min_to_tilt(mn):
    mn %= 720
    h = int(mn // 60); m = int(round(mn % 60))
    if m == 60:
        m = 0; h += 1
    h %= 12
    return f"{12 if h == 0 else h}:{m:02d}"


def circ_stats(mins):
    """circular mean + SD (in minutes on a 720-min clock)."""
    S = sum(math.sin(x / 720 * 2 * math.pi) for x in mins) / len(mins)
    C = sum(math.cos(x / 720 * 2 * math.pi) for x in mins) / len(mins)
    mean = (math.atan2(S, C) / (2 * math.pi) * 720) % 720
    diffs = []
    for x in mins:
        d = (x - mean) % 720
        diffs.append(d - 720 if d > 360 else d)
    sd = st.pstdev(diffs) if len(diffs) > 1 else 0.0
    return mean, (sd if sd >= 1 else 1.0)


def circ_diff(a, mean):
    d = (a - mean) % 720
    return d - 720 if d > 360 else d


def feed_spin(pk):
    try:
        plays = requests.get(FEED.format(pk), timeout=60).json()['liveData']['plays']['allPlays']
    except Exception:
        return None
    out = {}
    for play in plays:
        ab = play.get('atBatIndex', 0) + 1
        for ev in play.get('playEvents', []):
            if not ev.get('isPitch', False):
                continue
            pn = ev.get('pitchNumber') or 0
            br = ev.get('pitchData', {}).get('breaks', {})
            sr = br.get('spinRate'); sd = br.get('spinDirection')
            tilt = _DL.spin_axis_to_tilt(sd) if sd is not None else ''
            out[f"{pk}_{ab:03d}_{pn:02d}"] = (float(sr) if sr is not None else None, tilt or '')
    return out


def main():
    gc = gspread.service_account()
    tabs = []; pks = set()
    print("reading sheets ...", flush=True)
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for ws in sh.worksheets():
            t = ws.title.upper()
            if t not in B.ALL_TRACKED_TEAMS or t in ('ROC', 'AAA', 'FCL'):
                continue
            time.sleep(0.4)
            vals = ws.get_all_values()
            if not vals or 'PitchID' not in vals[0] or 'Spin Rate' not in vals[0]:
                continue
            ci = {n: j for j, n in enumerate(vals[0]) if n}
            tabs.append((ws.title, vals, ci))
            pc = ci['PitchID']
            for r in vals[1:]:
                p = (r[pc] if pc < len(r) else '').split('_')
                if len(p) == 3 and p[0].isdigit():
                    pks.add(int(p[0]))
    print(f"MLB tabs: {len(tabs)}   games: {len(pks)}", flush=True)

    print("pulling feeds ...", flush=True)
    feed = {}; failed = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(feed_spin, pk): pk for pk in pks}
        done = 0
        for fu in cf.as_completed(futs):
            res = fu.result()
            if res is None: failed.append(futs[fu])
            else: feed.update(res)
            done += 1
            if done % 300 == 0: print(f"  {done}/{len(pks)}", flush=True)
    print(f"feed pitches: {len(feed)}   failed: {len(failed)}", flush=True)

    # reference distributions per (pitcher, pitch type), from clean feed values (EP excluded)
    spin_g = defaultdict(list); tilt_g = defaultdict(list); rows = []
    for title, vals, ci in tabs:
        cols = {c: ci.get(c) for c in ['PitchID', 'Spin Rate', 'RTilt', 'Pitcher', 'Pitch Type', 'Velocity', 'Game Date']}
        for ri in range(1, len(vals)):
            r = vals[ri]
            def g(c):
                j = cols[c]; return r[j] if j is not None and j < len(r) else ''
            pid = g('PitchID'); f = feed.get(pid)
            if not f or f[0] is None:
                continue
            pitcher, ptype = g('Pitcher'), g('Pitch Type')
            if ptype == 'EP':
                continue
            if f[0] >= REF_MIN:          # clean pitch -> contributes to both references
                spin_g[(pitcher, ptype)].append(f[0])
                tm = tilt_to_min(f[1])
                if tm is not None:
                    tilt_g[(pitcher, ptype)].append(tm)
            rows.append((title, pid, g('Game Date'), pitcher, ptype, g('Velocity'),
                         sf(g('Spin Rate')), g('RTilt'), f[0], f[1]))
    spin_ref = {k: (st.mean(v), max(st.pstdev(v), 1.0), len(v)) for k, v in spin_g.items() if len(v) >= MIN_N}
    tilt_ref = {k: (circ_stats(v) + (len(v),)) for k, v in tilt_g.items() if len(v) >= MIN_N}

    def vlabel(z):
        z = abs(z)
        return 'good' if z <= 1 else ('review' if z <= 2 else 'bad')

    out = []
    for title, pid, date, pitcher, ptype, velo, stored, rtilt, fspin, ftilt in rows:
        if stored is not None and abs(stored - fspin) <= MIN_DELTA:
            continue
        cat = ('blank_recovered' if stored is None else
               'corrupt_repaired' if (stored < 1000 or abs(stored - fspin) > 800) else 'drift_updated')
        # spin verdict
        sr = spin_ref.get((pitcher, ptype))
        if sr:
            m, s, n = sr; zs = (fspin - m) / s
            sv, savg, ssd, szz = vlabel(zs), int(round(m)), int(round(s)), round(zs, 1)
        else:
            sv, savg, ssd, szz, n = 'low_n', '', '', '', len(spin_g.get((pitcher, ptype), []))
        # rtilt verdict
        tr = tilt_ref.get((pitcher, ptype)); ftm = tilt_to_min(ftilt)
        if tr and ftm is not None:
            (tmean, tsd, tn) = tr; zt = circ_diff(ftm, tmean) / tsd
            tv, tavg, tzz = vlabel(zt), min_to_tilt(tmean), round(zt, 1)
        else:
            tv, tavg, tzz = ('low_n' if ftm is not None else 'no_tilt'), '', ''
        # combined recommendation
        if sv == 'bad':
            rec = 'SKIP-spin'
        elif sv == 'good' and tv in ('good', 'low_n', 'no_tilt'):
            rec = 'APPLY'
        elif tv == 'bad':
            rec = 'REVIEW-tilt'
        else:
            rec = 'REVIEW'
        out.append([rec, sv, tv, cat, pitcher, ptype, title, date, velo, pid, n,
                    '' if stored is None else int(stored), int(round(fspin)), savg, ssd, szz,
                    rtilt, ftilt, tavg, tzz, '' if (rtilt or '') == (ftilt or '') else 'Y'])

    path = os.path.expanduser('~/Downloads/spin_rtilt_repair_review.csv')
    rorder = {'SKIP-spin': 0, 'REVIEW-tilt': 1, 'REVIEW': 2, 'APPLY': 3}
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['recommendation', 'spin_verdict', 'rtilt_verdict', 'category', 'Pitcher',
                    'PitchType', 'Team', 'GameDate', 'Velocity', 'PitchID', 'pitcher_n',
                    'sheet_Spin', 'feed_Spin', 'pitcher_avgSpin', 'pitcher_spinSD', 'spin_SDs_away',
                    'sheet_RTilt', 'feed_RTilt', 'pitcher_avgRTilt', 'rtilt_SDs_away', 'RTilt_differs'])
        for rr in sorted(out, key=lambda z: (rorder.get(z[0], 4), z[4], z[5], z[9])):
            w.writerow(rr)

    byrec = defaultdict(int); bysv = defaultdict(int); bytv = defaultdict(int)
    for z in out:
        byrec[z[0]] += 1; bysv[z[1]] += 1; bytv[z[2]] += 1
    print(f"\nrows (EP excluded): {len(out)}")
    print(f"  recommendation: {dict(byrec)}")
    print(f"  spin_verdict:   {dict(bysv)}")
    print(f"  rtilt_verdict:  {dict(bytv)}")
    print(f"CSV -> {path}")


if __name__ == '__main__':
    main()
