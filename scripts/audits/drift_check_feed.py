"""drift_check_feed.py — do the FEED-sourced columns change between the stored
(scrape-time) value and a fresh scrape NOW? READ-ONLY.

Re-scrapes a spread of games with the current Pitcher2026 and compares
Description, Event, BBType, RTilt, OTilt to what's stored. Event can change on
official-scorer rulings; RTilt/OTilt are derived from spin/movement, so if
Statcast reprocessed the movement the tilts shift too.
"""
import os, sys, pickle, time, warnings
from collections import defaultdict
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
import gspread
import scrapers.backfill_supplement as B
import scrapers.Pitcher2026 as Pitcher2026

COLS = ['Description', 'Event', 'BBType', 'RTilt', 'OTilt']


def main():
    meta = pickle.load(open(os.path.join(ROOT, 'scripts', '_feed_meta_cache.pkl'), 'rb'))
    dated = sorted(((d, pk) for pk, (d, _, _) in meta.items() if d))
    pick = [dated[int(i * (len(dated) - 1) / 15)][1] for i in range(16)]
    pick = sorted(set(pick))
    print(f"sample games: {pick}", flush=True)

    dl = Pitcher2026.BaseballSavantFocusedDownloader()
    cur = {}   # PitchID -> {col: value}
    for pk in pick:
        try:
            df = dl.download_game_data(pk)
        except Exception as e:
            print(f"  {pk}: scrape failed {e!r}"); continue
        for _, r in df.iterrows():
            cur[str(r['PitchID'])] = {c: ('' if r.get(c) is None else str(r.get(c))) for c in COLS}
        print(f"  {pk}: {len(df)} pitches", flush=True)
    print(f"re-scraped pitches: {len(cur)}", flush=True)

    sample = set(pick)
    stats = {c: {'n': 0, 'chg': 0, 'ex': []} for c in COLS}
    gc = gspread.service_account()
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for ws in sh.worksheets():
            t = ws.title.upper()
            if t not in B.ALL_TRACKED_TEAMS or t in ('ROC', 'AAA', 'FCL'):
                continue
            time.sleep(0.4)
            vals = ws.get_all_values()
            if not vals or 'PitchID' not in vals[0]:
                continue
            ci = {n: j for j, n in enumerate(vals[0]) if n}
            pc = ci['PitchID']
            for r in vals[1:]:
                pid = r[pc] if pc < len(r) else ''
                g = pid.split('_')[0]
                if not g.isdigit() or int(g) not in sample or pid not in cur:
                    continue
                for c in COLS:
                    if c not in ci:
                        continue
                    stored = str(r[ci[c]]) if ci[c] < len(r) else ''
                    now = cur[pid].get(c, '')
                    # normalize floats like '1.0'/'nan'
                    if now in ('nan', 'None'):
                        now = ''
                    s = stats[c]
                    s['n'] += 1
                    if stored.strip() != now.strip():
                        s['chg'] += 1
                        if len(s['ex']) < 8:
                            s['ex'].append(f"{pid}: {stored!r}->{now!r}")
    print(f"\n{'column':12s} {'compared':>9s} {'changed':>8s} {'%chg':>6s}")
    for c in COLS:
        s = stats[c]
        pct = 100 * s['chg'] / s['n'] if s['n'] else 0
        print(f"{c:12s} {s['n']:9d} {s['chg']:8d} {pct:5.1f}%")
    for c in COLS:
        if stats[c]['ex']:
            print(f"\n{c} changes: {stats[c]['ex']}")


if __name__ == '__main__':
    main()
