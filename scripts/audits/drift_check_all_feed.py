"""drift_check_all_feed.py — systematic drift check for every remaining
feed-derived tracking column. READ-ONLY.

Re-scrapes a date/park-spread sample with the current Pitcher2026 and diffs each
numeric tracking column (stored vs fresh). Reports per-column drift AND per-game
drift, so a park/date-localized glitch (the Progressive Field / PlateZ signature
= one game lighting up while the rest are flat) stands out.
"""
import os, sys, pickle, warnings
from collections import defaultdict
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
import gspread
import scrapers.backfill_supplement as B
import scrapers.Pitcher2026 as Pitcher2026

# sheet col -> tolerance (a "change" is |stored-fresh| > tol)
COLS = {'Velocity': 0.3, 'Spin Rate': 25, 'IndVertBrk': 0.5, 'HorzBrk': 0.5,
        'RelPosZ': 0.1, 'RelPosX': 0.1, 'Extension': 0.1, 'SzTop': 0.1, 'SzBot': 0.1,
        'VAA': 0.3, 'HAA': 0.3, 'PlateX': 0.15, 'PlateZ': 0.15,
        'ExitVelo': 0.5, 'LaunchAngle': 1.5, 'Distance': 5, 'HC_X': 2, 'HC_Y': 2}


def sf(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def main():
    meta = pickle.load(open(os.path.join(ROOT, 'scripts', '_feed_meta_cache.pkl'), 'rb'))
    dated = sorted(((d, pk) for pk, (d, _, _) in meta.items() if d))
    N = 30
    pick = sorted(set(dated[int(i * (len(dated) - 1) / (N - 1))][1] for i in range(N)))
    venue = {pk: vn for pk, (d, vid, vn) in meta.items()}
    print(f"sample games: {len(pick)}", flush=True)

    dl = Pitcher2026.BaseballSavantFocusedDownloader()
    cur = {}
    for pk in pick:
        try:
            df = dl.download_game_data(pk)
        except Exception as e:
            print(f"  {pk}: scrape failed {e!r}"); continue
        for _, r in df.iterrows():
            cur[str(r['PitchID'])] = {c: sf(r.get(c)) for c in COLS}
        print(f"  {pk} [{venue.get(pk,'?')[:18]}]: {len(df)}", flush=True)
    print(f"re-scraped pitches: {len(cur)}", flush=True)

    sample = set(pick)
    overall = {c: [0, 0, 0.0] for c in COLS}      # n, changed, absum
    bygame = defaultdict(lambda: defaultdict(int))  # game -> col -> changed
    gc = gspread.service_account()
    for label, sid in B.SPREADSHEET_IDS.items():
        sh = gc.open_by_key(sid)
        for ws in sh.worksheets():
            t = ws.title.upper()
            if t not in B.ALL_TRACKED_TEAMS or t in ('ROC', 'AAA', 'FCL'):
                continue
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
                for c, tol in COLS.items():
                    if c not in ci:
                        continue
                    stored = sf(r[ci[c]]) if ci[c] < len(r) else None
                    now = cur[pid].get(c)
                    if stored is None or now is None:
                        continue
                    overall[c][0] += 1
                    if abs(stored - now) > tol:
                        overall[c][1] += 1; overall[c][2] += abs(stored - now)
                        bygame[int(g)][c] += 1

    print("\n=== overall drift per column ===")
    for c in COLS:
        n, ch, s = overall[c]
        print(f"  {c:14s} n={n:6d}  changed={ch:5d} ({100*ch/n if n else 0:4.1f}%)  mean|chg|={s/ch if ch else 0:.2f}")
    print("\n=== games with notable drift (potential park/date glitch) ===")
    flagged = False
    for g in sorted(bygame, key=lambda x: -sum(bygame[x].values())):
        tot = sum(bygame[g].values())
        if tot < 15:
            continue
        flagged = True
        cols = ", ".join(f"{c}={n}" for c, n in sorted(bygame[g].items(), key=lambda x: -x[1]) if n >= 3)
        print(f"  {g} [{venue.get(g,'?')}]: {tot} drifted cells  ({cols})")
    if not flagged:
        print("  none — no game drifts more than noise.")


if __name__ == '__main__':
    main()
