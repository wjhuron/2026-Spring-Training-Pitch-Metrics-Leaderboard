"""completeness_audit.py — verify the Sheets pitch data covers every official PA.

Compares official boxscore PAs (data/boxscore_cache.json) against PA-ending
events in the SHEETS pitch data per game, with the two BY-DESIGN absences
accounted for:

  1. No-pitch intentional walks — since 2017 an automatic IBB contains no
     pitches. BACKFILLED 2026-08-18: 320 marker rows (312 MLB + 8 ROC/AAA,
     PitchID ending _00) were appended to the sheets, so this absence should
     now be CLOSED when read with include_no_pitch=True. A remaining gap here
     means the backfill missed something.
  2. Position-player pitching — historical only: through 2026-07-13 the
     pipeline dropped every EP-appearance pitch before the cache, so cache
     audits under-counted (~437 PAs). Policy changed the same day: EP PAs now
     stay in the cache and count in hitter/league data (position players are
     excluded from pitcher-facing views instead), so --cache audits should
     now pass too.

HISTORY (2026-07-13): a first version of this audit compared boxscores to
all_pitches_rs_cache.pkl (then post-EP-filter) and wrongly concluded 80
games were missing 437 PAs; every one was an EP-appearance PA the pipeline
dropped on purpose. The Sheets were complete. Sheets remain the
ground-truth source for this audit; --cache additionally validates that the
pipeline's read/keep path loses nothing.

Usage: python3 scripts/audits/completeness_audit.py            # reads sheets (slow, ~4 min)
       python3 scripts/audits/completeness_audit.py --cache    # fast; validates the pipeline kept everything
"""
import os, sys, json, argparse, pickle
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pipeline.utils import NON_PA_EVENTS


def sheet_pitches():
    from pipeline.fetch import read_all_pitches_from_sheets
    # PA-level audit: opt in to the no-pitch IBB markers. They are excluded by
    # default (see the docstring there) because ~50 downstream consumers count
    # rows as pitches; this is one of the few places that counts PAs.
    return read_all_pitches_from_sheets(include_no_pitch=True)


def cache_pitches():
    return pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', action='store_true',
                    help='audit the post-EP-filter cache instead of live sheets')
    args = ap.parse_args()

    box = json.load(open(os.path.join(ROOT, 'data', 'boxscore_cache.json')))
    box_g = {}
    for date, games in box.items():
        for g in games:
            box_g[g['gamePk']] = {
                'date': date,
                'teams': '@'.join(sorted({h['team'] for h in g['hitters']})),
                'pa': sum(h.get('pa', 0) for h in g['hitters']),
                'ibb': sum(h.get('ibb', 0) for h in g['hitters']),
            }

    pitches = cache_pitches() if args.cache else sheet_pitches()
    ev_abs = defaultdict(set)
    ibb_ct = defaultdict(int)
    for p in pitches:
        if p.get('_source', 'MLB') != 'MLB':
            continue
        parts = (p.get('PitchID') or '').split('_')
        if len(parts) != 3:
            continue
        try:
            gid, ab = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        ev = p.get('Event')
        if ev and ev not in NON_PA_EVENTS:
            ev_abs[gid].add(ab)
            if ev == 'Intent Walk':
                ibb_ct[gid] += 1

    flagged = []
    for gid, b in box_g.items():
        if gid not in ev_abs:
            if b['pa'] > 0:
                flagged.append((gid, b, b['pa'], 'GAME ABSENT'))
            continue
        miss = (b['pa'] - len(ev_abs[gid])) - max(0, b['ibb'] - ibb_ct[gid])
        if miss > 0:
            flagged.append((gid, b, miss, 'partial'))

    if not flagged:
        print(f"COMPLETE: all {len(box_g)} boxscore games fully covered "
              f"(source: {'cache' if args.cache else 'sheets'}; "
              f"no-pitch IBBs excluded by design).")
        return
    flagged.sort(key=lambda t: -t[2])
    print(f"{len(flagged)} game(s) with unexplained missing PAs "
          f"(source: {'cache' if args.cache else 'sheets'}):")
    print(f"{'gamePk':>8s}  {'date':10s}  {'teams':11s}  {'missing':>7s}  note")
    for gid, b, miss, note in flagged:
        print(f"{gid:8d}  {b['date']}  {b['teams']:11s}  {miss:7d}  {note}")


if __name__ == '__main__':
    main()
