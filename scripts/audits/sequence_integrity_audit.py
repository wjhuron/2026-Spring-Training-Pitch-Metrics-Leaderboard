"""sequence_integrity_audit.py — sheets-wide detector for feed SEQUENCE revisions.

The pitch-count audit only catches PAs where the pitch TOTAL changed. This catches
the same-count cousins: the feed re-sequenced a PA after the scrape (leadoff pitch
swapped, two pitches transposed, a mid-PA reorder), so a pitch's velo/outcome no
longer matches the feed at its position — which still corrupts per-pitch website data.

For every MLB pitch it aligns the sheet row to Savant (auto-ball-aware numbering)
and, per position, compares velocity and outcome category. A PA is flagged if any
position disagrees. Read-only. Writes ~/Downloads/sequence_revisions_2026.csv.

Usage: python3 scripts/audits/sequence_integrity_audit.py
"""
import os, sys, csv, time, pickle
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import gspread
import scrapers.backfill_supplement as B

VELO_THRESH = 1.5


def sf(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def cat_sheet(d):
    d = (d or '').lower()
    if 'in play' in d:
        return 'inplay'
    if 'hit by pitch' in d:
        return 'hbp'
    if 'foul tip' in d or 'missed bunt' in d:
        return 'strike'
    if 'foul' in d:  # foul, foul bunt
        return 'foul'
    if 'strike' in d:  # called/swinging (+ blocked)
        return 'strike'
    if 'ball' in d or 'pitchout' in d:
        return 'ball'
    return '?'


def cat_savant(d):
    d = (d or '').lower()
    if 'automatic' in d:   # MUST be first: automatic_ball/strike are not pitches
        return 'auto'
    if 'hit_into_play' in d:
        return 'inplay'
    if 'hit_by_pitch' in d:
        return 'hbp'
    if 'foul_tip' in d or 'missed_bunt' in d:
        return 'strike'
    if 'foul' in d:  # foul, foul_bunt
        return 'foul'
    if 'strike' in d:  # called_strike, swinging_strike(_blocked)
        return 'strike'
    if 'ball' in d or 'pitchout' in d:  # ball, blocked_ball, pitchout
        return 'ball'
    return '?'


def main():
    sc = pickle.load(open(os.path.join(ROOT, 'data', '_statcast2026_full.pkl'), 'rb'))
    bypa = defaultdict(list)
    for r in sc.itertuples(index=False):
        try:
            bypa[(int(r.game_pk), int(r.at_bat_number))].append((int(r.pitch_number), r))
        except Exception:
            continue
    aligned = {}   # (pk, ab, feednum) -> (velo, cat)
    for (pk, ab), evs in bypa.items():
        evs.sort(key=lambda t: t[0])
        feed = 0
        for pn, r in evs:
            if cat_savant(r.description) == 'auto':
                continue
            feed += 1
            aligned[(pk, ab, feed)] = (sf(r.release_speed), cat_savant(r.description))
    print(f"aligned savant pitches: {len(aligned)}", flush=True)

    gc = gspread.service_account()
    # sheet pitches per PA
    CACHE = os.path.join(ROOT, 'data', '_seq_sheet_pa.pkl')
    sheet_pa = defaultdict(dict)   # (pk,ab) -> {feednum: (velo, cat, pid, pitcher)}
    if os.path.exists(CACHE) and '--reread' not in sys.argv:
        sheet_pa = pickle.load(open(CACHE, 'rb'))
        print(f"loaded sheet PA cache ({len(sheet_pa)} PAs)", flush=True)
    else:
        for label, sid in B.SPREADSHEET_IDS.items():
            sh = gc.open_by_key(sid)
            for ws in sh.worksheets():
                t = ws.title.upper()
                if t not in B.ALL_TRACKED_TEAMS or t in ('ROC', 'AAA', 'FCL'):
                    continue
                time.sleep(0.7)
                vals = ws.get_all_values()
                if not vals or len(vals) < 2 or 'PitchID' not in vals[0]:
                    continue
                ci = {n: j for j, n in enumerate(vals[0]) if n}
                for r in vals[1:]:
                    pid = str(r[ci['PitchID']]) if ci['PitchID'] < len(r) else ''
                    pa = pid.split('_')
                    if len(pa) != 3:
                        continue
                    try:
                        pk, ab, pn = int(pa[0]), int(pa[1]), int(pa[2])
                    except ValueError:
                        continue
                    sheet_pa[(pk, ab)][pn] = (
                        sf(r[ci['Velocity']]) if 'Velocity' in ci else None,
                        cat_sheet(r[ci['Description']]) if 'Description' in ci else '?',
                        pid, r[ci['Pitcher']] if 'Pitcher' in ci else '')
                print(f"  [{label}/{ws.title}] read", flush=True)
        pickle.dump(dict(sheet_pa), open(CACHE, 'wb'))

    rows = []
    for (pk, ab), pitches in sheet_pa.items():
        n = len(pitches)
        sav_n = sum(1 for k in aligned if k[0] == pk and k[1] == ab)
        if sav_n == 0:
            continue  # no Savant coverage (skip; not a sequence issue)
        for pn, (sv_velo, sv_cat, pid, pitcher) in pitches.items():
            av = aligned.get((pk, ab, pn))
            if av is None:
                continue
            a_velo, a_cat = av
            velo_bad = (sv_velo is not None and a_velo is not None and abs(sv_velo - a_velo) > VELO_THRESH)
            cat_bad = (sv_cat != '?' and a_cat != '?' and sv_cat != a_cat)
            if velo_bad or cat_bad:
                kinds = []
                if velo_bad:
                    kinds.append(f"velo {sv_velo}->{a_velo}")
                if cat_bad:
                    kinds.append(f"outcome {sv_cat}->{a_cat}")
                rows.append({
                    'PitchID': pid, 'Pitcher': pitcher,
                    'sheet_count': n, 'feed_count': sav_n,
                    'issue': '; '.join(kinds),
                })

    # group by PA for reporting
    bypa_rows = defaultdict(list)
    for r in rows:
        bypa_rows['_'.join(r['PitchID'].split('_')[:2])].append(r)
    out = os.path.expanduser('~/Downloads/sequence_revisions_2026.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['PitchID', 'Pitcher', 'sheet_count', 'feed_count', 'issue'])
        w.writeheader()
        for pa in sorted(bypa_rows):
            for r in sorted(bypa_rows[pa], key=lambda x: x['PitchID']):
                w.writerow(r)

    print(f"\nwrote {out}")
    print(f"\n=== SEQUENCE-REVISION PAs: {len(bypa_rows)} ({len(rows)} pitches) ===")
    for pa in sorted(bypa_rows):
        rs = bypa_rows[pa]
        cm = '' if rs[0]['sheet_count'] == rs[0]['feed_count'] else f" [COUNT {rs[0]['sheet_count']}!={rs[0]['feed_count']}]"
        print(f"  {pa}  {rs[0]['Pitcher']:22s} {len(rs)} pitch(es){cm}: " +
              " | ".join(r['issue'] for r in rs[:4]))


if __name__ == '__main__':
    main()
