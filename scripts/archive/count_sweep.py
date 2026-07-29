"""count_sweep.py — find pitches with the wrong ball-strike count (MLB + ROC).

Read-only. The feed's raw `count` field is occasionally corrupt (phantom balls,
strikes past 2), so the TRUE count is rebuilt by rule-based replay of the pitch
RESULTS plus automatic balls/strikes. A PA is a TRUE ANOMALY only if its results
are irreconcilable: some pitch has an impossible pre-count (>3 balls or >2
strikes), or the stated outcome does not match (e.g. a strikeout that never
reaches a third strike). Otherwise the replay is the truth. Categories:

  AUTO_BALL_STRIKE  pickle missed an automatic ball/strike -> count too low. Fix: count.
  FEED_COUNT        no auto, results agree, but the pickle inherited the feed's
                    corrupt count field. Fix: count -> replay value.
  OUTCOME_MISMATCH  pickle disagrees with the feed on a pitch RESULT (Ball vs
                    Called Strike ...), which shifts the count. Fix: Description + count.
  TRUE_ANOMALY      feed results irreconcilable; no reliable value. Manual review.

Writes ~/Downloads/count_review_2026.csv. Nothing is written back.

Usage: python3 scripts/count_sweep.py
"""
import os, csv, pickle, requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PBP = "https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay"


def fetch(pk):
    try:
        return pk, requests.get(PBP.format(pk=pk), timeout=30).json()
    except Exception:
        return pk, None


def feed_class(e):
    d = e.get('details', {}) or {}
    if d.get('isBall'):
        return 'ball'
    desc = (d.get('description', '') or '').lower()
    # A caught foul tip (and a foul bunt) is a strike that CAN be strike three;
    # only a plain foul is capped at two strikes.
    if 'foul tip' in desc or 'foul bunt' in desc:
        return 'strike'
    if desc == 'foul':
        return 'foul'
    if d.get('isStrike'):
        return 'strike'
    return 'end'   # in play / HBP


def wally_desc(feed_desc):
    d = (feed_desc or ''); dl = d.lower()
    if dl.startswith('in play'):
        return 'In Play'
    m = {'ball': 'Ball', 'ball in dirt': 'Ball', 'called strike': 'Called Strike',
         'swinging strike': 'Swinging Strike', 'swinging strike (blocked)': 'Swinging Strike',
         'foul': 'Foul', 'foul tip': 'Swinging Strike', 'foul bunt': 'Foul Bunt',
         'missed bunt': 'Missed Bunt', 'hit by pitch': 'Hit By Pitch', 'pitchout': 'Pitchout'}
    return m.get(dl, d)


def pickle_class(desc):
    if desc == 'Ball':
        return 'ball'
    if desc in ('Called Strike', 'Swinging Strike'):
        return 'strike'
    if desc == 'Foul':
        return 'foul'
    return 'end'


def main():
    allp = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    idx = {p['PitchID']: p for p in allp if p.get('PitchID')}
    mlb = set(p['PitchID'].split('_')[0] for p in allp if p.get('_source') == 'MLB' and p.get('PitchID'))
    roc = set(p['PitchID'].split('_')[0] for p in allp if p.get('_source') in ('ROC', 'AAA') and p.get('PitchID'))
    ginfo = {}
    for p in allp:
        pid = p.get('PitchID')
        if not pid:
            continue
        pk = pid.split('_')[0]
        g = ginfo.setdefault(pk, {'date': p.get('Game Date'), 'teams': set()})
        if p.get('PTeam'):
            g['teams'].add(p['PTeam'])
    all_pks = sorted(mlb) + sorted(roc)
    print(f"games: {len(mlb)} MLB + {len(roc)} ROC = {len(all_pks)}; fetching ...", flush=True)

    pbps = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        for i, (pk, js) in enumerate(ex.map(fetch, all_pks)):
            pbps[pk] = js
            if (i + 1) % 400 == 0:
                print(f"  fetched {i + 1}/{len(all_pks)}", flush=True)

    rows = []
    for pk in all_pks:
        js = pbps.get(pk)
        if not js:
            continue
        src = 'MLB' if pk in mlb else 'ROC'
        g = ginfo.get(pk, {})
        for play in js.get('allPlays', []):
            ab = play.get('atBatIndex', 0) + 1
            outcome = (play.get('result', {}).get('event', '') or '').lower()
            b = s = 0
            auto_seen = False
            anomaly = False
            recs = []          # (pn, pre_b, pre_s, auto_before, feed_cls, feed_desc)
            for e in play.get('playEvents', []):
                if e.get('isPitch'):
                    if b > 3 or s > 2:
                        anomaly = True
                    cls = feed_class(e)
                    recs.append((e.get('pitchNumber'), b, s, auto_seen, cls,
                                 (e.get('details', {}) or {}).get('description', '')))
                    if cls == 'ball':
                        b += 1
                    elif cls == 'strike':
                        s += 1
                    elif cls == 'foul':
                        s = s + 1 if s < 2 else s
                elif 'automatic' in (e.get('details', {}).get('description', '') or '').lower():
                    dsc = e.get('details', {}).get('description', '').lower()
                    if 'ball' in dsc:
                        b += 1
                    elif 'strike' in dsc:
                        s += 1
                    auto_seen = True
            # outcome reconciliation: strikeout must reach 3 strikes; walk must reach 4 balls
            if 'strikeout' in outcome and s < 3:
                anomaly = True
            if outcome in ('walk', 'intent_walk') and b < 4:
                anomaly = True
            # PA-level pitch-result mismatch between pickle and feed
            pa_mismatch = False
            for pn, _b, _s, _a, cls, _fd in recs:
                cur = idx.get(f"{pk}_{ab:03d}_{(pn or 0):02d}")
                if cur is not None and pickle_class(cur.get('Description')) != cls:
                    pa_mismatch = True
                    break
            for pn, pb, ps, auto_before, cls, fdesc in recs:
                pid = f"{pk}_{ab:03d}_{(pn or 0):02d}"
                cur = idx.get(pid)
                if cur is None:
                    continue
                true_ct = f"{pb}-{ps}"
                cd = wally_desc(fdesc)
                count_wrong = cur.get('Count') != true_ct
                desc_wrong = pa_mismatch and cur.get('Description') != cd
                if not count_wrong and not desc_wrong:
                    continue
                if anomaly:
                    cat = 'TRUE_ANOMALY'
                elif pa_mismatch:
                    cat = 'OUTCOME_MISMATCH'
                elif auto_before:
                    cat = 'AUTO_BALL_STRIKE'
                else:
                    cat = 'FEED_COUNT'
                rows.append({
                    'PitchID': pid, 'league': src, 'date': g.get('date'),
                    'game': ' vs '.join(sorted(g.get('teams', []))) or pk,
                    'inn': play.get('about', {}).get('inning'), 'category': cat,
                    'stored_count': cur.get('Count'), 'correct_count': ('' if anomaly else true_ct),
                    'stored_desc': cur.get('Description'),
                    'correct_desc': (cd if cat == 'OUTCOME_MISMATCH' else ''),
                    'Pitcher': cur.get('Pitcher'), 'Batter': cur.get('Batter'),
                })

    order = {'AUTO_BALL_STRIKE': 0, 'FEED_COUNT': 1, 'OUTCOME_MISMATCH': 2, 'TRUE_ANOMALY': 3}
    rows.sort(key=lambda r: (order.get(r['category'], 9), r['date'] or '', r['PitchID']))
    out = os.path.expanduser('~/Downloads/count_review_2026.csv')
    cols = ['PitchID', 'league', 'date', 'game', 'inn', 'category', 'stored_count', 'correct_count',
            'stored_desc', 'correct_desc', 'Pitcher', 'Batter']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)

    counts = defaultdict(int); pas = defaultdict(set)
    for r in rows:
        counts[r['category']] += 1
        pas[r['category']].add('_'.join(r['PitchID'].split('_')[:2]))
    print(f"\n=== COUNT SWEEP COMPLETE ===  {len(rows)} pitches flagged")
    for cat in ('AUTO_BALL_STRIKE', 'FEED_COUNT', 'OUTCOME_MISMATCH', 'TRUE_ANOMALY'):
        print(f"  {cat:18s} {counts[cat]:4d} pitches / {len(pas[cat]):3d} PAs")
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
