"""pitch_count_audit.py — find every PA where your data's pitch COUNT differs
from the current official feed (missing or extra pitches from feed revisions).

Read-only. For every plate appearance in every game you have (MLB + ROC), compare
the number of pitches in the pickle against the current MLB Stats API play-by-play.
A difference means MLB added/removed a pitch after you scraped (a missing pitch if
feed > pickle, extra if pickle > feed). Writes a review CSV.

Usage: python3 scripts/audits/pitch_count_audit.py
"""
import os, csv, pickle, requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PBP = "https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay"


def fetch(pk):
    try:
        return pk, requests.get(PBP.format(pk=pk), timeout=30).json()
    except Exception:
        return pk, None


def main():
    allp = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    # pickle pitch count per (gamePk, atbat)
    pcount = defaultdict(int)
    ginfo = {}
    for p in allp:
        pid = p.get('PitchID')
        if not pid:
            continue
        parts = pid.split('_')
        pk, ab = parts[0], parts[1]
        pcount[(pk, ab)] += 1
        g = ginfo.setdefault(pk, {'date': p.get('Game Date'), 'teams': set(),
                                  'src': 'MLB' if p.get('_source') == 'MLB' else 'ROC'})
        if p.get('PTeam'):
            g['teams'].add(p['PTeam'])
    all_pks = sorted(ginfo)
    print(f"games: {len(all_pks)}; fetching play-by-play ...", flush=True)

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
        g = ginfo[pk]
        for play in js.get('allPlays', []):
            ab = f"{play.get('atBatIndex', 0) + 1:03d}"
            fn = sum(1 for e in play.get('playEvents', []) if e.get('isPitch'))
            pn = pcount.get((pk, ab), 0)
            if fn == 0 or pn == fn:
                continue
            rows.append({
                'game_pa': f"{pk}_{ab}", 'league': g['src'], 'date': g.get('date'),
                'game': ' vs '.join(sorted(g.get('teams', []))) or pk,
                'inning': play.get('about', {}).get('inning'),
                'batter': play.get('matchup', {}).get('batter', {}).get('fullName'),
                'pitcher': play.get('matchup', {}).get('pitcher', {}).get('fullName'),
                'result': play.get('result', {}).get('event'),
                'your_pitches': pn, 'feed_pitches': fn, 'diff': fn - pn,
                'kind': 'MISSING' if fn > pn else 'EXTRA',
            })

    rows.sort(key=lambda r: (r['kind'], r['date'] or '', r['game_pa']))
    out = os.path.expanduser('~/Downloads/pitch_count_audit_2026.csv')
    cols = ['game_pa', 'league', 'date', 'game', 'inning', 'batter', 'pitcher', 'result',
            'your_pitches', 'feed_pitches', 'diff', 'kind']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)

    miss = [r for r in rows if r['kind'] == 'MISSING']
    extra = [r for r in rows if r['kind'] == 'EXTRA']
    from collections import Counter
    print(f"\n=== PITCH-COUNT AUDIT COMPLETE ===")
    print(f"PAs with a pitch-count mismatch: {len(rows)}")
    print(f"  MISSING pitches (feed has more): {len(miss)} PAs, {sum(r['diff'] for r in miss)} pitches")
    print(f"  EXTRA pitches (you have more):   {len(extra)} PAs, {sum(-r['diff'] for r in extra)} pitches")
    print(f"  by league: {dict(Counter(r['league'] for r in rows))}")
    print(f"\nwrote {out}\n")
    for r in (miss + extra)[:40]:
        print(f"  {r['game_pa']} {r['league']:3s} {(r['date'] or ''):10s} you={r['your_pitches']} feed={r['feed_pitches']} "
              f"[{r['kind']}] {r['batter']} vs {r['pitcher']} ({r['result']})")
    if len(rows) > 40:
        print(f"  ... and {len(rows) - 40} more (see CSV)")


if __name__ == '__main__':
    main()
