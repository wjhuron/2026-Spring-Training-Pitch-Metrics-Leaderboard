"""midpa_sweep.py — full-season sweep for mid-PA substitution mislabels (MLB + ROC).

Read-only. Derives the game list from the pickle (exactly the games in your data),
fetches each game's MLB Stats API play-by-play in parallel, reconstructs the true
pitcher/batter/handedness per pitch, and diffs against the pickle. Writes a review
CSV sorted actionable-first. Nothing is written back to the Sheets or pipeline.

Usage: python3 scripts/midpa_sweep.py
"""
import sys, os, csv, pickle, requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import midpa_review as MR

PBP = "https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay"
PEOPLE = "https://statsapi.mlb.com/api/v1/people/{pid}"
STATUS_ORDER = {'MISLABELED': 0, 'REVIEW': 1, 'PITCH_NOT_IN_DATA': 2, 'ALREADY_FIXED': 3}


def fetch_pbp(pk):
    try:
        return pk, requests.get(PBP.format(pk=pk), timeout=30).json()
    except Exception:
        return pk, None


def fetch_person(pid):
    try:
        d = requests.get(PEOPLE.format(pid=pid), timeout=20).json()['people'][0]
        return pid, {'name': d.get('fullName'), 'bats': (d.get('batSide') or {}).get('code'),
                     'throws': (d.get('pitchHand') or {}).get('code')}
    except Exception:
        return pid, {'name': None, 'bats': None, 'throws': None}


def main():
    allp = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    idx = {p['PitchID']: p for p in allp if p.get('PitchID')}
    mlb_pks = sorted(set(p['PitchID'].split('_')[0] for p in allp
                         if p.get('_source') == 'MLB' and p.get('PitchID')))
    roc_pks = sorted(set(p['PitchID'].split('_')[0] for p in allp
                         if p.get('_source') in ('ROC', 'AAA') and p.get('PitchID')))
    ginfo = {}
    for p in allp:
        pid = p.get('PitchID')
        if not pid:
            continue
        pk = pid.split('_')[0]
        g = ginfo.setdefault(pk, {'date': p.get('Game Date'), 'teams': set()})
        if p.get('PTeam'):
            g['teams'].add(p['PTeam'])
    all_pks = mlb_pks + roc_pks
    print(f"games: {len(mlb_pks)} MLB + {len(roc_pks)} ROC = {len(all_pks)}; fetching play-by-play ...", flush=True)

    pbps = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        for i, (pk, js) in enumerate(ex.map(fetch_pbp, all_pks)):
            pbps[pk] = js
            if (i + 1) % 200 == 0:
                print(f"  fetched {i + 1}/{len(all_pks)}", flush=True)

    # pass 1: collect affected pitches (attribution differs from the finisher)
    affected = []
    for pk in all_pks:
        js = pbps.get(pk)
        if not js:
            continue
        src = 'MLB' if pk in set(mlb_pks) else 'ROC'
        for play in js.get('allPlays', []):
            events = play.get('playEvents', [])
            if not any(MR.is_pitching_sub(e) or MR.is_pinch_hitter(e) for e in events):
                continue
            recon = MR.reconstruct(play)
            ab = play.get('atBatIndex', 0) + 1
            fin_p = play.get('matchup', {}).get('pitcher', {}).get('id')
            fin_b = play.get('matchup', {}).get('batter', {}).get('id')
            for pn, pp, bb in recon:
                if pp.get('id') == fin_p and bb.get('id') == fin_b:
                    continue
                affected.append({'pk': pk, 'src': src, 'ab': ab, 'pn': pn, 'pp': pp, 'bb': bb,
                                 'fin_p': fin_p, 'fin_b': fin_b,
                                 'inn': play.get('about', {}).get('inning'),
                                 'half': (play.get('about', {}).get('halfInning', '') or '')[:3]})
    print(f"affected pitches (mid-PA): {len(affected)}; fetching handedness ...", flush=True)

    need = set()
    for a in affected:
        need |= {a['pp']['id'], a['bb']['id'], a['fin_p'], a['fin_b']}
    need.discard(None)
    with ThreadPoolExecutor(max_workers=12) as ex:
        for pid, h in ex.map(fetch_person, need):
            MR._hand[pid] = h

    # pass 2: build review rows
    rows = []
    for a in affected:
        pp, bb = a['pp'], a['bb']
        thr = MR.hand(pp['id'])['throws']
        cor_pitcher = MR.hand(pp['id'])['name']
        cor_batter = MR.hand(bb['id'])['name']
        cor_bats = MR.bats_of(bb['id'], thr)
        pid = f"{a['pk']}_{a['ab']:03d}_{(a['pn'] or 0):02d}"
        cur = idx.get(pid)
        pitcher_changed = pp.get('id') != a['fin_p']
        g = ginfo.get(a['pk'], {})
        game = ' vs '.join(sorted(g.get('teams', []))) or a['pk']
        if cur is None:
            status = 'PITCH_NOT_IN_DATA'
            cur_pitcher = cur_batter = cur_bats = cur_throws = ''
        else:
            cur_pitcher = cur.get('Pitcher'); cur_batter = cur.get('Batter')
            cur_bats = cur.get('Bats'); cur_throws = cur.get('Throws')
            if pitcher_changed:
                status = MR.classify(cur_pitcher, cor_pitcher, MR.hand(a['fin_p'])['name'],
                                     cur_throws, thr, MR.hand(a['fin_p'])['throws'])
            else:
                status = MR.classify(cur_batter, cor_batter, MR.hand(a['fin_b'])['name'],
                                     cur_bats, cor_bats, MR.bats_of(a['fin_b'], MR.hand(a['fin_p'])['throws']))
        rows.append({
            'PitchID': pid, 'league': a['src'], 'date': g.get('date'), 'game': game,
            'inn': a['inn'], 'half': a['half'], 'pitchNo': a['pn'],
            'change': 'PITCHER' if pitcher_changed else 'BATTER', 'status': status,
            'cur_Pitcher': cur_pitcher, 'cor_Pitcher': cor_pitcher,
            'cur_Throws': cur_throws, 'cor_Throws': thr,
            'cur_Batter': cur_batter, 'cor_Batter': cor_batter,
            'cur_Bats': cur_bats, 'cor_Bats': cor_bats,
        })

    rows.sort(key=lambda r: (STATUS_ORDER.get(r['status'], 9), r['date'] or '', r['PitchID']))
    out = os.path.expanduser('~/Downloads/midPA_review_2026.csv')
    cols = ['PitchID', 'league', 'date', 'game', 'inn', 'half', 'pitchNo', 'change', 'status',
            'cur_Pitcher', 'cor_Pitcher', 'cur_Throws', 'cor_Throws',
            'cur_Batter', 'cor_Batter', 'cur_Bats', 'cor_Bats']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)

    counts = defaultdict(int)
    ab_set = defaultdict(set)
    for r in rows:
        counts[r['status']] += 1
        ab_set[r['status']].add('_'.join(r['PitchID'].split('_')[:2]))
    print(f"\n=== SWEEP COMPLETE ===")
    print(f"affected pitches: {len(rows)} across {len(set('_'.join(r['PitchID'].split('_')[:2]) for r in rows))} plate appearances")
    for st in ('MISLABELED', 'REVIEW', 'PITCH_NOT_IN_DATA', 'ALREADY_FIXED'):
        print(f"  {st:18s} {counts[st]:4d} pitches / {len(ab_set[st]):3d} PAs")
    print(f"\nwrote {out}")
    actionable = [r for r in rows if r['status'] in ('MISLABELED', 'REVIEW', 'PITCH_NOT_IN_DATA')]
    print(f"\n=== ACTIONABLE ({len(actionable)} pitches you likely missed) ===")
    for r in actionable[:60]:
        if r['change'] == 'PITCHER':
            chg = f"P: {r['cur_Pitcher']}({r['cur_Throws']}) -> {r['cor_Pitcher']}({r['cor_Throws']})"
        else:
            chg = f"B: {r['cur_Batter']}({r['cur_Bats']}) -> {r['cor_Batter']}({r['cor_Bats']})"
        print(f"  {r['PitchID']:16s} {r['league']:3s} {r['status'][:5]:5s} {(r['date'] or ''):10s} {r['game']:14s} {chg}")


if __name__ == '__main__':
    main()
