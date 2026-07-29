"""midpa_review.py — find pitches mislabeled by mid-PA substitutions.

Ground truth = MLB Stats API play-by-play (the SAME feed Pitcher2026 builds from,
so PitchID = gamePk_atBat_pitch joins 1:1). For each play, reconstruct the true
pitcher/batter for every pitch by walking the substitution events in order, then
diff against the pickle (mirror of the Sheets). Also corrects handedness: outgoing
pitcher's throw hand, outgoing batter's bat side (switch hitters -> opposite the
pitcher's hand). Pinch-runners and defensive subs are ignored (batter at the plate
unchanged). Writes a review CSV.

Usage: python3 scripts/midpa_review.py <startDate> <endDate>   (default a sample window)
"""
import sys, os, csv, pickle
import requests
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = requests.Session()
SCHED = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={a}&endDate={b}"
PBP = "https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay"
PEOPLE = "https://statsapi.mlb.com/api/v1/people/{pid}"

_hand = {}
def hand(pid):
    if pid not in _hand:
        try:
            d = S.get(PEOPLE.format(pid=pid), timeout=15).json()['people'][0]
            _hand[pid] = {'name': d.get('fullName'), 'bats': (d.get('batSide') or {}).get('code'),
                          'throws': (d.get('pitchHand') or {}).get('code')}
        except Exception:
            _hand[pid] = {'name': None, 'bats': None, 'throws': None}
    return _hand[pid]


def _et(e):
    return (e.get('details', {}) or {}).get('eventType', '') or ''
def is_pitching_sub(e):
    return e.get('type') == 'action' and _et(e) == 'pitching_substitution'
def is_pinch_hitter(e):
    return (e.get('type') == 'action' and _et(e) == 'offensive_substitution'
            and 'pinch-hitter' in (e.get('details', {}).get('description', '') or '').lower())


def reconstruct(play):
    """Return list of (pitchNumber, pitcher_player, batter_player) for each pitch."""
    events = play.get('playEvents', [])
    mu = play.get('matchup', {})
    fin_p, fin_b = mu.get('pitcher', {}), mu.get('batter', {})
    # starting pitcher/batter of the PA = outgoing of the FIRST relevant sub, else the finisher
    start_p, start_b = fin_p, fin_b
    for e in events:
        if is_pitching_sub(e):
            start_p = e.get('replacedPlayer') or start_p; break
    for e in events:
        if is_pinch_hitter(e):
            start_b = e.get('replacedPlayer') or start_b; break
    cur_p, cur_b = start_p, start_b
    out = []
    for e in events:
        if e.get('type') == 'action':
            if is_pitching_sub(e):
                cur_p = e.get('player') or cur_p
            elif is_pinch_hitter(e):
                cur_b = e.get('player') or cur_b
            continue
        if e.get('isPitch'):
            out.append((e.get('pitchNumber'), cur_p, cur_b))
    return out


def bats_of(batter_id, pitcher_throws):
    bs = hand(batter_id)['bats']
    if bs == 'S':
        return 'L' if pitcher_throws == 'R' else 'R'
    return bs


SUFFIX = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}
def surname(name):
    """Core surname from either 'First Last [Jr.]' (feed) or 'Last [Jr.], First'
    (pickle). Lowercased, suffix stripped, so the two formats compare equal."""
    if not name:
        return ''
    part = name.split(',')[0] if ',' in name else name  # 'Last Jr.' or 'First Last Jr.'
    toks = [t for t in part.replace('.', '').split() if t.lower() not in SUFFIX]
    if ',' in name:
        return ' '.join(toks).lower().strip()          # whole pre-comma chunk is the surname
    return (toks[-1] if toks else part).lower().strip()  # last token of 'First Last'


def classify(cur_name, out_name, fin_name, cur_hand, out_hand, fin_hand):
    """FIXED if the pickle already shows the outgoing player; MISLABELED if it
    still shows the finisher; REVIEW if names are ambiguous (fall back to hand)."""
    cs, os_, fs = surname(cur_name), surname(out_name), surname(fin_name)
    if cs and cs == os_:
        return 'ALREADY_FIXED'
    if cs and cs == fs:
        return 'MISLABELED'
    if cur_hand and out_hand and fin_hand and out_hand != fin_hand:
        return 'ALREADY_FIXED' if cur_hand == out_hand else 'MISLABELED'
    return 'REVIEW'


def main():
    a = sys.argv[1] if len(sys.argv) > 1 else '2026-06-18'
    b = sys.argv[2] if len(sys.argv) > 2 else '2026-06-30'
    print(f"window {a} .. {b}", flush=True)
    idx = {}
    for p in pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb')):
        if p.get('_source') == 'MLB' and p.get('PitchID'):
            idx[p['PitchID']] = p
    print(f"indexed {len(idx)} MLB pitches from pickle", flush=True)

    sched = S.get(SCHED.format(a=a, b=b), timeout=20).json()
    games = []
    for d in sched.get('dates', []):
        for g in d.get('games', []):
            if g.get('gameType') == 'R' and g.get('status', {}).get('abstractGameState') == 'Final':
                games.append((g['gamePk'], d['date'],
                              g['teams']['away']['team'].get('name', '?'),
                              g['teams']['home']['team'].get('name', '?')))
    print(f"{len(games)} final games", flush=True)

    rows = []
    for pk, date, away, home in games:
        try:
            pbp = S.get(PBP.format(pk=pk), timeout=20).json()
        except Exception:
            continue
        for play in pbp.get('allPlays', []):
            events = play.get('playEvents', [])
            if not any((is_pitching_sub(e) or is_pinch_hitter(e)) for e in events):
                continue
            has_pitch_after = False  # only mid-PA (a sub with pitches after it) matters
            seen_pitch = False; seen_sub = False
            for e in events:
                if e.get('isPitch'):
                    seen_pitch = True
                    if seen_sub:
                        has_pitch_after = True
                if is_pitching_sub(e) or is_pinch_hitter(e):
                    if seen_pitch:
                        seen_sub = True
            if not has_pitch_after:
                continue
            recon = reconstruct(play)
            ab = play.get('atBatIndex', 0) + 1
            fin_p = play.get('matchup', {}).get('pitcher', {}).get('id')
            fin_b = play.get('matchup', {}).get('batter', {}).get('id')
            for pn, pp, bb in recon:
                pitcher_changed = pp.get('id') != fin_p
                batter_changed = bb.get('id') != fin_b
                if not pitcher_changed and not batter_changed:
                    continue  # this pitch belongs to the finisher (post-sub) — not mislabeled
                pid = f"{pk}_{ab:03d}_{(pn or 0):02d}"
                cur = idx.get(pid)
                thr = hand(pp['id'])['throws']
                cor_pitcher = hand(pp['id'])['name']
                cor_batter = hand(bb['id'])['name']
                cor_bats = bats_of(bb['id'], thr)
                inn = play.get('about', {}).get('inning')
                half = play.get('about', {}).get('halfInning', '')[:3]
                if cur is None:
                    status = 'PITCH_NOT_IN_DATA'
                    cur_pitcher = cur_batter = cur_bats = cur_throws = ''
                else:
                    cur_pitcher = cur.get('Pitcher'); cur_batter = cur.get('Batter')
                    cur_bats = cur.get('Bats'); cur_throws = cur.get('Throws')
                    if pitcher_changed:
                        status = classify(cur_pitcher, cor_pitcher, hand(fin_p)['name'],
                                          cur_throws, thr, hand(fin_p)['throws'])
                    else:
                        status = classify(cur_batter, cor_batter, hand(fin_b)['name'],
                                          cur_bats, cor_bats, bats_of(fin_b, hand(fin_p)['throws']))
                rows.append({
                    'PitchID': pid, 'date': date, 'game': f"{away} @ {home}", 'inn': inn, 'half': half,
                    'pitchNo': pn, 'status': status, 'change': 'PITCHER' if pitcher_changed else 'BATTER',
                    'cur_Pitcher': cur_pitcher, 'cur_Throws': cur_throws,
                    'cor_Pitcher': cor_pitcher, 'cor_Throws': thr,
                    'cur_Batter': cur_batter, 'cur_Bats': cur_bats,
                    'cor_Batter': cor_batter, 'cor_Bats': cor_bats,
                })

    out = os.path.expanduser('~/Downloads/midPA_review_sample.csv')
    cols = ['PitchID', 'date', 'game', 'inn', 'half', 'pitchNo', 'change', 'status',
            'cur_Pitcher', 'cor_Pitcher', 'cur_Throws', 'cor_Throws',
            'cur_Batter', 'cor_Batter', 'cur_Bats', 'cor_Bats']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)
    counts = defaultdict(int)
    for r in rows:
        counts[r['status']] += 1
    print(f"\naffected pitches: {len(rows)}  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"wrote {out}\n")
    for r in rows:
        if r['change'] == 'PITCHER':
            chg = f"P: {r['cur_Pitcher']}({r['cur_Throws']}) -> {r['cor_Pitcher']}({r['cor_Throws']})"
        else:
            chg = f"B: {r['cur_Batter']}({r['cur_Bats']}) -> {r['cor_Batter']}({r['cor_Bats']})"
        print(f"{r['PitchID']:16s} {r['status']:14s} {r['game']:34s} inn{r['inn']} {r['half']}  {chg}")


if __name__ == '__main__':
    main()
