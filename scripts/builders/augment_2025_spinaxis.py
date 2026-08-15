"""augment_2025_spinaxis.py — attach SpinAxis to the 2025 training pickle.

The 2025 set was built by fingerprint-joining Wally's retagged sheets to
public Statcast (build_2025_training_set.join: per (date, pitcher), greedy
match on velo ±0.25 + plate coords, total distance <= 0.5). The joined rows
kept Velocity/PlateX/PlateZ/_game_pk, so the SAME match can be re-run to
carry one more field — made STRICTER here by requiring the candidate's
game_pk to equal the row's stored _game_pk.

Usage: python3 scripts/builders/augment_2025_spinaxis.py
"""
import os, sys, pickle
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from build_2025_training_set import sf

PKL = os.path.join(ROOT, 'data', '_pitches2025_training.pkl')
pitches = pickle.load(open(PKL, 'rb'))
df = pickle.load(open(os.path.join(ROOT, 'data', '_statcast2025_cache.pkl'), 'rb'))
if 'game_type' in df.columns:
    df = df[df['game_type'] == 'R']

pub = defaultdict(list)
for row in df.itertuples(index=False):
    pub[(str(row.game_date)[:10], row.player_name)].append(row)

groups = defaultdict(list)
for p in pitches:
    groups[(p.get('Game Date'), p.get('Pitcher'))].append(p)

matched = unmatched = 0
for key, mine in groups.items():
    cands = pub.get(key, [])
    used = [False] * len(cands)
    for p in mine:
        v, px, pz = p.get('Velocity'), p.get('PlateX'), p.get('PlateZ')
        gpk = p.get('_game_pk')
        if v is None:
            p.setdefault('SpinAxis', None); unmatched += 1
            continue
        best_i, best_d = None, 1e9
        for i, c in enumerate(cands):
            if used[i]:
                continue
            if gpk is not None and sf(c.game_pk) is not None and int(c.game_pk) != gpk:
                continue
            cv, cx, cz = sf(c.release_speed), sf(c.plate_x), sf(c.plate_z)
            if cv is None or abs(cv - v) > 0.25:
                continue
            d = abs(cv - v) * 2.0
            if px is not None and cx is not None:
                d += abs(cx - px)
            if pz is not None and cz is not None:
                d += abs(cz - pz)
            if d < best_d:
                best_d, best_i = d, i
        if best_i is None or best_d > 0.5:
            p.setdefault('SpinAxis', None); unmatched += 1
            continue
        used[best_i] = True
        p['SpinAxis'] = sf(cands[best_i].spin_axis)
        matched += 1

n_sa = sum(1 for p in pitches if p.get('SpinAxis') is not None)
print(f're-join: {matched} matched, {unmatched} unmatched '
      f'({matched / max(matched + unmatched, 1):.1%}); '
      f'SpinAxis on {n_sa}/{len(pitches)} rows ({n_sa/len(pitches)*100:.1f}%)')
pickle.dump(pitches, open(PKL, 'wb'))
print('saved.')
