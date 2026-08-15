"""augment_priors_spinaxis.py — add SpinAxis to the 2021-24 training pickles.

The AXIS feature (OTilt-RTilt deviation) needs spin_axis on the prior rows;
the pickles were built without it. Rebuilding from scratch would lose the
weather-adjusted movement, so instead: re-run build_season_dicts with a
SpinAxis passthrough (in-memory source patch), VERIFY row-for-row alignment
with the existing pickle (length + sampled Velocity/Pitcher/Pitch Type
equality), then inject SpinAxis positionally into the existing rows.

2025's pickle comes from the sheets-statcast join (different builder, no
positional alignment available) — its rows keep SpinAxis=None; AXIS trains
on 2021-24 (2.8M rows) + 2026 (RTilt on every sheet row). OTilt needs no
augmentation anywhere: it derives from raw movement already in the rows.

Usage: python3 scripts/builders/augment_priors_spinaxis.py
"""
import os, sys, pickle, random

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

SRC_PATH = os.path.join(ROOT, 'scripts', 'build_historical_training_set.py')
src = open(SRC_PATH).read()
NEEDLE = "'_game_pk': int(r.game_pk) if sf(r.game_pk) is not None else None,"
assert NEEDLE in src, 'anchor not found in build_historical_training_set.py'
src = src.replace(NEEDLE, NEEDLE + "\n            'SpinAxis': sf(r.spin_axis),")
H = {'__name__': '_hist_mod', '__file__': SRC_PATH}
exec(compile(src, SRC_PATH, 'exec'), H)

for year in (2021, 2022, 2023, 2024):
    path = os.path.join(ROOT, 'data', f'_pitches{year}_training.pkl')
    existing = pickle.load(open(path, 'rb'))
    fresh = H['build_season_dicts'](year)
    assert len(existing) == len(fresh), \
        f'{year}: length mismatch {len(existing)} vs {len(fresh)} — builder drifted, aborting'
    def _eq(a, b):
        if a != a and b != b:      # both NaN
            return True
        return a == b
    rnd = random.Random(11)
    for i in rnd.sample(range(len(existing)), 500):
        for k in ('Velocity', 'Pitcher', 'Pitch Type', 'Game Date'):
            assert _eq(existing[i].get(k), fresh[i].get(k)), \
                f'{year} row {i}: {k} mismatch ({existing[i].get(k)!r} vs {fresh[i].get(k)!r})'
    n_sa = 0
    for e, f in zip(existing, fresh):
        e['SpinAxis'] = f.get('SpinAxis')
        if e['SpinAxis'] is not None:
            n_sa += 1
    pickle.dump(existing, open(path, 'wb'))
    print(f'{year}: verified alignment (500 samples), SpinAxis on '
          f'{n_sa}/{len(existing)} rows ({n_sa/len(existing)*100:.1f}%)', flush=True)
print('done.')
