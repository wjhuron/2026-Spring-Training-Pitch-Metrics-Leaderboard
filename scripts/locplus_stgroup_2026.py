"""locplus_stgroup_2026.py — 2026 replicate for the ST-own-group candidate.

Completes the six-replicate picture for locplus_structure_multiseason.py's
st_group variant (rel 4/5, partial flat on 2021-2025). Nothing about the
grouping was fitted on 2026, so it is a legitimate replicate — and it is the
production season. Same objectives, run on data/all_pitches_rs_cache.pkl
(already pipeline-schema; MLB only).

Usage: python3 scripts/locplus_stgroup_2026.py
"""
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import locplus_structure_multiseason as S
import locplus_constants_multiseason as base
from pipeline_sdplus import make_rv_xrv

S.CONFIGS = ('shipped', 'st_group')


def main():
    D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    pitches = [p for p in D if p.get('_source', 'MLB') == 'MLB']
    del D
    n_st = sum(1 for x in pitches if x.get('Pitch Type') in ('ST', 'SW'))
    print(f"2026: {len(pitches)} MLB pitches ({n_st} ST/SW)", flush=True)
    rv_fn = make_rv_xrv(base.LG, base.SCALE)
    res = S.eval_season(pitches, rv_fn)
    print(f"{'config':>10s} | {'PARTIAL|velo':>13s} | {'raw':>6s} "
          f"| {'r(velo)':>8s} | {'rel':>6s}")
    for name in S.CONFIGS:
        o = res.get(name)
        if o:
            print(f"{name:>10s} | {o['partial']:>13.3f} | {o['raw']:>6.3f} "
                  f"| {o['rlv']:>+8.3f} | {o['rel']:>6.3f}", flush=True)
    s, p = res.get('shipped'), res.get('st_group')
    if s and p:
        print(f"\nst_group minus shipped: partial {p['partial']-s['partial']:+.3f}"
              f"  rel {p['rel']-s['rel']:+.3f}")


if __name__ == '__main__':
    main()
