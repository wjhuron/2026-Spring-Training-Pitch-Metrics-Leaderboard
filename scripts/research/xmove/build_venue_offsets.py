"""Build data/venue_offsets.json — per-ballpark movement measurement offsets.

Hawkeye is calibrated per installation, and this repo already carries a
park-level tracking artifact on record (the Progressive Field SwingLength
glitch). Measured on 2021-2025, within-pitcher venue offsets replicate season
to season at r = 0.894 (along-axis) and 0.761 (cross-axis), which a roster
artifact could not do.

Deliberately MODEL-FREE: the offset is the mean of each venue's residual after
removing every (pitcher, pitch type, season) mean, so it depends on no movement
model and does not need re-deriving when the model changes. It is expressed in
the release-axis frame (along = Magnus direction, cross = seam direction) and
hand-mirrored so LHP and RHP pool.

Confounds handled:
  * roster -- killed by the within-pitcher demeaning; a park's own staff cannot
    create an offset once each pitcher is centred on his own mean
  * air density -- already removed upstream, since the movement columns are
    weather-adjusted (xIndVrtBrk/xHorzBrk)
  * small samples -- empirical-Bayes shrinkage, with the shrinkage constant
    DERIVED from the between-venue and within-venue variances rather than
    picked

What this does NOT establish is the CAUSE. Tropicana Field being the largest
offender is suggestive of rig calibration (a dome has no weather to mis-model),
but Coors ranking high after density adjustment points at the 1.05 exponent
being slightly off at altitude, and Oracle Park points at humidity. The
correction is empirical and agnostic.

Applied to the RESIDUAL, not to displayed IVB/HB: these are ~0.2" and changing
the displayed movement columns is a separate decision.

Usage: python3 scripts/research/xmove/build_venue_offsets.py [--apply]
"""
import os, sys, json, argparse
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from xmove_compare import load_np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DIR = os.environ.get('XMOVE_DIR', '/private/tmp/claude-501/-Users-wallyhuron-Huronalytics/8aed4efe-0775-4afc-b652-6ddab7be7e7d/scratchpad')
OUT = os.path.join(ROOT, 'data', 'venue_offsets.json')
MIN_VENUE_PITCHES = 2000


def demean(vals, keys):
    u, inv = np.unique(keys, return_inverse=True)
    cnt = np.bincount(inv)
    tot = np.bincount(inv, weights=vals)
    return vals - (tot / cnt)[inv]


def eb_shrink(obs, n, sigma2):
    """Empirical Bayes. tau2 (true between-venue variance) is backed out of the
    observed spread minus the mean sampling variance, so the shrinkage constant
    is measured rather than chosen."""
    samp = sigma2 / np.maximum(n, 1)
    tau2 = max(0.0, float(np.var(obs, ddof=1) - samp.mean()))
    w = tau2 / (tau2 + samp)
    return obs * w, tau2, w


def main(apply_it):
    A = load_np()
    with open(f'{DIR}/venue_by_gamepk.json') as f:
        vraw = json.load(f)
    vmap = {int(k): v[0] for k, v in vraw.items()}
    vname = {v[0]: v[1] for v in vraw.values()}
    venue = np.array([vmap.get(int(g), -1) for g in A['game']])

    # release-axis frame, hand-mirrored (both already handled in load_np)
    along, cross = A['along'], A['cross']
    key = np.array([f"{p}|{t}|{q}|{s}" for p, t, q, s in
                    zip(A['pitcher'], A['thr'], A['pt'], A['season'])])
    ok = venue > 0
    a_d = demean(along[ok], key[ok])
    c_d = demean(cross[ok], key[ok])
    v = venue[ok]
    ssn = A['season'][ok]

    sig2_a = float(np.var(a_d, ddof=1))
    sig2_c = float(np.var(c_d, ddof=1))
    print(f'{ok.sum():,} pitches; within-pitcher residual sd '
          f'along {np.sqrt(sig2_a):.2f}"  cross {np.sqrt(sig2_c):.2f}"')

    ids, ns, oa, oc, seasons = [], [], [], [], []
    for vid in np.unique(v):
        m = v == vid
        if m.sum() < MIN_VENUE_PITCHES:
            continue
        ids.append(int(vid)); ns.append(int(m.sum()))
        oa.append(float(a_d[m].mean())); oc.append(float(c_d[m].mean()))
        seasons.append(int(len(np.unique(ssn[m]))))
    ids = np.array(ids); ns = np.array(ns)
    oa = np.array(oa); oc = np.array(oc)

    sa, tau_a, wa = eb_shrink(oa, ns, sig2_a)
    sc, tau_c, wc = eb_shrink(oc, ns, sig2_c)
    print(f'\nempirical Bayes:')
    print(f'  between-venue true sd  along {np.sqrt(tau_a):.3f}"  '
          f'cross {np.sqrt(tau_c):.3f}"')
    print(f'  shrinkage weight       along {wa.min():.3f}-{wa.max():.3f}  '
          f'cross {wc.min():.3f}-{wc.max():.3f}')
    print(f'  (weights near 1.0 mean full-season parks have ample sample; '
          f'shrinkage only bites\n   on short samples such as a new park or a '
          f'partial season)')

    print(f'\n{len(ids)} venues, largest shrunk offsets:')
    print(f"  {'venue':<34} {'pitches':>9} {'seasons':>8} {'along':>7} {'cross':>7}")
    order = np.argsort(-np.hypot(sa, sc))
    for i in order[:10]:
        print(f'  {str(vname.get(ids[i], ids[i]))[:34]:<34} {ns[i]:>9,} '
              f'{seasons[i]:>8} {sa[i]:>+7.3f} {sc[i]:>+7.3f}')

    payload = dict(
        note=('Within-pitcher movement offset per ballpark, release-axis frame, '
              'hand-mirrored (arm side positive), inches. Empirical-Bayes '
              'shrunk. Subtract from the measured along/cross before computing '
              'the movement residual.'),
        seasons=[int(s) for s in sorted(set(int(x) for x in A['season']))],
        betweenVenueSd=dict(along=round(float(np.sqrt(tau_a)), 4),
                            cross=round(float(np.sqrt(tau_c)), 4)),
        venues={str(int(ids[i])): dict(name=vname.get(ids[i]), n=int(ns[i]),
                                       seasons=int(seasons[i]),
                                       along=round(float(sa[i]), 4),
                                       cross=round(float(sc[i]), 4))
                for i in range(len(ids))})
    if apply_it:
        with open(OUT, 'w') as f:
            json.dump(payload, f, indent=0, sort_keys=True)
        print(f'\nwrote {OUT}  ({len(ids)} venues)')
    else:
        print(f'\n(dry run -- {len(ids)} venues would be written; pass --apply)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    main(ap.parse_args().apply)
