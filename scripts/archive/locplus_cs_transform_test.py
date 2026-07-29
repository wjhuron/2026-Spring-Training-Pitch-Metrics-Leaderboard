"""locplus_cs_transform_test.py — does a COUNT TRANSFORM on the called-strike
surface help Loc+? (research item #3)

The physical surfaces are correctly count-collapsed EXCEPT the called-strike
surface: umpires expand the zone 3-0 and shrink it 0-2, so the SAME location is
called a strike at different rates by count. Rather than re-estimate a full CS
surface per count (sparse), apply a per-count logit INTERCEPT calibrated so the
predicted called-strike count matches the observed count among that count's taken
pitches (BP framing-model style). One baseline surface + 12 scalars.

  delta_c = logit(obs_cs_rate_c) - logit(mean predicted PCS over takes in count c)
  PCS_c(x,z) = sigmoid( logit(PCS(x,z)) + delta_c )

2026-only baseline (multi-season already shown to hurt). Reuses the machinery in
locplus_multiseason_test.py; evaluates the same 3 objectives.

Usage: python3 scripts/locplus_cs_transform_test.py
"""
import os, sys, math, pickle, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, ROOT)
import locplus_multiseason_test as L


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6); return math.log(p / (1 - p))
def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def score_cs(P, S, RV, PCSc):
    for p in P:
        t = p['_ft']
        if t is None: p['_v3'] = None; continue
        g, bh, ph, c, i, j, d, xw, re = t
        key = (g, bh, ph)
        psw = S['SW'][key][c][i][j]; pwh = S['WH'][key][i][j]; pfl = S['FL'][key][i][j]
        pbip = max(0.0, 1 - pwh - pfl); vbip = S['XW'][key][i][j]; pcs = PCSc[c][i][j]
        sv = pwh * RV['whiff'].get(c, 0) + pfl * RV['foul'].get(c, 0) + pbip * vbip
        tv = pcs * RV['cs'].get(c, 0) + (1 - pcs) * RV['ball'].get(c, 0)
        p['_v3'] = psw * sv + (1 - psw) * tv


def main():
    print('loading 2026 ...', flush=True)
    P = [p for p in pickle.load(open(L.PKL, 'rb')) if p.get('_source') == 'MLB']
    for p in P: p['_ft'] = L.feat_dict(p)
    P = [p for p in P if p['_ft'] is not None]
    print(f'  {len(P)} scorable', flush=True)
    RV = L.rv_from(P); infra = L.build_eval_infra(P)
    A, AC, csn, csd, n = L.accumulate([(p['_ft'] for p in P)])
    S = L.build_surfaces(A, AC, csn, csd, RV, count_specific=False)

    # baseline (count-independent CS surface)
    L.score(P, S)
    base = L.evaluate(P, infra)
    PCS = S['PCS']

    # per-count calibration of the CS surface over that count's takes
    agg = collections.defaultdict(lambda: {'cs': 0, 'tk': 0, 'pred': 0.0})
    for p in P:
        g, bh, ph, c, i, j, d, xw, re = p['_ft']
        if d in L.TAKE_DESC:
            a = agg[c]; a['tk'] += 1; a['pred'] += PCS[i][j]
            if d == 'Called Strike': a['cs'] += 1
    deltas = {}
    for c in L.COUNTS:
        a = agg.get(c)
        if not a or a['tk'] < 50 or a['pred'] <= 0:
            deltas[c] = 0.0; continue
        obs = a['cs'] / a['tk']; predr = a['pred'] / a['tk']
        deltas[c] = logit(obs) - logit(predr)
    PCSc = {c: [[sig(logit(PCS[i][j]) + deltas[c]) for j in range(L.NZ)] for i in range(L.NX)] for c in L.COUNTS}

    score_cs(P, S, RV, PCSc)
    cst = L.evaluate(P, infra)

    print(f"\n{'variant':26s} {'reliab':>7s} {'pred':>7s} {'rWhf':>6s} {'rVel':>6s}")
    print(f"{'BASELINE (CS count-indep)':26s} {base[0]:7.3f} {base[3]:7.3f} {base[1]:6.3f} {base[2]:6.3f}")
    print(f"{'CS count-transform':26s} {cst[0]:7.3f} {cst[3]:7.3f} {cst[1]:6.3f} {cst[2]:6.3f}")
    print(f"\ndelta: reliab {cst[0]-base[0]:+.3f}  pred {cst[3]-base[3]:+.3f}  "
          f"stuff-leak {cst[1]-base[1]:+.3f}")
    print("\nper-count CS logit shift (zone expand +, shrink -):")
    for s in range(3):
        row = "  ".join(f"{b}-{s}:{deltas[(b,s)]:+.2f}" for b in range(4))
        print(f"   {row}")


if __name__ == '__main__':
    main()
