"""locplus_mirror_prior_sweep.py — sweep K_MIRROR (2026-09-05).

Partial pooling of each (group, batter hand, pitcher hand) collapsed Loc+
surface toward the MIRRORED surface of its mirror pair (both hands flipped,
PlateX negated). K_MIRROR = 0 is the shipped model (no pooling); a very
large K is a full mirror. Kernel-weighted units like every other Loc+ K.

Protocol: locplus_fullseason_replicate.eval_season_full, verbatim — full-
season surfaces per season 2021-2026, first-half pitcher Loc+ vs second-
half luck-neutral xRV, raw and rendered units, FF-velocity partials, plus
the split-half reliability diagnostic. Paired across seasons against
K_MIRROR = 0.

RESULT 2026-09-05 (data/_loc_mirror_sweep.json): flat for K <= 100 (deltas
inside 0.001, 2/6 wins), monotonically WORSE from K300 up on every pitcher-
level objective (K100000 = full mirror: raw -0.018, rendered -0.024, 0/6).
K_MIRROR = 0 stands. Needs the K_MIRROR prior in pipeline/locplus.py, kept
as data/_loc_mirror_prior.patch (reverted from the working tree after the
sweep; K=0 reproduced the shipped surfaces exactly).

Usage: git apply data/_loc_mirror_prior.patch; python3 scripts/research/locplus/locplus_mirror_prior_sweep.py [--ks 0,10,...] [--seasons 2021,...]
Writes data/_loc_mirror_sweep.json.
"""
import argparse, gc, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import locplus_fullseason_replicate as R
from pipeline.sdplus import make_rv_xrv

_apply, _restore = R.apply, R.restore
def apply(cfg, zvariant=None):
    _apply({k: v for k, v in cfg.items() if k in R.KEYS}, zvariant)
    R.lp.K_MIRROR = cfg.get('K_MIRROR', 0)
def restore():
    _restore(); R.lp.K_MIRROR = 0
R.apply, R.restore = apply, restore

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ks', default='0,10,30,100,300,1000,3000,10000,100000')
    ap.add_argument('--seasons', default=None)
    a = ap.parse_args()
    ks = [float(x) for x in a.ks.split(',')]
    configs = {}
    for k in ks:
        name = 'shipped' if k == 0 else f'K{int(k)}'
        configs[name] = dict(R.SHIPPED, K_MIRROR=k)
    seasons = [int(s) for s in a.seasons.split(',')] if a.seasons else sorted(R.SEASONS)
    rv_fn = make_rv_xrv(R.LG, R.SCALE)
    full, meta = {}, {}
    for yr in seasons:
        t0 = time.time(); pitches = R.load_season(yr)
        print(f"=== {yr}: {len(pitches)} pitches loaded ({time.time() - t0:.0f}s)", flush=True)
        full[yr], meta[yr] = R.eval_season_full(pitches, rv_fn, configs, None)
        del pitches; gc.collect()
        json.dump({'configs': configs, 'full': {str(y): v for y, v in full.items()}, 'meta': {str(y): v for y, v in meta.items()}},
                  open(os.path.join(ROOT, 'data', '_loc_mirror_sweep.json'), 'w'), indent=1)
    metrics = ['raw', 'raw_partial', 'rendered', 'rendered_partial', 'rel_diag']
    R.summarise(full, configs, metrics, 'FULL-SEASON surfaces, first-half Loc+ vs second-half xRV')
    print("\nPAIRED vs shipped (K_MIRROR=0): mean delta, wins/n, paired-t SE; [x] = without the partial 2026 season")
    for m in metrics:
        print(f"-- {m}")
        for name in configs:
            if name == 'shipped': continue
            ds = [(yr, full[yr][name][m] - full[yr]['shipped'][m]) for yr in full]
            for tag, sub in (('all', ds), ('x', [x for x in ds if x[0] != 2026])):
                v = [x[1] for x in sub]; n = len(v)
                if n < 2: continue
                mu = sum(v) / n; se = math.sqrt(sum((x - mu) ** 2 for x in v) / (n - 1) / n)
                print(f"   {name:>8} [{tag:3}] {mu:+.4f}  wins {sum(1 for x in v if x > 0)}/{n}  SE {se:.4f}  t {mu / se if se > 0 else float('nan'):+.1f}")

if __name__ == '__main__':
    main()
