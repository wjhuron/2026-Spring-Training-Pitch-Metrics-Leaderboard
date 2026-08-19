"""bbplus_ev_derivation.py — BB+ gains an exit-velocity ingredient.

BB+ was the mean of Savant xwOBAcon over a hitter's batted balls, indexed to
league. A mean of a heavy-tailed per-BIP quantity is a weak estimator of the
underlying skill. A high exit-velocity percentile measures the same skill far
more stably, so BB+ becomes a two-ingredient blend.

    BB+raw = W_CON * shrink(conPlus, nBIP, N0_CON)
           + W_EV  * shrink(evPlus,  nBIP, N0_EV)

Derived config (see RESULTS below):
    EV percentile 95, W_CON 0.30, W_EV 0.70, N0_CON 200, N0_EV 0.
    Hitter+ stays 52/17/31 — refitting it is NOT worth it, measured.

GRADERS
  DESCRIPTIVE  atoms measured on half the season's games predict the OTHER
               half's value. Talent is constant across halves, so what is
               left is estimator quality. Split is game-level, 3 seeds.
  PREDICTIVE   full-season atoms in year N predict year N+1 value.

  The outer loop is LEAVE-ONE-SEASON-OUT, not leave-one-fold-out: the six
  descriptive folds inside a season are built from the same games, so
  holding out one leaves its own data in the training set through the other
  five. Seasons are the independent unit. Both the BB+ config and the
  composite weights are fitted on training seasons only.

TWO TRAPS THIS SCRIPT GUARDS, BOTH OF WHICH PRODUCED FALSE POSITIVES

  1. SAMPLE SIZE. The percentile sweep rises monotonically to ev_max, which
     looks like a grid that is too small. It is not. A maximum grows
     mechanically with the number of batted balls, and BIP count tracks
     playing time, which tracks quality. ev_max correlates .389 with nBIP
     against .225 for p90. With log(nBIP) controlled, ev_max LOSES. Any
     candidate whose value could rise with sample size must be re-graded
     with the nBIP control before it is believed.

  2. MODEL FAMILY. The natural descriptive target is xwOBA, but xwOBA is a
     function of exit velocity and launch angle, so an EV-family channel can
     win it by sharing a model with the target rather than by carrying
     skill. Grading a mean-EV channel against xwOBA gave +.0086 (6/6
     seasons); the same channel against ACTUAL wOBA gave +.0017 and flipped
     sign across seasons. Every candidate is graded on an outcome target
     before it is believed.

  The shipped EV95 change survives both guards. See audit().

REJECTED ALONG THE WAY, all on these graders
  ev_max as the summary            sample-size artifact, guard 1
  launch angle mean / SD / sweet   +.0004 to +.0007 on outcomes, sign flips
  mean EV / hard-hit as a 3rd      +.0017 within season, -.0016 across
  BB+ normalised by pitch quality  reliability delta within +/-.007
  BB+ count-mix adjustment         reliability loses 18/18
  refitting Hitter+ 52/17/31       gains .0034 descriptive, costs .0083
                                   predictive; wins 4/6 and 3/5 on its own

Usage (from the repo root):
    python3 scripts/research/hitter/bbplus_ev_derivation.py build
    python3 scripts/research/hitter/bbplus_ev_derivation.py sweep
    python3 scripts/research/hitter/bbplus_ev_derivation.py nested
    python3 scripts/research/hitter/bbplus_ev_derivation.py audit

`build` is the slow step (roughly an hour for 6 seasons x 7 slices); it
writes data/_bbplus_ev_atoms.json and the rest read it. Results land in
data/_bbplus_ev_derivation.json.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import statcast_hitter_adapter as A
import pipeline.sdplus as sd
import pipeline.contact as ct
from pipeline.utils import BUNT_BB_TYPES, safe_float

ATOMS = os.path.join(ROOT, 'data', '_bbplus_ev_atoms.json')
OUT = os.path.join(ROOT, 'data', '_bbplus_ev_derivation.json')

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
SEEDS = [0, 1, 2]
PCTS = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
GUTS_2026 = (0.3172, 1.2343)
# Floors are deliberately low. Shrinkage is one of the swept constants, so
# thin rows must reach the analysis and be shrunk there, not filtered here.
MIN_BIP, MIN_PA = 15, 60

SHIPPED_W = np.array([0.52, 0.17, 0.31])
DERIVED = {'pct': 'ev_p95', 'w_ev': 0.70, 'n0_con': 200, 'n0_ev': 0}


# ── build ───────────────────────────────────────────────────────────────

def _pa_rates(pitches, vocab):
    """Per-hitter full-PA wOBA and xwOBA, in the public linear weights."""
    BIP_T = {'Single': .9, 'Double': 1.25, 'Triple': 1.6, 'Home Run': 2.0,
             'Field Error': .9, 'Fielders Choice': .9,
             'Fielders Choice Out': .9}
    PA_T = {'Walk': .7, 'Hit By Pitch': .72}
    acc = defaultdict(lambda: [0.0, 0.0, 0])
    for p in pitches:
        h = p['Batter']
        if not h:
            continue
        if vocab == 'snake':
            ev = p.get('event_raw')
            if not isinstance(ev, str) or not ev or ev == 'intent_walk':
                continue
            bipw, paw, kmark = A.BIP_WOBA_PUB, A.PA_WOBA_PUB, 'strikeout'
        else:
            ev = p.get('Event')
            if not isinstance(ev, str) or not ev or ev == 'Intent Walk':
                continue
            bipw, paw, kmark = BIP_T, PA_T, 'Strikeout'
        if p['Description'] == 'In Play':
            xw = safe_float(p.get('xwOBA'))
            if xw is None:
                continue
            w = bipw.get(ev, 0.0)
        elif ev in paw:
            w = xw = paw[ev]
        elif kmark in ev:
            w = xw = 0.0
        else:
            continue
        a = acc[h]
        a[0] += w
        a[1] += xw
        a[2] += 1
    return {h: (v[0] / v[2], v[1] / v[2], v[2]) for h, v in acc.items()}


def atoms_of(pitches, guts, vocab):
    """Every raw quantity a sweep needs, unshrunk."""
    lg, sc = guts
    byh = defaultdict(list)
    for p in pitches:
        if p['Batter']:
            byh[(p['Batter'], 'X')].append(p)
    sd_res, _ = sd.compute_sd_plus(pitches, dict(byh), lg, sc)
    ct_res, _ = ct.compute_ct_plus(pitches, dict(byh), lg, sc)

    con = defaultdict(lambda: [0.0, 0])
    evs, las = defaultdict(list), defaultdict(list)
    lgx = [0.0, 0]
    for p in pitches:
        if p['Description'] != 'In Play':
            continue
        bb = p.get('BBType')
        if not bb or bb in BUNT_BB_TYPES:
            continue
        h = p['Batter']
        xw = safe_float(p.get('xwOBA'))
        if xw is not None:
            lgx[0] += xw
            lgx[1] += 1
            if h:
                con[h][0] += xw
                con[h][1] += 1
        if not h:
            continue
        e = safe_float(p.get('ExitVelo'))
        if e is not None:
            evs[h].append(e)
        v = safe_float(p.get('LaunchAngle'))
        if v is not None:
            las[h].append(v)
    lg_xwcon = lgx[0] / lgx[1]

    rates = _pa_rates(pitches, vocab)
    out = {}
    for h, c in con.items():
        rt, ev, la = rates.get(h), evs.get(h, []), las.get(h, [])
        if c[1] < MIN_BIP or len(ev) < MIN_BIP or rt is None or rt[2] < MIN_PA:
            continue
        sdv, ctv = sd_res.get((h, 'X')), ct_res.get((h, 'X'))
        if sdv is None or ctv is None:
            continue
        E, L = np.asarray(ev), np.asarray(la)
        row = {'nbip': c[1], 'xwcon': c[0] / c[1], 'sd': sdv['sdPlus'],
               'ct': ctv['ctPlus'], 'pa': rt[2], 'woba': rt[0],
               'xwoba': rt[1], 'ev_mean': float(E.mean()),
               'hard': float((E >= 95).mean()), 'ev_max': float(E.max())}
        for q in PCTS:
            row[f'ev_p{q}'] = float(np.percentile(E, q))
        if len(L) >= 10:
            row.update(la_mean=float(L.mean()), la_sd=float(L.std()),
                       la_sweet=float(((L >= 8) & (L <= 32)).mean()))
        out[h] = row
    return {'lg_xwcon': lg_xwcon, 'rows': out}


def _load_season(year):
    if year == 2026:
        import pickle
        D = pickle.load(open(os.path.join(ROOT, 'data',
                                          'all_pitches_rs_cache.pkl'), 'rb'))
        return [p for p in D if p.get('_source', 'MLB') == 'MLB'], GUTS_2026, 'title'
    return A.season_dicts(year), A.GUTS[year], 'snake'


def build():
    store = json.load(open(ATOMS)) if os.path.exists(ATOMS) else {}
    for year in SEASONS:
        keys = [f'{year}_full'] + [f'{year}_{s}{t}'
                                   for s in SEEDS for t in ('A', 'B')]
        if all(k in store for k in keys):
            print(f"{year}: cached", flush=True)
            continue
        P, guts, vocab = _load_season(year)
        print(f"{year}: {len(P):,} pitches", flush=True)
        store[f'{year}_full'] = atoms_of(P, guts, vocab)
        dates = sorted({p['Game Date'] for p in P if p.get('Game Date')})
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            side = dict(zip(dates, rng.random(len(dates)) < 0.5))
            for tag, want in (('A', True), ('B', False)):
                sub = [p for p in P
                       if side.get(p.get('Game Date'), not want) == want]
                k = f'{year}_{seed}{tag}'
                store[k] = atoms_of(sub, guts, vocab)
                print(f"  {k}: {len(store[k]['rows'])}", flush=True)
        json.dump(store, open(ATOMS, 'w'))
        del P
    print(f"wrote {ATOMS}")


# ── shared analysis machinery ───────────────────────────────────────────

def _bridge():
    """2026 comes from the pipeline cache and keys hitters by name; the
    statcast seasons key by MLBAM id. Bridge on the Savant bat-tracking
    leaderboard, which carries both. Absent that file, 2026 is skipped."""
    p = os.path.join(ROOT, 'data', '_bt_seasons.json')
    if not os.path.exists(p):
        # Announce the degrade. Without this file the 2026 rows keep their
        # name keys, never join to the id-keyed seasons, and silently vanish
        # from every fold — the results would still print, one replicate
        # short, with nothing to say so.
        print(f"WARNING: {p} missing — 2026 cannot be bridged from name keys "
              f"to MLBAM ids, so it is DROPPED from every fold. Results below "
              f"are 2021-2025 only. Repull with the Savant bat-tracking "
              f"leaderboard (seasonStart/seasonEnd, not season).",
              file=sys.stderr)
        return None
    m = {}
    for y in ('2026', '2025', '2024'):
        for pid, row in json.load(open(p)).get(y, {}).items():
            if row.get('name'):
                m.setdefault(row['name'], pid)
    return m


def z(v):
    v = np.asarray(v, float)
    s = v.std(ddof=0)
    return (v - v.mean()) / s if s > 0 else v * 0.0


def shrink(v, n, n0):
    return (n * v + n0 * 100.0) / (n + n0)


class Data:
    def __init__(self):
        self.store = json.load(open(ATOMS))
        self.bridge = _bridge()

    def slice(self, key, year):
        sl = self.store[key]
        rows = sl['rows']
        if year == 2026 and self.bridge:
            rows = {self.bridge[h]: v for h, v in rows.items()
                    if h in self.bridge}
        return rows, sl['lg_xwcon']

    def fold(self, src, lg, tgt, tkey, extra=()):
        ids = [h for h in src if h in tgt
               and all(k in src[h] for k in extra)]
        if len(ids) < 40:
            return None
        n = np.array([src[h]['nbip'] for h in ids], float)
        con = 100.0 * np.array([src[h]['xwcon'] for h in ids]) / lg
        ev = {}
        for p in [f'ev_p{q}' for q in PCTS] + ['ev_max']:
            r = np.array([src[h][p] for h in ids], float)
            ev[p] = 100.0 * r / float((r * n).sum() / n.sum())
        f = {'n': n, 'con': con, 'ev': ev, 'm': len(ids),
             'sd': z([src[h]['sd'] for h in ids]),
             'ct': z([src[h]['ct'] for h in ids]),
             'y': z([tgt[h][tkey] for h in ids])}
        for k in extra:
            f[k] = z([src[h][k] for h in ids])
        return f

    def groups(self, tkey='xwoba', extra=()):
        """Descriptive folds keyed by season, plus predictive pairs."""
        desc, pred = defaultdict(list), {}
        for y in SEASONS:
            for seed in SEEDS:
                ka, kb = f'{y}_{seed}A', f'{y}_{seed}B'
                if ka not in self.store or kb not in self.store:
                    continue
                A_, lgA = self.slice(ka, y)
                B_, lgB = self.slice(kb, y)
                for s, lg, t in ((A_, lgA, B_), (B_, lgB, A_)):
                    f = self.fold(s, lg, t, tkey, extra)
                    if f:
                        desc[y].append(f)
        ptkey = 'woba' if tkey == 'xwoba' else tkey
        for yn, yn1 in zip(SEASONS, SEASONS[1:]):
            if f'{yn}_full' not in self.store or f'{yn1}_full' not in self.store:
                continue
            S_, lg = self.slice(f'{yn}_full', yn)
            T_, _ = self.slice(f'{yn1}_full', yn1)
            f = self.fold(S_, lg, T_, ptkey, extra)
            if f:
                pred[yn] = [f]
        return desc, pred


def bb_of(f, cfg):
    """cfg None -> the pre-2026-08-19 BB+ (pure xwOBAcon, n0 60)."""
    if cfg is None:
        return z(shrink(f['con'], f['n'], 60))
    return z((1 - cfg['w_ev']) * shrink(f['con'], f['n'], cfg['n0_con'])
             + cfg['w_ev'] * shrink(f['ev'][cfg['pct']], f['n'], cfg['n0_ev']))


def fit_w(folds, cfg, extra=()):
    cols = [lambda f: bb_of(f, cfg), lambda f: f['sd'], lambda f: f['ct']]
    cols += [(lambda f, k=k: f[k]) for k in extra]
    X = np.column_stack([np.ones(sum(f['m'] for f in folds))] +
                        [np.concatenate([c(f) for f in folds]) for c in cols])
    y = np.concatenate([f['y'] for f in folds])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def apply_w(f, cfg, b, extra=()):
    p = b[0] + b[1] * bb_of(f, cfg) + b[2] * f['sd'] + b[3] * f['ct']
    for i, k in enumerate(extra):
        p = p + b[4 + i] * f[k]
    return p


def loso(groups, cfg, extra=(), fixed_w=None):
    """Leave-one-season-out held-out r. fixed_w pins the composite weights."""
    keys = sorted(groups)
    out = []
    for held in keys:
        tr = [f for k in keys if k != held for f in groups[k]]
        if fixed_w is not None:
            b = np.concatenate([[0.0], fixed_w])
        else:
            b = fit_w(tr, cfg, extra)
        rs = [np.corrcoef(apply_w(f, cfg, b, extra), f['y'])[0, 1]
              for f in groups[held]]
        out.append(np.mean(rs))
    return np.array(out)


# ── sweeps, nested selection, audit ─────────────────────────────────────

def sweep(res):
    d = Data()
    desc, pred = d.groups()
    allf = [f for v in desc.values() for f in v]

    def pooled(cfg):
        b = fit_w(allf, cfg)
        return float(np.mean([np.corrcoef(apply_w(f, cfg, b), f['y'])[0, 1]
                              for f in allf]))

    res['percentile'] = {
        p: pooled({'pct': p, 'w_ev': .70, 'n0_con': 200, 'n0_ev': 0})
        for p in [f'ev_p{q}' for q in PCTS] + ['ev_max']}
    res['w_ev'] = {
        f'{w:.2f}': pooled({'pct': 'ev_p95', 'w_ev': w,
                            'n0_con': 200, 'n0_ev': 0})
        for w in np.arange(0.45, 0.96, 0.05)}
    res['shrinkage'] = {
        f'{nc}/{ne}': pooled({'pct': 'ev_p95', 'w_ev': .70,
                              'n0_con': nc, 'n0_ev': ne})
        for nc in (130, 200, 280, 380, 500, 650) for ne in (0, 5, 15)}
    for k in ('percentile', 'w_ev', 'shrinkage'):
        print(f"\n{k}")
        for a, b in res[k].items():
            print(f"  {a:<12}{b:.4f}")
    return res


def nested(res):
    d = Data()
    desc, pred = d.groups()
    out = {}
    for label, g in (('descriptive', desc), ('predictive', pred)):
        arms = {
            'shipped BB+ / 52-17-31': (None, SHIPPED_W),
            'shipped BB+ / fitted':   (None, None),
            'new BB+ / 52-17-31':     (DERIVED, SHIPPED_W),
            'new BB+ / fitted':       (DERIVED, None),
        }
        base = loso(g, None, fixed_w=SHIPPED_W)
        out[label] = {}
        print(f"\n{label.upper()} — leave one season out ({len(g)} units)")
        for name, (cfg, w) in arms.items():
            v = loso(g, cfg, fixed_w=w)
            out[label][name] = {'r': float(v.mean()),
                                'delta': float(v.mean() - base.mean()),
                                'won': int((v > base).sum()), 'n': len(v)}
            print(f"  {name:<24}{v.mean():.4f}  "
                  f"{v.mean()-base.mean():+.4f}  {int((v>base).sum())}/{len(v)}")
    res['nested'] = out
    return res


def audit(res):
    """The two guards. Both caught a false positive during the derivation."""
    d = Data()
    out = {}

    # guard 1: sample size
    bands = [(15, 60), (60, 100), (100, 160), (160, 10 ** 9)]
    band_out = {}
    for lo, hi in bands:
        V, n = defaultdict(list), []
        for y in SEASONS:
            if f'{y}_full' not in d.store:
                continue
            rows, _ = d.slice(f'{y}_full', y)
            for h, r in rows.items():
                if lo <= r['nbip'] < hi:
                    for k in ('ev_p90', 'ev_p95', 'ev_max'):
                        V[k].append(r[k])
                    n.append(r['nbip'])
        if len(n) < 40:
            continue
        band_out[f'{lo}-{hi}'] = {
            k: float(np.corrcoef(V[k], n)[0, 1]) for k in V}
    out['nbip_correlation_by_band'] = band_out
    print("\nGUARD 1 — corr(statistic, nBIP) by sample-size band")
    for b, v in band_out.items():
        print(f"  {b:<14}" + "  ".join(f"{k} {x:+.3f}" for k, x in v.items()))

    # guard 2: model family. Re-grade against an OUTCOME target.
    print("\nGUARD 2 — the shipped change on a model target vs an outcome target")
    for tkey, label in (('xwoba', 'half B xwOBA (EV/LA model output)'),
                        ('woba', 'half B wOBA  (actual outcomes)')):
        g, _ = d.groups(tkey)
        a = loso(g, None)
        b = loso(g, DERIVED)
        out[f'target_{tkey}'] = {'shipped': float(a.mean()),
                                 'derived': float(b.mean()),
                                 'won': int((b > a).sum()), 'n': len(a)}
        print(f"  {label:<36} {a.mean():.4f} -> {b.mean():.4f}  "
              f"{b.mean()-a.mean():+.4f}  {int((b>a).sum())}/{len(a)}")
    res['audit'] = out
    return res


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'nested'
    if cmd == 'build':
        build()
        return
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res['config'] = DERIVED
    res = {'sweep': sweep, 'nested': nested, 'audit': audit}[cmd](res)
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
