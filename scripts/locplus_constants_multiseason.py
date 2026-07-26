"""locplus_constants_multiseason.py — do the tuned Loc+ constants transfer across seasons?

WHAT THIS IS NOT. It is NOT a repeat of the 2026-07-05 multi-season test, which
POOLED seasons into the surfaces and degraded every objective monotonically.
Location value is non-stationary, so surfaces stay current-season-only and that
is settled. Here every season builds its OWN surfaces from its OWN data — the
seasons are replicates, never pooled. The question is whether the CONSTANTS
(bandwidth + the six shrinkage pseudo-counts), which were fitted to 2026, also
win in seasons they never saw. Constants should transfer even where surfaces
cannot; that was an assumption behind shipping them, and this tests it.

OBJECTIVE SUBSTITUTION. The 2026 work scored on partial r(loc, xRV | Stuff+).
The historical Statcast caches carry no Stuff+ column, so this uses
partial r(loc, xRV | FF velo). Velocity is the dominant component of Stuff+, and
crucially the SAME control is applied to both configs within a season, so the
comparison stays apples-to-apples even though the absolute value differs from
the 2026 numbers. Raw prediction and reliability are reported alongside.

TWO CONVENTIONS THAT WILL SILENTLY INVERT RESULTS IF GOT WRONG:
  1. Statcast delta_run_exp is OFFENSE-perspective (called strike -0.065);
     Wally's feed RunExp is PITCHER-perspective (called strike +0.066). The
     adapter negates. Verified empirically against both sources.
  2. lg_woba / woba_scale are FanGraphs guts and season-specific; 2026 values
     are reused for every season. That shifts the BIP branch slightly in
     absolute terms but is applied identically to both configs, and
     correlations are scale-invariant, so it cannot flip a comparison.

Usage: python3 scripts/locplus_constants_multiseason.py
"""
import os, sys, math, pickle, gc
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pipeline_locplus as lp
from pipeline_sdplus import make_rv_xrv

LG, SCALE = 0.3169, 1.2393
MIN_SCORE, MIN_ACTUAL, MIN_REL, MIN_VELO = 150, 150, 100, 40

KEYS = ['K_WHIFF', 'K_FOUL', 'K_XWCON', 'K_SWING_COLL', 'K_SWING_COUNT', 'K_CS']
ORIG = {k: getattr(lp, k) for k in KEYS}

# Every configuration considered on 2026-07-25, so one run settles all of them.
# The bandwidth-only variants keep the ORIGINAL shrinkage constants, isolating
# the kernel change from the shrinkage retune.
CONFIGS = [
    ('shipped',   4.5, {k: 1 for k in KEYS}),
    ('x3.0',      3.0, {k: 1 for k in KEYS}),
    ('x2.5',      2.5, {k: 1 for k in KEYS}),
    ('x1.5',      1.5, {k: 1 for k in KEYS}),
    ('retuned-K', 1.5, {'K_WHIFF': 32, 'K_FOUL': 2, 'K_XWCON': 8,
                        'K_SWING_COLL': 64, 'K_SWING_COUNT': 2, 'K_CS': 2}),
]

DMAP = {'ball': 'Ball', 'blocked_ball': 'Ball', 'called_strike': 'Called Strike',
        'swinging_strike': 'Swinging Strike', 'swinging_strike_blocked': 'Swinging Strike',
        'foul_tip': 'Swinging Strike', 'foul': 'Foul', 'hit_into_play': 'In Play',
        'hit_into_play_no_out': 'In Play', 'hit_into_play_score': 'In Play',
        'hit_by_pitch': 'Hit By Pitch', 'foul_bunt': 'Foul Bunt',
        'missed_bunt': 'Missed Bunt', 'bunt_foul_tip': 'Bunt Foul Tip',
        'pitchout': 'Pitchout'}


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n; my = sum(ys) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    sy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if sx <= 0 or sy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)


def partial(r_ly, r_ls, r_sy):
    d = (1 - r_ls ** 2) * (1 - r_sy ** 2)
    return (r_ly - r_ls * r_sy) / math.sqrt(d) if d > 0 else None


def _f(v):
    """pandas NA/NaN -> None, else float. pipeline_utils.safe_float raises on
    pd.NA (`val == ''` is ambiguous), so every numeric field is sanitized at the
    adapter boundary rather than patching the pipeline for a research script."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def _s(v):
    """pandas NA -> None for string columns."""
    if v is None or not isinstance(v, str):
        return None
    return v


def adapt(path):
    """Statcast DataFrame -> pipeline_locplus-shaped dicts."""
    df = pickle.load(open(path, 'rb'))
    cols = ['pitch_type', 'plate_x', 'plate_z', 'sz_top', 'sz_bot', 'balls',
            'strikes', 'description', 'delta_run_exp',
            'estimated_woba_using_speedangle', 'stand', 'p_throws',
            'player_name', 'game_date', 'release_speed']
    sub = df[cols]
    out = []
    for r in sub.itertuples(index=False):
        d = DMAP.get(r.description)
        if d is None or r.pitch_type is None:
            continue
        try:
            b, s = int(r.balls), int(r.strikes)
        except (TypeError, ValueError):
            continue
        if not (0 <= b <= 3 and 0 <= s <= 2):
            continue
        # pandas NA breaks both `x != x` and float(); go through a guarded cast.
        try:
            re = float(r.delta_run_exp)
            if re != re:
                re = None
        except (TypeError, ValueError):
            re = None
        out.append({
            'Pitch Type': _s(r.pitch_type), 'Bats': _s(r.stand), 'Throws': _s(r.p_throws),
            'PlateX': _f(r.plate_x), 'PlateZ': _f(r.plate_z),
            'SzTop': _f(r.sz_top), 'SzBot': _f(r.sz_bot),
            'Count': f'{b}-{s}', 'Description': d,
            # NEGATED: statcast is offense-perspective, the pipeline expects
            # pitcher-perspective (see module docstring).
            'RunExp': (None if re is None else -re),
            'xwOBA': _f(r.estimated_woba_using_speedangle),
            'Pitcher': _s(r.player_name), 'Game Date': str(r.game_date)[:10],
            'Velocity': _f(r.release_speed), '_source': 'MLB',
            'Event': None, 'BBType': None,
        })
    del df, sub
    gc.collect()
    return out


def by_pitcher(pitches):
    d = defaultdict(list)
    for p in pitches:
        d[(p.get('Pitcher'), p.get('Throws'))].append(p)
    return d


def score_map(byp, S, min_n):
    out = {}
    for k, ps in byp.items():
        v = [s for s in (lp.score_pitch(p, S) for p in ps) if s is not None]
        if len(v) >= min_n:
            out[k] = sum(v) / len(v)
    return out


def eval_season(pitches, rv_fn):
    """Returns {config_name: metrics} for one season, surfaces built in-season."""
    base = [p for p in pitches if lp.is_eligible_baseline(p)]
    dates = sorted({p['Game Date'] for p in base if p['Game Date']})
    q = len(dates) // 4
    cuts = [dates[q], dates[2 * q], dates[3 * q]]
    seg = defaultdict(list)
    for p in base:
        d = p['Game Date']
        seg['A1' if d < cuts[0] else 'A2' if d < cuts[1]
            else 'B1' if d < cuts[2] else 'B2'].append(p)
    par = {d: i % 2 for i, d in enumerate(dates)}
    g0 = [p for p in base if par.get(p['Game Date']) == 0]
    g1 = [p for p in base if par.get(p['Game Date']) == 1]

    prep = {}
    for half, f, s in (('A', 'A1', 'A2'), ('B', 'B1', 'B2')):
        actual = {}
        for k, ps in by_pitcher(seg[s]).items():
            v = [x for x in (rv_fn(p) for p in ps) if x is not None]
            if len(v) >= MIN_ACTUAL:
                actual[k] = sum(v) / len(v)
        velo = {}
        for k, ps in by_pitcher(seg[f]).items():
            v = [x for x in (lp.safe_float(p['Velocity']) for p in ps
                             if p['Pitch Type'] == 'FF') if x is not None]
            if len(v) >= MIN_VELO:
                velo[k] = sum(v) / len(v)
        prep[half] = {'first': seg[f], 'byp_f': by_pitcher(seg[f]),
                      'actual': actual, 'velo': velo}

    res = {}
    for name, x, facs in CONFIGS:
        lp.PHYS_X_IN = x; lp.PHYS_Z_FRAC = 0.22
        lp._KX = lp._k1d(x / 2.0); lp._KZ = lp._k1d(0.22 / lp.BIN_Z)
        for k in KEYS:
            setattr(lp, k, ORIG[k] * facs[k])
        raws, parts, rlvs = [], [], []
        for half in ('A', 'B'):
            c = prep[half]
            S = lp.build_surfaces(c['first'], LG, SCALE)
            loc = score_map(c['byp_f'], S, MIN_SCORE)
            kk = [k for k in loc if k in c['actual'] and k in c['velo']]
            if len(kk) < 30:
                continue
            r_ly = pearson([loc[k] for k in kk], [c['actual'][k] for k in kk])
            r_ls = pearson([loc[k] for k in kk], [c['velo'][k] for k in kk])
            r_sy = pearson([c['velo'][k] for k in kk], [c['actual'][k] for k in kk])
            raws.append(r_ly); parts.append(partial(r_ly, r_ls, r_sy)); rlvs.append(r_ls)
        S0 = lp.build_surfaces(g0, LG, SCALE)
        S1 = lp.build_surfaces(g1, LG, SCALE)
        a0, a1 = score_map(by_pitcher(g0), S0, MIN_REL), score_map(by_pitcher(g1), S1, MIN_REL)
        ks = [k for k in a0 if k in a1]
        rel = pearson([a0[k] for k in ks], [a1[k] for k in ks])
        res[name] = {'raw': sum(raws) / len(raws), 'partial': sum(parts) / len(parts),
                     'rlv': sum(rlvs) / len(rlvs), 'rel': rel, 'n': len(ks)}
    return res


def main():
    rv_fn = make_rv_xrv(LG, SCALE)
    seasons = [(2021, 'data/_statcast2021_cache.pkl'),
               (2022, 'data/_statcast2022_cache.pkl'),
               (2023, 'data/_statcast2023_cache.pkl'),
               (2024, 'data/_statcast2024_cache.pkl'),
               (2025, 'data/_statcast2025_full_cache.pkl')]
    print(f"{'season':>7s} {'config':>9s} | {'PARTIAL|velo':>13s} | {'raw':>6s} "
          f"| {'r(velo)':>8s} | {'rel':>6s}")
    print('-' * 62)
    table = {}
    for yr, path in seasons:
        p = os.path.join(ROOT, path)
        if not os.path.exists(p):
            print(f"{yr:>7d}   cache missing, skipped", flush=True)
            continue
        print(f"adapting {yr}...", file=sys.stderr)
        pitches = adapt(p)
        print(f"  {yr}: {len(pitches)} usable pitches", file=sys.stderr)
        res = eval_season(pitches, rv_fn)
        table[yr] = res
        for name, _x, _f in CONFIGS:
            o = res.get(name)
            if o:
                print(f"{yr:>7d} {name:>9s} | {o['partial']:>13.3f} | {o['raw']:>6.3f} "
                      f"| {o['rlv']:>+8.3f} | {o['rel']:>6.3f}", flush=True)
        del pitches
        gc.collect()

    print()
    print("TRANSFER VERDICT — each config vs shipped, on partial|velo, per season")
    names = [c[0] for c in CONFIGS if c[0] != 'shipped']
    print(f"{'config':>10s} " + "".join(f"{yr:>9d}" for yr in sorted(table))
          + f"{'wins':>7s}")
    print('-' * (11 + 9 * len(table) + 7))
    for name in names:
        wins, cells = 0, ''
        for yr in sorted(table):
            s, p = table[yr].get('shipped'), table[yr].get(name)
            if not s or not p:
                cells += f"{'-':>9s}"
                continue
            d = p['partial'] - s['partial']
            wins += 1 if d > 0 else 0
            cells += f"{d:>+9.3f}"
        print(f"{name:>10s} {cells}{wins:>4d}/{len(table)}")
    print()
    print("Positive = beats shipped that season. A config has to win across")
    print("seasons it was never fitted to before it can be called structural;")
    print("winning only in the fitting season means it learned that season.")


if __name__ == '__main__':
    main()
