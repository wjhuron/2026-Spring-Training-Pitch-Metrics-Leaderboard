"""rail_fade_k.py — stabilization constants (k, in pitches) for the season
pitcher card's bubble-rail small-sample fade.

The fade is f = min(1, n_pitches / k) per bubble metric; full color at n = k
means split-half reliability 0.5 (the standard stabilization convention —
that threshold choice is a CONVENTION, the k values themselves are measured).

PROTOCOL — identical to the one that produced the shipped Pitcher+ component
constants (scripts/research/stuff/pitcherplus_search.py, pipeline/pitcherplus.py):
odd/even game split within season, r = Pearson(half A, half B) across
pitcher-seasons, k = mean(min(nA, nB)) * (1 - r) / r, with both halves
holding >= MIN_HALF pitches.

Part A  pooled 2021-2025 + per-season replicates, from data/_pplus_tables.pkl
        (the Pitcher+ build's own half table). Harness validation: the four
        shipped anchors (kPct 398, izWhiffPct 421, xRv100 1046, gbPct 333)
        must reproduce, or nothing else here is trusted.
Part B  low-n panel: halves rebuilt at ~100 / ~200 pitches from the 2023 and
        2025 caches (5 seeds, game-block assignment) — checks Spearman-Brown
        holds down at the fade's actual operating range.
Part C  2026 production-currency replicate from data/all_pitches_rs_cache.pkl
        (sheet rows): official Barrel column, exact InZone, Stuff+/Loc+
        atoms, per-pitch card xRV. Same split rule, keyed on Game Date.

Known definitional caveats (level, not reliability): Part A/B barrels are
the EV/LA recompute (official flag absent from the caches) — Part C anchors
barrel k on the official column; battery bbPct includes intentional walks
where the card's BB% is unintentional-only; battery iz is the rectangle,
Part C InZone is the exact rounded-rect.

Usage: PYTHONHASHSEED=0 python3 scripts/research/cards/rail_fade_k.py
Output: data/_rail_fade_k.json + printed tables.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research', 'misc'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'research', 'stuff'))
import leaderboard_metric_battery as bat            # noqa: E402
from pitcherplus_search import add_xrv              # noqa: E402

TABLES_PKL = os.path.join(ROOT, 'data', '_pplus_tables.pkl')
SHEET_PKL = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
OUT_JSON = os.path.join(ROOT, 'data', '_rail_fade_k.json')

MIN_HALF = 300          # protocol parity (pitcherplus_search MIN_HALF)
MIN_POOL = 80           # min pitcher-seasons in a correlation
SEEDS = (11, 22, 33, 44, 55)
LOW_N_TARGETS = (100, 200)
LOW_N_SEASONS = (2023, 2025)

# rail metric -> _pplus_tables column (Part A/B currency)
METRICS = {
    'kPct': 'kPct', 'bbPct': 'bbPct', 'kbbPct': 'kbbPct',
    'swStrPct': 'whiffPct', 'izWhiffPct': 'izWhiffPct', 'chasePct': 'chasePct',
    'xwOBAcon': 'xwobacon', 'babip': 'babip', 'hardHitPct': 'hardHitPct',
    'barrelPctAgainst': 'barrelPct', 'hrFbPct': 'hrFbPct',
    'gbPct': 'gbPct', 'puPct': 'puPct',
    'fbVelo': 'fbVelo', 'extension': 'extension', 'xRv100': 'xrv100',
}
# shipped anchors the harness must reproduce (pipeline/pitcherplus.py)
ANCHORS = {'kPct': 398.0, 'izWhiffPct': 421.0, 'xRv100': 1046.0, 'gbPct': 333.0}


def k_from_halves(a, b, na, nb):
    """Protocol k: Pearson r over paired halves, k = mean(min(n)) * (1-r)/r."""
    m = a.notna() & b.notna()
    if m.sum() < MIN_POOL:
        return None
    r = float(np.corrcoef(a[m], b[m])[0, 1])
    half_n = float(np.minimum(na[m], nb[m]).mean())
    if not np.isfinite(r) or r <= 0:
        return {'r': r, 'half_n': half_n, 'k': None, 'pool': int(m.sum())}
    return {'r': r, 'half_n': half_n, 'k': half_n * (1 - r) / r,
            'pool': int(m.sum())}


def part_a():
    t = pd.read_pickle(TABLES_PKL)
    A = t[(t['half'] == 'A') & (t['n'] >= MIN_HALF)]
    B = t[(t['half'] == 'B') & (t['n'] >= MIN_HALF)]
    ab = A.merge(B, on=['pid', 'season'], suffixes=('_a', '_b'))
    out = {}
    for rail_key, col in METRICS.items():
        pooled = k_from_halves(ab[col + '_a'], ab[col + '_b'],
                               ab['n_a'], ab['n_b'])
        per_season = {}
        for yr, g in ab.groupby('season'):
            res = k_from_halves(g[col + '_a'], g[col + '_b'],
                                g['n_a'], g['n_b'])
            if res:
                per_season[int(yr)] = res
        out[rail_key] = {'pooled': pooled, 'per_season': per_season}
    return out


def _half_metrics(g):
    """Lean per-half aggregation: rail metrics only (Part B)."""
    out = bat.rate_aggs(g)
    fb = g[g['pitch_type'].isin({'FF', 'SI', 'FA'})]
    out['fbVelo'] = fb['release_speed'].mean() if len(fb) >= 20 else np.nan
    out['extension'] = g['release_extension'].mean()
    v = g['xrv_pitch'].dropna()
    out['xrv100'] = v.sum() / len(g) * 100 if len(g) and len(v) else np.nan
    return out


def part_b():
    """Low-n halves: shuffled game-block assignment, both halves ~n_target."""
    out = {k: {} for k in METRICS}
    for year in LOW_N_SEASONS:
        df = bat.load_season(year)
        df = add_xrv(df, year)
        games = (df[['pitcher', 'game_pk']].drop_duplicates()
                 .groupby('pitcher')['game_pk'].apply(list).to_dict())
        pitch_by_game = df.groupby(['pitcher', 'game_pk']).indices
        for n_target in LOW_N_TARGETS:
            rows_by_seed = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                recs = []
                for pid, gpks in games.items():
                    if len(gpks) < 2:
                        continue
                    order = list(rng.permutation(gpks))
                    halves = {'A': [], 'B': []}
                    n_h = {'A': 0, 'B': 0}
                    for gpk in order:
                        # fill A to target, then B; extras dropped so both
                        # halves sit as close to n_target as game blocks allow
                        tgt = 'A' if n_h['A'] < n_target else (
                            'B' if n_h['B'] < n_target else None)
                        if tgt is None:
                            break
                        idx = pitch_by_game[(pid, gpk)]
                        halves[tgt].append(idx)
                        n_h[tgt] += len(idx)
                    if n_h['A'] < n_target or n_h['B'] < n_target:
                        continue
                    rec = {'pid': pid}
                    for h in ('A', 'B'):
                        sub = df.iloc[np.concatenate(halves[h])]
                        m = _half_metrics(sub)
                        for kk, vv in m.items():
                            rec[f'{kk}_{h}'] = vv
                        rec[f'n_{h}'] = len(sub)
                    recs.append(rec)
                rows_by_seed.append(pd.DataFrame(recs))
            for rail_key, col in METRICS.items():
                col_b = {'fbVelo': 'fbVelo', 'extension': 'extension',
                         'xRv100': 'xrv100'}.get(rail_key, METRICS[rail_key])
                rs, kvals, half_ns, pools = [], [], [], []
                for d in rows_by_seed:
                    if not len(d):
                        continue
                    res = k_from_halves(d[col_b + '_A'], d[col_b + '_B'],
                                        d['n_A'], d['n_B'])
                    if res:
                        rs.append(res['r']); half_ns.append(res['half_n'])
                        pools.append(res['pool'])
                        if res['k'] is not None:
                            kvals.append(res['k'])
                if rs:
                    out[rail_key][f'{year}@{n_target}'] = {
                        'r_mean': float(np.mean(rs)),
                        'half_n': float(np.mean(half_ns)),
                        'k_mean': float(np.mean(kvals)) if kvals else None,
                        'k_spread': ([round(v, 1) for v in kvals]
                                     if kvals else []),
                        'pool': int(np.mean(pools)),
                    }
    return out


# ── Part C: 2026 sheet replicate (production currency) ───────────────────
SHEET_SWING = ('Swinging Strike', 'Foul', 'In Play')
SHEET_K_EV = ('Strikeout', 'Strikeout Double Play', 'Strikeout - DP')
SHEET_HIT_EV = ('Single', 'Double', 'Triple', 'Home Run')


def part_c():
    from cards.pitcher import _compute_pitch_xrv, sf as card_sf   # noqa: E402
    raw = pd.read_pickle(SHEET_PKL)
    df = pd.DataFrame(raw)
    df = df[df['PTeam'] != 'ROC'].copy()
    for c in ('ExitVelo', 'Velocity', 'Extension', 'xwOBA', 'Stuff+', 'Loc+',
              'RunExp'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    d = df['Description'].fillna('')
    ev = df['Event'].fillna('').astype(str)
    df['swing'] = d.isin(SHEET_SWING)
    df['whiff'] = d == 'Swinging Strike'
    iz = df['InZone'].astype(str).isin(('1', 'True', 'TRUE', '1.0'))
    df['iz'] = iz
    df['chase_sw'] = ~iz & df['swing'] & df['PlateX'].astype(str).ne('')
    df['ooz'] = ~iz & df['PlateX'].astype(str).ne('')
    df['bip'] = d == 'In Play'
    df['pa_end'] = ev.str.len() > 0
    df['k'] = ev.isin(SHEET_K_EV) | ev.str.startswith('Strikeout')
    df['bbw'] = ev == 'Walk'
    df['hit'] = ev.isin(SHEET_HIT_EV)
    df['hr'] = ev == 'Home Run'
    bb = df['BBType'].fillna('').astype(str)
    df['gb'] = df['bip'] & (bb == 'ground_ball')
    df['fbb'] = df['bip'] & (bb == 'fly_ball')
    df['pu'] = df['bip'] & (bb == 'popup')
    df['hardhit'] = df['bip'] & (df['ExitVelo'] >= 95)
    df['barrel_official'] = df['Barrel'].astype(str).str.strip() == '6'

    def aggs(g):
        n = len(g)
        sw = g['swing'].sum(); ooz = g['ooz'].sum()
        izsw = (g['iz'] & g['swing']).sum()
        pa = g['pa_end'].sum(); bipn = g['bip'].sum()
        fbpu = g['fbb'].sum() + g['pu'].sum()
        hr = g['hr'].sum()
        babip_den = bipn - hr
        fbp = g[g['Pitch Type'].isin(('FF', 'SI'))]
        # NaN -> '' before to_dict: production feeds _compute_pitch_xrv sheet
        # blanks ('' -> sf None -> skip); raw NaN floats would pass sf() and
        # poison the sum.
        xrv_vals = _compute_pitch_xrv(
            g[['Description', 'xwOBA', 'RunExp']]
            .where(pd.notna(g[['Description', 'xwOBA', 'RunExp']]), '')
            .to_dict('records'))
        return pd.Series({
            'n': n, 'pa': pa,
            'kPct': g['k'].sum() / pa if pa else np.nan,
            'bbPct': g['bbw'].sum() / pa if pa else np.nan,
            'kbbPct': (g['k'].sum() - g['bbw'].sum()) / pa if pa else np.nan,
            'swStrPct': g['whiff'].sum() / sw if sw else np.nan,
            'izWhiffPct': (g['iz'] & g['whiff']).sum() / izsw
                          if izsw else np.nan,
            'chasePct': g['chase_sw'].sum() / ooz if ooz else np.nan,
            'xwOBAcon': g.loc[g['bip'], 'xwOBA'].mean(),
            'babip': ((g['hit'].sum() - hr) / babip_den
                      if babip_den > 0 else np.nan),
            'hardHitPct': g['hardhit'].sum() / bipn if bipn else np.nan,
            'barrelPctAgainst': (g['barrel_official'].sum() / bipn
                                 if bipn else np.nan),
            'hrFbPct': hr / fbpu if fbpu else np.nan,
            'gbPct': g['gb'].sum() / bipn if bipn else np.nan,
            'puPct': g['pu'].sum() / bipn if bipn else np.nan,
            'fbVelo': (fbp['Velocity'].mean() if len(fbp) >= 20 else np.nan),
            'extension': g['Extension'].mean(),
            'xRv100': (sum(xrv_vals) / n * 100 if n and xrv_vals else np.nan),
            'stuffScore': g['Stuff+'].mean(),
            'locPlus': g['Loc+'].mean(),
        })

    # odd/even game split, protocol parity, keyed on Game Date
    gdx = (df[['Pitcher', 'Game Date']].drop_duplicates()
           .sort_values(['Pitcher', 'Game Date']))
    gdx['half'] = np.where(gdx.groupby('Pitcher').cumcount() % 2 == 0,
                           'A', 'B')
    df = df.merge(gdx, on=['Pitcher', 'Game Date'])
    halves = {}
    for h in ('A', 'B'):
        sub = df[df['half'] == h]
        r = sub.groupby('Pitcher', group_keys=False).apply(aggs)
        halves[h] = r
    ab = halves['A'].merge(halves['B'], left_index=True, right_index=True,
                           suffixes=('_a', '_b'))
    ab = ab[(ab['n_a'] >= MIN_HALF) & (ab['n_b'] >= MIN_HALF)]
    out = {}
    keys = list(METRICS) + ['stuffScore', 'locPlus']
    for rail_key in keys:
        res = k_from_halves(ab[rail_key + '_a'], ab[rail_key + '_b'],
                            ab['n_a'], ab['n_b'])
        if res:
            out[rail_key] = res
    return out


def main():
    print('══ Part A: pooled 2021-2025 + per-season (protocol parity) ══')
    a = part_a()
    print(f"{'metric':<18}{'k pooled':>10}{'r':>8}{'half_n':>8}{'pool':>7}"
          f"   per-season k")
    for m, res in a.items():
        p = res['pooled']
        ks = ' '.join(
            f"{yr}:{(v['k'] and round(v['k'])) or '—'}"
            for yr, v in sorted(res['per_season'].items()))
        kp = f"{p['k']:.0f}" if p and p['k'] else '—'
        print(f"{m:<18}{kp:>10}{p['r']:>8.3f}{p['half_n']:>8.0f}"
              f"{p['pool']:>7}   {ks}")
        if m in ANCHORS and p and p['k']:
            drift = abs(p['k'] - ANCHORS[m]) / ANCHORS[m]
            tag = 'OK' if drift < 0.05 else 'DRIFT — DO NOT TRUST THIS RUN'
            print(f"{'':<18}anchor {ANCHORS[m]:.0f} vs {p['k']:.0f} "
                  f"({drift * 100:.1f}%) {tag}")

    print('\n══ Part B: low-n panel (fade operating range) ══')
    b = part_b()
    for m, levels in b.items():
        if levels:
            desc = '  '.join(
                f"{lv}: k={v['k_mean'] and round(v['k_mean']) or '—'} "
                f"(r={v['r_mean']:.3f}, n={v['half_n']:.0f}, "
                f"pool {v['pool']})"
                for lv, v in sorted(levels.items()))
            print(f"{m:<18}{desc}")

    print('\n══ Part C: 2026 sheet replicate (production currency) ══')
    c = part_c()
    for m, res in c.items():
        kk = f"{res['k']:.0f}" if res['k'] else '—'
        print(f"{m:<18}k={kk:>6}  r={res['r']:.3f}  "
              f"half_n={res['half_n']:.0f}  pool={res['pool']}")

    with open(OUT_JSON, 'w') as f:
        json.dump({'part_a': a, 'part_b': b, 'part_c': c,
                   'protocol': 'odd/even game split; k = half_n*(1-r)/r; '
                               'pitches basis; MIN_HALF %d' % MIN_HALF},
                  f, indent=1, default=str)
    print(f'\nsaved {OUT_JSON}')


if __name__ == '__main__':
    main()
