"""bsr_screen.py — does Bat Speed Responsiveness carry ANY next-season
signal for the quantities Hitter+ ships, beyond the bat-track levels the
priors already use?

CONTEXT. George Lewis (Teddy Should Not Have Won, 2026-08) defines BSR as
the per-hitter slope of bat speed on swing length, estimated within a
half-foot window of the hitter's own mean swing length. His result: BSR
predicts the mechanical bat-speed return on a longer swing, but NOT any
change in wOBA or run value. This screen asks the only question that
matters for this repo: does the SLOPE add next-season predictive power
for BB+ raw or raw_ct beyond (a) the same-season value of the target and
(b) the LEVELS already shipped as bat-track priors (bat speed, fast-swing
rate, swing length). If no, BSR stays an article topic; no full replicate
is built.

THIS IS A SCREEN, NOT AN ADOPTION TEST. Two season pairs only, crude
un-shrunk slopes, no constant sweeps. A pass buys the full battery
(shrinkage swept, contact-depth controls, 2024-2026 replicates); a fail
closes the question.

PRE-REGISTERED PROTOCOL (2026-08-25, written before any result was seen):
  BSR        per hitter-season OLS slope of bat_speed on swing_length,
             swings with bat_speed >= 50 and finite swing_length, within
             0.5 ft of the hitter's own mean swing length (the paper's
             window, taken as a convention for the screen), variables
             centered within hitter. >= 100 windowed swings qualifies.
             Two variants: bsr_raw (as above) and bsr_adj (both variables
             first residualized on pooled-league OLS controls per season:
             plate_z, side-signed plate_x, release_speed).
  Levels     from data/_bt_seasons.json: bs = avg_sweetspot_speed_mph,
             fast = avg_is_sweetspot_speed_high, sl = swing_length_qualified.
  Targets    next-season full-season values, joined on MLBAM id:
             bb  = BB+ raw, the shipped recipe (0.30 * con100 unshrunk +
                   0.70 * ev95_100), battery universe >= 150 non-bunt BIP;
             ct  = raw_ct from pipeline.contact against the season's own
                   cell tables, universe >= 400 CT-eligible swings.
  Pairs      2024 -> 2025 and 2025 -> 2026 (2026 = season to date).
  Analysis   per pair and target: r(BSR_Y, tgt_{Y+1}); partial r
             controlling tgt_Y; partial r controlling
             [tgt_Y, bs_Y, fast_Y, sl_Y] (multivariate residualization).
  DECISION   BSR earns the full replicate only if the multi-control
             partial r has the SAME SIGN in both pairs AND |r| >= 0.10 in
             both, for at least one target. Anything less: recorded as a
             rejection next to the 4th-atom result.
  Diagnostics (report only): odd/even split-half reliability of BSR
             within season (attenuation ceiling), year-to-year r of BSR
             (paper sanity check: ~.5-.6), league mean slope (paper:
             2.88 mph/ft), r(BSR, levels).

Data: data/_bsr_swing_{2024,2025}.pkl (bsr_swing_pull.py),
data/all_pitches_rs_cache.pkl (2026 sheet), data/_statcast{Y}_cache.pkl
(targets), data/_bt_seasons.json (levels + name bridge).

Usage: PYTHONHASHSEED=0 python3 scripts/research/hitter/bsr_screen.py
Output: data/_bsr_screen_results.json + printed tables.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from statcast_hitter_adapter import season_dicts, GUTS  # noqa: E402
from pipeline.contact import (  # noqa: E402
    is_ct_eligible, build_bip_count_offsets, build_contact_cell_weights,
    zone_level_contact_means, shrink_contact_cells, compute_hitter_ct)
from pipeline.sdplus import make_rv_xrv  # noqa: E402

D = lambda n: json.load(open(os.path.join(ROOT, 'data', n)))
SEASONS = (2024, 2025, 2026)
PAIRS = ((2024, 2025), (2025, 2026))
WINDOW_FT = 0.5
MIN_SWINGS = 100
MIN_BIP = 150
MIN_CT_SWINGS = 400
N0_CON = 200
W_EV = 0.70
BUNT_BB = {'bunt_grounder', 'bunt_popup', 'bunt_line_drive'}
SH_EV = {'sac_bunt', 'sac_bunt_double_play'}


def name_bridge():
    bt = D('_bt_seasons.json')
    m = {}
    for y in ('2026', '2025', '2024'):
        for pid, row in bt.get(y, {}).items():
            if row.get('name'):
                m.setdefault(row['name'], pid)
    return m


# ── swing frames (hid, bs, sl, plate_x, plate_z, velo, side) ────────────

def swing_frame(year):
    if year == 2026:
        raw = pd.read_pickle(os.path.join(ROOT, 'data',
                                          'all_pitches_rs_cache.pkl'))
        bridge = name_bridge()
        df = pd.DataFrame(raw)
        df = df[(df['PTeam'] != 'ROC') & (df['BTeam'] != 'ROC')]
        out = pd.DataFrame({
            'hid': df['Batter'].astype(str).map(bridge),
            'bs': pd.to_numeric(df['BatSpeed'], errors='coerce'),
            'sl': pd.to_numeric(df['SwingLength'], errors='coerce'),
            'px': pd.to_numeric(df['PlateX'], errors='coerce'),
            'pz': pd.to_numeric(df['PlateZ'], errors='coerce'),
            'velo': pd.to_numeric(df['Velocity'], errors='coerce'),
            'side': df['Bats'].astype(str),
        })
    else:
        bt = pd.read_pickle(os.path.join(ROOT, 'data',
                                         f'_bsr_swing_{year}.pkl'))
        out = pd.DataFrame({
            'hid': bt['batter'].astype('Int64').astype(str),
            'bs': pd.to_numeric(bt['bat_speed'], errors='coerce'),
            'sl': pd.to_numeric(bt['swing_length'], errors='coerce'),
            'px': pd.to_numeric(bt['plate_x'], errors='coerce'),
            'pz': pd.to_numeric(bt['plate_z'], errors='coerce'),
            'velo': pd.to_numeric(bt['release_speed'], errors='coerce'),
            'side': bt['stand'].astype(str),
        })
    m = out['bs'].notna() & (out['bs'] >= 50) & out['sl'].notna() \
        & out['hid'].notna()
    return out[m].reset_index(drop=True)


def pooled_residuals(sw):
    """Residualize bs and sl on [pz, side-signed px, velo], pooled league
    OLS, complete cases only. Returns a copy with bs_r / sl_r columns."""
    sw = sw.copy()
    sgn = np.where(sw['side'] == 'L', -1.0, 1.0)
    X = np.column_stack([np.ones(len(sw)), sw['pz'], sw['px'] * sgn,
                         sw['velo']])
    ok = np.isfinite(X).all(axis=1)
    for col in ('bs', 'sl'):
        y = sw[col].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        r = np.full(len(sw), np.nan)
        r[ok] = y[ok] - X[ok] @ beta
        sw[col + '_r'] = r
    return sw


def hitter_slope(bs, sl):
    """Windowed within-hitter slope. Returns (slope, n_window) or None."""
    mu = sl.mean()
    w = np.abs(sl - mu) <= WINDOW_FT
    if w.sum() < MIN_SWINGS:
        return None
    x = sl[w] - sl[w].mean()
    y = bs[w] - bs[w].mean()
    if x.std() < 0.03:
        return None
    return float(np.polyfit(x, y, 1)[0]), int(w.sum())


def season_bsr(year):
    sw = pooled_residuals(swing_frame(year))
    rows = {}
    for hid, g in sw.groupby('hid'):
        raw = hitter_slope(g['bs'].to_numpy(float), g['sl'].to_numpy(float))
        if raw is None:
            continue
        rec = {'bsr_raw': raw[0], 'n_sw': raw[1]}
        ga = g.dropna(subset=['bs_r', 'sl_r'])
        adj = hitter_slope(ga['bs_r'].to_numpy(float),
                           ga['sl_r'].to_numpy(float)) if len(ga) else None
        rec['bsr_adj'] = adj[0] if adj else np.nan
        # odd/even split-half of the raw slope (attenuation diagnostic)
        halves = []
        for k in (0, 1):
            h = g.iloc[k::2]
            s = hitter_slope(h['bs'].to_numpy(float), h['sl'].to_numpy(float))
            halves.append(s[0] if s else np.nan)
        rec['half_a'], rec['half_b'] = halves
        rows[hid] = rec
    return rows


# ── targets ─────────────────────────────────────────────────────────────

def bip_frame_cache(year):
    import pickle
    df = pickle.load(open(os.path.join(
        ROOT, 'data', f'_statcast{year}_cache.pkl'), 'rb'))
    if 'game_type' in df.columns:
        df = df[df['game_type'] == 'R']
    ev = df['events'].where(df['events'].astype(str).str.len() > 0)
    bip = ((df['description'] == 'hit_into_play')
           & df['bb_type'].fillna('').ne('')
           & ~df['bb_type'].isin(BUNT_BB)
           & ~ev.isin(SH_EV).fillna(False))
    out = pd.DataFrame({
        'hid': df.loc[bip, 'batter'].astype(int).astype(str),
        'xw': pd.to_numeric(
            df.loc[bip, 'estimated_woba_using_speedangle'], errors='coerce'),
        'ev': pd.to_numeric(df.loc[bip, 'launch_speed'], errors='coerce'),
    })
    return out.dropna(subset=['ev'])


def bip_frame_sheet():
    raw = pd.read_pickle(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'))
    df = pd.DataFrame(raw)
    df = df[(df['PTeam'] != 'ROC') & (df['BTeam'] != 'ROC')]
    bb = df['BBType'].fillna('').astype(str)
    bip = (df['Description'] == 'In Play') & bb.ne('') & ~bb.isin(BUNT_BB)
    bridge = name_bridge()
    out = pd.DataFrame({
        'hid': df.loc[bip, 'Batter'].astype(str).map(bridge),
        'xw': pd.to_numeric(df.loc[bip, 'xwOBA'], errors='coerce'),
        'ev': pd.to_numeric(df.loc[bip, 'ExitVelo'], errors='coerce'),
    })
    return out.dropna(subset=['ev', 'hid'])


def bb_targets(year):
    """dict[hid] -> full-season BB+ raw (unshrunk con arm)."""
    bips = bip_frame_sheet() if year == 2026 else bip_frame_cache(year)
    lg_con = float(bips['xw'].mean())
    groups = {h: g for h, g in bips.groupby('hid') if len(g) >= MIN_BIP}
    if not groups:
        return {}
    lg_ev95 = float(np.average(
        [np.percentile(g['ev'], 95) for g in groups.values()],
        weights=[len(g) for g in groups.values()]))
    out = {}
    for h, g in groups.items():
        xw = g['xw'].to_numpy(float)
        xw = xw[np.isfinite(xw)]
        con100 = 100.0 * xw.mean() / lg_con if len(xw) else np.nan
        ev100 = 100.0 * np.percentile(g['ev'], 95) / lg_ev95
        out[h] = (1 - W_EV) * con100 + W_EV * ev100
    return out


def sheet_pitches_2026():
    raw = pd.read_pickle(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'))
    bridge = name_bridge()
    out = []
    for p in raw:
        if p.get('PTeam') == 'ROC' or p.get('BTeam') == 'ROC':
            continue
        hid = bridge.get(str(p.get('Batter')))
        if hid is None or not p.get('Game Date'):
            continue
        q = dict(p)
        q['Batter'] = str(hid)
        q['BTeam'] = 'X'
        out.append(q)
    return out


def ct_targets(year):
    """dict[hid] -> full-season raw_ct, the ctplus battery's currency."""
    if year == 2026:
        pitches = sheet_pitches_2026()
        md = D('metadata_rs.json')
        G = md.get('gutsConstants') or {}
        lgw, ws = G.get('lgWOBA', 0.313), G.get('wOBAScale', 1.232)
    else:
        pitches = season_dicts(year)
        lgw, ws = GUTS[year]
    swings = [p for p in pitches if is_ct_eligible(p)]
    offsets = build_bip_count_offsets(swings, lgw, ws)
    rv_fn = make_rv_xrv(lgw, ws, offsets)
    cells = shrink_contact_cells(build_contact_cell_weights(swings, rv_fn),
                                 zone_level_contact_means(swings, rv_fn))
    by_h = defaultdict(list)
    for p in swings:
        by_h[p['Batter']].append(p)
    cuts = {(h + '|full', 'X'): ps for h, ps in by_h.items()
            if len(ps) >= MIN_CT_SWINGS}
    raw_ct = compute_hitter_ct(cuts, cells)
    out = {}
    for (key, _), v in raw_ct.items():
        h = key.split('|')[0]
        if v and v.get('raw_ct') is not None:
            out[h] = float(v['raw_ct'])
    return out


# ── stats helpers ───────────────────────────────────────────────────────

def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 25:
        return None, int(m.sum())
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


def partial_r(x, y, ctrls):
    """Partial r of x with y given the control columns (list of arrays)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    C = np.column_stack([np.ones(len(x))] +
                        [np.asarray(c, float) for c in ctrls])
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(C).all(axis=1)
    if m.sum() < 25:
        return None, int(m.sum())
    def resid(v):
        beta, *_ = np.linalg.lstsq(C[m], v[m], rcond=None)
        return v[m] - C[m] @ beta
    r, _ = pearson(resid(x), resid(y))
    return r, int(m.sum())


def main():
    bt = D('_bt_seasons.json')

    print('── BSR per season ──')
    bsr = {}
    for y in SEASONS:
        bsr[y] = season_bsr(y)
        raws = [v['bsr_raw'] for v in bsr[y].values()]
        ha = np.array([v['half_a'] for v in bsr[y].values()], float)
        hb = np.array([v['half_b'] for v in bsr[y].values()], float)
        sh, n_sh = pearson(ha, hb)
        rel = (2 * sh / (1 + sh)) if sh is not None else None
        print(f'  {y}: {len(bsr[y])} hitters, league mean slope '
              f'{np.mean(raws):.3f} mph/ft, split-half r '
              f'{sh if sh is not None else float("nan"):.3f} '
              f'(SB full-length {rel if rel is not None else float("nan"):.3f}, '
              f'n {n_sh})')

    print('\n── year-to-year BSR (paper: .611 / .530) ──')
    yy = {}
    for y0, y1 in PAIRS:
        common = sorted(set(bsr[y0]) & set(bsr[y1]))
        r, n = pearson([bsr[y0][h]['bsr_raw'] for h in common],
                       [bsr[y1][h]['bsr_raw'] for h in common])
        yy[f'{y0}-{y1}'] = {'r': r, 'n': n}
        print(f'  {y0}->{y1}: r {r:.3f} (n {n})')

    print('\n── targets ──')
    bb = {y: bb_targets(y) for y in SEASONS}
    ct = {y: ct_targets(y) for y in SEASONS}
    for y in SEASONS:
        print(f'  {y}: bb {len(bb[y])} hitters, ct {len(ct[y])} hitters')

    def levels(y, h):
        row = bt.get(str(y), {}).get(h)
        if not row:
            return None
        try:
            return (float(row['avg_sweetspot_speed_mph']),
                    float(row['avg_is_sweetspot_speed_high']),
                    float(row['swing_length_qualified']))
        except (KeyError, TypeError, ValueError):
            return None

    results = {'yy': yy, 'tables': {}}
    print('\n── the screen: BSR_Y vs target_{Y+1} ──')
    for tname, tgt in (('bb', bb), ('ct', ct)):
        for y0, y1 in PAIRS:
            pool = []
            for h, rec in bsr[y0].items():
                if h not in tgt[y0] or h not in tgt[y1]:
                    continue
                lv = levels(y0, h)
                if lv is None:
                    continue
                pool.append({'hid': h, 'bsr_raw': rec['bsr_raw'],
                             'bsr_adj': rec['bsr_adj'],
                             'tgt0': tgt[y0][h], 'tgt1': tgt[y1][h],
                             'bs': lv[0], 'fast': lv[1], 'sl': lv[2]})
            t = pd.DataFrame(pool)
            key = f'{tname}_{y0}_{y1}'
            if len(t) < 40:
                print(f'  {key}: pool {len(t)} too thin — skipped')
                results['tables'][key] = {'n': len(t)}
                continue
            row = {'n': len(t)}
            for var in ('bsr_raw', 'bsr_adj'):
                r0, _ = pearson(t[var], t['tgt1'])
                r1, _ = partial_r(t[var], t['tgt1'], [t['tgt0']])
                r2, n2 = partial_r(t[var], t['tgt1'],
                                   [t['tgt0'], t['bs'], t['fast'], t['sl']])
                row[var] = {'r': r0, 'partial_tgt0': r1,
                            'partial_full': r2, 'n_full': n2}
                print(f'  {key} {var}: r {r0:+.3f}  '
                      f'| tgt0 {r1:+.3f}  | tgt0+levels {r2:+.3f}  '
                      f'(n {n2})')
            results['tables'][key] = row

    # decision, mechanical
    print('\n── decision (pre-registered) ──')
    verdicts = {}
    for tname in ('bb', 'ct'):
        for var in ('bsr_raw', 'bsr_adj'):
            vals = []
            for y0, y1 in PAIRS:
                row = results['tables'].get(f'{tname}_{y0}_{y1}', {})
                v = row.get(var, {}).get('partial_full')
                vals.append(v)
            ok = (all(v is not None for v in vals)
                  and all(abs(v) >= 0.10 for v in vals)
                  and len({np.sign(v) for v in vals}) == 1)
            verdicts[f'{tname}_{var}'] = {'partials': vals, 'pass': bool(ok)}
            print(f'  {tname} {var}: partials '
                  f'{[None if v is None else round(v, 3) for v in vals]} '
                  f'-> {"PASS" if ok else "fail"}')
    results['verdicts'] = verdicts
    results['any_pass'] = any(v['pass'] for v in verdicts.values())
    print(f'\n  ANY PASS: {results["any_pass"]}')

    out = os.path.join(ROOT, 'data', '_bsr_screen_results.json')
    tmp = out + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=1, default=float)
    os.replace(tmp, out)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
