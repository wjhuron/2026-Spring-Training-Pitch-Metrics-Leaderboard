"""Prior-season active spin under WALLY'S pitch tags, not Savant's.

The problem: Savant's active_spin_slider is computed over the pitches SAVANT
calls sliders. Wally re-tags manually, so a class he calls FC may be the
pitches Savant calls SL, and joining on the label name mis-attributes those.

The bridge must be measured WITHIN a season on the SAME pitches. Comparing this
year's tags to last year's Savant arsenal cannot separate a relabel from a
pitcher simply changing his pitch mix -- Baz dropped a slider and added a
sinker between 2025 and 2026, which is not a re-tag at all, and a set-difference
rule pairs them and attaches a 23.5% efficiency to a sinker.

So: join per pitch on PitchID (game_pk_atbat_pitchno) to Savant's own pitch_type
for the SAME season, and take each Wally class's MAJORITY Savant tag. Measured
on 2026: per-pitch label agreement 92.86%, but mean class purity 98.3% -- each
Wally class really is one Savant class under a different name -- and 124 of 1955
classes (6.3%) carry a different majority tag. Prior-season active spin is then
looked up under that majority tag. If the pitcher had no such pitch last season
the lookup correctly returns nothing instead of a wrong number.

Then: does prior-season active spin still buy anything once the labels are
Wally's? Fit on 2026 pitches, 2025 active spin as the prior, release-axis frame,
cross-fit by game parity.
"""
import os, sys, json, math, pickle
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DIR = os.environ.get('XMOVE_DIR', '/private/tmp/claude-501/-Users-wallyhuron-Huronalytics/8aed4efe-0775-4afc-b652-6ddab7be7e7d/scratchpad')
COL = {'fourseam': 'FF', 'sinker': 'SI', 'cutter': 'FC', 'changeup': 'CH',
       'splitter': 'FS', 'curve': 'CU', 'slider': 'SL', 'sweeper': 'ST',
       'slurve': 'SV'}
PITCH_TYPES = ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS']
MIN_GROUP = 400
MIN_CLASS = 50
# The 1-for-1 rename bridge assumes an unmatched Wally class and an unmatched
# Savant class are the same pitches renamed. That is true for real relabels
# (Bradish SL->ST) but the rule also fires on coincidences -- Savant listing a
# class Wally has <50 of, or vice versa -- and then attaches a wildly wrong
# efficiency (a 23.5% "sinker"). Off by default: 80.7% direct coverage with
# values you can defend beats 83.3% with some you cannot.
DIRECT_ONLY = os.environ.get('XMOVE_BRIDGE', 'direct') == 'direct'


def sf(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def tilt_to_axis(rt):
    if not isinstance(rt, str) or ':' not in rt:
        return None
    try:
        h, m = rt.strip().split(':')[:2]
        return (((int(h) % 12) + int(m) / 60.0 + 6.0) * 30.0) % 360.0
    except (ValueError, IndexError):
        return None


def load_cache():
    with open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb') as f:
        raw = pickle.load(f)
    rows = []
    for p in raw:
        if p.get('_source') != 'MLB':
            continue
        pt, thr = p.get('Pitch Type'), p.get('Throws')
        if pt not in PITCH_TYPES or thr not in ('L', 'R'):
            continue
        ivb, hb = sf(p.get('xIndVrtBrk')), sf(p.get('xHorzBrk'))
        velo, spin = sf(p.get('Velocity')), sf(p.get('Spin Rate'))
        ext, aa = sf(p.get('Extension')), sf(p.get('ArmAngle'))
        axis = tilt_to_axis(p.get('RTilt'))
        gp = p.get('_game_pk') or p.get('PitchID')
        if None in (ivb, hb, velo, spin, ext, aa, axis):
            continue
        rows.append((p.get('Pitcher'), thr, pt, ivb, hb, velo, spin, ext, aa,
                     axis, p.get('Game Date'), p.get('PitchID')))
    d = pd.DataFrame(rows, columns=['pitcher', 'thr', 'pt', 'ivb', 'hb', 'velo',
                                    'spin', 'ext', 'aa', 'axis', 'date', 'pid'])
    # no game_pk in this cache -> split by date parity instead, same purpose:
    # the scoring model never sees the pitch's own day
    d['par'] = pd.to_datetime(d.date).dt.dayofyear % 2
    return d


def load_active(year):
    p = f'{DIR}/as_{year}.csv'
    s = pd.read_csv(p, encoding='utf-8-sig')
    out = {}
    for _, r in s.iterrows():
        key = (r.entity_name, r.pitch_hand)
        out[key] = {COL[c]: float(r[f'active_spin_{c}']) for c in COL
                    if pd.notna(r.get(f'active_spin_{c}'))}
    return out


def savant_tag_map():
    """(pitcher, hand, Wally class) -> majority Savant tag, from the SAME
    season's own pitches joined on PitchID."""
    sc = pd.read_pickle(os.path.join(ROOT, 'data', '_statcast2026_full.pkl'))
    sc = sc[['game_pk', 'at_bat_number', 'pitch_number', 'pitch_type']].dropna()
    sc['pitch_type'] = sc.pitch_type.replace({'KC': 'CU', 'FO': 'FS'})
    pid = (sc.game_pk.astype(int).astype(str) + '_' +
           sc.at_bat_number.astype(int).astype(str).str.zfill(3) + '_' +
           sc.pitch_number.astype(int).astype(str).str.zfill(2))
    return dict(zip(pid, sc.pitch_type))


def build_bridge(d, act, tagmap):
    """Map (pitcher, hand, WALLY class) -> prior-season active spin, keyed on
    the class's majority SAVANT tag rather than on its Wally name."""
    d = d.copy()
    d['savant'] = d.pid.map(tagmap)
    bridge = {}
    stats = {'direct': 0, 'renamed': 0, 'unmapped': 0, 'no_savant_tag': 0}
    renames = []
    for (pitcher, thr, pt), g in d.groupby(['pitcher', 'thr', 'pt']):
        if len(g) < MIN_CLASS:
            continue
        vc = g.savant.value_counts()
        if vc.empty:
            stats['no_savant_tag'] += len(g)
            continue
        major, purity = vc.index[0], vc.iloc[0] / vc.sum()
        sv = act.get((pitcher, thr), {})
        val = sv.get(major)
        if val is None:
            stats['unmapped'] += len(g)      # no such pitch last season
            continue
        bridge[(pitcher, thr, pt)] = val
        if major == pt:
            stats['direct'] += len(g)
        else:
            stats['renamed'] += len(g)
            renames.append((pitcher, thr, major, pt, val, purity, len(g)))
    return bridge, stats, renames


BASE = ['ext', 'aa', 'aa2', 'spin', 'velo', 'ax_sin', 'ax_cos', 'ax_sin2',
        'ax_cos2', 'sv_sin', 'sv_cos', 'aa_sin', 'aa_cos']
WITH = BASE + ['active', 'spin_t', 'svt_sin', 'svt_cos']


def prep(d):
    s = np.where(d.thr == 'R', 1.0, -1.0)
    d = d.copy()
    d['hb_s'] = d.hb * s
    th = np.radians(((d.axis - 180.0) % 360.0) * s)
    d['ct'], d['st'] = np.cos(th), np.sin(th)
    d['along'] = d.ivb * d.ct + d.hb_s * d.st
    d['cross'] = -d.ivb * d.st + d.hb_s * d.ct
    d['ax_sin'], d['ax_cos'] = np.sin(th), np.cos(th)
    d['ax_sin2'], d['ax_cos2'] = np.sin(2 * th), np.cos(2 * th)
    sv = d.spin / d.velo
    d['sv_sin'], d['sv_cos'] = sv * d.ax_sin, sv * d.ax_cos
    d['aa_sin'], d['aa_cos'] = d.aa * d.ax_sin, d.aa * d.ax_cos
    d['aa2'] = d.aa ** 2
    d['spin_t'] = d.spin * d.active / 100.0
    svt = d.spin_t / d.velo
    d['svt_sin'], d['svt_cos'] = svt * d.ax_sin, svt * d.ax_cos
    return d


def crossfit_r2(d, feats, target):
    num = den = 0.0
    per = {}
    for (pt, thr), g in d.groupby(['pt', 'thr']):
        if len(g) < MIN_GROUP:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[f].values for f in feats])
        y = g[target].values
        par = g.par.values
        pred = np.full(len(g), np.nan)
        for p in (0, 1):
            tr, te = par == p, par == 1 - p
            if tr.sum() < max(MIN_GROUP // 2, 20 * X.shape[1]) or te.sum() == 0:
                continue
            pred[te] = X[te] @ np.linalg.lstsq(X[tr], y[tr], rcond=1e-8)[0]
        ok = np.isfinite(pred)
        if ok.sum() == 0:
            continue
        a = ((y[ok] - pred[ok]) ** 2).sum()
        b = ((y[ok] - y[ok].mean()) ** 2).sum()
        num += a; den += b
        acc = per.setdefault(pt, [0.0, 0.0]); acc[0] += a; acc[1] += b
    return 1 - num / den, {k: 1 - v[0] / v[1] for k, v in per.items()}


if __name__ == '__main__':
    d = load_cache()
    act = load_active(2025)
    tagmap = savant_tag_map()
    bridge, stats, renames = build_bridge(d, act, tagmap)
    tot = sum(stats.values())
    print(f'{len(d):,} 2026 MLB pitches under Wally tags\n')
    print('active-spin attribution (share of pitches in classes >= 50):')
    for k in ('direct', 'renamed', 'unmapped', 'no_savant_tag'):
        print(f'  {k:>14}: {stats[k]:>8,}  {stats[k]/tot*100:>5.1f}%')
    print(f'\n{len(renames)} classes whose majority Savant tag differs from the '
          f'Wally name (largest first):')
    for r in sorted(renames, key=lambda x: -x[6])[:10]:
        print(f'  {r[0]:<24} {r[1]}  Savant {r[2]} -> Wally {r[3]:<3} '
              f'n={r[6]:>4}  purity {r[5]*100:>5.1f}%  active {r[4]:.1f}%')

    d['active'] = [bridge.get((p, t, pt)) for p, t, pt in
                   zip(d.pitcher, d.thr, d.pt)]
    have = d.active.notna()
    print(f'\n{have.sum():,}/{len(d):,} pitches ({have.mean()*100:.1f}%) carry a '
          f'2025 active-spin value')
    dd = prep(d[have])
    print(f'\n{"target":>7} {"base R2":>9} {"+active spin":>13} {"gain":>7}')
    print('-' * 40)
    res = {}
    for target in ('along', 'cross'):
        b = crossfit_r2(dd, BASE, target)
        w = crossfit_r2(dd, WITH, target)
        res[target] = (b, w)
        print(f'{target:>7} {b[0]:>9.3f} {w[0]:>13.3f} {w[0]-b[0]:>7.3f}')
    print(f'\n{"pt":>4} {"along base":>11} {"along +AS":>10} {"gain":>7} | '
          f'{"cross base":>11} {"cross +AS":>10} {"gain":>7}')
    for pt in PITCH_TYPES:
        ab = res['along'][0][1].get(pt); aw = res['along'][1][1].get(pt)
        cb = res['cross'][0][1].get(pt); cw = res['cross'][1][1].get(pt)
        if ab is None or cb is None:
            continue
        print(f'{pt:>4} {ab:>11.3f} {aw:>10.3f} {aw-ab:>7.3f} | '
              f'{cb:>11.3f} {cw:>10.3f} {cw-cb:>7.3f}')
