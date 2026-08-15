"""Does per-pitch DRAG carry information about the seam residual?

Motivation: seam-shifted wake works by making the ball's wake asymmetric. An
asymmetric wake changes drag as well as lift. So if crossOE (break
perpendicular to the measured release axis) is real physics rather than
tracking noise, pitches with anomalous drag should show anomalous crossOE.
If there is no relationship at all, that is a warning about the metric.

Drag must NOT go into the expectation itself: it is an in-flight outcome, not a
release property, so conditioning on it would weaken the counterfactual exactly
the way conditioning on OTilt would. This is a validator, not a feature.

Everything needed is already in the live feed and already parsed by
Pitcher2026, just not stored: pitchData.endSpeed and coordinates.aY. Speed loss
over the flight gives a drag coefficient up to a constant:
    Cd ~ -ln(endSpeed / startSpeed) / path_length

Drag also rises with air density and with spin, so the ANOMALY is what matters:
residualise Cd on velocity, spin, extension and the game's density factor,
within pitch type and hand, then correlate with the movement residuals.

Usage: python3 scripts/research/xmove/xmove_drag_probe.py [n_games]
"""
import os, sys, json, math, pickle, random
import urllib.request
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from xmove_gray_plot import load, fit_and_score, tilt_to_axis, sf

FEED = 'https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live'


def pull_game(pk):
    rows = []
    try:
        with urllib.request.urlopen(FEED.format(pk=pk), timeout=45) as r:
            j = json.load(r)
    except Exception:
        return rows
    for play in j.get('liveData', {}).get('plays', {}).get('allPlays', []):
        ab = play.get('about', {}).get('atBatIndex')
        for ev in play.get('playEvents', []):
            pd_ = ev.get('pitchData') or {}
            ss, es = pd_.get('startSpeed'), pd_.get('endSpeed')
            if not ss or not es:
                continue
            c = pd_.get('coordinates') or {}
            pid = f"{pk}_{(ab or 0) + 1:03d}_{ev.get('pitchNumber', 0):02d}"
            rows.append(dict(pid=pid, start=float(ss), end=float(es),
                             ext=sf(pd_.get('extension')), aY=sf(c.get('aY')),
                             y0=sf(c.get('y0')), plate_t=sf(pd_.get('plateTime'))))
    return rows


def resid_within(df, ycol, xcols, keycols):
    """OLS residual of ycol on xcols, fit separately within each key group."""
    out = np.full(len(df), np.nan)
    for _, g in df.groupby(keycols):
        if len(g) < 60:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[c].values for c in xcols])
        y = g[ycol].values
        b = np.linalg.lstsq(X, y, rcond=1e-8)[0]
        out[df.index.get_indexer(g.index)] = y - X @ b
    return out


def main(n_games=60):
    d = load()
    xi, xh = fit_and_score(d)
    with open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb') as f:
        raw = pickle.load(f)
    idx = {}
    k = 0
    for p in raw:
        if p.get('_source') != 'MLB':
            continue
        pt, thr = p.get('Pitch Type'), p.get('Throws')
        ivb, hb = sf(p.get('xIndVrtBrk')), sf(p.get('xHorzBrk'))
        velo, spin = sf(p.get('Velocity')), sf(p.get('Spin Rate'))
        ext, aa = sf(p.get('Extension')), sf(p.get('ArmAngle'))
        axis = tilt_to_axis(p.get('RTilt'))
        from xmove_gray_plot import PITCH_COLORS
        if None in (ivb, hb, velo, spin, ext, aa, axis) or thr not in ('L', 'R'):
            continue
        if pt not in PITCH_COLORS:
            continue
        idx[p.get('PitchID')] = k
        k += 1
    assert k == len(d['ivb']), f'index drift {k} vs {len(d["ivb"])}'

    pks = sorted({str(pid).split('_')[0] for pid in idx})
    random.Random(11).shuffle(pks)
    pks = pks[:n_games]
    cache = os.path.join(os.environ.get('XMOVE_DIR', '/tmp'), f'drag_pull_{n_games}.parquet')
    if os.path.exists(cache):
        f = pd.read_parquet(cache)
        print(f'reusing cached pull: {len(f):,} pitches from {len(pks)} games')
    else:
        print(f'pulling {len(pks)} games from the live feed...', flush=True)
        rows = []
        for i, pk in enumerate(pks):
            rows += pull_game(pk)
            if (i + 1) % 15 == 0:
                print(f'  {i+1}/{len(pks)}  ({len(rows):,} pitches)', flush=True)
        f = pd.DataFrame(rows)
        f.to_parquet(cache, index=False)
    f = f[f.pid.isin(idx)].dropna(subset=['ext', 'aY'])
    print(f'{len(f):,} pitches joined to the movement model\n')

    j = np.array([idx[p] for p in f.pid])
    io = d['ivb'][j] - xi[j]
    ho = (d['hb'][j] - xh[j]) * d['s'][j]
    f['along_oe'] = io * d['ct'][j] + ho * d['st'][j]
    f['cross_oe'] = -io * d['st'][j] + ho * d['ct'][j]
    f['pt'] = d['pt'][j]
    f['thr'] = d['thr'][j]
    f['velo'] = d['velo'][j]
    f['spin'] = d['spin'][j]
    f = f[np.isfinite(f.along_oe) & np.isfinite(f.cross_oe)].reset_index(drop=True)

    # drag coefficient up to a constant: speed decay over the tracked path
    path = (f.y0.fillna(50.0) - 17.0 / 12.0)
    f['cd'] = -np.log(f.end / f.start) / path
    f['sv'] = f.spin / f.velo
    # air density per game: drag is proportional to rho, and rho swings ~25%
    # between Coors and a sea-level night. Leaving it out would dump park
    # density straight into the "anomaly" -- and pitchers cluster by home park,
    # so it would contaminate the pitcher-level cross-check specifically.
    with open(os.path.join(ROOT, 'data', 'game_weather_rs.json')) as wf:
        wjs = json.load(wf)
    rho_by_pk = {k: v.get('rho') for k, v in wjs.items()}
    f['rho'] = [rho_by_pk.get(str(p).split('_')[0]) for p in f.pid]
    miss = f.rho.isna().sum()
    if miss:
        print(f'  {miss:,} pitches without a density value; dropped from the '
              f'drag anomaly')
    f = f[f.rho.notna()].reset_index(drop=True)
    f['cd_anom'] = resid_within(f, 'cd', ['velo', 'spin', 'sv', 'ext', 'rho'],
                                ['pt', 'thr'])
    f = f[np.isfinite(f.cd_anom)]
    print(f'{len(f):,} pitches with a drag anomaly\n')
    print(f'mean Cd {f.cd.mean():.5f}   sd {f.cd.std():.5f}   '
          f'anomaly sd {f.cd_anom.std():.5f}')

    print(f"\n{'pt':>4} {'n':>6} {'r(cd_anom, |crossOE|)':>22} "
          f"{'r(cd_anom, alongOE)':>20}")
    print('-' * 54)
    for pt, g in f.groupby('pt'):
        if len(g) < 400:
            continue
        r1 = np.corrcoef(g.cd_anom, g.cross_oe.abs())[0, 1]
        r2 = np.corrcoef(g.cd_anom, g.along_oe)[0, 1]
        print(f'{pt:>4} {len(g):>6} {r1:>22.3f} {r2:>20.3f}')
    print(f"{'ALL':>4} {len(f):>6} "
          f"{np.corrcoef(f.cd_anom, f.cross_oe.abs())[0,1]:>22.3f} "
          f"{np.corrcoef(f.cd_anom, f.along_oe)[0,1]:>20.3f}")
    print('\nSeam-shifted wake makes the wake asymmetric, which should raise BOTH '
          'the\ncross-axis break and drag. A positive r(cd_anom, |crossOE|) is '
          'independent\nphysical corroboration; ~0 would say the seam residual is '
          'not aerodynamic.')

    # Follow-up: drag tracking the ALONG residual is the spin-efficiency
    # signature -- lift and drag both scale with TRANSVERSE spin, so a pitch
    # with more transverse spin than its total implies gets both more Magnus
    # break and more drag. If that is what is happening, drag anomaly should
    # line up with Savant's active spin. It would then be a PER-PITCH proxy for
    # the one quantity Savant only publishes per season.
    try:
        from xmove_activespin_probe import load_active
    except Exception as e:
        print('  (active-spin cross-check skipped:', e, ')')
        return
    act = load_active(years=(2025,))
    act = act.rename(columns={'active': 'active25'})
    f2 = f.copy()
    f2['pitcher'] = d['pitcher'][j][np.isin(np.arange(len(j)), np.arange(len(j)))][:0].tolist() or None
    # rebuild aligned pitcher labels (f was filtered twice)
    jj = np.array([idx[p] for p in f2.pid])
    f2['pitcher'] = d['pitcher'][jj]
    f2['thr'] = d['thr'][jj]
    u = (f2.groupby(['pitcher', 'thr', 'pt'])
           .agg(n=('cd_anom', 'size'), cd=('cd_anom', 'mean'),
                ao=('along_oe', 'mean'), co=('cross_oe', 'mean')).reset_index())
    u = u[u.n >= 20].merge(act, on=['pitcher', 'thr', 'pt'], how='inner')
    print(f'\n{len(u)} pitcher x pitch-type units (>=20 pitches in sample) with '
          f'a 2025 active-spin value')
    if len(u) >= 40:
        print(f'  r(mean drag anomaly, active spin) = '
              f'{np.corrcoef(u.cd, u.active25)[0,1]:+.3f}')
        print(f'  r(mean alongOE,      active spin) = '
              f'{np.corrcoef(u.ao, u.active25)[0,1]:+.3f}')
        for pt, g in u.groupby('pt'):
            if len(g) >= 25:
                print(f'    {pt:>3} n={len(g):>3}  '
                      f'r(drag, active) {np.corrcoef(g.cd, g.active25)[0,1]:+.3f}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
