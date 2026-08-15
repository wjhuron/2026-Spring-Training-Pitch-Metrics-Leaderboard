"""Head-to-head of expected-movement model forms, scored the way production
actually runs (fit within the season, on held-out games) plus a gradient-
boosting ceiling so we know how much a linear form leaves on the table.

Protocol: split each season's games by game_pk parity, fit on one half and
score the other, both directions, so every pitch is scored by a model that
never saw its game. Five seasons = five independent replicates, which is the
bar in CLAUDE.md -- not a single-sample argmax.

Objectives per form:
  R2    out-of-sample fit. A DIAGNOSTIC, not the objective: conditioning on
        more of the movement's own causes trivially raises it while shrinking
        the residual toward zero.
  DIST  mean |corr(OE, raw movement)| across pitch types at the rendered unit.
        LOWER is better -- an OE that tracks the column beside it is a
        relabelled duplicate, which is what the shipped model produces.
  YoY   persistence of the residual, pitcher x pitch type, season to season.
"""
import os, sys, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from xmove_audit import PITCH_TYPES, MIN_N

DIR = os.environ.get('XMOVE_DIR', '/private/tmp/claude-501/-Users-wallyhuron-Huronalytics/8aed4efe-0775-4afc-b652-6ddab7be7e7d/scratchpad')
SEASONS = [2021, 2022, 2023, 2024, 2025]
UNIT_MIN = 50
RNG = np.random.default_rng(17)

FORMS = {
    'S1 shipped aa,ext,v': ['aa', 'ext', 'velo'],
    'S2 +spin':            ['aa', 'ext', 'velo', 'spin'],
    'S3 +spin,axis':       ['aa', 'ext', 'velo', 'spin', 'ax_sin', 'ax_cos'],
    'S3b +spin x axis':    ['aa', 'ext', 'velo', 'spin', 'ax_sin', 'ax_cos', 'sv_sin', 'sv_cos'],
    'S4 no arm angle':     ['ext', 'velo', 'spin', 'ax_sin', 'ax_cos', 'sv_sin', 'sv_cos'],
    'S5 +release point':   ['aa', 'ext', 'velo', 'spin', 'ax_sin', 'ax_cos', 'sv_sin',
                            'sv_cos', 'rz', 'rx_s'],
}
GBM_FEATS = ['aa', 'ext', 'velo', 'spin', 'ax_sin', 'ax_cos', 'rz', 'rx_s']


def load_np():
    """Load every season into one dict of numpy arrays (no giant DataFrame copies)."""
    frames = []
    for y in SEASONS:
        d = pd.read_parquet(f'{DIR}/xmove_{y}.parquet',
                            columns=['Pitch Type', 'Throws', 'Pitcher', 'Velocity',
                                     'Spin Rate', 'SpinAxis', 'xIndVrtBrk', 'xHorzBrk',
                                     'Extension', 'ArmAngle', 'RelPosZ', 'RelPosX',
                                     '_game_pk', 'season'])
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    del frames
    d = d[d['Pitch Type'].isin(PITCH_TYPES) & d['Throws'].isin(['L', 'R'])]
    d = d.dropna(subset=['xIndVrtBrk', 'xHorzBrk', 'Velocity', 'Extension',
                         'ArmAngle', 'SpinAxis', 'Spin Rate'])
    s = np.where(d['Throws'].values == 'R', 1.0, -1.0)
    ivb = d['xIndVrtBrk'].values.astype('f8')
    hb_s = d['xHorzBrk'].values.astype('f8') * s
    rtilt = (d['SpinAxis'].values.astype('f8') - 180.0) % 360.0
    th = np.radians(rtilt * s)
    spin = d['Spin Rate'].values.astype('f8')
    velo = d['Velocity'].values.astype('f8')
    spin_v = spin / velo
    A = dict(
        pt=d['Pitch Type'].values, thr=d['Throws'].values,
        pitcher=d['Pitcher'].values, season=d['season'].values.astype('i4'),
        game=d['_game_pk'].values.astype('i8'),
        ivb=ivb, hb_s=hb_s, velo=velo, spin=spin, spin_v=spin_v,
        ext=d['Extension'].values.astype('f8'), aa=d['ArmAngle'].values.astype('f8'),
        rz=d['RelPosZ'].values.astype('f8'), rx_s=d['RelPosX'].values.astype('f8') * s,
        ax_sin=np.sin(th), ax_cos=np.cos(th), ct=np.cos(th), st=np.sin(th),
    )
    A['sv_sin'] = spin_v * A['ax_sin']
    A['sv_cos'] = spin_v * A['ax_cos']
    A['along'] = ivb * A['ct'] + hb_s * A['st']
    A['cross'] = -ivb * A['st'] + hb_s * A['ct']
    # group id = pitch type x hand x season (production refits every season)
    A['gid'] = pd.factorize(pd.Series(d['Pitch Type'].values) + '_' +
                            pd.Series(d['Throws'].values) + '_' +
                            pd.Series(d['season'].values).astype(str))[0]
    A['ptid'] = pd.factorize(pd.Series(d['Pitch Type'].values))[0]
    A['unit'] = pd.factorize(pd.Series(d['Pitcher'].values) + '|' +
                             pd.Series(d['Throws'].values) + '|' +
                             pd.Series(d['Pitch Type'].values) + '|' +
                             pd.Series(d['season'].values).astype(str))[0]
    A['pkey'] = (pd.Series(d['Pitcher'].values) + '|' + pd.Series(d['Throws'].values)
                 + '|' + pd.Series(d['Pitch Type'].values)).values
    return A


def _design(A, feats, idx):
    return np.column_stack([np.ones(len(idx))] + [A[f][idx] for f in feats])


def run_linear(A, feats, polar=False):
    n = len(A['ivb'])
    x_ivb = np.full(n, np.nan); x_hb = np.full(n, np.nan)
    order = np.argsort(A['gid'], kind='stable')
    bounds = np.searchsorted(A['gid'][order], np.arange(A['gid'].max() + 2))
    for gi in range(A['gid'].max() + 1):
        idx = order[bounds[gi]:bounds[gi + 1]]
        if len(idx) == 0:
            continue
        par = A['game'][idx] % 2
        for p in (0, 1):
            tr, te = idx[par == p], idx[par == 1 - p]
            if len(tr) < MIN_N or len(te) == 0:
                continue
            Xt, Xs = _design(A, feats, tr), _design(A, feats, te)
            if polar:
                ba = np.linalg.lstsq(Xt, A['along'][tr], rcond=None)[0]
                bc = np.linalg.lstsq(Xt, A['cross'][tr], rcond=None)[0]
                a, c = Xs @ ba, Xs @ bc
                x_ivb[te] = a * A['ct'][te] - c * A['st'][te]
                x_hb[te] = a * A['st'][te] + c * A['ct'][te]
            else:
                bi = np.linalg.lstsq(Xt, A['ivb'][tr], rcond=None)[0]
                bh = np.linalg.lstsq(Xt, A['hb_s'][tr], rcond=None)[0]
                x_ivb[te] = Xs @ bi
                x_hb[te] = Xs @ bh
    return x_ivb, x_hb


def run_gbm(A):
    from sklearn.ensemble import HistGradientBoostingRegressor
    n = len(A['ivb'])
    x_ivb = np.full(n, np.nan); x_hb = np.full(n, np.nan)
    order = np.argsort(A['gid'], kind='stable')
    bounds = np.searchsorted(A['gid'][order], np.arange(A['gid'].max() + 2))
    X = np.column_stack([A[f] for f in GBM_FEATS])
    for gi in range(A['gid'].max() + 1):
        idx = order[bounds[gi]:bounds[gi + 1]]
        if len(idx) < MIN_N * 4:
            continue
        par = A['game'][idx] % 2
        for p in (0, 1):
            tr, te = idx[par == p], idx[par == 1 - p]
            if len(tr) < MIN_N * 4 or len(te) == 0:
                continue
            for tgt, out in (('ivb', x_ivb), ('hb_s', x_hb)):
                m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08,
                                                  min_samples_leaf=80,
                                                  l2_regularization=1.0, random_state=0)
                m.fit(X[tr], A[tgt][tr])
                out[te] = m.predict(X[te])
    return x_ivb, x_hb


def group_mean(vals, ids, nid, weights=None):
    s = np.bincount(ids, weights=vals, minlength=nid)
    c = np.bincount(ids, minlength=nid)
    with np.errstate(invalid='ignore', divide='ignore'):
        return s / c, c


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() >= 20 else np.nan


def evaluate(name, A, x_ivb, x_hb):
    ok = np.isfinite(x_ivb) & np.isfinite(x_hb)
    ivb_oe = A['ivb'] - x_ivb
    hb_oe = A['hb_s'] - x_hb
    # out-of-sample R^2, pooled within (pitch type x hand x season)
    ni = nh = di = dh = 0.0
    for gi in np.unique(A['gid'][ok]):
        m = ok & (A['gid'] == gi)
        ni += (ivb_oe[m] ** 2).sum(); nh += (hb_oe[m] ** 2).sum()
        di += ((A['ivb'][m] - A['ivb'][m].mean()) ** 2).sum()
        dh += ((A['hb_s'][m] - A['hb_s'][m].mean()) ** 2).sum()
    nid = A['unit'].max() + 1
    uid = np.where(ok, A['unit'], -1)
    keep = uid >= 0
    m_ivb, cnt = group_mean(A['ivb'][keep], uid[keep], nid)
    m_hb, _ = group_mean(A['hb_s'][keep], uid[keep], nid)
    m_io, _ = group_mean(ivb_oe[keep], uid[keep], nid)
    m_ho, _ = group_mean(hb_oe[keep], uid[keep], nid)
    # per-unit metadata (pitch type, season, pitcher key) via first occurrence
    meta = pd.DataFrame({'u': uid[keep], 'ptid': A['ptid'][keep],
                         'season': A['season'][keep], 'pkey': A['pkey'][keep],
                         'pt': A['pt'][keep]}).groupby('u', sort=True).first()
    upt = np.zeros(nid, dtype=int); useason = np.zeros(nid, dtype=int)
    upkey = np.empty(nid, dtype=object)
    ui = meta.index.values
    upt[ui] = meta.ptid.values
    useason[ui] = meta.season.values
    upkey[ui] = meta.pkey.values
    ptnames = dict(zip(meta.ptid.values, meta.pt.values))
    sel = cnt >= UNIT_MIN
    dist_i, dist_h, yoy_i, yoy_h = [], [], [], []
    for p in np.unique(upt[sel]):
        s = sel & (upt == p)
        if s.sum() < 100:
            continue
        dist_i.append(abs(corr(m_io[s], m_ivb[s])))
        dist_h.append(abs(corr(m_ho[s], m_hb[s])))
        yi, yh = [], []
        for y in SEASONS[:-1]:
            a = s & (useason == y); b = s & (useason == y + 1)
            ka = {upkey[u]: u for u in np.flatnonzero(a)}
            kb = {upkey[u]: u for u in np.flatnonzero(b)}
            common = [k for k in ka if k in kb]
            if len(common) >= 30:
                ua = np.array([ka[k] for k in common]); ub = np.array([kb[k] for k in common])
                yi.append(corr(m_io[ua], m_io[ub])); yh.append(corr(m_ho[ua], m_ho[ub]))
        if yi:
            yoy_i.append(np.mean(yi)); yoy_h.append(np.mean(yh))
    return dict(name=name, r2i=1 - ni / di, r2h=1 - nh / dh,
                rmse_i=math.sqrt(ni / ok.sum()), rmse_h=math.sqrt(nh / ok.sum()),
                dist_i=np.mean(dist_i), dist_h=np.mean(dist_h),
                yoy_i=np.mean(yoy_i), yoy_h=np.mean(yoy_h))


if __name__ == '__main__':
    A = load_np()
    print(f"{len(A['ivb']):,} pitches, 5 seasons, cross-fit by game parity within season\n", flush=True)
    res = []
    for name, feats in FORMS.items():
        xi, xh = run_linear(A, feats)
        res.append(evaluate(name, A, xi, xh)); print('  done', name, flush=True)
    xi, xh = run_linear(A, ['spin', 'velo', 'ext', 'aa', 'spin_v'], polar=True)
    res.append(evaluate('S6 polar axis-frame', A, xi, xh)); print('  done polar', flush=True)
    if '--gbm' in sys.argv:
        xi, xh = run_gbm(A)
        res.append(evaluate('S7 GBM (ceiling)', A, xi, xh)); print('  done gbm', flush=True)
    print(f"\n{'form':>21} {'R2 ivb':>7} {'R2 hb':>7} {'RMSEi':>6} {'RMSEh':>6} "
          f"{'DIST i':>7} {'DIST h':>7} {'YoY i':>6} {'YoY h':>6}")
    print('-' * 81)
    for r in res:
        print(f"{r['name']:>21} {r['r2i']:>7.3f} {r['r2h']:>7.3f} {r['rmse_i']:>6.2f} "
              f"{r['rmse_h']:>6.2f} {r['dist_i']:>7.3f} {r['dist_h']:>7.3f} "
              f"{r['yoy_i']:>6.3f} {r['yoy_h']:>6.3f}")
    print('\nDIST = mean |corr(OE, raw movement)| across pitch types; LOWER is better.')
