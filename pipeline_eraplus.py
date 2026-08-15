"""pipeline_eraplus.py — hdERA and hpERA (+ their 100-scale twins).

hdERA (deserved ERA, descriptive): what the pitcher's season deserves on
the ERA scale, luck stripped. Single channel by measurement — every
second channel was swept and rejected (FIP's gain is 95% its HR term,
HR/FB self-r 0.12; freed K/BB/xwOBAcon weights LOSE to packaged xwOBA in
every replicate season).

    hdERA = poolERA + DH_B * z(xwOBA against, shrunk at N0_XW PA)

hpERA (projected ERA, going forward): the Pitcher+ component set plus
role and park, calibrated to future ERA. Weights are the fold-mean OLS
fit on rest-of-season replicates 2021-2026 (one weight set serves both
horizons: next-season transfer cost 0.001). Held-out r: 0.51 next
season, 0.48 rest-of-season at the 60 IP gate — beats SIERA in all 11
replicates.

    hpERA = poolERA + sum_c W_PH[c] * z(channel_c)

Channels (ERA direction, i.e. higher value = more expected runs):
    xw     shrunk xwOBA against          stuff  -stuffScore (as published;
    k      -(shrunk K%)                          its own reliability
    izwh   -izWhiff%                             regression stands in for
    gb     -GB%                                  the measured n0=15)
    loc    +locPlusRaw (already prior-      xrv   +xRV/100 (make_rv_xrv,
           shrunk at n_prior=135 by               batter-positive, frozen
           pipeline_locplus; no double            LG/SCALE constants)
           shrink)                          gs    starter share (gs/g)
                                            park  home park factor

The 100-scale twins mirror wRC+/ERA- construction (a labeled convention,
not a fit): hdERA+ = 200 - 100 * hdERA / poolERA, higher = better,
average 100; likewise hpERA+. Their spread is genuine run-prevention
information (SD ~23 and ~19), landing naturally in wRC+ territory.

Shrinkage constants were measured by scripts/era_shrinkage_sweep.py
(split-half, interior optima in every replicate season); weights and
slopes by scripts/era_weights_final.py; full provenance in
data/_era_final_constants.json and the 2026-08-15 research notes.

Called from stuff_plus/train_stuff.py --inject (needs the fresh
stuffScore, like Pitcher+). Keys survive process_data-only runs via
XRVOE_KEYS carry-over. AAA/ROC rows get None (the pool, park factors,
and calibration are MLB).
"""
import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
PARK_PATH = os.path.join(DATA_DIR, 'era_park_factors.json')

POOL_MIN_OUTS = 90          # 30 IP: z-pool and anchor population
QUAL_OUTS = 180             # 60 IP: percentile pool (site convention)

# measured shrinkage (era_shrinkage_sweep.py; PA-denominated)
N0_XW = 250.0
N0_K = 90.0

DH_B = 0.917                # LOSO slope, 30+ IP display population

# hpERA fold-mean OLS weights (rest-of-season fit, gate 60), ERA direction
W_PH = {'stuff': 0.293, 'loc': 0.135, 'k': 0.101, 'izwh': 0.132,
        'xrv': 0.177, 'gb': 0.186, 'gs': 0.264, 'park': 0.169}

# frozen run-environment constants for make_rv_xrv (the values the weight
# fit used; z-scoring absorbs any drift in the true environment)
XRV_LG, XRV_SCALE = 0.3169, 1.2393

PH_CHANNELS = ('xw', 'stuff', 'loc', 'k', 'izwh', 'gb', 'xrv', 'gs',
               'park')
# NOTE: xw itself is not a hpERA channel (W_PH has no 'xw'): the K/contact
# information arrives through k/izwh/xrv/gb. It is listed here only so the
# z-pool statistics cover every channel one pass computes.


def _ip_outs(ip_str):
    if not ip_str:
        return 0
    s = str(ip_str)
    whole, _, frac = s.partition('.')
    try:
        return int(whole or 0) * 3 + int(frac or 0)
    except ValueError:
        return 0


def _pctl(v, pool):
    if v is None or not pool:
        return None
    below = sum(1 for x in pool if x < v)
    ties = sum(1 for x in pool if x == v)
    return round(100.0 * (below + 0.5 * ties) / len(pool), 1)


def compute_xrv_map(pitches, aaa_teams):
    """(Pitcher, PTeam) -> batter-positive xRV/100 over MLB pitches, plus
    a per-pitcher pooled entry for combined 2TM/3TM rows."""
    from pipeline_sdplus import make_rv_xrv
    from collections import defaultdict
    rv_fn = make_rv_xrv(XRV_LG, XRV_SCALE)
    acc = defaultdict(lambda: [0.0, 0])
    for p in pitches:
        if p.get('PTeam') in aaa_teams:
            continue
        v = rv_fn(p)
        if v is None:
            continue
        a = acc[(p.get('Pitcher'), p.get('PTeam'))]
        a[0] += v
        a[1] += 1
    out = {}
    pooled = defaultdict(lambda: [0.0, 0])
    for (name, team), (s, n) in acc.items():
        if n > 0:
            out[(name, team)] = 100.0 * s / n
        pooled[name][0] += s
        pooled[name][1] += n
    for name, (s, n) in pooled.items():
        if n > 0:
            out[(name, None)] = 100.0 * s / n
    return out


def _channels(row, xrv_map, park, is_combined):
    """Raw channel values in ERA direction, or None where unavailable."""
    ch = {}
    pa = row.get('pa') or 0
    xw = row.get('xwOBA')
    ch['xw'] = None
    ch['k'] = None
    if pa > 0:
        if xw is not None:
            ch['xw'] = (xw * pa + N0_XW * _channels.lg_xw) / (pa + N0_XW)
        kp = row.get('kPct')
        if kp is not None:
            ch['k'] = -((kp * pa + N0_K * _channels.lg_k) / (pa + N0_K))
    st = row.get('stuffScore')
    ch['stuff'] = -st if st is not None else None
    lr = row.get('locPlusRaw')
    ch['loc'] = lr if lr is not None else None
    iz = row.get('izWhiffPct')
    ch['izwh'] = -iz if iz is not None else None
    gb = row.get('gbPct')
    ch['gb'] = -gb if gb is not None else None
    key = (row.get('pitcher'), None if is_combined else row.get('team'))
    ch['xrv'] = xrv_map.get(key)
    g = row.get('g') or 0
    ch['gs'] = ((row.get('gs') or 0) / g) if g > 0 else None
    ch['park'] = park.get(row.get('team'), 100.0) / 100.0
    return ch


def apply_era_plus(rows, pitches, aaa_teams=('ROC', 'AAA'),
                   is_combined_fn=None):
    """Set hdERA / hpERA / hdERAPlus / hpERAPlus (+ _pctl each) in place.
    Returns the constants bundle for metadata, or None if the pool is too
    thin. `pitches` = the MLB+MiLB pitch dicts (sheet schema) for the
    xRV/100 channel; `is_combined_fn` identifies 2TM/3TM rows."""
    aaa = set(aaa_teams)
    if is_combined_fn is None:
        def is_combined_fn(team):
            return isinstance(team, str) and team.endswith('TM')
    try:
        with open(PARK_PATH) as f:
            park = json.load(f)
    except (OSError, json.JSONDecodeError):
        park = {}
        print('  eraplus WARNING: park factors missing, all parks neutral')

    xrv_map = compute_xrv_map(pitches, aaa)

    mlb = [r for r in rows if r.get('team') not in aaa]
    pool_rows = [r for r in mlb
                 if not is_combined_fn(r.get('team'))
                 and _ip_outs(r.get('ip')) >= POOL_MIN_OUTS
                 and r.get('era') is not None]
    if len(pool_rows) < 50:
        for r in rows:
            for k in ('hdERA', 'hpERA', 'hdERAPlus', 'hpERAPlus'):
                r[k] = None
                r[k + '_pctl'] = None
        return None

    # league rates for the shrink targets (PA-weighted over the pool)
    tot_pa = sum(r.get('pa') or 0 for r in pool_rows)
    _channels.lg_xw = sum((r.get('xwOBA') or 0) * (r.get('pa') or 0)
                          for r in pool_rows) / tot_pa
    _channels.lg_k = sum((r.get('kPct') or 0) * (r.get('pa') or 0)
                         for r in pool_rows) / tot_pa

    anchor = sum(r['era'] for r in pool_rows) / len(pool_rows)

    # z statistics per channel over the pool
    raw = {id(r): _channels(r, xrv_map, park, is_combined_fn(r.get('team')))
           for r in mlb}
    mu_sd = {}
    for c in PH_CHANNELS:
        v = [raw[id(r)][c] for r in pool_rows if raw[id(r)][c] is not None]
        if len(v) < 30:
            mu_sd[c] = None
            continue
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
        mu_sd[c] = (m, sd) if sd > 0 else None

    def z(c, val):
        if val is None or mu_sd.get(c) is None:
            return None
        m, sd = mu_sd[c]
        return (val - m) / sd

    for r in rows:
        if r.get('team') in aaa:
            for k in ('hdERA', 'hpERA', 'hdERAPlus', 'hpERAPlus'):
                r[k] = None
            continue
        ch = raw[id(r)]
        zxw = z('xw', ch['xw'])
        dh = (anchor + DH_B * zxw) if zxw is not None else None
        # hpERA only inside its calibrated domain (30+ IP): its pitch-level
        # channels (izWhiff/GB/xRV) are unshrunk, so tiny samples would
        # extrapolate garbage. hdERA's single channel is fully shrunk and
        # safe at any n.
        ph = None
        if _ip_outs(r.get('ip')) >= POOL_MIN_OUTS:
            zs = {c: z(c, ch[c]) for c in W_PH}
            if all(zs[c] is not None for c in W_PH):
                ph = anchor + sum(W_PH[c] * zs[c] for c in W_PH)
        r['hdERA'] = round(dh, 2) if dh is not None else None
        r['hpERA'] = round(ph, 2) if ph is not None else None
        r['hdERAPlus'] = (round(200.0 - 100.0 * dh / anchor)
                          if dh is not None else None)
        r['hpERAPlus'] = (round(200.0 - 100.0 * ph / anchor)
                          if ph is not None else None)

    # percentiles: qualified MLB non-combined pool, every row ranked
    # (site convention). hdERA/hpERA are lower-is-better -> invert.
    for key, invert in (('hdERA', True), ('hpERA', True),
                        ('hdERAPlus', False), ('hpERAPlus', False)):
        pool = [r[key] for r in mlb
                if r.get(key) is not None
                and not is_combined_fn(r.get('team'))
                and _ip_outs(r.get('ip')) >= QUAL_OUTS]
        for r in rows:
            p = _pctl(r.get(key), pool)
            if p is not None and invert:
                p = round(100.0 - p, 1)
            r[key + '_pctl'] = p

    n_dh = sum(1 for r in rows if r.get('hdERA') is not None)
    n_ph = sum(1 for r in rows if r.get('hpERA') is not None)
    print(f'  eraplus: anchor {anchor:.2f} (pool {len(pool_rows)}), '
          f'hdERA {n_dh} rows, hpERA {n_ph} rows')
    from pipeline_locplus import LOC_SCALE_K
    return {'anchor': round(anchor, 3), 'dhB': DH_B, 'weights': W_PH,
            'n0': {'xw': N0_XW, 'k': N0_K}, 'poolMinOuts': POOL_MIN_OUTS,
            # published so window/scratch contexts (NEW-tab cards) can score
            # hdERA/hpERA the way they already score Pitcher+ from its
            # baseline: z-pool stats per channel + shrink targets.
            'muSd': {c: (list(mu_sd[c]) if mu_sd.get(c) else None)
                     for c in PH_CHANNELS},
            'lgXw': round(_channels.lg_xw, 5),
            'lgK': round(_channels.lg_k, 5),
            'locScaleK': LOC_SCALE_K}


def score_scratch_row(row, pitches, g, gs, team, const):
    """Best-effort (hdERA, hpERA) for a WINDOW/SCRATCH pitcher (NEW-tab
    season cards): the same channels, z-scored against the PUBLISHED pool
    stats from `const` (metadata eraPlusConstants). Two documented
    approximations, matching the window-context convention: the loc
    channel arrives as site-scale locPlus and is inverted through
    LOC_SCALE_K (raw_loc_adj is not computed in window context), and
    NEW-tab pitch mixes can include MiLB lines. Returns (None, None)
    where channels are missing; hpERA additionally requires 30+ IP
    (g*outs unknown here, so the caller's box IP gates via `pitches`
    volume: 90+ PA-ending events approximates the domain floor poorly,
    so we gate on the row's pa >= 100 as the closest available proxy)."""
    mu_sd = const.get('muSd') or {}
    anchor = const.get('anchor')
    if anchor is None:
        return None, None

    def z(c, val):
        ms = mu_sd.get(c)
        if val is None or not ms:
            return None
        m, sd = ms
        return (val - m) / sd if sd else None

    pa = row.get('pa') or 0
    zxw = None
    if pa > 0 and row.get('xwOBA') is not None and const.get('lgXw'):
        xw_sh = (row['xwOBA'] * pa + N0_XW * const['lgXw']) / (pa + N0_XW)
        zxw = z('xw', xw_sh)
    dh = (anchor + const.get('dhB', DH_B) * zxw) if zxw is not None else None

    zs = {}
    if pa > 0 and row.get('kPct') is not None and const.get('lgK'):
        k_sh = -((row['kPct'] * pa + N0_K * const['lgK']) / (pa + N0_K))
        zs['k'] = z('k', k_sh)
    st = row.get('stuffScore')
    zs['stuff'] = z('stuff', -st) if st is not None else None
    lp = row.get('locPlus')
    lk = const.get('locScaleK') or 10
    zs['loc'] = ((100.0 - lp) / lk) if lp is not None else None
    iz = row.get('izWhiffPct')
    zs['izwh'] = z('izwh', -iz) if iz is not None else None
    gbp = row.get('gbPct')
    zs['gb'] = z('gb', -gbp) if gbp is not None else None
    if pitches:
        from pipeline_sdplus import make_rv_xrv
        rv_fn = make_rv_xrv(XRV_LG, XRV_SCALE)
        vals = [v for v in (rv_fn(p) for p in pitches) if v is not None]
        if len(vals) >= 50:
            zs['xrv'] = z('xrv', 100.0 * sum(vals) / len(vals))
        else:
            zs['xrv'] = None
    else:
        zs['xrv'] = None
    zs['gs'] = z('gs', (gs or 0) / g) if g else None
    try:
        with open(PARK_PATH) as f:
            park = json.load(f)
    except (OSError, json.JSONDecodeError):
        park = {}
    zs['park'] = z('park', park.get(team, 100.0) / 100.0)

    ph = None
    if pa >= 100 and all(zs.get(c) is not None for c in W_PH):
        ph = anchor + sum(W_PH[c] * zs[c] for c in W_PH)
    return (round(dh, 2) if dh is not None else None,
            round(ph, 2) if ph is not None else None)


def sort_rows_default(rows):
    """Default leaderboard order: hpERA good -> bad, valueless rows last
    (the client renders JSON order until a header is clicked)."""
    rows.sort(key=lambda r: (r.get('hpERA') is None,
                             r.get('hpERA') if r.get('hpERA') is not None
                             else 0.0))
    return rows
