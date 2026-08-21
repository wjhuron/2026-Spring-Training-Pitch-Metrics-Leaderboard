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

Shrinkage constants were measured by scripts/research/era/era_shrinkage_sweep.py
(split-half, interior optima in every replicate season); weights and
slopes by scripts/research/era/era_weights_final.py; full provenance in
data/era_final_constants.json and the 2026-08-15 research notes.

Called from stuff_plus/train_stuff.py --inject (needs the fresh
stuffScore, like Pitcher+). Keys survive process_data-only runs via
XRVOE_KEYS carry-over.

ROC/AAA ROWS SCORE hpERA AND NOT hdERA (2026-08-19). They are scored
against the MLB pool and never enter it -- no league rate, no z statistic,
no anchor -- the same translation framing Stuff+, Loc+ and xRVOE already
use for Rochester. The split is measured, not assumed: on ~800 pitcher-
seasons appearing at both levels in the same season, 2023-2025
(scripts/research/era/aaa_level_correction.py), the within-pitcher shift
is +0.077 ERA for hpERA and +0.765 for hdERA. hpERA survives because its
channels cancel and hdERA does not because it is nearly pure xwOBA. Their
home park is NEUTRAL: Savant publishes no minor-league park factors, so
the park channel z-scores a flat 1.00 for every ROC row. Neutral now
stands by MEASUREMENT, not just data absence (2026-08-21,
scripts/research/era/aaa_park_channel_validation.py): BA 2025 AAA park
factors were tested against next-season road xwOBA on 2023-2026 AAA
pitcher-seasons, and neither the naive forward channel nor the
retrodictive xrv correction beat the neutral baseline in the IL-only
decision test (park-partial sign flips across year-pairs). Note the
direction trap recorded there: for a translated arm the fitted forward
park weight points the WRONG way, so if factors ever improve, re-test
the retrodictive form, never just activate the channel.
"""
import json
import os

from pipeline.utils import DATA_DIR, TEAM_ABBREV_TO_ID, MLB_TEAMS
PARK_PATH = os.path.join(DATA_DIR, 'park_factors.json')


def _load_park(season):
    """{team abbrev -> runs park factor, 100 = neutral} for `season`.

    Reads data/park_factors.json — the SAME Savant file the hpERA weights
    were fit on (scripts/research/era/era_weights_final.py, through
    era_estimator_screen.park_exposure) — and resolves it through
    TEAM_ABBREV_TO_ID.

    The file is keyed by NUMERIC MLB club id on purpose. An abbreviation
    key silently misses whenever two sources spell a club differently, and
    that is exactly what happened: data/era_park_factors.json was a
    hand-copied 2026 snapshot keyed AZ/KC/SD/SF/TB with no Athletics row
    at all, while the leaderboard rows read ARI/KCR/SDP/SFG/TBR/ATH. Six
    clubs, 167 pitcher rows, scored hpERA against a neutral park from
    2026-08-15 to 2026-08-19. A club id cannot be spelled two ways.

    A club that fails to resolve is a bug, not a neutral park, so it is
    announced. Multi-team labels (2TM..10TM) and ROC are not franchises
    and stay neutral in silence.
    """
    try:
        with open(PARK_PATH) as f:
            allpf = json.load(f)
    except (OSError, json.JSONDecodeError):
        print('  eraplus WARNING: data/park_factors.json missing — '
              'ALL PARKS NEUTRAL. Rebuild with '
              'scripts/builders/park_factors_pull.py')
        return {}
    key = str(season)
    if key not in allpf:
        avail = sorted(k for k in allpf if k.isdigit())
        if not avail:
            print('  eraplus WARNING: park_factors.json holds no season — '
                  'ALL PARKS NEUTRAL')
            return {}
        key = avail[-1]
        print(f'  eraplus WARNING: no park factors for {season}, '
              f'falling back to {key}')
    byid = allpf[key]
    park = {}
    missing = []
    for abbr in MLB_TEAMS:
        tid = TEAM_ABBREV_TO_ID.get(abbr)
        pf = byid.get(str(tid)) if tid is not None else None
        if pf is None:
            if abbr != 'WBC':
                missing.append(abbr)
            continue
        park[abbr] = pf
    if missing:
        print(f'  eraplus WARNING: no {key} park factor for '
              f'{", ".join(sorted(missing))} — those rows score NEUTRAL. '
              f'Rebuild with scripts/builders/park_factors_pull.py')
    return park

POOL_MIN_OUTS = 90          # 30 IP: z-pool and anchor population
QUAL_OUTS = 180             # 60 IP: percentile pool (site convention)

# measured shrinkage (era_shrinkage_sweep.py; PA-denominated)
N0_XW = 250.0
N0_K = 90.0
# measured 2026-08-15 (same split-half sweep, interior optima 6/6 seasons)
# so hpERA can score EVERY pitcher, SIERA-style, without small-sample
# extrapolation: below these samples the channels shrink to league and
# hpERA pulls to the anchor instead of printing garbage.
N0_IZWH = 130.0        # in-zone swings (plateau 110-170)
N0_GB = 55.0           # BIP
N0_XRV = 800.0         # pitches (flat 500-1500)
IZSW_PER_PITCH = 0.33  # league iz-swings per pitch (.32-.34, 2021-2026);
                       # rows carry izWhiffPct but not the iz-swing count,
                       # so the shrink denominator is count * this ratio

DH_B = 0.917                # LOSO slope, 30+ IP display population

# hpERA fold-mean OLS weights (rest-of-season fit, gate 60), ERA direction.
# REFIT 2026-08-15 on the production-consistent shrinkage (izwh n0=130,
# gb 55, xrv 800): fitting on differently-shrunk inputs cost ~.02 held-out
# r; refit restores ROS .482 / NEXT .512.
W_PH = {'stuff': 0.297, 'loc': 0.136, 'k': 0.088, 'izwh': 0.117,
        'xrv': 0.139, 'gb': 0.162, 'gs': 0.277, 'park': 0.168}

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


from pipeline.utils import _pctl  # single-homed percentile convention


def compute_xrv_map(pitches, aaa_teams, roc_pitches=None):
    """(Pitcher, PTeam) -> batter-positive xRV/100, plus a per-pitcher
    pooled entry for combined 2TM/3TM rows.

    `roc_pitches` adds Triple-A entries so a ROC row can score the xRV
    channel. They are kept OUT of the pooled entry on purpose: the pooled
    value exists only to serve combined 2TM/3TM rows, and a traded pitcher
    with a Rochester stint would otherwise have his MLB pooled xRV diluted
    by minor-league pitches. Callers must hand in ROC pitches whose RunExp
    is ALREADY in MLB currency (train_stuff.py rescales them in place via
    compute_runexp_scale before inject); a MiLB-denominated RunExp would
    run about 1.2x hot and read as a worse pitcher.
    """
    from pipeline.sdplus import make_rv_xrv
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
            out[(name, team)] = (100.0 * s / n, n)
        pooled[name][0] += s
        pooled[name][1] += n
    for name, (s, n) in pooled.items():
        if n > 0:
            out[(name, None)] = (100.0 * s / n, n)
    if roc_pitches:
        roc = defaultdict(lambda: [0.0, 0])
        for p in roc_pitches:
            v = rv_fn(p)
            if v is None:
                continue
            a = roc[(p.get('Pitcher'), p.get('PTeam'))]
            a[0] += v
            a[1] += 1
        for key, (s, n) in roc.items():
            if n > 0:
                out[key] = (100.0 * s / n, n)
    return out


def combined_park_map(rows, park, aaa_teams, is_combined_fn):
    """{id(row) -> home park factor} for every combined 2TM/3TM/... row,
    IP-weighted over the pitcher's own MLB stint rows.

    A combined row carries a LABEL, not a franchise, so park.get('2TM') has
    always fallen through to a neutral 100 -- and a traded pitcher did not
    pitch in a neutral park. Measured on the 2026 board: 55 of 93 combined
    rows move at least 0.05 ERA once this is applied, up to 0.523.

    WEIGHTED BY INNINGS, which is a deliberate divergence from the research
    harness. era_estimator_screen.park_exposure takes an UNWEIGHTED mean
    over the pitcher's clubs, and the two disagree by a mean of 0.051 ERA
    and up to 0.278 on the 2026 board. Senzatela threw 52.3 innings at Coors
    and 4.7 at Milwaukee: unweighted calls that a 111 park, innings-weighted
    calls it 125. The weighted figure is the one that describes where he
    actually pitched. The fitted weight W_PH['park'] = 0.168 was estimated
    against the unweighted exposure, so it is if anything attenuated by the
    noisier measure -- the same direction as the abbreviation fix, and a
    refit belongs in the next replicate-validated battery.

    ROC/AAA stints are excluded: they have no MLB park factor and would
    otherwise pull a traded pitcher's exposure toward neutral.
    """
    from collections import defaultdict
    stints = defaultdict(list)
    for r in rows:
        t = r.get('team')
        if t in aaa_teams or is_combined_fn(t) or t not in park:
            continue
        key = r.get('mlbId') or r.get('pitcher')
        if key is None:
            continue
        o = _ip_outs(r.get('ip'))
        if o > 0:
            stints[key].append((park[t], o))
    out, unresolved = {}, []
    for r in rows:
        if not is_combined_fn(r.get('team')):
            continue
        sr = stints.get(r.get('mlbId') or r.get('pitcher')) or []
        tot = sum(o for _, o in sr)
        if tot > 0:
            out[id(r)] = sum(pf * o for pf, o in sr) / tot
        else:
            unresolved.append(r.get('pitcher'))
    if unresolved:
        # Neutral is a real park factor, so a silent fallback would look
        # like a measurement. Say which rows took it.
        print(f'  eraplus WARNING: {len(unresolved)} combined rows have no '
              f'resolvable MLB stint and score a NEUTRAL park: '
              f'{", ".join(sorted(unresolved)[:6])}'
              + (' ...' if len(unresolved) > 6 else ''))
    return out


def _channels(row, xrv_map, park, is_combined, combined_park=None):
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
    if iz is not None:
        n_izsw = (row.get('count') or 0) * IZSW_PER_PITCH
        ch['izwh'] = -((iz * n_izsw + N0_IZWH * _channels.lg_izwh)
                       / (n_izsw + N0_IZWH))
    else:
        ch['izwh'] = None
    gb = row.get('gbPct')
    if gb is not None:
        n_bip = row.get('nBip') or 0
        ch['gb'] = -((gb * n_bip + N0_GB * _channels.lg_gb)
                     / (n_bip + N0_GB))
    else:
        ch['gb'] = None
    key = (row.get('pitcher'), None if is_combined else row.get('team'))
    xr = xrv_map.get(key)
    if xr is not None:
        xv, xn = xr
        ch['xrv'] = (xv * xn + N0_XRV * _channels.lg_xrv) / (xn + N0_XRV)
    else:
        ch['xrv'] = None
    g = row.get('g') or 0
    ch['gs'] = ((row.get('gs') or 0) / g) if g > 0 else None
    if is_combined and combined_park is not None:
        pf = combined_park.get(id(row))
        ch['park'] = (pf if pf is not None else 100.0) / 100.0
    else:
        ch['park'] = park.get(row.get('team'), 100.0) / 100.0
    return ch


def apply_era_plus(rows, pitches, aaa_teams=('ROC', 'AAA'),
                   is_combined_fn=None, season=None, roc_pitches=None):
    """Set hdERA / hpERA / hdERAPlus / hpERAPlus (+ _pctl each) in place.
    Returns the constants bundle for metadata, or None if the pool is too
    thin. `pitches` = the MLB+MiLB pitch dicts (sheet schema) for the
    xRV/100 channel; `is_combined_fn` identifies 2TM/3TM rows."""
    aaa = set(aaa_teams)
    if is_combined_fn is None:
        def is_combined_fn(team):
            return isinstance(team, str) and team.endswith('TM')
    if season is None:
        from datetime import datetime as _dt
        season = _dt.now().year
    park = _load_park(season)

    xrv_map = compute_xrv_map(pitches, aaa, roc_pitches)

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

    # league rates for the shrink targets (denominator-weighted over the pool)
    tot_pa = sum(r.get('pa') or 0 for r in pool_rows)
    _channels.lg_xw = sum((r.get('xwOBA') or 0) * (r.get('pa') or 0)
                          for r in pool_rows) / tot_pa
    _channels.lg_k = sum((r.get('kPct') or 0) * (r.get('pa') or 0)
                         for r in pool_rows) / tot_pa
    _tc = sum(r.get('count') or 0 for r in pool_rows
              if r.get('izWhiffPct') is not None)
    _channels.lg_izwh = (sum((r['izWhiffPct']) * (r.get('count') or 0)
                             for r in pool_rows
                             if r.get('izWhiffPct') is not None) / _tc
                         if _tc else 0.19)
    _tb = sum(r.get('nBip') or 0 for r in pool_rows
              if r.get('gbPct') is not None)
    _channels.lg_gb = (sum((r['gbPct']) * (r.get('nBip') or 0)
                           for r in pool_rows
                           if r.get('gbPct') is not None) / _tb
                       if _tb else 0.42)
    _xs = _xn = 0.0
    for r in pool_rows:
        xr = xrv_map.get((r.get('pitcher'), r.get('team')))
        if xr is not None:
            _xs += xr[0] * xr[1]
            _xn += xr[1]
    _channels.lg_xrv = (_xs / _xn) if _xn else 0.0

    anchor = sum(r['era'] for r in pool_rows) / len(pool_rows)

    # z statistics per channel over the pool
    # ROC rows are SCORED against the MLB pool, never in it: they shape no
    # league rate, no z statistic and no anchor. Same translation framing
    # Stuff+, Loc+ and xRVOE already use for Rochester.
    scored = mlb + [r for r in rows if r.get('team') in aaa]
    cpark = combined_park_map(rows, park, aaa, is_combined_fn)
    raw = {id(r): _channels(r, xrv_map, park, is_combined_fn(r.get('team')),
                            cpark)
           for r in scored}
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
        is_aaa = r.get('team') in aaa
        ch = raw.get(id(r))
        if ch is None:
            for k in ('hdERA', 'hpERA', 'hdERAPlus', 'hpERAPlus'):
                r[k] = None
            continue
        zxw = z('xw', ch['xw'])
        dh = (anchor + DH_B * zxw) if zxw is not None else None
        # Every pitcher scores (SIERA convention, per Wally 2026-08-15):
        # all channels are now shrunk at measured constants (izWhiff n0=130
        # iz-swings, GB n0=55 BIP, xRV n0=800 pitches), so tiny samples
        # pull toward league/anchor instead of extrapolating outside the
        # calibrated domain. Qualification stays a render-time coloring
        # gate, exactly like SIERA.
        ph = None
        zs = {c: z(c, ch[c]) for c in W_PH}
        if all(zs[c] is not None for c in W_PH):
            ph = anchor + sum(W_PH[c] * zs[c] for c in W_PH)
        # hpERA ships for ROC; hdERA does NOT. Measured on ~800 paired
        # pitcher-seasons at both levels, 2023-2025
        # (scripts/research/era/aaa_level_correction.py):
        #
        #   hpERA   +0.077 ERA   the composite is level-neutral because its
        #                        channels cancel -- Triple-A flatters the
        #                        four outcome channels by +0.262 and its
        #                        stuff and location give back -0.185
        #   hdERA   +0.765 ERA   nearly pure xwOBA, so nothing offsets it
        #
        # and hdERA's honest correction is a REGRESSION, not a shift:
        # MLB = 3.630 + 0.226 * AAA, r = 0.209. That maps the whole
        # Triple-A range (1.58-5.93) into 3.99-4.97, so a corrected column
        # would read 4.2/4.3/4.4 down the page and look informative while
        # measuring the anchor. Blank is more honest than near-constant.
        if is_aaa:
            r['hdERA'] = None
            r['hdERAPlus'] = None
        else:
            r['hdERA'] = round(dh, 2) if dh is not None else None
            r['hdERAPlus'] = (round(200.0 - 100.0 * dh / anchor)
                              if dh is not None else None)
        r['hpERA'] = round(ph, 2) if ph is not None else None
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
    n_roc = sum(1 for r in rows
                if r.get('team') in aaa and r.get('hpERA') is not None)
    if roc_pitches and not n_roc:
        print('  eraplus WARNING: roc_pitches supplied but 0 ROC rows '
              'scored hpERA — check the xRV channel keying (Pitcher, PTeam)')
    print(f'  eraplus: anchor {anchor:.2f} (pool {len(pool_rows)}), '
          f'hdERA {n_dh} rows, hpERA {n_ph} rows ({n_roc} of them ROC/AAA)')
    from pipeline.locplus import LOC_SCALE_K
    return {'anchor': round(anchor, 3), 'dhB': DH_B, 'weights': W_PH,
            'n0': {'xw': N0_XW, 'k': N0_K}, 'poolMinOuts': POOL_MIN_OUTS,
            # published so window/scratch contexts (NEW-tab cards) can score
            # hdERA/hpERA the way they already score Pitcher+ from its
            # baseline: z-pool stats per channel + shrink targets.
            'muSd': {c: (list(mu_sd[c]) if mu_sd.get(c) else None)
                     for c in PH_CHANNELS},
            'lgXw': round(_channels.lg_xw, 5),
            'lgK': round(_channels.lg_k, 5),
            'lgIzwh': round(_channels.lg_izwh, 5),
            'lgGb': round(_channels.lg_gb, 5),
            'lgXrv': round(_channels.lg_xrv, 5),
            'locScaleK': LOC_SCALE_K}


def score_scratch_row(row, pitches, g, gs, team, const, season=None):
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
    if iz is not None and const.get('lgIzwh') is not None:
        _niz = (row.get('count') or 0) * IZSW_PER_PITCH
        zs['izwh'] = z('izwh', -((iz * _niz + N0_IZWH * const['lgIzwh'])
                                 / (_niz + N0_IZWH)))
    else:
        zs['izwh'] = None
    gbp = row.get('gbPct')
    if gbp is not None and const.get('lgGb') is not None:
        _nbip = row.get('nBip') or 0
        zs['gb'] = z('gb', -((gbp * _nbip + N0_GB * const['lgGb'])
                             / (_nbip + N0_GB)))
    else:
        zs['gb'] = None
    if pitches and const.get('lgXrv') is not None:
        from pipeline.sdplus import make_rv_xrv
        rv_fn = make_rv_xrv(XRV_LG, XRV_SCALE)
        vals = [v for v in (rv_fn(p) for p in pitches) if v is not None]
        if vals:
            _xv = 100.0 * sum(vals) / len(vals)
            zs['xrv'] = z('xrv', (_xv * len(vals) + N0_XRV * const['lgXrv'])
                          / (len(vals) + N0_XRV))
        else:
            zs['xrv'] = None
    else:
        zs['xrv'] = None
    zs['gs'] = z('gs', (gs or 0) / g) if g else None
    if season is None:
        from datetime import datetime as _dt
        season = _dt.now().year
    zs['park'] = z('park', _load_park(season).get(team, 100.0) / 100.0)

    ph = None
    if all(zs.get(c) is not None for c in W_PH):
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
