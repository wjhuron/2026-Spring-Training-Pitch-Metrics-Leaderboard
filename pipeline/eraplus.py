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

    hpERA = poolERA + sum_c W_PH[c] * z(channel_c) + W_LHP * (lhp - poolLHPshare)

Channels (ERA direction, i.e. higher value = more expected runs).
NOTE (corrected 2026-08-27): xw feeds hdERA only — W_PH carries no xw
term; the eight hpERA channels are the rest of this table.
    xw     shrunk xwOBA against          stuff  -stuffScore (as published;
    k      -(shrunk K%)                          its own reliability
    izwh   -izWhiff%                             regression stands in for
    gb     -GB%                                  the measured n0=15)
    loc    +locPlusRaw (UNSHRUNK raw       xrv   +xRV/100 (make_rv_xrv,
           ExpRV mean; Loc+ has had no            batter-positive, frozen
           pitcher prior since 2026-07-18,        LG/SCALE constants)
           and W_PH was fit on this input)  gs    starter share (gs/g)
                                            park  home park factor
    lhp    left-handed pitcher indicator, RAW and centered at the pool's
           LHP share (2026-09-05): without it LHP beat the forecast by
           0.11-0.20 ERA in 17 of 22 replicates, because the stuff channel
           grades their physics ~0.7 SD worse while their outcomes match
           RHP and Loc+ offsets only part of that. A calibration term for
           the FORECAST; deliberately not a Stuff+ feature (Loc+ already
           embeds the LHP outcome premium through its per-hand-pair
           surfaces, so it would be counted twice here).

The 100-scale twins mirror wRC+/ERA- construction (a labeled convention,
not a fit): hdERA+ = 200 - 100 * hdERA / poolERA, higher = better,
average 100; likewise hpERA+. Their spread is genuine run-prevention
information (SD ~23 and ~19), landing naturally in wRC+ territory.

Shrinkage constants were measured by scripts/research/era/era_shrinkage_sweep.py
(split-half, interior optima in every replicate season); weights and
slopes by scripts/research/era/era_weights_final.py; full provenance in
data/era_final_constants.json and the 2026-08-15 research notes. The
2026-09-05 hand refit: scripts/research/era/era_hand_weight_refit.py on
replicates rebuilt by era_targets_build.py (MLB Stats API lines), the
battery/cmdloc/xrv builders and era_stuff_loso_v2.py; result in
data/_era_hand_refit.json.

Called from stuff_plus/train_stuff.py --inject (needs the fresh
stuffScore, like Pitcher+). Keys survive process_data-only runs via
XRVOE_KEYS carry-over.

hWAR (2026-09-05): a deserved pitcher WAR on hdERA. One pitcher-season:

    RA9_d   = hdERA + (lgRA9 - lgERA)                 earned runs -> runs, the league gap (fWAR does this to FIP)
    RA9_dp  = RA9_d - WAR_PARK_PASS * (exposure - 1) * lgRA9 + shift
              exposure = (PF_home/100 + 1)/2 (half the games at home; combined rows resolve
              through combined_park_map); shift recenters the innings-weighted MLB mean to
              lgRA9, so runs above average sum to zero across the league (the hdERA anchor is
              the 30-IP pool, whose mean is not the league's)
    RAA     = (lgRA9 - RA9_dp) * IP / 9
    hWAR    = RAA / RPW + REPL * IP / 9
    RPW     = 4 r / (2 r)^WAR_PYTH_EXP with r = lgRA9, one season constant (the per-pitcher
              run-environment form is coded behind WAR_DYNAMIC_RPW and held off by decision).
              Team records read 10-11.7 (n 30, curvature)
    hWAR_se = (DH_B / sd_pool) x shrink x WAR_XW_PA_SD / sqrt(PA) x IP/9 / RPW   sampling error bar,
              on the runs scale hdERA itself uses (DH_B per pool SD of shrunk xwOBA, about 53
              runs/9 per xwOBA point; the linear-weights scale PA9/wOBAscale is 31 and reads 1.7x
              too narrow: war_error_bar.py, split-half variance ratio 1.09 on this scale)
    REPL    = REPL_RP + (WAR_REPL_SP - REPL_RP) * GS/G,  REPL_RP = WAR_REPL_SP - WAR_ROLE_GAP / RPW

Improvement battery (2026-09-05, scripts/research/era/war_improve_battery.py + _battery2.py,
war_calibration_slope.py; data/_war_improve_battery*.json). The rate is at a plateau: the
xwOBA shrink N0 0-1000 moves nothing outside noise (250 stays), the linear-weights form
(xwRAA) and K%/BB% channels lose, an xRV blend wins two objectives narrowly but loses
reliability. Per-pitch channels: framing runs RECEIVED are half-stable (rel .43) and carry
a NEGATIVE next-season weight, so neutralizing the catcher would remove pitcher signal
(rejected); pulled-air excess predicts next season 5/5 at 60 IP but loses reliability 0/6
(open: test it as a fixed per-BIP adjustment, not a fitted channel); running game and
WP/BK lose everywhere; the pitcher's actual home-pitch share changes nothing. Calibration:
the same-season slope of actual RA9 on the rate reads .86 because of selection on outcomes
(low-IP arms 1.33, high-IP .76, the winner's-curse signature), not over-dispersion, so
DH_B stays. Volume stays innings (TBF-based moves nobody more than .6 WAR and credits
dominance less). Role variable stays start share: innings per appearance explains none
of the within-pitcher change.

Why hdERA: three WARs built under identical conventions on the 2021-2026 replicates
(scripts/research/era/war_rate_validation.py, data/_war_rate_validation.json), differing only
in the rate. hdERA vs FIP: half-season reliability .467 vs .407 (5/6), next-season RA9 at 60 IP
.355 vs .342 (3/5), at 30 IP .263 vs .229 (5/5), rest-of-season .328 vs .303, and on the 50
pitchers per season where FIP-WAR and RA9-WAR disagree most .310 vs .270. Actual RA9 loses
every test (reliability .212). No leverage term, by decision: it credits the manager's
deployment, the same job-versus-pitcher line Pitcher+ drew. No league correction (one MLB
pool) and no opponent adjustment. Relievers are not credited for the job; the role split in
replacement CORRECTS the rate inflation the job gives them (WAR_ROLE_GAP 0.85 runs/9,
measured within season on swingmen who did both jobs; the adjacent-season version reads
0.64 and is a floor, because demoted starters rebound and promoted relievers are the good
ones; fWAR's split implies about the same 0.85). ROC rows have no hdERA and therefore no
hWAR.

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
import math
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
# REFIT 2026-09-05 with the pitcher-hand term below (scripts/research/era/
# era_hand_weight_refit.py on the rebuilt replicates). Held-out r ROS-60
# .4525 -> .4545 (4/6), NEXT-60 .5209 -> .5282 (4/5); the point is
# calibration, not r: the LHP residual goes to zero in every test. These
# eight are the same fit's fold means (the eight-channel control landed
# within .03 of the 08-15 set on 7 of 8; park .149 vs .168 because the
# rebuilt targets carry every stint club, as combined_park_map does). xrv
# is POSITIVE here: the research harness negates a batter-positive input
# and reads -0.160 for the same channel. Previous set (2026-08-15, refit on
# production-consistent shrinkage): stuff .297 loc .136 k .088 izwh .117
# xrv .139 gb .162 gs .277 park .168.
W_PH = {'stuff': 0.315, 'loc': 0.105, 'k': 0.053, 'izwh': 0.101,
        'xrv': 0.160, 'gb': 0.151, 'gs': 0.298, 'park': 0.152}
# Pitcher-hand term (2026-09-05): ERA credit for a left-hander, applied to
# the RAW indicator centered at the pool's LHP share, so the pool-mean
# forecast does not move. Fold-mean -0.211 ROS-60; -0.254 NEXT-60 with SE
# .013 across the five year-pairs. Kept outside the z-sum on purpose: an
# indicator's z would tie the weight to the pool share.
W_LHP = -0.211
# Scratch/window rows only, when the metadata bundle predates the term:
# mean LHP share of the 2021-2026 60-IP replicate pools (.314 .269 .247
# .276 .263 .301). The season path measures the live share.
LHP_SHARE_FALLBACK = 0.278

# ── hWAR (2026-09-05) ────────────────────────────────────────────────────
WAR_PYTH_EXP = 0.287     # PythagenPat exponent; RPW = 4r/(2r)^0.287 = 9.4-9.9 at 2021-2026 run environments
WAR_REPL_SP = 0.12       # wins per 9 IP a starter earns above replacement (fWAR's .380); the one
                         # convention that sets the league total (~430 WAR with the measured gap)
WAR_ROLE_GAP = 0.85      # runs per 9 the reliever job is worth to the same pitcher, measured WITHIN
                         # SEASON on 281 swingman pitcher-seasons 2021-2025 (>= 50 PA in each role,
                         # side and starter reconstructed from pitch order; -0.53..-1.28 by season,
                         # SE .09). The adjacent-season version read 0.64 and is a floor: demoted
                         # starters rebound and promoted relievers are the good ones. 0.85 puts the
                         # reliever bar at ~.032 wins/9, fWAR's published .03. Shipped .64 until
                         # 2026-09-05 (same day).
WAR_PARK_PASS = 0.91     # share of the PUBLISHED runs park factor that reaches hdERA: LOSO
                         # innings-weighted slope .85-1.05, 2021-2026. (Actual runs move 1.67x the
                         # published factor: Savant's factor is shrunk, so 1.0 here is not "all".)
WAR_DYNAMIC_RPW = False  # HELD (per Wally 2026-09-05): runs per win stays the season constant a
                         # reader can check. The dynamic form, (rate + lgRA9)/2 per pitcher as fWAR
                         # and bWAR do, was built and measured: aces +25% (Misiorowski 7.6 -> 9.6),
                         # league sum +8%. A convention, not a measurement; the simpler one won.
WAR_XW_PA_SD = 0.366     # sd of a single PA's xwOBA-against value (2024, 181,704 PA): the sampling
                         # noise behind hWAR_se, converted to runs at hdERA's own DH_B / sd(pool)
                         # (about +/-1.2 WAR at 190 IP, +/-0.7 at 90). The first version (2026-09-05,
                         # a0e152658) converted at the wOBA scale and read 1.7x too narrow; the
                         # split-half check (war_error_bar.py) puts the ratio on the DH_B scale at 1.09.

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
    """{id(row) -> home park factor} for every combined 2TM/3TM/... row:
    the park of the club the pitcher MOST RECENTLY pitched for
    (current_team_by_player, the same lastGameDate resolution the
    qualification denominator uses).

    A combined row carries a LABEL, not a franchise, so park.get('2TM')
    has always fallen through to a neutral 100 -- and a traded pitcher
    did not pitch in a neutral park.

    MOST-RECENT CLUB, not IP-weighted history, by measurement (2026-08-24,
    scripts/research/era/era_park_weight_refit.py + _era_team_outs.json).
    Two facts came out of that battery:
      1. The shipped W_PH['park'] = 0.168 was fit on FINAL-CLUB exposure
         all along: the bulk stats endpoint returns one season-combined
         row per pitcher with only the last club attached, so the
         harness's "mean over clubs" never saw a second club. (An earlier
         version of this docstring believed otherwise.)
      2. Re-fit on true per-stint exposure (person-hydrate pull),
         IP-weighted history LOSES to final-club in every LOSO test
         (ROS g60 1/6 folds, -.0023 mean r; NEXT g60 1/5, -.0028).
         hpERA forecasts future runs, and the future innings come at the
         CURRENT club's park, so where he pitched BEFORE the trade is the
         worse proxy.
    So production now matches both the fit convention and the measured
    best: a combined row scores the park of its most recent club. The
    previous IP-weighted convention shipped 2026-08-19..2026-08-24.

    ROC/AAA stints are excluded by current_team_by_player: a pitcher sent
    down still resolves to his last MLB club.
    """
    from pipeline.utils import current_team_by_player, player_key
    cur = current_team_by_player(rows, 'pitcher', set(aaa_teams))
    out, unresolved = {}, []
    for r in rows:
        if not is_combined_fn(r.get('team')):
            continue
        team = cur.get(player_key(r, 'pitcher'))
        pf = park.get(team) if team is not None else None
        if pf is not None:
            out[id(r)] = pf
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
    # raw indicator, not z-scored (see W_LHP); None when the hand is unknown
    ch['lhp'] = {'L': 1.0, 'R': 0.0}.get(row.get('throws'))
    if is_combined and combined_park is not None:
        pf = combined_park.get(id(row))
        ch['park'] = (pf if pf is not None else 100.0) / 100.0
    else:
        ch['park'] = park.get(row.get('team'), 100.0) / 100.0
    return ch


def apply_era_plus(rows, pitches, aaa_teams=('ROC', 'AAA'),
                   is_combined_fn=None, season=None, roc_pitches=None,
                   league_rates=None):
    """Set hdERA / hpERA / hdERAPlus / hpERAPlus (+ _pctl each) in place.
    Returns the constants bundle for metadata, or None if the pool is too
    thin. `pitches` = the MLB+MiLB pitch dicts (sheet schema) for the
    xRV/100 channel; `is_combined_fn` identifies 2TM/3TM rows.
    `league_rates` = metadata pitcherLeagueAverages (lgRA9/lgERA) for hWAR;
    without them hWAR is left as carried and the skip is logged."""
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

    # pool LHP share: the centering point of the hand term
    _lh = [raw[id(r)]['lhp'] for r in pool_rows if raw[id(r)]['lhp'] is not None]
    lhp_share = (sum(_lh) / len(_lh)) if _lh else LHP_SHARE_FALLBACK
    n_nohand = 0

    dh_raw = {}
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
            if ch.get('lhp') is None:
                n_nohand += 1        # unknown hand scores at the pool share
            else:
                ph += W_LHP * (ch['lhp'] - lhp_share)
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
            if dh is not None:
                dh_raw[id(r)] = dh
            r['hdERAPlus'] = (round(200.0 - 100.0 * dh / anchor)
                              if dh is not None else None)
        r['hpERA'] = round(ph, 2) if ph is not None else None
        r['hpERAPlus'] = (round(200.0 - 100.0 * ph / anchor)
                          if ph is not None else None)

    # ── hWAR: deserved WAR on the unrounded hdERA (module docstring) ──
    war_const = None
    lg_ra9 = (league_rates or {}).get('lgRA9')
    lg_era = (league_rates or {}).get('lgERA')
    if lg_ra9 and lg_era:
        rpw = 4.0 * lg_ra9 / (2.0 * lg_ra9) ** WAR_PYTH_EXP        # league environment: replacement split + fallback
        repl_rp = WAR_REPL_SP - WAR_ROLE_GAP / rpw
        _xw_sd = mu_sd['xw'][1] if mu_sd.get('xw') else None   # pool SD of shrunk xwOBA against, hdERA's z unit

        def _rate_dp(r):
            dh = dh_raw.get(id(r)); ch = raw.get(id(r))
            if dh is None or ch is None:
                return None
            exposure = (ch['park'] + 1.0) / 2.0
            return dh + (lg_ra9 - lg_era) - WAR_PARK_PASS * (exposure - 1.0) * lg_ra9

        # recenter so innings-weighted RAA over MLB club rows sums to zero
        _num = _den = 0.0
        for r in mlb:
            v = _rate_dp(r); o = _ip_outs(r.get('ip'))
            if v is not None and o > 0 and not is_combined_fn(r.get('team')):
                _num += v * o; _den += o
        shift = (lg_ra9 - _num / _den) if _den > 0 else 0.0
        n_war = 0
        for r in rows:
            if r.get('team') in aaa:
                r['hWAR'] = None; r['hWAR_se'] = None
                continue
            v = _rate_dp(r); o = _ip_outs(r.get('ip')); g = r.get('g') or 0
            if v is None or o <= 0:
                r['hWAR'] = None; r['hWAR_se'] = None
                continue
            ip9 = o / 27.0
            repl = repl_rp + (WAR_REPL_SP - repl_rp) * ((r.get('gs') or 0) / g if g else 0.0)
            rate_i = v + shift
            if WAR_DYNAMIC_RPW:
                _env = max(0.5 * (rate_i + lg_ra9), 1.0)
                rpw_i = 4.0 * _env / (2.0 * _env) ** WAR_PYTH_EXP
            else:
                rpw_i = rpw
            r['hWAR'] = round((lg_ra9 - rate_i) * ip9 / rpw_i + repl * ip9, 2)
            pa = r.get('pa') or r.get('tbf') or 0
            r['hWAR_se'] = (round((DH_B / _xw_sd) * (pa / (pa + N0_XW)) * WAR_XW_PA_SD / math.sqrt(pa) * ip9 / rpw_i, 2)
                            if pa > 0 and _xw_sd else None)
            n_war += 1
        war_const = {'lgRA9': round(lg_ra9, 4), 'lgERA': round(lg_era, 4), 'rpw': round(rpw, 4),
                     'dynamicRpw': WAR_DYNAMIC_RPW, 'xwPaSd': WAR_XW_PA_SD,
                     'replSp': WAR_REPL_SP, 'replRp': round(repl_rp, 4), 'roleGap': WAR_ROLE_GAP,
                     'parkPass': WAR_PARK_PASS, 'shift': round(shift, 4), 'pythExp': WAR_PYTH_EXP,
                     'nRows': n_war, 'sum': round(sum(r['hWAR'] for r in mlb if r.get('hWAR') is not None
                                                      and not is_combined_fn(r.get('team'))), 1)}
        print(f"  hWAR: {n_war} rows, RPW {rpw:.2f}, repl SP {WAR_REPL_SP:.3f} / RP {repl_rp:.3f} wins per 9, "
              f"shift {shift:+.3f}, MLB club-row sum {war_const['sum']:.1f}")
    else:
        print('  eraplus WARNING: no lgRA9/lgERA in metadata pitcherLeagueAverages — hWAR '
              'left as carried (run process_data before the inject)')

    # percentiles: qualified MLB non-combined pool, every row ranked
    # (site convention). hdERA/hpERA are lower-is-better -> invert.
    for key, invert in (('hdERA', True), ('hpERA', True),
                        ('hdERAPlus', False), ('hpERAPlus', False),
                        ('hWAR', False)):
        pool = [r[key] for r in mlb
                if r.get(key) is not None
                and not is_combined_fn(r.get('team'))
                and _ip_outs(r.get('ip')) >= QUAL_OUTS]
        for r in rows:
            p = _pctl(r.get(key), pool)
            if p is not None and invert:
                # int like every other shipped percentile (2026-08-27 audit:
                # round(...,1) here was the one float-typed rank in the JSON).
                p = int(round(100.0 - p))
            r[key + '_pctl'] = p

    n_dh = sum(1 for r in rows if r.get('hdERA') is not None)
    n_ph = sum(1 for r in rows if r.get('hpERA') is not None)
    n_roc = sum(1 for r in rows
                if r.get('team') in aaa and r.get('hpERA') is not None)
    if roc_pitches and not n_roc:
        print('  eraplus WARNING: roc_pitches supplied but 0 ROC rows '
              'scored hpERA — check the xRV channel keying (Pitcher, PTeam)')
    print(f'  eraplus: anchor {anchor:.2f} (pool {len(pool_rows)}), '
          f'hdERA {n_dh} rows, hpERA {n_ph} rows ({n_roc} of them ROC/AAA), '
          f'LHP share {lhp_share:.3f}'
          + (f', {n_nohand} rows with unknown hand scored at the pool share'
             if n_nohand else ''))
    from pipeline.locplus import LOC_SCALE_K
    return {'anchor': round(anchor, 3), 'dhB': DH_B, 'weights': W_PH,
            'wLhp': W_LHP, 'lhpShare': round(lhp_share, 4),
            'war': war_const,
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


# Display floor for scratch/window hpERA, a labeled CONVENTION rather than a
# swept constant: the docstring promised a pa >= 100 proxy for the season
# path's 30 IP domain floor, and until 2026-08-27 no gate existed at all, so
# tiny windows printed an hpERA. 100 PA ~ 25 IP of batters faced, the closest
# available proxy in window context (no g/outs there). If a measured gate is
# ever wanted, sweep window-vs-season hpERA agreement by PA per the repo rule.
SCRATCH_HP_MIN_PA = 100


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
    # The gate the docstring always promised (implemented 2026-08-27): below
    # SCRATCH_HP_MIN_PA the window is too small for a projection to mean
    # anything, so hpERA stays blank. hdERA already shrinks by pa and keeps
    # rendering.
    if pa >= SCRATCH_HP_MIN_PA and all(zs.get(c) is not None for c in W_PH):
        ph = anchor + sum(W_PH[c] * zs[c] for c in W_PH)
        _lh = {'L': 1.0, 'R': 0.0}.get(row.get('throws'))
        if _lh is not None:
            ph += (const.get('wLhp', W_LHP)
                   * (_lh - const.get('lhpShare', LHP_SHARE_FALLBACK)))
    return (round(dh, 2) if dh is not None else None,
            round(ph, 2) if ph is not None else None)


def sort_rows_default(rows):
    """Default leaderboard order: hpERA good -> bad, valueless rows last
    (the client renders JSON order until a header is clicked)."""
    rows.sort(key=lambda r: (r.get('hpERA') is None,
                             r.get('hpERA') if r.get('hpERA') is not None
                             else 0.0))
    return rows
