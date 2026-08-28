#!/usr/bin/env python3
"""platoon_cards.py — one player card per handedness split.

Renders the SAME visuals as the season cards (HitterCards / Cards), but with
every number, grade and percentile bubble computed from one side of the platoon
split only.

The percentile question is the one that matters. A vs-LHP xwOBAcon is ranked
against other MLB hitters' vs-LHP xwOBAcon, not against their season numbers:
same-hand and opposite-hand distributions sit in different places, so a season
pool would systematically flatter the opposite-hand cards and punish the
same-hand ones. Pools come from scripts/tools/platoon_splits.py, which is validated
against the shipped leaderboard.

League ANCHORS stay full-season and MLB (SD+/CT+ cell tables, BB+ denominator,
Hitter+ standardization, SACQ zones, xRV count offsets). Only the player's own
pitches are split, so the two cards are on one scale and comparable to the
season card they came from.

Level handling, per Wally:
  - ROC hitters and the four ROC arms: their own level only.
  - Bird / Dion / Cruz: MLB and minor league pitches COMBINED into one card
    per hand, since the question is what the whole arm has done vs that side.
  - Ortiz: ROC only, his 5 MLB PA excluded.

Usage:  python3 scripts/tools/platoon_cards.py [--role hitters|pitchers|both]
                                          [--outdir DIR]
"""

import argparse
import copy
import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))

import platoon_splits as PS                                      # noqa: E402
from pipeline.utils import (                                     # noqa: E402
    MLB_TEAMS, NON_PA_EVENTS, hitter_pa_per_game, pitcher_ip_per_game,
    SP_GS_RATIO, ip_str_to_float,
)

# (display, sheet name, role, level policy)
#   'own'      — only the level they play at
#   'combined' — MLB stint plus minor league pitches, folded into one card
TARGETS = [
    ('Abimelec Ortiz',  'Ortiz, Abimelec',  'H', 'own'),
    ('Yohandy Morales', 'Morales, Yohandy', 'H', 'own'),
    ('Seaver King',     'King, Seaver',     'H', 'own'),
    ('Andrew Pinckney', 'Pinckney, Andrew', 'H', 'own'),
    ('Phillip Glasser', 'Glasser, Phillip', 'H', 'own'),
    ('Jackson Kent',    'Kent, Jackson',    'P', 'own'),
    ('Luis Perales',    'Perales, Luis',    'P', 'own'),
    ('Erik Tolman',     'Tolman, Erik',     'P', 'own'),
    ('Jack Sinclair',   'Sinclair, Jack',   'P', 'own'),
    ('Yovanny Cruz',    'Cruz, Yovanny',    'P', 'combined'),
    ('Jake Bird',       'Bird, Jake',       'P', 'combined'),
    ('Will Dion',       'Dion, Will',       'P', 'combined'),
]

MIN_SPLIT_PA = 20          # below this a card asserts more than the sample can

# Every percentile key the hitter card's bubbles read, mapped to the metric key
# platoon_splits computes. Bat tracking is absent for the minors, so those three
# stay None rather than inheriting a stale season percentile.
HITTER_PCTL_KEYS = [
    'xwOBA', 'hitterPlus', 'sdPlus', 'ctPlus', 'bbPlus',
    'xwOBAcon', 'babip', 'maxEV', 'hardHitPct', 'barrelPct', 'airPullPct',
    'bbPct', 'kPct', 'chasePct', 'izContactPct', 'xwOBAsp',
]
HITTER_PCTL_BLANK = ['batSpeed', 'squaredUpPct', 'blastPct', 'ev50',
                     'avgEVAll', 'iso', 'avg', 'obp', 'slg', 'ops', 'hr',
                     'contactPct', 'whiffPct', 'izSwingPct', 'swingPct',
                     'gbPct', 'ldPct', 'fbPct', 'puPct', 'pullPct',
                     'twoStrikeWhiffPct', 'firstPitchSwingPct', 'sprintSpeed',
                     'wRCplus', 'xWRCplus', 'bbToK', 'hrFbPct', 'sb',
                     'avgFbDist', 'avgHrDist', 'izSwChase', 'attackAngle',
                     'attackDirection', 'idealAAPct', 'rv100', 'xRv100']


# ═════════════════════════════════════════════════════════════════════════
#  ENGINE  (reuses platoon_splits so the math is the validated one)
# ═════════════════════════════════════════════════════════════════════════

def build_engine():
    import json
    with open(os.path.join(PS.DATA_DIR, 'metadata_rs.json')) as f:
        metadata = json.load(f)
    woba_weights = metadata.get('wobaWeights') or PS.WOBA_WEIGHTS_FALLBACK
    print(f"  wOBA weights: {woba_weights}")

    all_pitches, new_rows = PS.load_pitches(use_new_tab=True)
    # preprocess must see the NEW rows or they never get the MiLB->MLB RunExp
    # rescale, the InZone recompute or the xwOBA fill — which would leave the
    # combined arms' run values inflated by ~1.34x on their minor league half.
    # It mutates in place, so the same dicts referenced by new_rows are fixed.
    # Groups are still built from the CACHE ONLY: the NEW rows carry a scratch
    # PTeam of WSH, so folding them into pitcher_groups would both mint a
    # phantom WSH row and double-count them against pitcher_pitches, which
    # adds new_rows itself.
    ep = PS.preprocess(all_pitches + new_rows, new_rows, metadata, woba_weights)
    lg = PS.League(all_pitches, ep, metadata, woba_weights)

    hitter_groups, pitcher_groups = defaultdict(list), defaultdict(list)
    for p in all_pitches:
        if not p.get('_roc_pitcher_pitch'):
            b, bt = p.get('Batter'), p.get('BTeam')
            if b and bt:
                hitter_groups[(b, bt)].append(p)
        if not p.get('_roc_hitter_pitch'):
            k = (p.get('Pitcher'), p.get('PTeam'))
            if k[0] and k[1] and k not in ep:
                pitcher_groups[k].append(p)
    sd_groups = {k: [q for q in v
                     if (q.get('Pitcher'), q.get('PTeam')) not in ep]
                 for k, v in hitter_groups.items()}
    lg.anchor_from_full_season(sd_groups)
    return dict(metadata=metadata, lg=lg, all_pitches=all_pitches,
                new_rows=new_rows, hitter_groups=dict(hitter_groups),
                pitcher_groups=dict(pitcher_groups), sd_groups=sd_groups)


def qualified_mlb(kind, lg):
    import json
    fn = 'hitter_leaderboard_rs.json' if kind == 'H' else 'pitcher_leaderboard_rs.json'
    with open(os.path.join(PS.DATA_DIR, fn)) as f:
        rows = json.load(f)
    out = []
    for r in rows:
        t = r.get('team')
        if t not in MLB_TEAMS:
            continue
        tg = lg.team_games.get(t, lg.max_tg)
        if kind == 'H':
            if (r.get('pa') or 0) >= hitter_pa_per_game(False) * tg:
                out.append((r['hitter'], t, None))
        else:
            ip = ip_str_to_float(r.get('ip')) if r.get('ip') is not None else 0
            g, gs = r.get('g') or 0, r.get('gs') or 0
            if ip >= tg * pitcher_ip_per_game(g > 0 and gs / g > SP_GS_RATIO, False):
                out.append((r['pitcher'], t, gs / g if g else 0.0))
    return rows, out


def hitter_split_pools(eng):
    """MLB pool of vs-RHP and vs-LHP values, one entry per qualified hitter."""
    lg = eng['lg']
    _, qual = qualified_mlb('H', lg)
    pools = {'R': defaultdict(list), 'L': defaultdict(list)}
    for name, team, _ in qual:
        ps_all = eng['hitter_groups'].get((name, team)) or []
        sd_all = eng['sd_groups'].get((name, team)) or []
        for hand in ('R', 'L'):
            ps = [p for p in ps_all if p.get('Throws') == hand]
            if not ps:
                continue
            sd = [p for p in sd_all if p.get('Throws') == hand]
            row = PS.hitter_row(ps, sd, lg)
            if (row.get('pa') or 0) < 25:
                continue
            for k in HITTER_PCTL_KEYS:
                if row.get(k) is not None:
                    pools[hand][k].append(row[k])
    print(f"  hitter pools: vs RHP {len(pools['R'].get('xwOBAcon', []))} players, "
          f"vs LHP {len(pools['L'].get('xwOBAcon', []))}")
    return pools


HIB = {k: hib for k, _l, _f, hib in PS.HITTER_METRICS}


def fit_aaa_wrc_line(min_pa=100):
    """Recover the Triple-A wRC+ baseline from the data, as (k, c) in
    wRC+ = k*wOBA + c.

    Two wRC+ constructions coexist for a ROC hitter: the pipeline's, built on
    MLB Guts constants with a park factor of 1.0, and FanGraphs' AAA figure,
    built on International League baselines and MiLB park factors. Both are
    affine in wOBA with k = 100/(wOBAScale x lgRPA), and Triple-A's higher run
    environment makes its k SMALLER — so the gap between them is a line in
    wOBA, not a constant. A flat offset is therefore right at the middle of
    the distribution and drifts at the tails, which is exactly where a platoon
    split sits.

    Fitting FG's AAA wRC+ directly on wOBA skips the offset and recovers the
    AAA baseline itself. Measured 2026-08-04: k 614.8, implied AAA lgWOBA
    .3377 against MLB's .3164, R2 .995, residual SD 1.6 points. Fit at runtime
    rather than hardcoded because it moves with the league as the season goes
    and with FanGraphs' park factors.

    Returns (k, c, n, resid_sd) or None if the inputs aren't there.
    """
    import json
    try:
        with open(os.path.join(PS.DATA_DIR, 'hitter_leaderboard_rs.json')) as f:
            lb = json.load(f)
        with open(os.path.join(PS.DATA_DIR, 'fg_overrides.json')) as f:
            fg = json.load(f).get('aaaHitters', {})
    except Exception:                                            # noqa: BLE001
        return None
    xs, ys = [], []
    for r in lb:
        if r.get('team') != 'ROC' or (r.get('pa') or 0) < min_pa:
            continue
        mid, w = r.get('mlbId'), r.get('wOBA')
        if mid is None or w is None:
            continue
        p = fg.get(str(int(mid))) or {}
        if p.get('wRCplus') is None:
            continue
        xs.append(w); ys.append(p['wRCplus'])
    if len(xs) < 6:
        return None
    n = len(xs)
    mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((v - mx) ** 2 for v in xs)
    if sxx <= 0:
        return None
    k = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / sxx
    c = my - k * mx
    resid = [ys[i] - (k * xs[i] + c) for i in range(n)]
    sd = (sum(r * r for r in resid) / (n - 2)) ** 0.5 if n > 2 else 0.0
    return k, c, n, sd


def split_wrc_plus(woba, lg, team, aaa_line=None):
    """wRC+ on the split's own wOBA, using process_data's exact formula.

    Only wOBA is split-specific; lgWOBA / wOBAScale / lgRPA are season
    constants and the park factor is the player's, not the split's. The league
    baseline stays SEASON-WIDE rather than same-hand, matching how FanGraphs
    reports split wRC+: 100 means "league average hitter", not "league average
    hitter against this hand", so a righty's vs-LHP figure is expected to run
    high.

    NOTE this is the MLB-baselined construction the pipeline applies to every
    row. It is NOT FanGraphs' AAA wRC+, which the season card shows via the FG
    override and which is built on International League baselines and MiLB park
    factors. The two will not reconcile, and the MLB-baselined one is the one
    consistent with every other number on a split card."""
    if woba is None:
        return None
    # ROC rows get the AAA baseline recovered by fit_aaa_wrc_line, so the split
    # figure lands on the same scale as the season card's FanGraphs number
    # instead of the MLB-baselined one.
    if aaa_line is not None and team in ('ROC', 'AAA'):
        k, c = aaa_line[0], aaa_line[1]
        return round(k * woba + c)
    scale, lg_woba, lg_rpa = lg.woba_scale, lg.lg_woba, lg.lg_rpa
    if not scale or not lg_rpa:
        return None
    pf = 1.0                       # no AAA park factors published; pipeline
                                   # already assumes 1.0 for ROC rows
    wraa_per_pa = (woba - lg_woba) / scale
    return round((wraa_per_pa + lg_rpa + (lg_rpa - pf * lg_rpa)) / lg_rpa * 100)


def build_hitter_row(base_row, split_row, pools, hand, lg=None, aaa_line=None):
    """A leaderboard-shaped row for one split: identity from the real row,
    every stat and percentile replaced by the split's own."""
    row = copy.deepcopy(base_row)
    for k, v in split_row.items():
        if not k.startswith('_'):
            row[k] = v
    # Percentiles the bubbles read, on the same-split pool.
    for k in HITTER_PCTL_KEYS:
        row[k + '_pctl'] = PS.percentile(split_row.get(k), pools[hand].get(k, []),
                                         HIB.get(k, True))
    # Anything not recomputed must NOT keep a season percentile.
    for k in HITTER_PCTL_BLANK:
        row[k + '_pctl'] = None
    for k in ('batSpeed', 'squaredUpPct', 'blastPct', 'xWRCplus'):
        row[k] = None
    row['wRCplus'] = (split_wrc_plus(split_row.get('wOBA'), lg, row.get('team'), aaa_line)
                      if lg else None)
    return row


# ═════════════════════════════════════════════════════════════════════════
#  HITTER CARDS
# ═════════════════════════════════════════════════════════════════════════

def render_hitters(eng, outdir):
    import cards.hitter as HC
    pools = hitter_split_pools(eng)
    aaa_line = fit_aaa_wrc_line()
    if aaa_line:
        k, c, n, sd = aaa_line
        print(f"  AAA wRC+ baseline fit: wRC+ = {k:.1f}*wOBA {c:+.1f}  "
              f"(n={n} ROC hitters, resid SD {sd:.1f}, implied AAA lgWOBA "
              f"{(100 - c) / k:.4f})")
    else:
        print("  WARNING: could not fit the AAA wRC+ baseline — "
              "split wRC+ will stay MLB-baselined")
    lb, _ = qualified_mlb('H', eng['lg'])
    lg = eng['lg']
    made, skipped = [], []

    for display, sheet, role, _policy in TARGETS:
        if role != 'H':
            continue
        base = next((r for r in lb if r.get('hitter') == sheet
                     and r.get('team') == 'ROC'), None)
        if base is None:
            skipped.append(f"{display}: no ROC leaderboard row")
            continue
        # Ortiz limited to ROC per Wally, so the group key is the level key.
        ps_all = eng['hitter_groups'].get((sheet, 'ROC')) or []
        sd_all = eng['sd_groups'].get((sheet, 'ROC')) or []

        for hand, label in (('R', 'vs RHP'), ('L', 'vs LHP')):
            ps = [p for p in ps_all if p.get('Throws') == hand]
            sd = [p for p in sd_all if p.get('Throws') == hand]
            split = PS.hitter_row(ps, sd, lg) if ps else None
            n_pa = (split or {}).get('pa') or 0
            if n_pa < MIN_SPLIT_PA:
                skipped.append(f"{display} {label}: {n_pa} PA")
                continue
            row = build_hitter_row(base, split, pools, hand, lg, aaa_line)

            # The card overwrites wRCplus (and xwOBA/xBA/xSLG for MLB hitters)
            # from the FanGraphs override cache AFTER taking h_row, which would
            # stamp a SEASON wRC+ onto a split card. Neutralize the cache for
            # the render: wRC+ has no per-split construction, so it shows as a
            # dash instead of a number that silently means something else.
            import pipeline.fg_overrides as _FG
            _pitches, _lb = HC.load_pitch_data, HC.load_hitter_leaderboard
            _fgr = _FG.refresh_if_stale
            HC.load_pitch_data = lambda *a, **k: ps
            HC.load_hitter_leaderboard = lambda *a, **k: [row]
            _FG.refresh_if_stale = lambda *a, **k: {}
            try:
                ok = HC.render_hitter_card(
                    sheet, team_abbrev='ROC',
                    year_label=f'2026 {label}', output_dir=outdir)
                (made if ok else skipped).append(f"{display} {label} ({n_pa} PA)")
            except Exception as e:                               # noqa: BLE001
                import traceback; traceback.print_exc()
                skipped.append(f"{display} {label}: {e}")
            finally:
                HC.load_pitch_data, HC.load_hitter_leaderboard = _pitches, _lb
                _FG.refresh_if_stale = _fgr
    return made, skipped


# ═════════════════════════════════════════════════════════════════════════
#  PITCHER CARDS
# ═════════════════════════════════════════════════════════════════════════

# Cards.py names two grades differently from platoon_splits.
_GRADE_ALIAS = {'stuffPlus': 'stuffScore'}


# Per-player hard date cap (inclusive). 2026-08-10 per Wally: caps LIFTED —
# platoon cards now include the FULL season for everyone, WSH call-up
# outings included, matching the season cards. (Historical: Cruz/Dion were
# capped at 8/4 while the article treated them as AAA arms awaiting a shot.)
DATE_CAP = {}


def pitcher_pitches(eng, sheet, policy):
    """This arm's pitches. 'own' = ROC tab only (the four Rochester arms have
    no MLB stint, and per Wally their cards are AAA-only regardless);
    'combined' folds the MLB stint together with the minor league rows: 25
    lefties faced in AAA plus 75 in MLB is 100 lefties faced."""
    out = []
    for (name, team), plist in eng['pitcher_groups'].items():
        if name != sheet:
            continue
        if team == 'ROC':
            out += plist
        elif policy == 'combined' and team in MLB_TEAMS:
            out += plist
    if policy == 'combined':
        out += [p for p in eng['new_rows'] if p.get('Pitcher') == sheet]
    cap = DATE_CAP.get(sheet)
    if cap:
        n0 = len(out)
        out = [p for p in out if (p.get('Game Date') or '') <= cap]
        if len(out) != n0:
            print(f"  {sheet}: date cap {cap} dropped {n0 - len(out)} pitches")
    return out


def pitcher_split_pools(eng):
    lg = eng['lg']
    _, qual = qualified_mlb('P', lg)
    pools = {'R': defaultdict(list), 'L': defaultdict(list)}
    for name, team, gs_ratio in qual:
        ps_all = eng['pitcher_groups'].get((name, team)) or []
        for hand in ('R', 'L'):
            ps = [p for p in ps_all if p.get('Bats') == hand]
            if not ps:
                continue
            row = PS.pitcher_row(ps, lg, gs_ratio)
            if (row.get('tbf') or 0) < 25:
                continue
            for k, v in list(row.items()):
                if v is not None:
                    pools[hand][_GRADE_ALIAS.get(k, k)].append(v)
    print(f"  pitcher pools: vs RHH {len(pools['R'].get('kPct', []))} arms, "
          f"vs LHH {len(pools['L'].get('kPct', []))}")
    return pools


def render_pitchers(eng, outdir):
    import cards.pitcher as CD
    pools = pitcher_split_pools(eng)
    lb, _ = qualified_mlb('P', eng['lg'])
    meta = eng['metadata']
    made, skipped = [], []

    # Reuse the pitches already in memory instead of letting Cards load the
    # 500k-pitch pickle a second time.
    CD._MLB_PICKLE_CACHE = [p for p in eng['all_pitches']
                            if p.get('_source') == 'MLB']
    mvn = CD.load_mvn_models()
    mlb_cache = CD.load_mlb_id_cache()

    for display, sheet, role, policy in TARGETS:
        if role != 'P':
            continue
        all_p = pitcher_pitches(eng, sheet, policy)
        if not all_p:
            skipped.append(f"{display}: no pitches")
            continue
        base = next((r for r in lb if r.get('pitcher') == sheet), None)
        levels = sorted({('MLB' if p.get('_source') == 'MLB' else 'MiLB')
                         for p in all_p})
        lvl = '+'.join(levels)

        for hand, label in (('R', 'vs RHH'), ('L', 'vs LHH')):
            ps = [p for p in all_p if p.get('Bats') == hand]
            tbf = sum(1 for p in ps
                      if p.get('Event') and p['Event'] not in NON_PA_EVENTS)
            if tbf < MIN_SPLIT_PA:
                skipped.append(f"{display} {label}: {tbf} TBF")
                continue

            norm = {sheet: [CD._normalize_scratch_pitch(p) for p in ps]}
            ctx = CD._build_scratch_league_context(norm, stuff_k_shrink=None)
            # Rank against the SAME split, not the season pool.
            ctx['pitcher_pools'] = {k: sorted(v) for k, v in pools[hand].items()}
            pctl_row, pitch_lb, locplus = CD._compute_scratch_pitcher_context(sheet, ctx)
            if pctl_row is None:
                skipped.append(f"{display} {label}: context failed")
                continue

            parts = sheet.split(', ')
            disp = (parts[1] + ' ' + parts[0]).upper() if len(parts) == 2 else sheet.upper()
            team_lbl = (base or {}).get('team') or 'WSH'
            mlb_id = CD.lookup_mlb_id(sheet, team_lbl, mlb_cache)
            headshot = CD.fetch_headshot(mlb_id) if mlb_id else None
            if headshot is None:
                from PIL import Image
                headshot = Image.new('RGB', (180, 180), (50, 50, 50))
            age = None
            try:
                age = CD.fetch_player_metadata(mlb_id).get('age') if mlb_id else None
            except Exception:                                    # noqa: BLE001
                pass

            # IP/ERA cannot be split by batter hand, so the header carries the
            # split's own counting line instead of a season pitching line.
            hdrs = ['TBF', 'Pitches', 'K%', 'BB%', 'xwOBA', 'Whiff%']
            vals = [str(tbf), str(len(ps)),
                    PS.fmt(pctl_row.get('kPct'), 'pct') + '%',
                    PS.fmt(pctl_row.get('bbPct'), 'pct') + '%',
                    PS.fmt(pctl_row.get('xwOBA'), 'avg'),
                    PS.fmt(pctl_row.get('swStrPct'), 'pct') + '%']

            config = {
                'display_name': disp,
                'hand': (ps[0].get('Throws') or (base or {}).get('throws') or 'R'),
                'team': team_lbl,
                'age': age or '—',
                'game_date': f'2026 {label} · {lvl}',
                'stat_headers': hdrs,
                'stat_values': vals,
                'headshot': headshot,
                'mlb_id': mlb_id,
                'league_avgs': meta.get('leagueAverages', {}),
                'overall_avgs': meta.get('pitcherLeagueAverages', {}),
                'pitcher_league_avgs': meta.get('pitcherLeagueAverages', {}),
                'mvn_models': mvn,
                'pctl_row': pctl_row,
                'pitch_locplus': locplus,
                'pitch_lb': pitch_lb,
                'loc_hands': (hand,),   # one side only; the other is empty
                'rv_mode': 'per100',
                'pitch_qual': None,
            }
            slug = (parts[0] + parts[1]).replace(' ', '') if len(parts) == 2 \
                else sheet.replace(' ', '').replace(',', '')
            out = os.path.join(outdir,
                               f"PitcherCard_{slug}_2026_{label.replace(' ', '_')}.png")
            try:
                ok = CD.render_card(config, ps, out)
                (made if ok else skipped).append(f"{display} {label} ({tbf} TBF, {lvl})")
            except Exception as e:                               # noqa: BLE001
                import traceback; traceback.print_exc()
                skipped.append(f"{display} {label}: {e}")
    CD.save_mlb_id_cache(mlb_cache)
    return made, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--role', default='both',
                    choices=['hitters', 'pitchers', 'both'])
    ap.add_argument('--outdir', default=os.path.expanduser('~/Downloads/platoon_cards'))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("=== Building split engine ===")
    eng = build_engine()

    made, skipped = [], []
    if args.role in ('hitters', 'both'):
        print("\n=== Hitter cards ===")
        m, s = render_hitters(eng, args.outdir)
        made += m; skipped += s
    if args.role in ('pitchers', 'both'):
        print("\n=== Pitcher cards ===")
        m, s = render_pitchers(eng, args.outdir)
        made += m; skipped += s

    print(f"\n{'='*60}\nRendered {len(made)} cards to {args.outdir}")
    for x in made:
        print('  ' + x)
    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for x in skipped:
            print('  ' + x)


if __name__ == '__main__':
    main()
