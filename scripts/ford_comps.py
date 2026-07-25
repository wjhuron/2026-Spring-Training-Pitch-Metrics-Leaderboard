"""ford_comps.py — closest player comparisons from a stat fingerprint.

Similarity = mean |z-difference| across a stat fingerprint, z-scored over the
full comparison pool (MLB + ROC, season sample floors; ROC is measured against
MLB baselines site-wide, so cross-level shape comps are legitimate).
Distances under ~0.45 are snug. Shape comps, not talent equivalences.

Hitters: 18-feature offensive fingerprint (contact quality, batted-ball
profile, swing decisions, contact skill incl. IZWhiff%, bat tracking,
outcomes).
Pitchers: 17-feature fingerprint (release/approach: FB velo, extension, arm
angle, VAA, |HAA|; whiff/zone: SwStr%, IZWhiff%, Chase%, IZ%;
outcomes: K%, BB%; contact against: GB%, PU%, EV, HardHit%, Barrel%, xwOBAcon)
PLUS a pitch-mix component (weight MIX_W): arsenals compared TAG-BLIND as
usage-weighted (velo, IVB, HB) shapes via a greedy earth-mover — an SL tag
here can match an FC tag there; shape beats label. LHP horizontal break is
mirrored onto the RHP axis. HAA is hand-signed, so the fingerprint uses its
magnitude. Comp lines and the CSV show the mix distance separately.

The comp pool is ALWAYS MLB-only — a ROC player is never comped to another
ROC player, and z-scales come from the MLB pool. The level setting
('mlb' / 'aaa' / 'both') picks which of the TARGET's stat lines to use: a
player with time at both levels (e.g. House WSH + ROC) has one row per
team, and 'aaa' comps his ROC (AAA-only) line against MLB players. 'both'
runs every row he has. Batch mode ignores level (team picks the row).

Date ranges are recomputed from the pitch cache (same features, lower pool
floors).

Run with NO arguments to use the SELECTION block at the bottom of the file:
edit role / player / team / level / date range there directly, then run.

Usage examples:
  python3 scripts/ford_comps.py                                  selection block
  python3 scripts/ford_comps.py --hitter "Ford, Harry" --start 2026-06-01
  python3 scripts/ford_comps.py --pitcher "Kolek, Bryce" --level mlb
  python3 scripts/ford_comps.py --team ROC --role both
  python3 scripts/ford_comps.py --team ROC --exclude "Jordan, Levi;Wallace, Cayden"
"""
import os, sys, csv, json, pickle, math, argparse
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline_utils import (safe_float as sf, NON_PA_EVENTS, BB_EVENTS,
                            K_EVENTS, BUNT_BB_TYPES,
                            spray_angle, spray_direction, SWING_DESCRIPTIONS)

FEATS_H = ['avgEVAll', 'hardHitPct', 'barrelPct', 'xwOBAcon',
           'gbPct', 'puPct', 'pullPct', 'airPullPct',
           'swingPct', 'izSwingPct', 'chasePct', 'whiffPct', 'izWhiffPct',
           'kPct', 'bbPct', 'batSpeed', 'swingLength', 'attackAngle']
NICE_H = {'avgEVAll': 'EV', 'hardHitPct': 'HardHit%',
          'barrelPct': 'Barrel%', 'xwOBAcon': 'xwOBAcon', 'gbPct': 'GB%',
          'puPct': 'PU%', 'pullPct': 'Pull%', 'airPullPct': 'AirPull%',
          'swingPct': 'Swing%', 'izSwingPct': 'IZSwing%', 'chasePct': 'Chase%',
          'whiffPct': 'Whiff%', 'izWhiffPct': 'IZWhiff%',
          'kPct': 'K%', 'bbPct': 'BB%',
          'batSpeed': 'BatSpd', 'swingLength': 'SwLen', 'attackAngle': 'AttackAng'}

FEATS_P = ['fbVelo', 'extension', 'armAngle', 'vaa', 'haa',
           'kPct', 'bbPct', 'swStrPct', 'izWhiffPct',
           'chasePct', 'izPct',
           'gbPct', 'puPct', 'avgEVAgainst', 'hardHitPct',
           'barrelPctAgainst', 'xwOBAcon']
NICE_P = {'fbVelo': 'FBVelo', 'extension': 'Ext', 'armAngle': 'ArmAng',
          'vaa': 'VAA', 'haa': '|HAA|', 'kPct': 'K%', 'bbPct': 'BB%',
          'swStrPct': 'SwStr%', 'izWhiffPct': 'IZWhiff%',
          'chasePct': 'Chase%', 'izPct': 'IZ%', 'gbPct': 'GB%', 'puPct': 'PU%',
          'avgEVAgainst': 'EV', 'hardHitPct': 'HardHit%',
          'barrelPctAgainst': 'Barrel%', 'xwOBAcon': 'xwOBAcon'}

MIN_FEATS = 12
# Pitch-mix component (pitchers): arsenals are compared TAG-BLIND — every
# pitch is just (usage, velo, IVB, HB[arm-side], spin tilt, spin rate) and
# usage mass is matched to the metrically nearest shapes in the other arsenal
# (greedy earth-mover). A pitch tagged SL here can match a pitch tagged FC
# there; shape > tag. Tilt is the RELEASE (spin-axis) tilt, not OTilt —
# OTilt is computed from IVB/HB so it would double-count movement; the
# spin axis adds the seam-shifted-wake/gyro info movement can't see. Tilt
# and spin get half weight since movement already encodes most of them.
MIX_W = 1 / 3        # share of the pitcher distance that comes from the mix
MIX_CORE = ('velo', 'ivb', 'hb')                 # required dims, weight 1
MIX_SOFT = (('tilt', 0.5), ('spin', 0.5))        # optional dims, half weight
MIX_MIN_USAGE = 0.03                             # drop show-me pitches
TILT_WRAP = 720                                  # tilt is minutes past 12:00


def tilt_minutes(s):
    """'10:52' -> 652; already-numeric minutes pass through."""
    if s is None:
        return None
    try:
        return float(s) % TILT_WRAP
    except (TypeError, ValueError):
        pass
    try:
        h, m = str(s).split(':')
        return (int(h) % 12) * 60 + int(m)
    except (TypeError, ValueError):
        return None


def mirror_tilt(t):
    """Mirror LHP spin tilt across the 12:00-6:00 axis onto the RHP clock."""
    return (TILT_WRAP - t) % TILT_WRAP


def tilt_diff(a, b):
    d = abs(a - b)
    return min(d, TILT_WRAP - d)
NL = {'hitter': 'PA', 'pitcher': 'TBF'}          # sample-count label
SEASON_POOL_MIN = {'hitter': 250, 'pitcher': 150}
WINDOW_POOL_MIN = {'hitter': 100, 'pitcher': 60}
SMALL_FLAG = {'hitter': 200, 'pitcher': 150}
AAA_TEAMS = {'ROC', 'AAA'}


def zstats(pool, feats):
    out = {}
    for f in feats:
        vals = [r[f] for r in pool if r.get(f) is not None]
        m = sum(vals) / len(vals)
        s = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
        out[f] = (m, s if s > 0 else 1.0)
    return out


def mix_zstats(pool):
    """Per-dimension scale over every arsenal pitch of the pool.

    Linear dims get (mean, sd); tilt gets a circular sd in minutes (via the
    mean resultant length) since the clock wraps at 12:00.
    """
    out = {}
    entries = [p for r in pool for p in (r.get('arsenal') or [])]
    for d in MIX_CORE + tuple(k for k, _ in MIX_SOFT):
        vals = [e[d] for e in entries if e.get(d) is not None]
        if not vals:
            out[d] = (0.0, 1.0)
            continue
        if d == 'tilt':
            sin_s = sum(math.sin(v / TILT_WRAP * 2 * math.pi) for v in vals)
            cos_s = sum(math.cos(v / TILT_WRAP * 2 * math.pi) for v in vals)
            r = min(max(math.hypot(sin_s, cos_s) / len(vals), 1e-9), 1 - 1e-9)
            s = math.sqrt(-2 * math.log(r)) * TILT_WRAP / (2 * math.pi)
            out[d] = (0.0, max(s, 1.0))
            continue
        m = sum(vals) / len(vals)
        s = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
        out[d] = (m, s if s > 0 else 1.0)
    return out


def arsenal_dist(A, B, mz):
    """Usage-weighted, tag-blind arsenal distance (greedy earth-mover).

    Cost between two pitches = weighted mean |z-diff| over the shape dims
    (velo/IVB/HB required; tilt/spin half-weight when both sides have them);
    each pitcher's usage mass flows to the metrically nearest shapes in the
    other arsenal.
    """
    pairs = []
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            cs = [abs(a[d] - b[d]) / mz[d][1] for d in MIX_CORE
                  if a.get(d) is not None and b.get(d) is not None]
            if len(cs) < len(MIX_CORE):
                continue
            num, den = sum(cs), float(len(cs))
            for d, w in MIX_SOFT:
                va, vb = a.get(d), b.get(d)
                if va is None or vb is None:
                    continue
                diff = tilt_diff(va, vb) if d == 'tilt' else abs(va - vb)
                num += w * diff / mz[d][1]
                den += w
            pairs.append((num / den, i, j))
    if not pairs:
        return None
    pairs.sort()
    ra = [a['usage'] for a in A]
    rb = [b['usage'] for b in B]
    cost = moved = 0.0
    for c, i, j in pairs:
        m = min(ra[i], rb[j])
        if m > 0:
            cost += c * m
            moved += m
            ra[i] -= m
            rb[j] -= m
    return cost / moved if moved else None


def dist(a, b, feats, stats, mix=None):
    ds = []
    for f in feats:
        va, vb = a.get(f), b.get(f)
        if va is None or vb is None:
            continue
        m, s = stats[f]
        ds.append((abs((va - m) / s - (vb - m) / s), f))
    if len(ds) < MIN_FEATS:
        return None, ds, None
    d = sum(dd for dd, _ in ds) / len(ds)
    md = None
    if mix is not None and a.get('arsenal') and b.get('arsenal'):
        md = arsenal_dist(a['arsenal'], b['arsenal'], mix)
    if md is not None:
        d = (1 - MIX_W) * d + MIX_W * md
    return d, ds, md


def why_parts(t, comp, ds, feats, stats, nice):
    """Shared extremes (both |z|>=0.75 same sign) + the biggest gap, as strings."""
    def z(r, f):
        v = r.get(f)
        if v is None:
            return None
        m, s = stats[f]
        return (v - m) / s
    shared = []
    for f in feats:
        za, zb = z(t, f), z(comp, f)
        if za is None or zb is None:
            continue
        if abs(za) >= 0.75 and abs(zb) >= 0.75 and za * zb > 0:
            shared.append((min(abs(za), abs(zb)), f, za))
    shared.sort(reverse=True)
    tr = ', '.join(f"{nice[f]} {'high' if zz > 0 else 'low'}"
                   for _, f, zz in shared[:4]) or 'league-average across the board'
    gap_d, gap_f = max(ds, key=lambda x: x[0])[0], max(ds, key=lambda x: x[0])[1]
    fmt = lambda v: f'{v:.4g}' if isinstance(v, float) else v
    gap = (f"{nice[gap_f]} (Δz {gap_d:.1f}: "
           f"{fmt(t.get(gap_f))} vs {fmt(comp.get(gap_f))})")
    return tr, gap


def why_lines(t, best, ds, feats, stats, nice):
    tr, gap = why_parts(t, best, ds, feats, stats, nice)
    return f"   shared: {tr}\n   biggest gap: {gap}"


CSV_ROWS = []
CSV_FIELDS = ['role', 'window', 'player', 'player_team', 'sample_type', 'sample',
              'rank', 'distance', 'mix', 'comp', 'comp_team', 'comp_side',
              'comp_sample', 'shared', 'biggest_gap']


def write_csv(path):
    if not CSV_ROWS:
        print('\n(no comp rows to write — CSV skipped)')
        return
    path = os.path.expanduser(path)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(CSV_ROWS)
    print(f"\nCSV written: {path} ({len(CSV_ROWS)} rows)")


def report(target, pool, feats, stats, label, nice, topn=5, verbose=True,
           nl='PA', small=200, csvmeta=None, mix=None):
    scored = []
    for r in pool:
        d, ds, md = dist(target, r, feats, stats, mix)
        if d is not None:
            scored.append((d, r, ds, md))
    scored.sort(key=lambda x: x[0])
    if not scored:
        print(f"\n{label}\n   (no pool candidates with >= {MIN_FEATS} shared features)")
        return []
    flag = ' [small sample]' if target['pa'] < small else ''
    print(f"\n{label}{flag}")
    htag = 'HP' if nl == 'TBF' else 'HB'         # RHP/LHP vs RHB/LHB/SHB
    for d, r, _, md in scored[:topn]:
        hand = f" ({r['side']}{htag})" if r.get('side') else ''
        mixs = f"  mix {md:.3f}" if md is not None else ''
        print(f"   {d:.3f}  {r['name']:24s} {r['team']:3s}  {nl} {r['pa']}{hand}{mixs}")
    for rank, (d, r, ds, md) in enumerate(scored[:3], 1):
        tr, gap = why_parts(target, r, ds, feats, stats, nice)
        CSV_ROWS.append(dict(csvmeta or {}, player=target['name'],
                             player_team=target['team'], sample_type=nl,
                             sample=target['pa'], rank=rank,
                             distance=f'{d:.3f}',
                             mix=f'{md:.3f}' if md is not None else '',
                             comp=r['name'], comp_team=r['team'],
                             comp_side=r.get('side') or '',
                             comp_sample=r['pa'], shared=tr, biggest_gap=gap))
    d0, b0, ds0 = scored[0][0], scored[0][1], scored[0][2]
    print(why_lines(target, b0, ds0, feats, stats, nice))
    if verbose:
        print(f"\n   feature detail vs {b0['name']} (z-units):")
        for f in feats:
            vf, vb = target.get(f), b0.get(f)
            if vf is None or vb is None:
                continue
            m, s = stats[f]
            print(f"    {nice[f]:10s} {target['name'].split(',')[0]:12s} {vf:8.3f} "
                  f"(z{(vf-m)/s:+.2f})   {b0['name'].split(',')[0]:12s} {vb:8.3f} "
                  f"(z{(vb-m)/s:+.2f})")
    return scored[:topn]


def in_level(team, level):
    if level == 'mlb':
        return team not in AAA_TEAMS
    if level == 'aaa':
        return team in AAA_TEAMS
    return True


def _mk_arsenal(entries):
    """entries: [(usage, velo, ivb, hb_armside, tilt_armside, spin)] ->
    cleaned, renormalized list. Core dims required; tilt/spin optional."""
    keep = [dict(usage=u, velo=v, ivb=iv, hb=hb, tilt=ti, spin=sp)
            for u, v, iv, hb, ti, sp in entries
            if u and u >= MIX_MIN_USAGE and v is not None
            and iv is not None and hb is not None]
    tot = sum(p['usage'] for p in keep)
    if not keep or tot <= 0:
        return None
    for p in keep:
        p['usage'] /= tot
    return keep


def load_arsenals():
    """(pitcher, team) -> tag-blind arsenal from the per-pitch-type leaderboard.

    HB is hand-signed in the data, so LHP break is mirrored to put both hands
    on the same arm-side/glove-side axis.
    """
    rows = json.load(open(os.path.join(ROOT, 'data', 'pitch_leaderboard_rs.json')))
    grouped = defaultdict(list)
    for r in rows:
        if (r.get('team') or '').endswith('TM'):
            continue
        lefty = r.get('throws') == 'L'
        hb = sf(r.get('horzBrk'))
        if hb is not None and lefty:
            hb = -hb
        ti = tilt_minutes(r.get('releaseTiltMinutes'))
        if ti is not None and lefty:
            ti = mirror_tilt(ti)
        grouped[(r['pitcher'], r['team'])].append(
            (sf(r.get('usagePct')), sf(r.get('velocity')),
             sf(r.get('indVertBrk')), hb, ti, sf(r.get('spinRate'))))
    return {k: _mk_arsenal(v) for k, v in grouped.items()}


BATS_CACHE = os.path.join(ROOT, 'data', 'hitter_bats_cache.json')


def _side_from_counts(c):
    """Per-pitch actual-side counts -> L / R / S (>=10% minority = switch)."""
    tot = sum(c.values())
    if len(c) > 1 and min(c.values()) / tot >= 0.10:
        return 'S'
    return max(c, key=c.get)


def hitter_sides():
    """Batter name -> L/R/S, derived from per-pitch Bats (actual side).

    Cached to data/hitter_bats_cache.json since the season leaderboard has no
    bats field; delete that file to rebuild it for new call-ups.
    """
    if os.path.exists(BATS_CACHE):
        return json.load(open(BATS_CACHE))
    counts = defaultdict(lambda: defaultdict(int))
    D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    for p in D:
        b, s = p.get('Batter'), p.get('Bats')
        if b and s in ('L', 'R'):
            counts[b][s] += 1
    out = {b: _side_from_counts(c) for b, c in counts.items()}
    json.dump(out, open(BATS_CACHE, 'w'))
    return out


def load_season(role):
    fn = 'hitter_leaderboard_rs.json' if role == 'hitter' else 'pitcher_leaderboard_rs.json'
    rows = json.load(open(os.path.join(ROOT, 'data', fn)))
    arsenals = load_arsenals() if role == 'pitcher' else {}
    sides = hitter_sides() if role == 'hitter' else {}
    season = []
    for r in rows:
        if (r.get('team') or '').endswith('TM'):
            continue
        if role == 'hitter':
            rec = {f: r.get(f) for f in FEATS_H}
            rec.update(name=r['hitter'], team=r['team'], pa=r.get('pa') or 0,
                       side=sides.get(r['hitter']))
        else:
            rec = {f: r.get(f) for f in FEATS_P}
            haa = r.get('haa')
            rec['haa'] = abs(haa) if haa is not None else None
            rec.update(name=r['pitcher'], team=r['team'], pa=r.get('tbf') or 0,
                       ip=sf(r.get('ip')), side=r.get('throws'),
                       arsenal=arsenals.get((r['pitcher'], r['team'])))
        season.append(rec)
    return season


def _load_cache(start, end):
    D = pickle.load(open(os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl'), 'rb'))
    for p in D:
        d = p.get('Game Date') or ''
        if d < start or (end and d > end):
            continue
        yield p


def window_hitters(start, end=None):
    """Recompute the hitter fingerprint from the pitch cache for the window."""
    agg = defaultdict(lambda: defaultdict(float))
    bsides = defaultdict(lambda: defaultdict(int))
    for p in _load_cache(start, end):
        h, t = p.get('Batter'), p.get('BTeam')
        if not h or not t:
            continue
        a = agg[(h, t)]
        if p.get('Bats') in ('L', 'R'):
            bsides[(h, t)][p['Bats']] += 1
        desc = p.get('Description')
        ev_field = sf(p.get('ExitVelo'))
        a['pitches'] += 1
        in_z = p.get('InZone') == 'Yes'
        if in_z:
            a['iz'] += 1
        is_swing = desc in SWING_DESCRIPTIONS and 'Bunt' not in (desc or '')
        if is_swing:
            a['sw'] += 1
            if in_z:
                a['izsw'] += 1
            else:
                a['oozsw'] += 1
            if desc == 'Swinging Strike':
                a['wh'] += 1
                if in_z:
                    a['izwh'] += 1
            bs, sl = sf(p.get('BatSpeed')), sf(p.get('SwingLength'))
            aa = sf(p.get('AttackAngle'))
            if bs is not None and bs > 50:
                a['bs_sum'] += bs; a['bs_n'] += 1
            if sl is not None:
                a['sl_sum'] += sl; a['sl_n'] += 1
            if aa is not None:
                a['aa_sum'] += aa; a['aa_n'] += 1
        if not in_z:
            a['ooz'] += 1
        ev = p.get('Event')
        if ev and ev not in NON_PA_EVENTS and ev != 'Intent Walk':
            a['pa'] += 1
            if ev in K_EVENTS:
                a['k'] += 1
            elif ev in BB_EVENTS:
                a['bb'] += 1
        bb_type = p.get('BBType')
        if desc == 'In Play' and bb_type and bb_type not in BUNT_BB_TYPES:
            a['bip'] += 1
            if bb_type == 'ground_ball':
                a['gb'] += 1
            if bb_type == 'popup':
                a['pu'] += 1
            if ev_field is not None:
                a['ev_sum'] += ev_field; a['ev_n'] += 1
                if ev_field >= 95:
                    a['hh'] += 1
            try:
                if int(sf(p.get('Barrel')) or 0) == 6:   # lsa code 6 = barrel
                    a['barrel'] += 1
            except (TypeError, ValueError):
                pass
            xw = sf(p.get('xwOBA'))
            if xw is not None:
                a['xw_sum'] += xw; a['xw_n'] += 1
            d_dir = spray_direction(spray_angle(sf(p.get('HC_X')), sf(p.get('HC_Y'))),
                                    p.get('Bats'))
            if d_dir in ('pull', 'pull_side'):
                a['pull'] += 1
                if bb_type in ('line_drive', 'fly_ball'):
                    a['airpull'] += 1
    out = []
    for (h, t), a in agg.items():
        bip, sw, pa_n = a['bip'], a['sw'], a['pa']
        if bip < 25 or sw < 50 or pa_n < 50:
            continue
        out.append(dict(
            name=h, team=t, pa=int(pa_n),
            side=_side_from_counts(bsides[(h, t)]) if bsides[(h, t)] else None,
            avgEVAll=a['ev_sum'] / a['ev_n'] if a['ev_n'] else None,
            hardHitPct=a['hh'] / a['ev_n'] if a['ev_n'] else None,
            barrelPct=a['barrel'] / bip,
            xwOBAcon=a['xw_sum'] / a['xw_n'] if a['xw_n'] else None,
            gbPct=a['gb'] / bip, puPct=a['pu'] / bip,
            pullPct=a['pull'] / bip, airPullPct=a['airpull'] / bip,
            swingPct=sw / a['pitches'],
            izSwingPct=a['izsw'] / a['iz'] if a['iz'] else None,
            chasePct=a['oozsw'] / a['ooz'] if a['ooz'] else None,
            whiffPct=a['wh'] / sw,
            izWhiffPct=a['izwh'] / a['izsw'] if a['izsw'] else None,
            kPct=a['k'] / pa_n, bbPct=a['bb'] / pa_n,
            batSpeed=a['bs_sum'] / a['bs_n'] if a['bs_n'] >= 20 else None,
            swingLength=a['sl_sum'] / a['sl_n'] if a['sl_n'] >= 20 else None,
            attackAngle=a['aa_sum'] / a['aa_n'] if a['aa_n'] >= 20 else None,
        ))
    return out


def window_pitchers(start, end=None):
    """Recompute the pitcher fingerprint from the pitch cache for the window.

    fbVelo / VAA / HAA come from the pitcher's primary fastball (the more-used
    of FF/SI in the window), matching the season leaderboard's FB framing.
    """
    agg = defaultdict(lambda: defaultdict(float))
    throws = {}
    ptypes = defaultdict(set)
    for p in _load_cache(start, end):
        h, t = p.get('Pitcher'), p.get('PTeam')
        if not h or not t:
            continue
        a = agg[(h, t)]
        if p.get('Throws'):
            throws[(h, t)] = p['Throws']
        desc = p.get('Description')
        a['pitches'] += 1
        ext, ang = sf(p.get('Extension')), sf(p.get('ArmAngle'))
        if ext is not None:
            a['ext_sum'] += ext; a['ext_n'] += 1
        if ang is not None:
            a['ang_sum'] += ang; a['ang_n'] += 1
        pt = p.get('Pitch Type')
        if pt and pt not in ('EP', 'PO'):
            ptypes[(h, t)].add(pt)
            a[f'mix_{pt}_n'] += 1
            for fld, key in (('Velocity', 'v'), ('IndVertBrk', 'iv'),
                             ('HorzBrk', 'hb'), ('Spin Rate', 'sp')):
                val = sf(p.get(fld))
                if val is not None:
                    a[f'mix_{pt}_{key}_sum'] += val
                    a[f'mix_{pt}_{key}_n'] += 1
            ti = tilt_minutes(p.get('RTilt'))
            if ti is not None:
                th = ti / TILT_WRAP * 2 * math.pi
                a[f'mix_{pt}_ti_sin'] += math.sin(th)
                a[f'mix_{pt}_ti_cos'] += math.cos(th)
                a[f'mix_{pt}_ti_n'] += 1
        if pt in ('FF', 'SI'):
            v, va, ha = sf(p.get('Velocity')), sf(p.get('VAA')), sf(p.get('HAA'))
            a[f'{pt}_n'] += 1
            if v is not None:
                a[f'{pt}_v_sum'] += v; a[f'{pt}_v_n'] += 1
            if va is not None:
                a[f'{pt}_vaa_sum'] += va; a[f'{pt}_vaa_n'] += 1
            if ha is not None:
                a[f'{pt}_haa_sum'] += ha; a[f'{pt}_haa_n'] += 1
        in_z = p.get('InZone') == 'Yes'
        if in_z:
            a['iz'] += 1
        else:
            a['ooz'] += 1
        is_swing = desc in SWING_DESCRIPTIONS and 'Bunt' not in (desc or '')
        if is_swing:
            a['sw'] += 1
            if in_z:
                a['izsw'] += 1
            else:
                a['oozsw'] += 1
            if desc == 'Swinging Strike':
                a['wh'] += 1
                if in_z:
                    a['izwh'] += 1
        ev = p.get('Event')
        if ev and ev not in NON_PA_EVENTS and ev != 'Intent Walk':
            a['tbf'] += 1
            if ev in K_EVENTS:
                a['k'] += 1
            elif ev in BB_EVENTS:
                a['bb'] += 1
        bb_type = p.get('BBType')
        if desc == 'In Play' and bb_type and bb_type not in BUNT_BB_TYPES:
            a['bip'] += 1
            if bb_type == 'ground_ball':
                a['gb'] += 1
            if bb_type == 'popup':
                a['pu'] += 1
            ev_f = sf(p.get('ExitVelo'))
            if ev_f is not None:
                a['ev_sum'] += ev_f; a['ev_n'] += 1
                if ev_f >= 95:
                    a['hh'] += 1
            try:
                if int(sf(p.get('Barrel')) or 0) == 6:   # lsa code 6 = barrel
                    a['barrel'] += 1
            except (TypeError, ValueError):
                pass
            xw = sf(p.get('xwOBA'))
            if xw is not None:
                a['xw_sum'] += xw; a['xw_n'] += 1
    out = []
    for (h, t), a in agg.items():
        bip, sw, tbf = a['bip'], a['sw'], a['tbf']
        if a['pitches'] < 200 or tbf < 50 or bip < 30:
            continue
        fb = 'FF' if a['FF_n'] >= a['SI_n'] else 'SI'
        haa = (a[f'{fb}_haa_sum'] / a[f'{fb}_haa_n']) if a[f'{fb}_haa_n'] else None
        n_typed = sum(a[f'mix_{pt}_n'] for pt in ptypes[(h, t)])
        lefty = throws.get((h, t)) == 'L'
        entries = []
        for pt in ptypes[(h, t)]:
            means = {k: a[f'mix_{pt}_{k}_sum'] / a[f'mix_{pt}_{k}_n']
                     if a[f'mix_{pt}_{k}_n'] else None
                     for k in ('v', 'iv', 'hb', 'sp')}
            hb = means['hb']
            if hb is not None and lefty:
                hb = -hb
            ti = None
            if a[f'mix_{pt}_ti_n']:
                th = math.atan2(a[f'mix_{pt}_ti_sin'], a[f'mix_{pt}_ti_cos'])
                ti = (th / (2 * math.pi) * TILT_WRAP) % TILT_WRAP
                if lefty:
                    ti = mirror_tilt(ti)
            entries.append((a[f'mix_{pt}_n'] / n_typed if n_typed else 0,
                            means['v'], means['iv'], hb, ti, means['sp']))
        out.append(dict(
            arsenal=_mk_arsenal(entries),
            # no outs-recorded field in the cache; IP estimated at 4.3 TBF/inning
            name=h, team=t, pa=int(tbf), ip=round(tbf / 4.3, 1),
            side=throws.get((h, t)),
            fbVelo=a[f'{fb}_v_sum'] / a[f'{fb}_v_n'] if a[f'{fb}_v_n'] else None,
            vaa=a[f'{fb}_vaa_sum'] / a[f'{fb}_vaa_n'] if a[f'{fb}_vaa_n'] else None,
            haa=abs(haa) if haa is not None else None,
            extension=a['ext_sum'] / a['ext_n'] if a['ext_n'] else None,
            armAngle=a['ang_sum'] / a['ang_n'] if a['ang_n'] else None,
            kPct=a['k'] / tbf, bbPct=a['bb'] / tbf,
            swStrPct=a['wh'] / a['pitches'],
            izWhiffPct=a['izwh'] / a['izsw'] if a['izsw'] else None,
            chasePct=a['oozsw'] / a['ooz'] if a['ooz'] else None,
            izPct=a['iz'] / a['pitches'],
            gbPct=a['gb'] / bip, puPct=a['pu'] / bip,
            avgEVAgainst=a['ev_sum'] / a['ev_n'] if a['ev_n'] else None,
            hardHitPct=a['hh'] / a['ev_n'] if a['ev_n'] else None,
            barrelPctAgainst=a['barrel'] / bip,
            xwOBAcon=a['xw_sum'] / a['xw_n'] if a['xw_n'] else None,
        ))
    return out


def get_rows(role, start=None, end=None):
    """Rows + feats + windowed flag for a role and optional date range."""
    feats = FEATS_H if role == 'hitter' else FEATS_P
    if start or end:
        rows = (window_hitters(start or '2026-01-01', end) if role == 'hitter'
                else window_pitchers(start or '2026-01-01', end))
        return rows, feats, True
    return load_season(role), feats, False


def range_label(start, end):
    if not start and not end:
        return 'full season'
    return f"{start or 'season start'} -> {end or 'now'}"


def big_enough(r, role, args):
    """Batch target floor: min_pa (PA) for hitters, min_ip (IP) for pitchers."""
    if role == 'hitter':
        return r['pa'] >= args.min_pa
    return r.get('ip') is not None and r['ip'] >= args.min_ip


def run_role(role, args):
    """One role's worth of reports (single player or team batch)."""
    nice = NICE_H if role == 'hitter' else NICE_P
    nl, small = NL[role], SMALL_FLAG[role]
    args.level = (args.level or 'both').lower()
    rows, feats, windowed = get_rows(role, args.start, args.end)
    pool_min = (WINDOW_POOL_MIN if windowed else SEASON_POOL_MIN)[role]
    # comp pool is ALWAYS MLB-only: never comp a ROC player to another ROC player
    mlb_pool = [r for r in rows if r['pa'] >= pool_min
                and r['team'] not in AAA_TEAMS]
    stats = zstats(mlb_pool, feats)          # z-scales from the MLB pool
    mix = mix_zstats(mlb_pool) if role == 'pitcher' else None
    rlab = range_label(args.start, args.end)
    meta = dict(role=role, window=rlab)

    def pool_for(t):
        return [r for r in mlb_pool
                if (r['name'], r['team']) != (t['name'], t['team'])]

    if args.team:
        excl = {n.strip() for n in args.exclude.split(';') if n.strip()}
        targets = [r for r in rows if r['team'] == args.team
                   and r['name'] not in excl and big_enough(r, role, args)]
        targets.sort(key=lambda r: -r['pa'])
        floor = (f"{args.min_pa} PA" if role == 'hitter' else f"{args.min_ip} IP")
        if not targets:
            print(f"\n(no {args.team} {role}s over {floor} — {rlab})")
        for t in targets:
            report(t, pool_for(t), feats, stats,
                   f"{t['name']} ({t['team']}, {nl} {t['pa']}) — {rlab}",
                   nice, topn=3, verbose=False, nl=nl, small=small,
                   csvmeta=meta, mix=mix)
        return

    # single player: level picks which of their stat lines to use as the target
    name = args.player
    matches = [r for r in rows if r['name'] == name
               and in_level(r['team'], args.level)]
    if args.player_team:
        matches = [r for r in matches if r['team'] == args.player_team]
    if not matches:
        print(f"\n({name} not found as a {role} at level '{args.level}' — {rlab}"
              f"{' (below window sample floors?)' if windowed else ''})")
        return
    for t in matches:
        pool = pool_for(t)
        report(t, pool, feats, stats,
               f"{name} ({t['team']}) — {rlab.upper()} ({nl} {t['pa']}, "
               f"pool {len(pool)})",
               nice, nl=nl, small=small, csvmeta=meta, mix=mix)


def main():
    if len(sys.argv) == 1:
        interactive()
        return
    ap = argparse.ArgumentParser(description='Closest player comps')
    ap.add_argument('--hitter', default=None, help='"Last, First" single-hitter mode')
    ap.add_argument('--pitcher', default=None, help='"Last, First" single-pitcher mode')
    ap.add_argument('--hitter-team', '--player-team', dest='player_team',
                    default=None, help='pin --hitter/--pitcher to one team row')
    ap.add_argument('--role', choices=['hitter', 'pitcher', 'both'],
                    default='hitter', help='batch mode roles (default hitter)')
    ap.add_argument('--level', choices=['mlb', 'aaa', 'both'], default='both',
                    help="which of the TARGET's stat lines to use (pool is always MLB)")
    ap.add_argument('--start', default=None, help='date window start (recomputed from cache)')
    ap.add_argument('--end', default=None, help='date window end (inclusive)')
    ap.add_argument('--team', default=None, help='batch mode: all players of this team')
    ap.add_argument('--exclude', default='', help='batch mode: semicolon-separated names to skip')
    ap.add_argument('--min-pa', type=int, default=100,
                    help='batch mode: min PA to include a hitter target')
    ap.add_argument('--min-ip', type=float, default=25,
                    help='batch mode: min IP to include a pitcher target')
    ap.add_argument('--csv', default=None,
                    help='write top-3 comps + reasons to this CSV path')
    args = ap.parse_args()

    if args.team:
        args.player = None
        for r in (['hitter', 'pitcher'] if args.role == 'both' else [args.role]):
            run_role(r, args)
    else:
        single = [('hitter', args.hitter), ('pitcher', args.pitcher)]
        ran = False
        for role, name in single:
            if not name:
                continue
            args.player = name
            # season report first, then the window if a range was given
            s, e = args.start, args.end
            args.start = args.end = None
            run_role(role, args)
            if s or e:
                args.start, args.end = s, e
                run_role(role, args)
                args.start = args.end = None
            args.start, args.end = s, e
            ran = True
        if not ran:
            args.player = 'Ford, Harry'
            run_role('hitter', args)
    if args.csv:
        write_csv(args.csv)


# ═══════════════════════════════════════════════════════════════════════════
# SELECTION
# ═══════════════════════════════════════════════════════════════════════════
def interactive():
    # — Settings (edit these directly, or pass command-line flags instead) —
    role       = "both"     # "hitter", "pitcher", or "both"
    player     = ""         # "Last, First" for one player, or "" for whole team
    team       = "ROC"      # batch mode: team whose players get comped
    level      = "both"    # which of the TARGET's stat lines to use: "mlb",
                            #   "aaa", or "both" (comp pool is ALWAYS MLB-only)
    start_date = None       # "yyyy-mm-dd", or None for full season
    end_date   = None       # "yyyy-mm-dd", or None for through today
    exclude    = ""         # batch mode: semicolon-separated names to skip
    min_pa     = 100        # batch mode: min PA to include a hitter
    min_ip     = 25         # batch mode: min IP to include a pitcher
    csv_path   = "~/Downloads/roc_comps.csv"   # "" = no CSV

    args = argparse.Namespace(player=player or None, player_team=None,
                              team=None if player else team,
                              level=level, start=start_date, end=end_date,
                              exclude=exclude, min_pa=min_pa, min_ip=min_ip)
    for r in (['hitter', 'pitcher'] if role == 'both' else [role]):
        run_role(r, args)
    if csv_path:
        write_csv(csv_path)


if __name__ == '__main__':
    main()
