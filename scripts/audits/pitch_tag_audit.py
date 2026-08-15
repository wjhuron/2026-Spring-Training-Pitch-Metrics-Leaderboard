"""pitch_tag_audit.py — audit retagged pitch classifications for likely errors.

READ-ONLY. Produces ~/Downloads/pitch_tag_audit_2026.xlsx.

Two audits:
  GOAL 1 (per-pitch, within-game): for each pitcher, build stable season
    per-type centroids, then test every individual pitch: is it closer to a
    DIFFERENT type this pitcher throws than to its own tag? Confidence is
    boosted when several pitches of the same tag in the SAME game flip the
    same direction (a real within-game trend, not tracking noise). One tab
    per team, pitchers alphabetical, Medium+ confidence only.

  GOAL 2 (whole pitch type): for each pitcher x type cluster, compare its
    profile to league per-hand type prototypes; flag clusters whose profile
    matches a different label better (e.g. Rico Garcia SL -> FC). FF<->SI
    swaps excluded per Wally.

Metrics: Velocity, Spin Rate, RTilt, OTilt (circular, clock->deg), IndVertBrk
(IVB), HorzBrk (HB), ArmAngle. Distances are RMS standardized z over available
metrics, with each axis scaled by its natural within-cluster noise (Goal 1) or
between-pitcher type spread (Goal 2), and per-metric reliability weights.

Usage:
    python3 scripts/audits/pitch_tag_audit.py            # diagnostics to stdout
    python3 scripts/audits/pitch_tag_audit.py --xlsx     # also write the workbook
"""
import os, sys, math, pickle, statistics as st
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
OUT = os.path.expanduser('~/Downloads/pitch_tag_audit_2026.xlsx')

# ---- metric config -------------------------------------------------------
LINEAR = ['Velocity', 'Spin Rate', 'IndVertBrk', 'HorzBrk', 'ArmAngle']
CIRC = ['RTilt', 'OTilt']
METRICS = LINEAR + CIRC
WEIGHT = {  # reliability weight per axis
    'Velocity': 1.0, 'IndVertBrk': 1.0, 'HorzBrk': 1.0, 'OTilt': 1.0,
    'RTilt': 0.8, 'Spin Rate': 0.5, 'ArmAngle': 0.4,
}

# ---- Goal 1 thresholds ---------------------------------------------------
MIN_STABLE = 10   # cluster size to contribute to noise-scale estimation
MIN_CLUSTER = 6   # per-pitcher per-type cluster size to be a valid centroid
G1_MARGIN = 1.5   # min (d_own - d_best) in RMS-z to flag
G1_DBEST = 2.0    # pitch must genuinely look like the target
G1_DSELF = 2.5    # pitch must genuinely NOT look like its own tag
MIN_METRICS = 5   # of the 7 axes a pitch must actually have measured.
#   Without this, a record whose entire tracking row is missing except ArmAngle
#   is scored on that one weak axis (discriminability 1.07 vs 14.9 for RTilt),
#   and n_agree returns 1/1, which maxes the agreement term in add_confidence.
#   Every such flag landed in the High tier. The observed `tot` distribution is
#   bimodal with a hole (1, then 6, then 7 -- nothing between 2 and 5), so any
#   value in [2, 6] is equivalent on current data; 5 is a convention chosen for
#   headroom, not a measured optimum.

# ---- Goal 2 thresholds ---------------------------------------------------
MIN_TYPE = 12      # pitcher-type cluster size to audit as a whole
G2_LEAGUE_MIN = 250  # league (hand,type) sample to be a prototype
G2_MARGIN = 1.2    # min (d_label - d_best) in type-spread z
G2_DBEST = 1.6     # cluster must look like the suggested prototype
G2_EXISTING_MIN = 6  # if the pitcher already throws the target type this many
#   times, don't suggest merging another type into it: the two are distinct
#   pitches (e.g. don't reclass an SL->FC when a real FC cluster already exists).

NEVER_SWAP = frozenset([frozenset(('FF', 'SI'))])  # excluded both goals

# A cutter sits close to the fastball; a gyro slider is much slower even
# when its shape/tilt looks cutter-like. Only suggest ->FC if the pitch is
# within this many mph of the pitcher's own primary fastball.
FC_MAX_GAP = 7.5


def sf(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def clock_deg(x):
    if not x or ':' not in str(x):
        return None
    try:
        h, m = str(x).split(':')
        return ((int(h) % 12) * 60 + int(m)) * 0.5
    except ValueError:
        return None


def deg_clock(deg):
    if deg is None:
        return ''
    total = round(deg / 0.5)  # minutes on 12h face
    h = (total // 60) % 12
    if h == 0:
        h = 12
    m = total % 60
    return f"{h}:{m:02d}"


def cdiff(a, b):
    if a is None or b is None:
        return None
    dd = abs(a - b) % 360
    return dd if dd <= 180 else 360 - dd


def cmean(degs):
    degs = [g for g in degs if g is not None]
    if not degs:
        return None
    s = sum(math.sin(math.radians(g)) for g in degs)
    c = sum(math.cos(math.radians(g)) for g in degs)
    if s == 0 and c == 0:
        return None
    return math.degrees(math.atan2(s, c)) % 360


def mvals(p):
    """metric-value dict for a pitch (tilt in degrees)."""
    d = {}
    for m in LINEAR:
        d[m] = sf(p.get(m))
    for m in CIRC:
        d[m] = clock_deg(p.get(m))
    return d


def centroid(pitch_mvals):
    """centroid dict from a list of metric-value dicts."""
    c = {}
    for m in LINEAR:
        vv = [d[m] for d in pitch_mvals if d[m] is not None]
        c[m] = st.median(vv) if vv else None
    for m in CIRC:
        c[m] = cmean([d[m] for d in pitch_mvals])
    return c


def dist(pv, cen, scales):
    """RMS standardized z-distance from pitch metric dict pv to centroid cen."""
    ss = 0.0
    wsum = 0.0
    for m in METRICS:
        a, b = pv.get(m), cen.get(m)
        if a is None or b is None:
            continue
        sc = scales.get(m)
        if not sc:
            continue
        dv = cdiff(a, b) if m in CIRC else (a - b)
        z = dv / sc
        w = WEIGHT[m]
        ss += w * z * z
        wsum += w
    if wsum == 0:
        return None
    return math.sqrt(ss / wsum)


SHORT = {'Velocity': 'velo', 'Spin Rate': 'spin', 'IndVertBrk': 'IVB',
         'HorzBrk': 'HB', 'ArmAngle': 'arm', 'RTilt': 'RTilt', 'OTilt': 'OTilt'}


def build_why(vals, own_cen, tgt_cen, own_label, tgt_label, scales, topk=3):
    """string naming the metrics that most favor the suggested type."""
    contribs = []
    for m in METRICS:
        a, co, ct = vals.get(m), own_cen.get(m), tgt_cen.get(m)
        if a is None or co is None or ct is None:
            continue
        sc = scales.get(m) or 1.0
        do = (cdiff(a, co) if m in CIRC else abs(a - co)) / sc
        dt = (cdiff(a, ct) if m in CIRC else abs(a - ct)) / sc
        contribs.append((do - dt, m, a, co, ct))  # positive -> favors target
    contribs.sort(reverse=True)
    parts = []
    for delta, m, a, co, ct in contribs[:topk]:
        if delta <= 0:
            break
        if m in CIRC:
            parts.append(f"{SHORT[m]} {deg_clock(a)} "
                         f"({own_label} {deg_clock(co)}/{tgt_label} {deg_clock(ct)})")
        else:
            parts.append(f"{SHORT[m]} {a:.1f} "
                         f"({own_label} {co:.1f}/{tgt_label} {ct:.1f})")
    return "; ".join(parts)


def n_agree(pv, own, tgt, scales):
    """how many metrics individually favor tgt over own, and total available."""
    agree = 0
    total = 0
    for m in METRICS:
        a = pv.get(m)
        co, ct = own.get(m), tgt.get(m)
        if a is None or co is None or ct is None:
            continue
        do = cdiff(a, co) if m in CIRC else abs(a - co)
        dt = cdiff(a, ct) if m in CIRC else abs(a - ct)
        total += 1
        if dt < do:
            agree += 1
    return agree, total


# =========================================================================
def load():
    d = pickle.load(open(CACHE, 'rb'))
    subj = [p for p in d if p.get('_source') in ('MLB', 'ROC')]
    for p in subj:
        p['_mv'] = mvals(p)
    return subj


def noise_scales(subj):
    """robust within-cluster noise per metric (residuals about pitcher-type
    centroids of stable clusters)."""
    clusters = defaultdict(list)
    for p in subj:
        clusters[(p['Pitcher'], p['Pitch Type'])].append(p['_mv'])
    resid = defaultdict(list)
    for key, mvs in clusters.items():
        if len(mvs) < MIN_STABLE:
            continue
        cen = centroid(mvs)
        for d in mvs:
            for m in METRICS:
                if d[m] is None or cen[m] is None:
                    continue
                r = cdiff(d[m], cen[m]) if m in CIRC else (d[m] - cen[m])
                resid[m].append(abs(r))
    scales = {}
    for m in METRICS:
        rr = resid[m]
        scales[m] = 1.4826 * st.median(rr) if rr else 1.0
        if scales[m] <= 0:
            scales[m] = 1.0
    return scales


def pitcher_centroids(subj):
    """{pitcher: {type: (centroid, n)}} for stable clusters."""
    byp = defaultdict(lambda: defaultdict(list))
    for p in subj:
        byp[p['Pitcher']][p['Pitch Type']].append(p['_mv'])
    out = {}
    for pit, types in byp.items():
        cd = {}
        for pt, mvs in types.items():
            if len(mvs) >= MIN_CLUSTER:
                cd[pt] = (centroid(mvs), len(mvs))
        if len(cd) >= 2:
            out[pit] = cd
    return out


def fastball_velos(subj):
    """{pitcher: primary fastball median velo} (FF preferred, else SI)."""
    byp = defaultdict(lambda: defaultdict(list))
    for p in subj:
        v = sf(p.get('Velocity'))
        if v is not None:
            byp[p['Pitcher']][p['Pitch Type']].append(v)
    out = {}
    for pit, types in byp.items():
        for fb in ('FF', 'SI'):
            if len(types.get(fb, [])) >= MIN_CLUSTER:
                out[pit] = st.median(types[fb])
                break
    return out


def fc_ok(fb_velo, pitch_velo):
    """True if a ->FC suggestion is velocity-plausible (or unjudgeable)."""
    if fb_velo is None or pitch_velo is None:
        return True
    return (fb_velo - pitch_velo) <= FC_MAX_GAP


def league_prototypes(subj):
    """{hand: {type: (centroid, spread_scales, n)}} and pooled type-spread."""
    byht = defaultdict(list)
    for p in subj:
        byht[(p['Throws'], p['Pitch Type'])].append(p['_mv'])
    protos = defaultdict(dict)
    # between-pitcher type-mean spread: per (hand,type), spread of per-pitcher
    # cluster centroids around the league centroid.
    pit_of = defaultdict(lambda: defaultdict(list))
    for p in subj:
        pit_of[(p['Throws'], p['Pitch Type'])][p['Pitcher']].append(p['_mv'])
    spread_resid = defaultdict(list)
    league_cen = {}
    for (hand, pt), mvs in byht.items():
        if len(mvs) < G2_LEAGUE_MIN:
            continue
        cen = centroid(mvs)
        league_cen[(hand, pt)] = cen
        for pit, pmvs in pit_of[(hand, pt)].items():
            if len(pmvs) < MIN_CLUSTER:
                continue
            pc = centroid(pmvs)
            for m in METRICS:
                if pc[m] is None or cen[m] is None:
                    continue
                r = cdiff(pc[m], cen[m]) if m in CIRC else (pc[m] - cen[m])
                spread_resid[m].append(abs(r))
    spread = {}
    for m in METRICS:
        rr = spread_resid[m]
        spread[m] = 1.4826 * st.median(rr) if rr else 1.0
        if spread[m] <= 0:
            spread[m] = 1.0
    for (hand, pt), cen in league_cen.items():
        protos[hand][pt] = (cen, len(byht[(hand, pt)]))
    return protos, spread


# =========================================================================
def audit_goal1(subj, scales, pcen, fbv):
    """returns list of per-pitch flag dicts (pre-confidence)."""
    flags = []
    for p in subj:
        pit = p['Pitcher']
        cd = pcen.get(pit)
        if not cd:
            continue
        own = p['Pitch Type']
        if own not in cd:
            continue
        pv = p['_mv']
        if sum(1 for m in METRICS if pv.get(m) is not None) < MIN_METRICS:
            continue  # sparse tracking record -- see MIN_METRICS
        d_own = dist(pv, cd[own][0], scales)
        if d_own is None:
            continue
        best_t, best_d = None, 1e9
        for pt, (cen, n) in cd.items():
            if pt == own:
                continue
            if frozenset((own, pt)) in NEVER_SWAP:
                continue
            if pt == 'FC' and not fc_ok(fbv.get(pit), pv.get('Velocity')):
                continue
            dd = dist(pv, cen, scales)
            if dd is not None and dd < best_d:
                best_d, best_t = dd, pt
        if best_t is None:
            continue
        margin = d_own - best_d
        if margin < G1_MARGIN or best_d > G1_DBEST or d_own < G1_DSELF:
            continue
        agree, tot = n_agree(pv, cd[own][0], cd[best_t][0], scales)
        flags.append({
            'pitcher': pit, 'team': p.get('PTeam'), 'opp': p.get('BTeam'),
            'date': p.get('Game Date'), 'game': str(p.get('PitchID', '')).split('_')[0],
            'pid': p.get('PitchID'), 'own': own, 'tgt': best_t,
            'd_own': d_own, 'd_best': best_d, 'margin': margin,
            'agree': agree, 'tot': tot, 'mv': pv,
            'why': build_why(pv, cd[own][0], cd[best_t][0], own, best_t, scales),
        })
    return flags


def add_confidence(flags, subj):
    """within-game reinforcement + confidence score/tier."""
    # count same-direction flips per (pitcher, game, own->tgt), and total
    # own-type pitches thrown in that game.
    grp = Counter()
    for f in flags:
        grp[(f['pitcher'], f['game'], f['own'], f['tgt'])] += 1
    own_in_game = Counter()
    for p in subj:
        own_in_game[(p['Pitcher'], str(p.get('PitchID', '')).split('_')[0],
                     p['Pitch Type'])] += 1
    for f in flags:
        k = (f['pitcher'], f['game'], f['own'], f['tgt'])
        nflip = grp[k]
        ntype = own_in_game[(f['pitcher'], f['game'], f['own'])] or 1
        frac = nflip / ntype
        f['nflip'] = nflip
        f['ntype'] = ntype
        f['frac'] = frac
        m = min(1.0, f['margin'] / 3.0)
        a = (f['agree'] / f['tot']) if f['tot'] else 0.0
        # cluster reinforcement: reward multiple same-direction flips + high fraction
        c = 0.5 * min(1.0, (nflip - 1) / 3.0) + 0.5 * min(1.0, frac / 0.5)
        score = 100 * (0.50 * m + 0.30 * a + 0.20 * c)
        f['conf'] = round(score)
        f['tier'] = 'High' if score >= 75 else 'Medium' if score >= 50 else 'Low'
    return flags


def audit_goal2(subj, protos, spread, fbv):
    """whole pitch-type reclass candidates."""
    byp = defaultdict(lambda: defaultdict(list))
    hand_of = {}
    team_of = defaultdict(Counter)
    for p in subj:
        byp[p['Pitcher']][p['Pitch Type']].append(p['_mv'])
        hand_of[p['Pitcher']] = p['Throws']
        team_of[p['Pitcher']][p.get('PTeam')] += 1
    out = []
    for pit, types in byp.items():
        hand = hand_of[pit]
        lib = protos.get(hand, {})
        for pt, mvs in types.items():
            if len(mvs) < MIN_TYPE or pt not in lib:
                continue
            cen = centroid(mvs)
            if sum(1 for m in METRICS if cen[m] is not None) < MIN_METRICS:
                continue  # latent form of the Goal 1 sparse-axis bug: centroid()
            #   yields None for any metric absent across the cluster and dist()
            #   silently skips it, so the comparison could rest on 1-2 axes.
            #   Measured: 28 of 3089 audited clusters are this thin and every one
            #   is a position player's EP. None currently reach the gates, so this
            #   removes a latent failure mode at zero cost to real flags.
            d_label = dist(cen, lib[pt][0], spread)
            if d_label is None:
                continue
            best_t, best_d = None, 1e9
            for lt, (lcen, ln) in lib.items():
                if lt == pt or frozenset((pt, lt)) in NEVER_SWAP:
                    continue
                if len(types.get(lt, [])) >= G2_EXISTING_MIN:
                    continue  # pitcher already throws this type as its own pitch
                if lt == 'FC' and not fc_ok(fbv.get(pit), cen.get('Velocity')):
                    continue
                dd = dist(cen, lcen, spread)
                if dd is not None and dd < best_d:
                    best_d, best_t = dd, lt
            if best_t is None:
                continue
            margin = d_label - best_d
            if margin < G2_MARGIN or best_d > G2_DBEST:
                continue
            m = min(1.0, margin / 2.5)
            nfac = min(1.0, len(mvs) / 40.0)
            score = 100 * (0.70 * m + 0.30 * nfac)
            # tier driven by margin (strength of the whole-type signal); a big
            # sample can't turn a weak profile gap into "High".
            tier = 'High' if margin >= 1.8 else 'Medium' if margin >= 1.2 else 'Low'
            out.append({
                'pitcher': pit, 'team': team_of[pit].most_common(1)[0][0],
                'hand': hand, 'type': pt, 'n': len(mvs),
                'tgt': best_t, 'd_label': d_label, 'd_best': best_d,
                'margin': margin, 'conf': round(score), 'tier': tier,
                'cen': cen,
                'why': build_why(cen, lib[pt][0], lib[best_t][0], pt, best_t, spread),
            })
    return out


# =========================================================================
def diagnostics(scales, spread, g1, g2):
    print("\n=== within-cluster noise scales (Goal 1 units) ===")
    for m in METRICS:
        print(f"  {m:12s} {scales[m]:.3f}")
    print("\n=== between-pitcher type spread (Goal 2 units) ===")
    for m in METRICS:
        print(f"  {m:12s} {spread[m]:.3f}")

    tiers = Counter(f['tier'] for f in g1)
    print(f"\n=== GOAL 1: {len(g1)} raw flags  tiers={dict(tiers)} ===")
    swaps = Counter((f['own'], f['tgt']) for f in g1 if f['tier'] in ('High', 'Medium'))
    print("  Medium+ swap directions:", swaps.most_common(15))
    top = sorted([f for f in g1 if f['tier'] in ('High', 'Medium')],
                 key=lambda f: -f['conf'])[:15]
    print("  top Medium+ examples:")
    for f in top:
        print(f"   {f['conf']:3d} {f['tier']:6s} {f['pitcher']:22s} {f['date']} "
              f"{f['own']}->{f['tgt']}  margin{f['margin']:.1f} agree{f['agree']}/{f['tot']} "
              f"gameflip{f['nflip']}/{f['ntype']}")

    t2 = Counter(x['tier'] for x in g2)
    print(f"\n=== GOAL 2: {len(g2)} raw type flags  tiers={dict(t2)} ===")
    sw2 = Counter((x['type'], x['tgt']) for x in g2 if x['tier'] in ('High', 'Medium'))
    print("  Medium+ type swaps:", sw2.most_common(20))
    for x in sorted([x for x in g2 if x['tier'] in ('High', 'Medium')],
                    key=lambda x: -x['conf'])[:25]:
        print(f"   {x['conf']:3d} {x['tier']:6s} {x['pitcher']:22s} {x['hand']} "
              f"{x['type']}->{x['tgt']}  n={x['n']:4d} margin{x['margin']:.1f}")


def main():
    subj = load()
    print(f"Loaded {len(subj)} subject pitches "
          f"({len(set(p['Pitcher'] for p in subj))} pitchers)")
    scales = noise_scales(subj)
    pcen = pitcher_centroids(subj)
    protos, spread = league_prototypes(subj)
    fbv = fastball_velos(subj)
    g1 = add_confidence(audit_goal1(subj, scales, pcen, fbv), subj)
    g2 = audit_goal2(subj, protos, spread, fbv)
    diagnostics(scales, spread, g1, g2)
    if '--xlsx' in sys.argv:
        from audit_write import write_workbook
        write_workbook(OUT, g1, g2)
        print(f"\nWrote {OUT}")


if __name__ == '__main__':
    main()
