"""roc_tag_audit.py -- ROC mistag audit, compared WITHIN individual games.

READ-ONLY. Covers the same pitcher-date outings as the ROC pitcher-reports PDF,
but flags INDIVIDUAL pitches instead of per-outing aggregates.

WHY GAME-LEVEL. A mistag is a within-game labeling error, so the right
reference is the pitcher's other pitches IN THAT GAME, not a season centroid.
Measured on ROC, 22-44% of each metric's season-scale variance is pure
game-to-game drift (park, weather, tracking calibration, day-to-day stuff):

    metric      season   in-game   drift share
    Velocity     1.186     0.890       44%
    Spin Rate   89.697    68.941       41%
    ArmAngle     2.669     2.076       40%
    HorzBrk      2.150     1.779       32%
    RTilt        6.639     5.495       31%
    IndVertBrk   1.927     1.631       28%
    OTilt       10.300     9.079       22%

Comparing within the game removes that drift from the noise floor, so genuine
mistags stand out further. Validated by synthetic injection (--bench): at a
matched false-flag rate, game centroids recover ~97% of injected mistags vs
~88% for season centroids.

DESIGN DECISIONS, EACH MEASURED (see --bench to reproduce):

  * LEAVE-ONE-OUT centroids. At in-game cluster sizes (median ~8) a pitch
    included in its own centroid drags that centroid toward itself and hides
    the very deviation being tested. Every centroid here excludes the pitch
    under test.

  * NO d_own GATE. Swept 1.5/2.0/2.5/3.0: results identical across all of them.
    Once centroids are game-level, any pitch clearing the margin gate already
    has a large d_own. The gate was inert, so it is gone.

  * NO centroid-uncertainty correction. Inflating scales by sqrt(1+1/n) for
    small in-game clusters was tested and was neutral-to-worse at every
    operating point. Rejected.

  * MIN_CLUSTER = 3 in-game. Swept 3/4/5/6; 3 and 4 were equivalent on the
    recall/FP frontier and 3 covers more pitches (96.3% vs 93.0%).

  * RELEASE IS A DIAGNOSTIC, NOT AN AXIS. Discriminability (between-type
    separation / within-type noise) is 0.84 for RelPosZ and 0.74 for RelPosX,
    both below 1.0: release varies more pitch-to-pitch within one pitch type
    than it differs between a pitcher's types. RelPosZ is also r=+0.79 with
    ArmAngle. Including them adds noise. They are reported so a flagged pitch
    with an anomalous release reads as a tracking glitch, not a mistag.

  * MIN_METRICS guard. Without it, records whose entire tracking row is missing
    except ArmAngle get flagged "High" off that one weak axis (agree/tot = 1/1
    maxes the agreement term). That produced every flag in the first version of
    this audit, all bogus.

THRESHOLDS ARE A CONVENTION. Recall rises monotonically in D_BEST_MAX and
saturates near 98.7% while the false-flag rate keeps climbing; there is no
interior optimum because recall and review burden have no stated exchange rate.
The frontier measured on ROC, reproduced on MLB as an independent replicate:

    d_best <= 2.5  ->  91.4% recall, 0.10 false flags per 1000
    d_best <= 3.0  ->  94.7% recall, 0.43            (shipped)
    d_best <= 4.0  ->  96.3% recall, 0.92
    d_best <= 5.0  ->  97.4% recall, 1.06

3.0 is chosen as the convention: ~5 expected false flags across ROC's ~12.3k
comparable pitches is a reviewable burden. Change D_BEST_MAX to move along the
frontier; --sweep reprints it.

Usage:
    python3 scripts/audits/roc_tag_audit.py            # ranked report
    python3 scripts/audits/roc_tag_audit.py --sweep    # threshold frontier
    python3 scripts/audits/roc_tag_audit.py --bench    # injection validation
    python3 scripts/audits/roc_tag_audit.py --csv      # + ~/Downloads CSV
    python3 scripts/audits/roc_tag_audit.py --plots    # + adjudication PNGs
"""
import os, sys, math, random, statistics as st
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pitch_tag_audit as A

TEAM = 'ROC'
REL = ['RelPosZ', 'RelPosX']
MIN_METRICS = 5      # of the 7 discriminating axes
MIN_CLUSTER = 3      # in-game pitches of a type needed to form a centroid
MARGIN_MIN = 1.0     # d_own - d_best, in within-game RMS-z
D_BEST_MAX = 3.0     # pitch must genuinely look like the target
D_BEST_REVIEW = 4.0  # wider band surfaced for manual adjudication
ORPHAN_DBEST = 2.0   # MEASURED, not reasoned. Injecting orphan mistags (relabel
#   a pitch to a type with no other instances that game, which is exactly how a
#   real mistag creates an orphan) gives a sharp precision cliff:
#       d_best <= 2.0   86.2% recall, 0.16 false flags per 1000 orphans
#       d_best <= 2.5   92.1% recall, 6.17   <-- 39x worse
#       d_best <= 3.0   94.6% recall, 19.97
#   ROC has ~440 real orphans, so 2.5 expects ~2.7 false flags and produced
#   exactly 3 -- indistinguishable from noise. An earlier version of this file
#   loosened the gate to 2.5 on the argument that 2.5 is where a pitch becomes
#   statistically indistinguishable from the cluster it sits near. That argument
#   was plausible and wrong; the injection curve settles it.
ORPHAN_REVIEW = 2.5  # surfaced for eyeballing, never tiered above Low
SCALE_MIN_N = 6      # cluster size contributing to noise-scale estimation
CSV_TIERS = {'High', 'Medium'}   # stdout still prints every tier
PLOTDIR = os.path.expanduser('~/Downloads/roc_tag_plots')

# pairs that actually get confused in practice, used by --bench injection
HARD_PAIRS = {frozenset(x) for x in [
    ('FF', 'FC'), ('FC', 'SL'), ('SL', 'ST'), ('SL', 'CU'), ('ST', 'CU'),
    ('CH', 'FS'), ('SI', 'CH'), ('SI', 'FS'), ('FC', 'ST'), ('FF', 'FS')]}


def prep(rows):
    for p in rows:
        p['_g'] = str(p.get('PitchID', '')).split('_')[0]
        p['_navail'] = sum(1 for m in A.METRICS if p['_mv'].get(m) is not None)
    return [p for p in rows if p['_navail'] >= MIN_METRICS]


def game_scales(rows):
    """within-GAME noise per metric: residuals about same-game type centroids."""
    cl = defaultdict(list)
    for p in rows:
        cl[(p['Pitcher'], p['_g'], p['Pitch Type'])].append(p['_mv'])
    resid = defaultdict(list)
    for _, mvs in cl.items():
        if len(mvs) < SCALE_MIN_N:
            continue
        cen = A.centroid(mvs)
        for d in mvs:
            for m in A.METRICS:
                a, b = d[m], cen[m]
                if a is None or b is None:
                    continue
                resid[m].append(abs(A.cdiff(a, b) if m in A.CIRC else a - b))
    out = {}
    for m in A.METRICS:
        rr = resid[m]
        out[m] = (1.4826 * st.median(rr)) if rr else 1.0
        if out[m] <= 0:
            out[m] = 1.0
    return out


def loo_centroid(ps, skip):
    mvs = [q['_mv'] for q in ps if q is not skip]
    return A.centroid(mvs) if mvs else None


def orphan_why(vals, tgt_cen, tgt_label, scl, topk=3):
    """An orphan has no own-centroid to contrast against, so report the axes on
    which it sits CLOSEST to the target cluster."""
    fits = []
    for m in A.METRICS:
        a, c = vals.get(m), tgt_cen.get(m)
        if a is None or c is None or not scl.get(m):
            continue
        z = (A.cdiff(a, c) if m in A.CIRC else abs(a - c)) / scl[m]
        fits.append((z, m, a, c))
    fits.sort()
    parts = []
    for z, m, a, c in fits[:topk]:
        if m in A.CIRC:
            parts.append(f"{A.SHORT[m]} {A.deg_clock(a)} (vs {tgt_label} "
                         f"{A.deg_clock(c)}, {z:.1f}z)")
        else:
            parts.append(f"{A.SHORT[m]} {a:.1f} (vs {tgt_label} {c:.1f}, {z:.1f}z)")
    return "matches " + tgt_label + " on: " + "; ".join(parts)


def score_game(rows, scl, tagkey='Pitch Type'):
    """Every pitch vs its OWN GAME's type centroids (leave-one-out).

    Returns (results, orphans):
      results[id(pitch)] = dict(d_own, d_best, margin, tgt, agree, tot)
      orphans            = list of dicts for pitches whose in-game type cluster
                           is too small to form a centroid.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for p in rows:
        groups[(p['Pitcher'], p['_g'])][p[tagkey]].append(p)
    res, orph = {}, []
    for _, types in groups.items():
        valid = {t: ps for t, ps in types.items() if len(ps) >= MIN_CLUSTER}
        # --- normal path: type has a usable in-game cluster ---
        if len(valid) >= 2:
            for own, ps in valid.items():
                for q in ps:
                    cen = loo_centroid(ps, q)
                    d_own = A.dist(q['_mv'], cen, scl) if cen else None
                    if d_own is None:
                        continue
                    bt, bd, bc = None, 1e9, None
                    for t, ps2 in valid.items():
                        if t == own or frozenset((own, t)) in A.NEVER_SWAP:
                            continue
                        c2 = loo_centroid(ps2, q)
                        dd = A.dist(q['_mv'], c2, scl) if c2 else None
                        if dd is not None and dd < bd:
                            bd, bt, bc = dd, t, c2
                    if bt is None:
                        continue
                    ag, tot = A.n_agree(q['_mv'], cen, bc, scl)
                    res[id(q)] = {'p': q, 'own': own, 'tgt': bt, 'd_own': d_own,
                                  'd_best': bd, 'margin': d_own - bd,
                                  'agree': ag, 'tot': tot, 'kind': 'cluster',
                                  'why': A.build_why(q['_mv'], cen, bc, own, bt, scl)}
        # --- orphan path: type has 1-2 pitches this game ---
        if not valid:
            continue
        for own, ps in types.items():
            if len(ps) >= MIN_CLUSTER:
                continue
            for q in ps:
                bt, bd, bc = None, 1e9, None
                for t, ps2 in valid.items():
                    if t == own or frozenset((own, t)) in A.NEVER_SWAP:
                        continue
                    c2 = loo_centroid(ps2, q)
                    dd = A.dist(q['_mv'], c2, scl) if c2 else None
                    if dd is not None and dd < bd:
                        bd, bt, bc = dd, t, c2
                if bt is None:
                    continue
                orph.append({'p': q, 'own': own, 'tgt': bt, 'd_own': None,
                             'd_best': bd, 'margin': None, 'kind': 'orphan',
                             'nsame': len(ps), 'agree': None, 'tot': None,
                             'why': orphan_why(q['_mv'], bc, bt, scl)})
    return res, orph


def release_stats(rows):
    cl = defaultdict(list)
    for p in rows:
        cl[(p['Pitcher'], p['Pitch Type'])].append(p)
    out = {}
    for k, ps in cl.items():
        if len(ps) < 6:
            continue
        stat = []
        for m in REL:
            vv = [A.sf(x.get(m)) for x in ps]
            vv = [v for v in vv if v is not None]
            if len(vv) < 6:
                stat = None
                break
            med = st.median(vv)
            scl = 1.4826 * st.median([abs(v - med) for v in vv]) or 0.05
            stat.append((med, scl))
        if stat:
            out[k] = stat
    return out


def rel_outlier(p, rs):
    k = (p['Pitcher'], p['Pitch Type'])
    if k not in rs:
        return None
    zs = []
    for (med, scl), m in zip(rs[k], REL):
        v = A.sf(p.get(m))
        if v is not None:
            zs.append(abs(v - med) / scl)
    return max(zs) if zs else None


def game_drift(rows, scl):
    """per (pitcher,game,type): shift of the whole in-game cluster vs season.
    A large shift with no per-pitch flags = calibration/park, not a mistag."""
    season = defaultdict(list)
    for p in rows:
        season[(p['Pitcher'], p['Pitch Type'])].append(p['_mv'])
    scen = {k: A.centroid(v) for k, v in season.items() if len(v) >= 20}
    gcl = defaultdict(list)
    for p in rows:
        gcl[(p['Pitcher'], p['_g'], p['Pitch Type'], p['Game Date'])].append(p['_mv'])
    out = []
    for (pit, g, pt, dt), mvs in gcl.items():
        if len(mvs) < 8 or (pit, pt) not in scen:
            continue
        d = A.dist(A.centroid(mvs), scen[(pit, pt)], scl)
        if d is not None:
            out.append((d, pit, dt, pt, len(mvs)))
    return sorted(out, reverse=True)


def confidence(flags, rows):
    grp = Counter()
    for f in flags:
        grp[(f['p']['Pitcher'], f['p']['_g'], f['own'], f['tgt'])] += 1
    own_in_game = Counter()
    for p in rows:
        own_in_game[(p['Pitcher'], p['_g'], p['Pitch Type'])] += 1
    for f in flags:
        p = f['p']
        nflip = grp[(p['Pitcher'], p['_g'], f['own'], f['tgt'])]
        ntype = own_in_game[(p['Pitcher'], p['_g'], f['own'])] or 1
        f['nflip'], f['ntype'] = nflip, ntype
        if f['kind'] == 'orphan':
            # Tier straight off the measured precision cliff (see ORPHAN_DBEST),
            # not off an invented continuous formula. The previous formula could
            # not exceed ~40 in practice, so no orphan could ever reach Medium
            # and any "Medium+" filter silently deleted the whole mechanism.
            d = f['d_best']
            s = 82 if d <= 1.75 else 62 if d <= ORPHAN_DBEST else 30
        else:
            m = min(1.0, f['margin'] / 3.0)
            a = (f['agree'] / f['tot']) if f['tot'] else 0.0
            c = 0.5 * min(1.0, (nflip - 1) / 3.0) + 0.5 * min(1.0, nflip / ntype / 0.5)
            s = 100 * (0.50 * m + 0.30 * a + 0.20 * c)
        f['conf'] = round(s)
        f['tier'] = 'High' if s >= 75 else 'Medium' if s >= 50 else 'Low'
    return flags


# ---------------------------------------------------------------- validation
def bench(rows, scl, seeds=6):
    """inject known mistags between confusable types; measure recall / FP."""
    trials = []
    for seed in range(seeds):
        rng = random.Random(seed)
        for p in rows:
            p['_tag'] = p['Pitch Type']
        bg = defaultdict(lambda: defaultdict(list))
        for p in rows:
            bg[(p['Pitcher'], p['_g'])][p['Pitch Type']].append(p)
        inj = {}
        for _, types in bg.items():
            elig = [t for t, ps in types.items() if len(ps) >= MIN_CLUSTER + 1]
            pairs = [(a, b) for i, a in enumerate(elig) for b in elig[i + 1:]
                     if frozenset((a, b)) in HARD_PAIRS]
            if not pairs:
                continue
            a, b = rng.choice(pairs)
            if rng.random() < .5:
                a, b = b, a
            v = rng.choice(types[a])
            v['_tag'] = b
            inj[id(v)] = a
        res, _ = score_game(rows, scl, tagkey='_tag')
        trials.append((res, inj))
    for p in rows:
        p.pop('_tag', None)
    return trials


def bench_report(trials):
    print(f"\n{'d_best<=':>9s}" + "".join(f"{f'margin>={m}':>16s}"
                                          for m in (0.5, 1.0, 1.5, 2.0)))
    for dbe in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 99):
        row = f"{dbe:9.1f}"
        for mg in (0.5, 1.0, 1.5, 2.0):
            rr, ff = [], []
            for res, inj in trials:
                tot = sum(1 for k in inj if k in res)
                clean = [k for k in res if k not in inj]
                ok = lambda v: v['margin'] >= mg and v['d_best'] <= dbe
                hit = sum(1 for k, a in inj.items()
                          if k in res and ok(res[k]) and res[k]['tgt'] == a)
                rr.append(hit / tot if tot else 0)
                ff.append(1000 * sum(1 for k in clean if ok(res[k])) / len(clean))
            row += f"{100*st.mean(rr):10.1f}%/{st.mean(ff):4.2f}"
        print(row)
    print("  cells = recall% / false flags per 1000 clean pitches")


# ---------------------------------------------------------------- plotting
#
# These used to be HB x IVB and Velocity x OTilt scatters. Measured across the
# Medium+ candidates, that pair of panels carried only 34% of the evidence the
# flags were built on: RTilt, Spin Rate and ArmAngle were drawn nowhere, and
# they are disproportionately the axes that drive flags. For the strongest
# finding (Lara 2026-04-05) the entire case is Spin Rate at +13.9z, roughly six
# times the next contributor, and the scatter could not show it at all. Worse,
# a projection in which a flagged pitch looks unremarkable is exactly the case
# where the eye overrides the arithmetic in the wrong direction -- that happened
# once during development on a Champlain pitch that looked interior in 2D while
# sitting 4.25z from its own cluster in the full space.
#
# The plot now draws the discriminant itself: per-axis pull toward the suggested
# type, in the same z units the decision is made in, all 7 axes, nothing hidden.


def axis_pulls(f, groups, scl):
    """[(pull_z, metric, value, own_centroid, tgt_centroid)] favouring the target.

    pull > 0 means the axis argues for the suggested type. For orphans there is
    no own-cluster centroid, so pull is reported as -|z to target|: bars near
    zero mean the pitch matches the target on that axis.
    """
    p = f['p']
    g = groups[(p['Pitcher'], p['_g'])]
    tgt = A.centroid([x['_mv'] for x in g[f['tgt']] if x is not p])
    own = None
    if f['kind'] != 'orphan':
        own = A.centroid([x['_mv'] for x in g[f['own']] if x is not p])
    out = []
    for m in A.METRICS:
        a, ct = p['_mv'].get(m), tgt.get(m)
        if a is None or ct is None or not scl.get(m):
            continue
        dt = (A.cdiff(a, ct) if m in A.CIRC else abs(a - ct)) / scl[m]
        if own is None:
            out.append((-dt, m, a, None, ct))
        else:
            co = own.get(m)
            if co is None:
                continue
            do = (A.cdiff(a, co) if m in A.CIRC else abs(a - co)) / scl[m]
            out.append((do - dt, m, a, co, ct))
    out.sort(reverse=True)
    return out


def _fmt(m, v):
    if v is None:
        return '--'
    return A.deg_clock(v) if m in A.CIRC else (f'{v:.0f}' if m == 'Spin Rate'
                                              else f'{v:.1f}')


def make_plot(pitcher, date, flagged, groups, scl, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(flagged)
    fig, axes = plt.subplots(1, n, figsize=(7.6 * n, 5.4), squeeze=False)
    for ax, f in zip(axes[0], sorted(flagged, key=lambda f: -f['conf'])):
        pulls = axis_pulls(f, groups, scl)
        if not pulls:
            continue
        ys = list(range(len(pulls)))[::-1]
        vals = [d for d, *_ in pulls]
        cols = ['#B00020' if d > 0 else '#4878A8' for d in vals]
        ax.barh(ys, vals, color=cols, height=.62, zorder=2)
        ax.axvline(0, color='#333', lw=1.0, zorder=3)
        labels = []
        for d, m, a, co, ct in pulls:
            if co is None:
                labels.append(f"{A.SHORT[m]}   {_fmt(m,a)}  ({f['tgt']} {_fmt(m,ct)})")
            else:
                labels.append(f"{A.SHORT[m]}   {_fmt(m,a)}  "
                              f"({f['own']} {_fmt(m,co)} / {f['tgt']} {_fmt(m,ct)})")
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=9, fontfamily='monospace')
        for y, d in zip(ys, vals):
            ax.annotate(f'{d:+.1f}', (d, y), fontsize=9, fontweight='bold',
                        va='center', ha='left' if d >= 0 else 'right',
                        xytext=(4 if d >= 0 else -4, 0),
                        textcoords='offset points', color='#222', zorder=4)
        span = max(abs(min(vals)), abs(max(vals))) or 1.0
        ax.set_xlim(-span * 1.45, span * 1.45)
        if f['kind'] == 'orphan':
            sub = (f"ORPHAN, only {f['nsame']} tagged {f['own']} this game   "
                   f"d_best {f['d_best']:.2f}   bars = -|z| to {f['tgt']}")
            ax.set_xlabel(f"closer to {f['tgt']}  ->", fontsize=10)
        else:
            sub = (f"margin {f['margin']:.2f}   d_own {f['d_own']:.2f}   "
                   f"d_best {f['d_best']:.2f}   agree {f['agree']}/{f['tot']}   "
                   f"same-game flips {f['nflip']}/{f['ntype']}")
            ax.set_xlabel(f"<-  favours {f['own']}        favours {f['tgt']}  ->",
                          fontsize=10)
        ax.set_title(f"{pitcher}   {date}\n{f['own']} → {f['tgt']}   "
                     f"conf {f['conf']} ({f['tier']})\n{sub}",
                     fontsize=11, linespacing=1.5)
        ax.grid(axis='x', color='#e6e6e6', zorder=0)
        ax.set_axisbelow(True)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    path = os.path.join(outdir, f"{pitcher.replace(', ', '_').replace(' ', '_')}_{date}.png")
    fig.savefig(path, dpi=125, bbox_inches='tight')
    plt.close(fig)
    return path


# ---------------------------------------------------------------- main
def main():
    subj = A.load()
    roc_all = [p for p in subj if p.get('PTeam') == TEAM]
    roc = prep(roc_all)
    scl = game_scales(roc)
    rs = release_stats(roc)

    print(f"ROC pitches {len(roc_all)} | pitchers {len({p['Pitcher'] for p in roc_all})} "
          f"| outings {len({(p['Pitcher'], p['Game Date']) for p in roc_all})}")
    print(f"Usable ({MIN_METRICS}+ of 7 metrics): {len(roc)}; "
          f"{len(roc_all) - len(roc)} skipped for sparse tracking.")
    print("\nWithin-GAME noise scales (z units):")
    for m in A.METRICS:
        print(f"  {m:12s} {scl[m]:8.3f}")

    if '--bench' in sys.argv:
        print("\n=== INJECTION VALIDATION (confusable-type mistags) ===")
        bench_report(bench(roc, scl))

    res, orph = score_game(roc, scl)
    print(f"\nCompared within-game: {len(res)} pitches in usable clusters, "
          f"{len(orph)} orphans (in-game type count < {MIN_CLUSTER}).")

    if '--sweep' in sys.argv:
        print("\n=== live candidate counts by threshold ===")
        print(f"{'d_best<=':>9s}" + "".join(f"{f'margin>={m}':>12s}" for m in (0.5, 1.0, 1.5, 2.0)))
        for dbe in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
            row = f"{dbe:9.1f}"
            for mg in (0.5, 1.0, 1.5, 2.0):
                row += f"{sum(1 for v in res.values() if v['margin'] >= mg and v['d_best'] <= dbe):12d}"
            print(row)

    flags = [v for v in res.values()
             if v['margin'] >= MARGIN_MIN and v['d_best'] <= D_BEST_MAX]
    review = [v for v in res.values()
              if v['margin'] >= MARGIN_MIN and D_BEST_MAX < v['d_best'] <= D_BEST_REVIEW]
    orphf = [o for o in orph if o['d_best'] <= ORPHAN_REVIEW]
    for f in flags + review + orphf:
        f['rel'] = rel_outlier(f['p'], rs)
    confidence(flags + review + orphf, roc)

    def show(items, header):
        print(f"\n=== {header}: {len(items)} ===")
        if items:
            print("  swaps:", Counter((f['own'], f['tgt']) for f in items).most_common())
        pages = defaultdict(list)
        for f in items:
            pages[(f['p']['Pitcher'], f['p']['Game Date'])].append(f)
        for (pit, dt), fs in sorted(pages.items()):
            fs.sort(key=lambda f: -f['conf'])
            print(f"\n  {pit}  {dt}   {len(fs)} flagged")
            for f in fs:
                rel = f['rel']
                rstr = (f"rel z{rel:.1f}" + ("  <-- LIKELY TRACKING GLITCH" if rel >= 3 else "")
                        ) if rel is not None else "rel n/a"
                if f['kind'] == 'orphan':
                    print(f"    {f['conf']:3d} {f['tier']:6s} {f['own']}->{f['tgt']} "
                          f"ORPHAN (only {f['nsame']} tagged {f['own']} this game) "
                          f"d_best{f['d_best']:.2f} {rstr}")
                else:
                    print(f"    {f['conf']:3d} {f['tier']:6s} {f['own']}->{f['tgt']} "
                          f"margin{f['margin']:.2f} d_own{f['d_own']:.2f} "
                          f"d_best{f['d_best']:.2f} agree{f['agree']}/{f['tot']} "
                          f"flip{f['nflip']}/{f['ntype']} {rstr}")
                print(f"        {f['why']}")

    show(flags, f"PER-PITCH FLAGS (margin>={MARGIN_MIN}, d_best<={D_BEST_MAX})")
    show(review, f"REVIEW BAND (d_best {D_BEST_MAX}-{D_BEST_REVIEW})")
    show(orphf, f"ORPHAN PITCHES (flag d_best<={ORPHAN_DBEST}, "
                f"review to {ORPHAN_REVIEW})")

    print("\n=== WHOLE-GAME DRIFT (cluster shifted vs season; calibration, not mistag) ===")
    for d, pit, dt, pt, n in game_drift(roc, scl)[:10]:
        print(f"  {pit:22s} {dt} {pt:3s} n={n:3d}  shift {d:.2f} z")

    allf = flags + review + orphf
    if '--csv' in sys.argv:
        import csv
        out = os.path.expanduser('~/Downloads/roc_tag_audit_2026.csv')
        rows = [f for f in allf if f['tier'] in CSV_TIERS]
        with open(out, 'w', newline='') as fh:
            w = csv.writer(fh)
            # identity, then the raw pitch metrics, then the scoring block.
            w.writerow(['Pitcher', 'Game Date', 'Kind', 'Tagged', 'Suggested',
                        'Tier', 'Velocity', 'SpinRate', 'RTilt', 'OTilt', 'IVB',
                        'HB', 'RelPosZ', 'RelPosX', 'ArmAngle', 'ReleaseOutlierZ',
                        'Confidence', 'Margin', 'DistOwn', 'DistBest',
                        'MetricsAgree', 'SameGameFlips', 'TypePitchesInGame',
                        'Why', 'PitchID'])
            for f in sorted(rows, key=lambda f: (f['p']['Pitcher'],
                                                 f['p']['Game Date'], -f['conf'])):
                p = f['p']
                w.writerow([p['Pitcher'], p['Game Date'], f['kind'], f['own'],
                            f['tgt'], f['tier'],
                            p.get('Velocity'), p.get('Spin Rate'), p.get('RTilt'),
                            p.get('OTilt'), p.get('IndVertBrk'), p.get('HorzBrk'),
                            p.get('RelPosZ'), p.get('RelPosX'), p.get('ArmAngle'),
                            '' if f['rel'] is None else round(f['rel'], 2),
                            f['conf'],
                            '' if f['margin'] is None else round(f['margin'], 2),
                            '' if f['d_own'] is None else round(f['d_own'], 2),
                            round(f['d_best'], 2),
                            '' if f['tot'] is None else f"{f['agree']}/{f['tot']}",
                            f.get('nflip'), f.get('ntype'), f['why'],
                            p.get('PitchID')])
        print(f"\nWrote {out}  ({len(rows)} rows, tiers {'/'.join(sorted(CSV_TIERS))}; "
              f"{len(allf) - len(rows)} Low-tier candidates omitted)")

    if '--plots' in sys.argv:
        os.makedirs(PLOTDIR, exist_ok=True)
        groups = defaultdict(lambda: defaultdict(list))
        for p in roc:
            groups[(p['Pitcher'], p['_g'])][p['Pitch Type']].append(p)
        pages = defaultdict(list)
        for f in allf:
            pages[(f['p']['Pitcher'], f['p']['Game Date'])].append(f)
        made = [make_plot(pit, dt, fs, groups, scl, PLOTDIR)
                for (pit, dt), fs in sorted(pages.items())]
        print(f"\nWrote {len([m for m in made if m])} plots to {PLOTDIR}")


if __name__ == '__main__':
    main()
