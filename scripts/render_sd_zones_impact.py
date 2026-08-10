#!/usr/bin/env python3
"""Zone x count SD+/CT+ impact matrices for the ROC article hitters.

Hitter-side port of scripts/render_loc_zones_impact.py (the "What His
Locations Add and Cost" pages): every cell of the 5-zone x 3-count-state
matrix shows its IMPACT in metric points — his share of decisions (swings
for CT+) there times his grade, minus the league share times the league
grade — so the 15 cells sum to the gap between his overall grade and the
league's. Panels: all pitches + FB / BRK / OFF pitch categories (SD+'s own
category dimension). Pages: season, vs RHP, vs LHP.

Per-pitch grades (both scale the league overall grade to exactly 100):
  SD+: atom = 100 * dv / lg_mean_dv, dv = RV(chosen) - RV(opposite) from
       the shipped sdPlusWeights table (metadata_rs.json).
  CT+: atom = 100 * lev * I[contact] / lg_mean(lev * (1 - p_whiff)) from
       the shipped ctPlusWeights table — leverage-weighted contact vs the
       league expectation, the numerator/denominator atoms of shipped CT+.
Panel grades are the unregressed mean of those atoms; the leaderboard
SD+/CT+ additionally mix-neutralizes and regresses, so the header carries
the printed leaderboard value separately.

Constants (swept on MLB hitter units capped to ROC per-category volumes,
scripts/sdzone_impact_sweeps.py, 2026-08-10):
  SD+: shrinkage k=0 — cross-half predictive r declines monotonically in k
       (0.1928 at k=0, flat through k=5, 0.0817 at k=640); k=0 is the
       parameter-space boundary, so unshrunken grades ship. Callout
       discount lam=0.25 (interior: expected realized spread 3.546 /
       3.600 / 3.572 at lam 0 / 0.25 / 0.5).
  CT+: shrinkage k=10 (interior: r 0.1540 / 0.1595 / 0.1476 at k 0 / 10 /
       320, flat 5-20). Callout lam=0.75 (interior: 2.312 / 2.402 / 2.287
       at 0.5 / 0.75 / 1.0).
  The callout objective is EXPECTED realized spread (no-pick pages score
  0) — the raw conditional spread is gameable by coverage collapse.
  Color spans and panel gates are display conventions, stated below.

Usage: python3 scripts/render_sd_zones_impact.py
Outputs ~/Downloads/ArticleVisuals/<Last>/{SD,CT}Zones_<LastFirst>_impact*.png
"""
import json
import math
import os
import pickle
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline_sdplus import (  # noqa: E402
    is_eligible, classify_zone, classify_decision, get_count, cat_of,
)
from pipeline_contact import is_ct_eligible, classify_contact_outcome  # noqa: E402

PICKLE = os.path.join(ROOT, 'data', 'all_pitches_rs_cache.pkl')
METADATA = os.path.join(ROOT, 'data', 'metadata_rs.json')
HITTER_LB = os.path.join(ROOT, 'data', 'hitter_leaderboard_rs.json')
OUT_ROOT = os.path.expanduser('~/Downloads/ArticleVisuals')

for f in os.listdir(os.path.join(ROOT, 'assets', 'fonts')):
    if f.endswith('.ttf'):
        fm.fontManager.addfont(os.path.join(ROOT, 'assets', 'fonts', f))
plt.rcParams['font.family'] = 'IBM Plex Sans'
TITLE_FONT = {'fontfamily': 'Bitter', 'fontweight': 700}

PAPER = (240 / 255, 232 / 255, 216 / 255)
CREAM = '#e8dfcb'
CREAM_RGB = (232 / 255, 223 / 255, 203 / 255)
INK = (58 / 255, 48 / 255, 38 / 255)
BRICK = (176 / 255, 64 / 255, 47 / 255)
SLATE = (66 / 255, 100 / 255, 138 / 255)

HITTERS = ['Ortiz, Abimelec', 'Morales, Yohandy', 'King, Seaver',
           'Glasser, Phillip', 'Pinckney, Andrew']

ZONES = ['heart', 'shadow_in', 'shadow_out', 'chase', 'waste']
ZONE_NAMES = {'heart': 'Heart', 'shadow_in': 'Shadow-In',
              'shadow_out': 'Shadow-Out', 'chase': 'Chase', 'waste': 'Waste'}
STATES = ['ahead', 'even', 'behind']
STATE_NAMES = {'ahead': 'AHEAD', 'even': 'EVEN', 'behind': 'BEHIND'}
STATE_PROSE = {'ahead': 'while ahead', 'even': 'in even counts',
               'behind': 'while behind'}
CATS = ['FB', 'BRK', 'OFF']
CAT_NAMES = {'FB': 'Fastballs', 'BRK': 'Breaking', 'OFF': 'Offspeed'}

# Per-metric config: (metric label, swept k, swept lam, color span in pts,
# panel gate, unit noun, page title verb phrase).
METRICS = {
    'SD': {'label': 'SD+', 'k': 0, 'lam': 0.25, 'span': 4.0,
           'gate': 100, 'noun': 'decisions',
           'title': 'What His Swing Decisions Add and Cost',
           'lb_key': 'sdPlus'},
    'CT': {'label': 'CT+', 'k': 10, 'lam': 0.75, 'span': 4.0,
           'gate': 60, 'noun': 'swings',
           'title': 'What His Contact Adds and Costs',
           'lb_key': 'ctPlus'},
}
FADE_N = 10


def sf(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def count_state_hitter(p):
    c = get_count(p)
    if c is None:
        return None
    b, s = c
    return 'ahead' if b > s else ('behind' if s > b else 'even')


def heat_color(t):
    t = min(1.0, max(0.0, t))
    target = BRICK if t >= 0.5 else SLATE
    p = (abs(t - 0.5) / 0.5) ** 0.9
    return tuple(b + (c - b) * p for b, c in zip(PAPER, target))


def draw_zone_legend(ax):
    ZONE_X, HEART_X, SHADOW_X, CHASE_X = 0.83, 6.7 / 12, 13.3 / 12, 20.0 / 12
    DXl, Z0, Z1 = 1.95, -0.75, 1.75
    ax.set_xlim(-DXl, DXl)
    ax.set_ylim(Z0, Z1)
    ax.set_aspect('equal')
    ax.axis('off')
    shades = {'waste': (0.78, 0.73, 0.62), 'chase': (0.85, 0.80, 0.69),
              'shadow_out': (0.90, 0.86, 0.76), 'shadow_in': (0.94, 0.90, 0.81),
              'heart': (0.97, 0.94, 0.86)}
    rects = [('waste', (-DXl, Z0, 2 * DXl, Z1 - Z0)),
             ('chase', (-CHASE_X, -0.5, 2 * CHASE_X, 2.0)),
             ('shadow_out', (-SHADOW_X, -1 / 6, 2 * SHADOW_X, 1 + 2 / 6)),
             ('shadow_in', (-ZONE_X, 0, 2 * ZONE_X, 1.0)),
             ('heart', (-HEART_X, 1 / 6, 2 * HEART_X, 4 / 6))]
    for z, (x, y, w, h) in rects:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=shades[z],
                               edgecolor=(*INK, 0.4), linewidth=0.8))
    ax.add_patch(Rectangle((-ZONE_X, 0), 2 * ZONE_X, 1, fill=False,
                           edgecolor=(*INK, 0.85), linewidth=1.8))
    for txt, x, y in [('Heart', 0, 0.5), ('Shadow-In', 0, 0.08),
                      ('Shadow-Out', 0, 1.075), ('Chase', 0, 1.33),
                      ('Waste', 0, 1.615)]:
        ax.text(x, y, txt, ha='center', va='center', fontsize=8,
                color=INK, fontweight=600)
    ax.set_title('the five zones (catcher view)', fontsize=9,
                 color=(*INK, 0.75), pad=5)


def cell_impacts(acc, n_total, lg, k):
    out = {}
    for z in ZONES:
        for st in STATES:
            cell = (z, st)
            ls, lgg, _sd = lg['joint'].get(cell, (0.0, 0.0, 0.0))
            s_sum, n = acc.get(cell, [0.0, 0])
            share = n / n_total if n_total else 0.0
            g_raw = (s_sum / n) if n else None
            g_shr = ((n * (s_sum / n) + k * lgg) / (n + k)
                     if n else lgg)
            impact = share * g_shr - ls * lgg
            mix = lgg * (share - ls)
            execu = share * (g_shr - lgg)
            out[cell] = {'n': n, 'share': share, 'g_raw': g_raw,
                         'g_shr': g_shr, 'lg_share': ls, 'lg_grade': lgg,
                         'impact': impact, 'mix': mix, 'exec': execu}
    return out


def driver_phrase(c):
    m, e = abs(c['mix']), abs(c['exec'])
    if m > 2 * e:
        return 'mostly mix'
    if e > 2 * m:
        return 'mostly execution'
    return 'mix and execution'


def draw_panel(ax, title, label, cells, state_n, lg, n_total, lg_tot,
               span, noun):
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.95, 5.45)
    ax.invert_yaxis()
    ax.axis('off')
    for j, st in enumerate(STATES):
        hs = 100.0 * state_n.get(st, 0) / n_total if n_total else 0.0
        ls = 100.0 * lg['state'].get(st, 0.0)
        ax.text(j + 0.5, -0.52, STATE_NAMES[st], ha='center', fontsize=8,
                color=INK, fontweight=700)
        ax.text(j + 0.5, -0.18, f'{hs:.1f}% (lg {ls:.1f}%)',
                ha='center', fontsize=6.2, color=(*INK, 0.85))
    for i, z in enumerate(ZONES):
        ax.text(-0.06, i + 0.5, ZONE_NAMES[z], ha='right', va='center',
                fontsize=9, color=INK, fontweight=600)
        for j, st in enumerate(STATES):
            c = cells[(z, st)]
            t = 0.5 + c['impact'] / (2 * span)
            face = heat_color(t)
            dim = 0 < c['n'] < FADE_N
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face,
                                   alpha=0.45 if dim else 1.0,
                                   edgecolor=(*INK, 0.25), linewidth=0.6))
            txt = CREAM_RGB if (abs(t - 0.5) > 0.28 and not dim) else INK
            star = '*' if dim else ''
            imp_txt = f'{c["impact"]:.1f}'.replace('-0.0', '0.0')
            ax.text(j + 0.5, i + 0.24, f'{imp_txt} pts{star}',
                    ha='center', va='center', fontsize=8.6, fontweight=700,
                    color=txt)
            sn = state_n.get(st, 0)
            cshare = 100.0 * c['n'] / sn if sn else 0.0
            lg_state = lg['state'].get(st, 0.0)
            lgc = (100.0 * c['lg_share'] / lg_state) if lg_state else 0.0
            sub_c = (*txt, 0.92)
            ax.text(j + 0.5, i + 0.53, f'{cshare:.1f}% (lg {lgc:.1f}%)',
                    ha='center', va='center', fontsize=6.2, color=sub_c)
            g_txt = f'{c["g_raw"]:.0f}' if c['g_raw'] is not None else '–'
            ax.text(j + 0.5, i + 0.78, f'{label} {g_txt} (lg {c["lg_grade"]:.0f})',
                    ha='center', va='center', fontsize=6.2, color=sub_c)
    total = sum(c['impact'] for c in cells.values())
    ax.text(0, 5.32, f'cells sum to {total:.1f} pts vs a league-average map '
            f'(league overall grade {lg_tot:.0f}) · * = under {FADE_N} {noun}',
            fontsize=6.4, color=(*INK, 0.8))
    ax.set_title(title, fontsize=11, color=INK, pad=10, loc='left', **TITLE_FONT)


def fit_font(text, base=11.0, max_chars=116):
    n = len(text)
    return base if n <= max_chars else max(7.2, base * max_chars / n)


def league_tables(pool, atom_fn):
    """(hand, cat) + (hand, 'ALL') league tables: joint share/grade/sd per
    cell, state shares, overall grade."""
    acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))
    state_n = defaultdict(lambda: defaultdict(int))
    n_tot = defaultdict(int)
    for p in pool:
        z, st = classify_zone(p), count_state_hitter(p)
        if st is None:
            continue
        a = atom_fn(p)
        hands = ['ALL'] + ([p.get('Throws')] if p.get('Throws') in ('L', 'R') else [])
        for h in hands:
            for g in ('ALL', cat_of(p)):
                cell = acc[(h, g)][(z, st)]
                cell[0] += a
                cell[1] += a * a
                cell[2] += 1
                state_n[(h, g)][st] += 1
                n_tot[(h, g)] += 1
    out = {}
    for key, cells in acc.items():
        tot = n_tot[key]
        joint, sd = {}, {}
        for cell in [(z, s) for z in ZONES for s in STATES]:
            s, ss, n = cells.get(cell, [0.0, 0.0, 0])
            if n:
                mean = s / n
                joint[cell] = (n / tot, mean,
                               math.sqrt(max(ss / n - mean ** 2, 0.0)))
            else:
                joint[cell] = (0.0, 0.0, 0.0)
        out[key] = {'joint': joint,
                    'state': {st: state_n[key][st] / tot for st in STATES},
                    'total': sum(sh * g for sh, g, _ in joint.values())}
    return out


def main():
    print('Loading pitch cache + tables ...')
    with open(PICKLE, 'rb') as f:
        allp = pickle.load(f)
    meta = json.load(open(METADATA))
    sdw, ctw = meta['sdPlusWeights'], meta['ctPlusWeights']
    lb = json.load(open(HITTER_LB))
    lb_row = {r['hitter']: r for r in lb
              if r.get('hitter') in HITTERS and r.get('team') == 'ROC'}

    mlb = [p for p in allp if p.get('_source', 'MLB') == 'MLB' and is_eligible(p)]

    def dv(p):
        z, c, cat = classify_zone(p), get_count(p), cat_of(p)
        s = sdw[f'{z}|{cat}|{c[0]}-{c[1]}|swing']['rv']
        t = sdw[f'{z}|{cat}|{c[0]}-{c[1]}|take']['rv']
        return (s - t) if classify_decision(p) == 'swing' else (t - s)

    lg_mean_dv = sum(dv(p) for p in mlb) / len(mlb)

    def sd_atom(p):
        return 100.0 * dv(p) / lg_mean_dv

    mlb_sw = [p for p in mlb if is_ct_eligible(p)]

    def lev_pw(p):
        cell = ctw[f'{classify_zone(p)}|{get_count(p)[0]}-{get_count(p)[1]}']
        return cell['rv_contact'] - cell['rv_whiff'], cell['p_whiff']

    D = sum(lv * (1 - pw) for lv, pw in (lev_pw(p) for p in mlb_sw)) / len(mlb_sw)

    def ct_atom(p):
        lv, _pw = lev_pw(p)
        made = 1.0 if classify_contact_outcome(p) == 'contact' else 0.0
        return 100.0 * lv * made / D

    print('Building league tables ...')
    lg_tabs = {'SD': league_tables(mlb, sd_atom),
               'CT': league_tables(mlb_sw, ct_atom)}
    atom_fns = {'SD': sd_atom, 'CT': ct_atom}
    elig_fns = {'SD': is_eligible, 'CT': is_ct_eligible}

    by_hitter = defaultdict(list)
    for p in allp:
        if p.get('Batter') in HITTERS and p.get('BTeam') == 'ROC':
            by_hitter[p['Batter']].append(p)

    PAGES = [('ALL', '', ''), ('R', '_vsRHP', ' vs RHP'), ('L', '_vsLHP', ' vs LHP')]

    for name in HITTERS:
      last, first = [s.strip() for s in name.split(',')]
      outdir = os.path.join(OUT_ROOT, last)
      os.makedirs(outdir, exist_ok=True)
      row = lb_row.get(name, {})

      for mkey, cfg in METRICS.items():
        atom_fn, elig = atom_fns[mkey], elig_fns[mkey]
        label, noun = cfg['label'], cfg['noun']
        all_elig = [p for p in by_hitter[name] if elig(p)]

        for hand, suffix, hand_label in PAGES:
            pitches = (all_elig if hand == 'ALL'
                       else [p for p in all_elig if p.get('Throws') == hand])
            if not pitches:
                continue
            lgh = lg_tabs[mkey]

            acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
            state_n = defaultdict(lambda: defaultdict(int))
            for p in pitches:
                z, st = classify_zone(p), count_state_hitter(p)
                if st is None:
                    continue
                a = atom_fn(p)
                for key in ('ALL', cat_of(p)):
                    acc[key][(z, st)][0] += a
                    acc[key][(z, st)][1] += 1
                    state_n[key][st] += 1

            panels = [('ALL', acc['ALL'])]
            for cat in CATS:
                zs = acc.get(cat)
                if zs and sum(n for _, n in zs.values()) >= cfg['gate']:
                    panels.append((cat, zs))

            # Callouts from category cells, impact minus lam SE (swept).
            ranked = []
            cat_sums = []
            for key, zs in panels[1:]:
                n_total = sum(n for _, n in zs.values())
                L = lgh[(hand, key)]
                cells = cell_impacts(zs, n_total, L, cfg['k'])
                cat_sums.append((sum(c['impact'] for c in cells.values()),
                                 key, n_total))
                for (z, st), c in cells.items():
                    sd = L['joint'].get((z, st), (0, 0, 0))[2]
                    se_g = sd / math.sqrt(c['n']) if c['n'] else 0.0
                    se_s = math.sqrt(max(c['share'] * (1 - c['share']), 1e-9)
                                     / n_total)
                    se = math.sqrt((c['share'] * se_g) ** 2
                                   + (c['lg_grade'] * se_s) ** 2)
                    d = abs(c['impact']) - cfg['lam'] * se
                    score = (math.copysign(max(0.0, d), c['impact'])
                             if d > 0 else 0.0)
                    ranked.append((score, c['impact'], key, z, st, c))
            ranked.sort(key=lambda r: r[0])
            costs = [r for r in ranked[:2] if r[0] < 0]
            helps = [r for r in ranked[-2:][::-1] if r[0] > 0]

            def sentence(r):
                _, imp, key, z, st, c = r
                verb = 'adds' if imp > 0 else 'costs'
                return (f'{CAT_NAMES[key]} in {ZONE_NAMES[z]} '
                        f'{STATE_PROSE[st]}: {verb} {abs(imp):.1f} pts '
                        f'({driver_phrase(c)})')

            mm_cols = 3
            mm_rows = math.ceil(len(panels) / mm_cols)
            fig = plt.figure(figsize=(17.0, 3.6 + 5.2 * mm_rows), dpi=200)
            fig.patch.set_facecolor(CREAM)
            gs = GridSpec(mm_rows + 1, mm_cols, figure=fig,
                          height_ratios=[0.78] + [1.3] * mm_rows,
                          hspace=0.30, wspace=0.35,
                          left=0.055, right=0.985, top=0.955, bottom=0.04)
            hd = fig.add_subplot(gs[0, :2])
            hd.axis('off')
            hd.text(0, 0.97, f'{first} {last}{hand_label}: {cfg["title"]}',
                    fontsize=23, color=INK, va='top', **TITLE_FONT)
            shr = (f'shrinkage k={cfg["k"]}' if cfg['k']
                   else 'grades unshrunken (k=0 won the sweep)')
            sub = (f'2026 season (ROC){hand_label} · zone × count impact '
                   f'matrix · {shr}, callouts impact minus {cfg["lam"]} SE '
                   f'(both split-half tuned)')
            lb_val = row.get(cfg['lb_key'])
            if hand == 'ALL' and lb_val is not None:
                pct = row.get(cfg['lb_key'] + '_pctl')
                sub += (f' · leaderboard {label} {lb_val:.0f}'
                        + (f' ({pct} pctl)' if pct is not None else ''))
            if hand != 'ALL':
                sub += f' · league ={hand_label} only'
            hd.text(0, 0.78, sub, fontsize=fit_font(sub, base=11.5),
                    color=(*INK, 0.75), va='top')
            counts_word = ('AHEAD/BEHIND are the hitter’s counts. ' )
            hd.text(0, 0.68,
                    f'Each cell: its IMPACT in {label} points, his share of '
                    f'{noun} there times his grade minus the league’s share '
                    'times the league grade; red adds value, blue costs it,\n'
                    'and the 15 cells sum to the gap between his overall '
                    f'grade and league. {counts_word}Mix = the pitch diet he '
                    'sees and the counts he reaches; execution = his grade '
                    'on\nthose pitches. Share percents are within-column '
                    f'(league in parentheses); the {label} line shows the '
                    'unshrunken grade of those pitches. Panel grade = mean '
                    'atom;\nthe leaderboard number additionally '
                    'mix-neutralizes and regresses.',
                    fontsize=9.8, color=INK, va='top', linespacing=1.6)
            if helps:
                helps_line = ('Helps him most:  '
                              + '   |   '.join(sentence(r) for r in helps))
                hd.text(0, 0.20, helps_line, fontsize=fit_font(helps_line),
                        color=(140 / 255, 52 / 255, 38 / 255), va='top',
                        fontweight=700)
            if costs:
                costs_line = ('Costs him most:  '
                              + '   |   '.join(sentence(r) for r in costs))
                hd.text(0, 0.03, costs_line, fontsize=fit_font(costs_line),
                        color=(52 / 255, 80 / 255, 110 / 255), va='top',
                        fontweight=700)
            if cat_sums:
                best_map = max(cat_sums, key=lambda t: t[0])
                worst_map = min(cat_sums, key=lambda t: t[0])
                maps_line = (f'Whole category maps: best vs '
                             f'{CAT_NAMES[best_map[1]]} {best_map[0]:.1f} pts '
                             f'({best_map[2]} {noun}) · worst vs '
                             f'{CAT_NAMES[worst_map[1]]} {worst_map[0]:.1f} pts '
                             f'({worst_map[2]} {noun})')
                hd.text(0, -0.13, maps_line,
                        fontsize=fit_font(maps_line, base=10.5),
                        color=INK, va='top', fontweight=600)
            lg_ax = fig.add_subplot(gs[0, 2])
            draw_zone_legend(lg_ax)

            for i, (key, zs) in enumerate(panels):
                n_total = sum(n for _, n in zs.values())
                overall = sum(s for s, _ in zs.values()) / n_total
                L = lgh[(hand, key)]
                cells = cell_impacts(zs, n_total, L, cfg['k'])
                pname = ('All pitches' if key == 'ALL'
                         else f'vs {CAT_NAMES[key]}')
                r, c = divmod(i, mm_cols)
                ax = fig.add_subplot(gs[1 + r, c])
                draw_panel(ax, f'{pname} · {label} {overall:.0f} · '
                           f'{n_total} {noun}',
                           label, cells, state_n[key], L, n_total,
                           L['total'], cfg['span'], noun)

            out = os.path.join(
                outdir, f'{mkey}Zones_{last}{first}_impact{suffix}.png')
            fig.savefig(out, facecolor=CREAM, bbox_inches='tight')
            plt.close(fig)
            imps = [c['impact']
                    for _k2, zs in panels
                    for c in cell_impacts(
                        zs, sum(n for _, n in zs.values()),
                        lgh[(hand, _k2)], cfg['k']).values()]
            print(f'{name} {mkey}{hand_label or " season"} -> {out}  '
                  f'(cell impacts p5 {min_p(imps, 5):+.1f} / '
                  f'p95 {min_p(imps, 95):+.1f})')


def min_p(vals, q):
    vs = sorted(vals)
    i = max(0, min(len(vs) - 1, int(round(q / 100 * (len(vs) - 1)))))
    return vs[i]


if __name__ == '__main__':
    main()
