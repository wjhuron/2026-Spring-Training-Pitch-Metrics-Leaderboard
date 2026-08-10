#!/usr/bin/env python3
"""Zone x count SD+/CT+ impact matrices, CARD-COHERENT version.

v2 (2026-08-10): cells now decompose the SHIPPED metric's own aggregate,
so the season page's 15 cells sum to (leaderboard value - 100) and the
headline number IS the hitter-card number. The v1 decomposition counted
pitch diet (SD+) / swing selection (CT+), which the leaderboard metrics
deliberately neutralize, so page and card disagreed (Pinckney CT+ 98 vs
card 91 — the gap was his easy-contact swing mix).

  SD+ (diet-neutral, mirroring compute_hitter_sd's mix-neutral form):
      impact(z,st) = kappa * w_z_lg * [ s_h(st|z)*g'_h - s_lg(st|z)*g_lg ]
      w_z_lg = league zone share (diet neutralized by construction),
      s(st|z) = within-zone count-state share, g = mean decision atom
      (100 * dv / lg_mean_dv from the shipped sdPlusWeights table), g'_h
      shrunk toward g_lg with k=5. Count mix + execution count; which
      zones pitchers put him in does not.

  CT+ (execution-only, mirroring raw_ct = sum(lev*I)/sum(lev*E)):
      impact(z,st) = kappa * 100 * [n/(n+5)] * sum_cell[lev*(I-E)] / T
      E = league contact expectation for the pitch's (zone x count) cell,
      T = sum(lev*E) over ALL his swings. A cell where he converts at the
      league rate scores 0 no matter how often he swings there — swing
      selection is SD+'s job. Category panels partition the season total.

  kappa: per-hitter calibration absorbing the shipped regression +
  qualified-pool re-anchor — exact ((printed-100)/page_gap) when
  well-conditioned, else the theoretical f * n/(n+n0).

Constants k=5 / lam=0.25 (both metrics) re-swept for these formulas —
interior optima bracketed (scripts/sdzone_impact_sweeps.py v2; the CT lam
curve is flat 0-0.75). Color span is a display convention.

Renders only the requested article pages (season, no platoon splits):
King SD+; Morales, Glasser, Pinckney CT+.

Usage: python3 scripts/render_sd_zones_impact.py
Outputs ~/Downloads/ArticleVisuals/<Last>/{SD,CT}Zones_<LastFirst>_impact.png
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
    HITTER_PRIOR_N as SD_N0,
)
from pipeline_contact import (  # noqa: E402
    is_ct_eligible, classify_contact_outcome, HITTER_PRIOR_N as CT_N0,
)

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

# (hitter, metric) pages requested — season only, no platoon splits.
REQUESTS = [('King, Seaver', 'SD'),
            ('Morales, Yohandy', 'CT'),
            ('Glasser, Phillip', 'CT'),
            ('Pinckney, Andrew', 'CT')]

ZONES = ['heart', 'shadow_in', 'shadow_out', 'chase', 'waste']
ZONE_NAMES = {'heart': 'Heart', 'shadow_in': 'Shadow-In',
              'shadow_out': 'Shadow-Out', 'chase': 'Chase', 'waste': 'Waste'}
STATES = ['ahead', 'even', 'behind']
STATE_NAMES = {'ahead': 'AHEAD', 'even': 'EVEN', 'behind': 'BEHIND'}
STATE_PROSE = {'ahead': 'while ahead', 'even': 'in even counts',
               'behind': 'while behind'}
CATS = ['FB', 'BRK', 'OFF']
CAT_NAMES = {'FB': 'Fastballs', 'BRK': 'Breaking', 'OFF': 'Offspeed'}
CELLS = [(z, s) for z in ZONES for s in STATES]

METRICS = {
    'SD': {'label': 'SD+', 'k': 5, 'lam': 0.25, 'span': 3.0,
           'gate': 100, 'noun': 'decisions', 'n0': SD_N0,
           'title': 'What His Swing Decisions Add and Cost',
           'lb_key': 'sdPlus'},
    'CT': {'label': 'CT+', 'k': 5, 'lam': 0.25, 'span': 1.5,
           'gate': 60, 'noun': 'swings', 'n0': CT_N0,
           'title': 'What His Contact Adds and Costs',
           'lb_key': 'ctPlus'},
}
FADE_N = 10


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


def fit_font(text, base=11.0, max_chars=116):
    n = len(text)
    return base if n <= max_chars else max(7.2, base * max_chars / n)


# ── SD cell machinery ───────────────────────────────────────────────────

def sd_league(atoms):
    acc = defaultdict(lambda: [0.0, 0.0, 0])
    for v, z, st in atoms:
        c = acc[(z, st)]
        c[0] += v
        c[1] += v * v
        c[2] += 1
    tot = len(atoms)
    joint = {}
    for cell in CELLS:
        s, ss, n = acc.get(cell, [0.0, 0.0, 0])
        if n:
            m = s / n
            joint[cell] = (n / tot, m, math.sqrt(max(ss / n - m * m, 0.0)))
        else:
            joint[cell] = (0.0, 0.0, 0.0)
    wz = {z: sum(joint[(z, st)][0] for st in STATES) for z in ZONES}
    slg = {(z, st): (joint[(z, st)][0] / wz[z] if wz[z] else 0.0)
           for z in ZONES for st in STATES}
    state = {st: sum(joint[(z, st)][0] for z in ZONES) for st in STATES}
    return {'joint': joint, 'wz': wz, 'slg': slg, 'state': state}


def sd_cells(atoms, lg, k, kappa):
    """{cell: dict} with impact (calibrated pts), mix/exec, display stats."""
    acc = defaultdict(lambda: [0.0, 0])
    for v, z, st in atoms:
        acc[(z, st)][0] += v
        acc[(z, st)][1] += 1
    nz = {z: sum(acc.get((z, st), [0, 0])[1] for st in STATES) for z in ZONES}
    out = {}
    for z in ZONES:
        for st in STATES:
            _ls, lgg, sd = lg['joint'][(z, st)]
            s_sum, n = acc.get((z, st), [0.0, 0])
            s_lg = lg['slg'][(z, st)]
            if nz[z] == 0:
                out[(z, st)] = {'n': 0, 'nz': 0, 'share': None, 'g_raw': None,
                                'lg_share': s_lg, 'lg_grade': lgg, 'sd': sd,
                                'impact': 0.0, 'mix': 0.0, 'exec': 0.0,
                                'wz': lg['wz'][z]}
                continue
            s_h = n / nz[z]
            g_raw = (s_sum / n) if n else None
            g_shr = ((n * (s_sum / n) + k * lgg) / (n + k)) if n else lgg
            w = lg['wz'][z]
            impact = kappa * w * (s_h * g_shr - s_lg * lgg)
            mix = kappa * w * lgg * (s_h - s_lg)
            execu = kappa * w * s_h * (g_shr - lgg)
            out[(z, st)] = {'n': n, 'nz': nz[z], 'share': s_h, 'g_raw': g_raw,
                            'lg_share': s_lg, 'lg_grade': lgg, 'sd': sd,
                            'impact': impact, 'mix': mix, 'exec': execu,
                            'wz': w}
    return out


def sd_se(c, kappa):
    if not c['n'] or not c['nz']:
        return 0.0
    se_g = c['sd'] / math.sqrt(c['n'])
    se_s = math.sqrt(max(c['share'] * (1 - c['share']), 1e-9) / c['nz'])
    return kappa * c['wz'] * math.sqrt((c['share'] * se_g) ** 2
                                       + (c['lg_grade'] * se_s) ** 2)


# ── CT cell machinery ───────────────────────────────────────────────────

def ct_cells(tuples, T, k, kappa):
    """tuples: (e, var, made, lev_e, z, st). Impacts in calibrated CT+ pts;
    cell ratio = sum(lev*I)/sum(lev*E) for the grade line."""
    acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])  # e, var, levI, levE, n
    for e, var, levI, levE, z, st in tuples:
        c = acc[(z, st)]
        c[0] += e
        c[1] += var
        c[2] += levI
        c[3] += levE
        c[4] += 1
    out = {}
    for cell in CELLS:
        s_e, s_v, s_i, s_x, n = acc.get(cell, [0.0, 0.0, 0.0, 0.0, 0])
        shrink = n / (n + k) if n else 0.0
        out[cell] = {'n': n,
                     'impact': kappa * 100.0 * shrink * s_e / T,
                     'se': kappa * 100.0 * shrink * math.sqrt(s_v) / T,
                     'ratio': (100.0 * s_i / s_x) if s_x > 0 else None,
                     'levE': s_x}
    return out


# ── Panels ──────────────────────────────────────────────────────────────

def draw_panel_sd(ax, title, cells, state_n, lg, n_total, sum_note, span,
                  noun):
    _draw_grid(ax, title, cells, state_n, lg_state=lg['state'],
               n_total=n_total, span=span,
               row_sub=lambda z: f'lg {100 * lg["wz"][z]:.0f}%',
               line2=lambda c: (f'{100 * c["share"]:.1f}% of zone '
                                f'(lg {100 * c["lg_share"]:.1f}%)'
                                if c['share'] is not None else '–'),
               line3=lambda c: (f'grade {c["g_raw"]:.0f} '
                                f'(lg {c["lg_grade"]:.0f})'
                                if c['g_raw'] is not None
                                else f'grade – (lg {c["lg_grade"]:.0f})'),
               sum_note=sum_note, noun=noun)


def draw_panel_ct(ax, title, cells, state_n, lg_state, n_total, sum_note,
                  span, noun):
    _draw_grid(ax, title, cells, state_n, lg_state=lg_state,
               n_total=n_total, span=span,
               row_sub=lambda z: None,
               line2=lambda c: None,
               line3=lambda c: (f'cell CT+ {c["ratio"]:.0f}'
                                if c['ratio'] is not None else 'cell CT+ –'),
               sum_note=sum_note, noun=noun)


def _draw_grid(ax, title, cells, state_n, lg_state, n_total, span,
               row_sub, line2, line3, sum_note, noun):
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.95, 5.45)
    ax.invert_yaxis()
    ax.axis('off')
    for j, st in enumerate(STATES):
        hs = 100.0 * state_n.get(st, 0) / n_total if n_total else 0.0
        ls = 100.0 * lg_state.get(st, 0.0)
        ax.text(j + 0.5, -0.52, STATE_NAMES[st], ha='center', fontsize=8,
                color=INK, fontweight=700)
        ax.text(j + 0.5, -0.18, f'{hs:.1f}% (lg {ls:.1f}%)',
                ha='center', fontsize=6.2, color=(*INK, 0.85))
    for i, z in enumerate(ZONES):
        ax.text(-0.06, i + 0.44, ZONE_NAMES[z], ha='right', va='center',
                fontsize=9, color=INK, fontweight=600)
        sub = row_sub(z)
        if sub:
            ax.text(-0.06, i + 0.70, sub, ha='right', va='center',
                    fontsize=6.0, color=(*INK, 0.6))
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
            has2 = line2(c) is not None
            y_imp = 0.24 if has2 else 0.30
            ax.text(j + 0.5, i + y_imp, f'{imp_txt} pts{star}',
                    ha='center', va='center', fontsize=8.6, fontweight=700,
                    color=txt)
            sub_c = (*txt, 0.92)
            if has2:
                ax.text(j + 0.5, i + 0.53, line2(c), ha='center',
                        va='center', fontsize=6.2, color=sub_c)
                y3 = 0.78
            else:
                y3 = 0.62
            ax.text(j + 0.5, i + y3, line3(c), ha='center', va='center',
                    fontsize=6.2, color=sub_c)
    ax.text(0, 5.32, f'{sum_note} · * = under {FADE_N} {noun}',
            fontsize=6.4, color=(*INK, 0.8))
    ax.set_title(title, fontsize=11, color=INK, pad=10, loc='left',
                 **TITLE_FONT)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print('Loading pitch cache + tables ...')
    with open(PICKLE, 'rb') as f:
        allp = pickle.load(f)
    meta = json.load(open(METADATA))
    sdw, ctw = meta['sdPlusWeights'], meta['ctPlusWeights']
    reanchor = meta.get('plusReanchor', {})
    lb = json.load(open(HITTER_LB))
    lb_row = {r['hitter']: r for r in lb
              if r.get('hitter') in {h for h, _ in REQUESTS}
              and r.get('team') == 'ROC'}

    mlb = [p for p in allp if p.get('_source', 'MLB') == 'MLB' and is_eligible(p)]

    def dv(p):
        z, c, cat = classify_zone(p), get_count(p), cat_of(p)
        s = sdw[f'{z}|{cat}|{c[0]}-{c[1]}|swing']['rv']
        t = sdw[f'{z}|{cat}|{c[0]}-{c[1]}|take']['rv']
        return (s - t) if classify_decision(p) == 'swing' else (t - s)

    lg_mean_dv = sum(dv(p) for p in mlb) / len(mlb)

    def sd_atom(p):
        st = count_state_hitter(p)
        return (100.0 * dv(p) / lg_mean_dv, classify_zone(p), st)

    def ct_tuple(p):
        cell = ctw[f'{classify_zone(p)}|{get_count(p)[0]}-{get_count(p)[1]}']
        lev = cell['rv_contact'] - cell['rv_whiff']
        E = 1.0 - cell['p_whiff']
        made = 1.0 if classify_contact_outcome(p) == 'contact' else 0.0
        return (lev * (made - E), lev * lev * E * (1 - E), lev * made,
                lev * E, classify_zone(p), count_state_hitter(p))

    print('Building SD league tables ...')
    lg_sd_atoms = defaultdict(list)
    for p in mlb:
        a = sd_atom(p)
        if a[2] is None:
            continue
        lg_sd_atoms['ALL'].append(a)
        lg_sd_atoms[cat_of(p)].append(a)
    sd_lg = {g: sd_league(v) for g, v in lg_sd_atoms.items()}

    # League state shares for CT column headers.
    ct_lg_state = defaultdict(lambda: defaultdict(int))
    for p in mlb:
        if not is_ct_eligible(p):
            continue
        st = count_state_hitter(p)
        if st is None:
            continue
        for g in ('ALL', cat_of(p)):
            ct_lg_state[g][st] += 1
    ct_lg_state = {g: {st: n / sum(d.values()) for st, n in d.items()}
                   for g, d in ct_lg_state.items()}

    for name, mkey in REQUESTS:
        cfg = METRICS[mkey]
        label, noun = cfg['label'], cfg['noun']
        last, first = [s.strip() for s in name.split(',')]
        outdir = os.path.join(OUT_ROOT, last)
        os.makedirs(outdir, exist_ok=True)
        row = lb_row.get(name, {})
        printed = row.get(cfg['lb_key'])
        pctl = row.get(cfg['lb_key'] + '_pctl')
        f_anchor = reanchor.get(cfg['lb_key'], 1.0)

        pitches = [p for p in allp
                   if p.get('Batter') == name and p.get('BTeam') == 'ROC']

        if mkey == 'SD':
            elig = [p for p in pitches if is_eligible(p)]
            atoms = {'ALL': [], 'FB': [], 'BRK': [], 'OFF': []}
            for p in elig:
                a = sd_atom(p)
                if a[2] is None:
                    continue
                atoms['ALL'].append(a)
                atoms[cat_of(p)].append(a)
            n_all = len(atoms['ALL'])
            # Uncalibrated ALL-panel gap -> exact kappa vs theory fallback.
            raw_cells = sd_cells(atoms['ALL'], sd_lg['ALL'], cfg['k'], 1.0)
            G = sum(c['impact'] for c in raw_cells.values())
        else:
            elig = [p for p in pitches if is_ct_eligible(p)]
            tuples = {'ALL': [], 'FB': [], 'BRK': [], 'OFF': []}
            for p in elig:
                t = ct_tuple(p)
                if t[5] is None:
                    continue
                tuples['ALL'].append(t)
                tuples[cat_of(p)].append(t)
            n_all = len(tuples['ALL'])
            T = sum(t[3] for t in tuples['ALL'])
            raw_cells = ct_cells(tuples['ALL'], T, cfg['k'], 1.0)
            G = sum(c['impact'] for c in raw_cells.values())

        kappa_theory = f_anchor * n_all / (n_all + cfg['n0'])
        if printed is not None and abs(G) >= 1.0:
            kappa = (printed - 100.0) / G
            k_src = 'exact'
            if not (0.2 <= kappa <= 2.5):
                kappa, k_src = kappa_theory, 'theory (exact ill-conditioned)'
        else:
            kappa, k_src = kappa_theory, 'theory (gap too small)'
        print(f'{name} {label}: card {printed}, page gap {G:.2f} pts, '
              f'kappa {kappa:.3f} ({k_src})')

        # Build panels.
        panels = []
        for key in ['ALL'] + CATS:
            if mkey == 'SD':
                sub = atoms[key]
                if key != 'ALL' and len(sub) < cfg['gate']:
                    continue
                cells = sd_cells(sub, sd_lg[key], cfg['k'], kappa)
                state_n = defaultdict(int)
                for _v, _z, st in sub:
                    state_n[st] += 1
                panels.append((key, cells, state_n, len(sub)))
            else:
                sub = tuples[key]
                if key != 'ALL' and len(sub) < cfg['gate']:
                    continue
                cells = ct_cells(sub, T, cfg['k'], kappa)
                state_n = defaultdict(int)
                for t in sub:
                    state_n[t[5]] += 1
                panels.append((key, cells, state_n, len(sub)))

        # Callouts from category panels, impact minus lam * SE.
        ranked = []
        for key, cells, _sn, _n in panels[1:]:
            for cell, c in cells.items():
                se = sd_se(c, kappa) if mkey == 'SD' else c['se']
                d = abs(c['impact']) - cfg['lam'] * se
                score = math.copysign(max(0.0, d), c['impact']) if d > 0 else 0.0
                ranked.append((score, c['impact'], key, cell, c))
        ranked.sort(key=lambda r: r[0])
        costs = [r for r in ranked[:2] if r[0] < 0]
        helps = [r for r in ranked[-2:][::-1] if r[0] > 0]

        def sentence(r):
            _, imp, key, (z, st), c = r
            verb = 'adds' if imp > 0 else 'costs'
            base = (f'{CAT_NAMES[key]} in {ZONE_NAMES[z]} {STATE_PROSE[st]}: '
                    f'{verb} {abs(imp):.1f} pts')
            if mkey == 'SD':
                m, e = abs(c['mix']), abs(c['exec'])
                drv = ('mostly count mix' if m > 2 * e else
                       'mostly execution' if e > 2 * m else
                       'count mix and execution')
                base += f' ({drv})'
            return base

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
        hd.text(0, 0.97, f'{first} {last}: {cfg["title"]}',
                fontsize=23, color=INK, va='top', **TITLE_FONT)
        sub = (f'2026 season (ROC) · zone × count impact matrix on the '
               f'leaderboard {label} scale · shrinkage k={cfg["k"]}, '
               f'callouts impact minus {cfg["lam"]} SE (both split-half '
               f'tuned)')
        hd.text(0, 0.78, sub, fontsize=fit_font(sub, base=11.5),
                color=(*INK, 0.75), va='top')
        if mkey == 'SD':
            expl = (
                f'Each cell: its IMPACT in {label} points; the 15 cells sum '
                f'to his card {label} minus 100. Cells match the card’s '
                'diet-neutral math: every zone carries the LEAGUE’s share of '
                'pitches\n(so which zones pitchers put him in moves nothing), '
                'and within a zone his impact is his count-state share times '
                'his decision grade there versus the league’s.\nAHEAD/BEHIND '
                'are the hitter’s counts. Grade = mean per-decision value on '
                'the 100 scale, unshrunken; share = his split of that zone’s '
                'decisions across count states.')
        else:
            expl = (
                f'Each cell: its IMPACT in {label} points; the 15 cells sum '
                f'to his card {label} minus 100, and the category panels '
                'split that gap (approximately — each panel\napplies its own '
                'small-sample shrinkage). Cells match the card’s '
                'execution-only math: leverage-weighted contact made minus '
                'contact\nexpected on HIS swings there — a cell where he '
                'converts at the league rate scores 0 no matter how often '
                'he swings there (swing selection is SD+’s job).\n'
                'AHEAD/BEHIND are the hitter’s counts. Cell CT+ = that '
                'cell’s own contact-vs-expected ratio on the 100 scale.')
        hd.text(0, 0.68, expl, fontsize=9.8, color=INK, va='top',
                linespacing=1.6)
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
        cat_line = ' · '.join(
            f'{CAT_NAMES[k]} {sum(c["impact"] for c in cells.values()):.1f} pts'
            for k, cells, _sn, _n in panels[1:])
        if cat_line:
            hd.text(0, -0.13, f'Category totals: {cat_line}',
                    fontsize=fit_font(f'Category totals: {cat_line}',
                                      base=10.5),
                    color=INK, va='top', fontweight=600)
        lg_ax = fig.add_subplot(gs[0, 2])
        draw_zone_legend(lg_ax)

        for i, (key, cells, state_n, n_sub) in enumerate(panels):
            total = sum(c['impact'] for c in cells.values())
            if key == 'ALL':
                ttl = (f'All pitches · {label} {printed:.0f} (card) · '
                       f'{n_sub} {noun}')
                note = (f'cells sum to {label} − 100 = {total:.1f} pts')
            else:
                ttl = f'vs {CAT_NAMES[key]} · {total:.1f} pts · {n_sub} {noun}'
                note = (f'cells sum to {total:.1f} pts'
                        + (' of the season total' if mkey == 'CT' else
                           f' vs a league-average {CAT_NAMES[key]} map'))
            r, c = divmod(i, mm_cols)
            ax = fig.add_subplot(gs[1 + r, c])
            if mkey == 'SD':
                draw_panel_sd(ax, ttl, cells, state_n, sd_lg[key], n_sub,
                              note, cfg['span'], noun)
            else:
                draw_panel_ct(ax, ttl, cells, state_n, ct_lg_state[key],
                              n_sub, note, cfg['span'], noun)

        out = os.path.join(outdir, f'{mkey}Zones_{last}{first}_impact.png')
        fig.savefig(out, facecolor=CREAM, bbox_inches='tight')
        plt.close(fig)
        all_sum = sum(c['impact'] for c in panels[0][1].values())
        print(f'  -> {out}')
        print(f'  VERIFY: ALL cells sum {all_sum:.2f} vs card-100 = '
              f'{(printed - 100):.2f}' if printed is not None else '')


if __name__ == '__main__':
    main()
