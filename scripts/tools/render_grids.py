#!/usr/bin/env python3
"""Render LA x Spray appendix grids (league reference + per-hitter BIP)."""
import json, gzip
BASE = '/Users/wallyhuron/Huronalytics'
OUT = json.load(open(f'{BASE}/scripts/wsh_analysis_output.json'))
# 6/17 published reference tables (sacqZones) come from the embedded data, matching wsh_hitter_analysis.py
META = json.loads(gzip.open(f'{BASE}/data/data_embedded.json.gz', 'rt').read())['metadata']

LA_LABELS = ['<-10','-10/0','0/5','5/10','10/15','15/20','20/25','25/30','30/35','35/40','40/50','50+']
SPRAY = ['pull','pull_side','center_pull','center_oppo','oppo_side','oppo']
SPRAY_LBL = {'pull':'Pull','pull_side':'Pull-sd','center_pull':'Ctr-pull',
             'center_oppo':'Ctr-opp','oppo_side':'Oppo-sd','oppo':'Oppo'}
MIN = 20

hand_z, pool_z = {}, {}
for z in META['sacqZones']:
    rec = {'woba': z['woba'], 'quality': z['quality'], 'count': z['count']}
    if z['bats'] is None:
        pool_z[(z['spray'], z['laBin'])] = rec
    else:
        hand_z[(z['spray'], z['laBin'], z['bats'])] = rec

def zone(direction, lab, bats):
    h = hand_z.get((direction, lab, bats))
    if h and h['count'] >= MIN and h['woba'] is not None:
        return h['woba'], h['quality']
    p = pool_z.get((direction, lab))
    if p and p['count'] >= MIN and p['woba'] is not None:
        return p['woba'], p['quality']
    return None

def ref_table(bats):
    lines = [f"LEAGUE xwOBAsp ZONE wOBA, {bats}HH  ( ()=quality zone, wOBA>=.500 )"]
    hdr = f"{'spray\\LA':<9}" + "".join(f"{l:>7}" for l in LA_LABELS)
    lines.append(hdr)
    for d in SPRAY:
        cells = []
        for lab in range(12):
            z = zone(d, lab, bats)
            if z is None:
                cells.append(f"{'.':>7}")
            else:
                w, q = z
                s = f"{w:.3f}".lstrip('0')
                cells.append(f"{('('+s+')') if q else (' '+s+' '):>7}")
        lines.append(f"{SPRAY_LBL[d]:<9}" + "".join(cells))
    return "\n".join(lines)

def hitter_grid(title, grid, bats, xsp, sacq, sacq_q, sacq_tot):
    xs = f"{xsp:.3f}".lstrip('0') if xsp is not None else 'n/a'
    lines = [f"{title}   xwOBAsp={xs}  SACQ%={sacq}% ({sacq_q}/{sacq_tot} BIP in quality zones)"]
    hdr = f"{'spray\\LA':<9}" + "".join(f"{l:>7}" for l in LA_LABELS) + f"{'TOT':>6}"
    lines.append(hdr)
    col_tot = [0]*12
    for d in SPRAY:
        row = grid.get(d, {})
        cells = []
        rt = 0
        for lab in range(12):
            n = row.get(str(lab), row.get(lab, 0)) or 0
            rt += n; col_tot[lab] += n
            if n == 0:
                cells.append(f"{'.':>7}")
            else:
                z = zone(d, lab, bats)
                q = z[1] if z else False
                cells.append(f"{('('+str(n)+')') if q else str(n):>7}")
        lines.append(f"{SPRAY_LBL[d]:<9}" + "".join(cells) + f"{rt:>6}")
    lines.append(f"{'TOTAL':<9}" + "".join(f"{t:>7}" for t in col_tot) + f"{sum(col_tot):>6}")
    return "\n".join(lines)

res = OUT['results']
blocks = []
blocks.append(ref_table('L'))
blocks.append(ref_table('R'))

SINGLE_BATS = {'Wood, James':'L','Lile, Daylen':'L','Abrams, CJ':'L','Young, Jacob':'R',
               'García Jr., Luis':'L','Mead, Curtis':'R','Vivas, Jorbit':'L','Tena, José':'L',
               'Crews, Dylan':'R'}  # MLB (WSH) only
for name, bats in SINGLE_BATS.items():
    m = res[name]['all']
    blocks.append(hitter_grid(f"{name} ({bats}HH)", m['grid'], bats, m['xwOBAsp'], m['sacqPct'], m['sacqQ'], m['sacqTot']))

for name in ['Nuñez, Nasim','Ruiz, Keibert','Millas, Drew']:
    for side, bats, lbl in [('L','L','L-swing/vs RHP'),('R','R','R-swing/vs LHP')]:
        m = res[name][side]
        blocks.append(hitter_grid(f"{name}: {lbl}", m['grid'], bats, m['xwOBAsp'], m['sacqPct'], m['sacqQ'], m['sacqTot']))

txt = "\n\n".join(blocks)
open(f'{BASE}/scripts/wsh_grids.txt','w').write(txt)
print(txt)
