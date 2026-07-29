"""
Batch Pitcher Card Generator
Generates dark-themed pitcher stat cards for all pitchers on a team for a given date.

Usage:
    1. Edit the Settings block at the top of main()
    2. python3 generate_batch_cards.py
"""

import sys
import os
import json
import urllib.request
import urllib.parse
import time as time_module
from datetime import datetime
from collections import defaultdict
from math import atan2, sin, cos

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle
from PIL import Image
from io import BytesIO
import numpy as np
import gspread
from sheets_append import _workbook_id_for_team

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
# Sheet routing comes from sheets_append._workbook_id_for_team (per team).

AL_TEAMS = {'ATH','BAL','BOS','CLE','CWS','DET','HOU','KCR','LAA','MIN','NYY','SEA','TBR','TEX','TOR'}
NL_TEAMS = {'ARI','ATL','CHC','CIN','COL','LAD','MIA','MIL','NYM','PHI','PIT','SDP','SFG','STL','WSH'}

TEAM_ABBREV_TO_ID = {
    'ARI':109,'ATL':144,'BAL':110,'BOS':111,'CHC':112,'CWS':145,'CIN':113,'CLE':114,
    'COL':115,'DET':116,'HOU':117,'KCR':118,'LAA':108,'LAD':119,'MIA':146,'MIL':158,
    'MIN':142,'NYM':121,'NYY':147,'ATH':133,'PHI':143,'PIT':134,'SDP':135,'SFG':137,
    'SEA':136,'STL':138,'TBR':139,'TEX':140,'TOR':141,'WSH':120,
}

TEAM_NAME_TO_ABBREV = {
    'Arizona Diamondbacks':'ARI','Athletics':'ATH','Atlanta Braves':'ATL',
    'Baltimore Orioles':'BAL','Boston Red Sox':'BOS','Chicago Cubs':'CHC',
    'Chicago White Sox':'CWS','Cincinnati Reds':'CIN','Cleveland Guardians':'CLE',
    'Colorado Rockies':'COL','Detroit Tigers':'DET','Houston Astros':'HOU',
    'Kansas City Royals':'KCR','Los Angeles Angels':'LAA','Los Angeles Dodgers':'LAD',
    'Miami Marlins':'MIA','Milwaukee Brewers':'MIL','Minnesota Twins':'MIN',
    'New York Mets':'NYM','New York Yankees':'NYY','Philadelphia Phillies':'PHI',
    'Pittsburgh Pirates':'PIT','San Diego Padres':'SDP','San Francisco Giants':'SFG',
    'Seattle Mariners':'SEA','St. Louis Cardinals':'STL','Tampa Bay Rays':'TBR',
    'Texas Rangers':'TEX','Toronto Blue Jays':'TOR','Washington Nationals':'WSH',
}

PITCH_COLORS = {
    'FF':'#0072B2','SI':'#FFD700','FC':'#8B5A2B','SL':'#D55E00',
    'ST':'#56B4E9','CU':'#332288','SV':'#882255','CH':'#009E73','FS':'#CC79A7','KN':'#AAAAAA',
    'EP':'#888888'
}
PITCH_NAMES = {
    'FF':'Fastball','SI':'Sinker','FC':'Cutter','SL':'Slider',
    'ST':'Sweeper','CU':'Curveball','SV':'Slurve','CH':'Changeup','FS':'Splitter','KN':'Knuckleball',
    'EP':'Eephus'
}
PITCH_ORDER = ['FF','SI','FC','SL','ST','CU','SV','CH','FS','KN']
# Bunts are not swings (matches Cards.SWING_DESC / pipeline_utils.SWING_DESCRIPTIONS)
# so Whiff%/Chase%/Swing% match the leaderboard. STRIKE_DESC still counts foul bunt.
SWING_DESC = ['Swinging Strike','Foul','In Play']
STRIKE_DESC = ['Called Strike','Swinging Strike','Foul','Foul Bunt','In Play']

# Batted ball type colors (for distribution chart)
BB_COLORS = {
    'ground_ball': '#00d4ff',   # cyan (matches card accent)
    'line_drive':  '#FF6B6B',   # soft coral
    'fly_ball':    '#7B68EE',   # medium slate blue
    'popup':       '#FF9F43',   # warm amber
}
BB_LABELS = {
    'ground_ball': 'Ground Ball',
    'line_drive':  'Line Drive',
    'fly_ball':    'Fly Ball',
    'popup':       'Popup',
}
BB_TYPES = ['ground_ball', 'line_drive', 'fly_ball', 'popup']

# Layout constants (v30)
TABLE_LEFT_FIG=0.01; TABLE_RIGHT_FIG=0.99; PLOT_LEFT=0.585; PLOT_RIGHT=0.99
USAGE_SHIFT=0.18; DIVIDER_COL=14; PLATE_HALF=17/12/2
FIG_W=16; FIG_H=15; DPI=100; SAVE_DPI=150
BG='#141619'; ACCENT='#00d4ff'; DARK_CELL='#1e2127'; DARKER='#0d0f12'

MLB_ID_CACHE_PATH = 'mlb_id_cache.json'
OUTPUT_DIR = '/Users/wallyhuron/Downloads/'
METADATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'metadata_rs.json')

# Mapping: card column header → metadata league average key
PCT_COLOR_COLS = {
    'Strike%': 'strikePct',
    'Zone%':   'izPct',
    'Whiff%':  'swStrPct',
}


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def sf(v):
    if v is None or v == '': return None
    try: return float(v)
    except: return None

def pct_cell_color(value_str, league_avg, row_bg_hex):
    """Return cell background color based on how a percentage compares to league average.
    value_str: cell text like '65.3%'
    league_avg: league average as decimal (e.g. 0.6587)
    row_bg_hex: base row background color (e.g. '#1e2127')
    """
    if league_avg is None or not value_str or value_str == '—':
        return None
    try:
        val = float(value_str.replace('%', ''))
    except (ValueError, AttributeError):
        return None
    avg_pct = league_avg * 100
    diff = val - avg_pct  # positive = above average
    # Scale: ±10 pp maps to full intensity
    intensity = max(-1.0, min(1.0, diff / 10.0))
    rb = int(row_bg_hex[1:3], 16)
    rg = int(row_bg_hex[3:5], 16)
    rbb = int(row_bg_hex[5:7], 16)
    if intensity >= 0:
        target = (0, 160, 0)
    else:
        target = (160, 0, 0)
        intensity = abs(intensity)
    alpha = intensity * 0.4
    r = int(rb * (1 - alpha) + target[0] * alpha)
    g = int(rg * (1 - alpha) + target[1] * alpha)
    b = int(rbb * (1 - alpha) + target[2] * alpha)
    return f'#{r:02x}{g:02x}{b:02x}'

def avg_tilt(tilts):
    valid = [t for t in tilts if t and t != '']
    if not valid: return '—'
    sins, coss = [], []
    for t in valid:
        parts = str(t).split(':')
        if len(parts) != 2: continue
        h, m = int(parts[0]), int(parts[1])
        if h == 12: h = 0
        a = (h*60+m)/720*2*3.14159
        sins.append(sin(a)); coss.append(cos(a))
    if not sins: return '—'
    am = (atan2(sum(sins)/len(sins), sum(coss)/len(coss))*720/(2*3.14159)) % 720
    h, m = int(am//60), int(am%60)
    if h == 0: h = 12
    return f'{h}:{m:02d}'

def fmt_fi(v):
    if v is None: return '—'
    neg = v < 0; av = abs(v); ft = int(av); inc = round((av-ft)*12)
    if inc == 12: ft += 1; inc = 0
    s = f"{ft}'{inc}\""; return f"-{s}" if neg else s

def compute_iz(p):
    px, pz, st, sb = sf(p.get('PlateX')), sf(p.get('PlateZ')), sf(p.get('SzTop')), sf(p.get('SzBot'))
    if any(v is None for v in [px, pz, st, sb]): return None
    return abs(px) <= 0.83 and pz >= sb-0.121 and pz <= st+0.121

def luminance(hc):
    r, g, b = int(hc[1:3],16)/255, int(hc[3:5],16)/255, int(hc[5:7],16)/255
    r = r/12.92 if r<=0.03928 else ((r+0.055)/1.055)**2.4
    g = g/12.92 if g<=0.03928 else ((g+0.055)/1.055)**2.4
    b = b/12.92 if b<=0.03928 else ((b+0.055)/1.055)**2.4
    return 0.2126*r + 0.7152*g + 0.0722*b

def badge_text_color(hc):
    return 'black' if luminance(hc) > 0.25 else 'white'

def is_barrel(ev, la):
    """Statcast barrel definition (LA in [8,50], EV>=98, EV*1.5-LA>=117, EV+LA>=124)."""
    if ev is None or la is None:
        return False
    return (la >= 8 and la <= 50 and ev >= 98 and
            ev * 1.5 - la >= 117 and
            ev + la >= 124)

def outs_to_ip_str(outs):
    return f"{outs//3}.{outs%3}"

def _fullname_to_lastfirst(full_name):
    parts = full_name.strip().split()
    if len(parts) <= 1: return full_name
    suffixes = {'jr.','jr','sr.','sr','ii','iii','iv','v'}
    suffix = ''
    if len(parts) > 2 and parts[-1].lower().rstrip('.') in suffixes:
        suffix = ' ' + parts.pop()
    return parts[-1] + suffix + ', ' + ' '.join(parts[:-1])


# ═══════════════════════════════════════════════════════════════
# MLB API FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def load_mlb_id_cache():
    if os.path.exists(MLB_ID_CACHE_PATH):
        with open(MLB_ID_CACHE_PATH) as f:
            return json.load(f)
    return {}

def save_mlb_id_cache(cache):
    with open(MLB_ID_CACHE_PATH, 'w') as f:
        json.dump(cache, f)

def lookup_mlb_id(player_name, team_abbrev, cache):
    cache_key = f"{player_name}|{team_abbrev}"
    if cache_key in cache:
        return cache[cache_key]
    parts = player_name.split(', ')
    search_name = f"{parts[1]} {parts[0]}" if len(parts) == 2 else player_name
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/search?names={urllib.parse.quote(search_name)}&sportIds=1,11,12,13,14&hydrate=currentTeam&limit=25"
        req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        team_id = TEAM_ABBREV_TO_ID.get(team_abbrev)
        people = data.get('people', [])
        if team_id and people:
            for person in people:
                ct = person.get('currentTeam', {})
                parent = ct.get('parentOrgId') or ct.get('id')
                if parent == team_id or ct.get('id') == team_id:
                    cache[cache_key] = person['id']; return person['id']
        if people:
            last_name = parts[0] if len(parts) == 2 else player_name.split()[-1]
            for person in people:
                if person.get('lastName','').lower() == last_name.lower():
                    cache[cache_key] = person['id']; return person['id']
            cache[cache_key] = people[0]['id']; return people[0]['id']
    except Exception as e:
        print(f"  Warning: MLB ID lookup failed for {player_name}: {e}")
    cache[cache_key] = None
    return None

def fetch_player_metadata(mlb_id):
    """Fetch age and hand from MLB API."""
    if not mlb_id: return {'age': '??', 'hand': 'R'}
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{mlb_id}"
        req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        p = data.get('people', [{}])[0]
        return {
            'age': str(p.get('currentAge', '??')),
            'hand': p.get('pitchHand', {}).get('code', 'R'),
        }
    except:
        return {'age': '??', 'hand': 'R'}

def fetch_headshot(mlb_id):
    """Fetch and process MLB headshot."""
    try:
        url = f'https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_426,q_auto:best/v1/people/{mlb_id}/headshot/67/current'
        req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
        img = Image.open(BytesIO(urllib.request.urlopen(req, timeout=10).read()))
        ha = np.array(img.convert('RGBA'))
        gm = ((np.abs(ha[:,:,0].astype(int)-ha[:,:,1].astype(int))<15) &
              (np.abs(ha[:,:,1].astype(int)-ha[:,:,2].astype(int))<15) &
              (ha[:,:,0]>170) & (ha[:,:,0]<230))
        ha[gm] = [255,255,255,255]
        return Image.fromarray(ha)
    except:
        # Return a placeholder
        img = Image.new('RGBA', (213, 320), (50, 50, 50, 255))
        return img

def fetch_game_pks_for_date(date_str, include_live=False):
    url = f"https://statsapi.mlb.com/api/v1/schedule?date={date_str}&sportId=1&gameType=R,F,D,L,W"
    try:
        req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        game_pks = []
        for date_data in data.get('dates', []):
            for game in date_data.get('games', []):
                state = game.get('status', {}).get('abstractGameState', '')
                if state == 'Final' or (include_live and state == 'Live'):
                    game_pks.append(game['gamePk'])
        return game_pks
    except Exception as e:
        print(f"  Error fetching schedule: {e}")
        return []

def fetch_boxscore(game_pk):
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    try:
        req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            box = json.loads(resp.read())
    except Exception as e:
        print(f"  Error fetching boxscore for {game_pk}: {e}")
        return None
    result = {'pitchers': []}
    for side in ['away', 'home']:
        team_data = box.get('teams', {}).get(side, {})
        team_name = team_data.get('team', {}).get('name', '')
        team_abbrev = TEAM_NAME_TO_ABBREV.get(team_name, team_name)
        pitcher_ids = team_data.get('pitchers', [])
        players = team_data.get('players', {})
        for idx, pid in enumerate(pitcher_ids):
            p = players.get(f'ID{pid}', {})
            full_name = p.get('person', {}).get('fullName', '')
            stats = p.get('stats', {}).get('pitching', {})
            if not stats: continue
            last_first = p.get('person', {}).get('lastFirstName', '')
            if not last_first and full_name:
                last_first = _fullname_to_lastfirst(full_name)
            result['pitchers'].append({
                'name': last_first, 'team': team_abbrev,
                'outs': stats.get('outs', 0),
                'r': stats.get('runs', 0), 'er': stats.get('earnedRuns', 0),
                'h': stats.get('hits', 0), 'hr': stats.get('homeRuns', 0),
                'so': stats.get('strikeOuts', 0), 'bb': stats.get('baseOnBalls', 0),
                'tbf': stats.get('battersFaced', 0),
            })
    return result

def fetch_boxscores_for_team(date_str, team_abbrev, include_live=False, game_pk=None):
    """Fetch boxscore stats for all pitchers on a team for a given date."""
    if game_pk:
        game_pks = [int(game_pk)]
        print(f"  Using game PK: {game_pk}")
    else:
        print(f"  Fetching boxscores for {date_str}...")
        game_pks = fetch_game_pks_for_date(date_str, include_live=include_live)
        status = "games (including live)" if include_live else "completed games"
        print(f"  Found {len(game_pks)} {status}")
    pitcher_stats = {}
    for gpk in game_pks:
        box = fetch_boxscore(gpk)
        if not box: continue
        for p in box['pitchers']:
            if p['team'] == team_abbrev:
                pitcher_stats[p['name']] = p
        time_module.sleep(0.1)
    return pitcher_stats


# ═══════════════════════════════════════════════════════════════
# CARD RENDERING (v30)
# ═══════════════════════════════════════════════════════════════
def render_card(config, pitches, output_file):
    """Render a single pitcher card. config has display_name, hand, team, age, game_date, stat_headers, stat_values, headshot, mlb_id."""
    headshot = config['headshot']

    # Compute pitch data
    locations = {'L': defaultdict(list), 'R': defaultdict(list)}
    sz_tops, sz_bots = [], []
    groups = defaultdict(list)

    for p in pitches:
        pt = p.get('Pitch Type', '')
        hb, ivb = p.get('HorzBrk'), p.get('IndVertBrk')
        if pt and hb is not None and hb != '' and ivb is not None and ivb != '':
            try: groups[pt].append((float(hb), float(ivb)))
            except: pass
        bh = p.get('Bats', '')
        px, pz = p.get('PlateX'), p.get('PlateZ')
        szt, szb = p.get('SzTop'), p.get('SzBot')
        if bh in ('L','R') and pt and px is not None and px != '' and pz is not None and pz != '':
            try:
                desc = p.get('Description', '')
                is_b = str(p.get('Barrel', '')).strip() == '6'
                locations[bh][pt].append((float(px), float(pz), desc, is_b))
            except: pass
        if szt is not None and szt != '' and szb is not None and szb != '':
            try: sz_tops.append(float(szt)); sz_bots.append(float(szb))
            except: pass

    avg_top = np.mean(sz_tops) if sz_tops else 3.5
    avg_bot = np.mean(sz_bots) if sz_bots else 1.5
    sorted_types = [pt for pt in PITCH_ORDER if pt in groups]

    # Batted ball distribution per pitch type
    bb_by_pitch = defaultdict(lambda: {'ground_ball': 0, 'line_drive': 0, 'fly_ball': 0, 'popup': 0, 'hh': 0, 'brl': 0})
    for p in pitches:
        pt = p.get('Pitch Type', '')
        if not pt or p.get('Description') != 'In Play':
            continue
        bbt = str(p.get('BBType', '')).strip()
        if not bbt or bbt.startswith('bunt'):
            continue
        if bbt in BB_TYPES:
            bb_by_pitch[pt][bbt] += 1
        ev = sf(p.get('ExitVelo'))
        if ev is not None and ev >= 95:
            bb_by_pitch[pt]['hh'] += 1
        if str(p.get('Barrel', '')).strip() == '6':
            bb_by_pitch[pt]['brl'] += 1

    if not sorted_types:
        print(f"  WARNING: No pitch type data for {config['display_name']}, skipping")
        return False

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax_main = fig.add_axes([0,0,1,1])
    ax_main.set_xlim(0, FIG_W); ax_main.set_ylim(0, FIG_H)
    ax_main.axis('off'); ax_main.set_facecolor(BG)

    # Stripe — usage-ordered, equal widths, aligned with photo
    photo_left = TABLE_LEFT_FIG * FIG_W
    stripe_bottom = 14.90
    stripe_height = 0.22
    stripe_x = photo_left
    total_w = FIG_W * TABLE_RIGHT_FIG - photo_left
    stripe_counts = {pt: sum(1 for p in pitches if p.get('Pitch Type') == pt) for pt in sorted_types}
    stripe_order = sorted(sorted_types,
        key=lambda pt: (-stripe_counts[pt], PITCH_ORDER.index(pt) if pt in PITCH_ORDER else 999))
    for pt in stripe_order:
        w = total_w / len(sorted_types)
        ax_main.add_patch(Rectangle((stripe_x, stripe_bottom), w, stripe_height,
            facecolor=PITCH_COLORS.get(pt, '#999'), edgecolor='none', zorder=6))
        stripe_x += w

    # Photo
    photo_w = 1.4; photo_h = photo_w * headshot.size[1] / headshot.size[0]
    photo_top = 14.85; photo_bottom = photo_top - photo_h
    ax_main.imshow(np.array(headshot), extent=[photo_left, photo_left+photo_w, photo_bottom, photo_top], aspect='auto', zorder=2, interpolation='antialiased')
    ax_main.add_patch(Rectangle((photo_left, photo_bottom), photo_w, photo_h, fill=False, edgecolor=ACCENT, linewidth=1.5, alpha=0.5, zorder=3))

    photo_right = photo_left + photo_w; text_x = photo_right + 0.3
    ax_main.text(text_x, photo_top-0.1, config['display_name'], fontsize=32, fontfamily='DIN Condensed', color='white', va='top', fontweight='bold')
    hand_code = 'LHP' if config['hand'] == 'L' else 'RHP'
    ax_main.text(text_x, photo_top-0.85, f"{hand_code}  |  {config['team']}  |  Age: {config['age']}", fontsize=12, fontfamily='Avenir Next', color='#888', va='top')
    ax_main.text(text_x, photo_top-1.5, config['game_date'], fontsize=24, fontfamily='DIN Condensed', color=ACCENT, va='top')

    # Stat line
    col_w = 0.6; cell_h = 0.42
    stat_y_header = photo_bottom - 0.5; stat_y_value = stat_y_header - cell_h
    for i in range(len(config['stat_headers'])):
        x = photo_left + i * col_w
        ax_main.add_patch(Rectangle((x, stat_y_header), col_w, cell_h, facecolor=DARKER, edgecolor='#333840', linewidth=0.8))
        ax_main.text(x+col_w/2, stat_y_header+cell_h/2, config['stat_headers'][i], fontsize=10, ha='center', va='center', color=ACCENT, fontweight='bold', fontfamily='Avenir Next')
        ax_main.add_patch(Rectangle((x, stat_y_value), col_w, cell_h, facecolor=DARK_CELL, edgecolor='#333840', linewidth=0.8))
        ax_main.text(x+col_w/2, stat_y_value+cell_h/2, config['stat_values'][i], fontsize=12, ha='center', va='center', color='white', fontweight='bold', fontfamily='Avenir Next')
    ax_main.add_patch(Rectangle((photo_left, stat_y_value), len(config['stat_headers'])*col_w, stat_y_header+cell_h-stat_y_value, fill=False, edgecolor=ACCENT, linewidth=2, zorder=5))

    # Movement plot
    ax_plot = fig.add_axes([PLOT_LEFT, 0.58, PLOT_RIGHT-PLOT_LEFT, 0.40])
    ax_plot.set_xlim(-25,25); ax_plot.set_ylim(-25,25)
    ax_plot.axhline(y=0, color='#333840', linestyle='--', linewidth=0.5)
    ax_plot.axvline(x=0, color='#333840', linestyle='--', linewidth=0.5)
    ax_plot.set_xlabel('Horizontal Break (in)', fontsize=10, color='#ccc', fontweight='bold', fontfamily='Avenir Next')
    ax_plot.set_ylabel('Induced Vertical Break (in)', fontsize=10, color='#ccc', fontweight='bold', fontfamily='Avenir Next')
    ax_plot.tick_params(labelsize=8, colors='#999')
    ax_plot.set_xticks(range(-25,26,5)); ax_plot.set_yticks(range(-25,26,5))
    ax_plot.grid(True, alpha=0.2, color='#333840'); ax_plot.set_facecolor('#1a1d21')
    for spine in ax_plot.spines.values(): spine.set_color('#444')

    for pt in PITCH_ORDER:
        if pt not in groups: continue
        xs, ys = zip(*groups[pt]); color = PITCH_COLORS[pt]
        ax_plot.scatter(xs, ys, c=color, s=65, alpha=1.0, edgecolors='none', zorder=3)
        if len(groups[pt]) >= 4:
            cov = np.cov(xs, ys); vals, vecs = np.linalg.eigh(cov)
            ax_plot.add_patch(Ellipse((np.mean(xs), np.mean(ys)), 2*1.5*np.sqrt(vals[1]), 2*1.5*np.sqrt(vals[0]),
                angle=np.degrees(np.arctan2(vecs[1,1], vecs[0,1])), fill=False, edgecolor=color, linewidth=1.2, linestyle='--', alpha=0.7))

    legend_handles = [mpatches.Patch(color=PITCH_COLORS[pt], label=f'{pt} - {"Fastball" if pt=="FF" else PITCH_NAMES[pt]}') for pt in sorted_types]
    leg = ax_plot.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5,-0.07), ncol=min(len(sorted_types),4), fontsize=7.5, frameon=False, handlelength=1.2, columnspacing=1.2)
    for t in leg.get_texts(): t.set_color('#ccc')

    # Location plots
    LOC_TITLE_Y=0.565; LOC_BOTTOM=0.29; LOC_HEIGHT=0.26
    LOC_L_X=0.01; LOC_R_X=0.24; LOC_W=0.22

    # Fixed zone bounds — same size for every pitcher, every card
    def draw_zone(ax, hand):
        ax.set_facecolor('#1a1d21')
        ax.set_xlim(-1.9, 1.9); ax.set_ylim(0.5, 4.2)
        ax.add_patch(Rectangle((-PLATE_HALF, avg_bot), PLATE_HALF*2, avg_top-avg_bot, fill=False, edgecolor='#888', linewidth=1.5, zorder=2))
        tw = PLATE_HALF*2/3; th = (avg_top-avg_bot)/3
        for i in range(1,3):
            ax.plot([-PLATE_HALF+i*tw, -PLATE_HALF+i*tw], [avg_bot, avg_top], color='#444', linewidth=0.5, zorder=2)
            ax.plot([-PLATE_HALF, PLATE_HALF], [avg_bot+i*th, avg_bot+i*th], color='#444', linewidth=0.5, zorder=2)
        pt_y = avg_bot - 0.15
        ax.plot([-PLATE_HALF,-PLATE_HALF,0,PLATE_HALF,PLATE_HALF,-PLATE_HALF], [pt_y,pt_y-0.10,pt_y-0.20,pt_y-0.10,pt_y,pt_y], color='#888', linewidth=1.2, zorder=2)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values(): spine.set_color('#444')
        for pt in PITCH_ORDER:
            if pt not in locations[hand]: continue
            color = PITCH_COLORS[pt]
            for px_val, pz_val, desc, barrel_flag in locations[hand][pt]:
                if desc == 'Swinging Strike':
                    ax.text(px_val, pz_val, 'W', fontsize=8, fontweight='bold', color=color, ha='center', va='center', zorder=3)
                elif barrel_flag:
                    ax.text(px_val, pz_val, 'B', fontsize=8, fontweight='bold', color=color, ha='center', va='center', zorder=3)
                else:
                    ax.scatter([px_val], [pz_val], c=color, s=55, alpha=1.0, edgecolors='none', zorder=3)

    ax_loc_l = fig.add_axes([LOC_L_X, LOC_BOTTOM, LOC_W, LOC_HEIGHT])
    ax_loc_r = fig.add_axes([LOC_R_X, LOC_BOTTOM, LOC_W, LOC_HEIGHT])
    draw_zone(ax_loc_l, 'R'); draw_zone(ax_loc_r, 'L')
    fig.text(LOC_L_X+LOC_W/2, LOC_TITLE_Y, 'VS RHH', fontsize=14, fontweight='bold', color='white', fontfamily='DIN Condensed', ha='center', va='center')
    fig.text(LOC_R_X+LOC_W/2, LOC_TITLE_Y, 'VS LHH', fontsize=14, fontweight='bold', color='white', fontfamily='DIN Condensed', ha='center', va='center')

    # W/B legend below pitch location plots
    fig.text(LOC_R_X + LOC_W + 0.005, LOC_BOTTOM + 0.015, 'W = Whiff',
        fontsize=7, color='#ccc', va='bottom', ha='left', fontfamily='Avenir Next')
    fig.text(LOC_R_X + LOC_W + 0.005, LOC_BOTTOM, 'B = Barrel',
        fontsize=7, color='#ccc', va='bottom', ha='left', fontfamily='Avenir Next')

    # ── Batted ball distribution (donut + stacked bars) ──
    BB_CHART_BOTTOM = 0.58
    BB_CHART_HEIGHT = 0.17

    overall_bb = {bb: sum(bb_by_pitch[pt][bb] for pt in bb_by_pitch) for bb in BB_TYPES}
    total_bip = sum(overall_bb.values())

    # Donut (left side) — top aligned with top of first badge row
    _n_bb_tmp = max(len([pt for pt in sorted_types if pt in bb_by_pitch and
        sum(bb_by_pitch[pt][bb] for bb in BB_TYPES) > 0]), 1)
    _rh_bb_tmp = min(0.12, 0.85 / _n_bb_tmp)
    _badge_top_axes = 0.90 + _rh_bb_tmp * 0.4
    _badge_top_fig = BB_CHART_BOTTOM + BB_CHART_HEIGHT * _badge_top_axes
    _donut_bottom = _badge_top_fig - BB_CHART_HEIGHT + 0.02
    ax_bb_donut = fig.add_axes([0.01, _donut_bottom, 0.16, BB_CHART_HEIGHT])
    ax_bb_donut.set_facecolor(BG)

    if total_bip > 0:
        bb_vals = [overall_bb[bb] for bb in BB_TYPES]
        bb_cols = [BB_COLORS[bb] for bb in BB_TYPES]
        bb_wedges, _ = ax_bb_donut.pie(
            bb_vals, colors=bb_cols, startangle=90,
            wedgeprops=dict(width=0.30, edgecolor=BG, linewidth=2.0),
            counterclock=False
        )
        ax_bb_donut.text(0, 0, f'{total_bip}\nBIP', ha='center', va='center',
                         fontsize=11, fontweight='bold', color='#e8e8e8', linespacing=1.15)
        angle = 90
        for bb, val in zip(BB_TYPES, bb_vals):
            if val == 0:
                continue
            ang_span = val / total_bip * 360
            mid_angle = angle - ang_span / 2
            mid_rad = np.radians(mid_angle)
            r = 0.85
            x = r * np.cos(mid_rad)
            y_pos = r * np.sin(mid_rad)
            tc_w = badge_text_color(BB_COLORS[bb])
            ax_bb_donut.text(x, y_pos, str(val), ha='center', va='center',
                             fontsize=7, fontweight='bold', color=tc_w)
            angle -= ang_span

    # Stacked bars (right of donut)
    ax_bb_bars = fig.add_axes([0.17, BB_CHART_BOTTOM, 0.30, BB_CHART_HEIGHT])
    ax_bb_bars.set_xlim(0, 1)
    ax_bb_bars.set_ylim(0, 1)
    ax_bb_bars.set_xticks([])
    ax_bb_bars.set_yticks([])
    ax_bb_bars.axis('off')
    ax_bb_bars.set_facecolor(BG)

    bb_pitch_order = sorted(
        [pt for pt in sorted_types if pt in bb_by_pitch and
         sum(bb_by_pitch[pt][bb] for bb in BB_TYPES) > 0],
        key=lambda pt: sum(bb_by_pitch[pt][bb] for bb in BB_TYPES),
        reverse=True
    )

    if bb_pitch_order:
        n_bb = len(bb_pitch_order)
        # Reserve space for legend (2 rows of labels) at bottom
        legend_reserve = 0.18
        available = 0.90 - legend_reserve  # usable vertical space for bars
        gap_bb = 0.012
        rh_bb = min(0.12, (available - (n_bb - 1) * gap_bb) / n_bb)

        for i, pt in enumerate(bb_pitch_order):
            y = 0.90 - i * (rh_bb + gap_bb)
            color = PITCH_COLORS.get(pt, '#999')
            tc_b = badge_text_color(color)
            total_pt = sum(bb_by_pitch[pt][bb] for bb in BB_TYPES)
            brl = bb_by_pitch[pt]['brl']

            # Badge
            ax_bb_bars.add_patch(FancyBboxPatch(
                (0.02, y - rh_bb * 0.4), 0.08, rh_bb * 0.8,
                boxstyle="round,pad=0.008", facecolor=color, edgecolor='none'))
            ax_bb_bars.text(0.06, y, pt, fontsize=8, ha='center', va='center',
                            color=tc_b, fontweight='bold')

            # Gray track
            ax_bb_bars.add_patch(Rectangle(
                (0.13, y - rh_bb * 0.275), 0.48, rh_bb * 0.55,
                facecolor='#333840', edgecolor='none'))

            # Stacked segments with counts
            left = 0.13
            for bb in BB_TYPES:
                cnt = bb_by_pitch[pt][bb]
                pct = cnt / total_pt if total_pt > 0 else 0
                if pct > 0:
                    seg_w = 0.48 * pct
                    ax_bb_bars.add_patch(Rectangle(
                        (left, y - rh_bb * 0.275), seg_w, rh_bb * 0.55,
                        facecolor=BB_COLORS[bb], edgecolor=BG, linewidth=0.5))
                    seg_tc = badge_text_color(BB_COLORS[bb])
                    ax_bb_bars.text(left + seg_w / 2, y, str(cnt),
                                    ha='center', va='center', fontsize=7,
                                    color=seg_tc, fontweight='bold')
                    left += seg_w

            # Count label with barrel info
            label = str(total_pt)
            if brl > 0:
                label += f'  ({brl} Brl)'
            ax_bb_bars.text(0.63, y, label, fontsize=8, va='center', ha='left',
                            color='#e8e8e8', fontweight='bold')

        # Legend below the last bar
        last_bar_y = 0.90 - (n_bb - 1) * (rh_bb + gap_bb)
        legend_y = last_bar_y - rh_bb * 0.5 - 0.04
        bb_legend_patches = [mpatches.Patch(color=BB_COLORS[bb], label=f'{BB_LABELS[bb]} ({overall_bb[bb]})')
                             for bb in BB_TYPES if overall_bb[bb] > 0]
        ax_bb_bars.legend(handles=bb_legend_patches, loc='upper left',
                          bbox_to_anchor=(0.02, legend_y), ncol=2, fontsize=7,
                          frameon=False, labelcolor='#bbb', handlelength=1.0,
                          columnspacing=0.8)

    # Pitch usage
    usage = {'L': defaultdict(int), 'R': defaultdict(int)}
    tots = {'L': 0, 'R': 0}
    for p in pitches:
        bh, pt = p.get('Bats',''), p.get('Pitch Type','')
        if bh in ('L','R') and pt: usage[bh][pt] += 1; tots[bh] += 1

    def draw_usage(ax, data, total, title):
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off'); ax.set_facecolor(BG)
        S = USAGE_SHIFT; content_center = (0.04+S+0.80+S)/2
        ax.text(content_center, 0.96, title, fontsize=13, fontweight='bold', ha='center', va='top', color='white', fontfamily='DIN Condensed')
        spts = sorted(data.keys(), key=lambda x: (-data[x], PITCH_ORDER.index(x) if x in PITCH_ORDER else 999)); n = len(spts)
        if n == 0: return
        rh = min(0.10, 0.78/n); gap = 0.012
        for i, pt in enumerate(spts):
            y = 0.84 - i*(rh+gap); pct = data[pt]/total if total > 0 else 0
            color = PITCH_COLORS.get(pt, '#999'); tc = badge_text_color(color)
            ax.add_patch(FancyBboxPatch((0.04+S, y-rh*0.4), 0.08, rh*0.8, boxstyle="round,pad=0.008", facecolor=color, edgecolor='none'))
            ax.text(0.08+S, y, pt, fontsize=8, ha='center', va='center', color=tc, fontweight='bold')
            ax.add_patch(Rectangle((0.15+S, y-rh*0.275), 0.55, rh*0.55, facecolor='#333840', edgecolor='none'))
            if pct > 0: ax.add_patch(Rectangle((0.15+S, y-rh*0.275), 0.55*pct, rh*0.55, facecolor=color, edgecolor='none'))
            ax.text(0.74+S, y, f'{pct*100:.1f}%', fontsize=10, va='center', ha='left', color='white', fontweight='bold', fontfamily='Avenir Next')

    usage_left = 0.50; usage_mid = (usage_left + TABLE_RIGHT_FIG) / 2
    ax_ul = fig.add_axes([usage_left, 0.32, usage_mid-usage_left, 0.17])
    ax_ur = fig.add_axes([usage_mid, 0.32, TABLE_RIGHT_FIG-usage_mid, 0.17])
    draw_usage(ax_ul, usage['R'], tots['R'], 'VS RHH')
    draw_usage(ax_ur, usage['L'], tots['L'], 'VS LHH')

    # Metrics table
    ax_table = fig.add_axes([TABLE_LEFT_FIG, 0.01, TABLE_RIGHT_FIG-TABLE_LEFT_FIG, 0.27])
    ax_table.axis('off'); ax_table.set_facecolor(BG)

    tc = len(pitches)
    pitch_stats = []

    # Sort pitch types by usage (descending), with PITCH_ORDER as tiebreaker
    pitch_counts = {}
    for p in pitches:
        pt = p.get('Pitch Type', '')
        if pt:
            pitch_counts[pt] = pitch_counts.get(pt, 0) + 1
    table_pitch_order = sorted(
        pitch_counts.keys(),
        key=lambda pt: (-pitch_counts[pt], PITCH_ORDER.index(pt) if pt in PITCH_ORDER else 999)
    )

    for pt in table_pitch_order:
        pp = [p for p in pitches if p.get('Pitch Type') == pt]
        if not pp: continue
        n = len(pp)
        velos=[v for v in (sf(p.get('Velocity')) for p in pp) if v]
        spins=[v for v in (sf(p.get('Spin Rate')) for p in pp) if v]
        ivbs=[v for v in (sf(p.get('IndVertBrk')) for p in pp) if v is not None]
        hbs=[v for v in (sf(p.get('HorzBrk')) for p in pp) if v is not None]
        relzs=[v for v in (sf(p.get('RelPosZ')) for p in pp) if v is not None]
        relxs=[v for v in (sf(p.get('RelPosX')) for p in pp) if v is not None]
        exts=[v for v in (sf(p.get('Extension')) for p in pp) if v is not None]
        armangles=[v for v in (sf(p.get('ArmAngle')) for p in pp) if v is not None]
        swings=[p for p in pp if p.get('Description') in SWING_DESC]
        whiffs=[p for p in pp if p.get('Description')=='Swinging Strike']
        strikes=[p for p in pp if p.get('Description') in STRIKE_DESC]
        iz_n=0
        for p in pp:
            r = compute_iz(p)
            if r is None: continue
            if r: iz_n+=1
        rvs=[v for v in (sf(p.get('RunExp')) for p in pp) if v is not None]
        rv=round(sum(rvs),1) if rvs else None
        pt_name='Fastball' if pt=='FF' else PITCH_NAMES.get(pt,pt)
        row=[pt_name,str(n),f"{n/tc*100:.1f}%",
            f"{sum(velos)/len(velos):.1f}" if velos else '—',f"{max(velos):.1f}" if velos else '—',
            f"{int(sum(spins)/len(spins))}" if spins else '—',
            f'{sum(ivbs)/len(ivbs):.1f}"' if ivbs else '—',f'{sum(hbs)/len(hbs):.1f}"' if hbs else '—',
            fmt_fi(sum(relzs)/len(relzs)) if relzs else '—',fmt_fi(sum(relxs)/len(relxs)) if relxs else '—',
            fmt_fi(sum(exts)/len(exts)) if exts else '—',
            f"{sum(armangles)/len(armangles):.1f}°" if armangles else '—',
            f"{len(strikes)/n*100:.1f}%",f"{iz_n/n*100:.1f}%" if n else '—',
            f"{len(whiffs)/len(swings)*100:.1f}%" if swings else '—',
            str(rv) if rv is not None else '—']
        pitch_stats.append((pt, row))

    t_str=[p for p in pitches if p.get('Description') in STRIKE_DESC]
    t_sw=[p for p in pitches if p.get('Description') in SWING_DESC]
    t_wh=[p for p in pitches if p.get('Description')=='Swinging Strike']
    t_iz=sum(1 for p in pitches if compute_iz(p)==True)
    t_rv=[v for v in (sf(p.get('RunExp')) for p in pitches) if v is not None]
    total_row=['Total',str(tc),'100.0%','—','—','—','—','—','—','—','—','—',
        f"{len(t_str)/tc*100:.1f}%" if tc else '—',f"{t_iz/tc*100:.1f}%" if tc else '—',
        f"{len(t_wh)/len(t_sw)*100:.1f}%" if t_sw else '—',
        str(round(sum(t_rv),1)) if t_rv else '—']

    # Check if certain source columns are populated at all in the raw pitch data
    has_rv_data = any(p.get('RunExp') is not None and str(p.get('RunExp','')).strip() != '' for p in pitches)

    all_col_headers=['Pitch Type','Count','Usage','Avg Velo','Max Velo','Spin Rate','IVB','HB','RelZ','RelX','Ext','ArmAngle','Strike%','Zone%','Whiff%','PitchRV']
    all_cell_data=[r[1] for r in pitch_stats]+[total_row]

    # Columns to force-exclude if source data doesn't exist yet
    force_exclude = set()
    if not has_rv_data: force_exclude.add('PitchRV')

    # Drop columns where ALL pitch-type rows have '—' OR source data is missing
    pitch_rows_only = [r[1] for r in pitch_stats]  # exclude total row
    cols_to_keep = []
    for ci in range(len(all_col_headers)):
        col_name = all_col_headers[ci]
        # Always keep Pitch Type, Count, Usage
        if ci < 3:
            cols_to_keep.append(ci)
        elif col_name in force_exclude:
            continue  # skip — source data not available
        else:
            # Keep if at least one pitch-type row has a real value (not '—')
            if any(row[ci] != '—' for row in pitch_rows_only):
                cols_to_keep.append(ci)

    col_headers = [all_col_headers[i] for i in cols_to_keep]
    cell_data = [[row[i] for i in cols_to_keep] for row in all_cell_data]
    pt_codes=[r[0] for r in pitch_stats]+[None]

    # Recalculate divider column index (Strike% position in filtered columns)
    divider_col = col_headers.index('Strike%') if 'Strike%' in col_headers else None

    table = ax_table.table(cellText=cell_data, colLabels=col_headers, loc='upper center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.6)

    for (r,c), cell in table.get_celld().items():
        cell.set_edgecolor('#333840'); cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(DARKER); cell.set_text_props(color=ACCENT, fontweight='bold', fontsize=10)
        elif r == len(cell_data):
            cell.set_facecolor(DARKER); cell.set_text_props(fontweight='bold', color='white')
        else:
            bg = DARK_CELL if r%2==1 else '#252930'
            cell.set_facecolor(bg); cell.set_text_props(color='#e8e8e8', fontweight='bold')
        if c == 0 and r > 0:
            pc = pt_codes[r-1]
            if pc:
                cell.set_facecolor(PITCH_COLORS.get(pc,'#999'))
                cell.set_text_props(color=badge_text_color(PITCH_COLORS.get(pc,'#999')), fontweight='bold')

    # Percentile-based coloring for Strike%, Zone%, Whiff%
    league_avgs = config.get('league_avgs', {})
    overall_avgs = config.get('overall_avgs', {})
    for c, col_name in enumerate(col_headers):
        meta_key = PCT_COLOR_COLS.get(col_name)
        if not meta_key:
            continue
        for r in range(1, len(cell_data) + 1):
            if r == len(cell_data):
                # Total row — use overall league averages
                la = overall_avgs.get(meta_key)
                row_bg = DARKER
            else:
                pc = pt_codes[r - 1]
                if not pc or pc not in league_avgs:
                    continue
                la = league_avgs[pc].get(meta_key)
                row_bg = DARK_CELL if r % 2 == 1 else '#252930'
            if la is None:
                continue
            val_str = cell_data[r - 1][c]
            tinted = pct_cell_color(val_str, la, row_bg)
            if tinted:
                table.get_celld()[(r, c)].set_facecolor(tinted)

    # Divider + borders
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer(); fig_bbox = fig.get_window_extent(renderer)
    if divider_col is not None:
        strike_cell = table.get_celld()[0, divider_col]
        x_fig = strike_cell.get_window_extent(renderer).x0 / fig_bbox.width
        top_y = strike_cell.get_window_extent(renderer).y1 / fig_bbox.height - 0.001
        bot_cell = table.get_celld()[len(cell_data), divider_col]
        bot_y = bot_cell.get_window_extent(renderer).y0 / fig_bbox.height
        fig.add_artist(plt.Line2D([x_fig,x_fig], [bot_y,top_y], transform=fig.transFigure, color=ACCENT, linewidth=2, zorder=10))

    tl = table.get_celld()[0,0].get_window_extent(renderer)
    br = table.get_celld()[len(cell_data), len(col_headers)-1].get_window_extent(renderer)
    l = tl.x0/fig_bbox.width; r_ = br.x1/fig_bbox.width
    t = tl.y1/fig_bbox.height - 0.001; b = br.y0/fig_bbox.height
    for x1,y1,x2,y2 in [(l,b,r_,b),(l,t,r_,t),(l,b,l,t),(r_,b,r_,t)]:
        fig.add_artist(plt.Line2D([x1,x2],[y1,y2], transform=fig.transFigure, color=ACCENT, linewidth=2, zorder=10))

    # Place watermark just below the metrics table border (measured after draw)
    fig.text(0.99, b - 0.008, 'Huronalytics', fontsize=9, ha='right', va='top', color='#555', style='italic', fontfamily='DIN Condensed')
    plt.savefig(output_file, dpi=SAVE_DPI, bbox_inches='tight', facecolor=BG, pad_inches=0.1)
    plt.close()

    # Crop bottom dead space from saved PNG
    card_img = Image.open(output_file)
    pixels = np.array(card_img)
    bg_rgb = (20, 22, 25)  # BG=#141619
    # Scan from bottom up to find last non-background row
    for y in range(pixels.shape[0]-1, 0, -1):
        row = pixels[y, :, :3]
        if not np.all(np.abs(row.astype(int) - np.array(bg_rgb)) < 10):
            # Found content — add small padding below
            crop_y = min(y + 30, pixels.shape[0])
            card_img = card_img.crop((0, 0, card_img.width, crop_y))
            card_img.save(output_file)
            break
    return True


# ═══════════════════════════════════════════════════════════════
# MAIN BATCH LOGIC
# ═══════════════════════════════════════════════════════════════
def main():
    # ── Settings ──────────────────────────────────────────────
    team            = "WSH"             # team abbreviation (e.g., "WSH", "NYY")
    game_date       = "2026-03-30"      # YYYY-MM-DD
    filter_pitchers = None              # None = all pitchers, or list: ["Corbin, Patrick", "Williams, Trevor"]
    game_pk         = ""                # optional game PK (e.g., "823484") — use for live/in-progress games
    # ──────────────────────────────────────────────────────────

    # Determine league
    if team in AL_TEAMS:
        league = 'AL'
    elif team in NL_TEAMS:
        league = 'NL'
    else:
        print(f"Error: Unknown team '{team}'")
        sys.exit(1)

    sheet_key = _workbook_id_for_team(team)
    date_str = game_date

    # Format date for display
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    display_date = date_obj.strftime('%B %d, %Y').replace(' 0', ' ')  # "March 26, 2026"
    date_short = date_obj.strftime('%m-%d')  # "03-26"

    if filter_pitchers:
        print(f"═══ Generating cards for {', '.join(filter_pitchers)} ({team}) on {date_str} ({league}) ═══\n")
    else:
        print(f"═══ Generating cards for {team} on {date_str} ({league}) ═══\n")

    # Load league averages for percentile coloring
    league_avgs = {}
    overall_avgs = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        league_avgs = meta.get('leagueAverages', {})
        overall_avgs = meta.get('pitcherLeagueAverages', {})

    # Step 1: Load pitch data from Google Sheets
    print("Step 1: Loading pitch data from Google Sheets...")
    gc = gspread.service_account()
    sh = gc.open_by_key(sheet_key)
    ws = sh.worksheet(team)
    all_rows = ws.get_all_records()

    # Filter by date (and optionally by pitcher name)
    pitches_by_pitcher = defaultdict(list)
    for row in all_rows:
        if row.get('Game Date') == date_str:
            pitcher_name = row.get('Pitcher', '')
            if pitcher_name:
                if filter_pitchers and pitcher_name not in filter_pitchers:
                    continue
                pitches_by_pitcher[pitcher_name].append(row)

    pitcher_names = sorted(pitches_by_pitcher.keys())
    print(f"  Found {len(pitcher_names)} pitchers: {', '.join(pitcher_names)}")

    if not pitcher_names:
        print(f"  No pitch data found for {team} on {date_str}")
        if filter_pitchers:
            print(f"  (filter_pitchers was set to: {filter_pitchers})")
        sys.exit(0)

    # Step 2: Fetch boxscore stats
    print("\nStep 2: Fetching boxscore stats from MLB API...")
    box_stats = fetch_boxscores_for_team(date_str, team, include_live=bool(game_pk), game_pk=game_pk)
    print(f"  Found boxscore data for: {', '.join(box_stats.keys())}")

    # Step 3: Look up MLB IDs and metadata
    print("\nStep 3: Looking up MLB player IDs...")
    mlb_cache = load_mlb_id_cache()

    # Step 4: Generate cards
    print("\nStep 4: Generating cards...")
    generated = []

    for pitcher_name in pitcher_names:
        pitches = pitches_by_pitcher[pitcher_name]
        print(f"\n  --- {pitcher_name} ({len(pitches)} pitches) ---")

        # Get hand from pitch data
        hand = pitches[0].get('Throws', 'R') if pitches else 'R'

        # Look up MLB ID
        mlb_id = lookup_mlb_id(pitcher_name, team, mlb_cache)
        print(f"  MLB ID: {mlb_id}")

        # Get age from MLB API
        meta = fetch_player_metadata(mlb_id)
        age = meta['age']
        # Use hand from sheet data (more reliable for current game)
        if not hand: hand = meta['hand']

        # Get boxscore stats
        box = box_stats.get(pitcher_name)
        if not box:
            print(f"  WARNING: No boxscore data found for {pitcher_name}, skipping")
            continue

        ip_str = outs_to_ip_str(box['outs'])
        pitch_count = len(pitches_by_pitcher[pitcher_name])
        stat_headers = ['IP', 'P', 'TBF', 'R', 'ER', 'H', 'K', 'BB', 'HR']
        stat_values = [ip_str, str(pitch_count), str(box['tbf']), str(box['r']), str(box['er']),
                       str(box['h']), str(box['so']), str(box['bb']), str(box['hr'])]

        print(f"  Stat line: {' | '.join(f'{h}:{v}' for h,v in zip(stat_headers, stat_values))}")

        # Fetch headshot
        headshot = fetch_headshot(mlb_id)

        # Format display name
        parts = pitcher_name.split(', ')
        if len(parts) == 2:
            display_name = f"{parts[1]} {parts[0]}".upper()
            last_name = parts[0]
        else:
            display_name = pitcher_name.upper()
            last_name = pitcher_name

        # Build config
        config = {
            'display_name': display_name,
            'hand': hand,
            'team': team,
            'age': age,
            'game_date': display_date,
            'stat_headers': stat_headers,
            'stat_values': stat_values,
            'headshot': headshot,
            'mlb_id': mlb_id,
            'league_avgs': league_avgs,
            'overall_avgs': overall_avgs,
        }

        # Output file — MMDDYYYY-LastFirst format
        date_mmddyyyy = date_obj.strftime('%m%d%Y')
        # Build LastFirst from pitcher_name ("Last, First" -> "LastFirst")
        if len(parts) == 2:
            name_slug = f"{parts[0]}{parts[1]}".replace(' ', '')
        else:
            name_slug = pitcher_name.replace(' ', '').replace(',', '')
        output_file = os.path.join(OUTPUT_DIR, f"{date_mmddyyyy}-{name_slug}.png")

        # Render
        success = render_card(config, pitches, output_file)
        if success:
            print(f"  ✅ Saved: {output_file}")
            generated.append(output_file)
        else:
            print(f"  ❌ Failed to render")

    # Save MLB ID cache
    save_mlb_id_cache(mlb_cache)

    # Summary
    print(f"\n{'═'*60}")
    print(f"Generated {len(generated)} cards for {team} on {date_str}:")
    for f in generated:
        print(f"  {os.path.basename(f)}")
    print(f"{'═'*60}")


if __name__ == '__main__':
    main()
