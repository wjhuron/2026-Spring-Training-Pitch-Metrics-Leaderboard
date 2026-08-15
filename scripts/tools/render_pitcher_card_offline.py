#!/usr/bin/env python3
"""Offline pitcher-card render harness — iterate on Cards.py without the
Google Sheets / boxscore network path.

Pulls a pitcher's pitches straight from the season pickle
(data/all_pitches_rs_cache.pkl) and their stats + percentiles from
data/pitcher_leaderboard_rs.json, builds the render_card config, and
renders. Use during the pitcher-card redesign to render Foster Griffin
(or any pitcher) in ~2s.

    python3 scripts/tools/render_pitcher_card_offline.py "Griffin, Foster" WSH
"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cards.pitcher as Cards
from PIL import Image

PICKLE = '/Users/wallyhuron/Huronalytics/data/all_pitches_rs_cache.pkl'
LEADERBOARD = '/Users/wallyhuron/Huronalytics/data/pitcher_leaderboard_rs.json'
METADATA = '/Users/wallyhuron/Huronalytics/data/metadata_rs.json'


def fmt_pct(v):
    return f'{v*100:.1f}%' if v is not None else '—'


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'Griffin, Foster'
    team = sys.argv[2] if len(sys.argv) > 2 else 'WSH'
    out = sys.argv[3] if len(sys.argv) > 3 else f'/Users/wallyhuron/Downloads/_pitchercard_test.png'

    print(f'Loading pickle …')
    with open(PICKLE, 'rb') as f:
        allp = pickle.load(f)
    pitches = [p for p in allp if p.get('Pitcher') == name and p.get('PTeam') == team]
    print(f'  {len(pitches)} pitches for {name} ({team})')
    if not pitches:
        print('  no pitches — aborting'); return

    with open(LEADERBOARD) as f:
        lb = json.load(f)
    rows = [r for r in lb if r.get('pitcher') == name and r.get('team') == team]
    if not rows:
        rows = [r for r in lb if r.get('pitcher') == name]
    p_row = rows[0] if rows else {}
    print(f'  leaderboard row: {"FOUND" if p_row else "MISSING"}  '
          f'(era={p_row.get("era")}, siera={p_row.get("siera")})')

    # Per-pitch-type Loc+ for the table.
    PITCH_LB = '/Users/wallyhuron/Huronalytics/data/pitch_leaderboard_rs.json'
    pitch_locplus = {}
    with open(PITCH_LB) as f:
        for r in json.load(f):
            if r.get('pitcher') == name and r.get('team') == team and r.get('locPlus') is not None:
                pitch_locplus[r.get('pitchType')] = r['locPlus']
    print(f'  per-pitch Loc+: {pitch_locplus}')

    with open(METADATA) as f:
        meta = json.load(f)
    league_avgs = meta.get('leagueAverages', {})
    overall_avgs = meta.get('pitcherLeagueAverages', {})

    mvn_models = Cards.load_mvn_models()

    mlb_id = p_row.get('mlbId')
    try:
        headshot = Cards.fetch_headshot(mlb_id) if mlb_id else None
    except Exception:
        headshot = None
    if headshot is None:
        headshot = Image.new('RGB', (180, 270), (90, 90, 90))

    # Season statline = context + the two non-bubble rate stats (ERA/SIERA).
    stat_headers = ['G', 'GS', 'IP', 'ERA', 'SIERA']
    stat_values = [
        str(p_row.get('g', '—')),
        str(p_row.get('gs', 0)),
        str(p_row.get('ip', '—')),
        f"{p_row.get('era'):.2f}" if p_row.get('era') is not None else '—',
        f"{p_row.get('siera'):.2f}" if p_row.get('siera') is not None else '—',
    ]

    parts = name.split(', ')
    display_name = f'{parts[1]} {parts[0]}'.upper() if len(parts) == 2 else name.upper()

    # Season label with a "Through <date>" freshness stamp (matches the hitter
    # card). Latest game date comes from the pitches.
    import datetime as _dt
    _dates = sorted(str(p.get('Game Date')) for p in pitches if p.get('Game Date'))
    game_date = '2026 Season'
    if _dates:
        try:
            _ld = _dt.datetime.strptime(_dates[-1], '%Y-%m-%d')
            game_date = f"2026 Season  ·  Through {_ld.strftime('%b %d').replace(' 0', ' ')}"
        except Exception:
            pass

    config = {
        'display_name': display_name,
        'hand': (pitches[0].get('Throws') or p_row.get('throws') or 'R'),
        'team': team,
        'age': p_row.get('age', '—'),
        'game_date': game_date,
        'stat_headers': stat_headers,
        'stat_values': stat_values,
        'headshot': headshot,
        'mlb_id': mlb_id,
        'league_avgs': league_avgs,
        'overall_avgs': overall_avgs,
        'pitcher_league_avgs': overall_avgs,
        'mvn_models': mvn_models,
        'pctl_row': p_row,   # NEW: source of all _pctl values for bubbles
        'pitch_locplus': pitch_locplus,   # per-pitch-type Loc+ for the table
    }

    print('Rendering …')
    Cards.render_card(config, pitches, out)
    print(f'  wrote {out}')


if __name__ == '__main__':
    main()
