"""pull_statcast_multiyear.py — cache full public Statcast for 2021-2024.

Feeds the multi-season Stuff+ expansion (2021-24 join the training set alongside
the retagged 2025 + OOF 2026). No retagged sheets exist for these years, so we
reconstruct the feature set from PUBLIC Statcast — fine because Stuff+ is pitch-
type agnostic (public tags feed only the fastball anchor + display grouping).

Keeps the full physics column set (superset of what the 2025 build kept, since
those years have no sheet to join). arm_angle exists only from ~mid-2024;
bat_speed/swing_length from 2024 — [c for c in keep if in df] drops them silently
for earlier years. Caches one pickle per season (resumable: skips cached years).

Usage: python3 scripts/builders/pull_statcast_multiyear.py
"""
import os, pickle, warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'data')

# generous regular-season windows; game_type=='R' filter drops spring/postseason
SEASONS = {
    2021: ('2021-04-01', '2021-10-03'),
    2022: ('2022-04-07', '2022-10-05'),
    2023: ('2023-03-30', '2023-10-01'),
    2024: ('2024-03-20', '2024-09-30'),
}

KEEP = ['game_date', 'game_pk', 'player_name', 'pitcher', 'batter', 'p_throws',
        'stand', 'pitch_type', 'release_speed', 'release_pos_x', 'release_pos_z',
        'release_extension', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z',
        'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'arm_angle',
        'description', 'events', 'type', 'bb_type', 'launch_speed', 'launch_angle',
        'bat_speed', 'swing_length', 'estimated_woba_using_speedangle',
        'delta_run_exp', 'delta_pitcher_run_exp', 'game_type', 'balls', 'strikes',
        'outs_when_up', 'sz_top', 'sz_bot']


def main():
    from pybaseball import statcast
    for year, (start, end) in SEASONS.items():
        out = os.path.join(DATA, f'_statcast{year}_cache.pkl')
        if os.path.exists(out):
            print(f'{year}: cached, skipping ({os.path.getsize(out)/1e6:.0f} MB)', flush=True)
            continue
        print(f'{year}: downloading {start}..{end} ...', flush=True)
        df = statcast(start_dt=start, end_dt=end, verbose=False)
        keep = [c for c in KEEP if c in df.columns]
        df = df[keep]
        if 'game_type' in df.columns:
            df = df[df['game_type'] == 'R']
        df['game_date'] = df['game_date'].astype(str).str[:10]
        pickle.dump(df, open(out, 'wb'))
        miss = [c for c in KEEP if c not in df.columns]
        print(f'{year}: cached {len(df)} pitches -> {out}  (missing cols: {miss})', flush=True)
    print('done.')


if __name__ == '__main__':
    main()
