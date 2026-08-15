"""pull_statcast_2025_full.py — re-pull public Statcast 2025 with the FULL
column set (the original _statcast2025_cache.pkl kept only a narrow slice;
the Pitching+ joint-model experiment needs balls/strikes/sz_top/sz_bot).
Writes data/_statcast2025_full_cache.pkl. Same KEEP as pull_statcast_multiyear.
"""
import os, pickle, warnings
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys_keep = None
from pull_statcast_multiyear import KEEP
OUT = os.path.join(ROOT, 'data', '_statcast2025_full_cache.pkl')

def main():
    if os.path.exists(OUT):
        print(f'cached, skipping ({os.path.getsize(OUT)/1e6:.0f} MB)')
        return
    from pybaseball import statcast
    print('downloading 2025-03-18..2025-09-28 ...', flush=True)
    df = statcast(start_dt='2025-03-18', end_dt='2025-09-28', verbose=False)
    keep = [c for c in KEEP if c in df.columns]
    df = df[keep]
    if 'game_type' in df.columns:
        df = df[df['game_type'] == 'R']
    df['game_date'] = df['game_date'].astype(str).str[:10]
    pickle.dump(df, open(OUT, 'wb'))
    print(f'cached {len(df)} pitches -> {OUT}', flush=True)

if __name__ == '__main__':
    main()
