"""Build a compact parquet cache of the fields the expected-movement (xIVB/xHB)
audit needs, from data/_pitches{YEAR}_training.pkl.

Writes one parquet per season to the scratch dir. Run once; every other
xmove_* script reads the parquet.
"""
import pickle, sys, os
import numpy as np
import pandas as pd

OUT = os.environ.get('XMOVE_DIR', '/private/tmp/claude-501/-Users-wallyhuron-Huronalytics/8aed4efe-0775-4afc-b652-6ddab7be7e7d/scratchpad')
COLS = ['Game Date', 'PTeam', 'Pitcher', 'Throws', 'Bats', 'Pitch Type', 'Velocity',
        'Spin Rate', 'SpinAxis', 'IndVertBrk', 'HorzBrk', 'xIndVrtBrk', 'xHorzBrk',
        'RelPosZ', 'RelPosX', 'Extension', 'ArmAngle', 'PlateZ', 'PlateX',
        'VAA', '_game_pk', '_source']

def build(year):
    src = f'/Users/wallyhuron/Huronalytics/data/_pitches{year}_training.pkl'
    with open(src, 'rb') as f:
        rows = pickle.load(f)
    df = pd.DataFrame([{c: r.get(c) for c in COLS} for r in rows])
    for c in ['Velocity', 'Spin Rate', 'SpinAxis', 'IndVertBrk', 'HorzBrk',
              'xIndVrtBrk', 'xHorzBrk', 'RelPosZ', 'RelPosX', 'Extension',
              'ArmAngle', 'PlateZ', 'PlateX', 'VAA']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['season'] = year
    path = f'{OUT}/xmove_{year}.parquet'
    df.to_parquet(path, index=False)
    print(f'{year}: {len(df):,} rows -> {path}')
    return df

if __name__ == '__main__':
    years = [int(a) for a in sys.argv[1:]] or [2021, 2022, 2023, 2024, 2025]
    for y in years:
        build(y)
