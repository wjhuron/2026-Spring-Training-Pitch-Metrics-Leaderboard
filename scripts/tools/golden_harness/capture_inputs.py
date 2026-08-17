#!/usr/bin/env python3
"""Capture frozen pipeline inputs for golden-output testing.

Reads the six division workbooks + NEW tab ONCE and pickles the raw rows.
Later golden runs monkeypatch the fetch layer to replay these rows, so a
code change can be diffed against a byte-stable baseline without network.
"""
import os
import pickle
import sys

REPO = '/Users/wallyhuron/Huronalytics'
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from pipeline.fetch import read_all_pitches_from_sheets, read_new_tab_pitches

rs = read_all_pitches_from_sheets()
print(f"captured {len(rs)} RS rows")
with open(os.path.join(OUT, 'golden_input_rs.pkl'), 'wb') as f:
    pickle.dump(rs, f)

try:
    new_tab = read_new_tab_pitches()
except Exception as e:  # NEW tab is best-effort in main() too
    print(f"NEW tab read failed ({e}); freezing empty list")
    new_tab = []
print(f"captured {len(new_tab)} NEW-tab rows")
with open(os.path.join(OUT, 'golden_input_new.pkl'), 'wb') as f:
    pickle.dump(new_tab, f)

print("capture complete")
