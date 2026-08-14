"""locplus_groupbw_extend2.py — round 3 of the per-group x-bandwidth search.
Round 2 was still monotone at the 8.0 edge for FF/SI/CU (pred_g and rel_g
both rising). Extends to {8.0, 10.0, 14.0}; 14 inches approaches the
vertical-only limit for these groups.

Usage: python3 scripts/locplus_groupbw_extend2.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import locplus_groupbw_multiseason as S

S.SWEEP_GROUPS = ('FF', 'SI', 'CU')
S.BWS = (8.0, 10.0, 14.0)

if __name__ == '__main__':
    S.main()
