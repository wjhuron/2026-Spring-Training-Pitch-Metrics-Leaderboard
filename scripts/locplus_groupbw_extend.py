"""locplus_groupbw_extend.py — grid extension for the per-group bandwidth
sweep. Round 1 (locplus_groupbw_multiseason.py) had FF 5/5, SI 4/5, CU 4/5
better at x=5.5 with reliability moving the same direction — but 5.5 was the
grid edge. This extends the grid to {5.5, 6.5, 8.0} for those three groups
so the optimum is bracketed (per the never-take-the-edge rule).

Usage: python3 scripts/locplus_groupbw_extend.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import locplus_groupbw_multiseason as S

S.SWEEP_GROUPS = ('FF', 'SI', 'CU')
S.BWS = (5.5, 6.5, 8.0)

if __name__ == '__main__':
    S.main()
