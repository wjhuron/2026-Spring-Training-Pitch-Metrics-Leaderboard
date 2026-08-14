"""locplus_groupbw_extend3.py — round 4 of the per-group x-bandwidth search.
FF/CU were still monotone at the 14.0 edge. This tests {14, 24, 200}; 200
inches is effectively an x-flat physical surface (uniform kernel across the
36-inch grid), i.e. the explicit "value is vertical + CS edge only" limit.
(The called-strike surface always keeps the global 4.5/0.22 pair, so
zone-edge x-structure is preserved in every variant.)

Usage: python3 scripts/locplus_groupbw_extend3.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import locplus_groupbw_multiseason as S

S.SWEEP_GROUPS = ('FF', 'CU')
S.BWS = (14.0, 24.0, 200.0)

if __name__ == '__main__':
    S.main()
