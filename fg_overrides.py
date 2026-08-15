#!/usr/bin/env python3
"""Entry stub — launchd com.huronalytics.refreshfg pins this path.

The implementation lives in pipeline.fg_overrides. Delete this stub only after
updating the launchd plist (see docs/launchd/ for a ready replacement)
and running `launchctl unload && launchctl load` on it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.fg_overrides import *          # re-export for any stale imports
from pipeline.fg_overrides import main

if __name__ == '__main__':
    main()
