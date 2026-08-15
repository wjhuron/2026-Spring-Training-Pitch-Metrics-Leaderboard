#!/usr/bin/env python3
"""Entry stub — launchd com.huronalytics.refreshpickle pins this path.

The implementation lives in pipeline.refresh_pickle. Delete this stub only after
updating the launchd plist (see docs/launchd/ for a ready replacement)
and running `launchctl unload && launchctl load` on it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.refresh_pickle import *          # re-export for any stale imports
from pipeline.refresh_pickle import main

if __name__ == '__main__':
    main()
