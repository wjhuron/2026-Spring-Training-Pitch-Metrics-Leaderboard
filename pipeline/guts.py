#!/usr/bin/env python3
"""Scrape the 2026 row from FanGraphs Guts page (https://www.fangraphs.com/tools/guts).

Usage: python3 guts.py
"""

import urllib.request
import json
import re


def scrape_guts(year=2026):
    """Fetch FanGraphs Guts page and extract the specified year's row."""
    url = 'https://www.fangraphs.com/tools/guts'
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
        'Accept': 'text/html',
    })
    html = urllib.request.urlopen(req, timeout=15).read().decode('utf-8')

    # FanGraphs uses Next.js — data is embedded in __NEXT_DATA__ script tag
    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
    if not match:
        raise RuntimeError('Could not find __NEXT_DATA__ on FanGraphs Guts page')

    data = json.loads(match.group(1))
    props = data.get('props', {})
    page_props = props.get('pageProps', {})
    dehydrated = page_props.get('dehydratedState', {})
    queries = dehydrated.get('queries', [])
    if not queries:
        raise RuntimeError('FanGraphs Guts page structure changed — no queries found in __NEXT_DATA__')

    for q in queries:
        rows = q.get('state', {}).get('data', [])
        if isinstance(rows, list) and rows and isinstance(rows[0], dict) and 'Season' in rows[0]:
            for row in rows:
                if row.get('Season') == year:
                    return row

    raise RuntimeError(f'Could not find {year} row in FanGraphs Guts data')


if __name__ == '__main__':
    row = scrape_guts()

    fields = [
        ('Season',    'd',  0),
        ('wOBA',      'f',  3),
        ('wOBAScale', 'f',  3),
        ('wBB',       'f',  3),
        ('wHBP',      'f',  3),
        ('w1B',       'f',  3),
        ('w2B',       'f',  3),
        ('w3B',       'f',  3),
        ('wHR',       'f',  3),
        ('runSB',     'f',  3),
        ('runCS',     'f',  3),
        ('R/PA',      'f',  3),
        ('R/W',       'f',  3),
        ('cFIP',      'f',  3),
    ]

    print(f"\n{'─' * 50}")
    print(f"  FanGraphs Guts — {row['Season']}")
    print(f"{'─' * 50}")
    for key, fmt, dec in fields:
        val = row.get(key)
        if val is None:
            continue
        if fmt == 'd':
            print(f"  {key:12s}  {val}")
        else:
            print(f"  {key:12s}  {val:.{dec}f}")
    print(f"{'─' * 50}\n")
