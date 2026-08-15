#!/usr/bin/env python3
"""HitterCards_light_gray.py — Clean light-gray theme.

Modern, Apple-like light theme. Off-white background, dark text,
confident blue accent. Reads well on bright phone screens.

Usage:
    python3 HitterCards_light_gray.py --hitters "Wood, James"
"""

import argparse

import cards.hitter as HitterCards
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────────────
# LIGHT GRAY PALETTE
# ─────────────────────────────────────────────────────────────────────
HitterCards.BG          = '#f5f5f7'   # off-white (Apple-like)
HitterCards.DARK_CELL   = '#e9e9ed'   # slightly darker cell bg
HitterCards.DARKER      = '#dadbe1'   # header/Total bg
HitterCards.ACCENT      = '#0066cc'   # confident blue accent

# Text — dark on light
HitterCards.TEXT_PRIMARY       = '#1c1c1e'   # near-black
HitterCards.TEXT_SECONDARY     = '#3a3a3c'   # dark gray
HitterCards.TEXT_MUTED         = '#6b6b70'   # mid gray
HitterCards.TEXT_FAINT         = '#8b8b90'
HitterCards.TEXT_DIMMED        = '#9a9aa0'

# Borders + grid
HitterCards.SUBTLE_BORDER      = '#d2d2d6'
HitterCards.TOTAL_BORDER       = '#6b6b70'
HitterCards.ALT_ROW_BG         = '#ececf0'
HitterCards.GRID_COLOR         = '#3a3a3c'   # dark grid on light bg
HitterCards.TICK_COLOR         = '#6b6b70'
HitterCards.SPINE_COLOR        = '#9a9aa0'
HitterCards.PHOTO_BORDER       = '#9a9aa0'
HitterCards.PERCENTILE_NEUTRAL = '#8b8b90'

# Heat-map colormap — diverging dark blue → light-gray bg → dark red.
HitterCards.HEAT_CMAP = HitterCards.make_heat_cmap(HitterCards.BG)

# WOBA_CMAP — leave as the saturated default (matches hitter page).


# Patch savefig
_orig_savefig = plt.savefig
def _light_savefig(out_path, *a, **k):
    if 'HitterCard_' in str(out_path) and '_light' not in str(out_path):
        out_path = str(out_path).replace('.png', '_light_gray.png')
    return _orig_savefig(out_path, *a, **k)


def main():
    parser = argparse.ArgumentParser(description='Light-gray hitter cards')
    parser.add_argument('--hitters', default='Wood, James')
    parser.add_argument('--year-label', default='2026 Season')
    parser.add_argument('--output-dir', default=HitterCards.OUTPUT_DIR)
    args = parser.parse_args()
    names = [h.strip() for h in args.hitters.split(';') if h.strip()]

    plt.savefig = _light_savefig
    try:
        for name in names:
            HitterCards.render_hitter_card(name, year_label=args.year_label,
                                             output_dir=args.output_dir)
    finally:
        plt.savefig = _orig_savefig


if __name__ == '__main__':
    main()
