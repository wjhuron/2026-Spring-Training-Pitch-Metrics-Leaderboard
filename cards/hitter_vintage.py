#!/usr/bin/env python3
"""HitterCards_vintage.py — Vintage broadcast theme.

1980s ABC Monday Night Baseball graphics aesthetic. Warm charcoal bg,
amber neon accent, cream warm-tone text. Note: stays in the dark realm
but with a totally different mood than the default — warm and analog
rather than cool and modern.

Usage:
    python3 HitterCards_vintage.py --hitters "Wood, James"
"""

import argparse

import cards.hitter as HitterCards
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────────────
# VINTAGE BROADCAST PALETTE
# ─────────────────────────────────────────────────────────────────────
HitterCards.BG          = '#1d1f23'   # warm charcoal
HitterCards.DARK_CELL   = '#262830'   # cell bg
HitterCards.DARKER      = '#191b1f'   # darker for headers/Total bg
HitterCards.ACCENT      = '#ffb74d'   # amber neon — the broadcast accent

# Text — warm cream tones on charcoal (not pure white)
HitterCards.TEXT_PRIMARY       = '#ffe9b8'   # warm cream
HitterCards.TEXT_SECONDARY     = '#d4b88a'   # mid amber
HitterCards.TEXT_MUTED         = '#9a8866'   # muted amber
HitterCards.TEXT_FAINT         = '#5c5340'   # deep amber-gray
HitterCards.TEXT_DIMMED        = '#6e6346'

# Borders + grid
HitterCards.SUBTLE_BORDER      = '#3a3530'   # warm dark border
HitterCards.TOTAL_BORDER       = '#8a7548'   # amber-tinged border
HitterCards.ALT_ROW_BG         = '#222428'
HitterCards.GRID_COLOR         = '#ffe9b8'   # cream gridlines on charcoal
HitterCards.TICK_COLOR         = '#9a8866'
HitterCards.SPINE_COLOR        = '#5c5340'
HitterCards.PHOTO_BORDER       = '#9a8866'
HitterCards.PERCENTILE_NEUTRAL = '#9a8866'

# Heat-map colormap — diverging dark blue → charcoal bg → dark red.
HitterCards.HEAT_CMAP = HitterCards.make_heat_cmap(HitterCards.BG)

# WOBA_CMAP — leave as the saturated default (matches hitter page).


# Patch savefig
_orig_savefig = plt.savefig
def _vintage_savefig(out_path, *a, **k):
    if 'HitterCard_' in str(out_path) and '_vintage' not in str(out_path):
        out_path = str(out_path).replace('.png', '_vintage.png')
    return _orig_savefig(out_path, *a, **k)


def main():
    parser = argparse.ArgumentParser(description='Vintage broadcast hitter cards')
    parser.add_argument('--hitters', default='Wood, James')
    parser.add_argument('--year-label', default='2026 Season')
    parser.add_argument('--output-dir', default=HitterCards.OUTPUT_DIR)
    args = parser.parse_args()
    names = [h.strip() for h in args.hitters.split(';') if h.strip()]

    plt.savefig = _vintage_savefig
    try:
        for name in names:
            HitterCards.render_hitter_card(name, year_label=args.year_label,
                                             output_dir=args.output_dir)
    finally:
        plt.savefig = _orig_savefig


if __name__ == '__main__':
    main()
