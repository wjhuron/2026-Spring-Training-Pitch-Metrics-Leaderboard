#!/usr/bin/env python3
"""Simulate the Okabe-Ito pitch palette (Palette A) under colorblindness to
verify the 9 colors stay mutually distinguishable.

Uses Machado et al. (2009) dichromacy matrices applied in linear RGB, for
deuteranopia, protanopia, and tritanopia (severity 1.0). Renders a swatch grid
(Normal + 3 CVD types) and prints the closest color pairs under each type.
Output: ~/Downloads/okabe_cvd_check.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

BG = '#f0e8d8'
TEXT_PRIMARY = '#1a1612'
TEXT_MUTED = '#6a5f55'

PAL = {'FF': '#0072B2', 'SI': '#E69F00', 'FC': '#8B5A2B', 'CH': '#009E73',
       'FS': '#CC79A7', 'SL': '#D55E00', 'ST': '#56B4E9', 'SV': '#882255', 'CU': '#332288'}
ORDER = ['FF', 'SI', 'FC', 'CH', 'FS', 'SL', 'ST', 'SV', 'CU']
NAMES = {'FF': '4-Seam', 'SI': 'Sinker', 'FC': 'Cutter', 'CH': 'Change', 'FS': 'Split',
         'SL': 'Slider', 'ST': 'Sweeper', 'SV': 'Slurve', 'CU': 'Curve'}

MACHADO = {
    'Deuteranopia (~6% of men)': np.array([[0.367322, 0.860646, -0.227968],
                                           [0.280085, 0.672501, 0.047413],
                                           [-0.011820, 0.042940, 0.968881]]),
    'Protanopia (~2%)':          np.array([[0.152286, 1.052583, -0.204868],
                                           [0.114503, 0.786281, 0.099216],
                                           [-0.003882, -0.048116, 1.051998]]),
    'Tritanopia (rare)':         np.array([[1.255528, -0.076749, -0.178779],
                                           [-0.078411, 0.930809, 0.147602],
                                           [0.004733, 0.691367, 0.303900]]),
}


def hex2rgb(h):
    return np.array([int(h[i:i+2], 16) / 255 for i in (1, 3, 5)])


def srgb2lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin2srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def rgb2hex(rgb):
    return '#' + ''.join(f'{int(round(np.clip(x,0,1)*255)):02X}' for x in rgb)


def simulate(hexc, mat):
    return lin2srgb(mat @ srgb2lin(hex2rgb(hexc)))


def lum(rgb):
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


rows = [('Normal vision', None)] + list(MACHADO.items())

fig, ax = plt.subplots(figsize=(13, 6.2), dpi=150)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, len(ORDER)); ax.set_ylim(0, len(rows))
ax.axis('off')
ax.set_title('Palette A (Okabe-Ito) under colorblindness — do any two pitches collapse?',
             fontsize=14, fontweight='bold', color=TEXT_PRIMARY, pad=14)

for ri, (label, mat) in enumerate(rows):
    y = len(rows) - 1 - ri
    ax.text(-0.12, y + 0.5, label, ha='right', va='center', fontsize=10.5,
            fontweight='bold', color=TEXT_MUTED)
    for ci, pt in enumerate(ORDER):
        rgb = hex2rgb(PAL[pt]) if mat is None else simulate(PAL[pt], mat)
        ax.add_patch(plt.Rectangle((ci + 0.04, y + 0.08), 0.92, 0.84,
                     facecolor=tuple(np.clip(rgb, 0, 1)), edgecolor='none'))
        tc = '#1a1612' if lum(rgb) > 0.55 else '#ffffff'
        ax.text(ci + 0.5, y + 0.5, pt, ha='center', va='center',
                fontsize=10, fontweight='bold', color=tc)

# pitch-name header under the grid
for ci, pt in enumerate(ORDER):
    ax.text(ci + 0.5, -0.18, NAMES[pt], ha='center', va='top', fontsize=8, color=TEXT_MUTED)

out = os.path.expanduser('~/Downloads/okabe_cvd_check.png')
fig.savefig(out, dpi=150, facecolor=BG, bbox_inches='tight', pad_inches=0.25)
print('Saved:', out)

# ── closest pairs under each CVD type (Euclidean in simulated sRGB) ──
print('\nClosest color pairs (smaller = harder to tell apart):')
for label, mat in MACHADO.items():
    sims = {pt: (hex2rgb(PAL[pt]) if mat is None else simulate(PAL[pt], mat)) for pt in ORDER}
    pairs = []
    for i in range(len(ORDER)):
        for j in range(i + 1, len(ORDER)):
            a, b = ORDER[i], ORDER[j]
            d = float(np.linalg.norm(np.clip(sims[a], 0, 1) - np.clip(sims[b], 0, 1)))
            pairs.append((d, a, b))
    pairs.sort()
    closest = ', '.join(f'{a}-{b} ({d:.2f})' for d, a, b in pairs[:3])
    print(f'  {label:28s} closest: {closest}')
