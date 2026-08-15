# Pitch subtype reference profiles

Preserved 2026-08-15 from `pitch_subtype_classifier.py` before its deletion
(the classifier was rejected in `expected_movement_review.md`; these hand-tuned
(IVB, HB) reference profiles are the naming source the cluster work cited).

```python
SUBTYPES = [
    # ── FF (Four-Seam Fastball) ──
    ('Gyro Fastball',    'FF',   9,     0),
    ('Inefficient FF',   'FF',  13,     6),
    ('Deadzone FB',      'FF',  14,    13.5),
    ('Running Fastball',  'FF',  15,    16),
    ('Relative Cut FF',  'FF',  16,     1.5),
    ('Standard FF',      'FF',  17,     9),
    ('Rider',            'FF',  19.5,   6.5),
    ('Ride n\' Run',     'FF',  19.5,  11),

    # ── SI (Sinker) ──
    ('Gyro Sinker',      'SI',   9,    12),
    ('Sinker',           'SI',   9,    16),
    ('Running Sinker',   'SI',   9.5,  19.5),
    ('Heavy Sinker',     'SI',   5,    15.5),
    ('Heavy Runner',     'SI',   5,    20),
    ('Diver',            'SI',  -1,    17),

    # ── FC (Cutter) ──
    ('Gyro Cutter',      'FC',   8,     0),
    ('Standard Cutter',  'FC',  10,    -3),
    ('Sweeping Cutter',  'FC',   9.5,  -6.5),
    ('Backspinner',      'FC',  13.5,   0.5),

    # ── SL (Slider) ──
    ('Gyro SL',          'SL',   1,    -2),
    ('Slutter',          'SL',   5,    -4.5),
    ('Standard SL',      'SL',   0.5,  -6.5),

    # ── ST (Sweeper) ──
    ('Sweeper',          'ST',  -1.5, -15),

    # ── CU (Curveball) ──
    ('Gyro CB',          'CU',  -5,    -2),
    ('IE Downer',        'CU',  -9,    -2),
    ('Standard CB',      'CU', -13,   -10),
    ('Downer',           'CU', -15,    -5),
    ('Efficient CB',     'CU', -18,   -14),

    # ── SV (Slurve) ──
    ('IE Slurve',        'SV',  -6.5,  -7),
    ('Slurve',           'SV',  -8,   -14),
    ('Efficient Slurve', 'SV', -12,   -19),
]
```
