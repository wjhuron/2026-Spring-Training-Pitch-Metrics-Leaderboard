# Expected movement (xIVB / xHB, IVBOE / HBOE) — review and rebuild

Status: the Finding 5 basis (spin + release axis) SHIPPED on 2026-09-03 and was
PULLED the same day. It explains movement far better, but its residual explains
pitch RESULTS worse than raw movement, because it credits away the spin and
axis that are the weapon (`scripts/research/xmove/xmove_residual_value.py`,
5-season LOSO: whiff r 0.05 vs 0.13 for the slot + extension + velocity
residual). The chart on the player page is about performance, so
`pipeline/xmove.py` ships the HITTER basis (slot, extension, velocity, the
same inputs as the pre-review model) with the per-pitch scoring and micro-row
sums built for the physics basis; the physics basis stays in the module for
research. Everything below is the movement-fit analysis. Scripts under
`scripts/research/xmove/`.
Findings 1-5 are the model rebuild; 6 covers whether the pitch-type label can be
removed (it cannot) and what a retag costs; 7 covers cross-axis break, which is
the only measurement here that needs no model and no label; 8 renders the two
surviving forms on one identical plate and measures what the per-class
intercept actually does; 9 is the neighbour contamination that decides between
them.
Data: `data/_pitches{2021..2025}_training.pkl`, 3,430,815 MLB pitches with
movement, spin rate, release spin axis, arm angle, extension and release point.

## What ships today

`process_data.fit_mvn_models` fits, per (pitch type, hand), a joint normal over

    [IVB, HB, ArmAngle, Extension, Velocity]

and takes the conditional mean of the first two given the last three. That
conditional mean is algebraically the per-group OLS fit, so everything below
uses OLS — identical numbers, far faster. `Cards.py` and `js/aggregator.js`
carry copies of the same math, fed by `mvnModels` in `data/metadata_rs.json`.
`ivbOE = IVB - xIVB`, `hbOE = HB - xHB`. The ROC variant swaps arm angle for
`RelPosZ/RelPosX`. Both movement columns are weather-adjusted
(`xIndVrtBrk`/`xHorzBrk`), and so is the leaderboard's `indVertBrk`, so the
residual is in one consistent currency. The card ellipses were fixed 7-inch
hatched circles, disabled at `Cards.py:1665`; the leaderboard and player page
never render `ivbOE`/`hbOE` at all.

## Finding 1 — the model omits the two dominant causes of movement

Spin rate and the measured release spin axis are both in the data at 99%+
coverage for every season 2021-2025, and neither is a regressor.

Pooled pitch-level R², per (pitch type, hand), 3.43M pitches:

| regressor set | R² IVB | R² HB | RMSE IVB | RMSE HB |
|---|---|---|---|---|
| group mean only | 0.000 | 0.000 | 4.13" | 3.92" |
| **shipped** — arm angle, extension, velo | 0.303 | 0.210 | 3.45" | 3.49" |
| + spin rate | 0.352 | 0.287 | 3.33" | 3.31" |
| + spin rate + release axis | 0.494 | 0.431 | 2.94" | 2.96" |
| all of the above + release point | 0.500 | 0.433 | 2.92" | 2.96" |

Worst cases for the shipped set: slider IVB R² 0.102, sweeper IVB 0.105,
sinker HB 0.095, changeup HB 0.089. With spin and axis those become 0.475,
0.467, 0.367, 0.315.

## Finding 2 — the residual is a near-duplicate of the column beside it

Because the model explains so little, `IVB - xIVB` is close to `IVB` minus a
league constant. At the rendered unit (pitcher × hand × pitch type × season,
≥50 pitches), correlation between the residual and the raw movement it is
supposed to contextualise:

| pitch type | corr(ivbOE, IVB) | corr(hbOE, HB) |
|---|---|---|
| SI | 0.632 | **0.925** |
| CH | 0.675 | **0.914** |
| SL | **0.905** | 0.864 |
| ST | **0.922** | 0.861 |
| FC | 0.869 | 0.675 |
| FS | 0.851 | 0.723 |

Reliability is not the problem — split-half reliability is 0.99 and
year-over-year persistence 0.71-0.90 for both the shipped and the candidate
models, because movement is one of the most stable things a pitcher does.
Distinctness is the problem.

## Finding 3 — the SSW premise, measured

`SpinAxis` is a genuine release measurement, not the movement-derived axis: if
it were the latter, OTilt − RTilt would be identically zero. Hand-signed
circular deviation (positive = arm side of the release axis), 2021-25:

| pitch type | n | mean dev | sd | mean abs |
|---|---|---|---|---|
| FF | 1,150,873 | −8.8° | 8.9 | 10.0 |
| SI | 539,106 | **+18.3°** | 12.1 | 19.4 |
| FC | 274,205 | −23.9° | 21.3 | 27.7 |
| SL | 540,575 | −3.1° | 40.6 | 32.7 |
| ST | 203,257 | **+31.3°** | 16.1 | 32.0 |
| CU | 243,007 | +0.7° | 18.1 | 13.0 |
| CH | 375,130 | +9.7° | 15.3 | 14.5 |
| FS | 84,801 | +16.7° | 21.3 | 22.5 |

For the 1,237 pitcher-seasons throwing both a four-seam and a sinker (≥50
each):

* release axis: correlation **0.996**, median shift **2.6°** — effectively the
  same axis, exactly as expected
* observed break direction: median shift **28.4°**
* non-Magnus (cross-axis) break: **−3.16" on the FF, +5.60" on the SI** —
  **8.8 inches** of the separation is seam-shifted wake, not spin orientation

Contrast FF → FC: the release axis genuinely moves (median 65°) and the
cross-axis component barely changes (−2.84" → −3.45"). Cutters are a spin
orientation story. SL → ST: cross-axis +7.6", a seam story.

## The release-axis frame

With `u = (cos RTilt, sin RTilt)` a unit vector in (IVB, arm-side HB) space:

    along = break · u        Magnus-direction break, inches
    cross = break · u_perp   non-Magnus / seam-shifted deflection, inches

Both are linear in (IVB, HB), so a gyro slider's tiny break does not blow the
residual up the way a tilt-in-degrees residual does (slider dev sd is 40.6°
precisely because of this). `cross` is the SSW number in the unit a pitching
coach already thinks in.

## Finding 4 — head-to-head, cross-fit within season by game parity

Every pitch is scored by a model that never saw its game; the split is repeated
in both directions and across all five seasons, matching how production refits
each season. `DIST` is the mean |corr(OE, raw movement)| across pitch types at
the rendered unit, and lower is better.

| form | R² IVB | R² HB | RMSE i | RMSE h | DIST i | DIST h | YoY i | YoY h |
|---|---|---|---|---|---|---|---|---|
| S1 shipped — arm angle, ext, velo | 0.309 | 0.214 | 3.41 | 3.47 | 0.796 | 0.842 | 0.845 | 0.846 |
| S2 + spin rate | 0.360 | 0.292 | 3.28 | 3.29 | 0.757 | 0.759 | 0.837 | 0.825 |
| S3 + spin + release axis | 0.500 | 0.436 | 2.90 | 2.94 | 0.641 | 0.677 | 0.789 | 0.786 |
| S3b + spin×axis interactions | 0.505 | 0.445 | 2.89 | 2.91 | 0.636 | 0.672 | 0.788 | 0.784 |
| S4 as S3b, no arm angle | 0.457 | 0.436 | 3.02 | 2.94 | 0.692 | 0.693 | 0.795 | 0.788 |
| S5 + release point | 0.511 | 0.448 | 2.87 | 2.91 | 0.628 | 0.667 | 0.780 | 0.777 |
| S6 polar (release-axis frame) | 0.417 | 0.392 | 3.13 | 3.05 | **0.478** | **0.560** | 0.781 | 0.775 |
| S7 GBM (ceiling) | **0.657** | **0.598** | 2.40 | 2.48 | 0.544 | 0.568 | 0.663 | 0.639 |

Reads:

* Arm angle earns its place (S4 vs S3b) but is worth far less than spin+axis.
* Release point adds ~0.006 R² over S3b. Not worth the extra dimensions.
* The polar frame buys the best distinctness per unit of fit — it removes the
  axis direction exactly rather than approximating it with additive trig.
* **There is a large nonlinear gap**: the GBM reaches 0.657/0.598 where the
  best linear form manages 0.511/0.448. The Magnus lift coefficient saturates
  with spin factor and interacts with the axis, and no additive linear term
  captures that.
* Falling YoY across the table is expected, not a defect: a better model moves
  more of the pitcher's own stable inputs out of the residual, so what is left
  is a smaller, purer quantity. All forms stay comfortably persistent.

### Two sweep bugs worth recording

The first two passes of `xmove_sweep.py` produced R² of −1e16 and are void. Both
causes are easy to hit again:

1. **Basis conditioning.** A hand-rolled truncated-power spline on spin rate
   puts columns of order 1e10 (spin cubed) next to columns of order 1, and
   `lstsq(..., rcond=None)` returns garbage. Fixed by using a cubic B-spline
   basis, which is compactly supported and bounded in [0, 1].
2. **Non-constant coverage.** Gating each config on its own `k` meant wider
   designs blanked more groups, so the R² denominators differed between grid
   points and the "comparison" was not one. Fixed by setting the per-group
   training floor from the widest design in the grid, so every config blanks
   the same groups. Coverage is now reported per row (0.994) precisely so that
   a config quietly scoring fewer rows is visible rather than flattering.

### A note on the objective

Out-of-sample R² is used as the primary here, which normally would be gameable
— conditioning on more of a thing's own causes always raises it while shrinking
the residual toward zero. It is admissible in this specific case because every
candidate regressor is strictly *upstream* of movement: spin rate, release spin
axis, arm angle, extension, velocity. None is derived from the break. The
inputs that WOULD make R² meaningless are exactly the circular ones — OTilt
(`atan2(HB, IVB)`, the answer restated) and the movement-derived pitch subtype
from `pitch_subtype_classifier.py`. Neither is admitted. DIST is carried as the
guard against a config that buys R² by pushing the residual back toward the raw
column.

## Finding 5 — how much of the nonlinear gap a fixed basis recovers

Why this decides the shipping path: the site does not run a model.
`js/aggregator.js` reads mu/cov per group out of `metadata_rs.json` and takes
an MVN conditional mean, which is exactly OLS on whatever sits in the vector.
So **any fixed basis — splines, axis harmonics, tensor interactions — still
ships as a covariance matrix and needs no new plumbing.** A GBM does not: it
would have to be scored per pitch in `process_data.py` and summed into every
aggregation cell, or filtered views on the site would go wrong.

Release-axis frame, cross-fit by game parity within season, coverage held
constant at 0.994 across the grid. `wrstY` is the worst of the five seasons.

| spin df | harmonics | k | R² IVB | R² HB | DIST i | DIST h | wrstY i | wrstY h |
|---|---|---|---|---|---|---|---|---|
| linear | 1 | 14 | 0.519 | 0.464 | 0.624 | 0.657 | 0.508 | 0.457 |
| linear | **2** | 16 | **0.528** | **0.471** | 0.611 | 0.653 | 0.513 | 0.459 |
| linear | 3 | 18 | 0.517 | 0.455 | 0.610 | 0.650 | 0.456 | 0.370 |
| 2 | 1 | 20 | 0.521 | 0.459 | 0.618 | 0.647 | 0.499 | 0.417 |
| 2 | 2 | 22 | 0.526 | 0.466 | 0.607 | 0.646 | 0.500 | 0.421 |
| 2 | 3 | 24 | 0.517 | 0.449 | 0.605 | 0.644 | 0.453 | 0.335 |
| 4 | 1 | 23 | 0.501 | 0.445 | 0.617 | 0.645 | 0.478 | 0.414 |
| 4 | 2 | 25 | 0.501 | 0.455 | 0.606 | 0.645 | 0.462 | 0.431 |
| 4 | 3 | 27 | 0.494 | 0.442 | 0.605 | 0.642 | 0.437 | 0.359 |
| 6 | 1 | 29 | 0.429 | 0.351 | 0.614 | 0.633 | 0.152 | 0.113 |
| 6 | 2 | 31 | 0.442 | 0.383 | 0.600 | 0.633 | 0.230 | 0.233 |
| 6 | 3 | 33 | 0.429 | 0.342 | 0.596 | 0.625 | 0.184 | 0.171 |

**Both axes bracket an interior optimum**, so this is a measured argmax and not
an edge of grid:

* harmonics 1 → 2 → 3 gives 0.519 → 0.528 → 0.517
* spin linear → df2 → df4 → df6 gives 0.528 → 0.526 → 0.501 → 0.442, and
  dropping spin entirely (Finding 4, S2 vs S3) costs far more still

Winner: **linear in spin, two axis harmonics, spin×axis tensor, k = 16.**
The spin spline buys nothing, which says the nonlinearity the GBM is exploiting
is *interaction* structure, not curvature in spin.

### Reading DIST correctly

DIST has a floor and is not a minimand. Seam effect genuinely correlates with
movement — a sinker with more seam gets more arm-side run — so `corr(OE, HB)`
stays well above zero even for a perfect model. The GBM, the best-fitting form
available, sits at 0.544/0.568. So the ladder reads:

* shipped 0.796 / 0.842 — genuinely redundant with the raw column
* basis 0.611 / 0.653 — a real gain
* GBM 0.544 / 0.568 — approximately the floor
* the no-harmonic polar form from Finding 4, 0.478 / 0.560, is *below* the best
  model's floor. That is over-removal — signal discarded, not a cleaner
  residual — and it is why the R² optimum is taken as the winner with DIST used
  only to confirm the residual is not drifting back toward the raw column.

## Finding 6 — the label cannot be removed, and it is worth more than the residual

Two questions the earlier findings left open, both about the pitch-type term.
Scripts: `xmove_agnostic.py`, `xmove_agnostic_basis.py`,
`xmove_agnostic_vs_class.py`, `xmove_retag_sensitivity.py`.

### 6a. The pitch-type-agnostic variant, measured

Recommended design section 1 offered a pooled variant (drop pitch type from the
grouping key) as "a one-line change, a different metric, not a better one".
Measured, pooled by hand x season with the Finding 5 feature set:

| | pooled R² | mean within-type R² |
|---|---|---|
| IVB | 0.708 | **-0.320** |
| HB | 0.793 | **-0.775** |

**The pooled number is an artifact and must not be quoted.** Pooled across
classes most of the variance is between classes, so "a downward axis breaks
down" scores 0.7 while every within-class fit is worse than that class's own
mean. Only the within-type column is meaningful, and it is the unit the card
renders. This is the same wrong-unit trap that Finding 4 avoids for DIST and
does not avoid for R².

Adding basis does not rescue it. Mean within-type R², harmonics with the spin
and velo tensors:

| form | k | IVB | HB |
|---|---|---|---|
| H1 | 8 | -0.320 | -0.775 |
| H2 | 12 | -0.058 | -0.396 |
| H3 | 16 | 0.012 | -0.285 |
| H2 + velo x axis | 16 | 0.077 | -0.355 |
| H3 + velo x axis | 22 | **0.094** | **-0.277** |

Monotone to the edge of the grid, so by the usual rule the grid is too small.
But the increments are decelerating and would need to quintuple to reach the
per-type ~0.48 / ~0.41, so extending it is not worth the run.

**The reason is physical, not a modelling failure.** Two pitches with identical
arm angle, extension, velocity, spin rate and release axis genuinely move
differently, and that difference *is* the seam-shifted wake. SSW is a function
of seam orientation and gyro fraction, neither of which the feed measures. A
release-only expectation therefore cannot fit within pitch type by
construction. The negative R² is the statement that pitch type carries real
physical information the sensors do not capture.

What the agnostic form does deliver is the FF/SI framing, exactly as section 1
predicted. For the 1,237 pitcher-seasons throwing both:

| | expected gap | residual gap |
|---|---|---|
| per-type | IVB 6.70", HB -6.34" | IVB -0.96", HB -0.93" |
| agnostic | IVB 0.79", HB -0.42" | **IVB 4.78", HB -6.59"** |

**And the shortcut does not exist.** If the per-class expectation were the
agnostic one plus a class constant, that constant would cancel out of any
within-class comparison, and one could compute the retag-proof agnostic
residual and do the class comparison at display time. It is not a constant:
the per-class fit changes the slopes too. Correlation between the two
residuals within class, across pitcher-seasons:

| | corr IVB | rank IVB | sd(difference) | sd(the signal) |
|---|---|---|---|---|
| FF | 0.711 | 0.632 | 1.27 | 1.37 |
| SI | 0.882 | 0.877 | 1.17 | 2.26 |
| FC | 0.753 | 0.717 | 1.61 | 1.99 |
| SL | 0.663 | 0.635 | 1.95 | 1.87 |

The two models disagree by as much as the entire spread of pitchers within a
class. Class is an interaction with the release parameters, not an offset.

### 6b. What a retag costs

Limitations already noted that a re-tagged pitch moves its own baseline. The
size was never measured. Score every pitch tagged X under the X-fitted model
and again under the Y-fitted model; the pitches are identical, only the model
changes. RHP, Finding 5 feature set:

| retag | n | expectation shift | vs the model's own median residual |
|---|---|---|---|
| SL → ST | 404k | 5.93" | **1.76x** |
| ST → SL | 154k | 5.82" | 1.51x |
| FF → SI | 832k | 8.78" | **3.35x** |
| SI → FF | 383k | 8.69" | 2.92x |
| FC → SL | 205k | 3.12" | 1.06x |

**Every boundary moves the expectation by more than the residual it is meant to
measure.** Concretely, Luis Medina's 2026 slider deviates 4.4" from expectation
as a slider and 2.1" as a sweeper; his sinker deviates 0.7" as a sinker and
8.8" as a four-seam. Not one pitch changes.

So the per-class residual cannot be presented as a physics claim. It is a peer
comparison, and under that reading a retag changing the answer is correct
semantics rather than a defect: renaming his sliders sweepers means asking how
he compares to sweeper-throwers instead. The failure is only in the framing.

## Finding 7 — cross-axis break, the model-free measurement

Scripts: `xmove_tilt_gap.py`, `xmove_cross_validate.py`.

Finding 3 established the release-axis frame. Taking `cross` on its own, with
no expectation subtracted and no model at all, gives a direct SSW measurement
from two measured columns and trigonometry. It has three properties nothing
else in this document has: no fitted expectation, no league baseline, and no
pitch type, so **there is nothing for a retag to move**.

The unit matters. `cross = total break x sin(tilt gap)`, so degrees and inches
are not the same ranking. Within pitch type they correlate 0.93 for sinkers,
0.94 for four-seams, 0.96 for changeups, but only 0.54 for sweepers. And at the
tail, where a leaderboard lives, the top five RHP sinkers of 2025 by degrees
and by inches **overlap only 2 of 5**: Stroman has the largest angle in
baseball (37.1°) and the sixth-largest deflection (6.9"), while Keller carries
8.9" with seven fewer degrees. Report inches.

Per class, means over pitcher-hand-type-seasons, which reproduce Finding 3
independently (sinker +18.1° here against +18.3° there, sweeper +30.9° against
+31.3°, cutter -23.9° against -23.9°):

| | cross (in) | tilt gap (deg) |
|---|---|---|
| FC | -3.55 | -23.9 |
| FF | -2.78 | -9.3 |
| SL | 0.13 | -1.5 |
| CU | 0.49 | 0.7 |
| CH | 2.52 | 9.8 |
| FS | 3.07 | 15.2 |
| SI | 5.44 | 18.1 |
| ST | 7.30 | 30.9 |

### Is it a pitcher trait, or a pitch-type detector?

Both, and the split is worth stating plainly. **70.4% of the variance sits
between classes and 29.6% within** (sd 3.44" between, 2.23" within). So the
headline number is substantially a classifier. That is a virtue for the
descriptive job and a limit for an evaluation one.

The within-class part is not noise:

| | split-half reliability | year over year | vs raw HB | vs raw IVB |
|---|---|---|---|---|
| FF | 0.956 | 0.819 | **0.19** | 0.29 |
| SI | 0.974 | 0.866 | **0.00** | 0.16 |
| FC | 0.966 | 0.720 | 0.17 | 0.52 |
| SL | 0.986 | 0.821 | 0.63 | 0.13 |
| ST | 0.938 | 0.639 | 0.53 | 0.14 |
| CU | 0.984 | 0.912 | 0.62 | 0.24 |
| CH | 0.983 | 0.869 | **0.01** | 0.39 |
| FS | 0.972 | 0.802 | 0.59 | 0.24 |

Mean within-class reliability 0.970 and persistence 0.806, split-half by game
parity at the rendered unit. That sits in the same family as the movement
metrics in Finding 4 and above Command+ at 0.795.

**The distinctness column reverses the pitch-table verdict.** IVBOE/HBOE within
(type, hand) restate their raw column at 0.62 to 0.99 and therefore do not earn
a column beside it. Cross-axis is 0.00 for sinkers, 0.01 for changeups and 0.19
for four-seams: essentially orthogonal on exactly the pitches where seam effect
is most interesting and least visible. On breaking balls it runs 0.53 to 0.63,
because a sweeper's horizontal break largely *is* its seam effect, so it is
partially redundant there.

### The intended job is descriptive, which changes how to read all of the above

Per Wally, cross-axis is not meant to be a skill grade. It is there to explain
*why* pitches move the way they do. Under that framing the 70/30 split inverts
from a limitation into the whole point: a single measured quantity that orders
cutter (-3.55") through sweeper (+7.30") along a physical axis is an
explanation of pitch identity, not a contaminated evaluation metric. It answers
"why does his sinker run six inches more than his four-seam off the same
release" with a number rather than an assertion.

Two consequences follow:

* The reliability and persistence above stop being the case for shipping and
  become a guarantee instead: the descriptive number is stable, so it needs no
  small-sample caveat at the rendered unit.
* **It should probably not be coloured as good or bad.** A percentile ramp
  asserts a value judgement the number does not carry; a diverging scale
  centred on zero (glove side against arm side) states what it actually
  measures. This is a genuine open design question rather than a settled one.

Because 70% of the variance is between classes, any cross-pitcher *ranking*
still has to be **within pitch type** or it just ranks pitch types. The
pipeline already computes per-pitch-type percentiles for `PITCH_PCTL_KEYS`. For
a descriptive display the cross-class ordering is the useful view and should be
shown on one shared axis, as `xmove_seam_panel.py` does.

Production would need `cross` computed in `process_data.py` from `RTilt` and
the two break columns, about three lines. It does not exist outside the xmove
harness today.

## Finding 8 — what the per-class intercept actually does

Findings 6 and 7 left two forms standing, and every comparison between them so
far was confounded: the option-2 render predated the card redesign, so it
differed in typography, palette and pitcher as well as in model. `--per-class`
on `xmove_plate_v2.py` fixes that. Same layout, same data, same pitcher, one
switch. Any difference between the two images is the model.

Naming, to keep the rest of this readable:

* **Option 1**, label-agnostic. One surface per hand. The label is never read,
  so a retag cannot move the expectation.
* **Option 2**, per class. One surface per (pitch type, hand), which is what
  `fit_mvn_models` ships today.

Max Fried, 2026, both forms:

| | opt 1 xIVB / xHB | opt 2 xIVB / xHB | pctl 1 → 2 |
|---|---|---|---|
| FC | 14.1 / -2.9 | 9.1 / 1.0 | 88 → 55 |
| SI | 14.3 / -9.5 | 11.5 / -13.9 | 99 → 100 |
| FF | 16.4 / -8.3 | 17.3 / -6.0 | 91 → 59 |
| CU | -16.6 / 12.2 | -17.4 / 12.4 | 22 → 35 |
| CH | 8.1 / -13.1 | 7.5 / -13.9 | 60 → 68 |
| ST | -2.9 / 13.4 | -0.8 / 15.9 | 24 → 55 |

The cutter and four-seam fall off a cliff, 88 → 55 and 91 → 59, because their
deflection is ordinary *for a cutter and a four-seam*: option 1 was partly
crediting Fried for the pitch type. The sinker survives and sharpens, because
it is strange even against other sinkers. Medina went the other way entirely,
his sinker deviation collapsing from 7.4" to 0.7" under option 2, so his was
almost purely the class effect. Both behaviours are correct, and which one a
viewer wants depends on the question being asked.

**The absorption is algebraic, not empirical.** `xmove_ssw_absorption.py` fits
both forms and takes the league-mean residual per class:

| RHP | SEAM | opt 1 magnitude | opt 2 magnitude |
|---|---|---|---|
| FC | -3.5 | 2.77 | 0.00 |
| FF | -2.6 | 2.46 | 0.00 |
| SL | 0.0 | 0.65 | 0.00 |
| CU | 0.5 | 2.64 | 0.00 |
| CH | 2.8 | 1.37 | 0.00 |
| SI | 5.8 | 5.01 | 0.00 |
| ST | 7.6 | 4.14 | 0.00 |

Option 2's column is zero everywhere because an intercept per class forces it,
not because the model explains anything. LHP reproduces the option-1 ordering
off a fully separate fit (SI 4.41, ST 3.66, FC 3.38, FF 2.43, CU 2.10, SL 0.92).

So the honest description is not that option 2 *accounts for* seam-shifted
wake. It **charges the class for it** and option 1 **charges the pitcher**. The
sinker's 5.0" and the sweeper's 4.1" do not get explained by option 2, they get
defined away. Neither form has any seam input; the only difference is the
baseline the leftover is measured against. Option 1's baseline is the average
seam effect across a whole arsenal mix, which is not zero, so option 1 is not a
"pure Magnus" model either and should never be described as one.

Under option 1 the magnitudes track SEAM in the expected order, but **the axis
they land on rotates with the spin axis**: the sinker's shows up as horizontal
(3.95" of the 5.01") and the sweeper's as vertical (2.79" of 4.14"). Magnitude
is comparable across classes; the IVB/HB decomposition is not.

## Finding 9 — neighbour contamination, and why the curveball gave it away

One row of the Finding 8 table does not fit. The curveball's model-free
cross-axis break is 0.5", near nothing, yet option 1 leaves a 2.64" residual.
Every other class's residual is smaller than its SEAM. This one is five times
larger, which means the model is inventing a deflection that is not there.

**It is a direction error, not a magnitude error.** Decomposing the residual in
the same release-axis frame SEAM uses (`xmove_cu_leftover.py`):

| RHP | total cross | model predicts | leftover cross | leftover along |
|---|---|---|---|---|
| CU | 0.5 | 2.7 | -2.22 | 0.91 |
| SI | 5.8 | 0.7 | 5.02 | 0.08 |
| ST | 7.6 | 3.8 | 3.84 | 1.25 |
| FF | -2.6 | -0.5 | -2.10 | 0.94 |

The along-axis part is under an inch. The model predicts 2.7" of perpendicular
deflection for a pitch that has 0.5". And the pattern across the whole table is
shrinkage toward the middle: it under-predicts the extremes (SI, ST) and
over-predicts everything near the centre (CU, FF). LHP replicates it.

**Velocity is not the cause.** The obvious story, that the curveball is simply
the slowest thing in a pool dominated by 95 mph fastballs, does not survive:
the all-types along-axis residual by velocity bin is flat and unordered (-0.37
to 0.54 with no monotone trend), and within pitch type CU and SL move in
*opposite* directions across overlapping velocity bands.

**It is sweeper contamination.** CU and ST overlap on every release input the
model can see. Tilt 202-242 against 223-265 at the 10th/90th, velocity 74.7-83.9
against 78.9-85.9, spin 2583 against 2602. Arm angle (43.9 against 31.7) is the
only input with real separation, and it is already a regressor. Yet their
cross-axis break differs by 7 inches. One pooled surface cannot separate them,
so it splits the difference. Binning curveballs by how sweeper-dense their
axis x velocity cell is (`xmove_cu_confusion.py`) gives a clean dose-response:

| sweeper share of cell | 1.8% | 5.9% | 19% | 39% | 54% |
|---|---|---|---|---|---|
| CU leftover cross | -0.43 | -1.49 | -2.56 | -3.40 | -4.25 |

**And it is irreducible.** If this were basis starvation, more flexibility would
kill it. It does not move at all from 22 parameters to 55:

| | H3 (22p) | H5 (34p) | H8 (52p) | H8 + v², s² (55p) |
|---|---|---|---|---|
| CU | -2.22 | -2.14 | -2.11 | -2.14 |
| SI | 5.02 | 5.00 | 4.99 | 4.86 |
| ST | 3.84 | 3.66 | 3.65 | 3.67 |

That is the important result in this finding, and it cuts both ways. Two pitches
with the same velocity, spin rate, release axis, arm angle and extension get
systematically different perpendicular deflection, and **no function of those
five inputs can separate them.** The difference lives in something not
measured: seam orientation at release, grip, gyro fraction. So the residual is
carrying real physics rather than the model's own misfit, which is what a seam
metric is supposed to do. Had more parameters killed it, the whole metric would
have been measuring its own inadequacy.

**Causally confirmed by targeted holdout.** Refit with one class removed from
training only, then score curveballs with it (`xmove_cu_holdout.py`):

| RHP training pool | CU leftover cross |
|---|---|
| full (shipped) | -2.22 |
| drop ST+SV (sweepers) | -0.76 |
| drop SL — control | -2.23 |
| drop FF+SI — control | -2.17 |
| drop ST+SV+SL | -0.01 |

Removing sweepers erases two-thirds. Removing sliders, which sit at a similar
place on the clock at similar velocity but carry ~0 cross-axis break, moves it
by 0.01". Removing the entire fastball population moves it by 0.05". LHP
replicates off an independent fit: -1.74 to -0.77 dropping sweepers, -1.60 on
the slider control.

Three caveats on that table. The -0.01" row is close to tautological, since with
sweepers and sliders both gone the curveball is nearly alone in its corner. The
reverse direction is messier: sweepers go 3.84 to 2.28 without curveballs, but
also to 2.60 without sliders, so sweeper contamination is shared rather than
curveball-specific. And this is a training-pool intervention, not a randomised
one; it establishes that the sweeper rows cause the offset, not that nothing
else contributes.

### How much of the displayed number this is

The class *mean* is already removed at the percentile stage, which centres per
pitch type. What survives is the spread, since the offset ranges -0.43 to -4.25
depending on neighbourhood. Measured at the rendered unit, pitcher x pitch type
x season with 50+ pitches, as the share of class-centred spread explained by
neighbour composition (`xmove_neighbour_contam.py`):

| | residual sd | share | SEAM sd | share |
|---|---|---|---|---|
| RHP FF | 1.62 | 0.248 | 1.27 | 0.073 |
| RHP SI | 2.20 | 0.240 | 1.65 | 0.062 |
| LHP FC | 1.94 | 0.212 | 1.48 | 0.001 |
| LHP ST | 1.73 | 0.204 | 1.47 | 0.001 |
| RHP CU | 2.80 | 0.127 | 2.58 | 0.005 |
| RHP SL | 2.56 | 0.004 | 3.24 | 0.007 |

**The curveball is not special.** Every class carries this and the four-seam and
sinker are worse. The curveball only stood out because its total cross-axis
break is near zero, so the offset was visible against nothing; FF and SI have
large real seam effects that were masking equally large contamination.

In inches, roughly 1.0" of a typical curveball's displayed deviation is
neighbourhood rather than pitcher (2.80 x sqrt(0.127)), 1.08" for the sinker,
0.81" for the four-seam. Against sds of 1.6-2.8" that is material and not fatal:
three-quarters of the displayed number is still the pitcher.

**SEAM is nearly immune**, 0.005 for the RHP curveball and 0.001 for both LHP
cases, because it has no model to contaminate. That cuts the artifact from about
1.0" to 0.18" on a curveball.

Two things to hold loosely. Changeups run the other way (RHP SEAM 0.166 against
residual 0.079, LHP 0.126 against 0.098); since SEAM is model-free, neighbour
composition cannot be corrupting it, so that is more likely a real confound,
plausibly velocity separation from the fastball, which would legitimately track
seam behaviour. And the "foreign share of cell" predictor treats a curveball
surrounded by sliders the same as one surrounded by sweepers when only the
latter should bias it, so these shares understate the contamination rather than
inflate it.

### What this does to the choice

Option 2 is immune to all of this by construction: fitting each class alone
means there are no foreign pitches in the training pool to contaminate it. So
each form now fails exactly one column, and SEAM fails none:

| | retag-proof | explains why | neighbour-clean |
|---|---|---|---|
| Option 1, agnostic | yes | yes | no, ~1"/pitch |
| Option 2, per class | no (Finding 6b) | no | yes |
| SEAM, model-free | yes | yes | yes |

SEAM's cost is that it answers a narrower question. It reports the perpendicular
deflection, not what *should* have happened, so it cannot by itself produce an
expected-movement ellipse. It dominates on robustness and loses on scope.

## Recommended design

### 1. The expected value: release axis + pitch-class seam term

This resolves the FF/SI tension directly. The model is already fit per (pitch
type, hand), so the pitch class's *typical* seam deflection lands in that
group's intercept for free. Concretely, for a pitcher whose FF and SI share a
release axis:

* xFF is built from his axis plus the four-seam class's −3.2" cross-axis term
* xSI is built from the same axis plus the sinker class's +5.6" term

so the two expected ellipses land ~9 inches apart even though RTilt is
identical — the "expected" is genuinely expected, not a restatement of what the
pitch did. What is left in the residual is *his* seam skill on top of what a
sinker normally gets, which is the Sonny Gray number worth showing.

If instead you want the pure-physics version — one pooled model, no pitch-type
term, so FF and SI share an expectation and the entire difference reads as SSW
— that is a one-line change (drop pitch type from the grouping key). It is a
different metric, not a better one, and the two should not be mixed on one
plot.

Findings 8 and 9 price that choice rather than settle it. The per-class form
absorbs the class seam effect by construction and is neighbour-clean, but its
expectation moves when a pitch is retagged. The agnostic form is retag-proof and
is the one that explains *why* a sinker sinks, but roughly an inch per pitch of
its displayed deviation is neighbourhood composition rather than the pitcher.
SEAM avoids both problems and answers a narrower question. The current
recommendation stands for an *expected-movement* display; if the goal is the
descriptive job of Finding 7, SEAM should be the number of record and the
expected-versus-actual arrows the illustration of it.

### 2. The residual: report it in the release-axis frame

Alongside IVBOE/HBOE (keep them; they are the Cartesian projection of the same
vector and existing users know them), expose the rotation:

* **`sswIn`** — `cross`, observed break perpendicular to the measured release
  axis, in inches. Descriptive: the FF/SI signature above.
* **`sswOE`** — `cross` minus its expectation for the pitch class. The skill
  version, and the one that belongs on a leaderboard, since raw `cross` is
  close to a restatement of pitch type.
* **`magOE`** — `along` residual. Reads as spin efficiency plus seam
  contribution along the axis; it cannot be split further without an active-spin
  measurement (see Limitations).

### 3. The plot: replace the fixed 7-inch hatched circle

`Cards.py:1668-1678` drew the expectation as a 7.0-inch hatched disc. The
radius was arbitrary and the disc reads as an error bar it never was. Better:

* a small open marker at (xHB, xIVB) per pitch type,
* an arrow from expected to actual — the OE vector, whose length and direction
  are the whole story in one mark,
* the ring, if kept at all, at ±1 SD of the *league distribution of
  pitcher-level OE for that pitch type* (measured per type, not a constant), so
  "outside the ring" means something falsifiable: roughly top or bottom 16%.

## Limitations to state on the page

* **Spin efficiency is not measured per pitch.** The feed gives total spin rate
  and the 2D release axis, not the transverse/gyro split. So `magOE` mixes
  spin efficiency with the along-axis part of the seam effect. `sswOE` does not
  have this problem: gyro spin scales break magnitude but does not rotate it,
  so a cross-axis residual is gyro-independent. If per-pitch active spin ever
  becomes available, `magOE` splits cleanly.
* **Pitch type is partly movement-derived, and the cost is now measured.**
  MLB's classifier uses movement, so conditioning on pitch type conditions
  weakly on the answer, and a re-tagged pitch moves its own baseline. Finding 6b
  puts a number on it: the shift is 1.06x to 3.35x the residual it is meant to
  measure, so a per-class residual is a peer comparison and not a physics claim.
  Finding 6a shows the label cannot simply be dropped either, because class is
  an interaction with the release parameters rather than an offset. Cross-axis
  break (Finding 7) is the one measurement here with no exposure to this.
* **Neighbour contamination in the agnostic form.** Where two classes share a
  release signature and differ in seam behaviour, one pooled surface averages
  them, and no amount of basis flexibility fixes it (Finding 9: 22 to 55
  parameters, no change). About a quarter of the class-centred spread in the
  four-seam and sinker residual, and an eighth of the curveball's, is neighbour
  composition rather than the pitcher. The per-class form and raw cross-axis
  break are both immune.
* **Sliders.** `SL` spans gyro sliders and sweepers, so its axis-deviation sd is
  40.6° — by far the widest class. The release-axis frame handles this (the
  decomposition is linear in IVB/HB, so small-break pitches do not blow up),
  but any *degrees*-based SSW display would be unusable for SL.
