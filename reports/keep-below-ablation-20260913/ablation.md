## Fit rows per arm (batch 256; validation is the pinned v4 holdout)

| arm | fit rows | seeds |
|---|---|---|
| v4only | 313560 | 0, 1, 2 |
| mixed | 429078 | 0, 1, 2 |
| ablated | 358086 | 0, 1, 2 |
| random | 389527 | 0 |

## Validation cross-entropy by epoch

| epoch | v4only s0 | v4only s1 | v4only s2 | mixed s0 | mixed s1 | mixed s2 | ablated s0 | ablated s1 | ablated s2 | random s0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.9519 | 0.9651 | 0.9650 | 0.8983 | 0.9518 | 0.9195 | 0.9117 | 0.9474 | 0.9216 | 0.9140 |
| 2 | 0.9556 | 0.9454 | 0.9974 | 0.8434 | 0.9003 | 0.8802 | 0.8716 | 0.8657 | 0.8756 | 0.9135 |
| 3 | 1.0939 | 1.0607 | 1.0991 | 1.0010 | 0.8764 | 0.8606 | 0.9002 | 0.9290 | 0.9122 | 0.9589 |
| 4 | 1.2100 | 1.1870 | 1.2345 | 0.9827 | 0.8689 | 0.8725 | 0.9524 | 0.9339 | 0.9557 | 0.9959 |
| 5 | 1.3618 | 1.2568 | 1.2857 | 0.9354 | 0.8793 | 0.9011 | 1.0349 | 1.0112 | 0.9964 | 1.0463 |
| 6 | 1.5090 | 1.3577 | 1.4123 | 1.0123 | 0.9312 | 0.9361 | 1.0903 | 1.0323 | 1.0236 | 1.0911 |

Best epoch per seed (minimum validation CE), and the paired difference against the reference arm on the same seed.

| arm | best CE per seed | mean ± sd | minus mixed |
|---|---|---|---|
| v4only | 0.9519 (e1), 0.9454 (e2), 0.9650 (e1) | 0.9541 ± 0.0100 | +0.1085, +0.0765, +0.1044 |
| mixed | 0.8434 (e2), 0.8689 (e4), 0.8606 (e3) | 0.8576 ± 0.0130 | - |
| ablated | 0.8716 (e2), 0.8657 (e2), 0.8756 (e2) | 0.8710 ± 0.0050 | +0.0282, -0.0032, +0.0151 |
| random | 0.9135 (e2) | 0.9135 | +0.0701 |

## Verified solves by epoch (v4 cohort, beam 64 x depth 48; cells read all / `keep_below`)

| epoch | v4only s0 | v4only s1 | v4only s2 | mixed s0 | mixed s1 | mixed s2 | ablated s0 | ablated s1 | ablated s2 | random s0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 48/0* | 56/0 | 46/0* | 53/7 | 43/0 | 57/11 | 48/0 | 55/0 | 47/0 | 50/3 |
| 2 | 55/0 | 63/0* | 53/0 | 52/7* | 64/15 | 82/15 | 57/0 | 64/0 | 65/0 | 67/17 |
| 3 | 84/0 | 77/0 | 80/0 | 89/11 | 89/19 | 102/16* | 58/0 | 73/1 | 82/0 | 77/16 |
| 4 | 94/0 | 74/0 | 88/0 | 85/7 | 104/8* | 94/16 | 80/0 | 90/1 | 78/1 | 86/17 |
| 5 | 83/0 | 77/0 | 90/0 | 104/8 | 109/19 | 90/8 | 75/0 | 100/2 | 83/1 | 102/19 |
| 6 | 86/0 | 72/0 | 75/0 | 75/1 | 70/7 | 80/11 | 61/0 | 94/1 | 78/0 | 108/15 |

`*`: cell taken from the original best-checkpoint run.

## Matched-epoch paired difference, v4only minus mixed

Columns: all verified solves, then with `keep_below` removed.

| epoch | s0 | s1 | s2 | mean | other s0 | other s1 | other s2 | other mean |
|---|---|---|---|---|---|---|---|---|
| 1 | -5 | +13 | -11 | -1.0 | +2 | +13 | +0 | +5.0 |
| 2 | +3 | -1 | -29 | -9.0 | +10 | +14 | -14 | +3.3 |
| 3 | -5 | -12 | -22 | -13.0 | +6 | +7 | -6 | +2.3 |
| 4 | +9 | -30 | -6 | -9.0 | +16 | -22 | +10 | +1.3 |
| 5 | -21 | -32 | +0 | -17.7 | -13 | -13 | +8 | -6.0 |
| 6 | +11 | +2 | -5 | +2.7 | +12 | +9 | +6 | +9.0 |

## Matched-epoch paired difference, ablated minus mixed

Columns: all verified solves, then with `keep_below` removed.

| epoch | s0 | s1 | s2 | mean | other s0 | other s1 | other s2 | other mean |
|---|---|---|---|---|---|---|---|---|
| 1 | -5 | +12 | -10 | -1.0 | +2 | +12 | +1 | +5.0 |
| 2 | +5 | +0 | -17 | -4.0 | +12 | +15 | -2 | +8.3 |
| 3 | -31 | -16 | -20 | -22.3 | -20 | +2 | -4 | -7.3 |
| 4 | -5 | -14 | -16 | -11.7 | +2 | -7 | -1 | -2.0 |
| 5 | -29 | -9 | -7 | -15.0 | -21 | +8 | +0 | -4.3 |
| 6 | -14 | +24 | -2 | +2.7 | -13 | +30 | +9 | +8.7 |

## Matched-epoch paired difference, random minus mixed

Columns: all verified solves, then with `keep_below` removed.

| epoch | s0 | mean | other s0 | other mean |
|---|---|---|---|---|
| 1 | -3 | -3.0 | +1 | +1.0 |
| 2 | +15 | +15.0 | +5 | +5.0 |
| 3 | -12 | -12.0 | -17 | -17.0 |
| 4 | +1 | +1.0 | -9 | -9.0 |
| 5 | -2 | -2.0 | -13 | -13.0 |
| 6 | +33 | +33.0 | +19 | +19.0 |

`keep_below` verified solves summed over searched cells: v4only 0, mixed 186, ablated 7, random 87.

