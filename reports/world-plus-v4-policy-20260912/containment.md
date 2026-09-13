## Verified solves by witness containment (v4 cohort, beam 64 x depth 48)

Contained: a behaviour's eval witness occurs, as whole ops, inside at least one world-v3 fit program and inside no v4 fit program (fit = train split, minus the drop list, minus validation skeletons). 5 of the cohort's 43 behaviours are contained; fit programs: v4 159, world 473.

| contained behaviour | witness ops | world fit programs containing it |
|---|---|---|
| `drop_last` | 12 | 1 |
| `increment_each` | 5 | 44 |
| `keep_below` | 6 | 156 |
| `n_minus_each` | 7 | 2 |
| `negate_each` | 6 | 3 |

Solves summed over seeds; the last column is the other behaviours' paired difference per seed.

| epoch | contained v4-only | contained v4+world | difference | other v4-only | other v4+world | difference | other, per seed |
|---|---|---|---|---|---|---|---|
| 1 | 26 | 38 | +12 | 124 | 115 | -9 | +0, -9, +0 (mean -3.0) |
| 2 | 26 | 58 | +32 | 145 | 140 | -5 | -10, -4, +9 (mean -1.7) |
| 3 | 59 | 113 | +54 | 182 | 167 | -15 | -5, -16, +6 (mean -5.0) |
| 4 | 65 | 91 | +26 | 191 | 192 | +1 | -8, +5, +4 (mean +0.3) |
| 5 | 60 | 110 | +50 | 190 | 193 | +3 | -2, +1, +4 (mean +1.0) |
| 6 | 51 | 60 | +9 | 182 | 165 | -17 | -11, -7, +1 (mean -5.7) |
| each seed's peak | 67 | 125 | +58 | 194 | 190 | -4 | -5, +0, +1 (mean -1.3) |

