## Validation cross-entropy on the pinned v4 holdout (26,040 rows, the same rows in both arms)

| seed | v4-only | best epoch | v4+world | best epoch | delta |
|---|---|---|---|---|---|
| 0 | 0.9519 | 1 | 0.8434 | 2 | -0.1085 |
| 1 | 0.9454 | 2 | 0.8689 | 4 | -0.0765 |
| 2 | 0.9650 | 1 | 0.8606 | 3 | -0.1044 |
| **mean ± sd** | 0.9541 ± 0.0100 | | 0.8576 ± 0.0130 | | -0.0965 ± 0.0174 |

A delta is only worth reading against the seed spread in the same row.

## Verified solves, v4 eval cohort (300 tasks, beam 64 x depth 48)

A graded solve passes the task's held-out examples. A verified solve also computes the behaviour: its program is a catalogue witness, or it agrees with the Python reference on 2,000 sampler draws and 2,000 perturbed draws. Cells read verified / graded.

| bucket | v4-only s0 | v4-only s1 | v4-only s2 | v4+world s0 | v4+world s1 | v4+world s2 |
|---|---|---|---|---|---|---|
| 1-5 | 32/32 | 41/41 | 31/31 | 32/32 | 60/60 | 51/51 |
| 6-10 | 0/0 | 6/6 | 0/0 | 8/8 | 28/28 | 33/33 |
| 11-20 | 16/16 | 16/16 | 15/17 | 11/12 | 10/10 | 13/14 |
| 21-40 | 0/0 | 0/0 | 0/0 | 1/1 | 6/6 | 5/5 |
| 41+ | 0/0 | 0/0 | 0/0 | 0/0 | 0/1 | 0/0 |
| **all** | 48/48 | 63/63 | 46/48 | 52/53 | 104/105 | 102/103 |

Programs that pass held-out grading without computing the behaviour:

- `all_positive`, 28 ops: fails 96/2000 sampler draws and 339/2000 perturbed; e.g. {'args': [[6, 4]], 'reference': 1}
- `bounds_cover_list_flag`, 44 ops: fails 498/2000 sampler draws and 229/1827 perturbed; e.g. {'args': [[7, -3, 8, -5, -9], 6, 7], 'reference': 0}

## Solve rate, v4 eval cohort (300 tasks, beam 64 x depth 48)

| bucket | v4-only s0 | v4-only s1 | v4-only s2 | v4+world s0 | v4+world s1 | v4+world s2 |
|---|---|---|---|---|---|---|
| 1-5 | 32/60 | 41/60 | 31/60 | 32/60 | 60/60 | 51/60 |
| 6-10 | 0/60 | 6/60 | 0/60 | 8/60 | 28/60 | 33/60 |
| 11-20 | 16/60 | 16/60 | 17/60 | 12/60 | 10/60 | 14/60 |
| 21-40 | 0/60 | 0/60 | 0/60 | 1/60 | 6/60 | 5/60 |
| 41+ | 0/60 | 0/60 | 0/60 | 0/60 | 1/60 | 0/60 |
| **all scored** | 48/300 | 63/300 | 48/300 | 53/300 | 105/300 | 103/300 |

## Solve rate, world eval (55 tasks, same search)

Scored on the 44 clean behaviours; the 11 whose program or skeleton already appears in a training set are counted in the last row only.

| bucket | v4-only s0 | v4-only s1 | v4-only s2 | v4+world s0 | v4+world s1 | v4+world s2 |
|---|---|---|---|---|---|---|
| 1-5 | - | - | - | - | - | - |
| 6-10 | 0/1 | 0/1 | 0/1 | 1/1 | 0/1 | 0/1 |
| 11-20 | - | - | - | - | - | - |
| 21-40 | 1/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| 41+ | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| **all scored** | 1/44 | 0/44 | 0/44 | 1/44 | 0/44 | 0/44 |
| excluded (seen in training) | 2/11 | 2/11 | 3/11 | 2/11 | 2/11 | 3/11 |

## Cross-entropy on held-out row sets (experiments/step3_ce.py)

| checkpoint | v4-fit | v4-holdout | world-eval-all | world-eval-clean | world-eval-dirty | world-holdout |
|---|---|---|---|---|---|---|
| mixed/policies/1m-s0 | 0.3885 | 0.8434 | 1.0249 | 1.0782 | 0.6517 | 1.1742 |
| mixed/policies/1m-s1 | 0.2296 | 0.8689 | 0.9141 | 0.9786 | 0.4631 | 1.1111 |
| mixed/policies/1m-s2 | 0.2982 | 0.8606 | 0.9498 | 1.0062 | 0.5546 | 1.1394 |
| v4only/policies/1m-s0 | 0.5484 | 0.9519 | 2.7462 | 2.7726 | 2.5614 | 2.6724 |
| v4only/policies/1m-s1 | 0.3365 | 0.9454 | 3.8145 | 3.8636 | 3.4705 | 3.7003 |
| v4only/policies/1m-s2 | 0.5554 | 0.9650 | 2.8202 | 2.8467 | 2.6345 | 2.6847 |

Row-set sizes: v4-fit 6360 tasks / 313560 rows, v4-holdout 640 tasks / 26040 rows, world-eval-all 55 tasks / 11305 rows, world-eval-clean 44 tasks / 9892 rows, world-eval-dirty 11 tasks / 1413 rows, world-holdout 46 tasks / 11312 rows

