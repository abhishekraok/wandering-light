## Solves by epoch on the 300-task v4 cohort (verified / graded; beam 64 x depth 48)

Each cell: solves, then validation CE on the pinned v4 holdout at that epoch. `*` marks a cell reused from the best-checkpoint run.

| epoch | v4-only s0 | v4-only s1 | v4-only s2 | v4+world s0 | v4+world s1 | v4+world s2 |
|---|---|---|---|---|---|---|
| 1 | 48/48* (0.952) | 56/56 (0.965) | 46/48* (0.965) | 53/53 (0.898) | 43/43 (0.952) | 57/59 (0.920) |
| 2 | 55/55 (0.956) | 63/63* (0.945) | 53/54 (0.997) | 52/53* (0.843) | 64/66 (0.900) | 82/82 (0.880) |
| 3 | 84/85 (1.094) | 77/77 (1.061) | 80/80 (1.099) | 89/89 (1.001) | 89/89 (0.876) | 102/103* (0.861) |
| 4 | 94/94 (1.210) | 74/74 (1.187) | 88/89 (1.234) | 85/86 (0.983) | 104/105* (0.869) | 94/97 (0.872) |
| 5 | 83/84 (1.362) | 77/80 (1.257) | 90/92 (1.286) | 104/109 (0.935) | 109/111 (0.879) | 90/91 (0.901) |
| 6 | 86/91 (1.509) | 72/72 (1.358) | 75/76 (1.412) | 75/77 (1.012) | 70/70 (0.931) | 80/82 (0.936) |

## Matched-epoch paired difference in verified solves (v4+world minus v4-only)

| epoch | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| 1 | +5 | -13 | +11 | +1.0 |
| 2 | -3 | +1 | +29 | +9.0 |
| 3 | +5 | +12 | +22 | +13.0 |
| 4 | -9 | +30 | +6 | +9.0 |
| 5 | +21 | +32 | +0 | +17.7 |
| 6 | -11 | -2 | +5 | -2.7 |

Steps per epoch: v4-only 1225, v4+world 1677 (batch 256), a ratio of 1.37.

## Matched-step paired difference in verified solves (v4+world minus v4-only)

Each v4+world epoch against the v4-only epoch nearest in optimizer steps, kept only within half a v4-only epoch. The gap is v4-only steps minus v4+world steps.

| v4+world epoch | v4-only epoch | step gap | s0 | s1 | s2 | mean |
|---|---|---|---|---|---|---|
| 1 | 1 | -452 | +5 | -13 | +11 | +1.0 |
| 2 | 3 | +321 | -32 | -13 | +2 | -14.3 |
| 3 | 4 | -131 | -5 | +15 | +14 | +8.0 |
| 4 | 5 | -583 | +2 | +27 | +4 | +11.0 |

## Per-seed summaries over searched epochs (verified solves)

Peak is the best searched epoch chosen by solves on this cohort, so it is an upper bound for each arm, not a selection rule. Mean over epochs is shown only when both arms were searched at every epoch.

| summary | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| v4-only peak (epoch) | 94 (4) | 77 (3) | 90 (5) | 87.0 |
| v4+world peak (epoch) | 104 (5) | 109 (5) | 102 (3) | 105.0 |
| difference | +10 | +32 | +12 | +18.0 |
| v4-only mean over epochs 1-6 | 75.0 | 69.8 | 72.0 | 72.3 |
| v4+world mean over epochs 1-6 | 76.3 | 79.8 | 84.2 | 80.1 |
| difference of means | +1.3 | +10.0 | +12.2 | +7.8 |

