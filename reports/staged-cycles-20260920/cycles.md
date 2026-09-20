# staged-cycles: one cycle to the next closed state as target

Beam 64, depth = cycle length + 4, guard `both`, policy epoch 3. A transition is solved when the returned program replays the start stacks onto the target stacks on every shown example. Shares are the mean over seeds.

**Read the buckets, not the overall share.** `beam_solve` tests the hit while it expands, before the policy ranks anything, so every one-op transition is solved by exhaustive one-step search whatever the policy says. The two witness kinds have very different shares of one-op cycles, so the overall column mixes that in; `2+` is the same share over the cycles the policy actually has to write.

| witness | seeds | witnesses | transitions | solved | share | 2+ | 1 | 2-5 | 6-12 | 13+ | whole witnesses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fused | 3 | 70 | 500 | 439.0 | 0.878 | 0.445 | 1.000 | 0.867 | 0.756 | 0.308 | 0.152 |
| staged | 3 | 70 | 750 | 723.7 | 0.965 | 0.914 | 1.000 | 0.997 | 0.856 | 1.000 | 0.795 |

Per cell:

| arm | seed | transitions | solved | share | 2+ | whole witnesses | wall (s) | executions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fused-subgoal` | 0 | 500 | 446 | 0.892 | 0.509 | 0.229 | 56 | 19,869,863 |
| `fused-subgoal` | 1 | 500 | 433 | 0.866 | 0.391 | 0.114 | 57 | 20,505,538 |
| `fused-subgoal` | 2 | 500 | 438 | 0.876 | 0.436 | 0.114 | 57 | 20,825,189 |
| `staged-subgoal` | 0 | 750 | 726 | 0.968 | 0.921 | 0.829 | 29 | 11,077,912 |
| `staged-subgoal` | 1 | 750 | 720 | 0.960 | 0.902 | 0.771 | 31 | 11,451,603 |
| `staged-subgoal` | 2 | 750 | 725 | 0.967 | 0.918 | 0.786 | 31 | 11,169,134 |

By behaviour, transitions and solved share (all cycles, then 2+ ops):

| behaviour | fused n / share / 2+ | staged n / share / 2+ |
|---|---:|---:|
| `choose_larger_total` | 5 / 0.733 / 0.733 | 50 / 1.000 / 1.000 |
| `gap_between_intervals` | 20 / 0.850 / 0.800 | 35 / 1.000 / 1.000 |
| `interval_region_codes` | 5 / 0.000 / 0.000 | 45 / 0.763 / 0.695 |
| `inversion_count` | 40 / 0.875 / 0.000 | 40 / 1.000 / 1.000 |
| `largest_adjacent_sum` | 85 / 0.922 / 0.556 | 45 / 0.778 / 0.600 |
| `magnitude_ordered_pair` | 5 / 0.000 / 0.000 | 75 / 1.000 / 1.000 |
| `maximum_drawdown` | 60 / 0.928 / 0.133 | 55 / 1.000 / 1.000 |
| `ordered_pair` | 5 / 0.000 / 0.000 | 35 / 1.000 / 1.000 |
| `pair_gap_total` | 55 / 0.909 / 0.000 | 105 / 1.000 / 1.000 |
| `positive_majority` | 55 / 0.909 / 0.667 | 65 / 1.000 / 1.000 |
| `running_mean_floor` | 80 / 0.938 / 0.500 | 85 / 0.937 / 0.787 |
| `sum_absolute_values` | 10 / 0.800 / 0.600 | 50 / 0.993 / 0.987 |
| `zero_balance_prefix_count` | 55 / 0.909 / 0.000 | 45 / 1.000 / 1.000 |
| `zero_base_by_minimum` | 20 / 0.817 / 0.633 | 20 / 1.000 / 1.000 |
