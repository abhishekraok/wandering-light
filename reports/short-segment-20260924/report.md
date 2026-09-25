| arm | encoding | search | run | tasks | verified /43 per seed | mean | primitive executions (sum) | fuel (sum) | median exec / task | clock (s) | median s / task | tasks/h/lane | capped | held-out fails (tasks) | unverified solves (tasks) | budget spent |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| capped | difference | flat-w64 | short-segment-difference-20260923 | 300 | s0:13 | 13.00 | 38,592,336 | 43,122,538 | 1,191 | 373 | 0.03 | 2861 | 0.0 | 21.0 | 10.0 |  |
| capped | difference | segment-k16-d8 | short-segment-difference-20260923 | 300 | s0:14 | 14.00 | 816,426,371 | 879,303,035 | 378 | 8113 | 0.00 | 133 | 6.0 | 26.0 | 6.0 |  |
| capped | difference | segment-local-k64-d8 (exploratory) | short-segment-difference-20260923 | 300 | s0:13 | 13.00 | 1,354,736,908 | 1,476,125,173 | 457 | 13149 | 0.00 | 81 | 98.0 | 39.0 | 12.0 |  |
| capped | difference | flat-widening-k16-d8-h96-fuel | short-segment-difference-20260923 | 300 | s0:12 | 12.00 | 686,265,917 | 764,299,347 | 838 | 10863 | 0.00 | 99 | 0.0 | 21.0 | 10.0 | 143/300 ended by budget, 26 set by flat, 0.87 of fuel spent, overshoot 395 on 94, widest 512 |
| capped | difference | flat-widening-local-k64-d8-h96-fuel (exploratory) | short-segment-difference-20260923 | 300 | s0:14 | 14.00 | 1,066,676,232 | 1,183,139,532 | 458 | 15087 | 0.00 | 72 | 0.0 | 23.0 | 10.0 | 132/300 ended by budget, 96 set by flat, 0.80 of fuel spent, overshoot 576 on 96, widest 1024 |
| capped | machine | flat-w64 | short-segment-machine-20260923 | 300 | s0:13 | 13.00 | 38,756,817 | 43,087,457 | 792 | 337 | 0.00 | 3165 | 0.0 | 27.0 | 5.0 |  |
| capped | machine | segment-k16-d8 | short-segment-machine-20260923 | 300 | s0:14 | 14.00 | 829,475,763 | 891,957,422 | 378 | 7670 | 0.00 | 140 | 6.0 | 25.0 | 5.0 |  |
| capped | machine | flat-widening-k16-d8-h96-fuel | short-segment-machine-20260923 | 300 | s0:12 | 12.00 | 713,231,329 | 789,266,257 | 673 | 10662 | 0.00 | 101 | 0.0 | 34.0 | 5.0 | 117/300 ended by budget, 29 set by flat, 0.88 of fuel spent, overshoot 499 on 79, widest 512 |
| **staged** (BEST s0) | machine | flat-w64 | staged-arms-20260920 | 300 | s0:16 | 16.00 | 34,785,672 | 38,054,943 | 492 | 248 | 0.00 | 4307 | 0.0 | 12.0 | 4.0 |  |
| **staged** (BEST s0) | machine | segment-k16-d16 | staged-arms-20260920 | 300 | s0:15 | 15.00 | 1,438,015,529 | 1,538,328,174 | 425 | 10077 | 0.00 | 105 | 36.0 | 25.0 | 14.0 |  |
| **staged** (BEST s0) | machine | flat-widening-k16-d16-h192-fuel | staged-arms-20260920 | 300 | s0:17 | 17.00 | 1,021,731,975 | 1,208,743,986 | 396 | 19748 | 0.00 | 55 | 21.0 | 18.0 | 3.0 | 83/300 ended by budget, 43 set by flat, 0.78 of fuel spent, overshoot 1,556 on 61, widest 512 |

Equivalence check: `BEST` (staged s0 e3) under flat verifies 16; best-system-command (#158) measured 16. Count reproduced; 300 of 300 tasks identical.
Per task against #158's b64 cell: 300/300 with the same program and solved; 0 of BEST's tasks hit the 120 s cap (the whole cohort took 154 s in #158).

Executions, fuel, clock, capped and shortcut columns are means over seeds. Every row records seconds, executions and fuel. The clock is the sum of per-task seconds (library replay, search and policy inference). Budget spent is flat-widening only, per seed.

## Per checkpoint against flat-48

Solved within m x flat-48's own per-task cost, in fuel: graded tasks / verified behaviours.

| arm | encoding | run | seed | search | verified | 1x | 4x | 16x | 45x | gains over flat-48 | losses |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|
| capped | difference | short-segment-difference-20260923 | 0 | flat-w64 | 13 | 161 / 13 | 161 / 13 | 161 / 13 | 161 / 13 | - | - |
| capped | difference | short-segment-difference-20260923 | 0 | flat-widening-k16-d8-h96-fuel | 12 | 128 / 11 | 135 / 12 | 136 / 12 | 136 / 12 | shared_value_occurrences | reverse, rotate_left |
| capped | difference | short-segment-difference-20260923 | 0 | flat-widening-local-k64-d8-h96-fuel | 14 | 129 / 12 | 143 / 14 | 144 / 14 | 145 / 14 | prefix_sums, shared_value_occurrences | sum_negative |
| capped | difference | short-segment-difference-20260923 | 0 | segment-k16-d8 | 14 | 156 / 14 | 161 / 14 | 166 / 14 | 166 / 14 | all_positive | - |
| capped | difference | short-segment-difference-20260923 | 0 | segment-local-k64-d8 | 13 | 125 / 11 | 157 / 12 | 159 / 12 | 162 / 12 | - | - |
| capped | machine | short-segment-machine-20260923 | 0 | flat-w64 | 13 | 159 / 13 | 159 / 13 | 159 / 13 | 159 / 13 | - | - |
| capped | machine | short-segment-machine-20260923 | 0 | flat-widening-k16-d8-h96-fuel | 12 | 145 / 12 | 149 / 12 | 149 / 12 | 149 / 12 | - | keep_zeros |
| capped | machine | short-segment-machine-20260923 | 0 | segment-k16-d8 | 14 | 152 / 12 | 161 / 14 | 166 / 14 | 166 / 14 | all_positive | - |
| staged | machine | staged-arms-20260920 | 0 | flat-w64 | 16 | 180 / 16 | 180 / 16 | 180 / 16 | 180 / 16 | - | - |
| staged | machine | staged-arms-20260920 | 0 | flat-widening-k16-d16-h192-fuel | 17 | 165 / 16 | 174 / 17 | 175 / 17 | 178 / 17 | pair_floor_averages, prefix_sums | keep_zeros |
| staged | machine | staged-arms-20260920 | 0 | segment-k16-d16 | 15 | 148 / 11 | 177 / 15 | 182 / 15 | 183 / 15 | - | all_positive |

## Segment tasks the clock truncated

A segment run that ended at the clock cap is a loss the harness caused, not a search result.

- capped / difference segment-k16-d8 (short-segment-difference-20260923) s0: task 470 replace_value, 120 s, 9,323,579 executions
- capped / difference segment-k16-d8 (short-segment-difference-20260923) s0: task 121 prefix_sums, 120 s, 9,845,287 executions
- capped / difference segment-k16-d8 (short-segment-difference-20260923) s0: task 1343 keep_ascending_pairs, 120 s, 9,535,273 executions
- capped / difference segment-k16-d8 (short-segment-difference-20260923) s0: task 1206 furthest_squared_radius, 120 s, 9,211,203 executions
- capped / difference segment-k16-d8 (short-segment-difference-20260923) s0: task 1202 furthest_squared_radius, 120 s, 8,691,495 executions
- capped / difference segment-k16-d8 (short-segment-difference-20260923) s0: task 1172 pair_floor_averages, 120 s, 8,972,087 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1579 cartesian_pair_sums, 120 s, 9,361,969 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 709 capped_sum, 120 s, 14,204,300 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1581 cartesian_pair_sums, 120 s, 10,707,688 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1594 cartesian_pair_sums, 120 s, 9,981,389 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1573 cartesian_pair_sums, 120 s, 10,773,029 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1588 cartesian_pair_sums, 120 s, 11,783,616 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1577 cartesian_pair_sums, 120 s, 11,067,510 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1564 cartesian_pair_sums, 120 s, 10,238,720 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1596 cartesian_pair_sums, 120 s, 10,129,249 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 701 capped_sum, 120 s, 12,868,374 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 704 capped_sum, 120 s, 13,642,875 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1587 cartesian_pair_sums, 120 s, 13,937,299 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1563 cartesian_pair_sums, 120 s, 10,527,643 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1592 cartesian_pair_sums, 120 s, 10,900,673 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1580 cartesian_pair_sums, 120 s, 10,383,812 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 714 capped_sum, 120 s, 15,181,830 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 689 capped_sum, 120 s, 15,301,671 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 696 capped_sum, 120 s, 15,010,347 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 715 capped_sum, 120 s, 16,961,254 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1603 aligned_matches, 120 s, 13,415,488 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 479 replace_value, 120 s, 15,932,969 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 815 nearest_interval_endpoint_each, 120 s, 14,076,626 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 456 replace_value, 120 s, 16,807,590 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1019 concatenate_shorter_first, 120 s, 9,923,284 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 470 replace_value, 120 s, 17,550,893 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 128 prefix_sums, 120 s, 11,854,239 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1005 concatenate_shorter_first, 120 s, 10,050,959 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1604 aligned_matches, 120 s, 13,510,087 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 453 replace_value, 120 s, 16,470,610 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 810 nearest_interval_endpoint_each, 120 s, 12,395,549 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 838 nearest_interval_endpoint_each, 120 s, 12,403,637 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 464 replace_value, 120 s, 14,662,675 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1014 concatenate_shorter_first, 120 s, 9,495,168 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 772 shared_value_occurrences, 120 s, 13,772,889 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1023 concatenate_shorter_first, 120 s, 9,171,093 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1616 aligned_matches, 120 s, 13,829,953 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1724 largest_inside_interval, 120 s, 19,085,928 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 807 nearest_interval_endpoint_each, 120 s, 16,312,287 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1022 concatenate_shorter_first, 120 s, 10,164,773 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 468 replace_value, 120 s, 10,692,807 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1605 aligned_matches, 120 s, 12,804,357 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 794 shared_value_occurrences, 120 s, 15,308,243 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 459 replace_value, 120 s, 16,366,779 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 458 replace_value, 120 s, 14,634,791 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 820 nearest_interval_endpoint_each, 120 s, 16,511,506 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 824 nearest_interval_endpoint_each, 120 s, 13,894,329 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 143 prefix_sums, 120 s, 11,080,726 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 159 prefix_sums, 120 s, 11,215,257 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 440 replace_value, 120 s, 15,652,416 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1636 aligned_matches, 120 s, 9,737,413 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 121 prefix_sums, 120 s, 11,911,633 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 152 prefix_sums, 120 s, 9,938,118 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 776 shared_value_occurrences, 120 s, 11,507,099 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 831 nearest_interval_endpoint_each, 120 s, 11,493,301 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 764 shared_value_occurrences, 120 s, 11,245,381 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 120 prefix_sums, 120 s, 11,603,938 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1156 widest_pair, 120 s, 8,217,492 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1408 positive_balance_prefix_count, 120 s, 11,984,668 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1499 adjacent_sign_change_count, 120 s, 9,983,413 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1266 axis_point_count, 120 s, 8,625,406 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1466 median_floor, 120 s, 12,321,126 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1347 keep_ascending_pairs, 120 s, 10,182,431 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1159 widest_pair, 120 s, 11,620,810 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 897 farther_choice_from_target, 120 s, 19,557,172 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1772 farthest_from_target_last_tie, 120 s, 13,419,801 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 846 span_inside_interval, 120 s, 13,235,468 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1790 farthest_from_target_last_tie, 120 s, 12,387,783 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1367 polyline_manhattan_length, 120 s, 9,805,236 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1147 widest_pair, 120 s, 8,062,200 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 851 span_inside_interval, 120 s, 14,392,765 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1343 keep_ascending_pairs, 120 s, 10,321,476 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1268 axis_point_count, 120 s, 9,618,977 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1398 polyline_manhattan_length, 120 s, 9,852,651 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 610 second_largest, 120 s, 12,911,046 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1206 furthest_squared_radius, 120 s, 10,960,008 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1375 polyline_manhattan_length, 120 s, 9,714,614 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1314 manhattan_point_radii, 120 s, 8,552,362 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 623 second_largest, 120 s, 12,032,177 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 752 aligned_maxima, 120 s, 9,197,682 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 735 aligned_maxima, 120 s, 10,669,774 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1129 widest_pair, 120 s, 11,397,593 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1326 keep_ascending_pairs, 120 s, 8,075,836 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1476 median_floor, 120 s, 12,590,505 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1175 pair_floor_averages, 120 s, 8,201,632 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1787 farthest_from_target_last_tie, 120 s, 12,871,940 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1534 nondecreasing_indicator, 120 s, 9,720,168 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1202 furthest_squared_radius, 120 s, 10,399,655 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1132 widest_pair, 120 s, 10,656,196 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1396 polyline_manhattan_length, 120 s, 9,466,270 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 759 aligned_maxima, 120 s, 9,891,460 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 755 aligned_maxima, 120 s, 8,696,939 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1172 pair_floor_averages, 120 s, 9,580,745 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 758 aligned_maxima, 120 s, 11,451,035 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 625 second_largest, 120 s, 9,858,994 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1122 widest_pair, 120 s, 10,835,252 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1334 keep_ascending_pairs, 120 s, 7,551,778 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 909 farther_choice_from_target, 120 s, 17,941,110 executions
- capped / difference segment-local-k64-d8 (short-segment-difference-20260923) s0: task 1157 widest_pair, 120 s, 9,682,547 executions
- capped / machine segment-k16-d8 (short-segment-machine-20260923) s0: task 1592 cartesian_pair_sums, 120 s, 9,751,546 executions
- capped / machine segment-k16-d8 (short-segment-machine-20260923) s0: task 1343 keep_ascending_pairs, 120 s, 8,840,725 executions
- capped / machine segment-k16-d8 (short-segment-machine-20260923) s0: task 1206 furthest_squared_radius, 120 s, 7,581,219 executions
- capped / machine segment-k16-d8 (short-segment-machine-20260923) s0: task 1375 polyline_manhattan_length, 120 s, 7,518,832 executions
- capped / machine segment-k16-d8 (short-segment-machine-20260923) s0: task 1202 furthest_squared_radius, 120 s, 8,246,326 executions
- capped / machine segment-k16-d8 (short-segment-machine-20260923) s0: task 1132 widest_pair, 120 s, 9,738,458 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1579 cartesian_pair_sums, 120 s, 8,906,874 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1581 cartesian_pair_sums, 120 s, 10,352,331 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1594 cartesian_pair_sums, 120 s, 11,814,799 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1573 cartesian_pair_sums, 120 s, 12,502,665 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1588 cartesian_pair_sums, 120 s, 11,756,347 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1577 cartesian_pair_sums, 120 s, 11,797,981 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1564 cartesian_pair_sums, 120 s, 10,986,470 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1596 cartesian_pair_sums, 120 s, 10,770,903 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1587 cartesian_pair_sums, 120 s, 15,368,778 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1592 cartesian_pair_sums, 120 s, 11,871,749 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1580 cartesian_pair_sums, 120 s, 11,771,468 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1019 concatenate_shorter_first, 120 s, 12,354,963 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 470 replace_value, 120 s, 19,780,049 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 128 prefix_sums, 120 s, 12,335,038 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1005 concatenate_shorter_first, 120 s, 18,136,055 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 807 nearest_interval_endpoint_each, 120 s, 18,801,894 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1022 concatenate_shorter_first, 120 s, 19,229,600 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 159 prefix_sums, 120 s, 10,442,024 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 440 replace_value, 120 s, 14,131,785 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 121 prefix_sums, 120 s, 13,297,916 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 120 prefix_sums, 120 s, 14,013,583 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1772 farthest_from_target_last_tie, 120 s, 17,870,163 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1790 farthest_from_target_last_tie, 120 s, 17,081,795 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1367 polyline_manhattan_length, 120 s, 14,449,918 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1343 keep_ascending_pairs, 120 s, 13,115,222 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 610 second_largest, 120 s, 18,898,499 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1206 furthest_squared_radius, 120 s, 13,271,032 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1314 manhattan_point_radii, 120 s, 10,810,582 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 752 aligned_maxima, 120 s, 11,067,728 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1326 keep_ascending_pairs, 120 s, 10,437,933 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1202 furthest_squared_radius, 120 s, 15,096,424 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1132 widest_pair, 120 s, 19,265,344 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1172 pair_floor_averages, 120 s, 12,211,330 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 758 aligned_maxima, 120 s, 18,191,031 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 1334 keep_ascending_pairs, 120 s, 9,125,688 executions
- staged / machine segment-k16-d16 (staged-arms-20260920) s0: task 909 farther_choice_from_target, 120 s, 21,551,639 executions

## The capped tasks without the clock

Each segment cell's `wall_clock_cap` tasks rerun alone, in cohort order, under the rerun's clock (GPU, 1 thread; the library starts empty).

- capped / difference segment-k16-d8 s0: 6/6 rerun at 1800 s, 6 max_segments; median 37 s; 0 solved; verified 14 -> 14
- capped / difference segment-local-k64-d8 s0: 98/98 rerun at 1800 s in 3 parts, 5 fit, 93 max_segments; median 114 s; 0 solved; verified 13 -> 13
- capped / machine segment-k16-d8 s0: 6/6 rerun at 1800 s, 6 max_segments; median 46 s; 0 solved; verified 14 -> 14
- staged / machine segment-k16-d16 s0: 36/36 rerun at 1800 s, 36 max_segments; median 66 s; 0 solved; verified 15 -> 15

## Rules, one line each

- **Executions:** one per op applied to one shown example stack, counted at the same place (before each op attempt) in flat, widening and segment, library replay included; the registered "primitive executions" column.
- **Fuel:** the same applications' machine cost, 1 per op plus every quotation-body step, so fuel - executions is body steps.
- **Widening budget unit:** each cell's `budget_unit`, `fuel` by default.
- **Widening budget:** per task, max(the segment row's spend, flat-48's own spend on that task) in that unit; `budget_term` names the larger, and `set by flat` counts the tasks flat-48 set.
- **Widening stop:** widths 64, 128, 256, ... from scratch on one shared counter, until the first shown fit, the budget, the clock, or a restart whose width never turned a successor away (`saturated`: every wider beam would run the same search).
- **Overshoot:** a restart that passes its budget mid-attempt is charged in full and reported as `overshoot`.
- **Clock:** 120 s per task for flat-48 and segment, 300 s for flat-widening, recorded per cell; `capped` counts `wall_clock_cap` rows per column.
- **Fit check:** every executable successor is tested with `moved == outputs` as it is made, before ranking or pruning (in `_run_segment` also before the open/closed split); held-out examples are read only at grading.
- **Stop at fit:** all three searches return at the first shown fit (the segment beam through `stop_at_fit=True`), so a row's counters with a program are its counters at the fit.
- **Merge rule:** equal closed states (structurally equal products) merge across roots, keeping the highest cumulative log-probability and its program (a tie keeps the first merged); the top K unexpanded ones by that score are the next roots.
- **Local scoring (exploratory, capped/difference s0 only):** at each cut the next roots are ranked by what they gained in the segment just finished and restart at 0, and equal closed states keep the best gain; K = 64, depth 8, width 64, 12 segments, 120 s, no budget, with its own flat-widening control budgeted on its fuel.
- **Frozen per cell:** K = 16, segment depth {'capped': 8, 'staged': 16, 'fused': 48} by arm, 12 segments.
- **Horizon:** flat-48 is `BEST`'s search at depth 48; flat-widening runs at the segment cell's horizon, 12 x its depth.
- **Order and library:** every column runs the #101 cohort in the same order (`co.cohort`, seed 17, bucket order), and each column's library starts empty and holds only its own solves (`solve.solve_tasks`).
- **Why the flat-48 floor:** on increment_each segment-reset fitted at 10,705 executions against about 10,750 for flat width 64, a hair cheaper; the flat-48 floor is why the control still solves it.
- **Matched computation, a limit:** a combinator that fails on an example drops the body cost it had already run there (inherited from `step_all_quiet`), so fuel is the recorded counter, not every step attempted.
- **K diagnostic:** the 0.46 and 0.27 local K figures were computed on cumulatively chosen roots; the local-scoring diagnostic (`segment_k_diagnostic.py --scoring local`) replaces them.
- **Bearing, not a result:** the 6-task CPU smoke of 2026-09-24 ran `BEST` at the old segment depth 12: flat-48 5/6 at 613k executions, segment 4/6 at 27.5M (45x).

