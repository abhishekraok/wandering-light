| arm | graded tasks | verified tasks | verified behaviours /43 | union over seeds | 1-5 (beh.) | 6-10 (beh.) | 11-20 (beh.) | 21-40 (beh.) | 41+ (beh.) | longest verified program |
|---|---|---|---|---|---|---|---|---|---|---|
| fused | 86.00 (88 / 81 / 89) | 83.67 (86 / 79 / 86) | 12.33 (13 / 11 / 13) | 15 | 6.00 | 1.00 | 3.67 | 1.33 | 0.33 | 40 ops |
| refused | 90.33 (86 / 90 / 95) | 87.00 (83 / 86 / 92) | 12.67 (12 / 13 / 13) | 15 | 4.67 | 1.33 | 4.00 | 2.33 | 0.33 | 38 ops |
| staged | 139.67 (143 / 128 / 148) | 137.67 (139 / 127 / 147) | 15.67 (16 / 16 / 15) | 16 | 6.00 | 3.00 | 5.00 | 1.67 | 0.00 | 44 ops |
| both | 130.33 (121 / 147 / 123) | 128.00 (120 / 145 / 119) | 14.00 (14 / 15 / 13) | 15 | 6.00 | 3.00 | 4.67 | 0.33 | 0.00 | 26 ops |

Behaviours verified by some seed of an arm and by no seed of `fused`, and the reverse:

- `refused`: gains ['largest_inside_interval', 'n_minus_each', 'pair_floor_averages', 'prefix_sums']; loses ['capped_sum', 'farther_choice_from_target', 'keep_negative', 'shared_value_occurrences']
- `staged`: gains ['keep_below', 'largest_inside_interval', 'n_minus_each']; loses ['farther_choice_from_target', 'shared_value_occurrences']
- `both`: gains ['keep_below', 'largest_inside_interval', 'n_minus_each']; loses ['count_in_range', 'farther_choice_from_target', 'shared_value_occurrences']
