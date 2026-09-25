## world arm 1m-s0-e5, #101 cohort, 10-draw gate

5,127 closed cross-path meetings in 288 tasks, 10 fresh draws each.

| pairs | n | pass | fail | inconclusive | pass rate of judged |
|---|---:|---:|---:|---:|---:|
| all | 5,127 | 4,440 | 686 | 1 | **86.6%** |
| same-layer | 1,172 | 1,086 | 86 | 0 | **92.7%** |
| earlier-layer | 3,955 | 3,354 | 600 | 1 | **84.8%** |
| same-layer/commutation | 159 | 148 | 11 | 0 | **93.1%** |
| same-layer/other | 1,013 | 938 | 75 | 0 | **92.6%** |
| earlier-layer/insertion | 1,317 | 987 | 329 | 1 | **75.0%** |
| earlier-layer/other | 2,638 | 2,367 | 271 | 0 | **89.7%** |

- passing earlier-layer pairs: the shorter route saves a median of 2.0 ops; distribution 1: 1,564, 2: 462, 3: 124, 4: 20, 5: 295, 6: 35, 7: 48, 8: 27, 9: 6, 10: 55, 11: 39, 12: 28, 13: 52, 14: 10, 15: 25, 16: 25, 17: 47, 18: 44, 19: 21, 20: 19, 21: 13, 22: 6, 23: 16, 24: 31, 25: 5, 26: 19, 27: 22, 28: 2, 29: 33, 30: 9, 31: 16, 32: 3, 33: 8, 34: 7, 35: 2, 36: 24, 37: 5, 38: 11, 39: 2, 40: 1, 41: 12, 42: 9, 43: 3, 44: 21, 45: 5, 46: 86, 47: 37
- distinct next-op labels among them (task, divergence prefix, preferred op, dispreferred op): 2,767 in 288 tasks
- failures by behaviour: sum_negative 102, capped_sum 57, n_minus_each 55, widest_pair 50, keep_below 43, positive_majority_three 38, all_positive 30, replace_value 24, cartesian_pair_sums 22, aligned_matches 22, drop_last 21, polyline_manhattan_length 21, largest_inside_interval 19, aligned_maxima 18, adjacent_sign_change_count 17, nondecreasing_indicator 17, shared_value_occurrences 16, median_floor 12, second_largest 12, increment_each 10, nearest_interval_endpoint_each 10, positive_balance_prefix_count 10, concatenate_shorter_first 9, prefix_sums 9, farther_choice_from_target 8, bounds_cover_list_flag 5, axis_point_count 4, keep_ascending_pairs 4, lexicographic_order 4, farthest_from_target_last_tie 4, negate_each 3, furthest_squared_radius 3, min_pair 2, manhattan_point_radii 2, keep_negative 1, span_inside_interval 1, pair_floor_averages 1

Failing pairs (first / second / the input that parts them):

- `increment_each` (earlier-layer, insertion): `uncons drop` vs `uncons [ [ over 0 eq? ] [ swap drop ] [ drop ] ifte ] swap cons map` on `Lst(items=(Int(value=8), Int(value=0), Int(value=6), Int(value=7), Int(value=2), Int(value=-2)))`
- `increment_each` (earlier-layer, other): `nil [ cons ] fold uncons drop` vs `uncons [ [ over lt? ] [ drop ] [ swap drop ] ifte ] swap cons map` on `Lst(items=(Int(value=6), Int(value=7), Int(value=5), Int(value=-6)))`
- `increment_each` (earlier-layer, insertion): `nil [ cons ] fold uncons drop` vs `nil [ cons ] fold uncons [ [ over lt? ] [ drop ] [ swap drop ] ifte ] swap cons map` on `Lst(items=(Int(value=3), Int(value=-7), Int(value=-4), Int(value=1), Int(value=6)))`
- `increment_each` (earlier-layer, insertion): `uncons [ 0 [ swap drop ] fold ] dip` vs `nil [ cons ] fold uncons [ 0 [ swap drop ] fold ] dip [ over lt? ] [ swap ] [ ] ifte` on `Lst(items=(Int(value=-9), Int(value=-2), Int(value=-1), Int(value=7), Int(value=-8), Int(value=7)))`
- `increment_each` (earlier-layer, insertion): `uncons [ [ over lt? ] [ drop ] [ swap drop ] ifte ] fold` vs `uncons [ [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte [ over lt? ] [ drop ] [ swap drop ] ifte ] fold` on `Lst(items=(Int(value=-3), Int(value=-4)))`
- `increment_each` (earlier-layer, other): `uncons drop 0 [ add ] fold` vs `0 [ swap drop ] fold [ dup 0 lt? ] [ 0 swap sub 1 sub ] [ ] ifte` on `Lst(items=(Int(value=-5), Int(value=-4), Int(value=0), Int(value=-6), Int(value=3), Int(value=-8)))`
- `increment_each` (earlier-layer, insertion): `uncons drop` vs `uncons [ [ over lt? ] [ drop ] [ swap drop ] ifte ] swap cons map` on `Lst(items=(Int(value=-1), Int(value=-8), Int(value=-9)))`
- `increment_each` (same-layer, commutation): `nil [ [ [ eq? ] swap cons filter null? ] [ cons ] [ drop ] ifte ] fold nil [ cons ] fold` vs `nil [ cons ] fold nil [ [ [ eq? ] swap cons filter null? ] [ cons ] [ drop ] ifte ] fold` on `Lst(items=(Int(value=9), Int(value=0), Int(value=3), Int(value=4), Int(value=9), Int(value=9)))`
- `increment_each` (earlier-layer, other): `nil [ [ [ eq? ] swap cons filter null? ] [ cons ] [ drop ] ifte ] fold` vs `nil [ [ over null? ] [ cons ] [ [ over uncons swap drop eq? ] [ drop ] [ cons ] ifte ] ifte ] fold` on `Lst(items=(Int(value=0), Int(value=-3), Int(value=7), Int(value=1), Int(value=7), Int(value=9)))`
- `increment_each` (earlier-layer, other): `nil [ [ [ eq? ] swap cons filter null? ] [ cons ] [ drop ] ifte ] fold nil [ cons ] fold` vs `nil [ cons ] fold nil [ [ over null? ] [ cons ] [ [ over uncons swap drop eq? ] [ drop ] [ cons ] ifte ] ifte ] fold` on `Lst(items=(Int(value=-7), Int(value=-8), Int(value=2), Int(value=-8), Int(value=-9)))`
- `keep_negative` (earlier-layer, insertion): `nil [ cons ] fold` vs `nil [ [ over null? ] [ cons ] [ [ over uncons swap drop eq? ] [ drop ] [ cons ] ifte ] ifte ] fold` on `Lst(items=(Int(value=1), Int(value=1), Int(value=-4), Int(value=-3)))`
- `keep_below` (same-layer, other): `dup div` vs `drop 1` on `Lst(items=(Int(value=9), Int(value=3), Int(value=2), Int(value=0))) Int(value=0)`
