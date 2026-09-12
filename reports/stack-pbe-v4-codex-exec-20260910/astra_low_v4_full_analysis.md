astra low, full split: solved 995/1800 (55.3%), memorised 714, shown-only 94.9%, tool attempts 0, input tok 14,311,678, output tok 851,261, reasoning tok 777,256, wall 33,487 s
aligned_matches solves (sampler shortcut): 20
BFS 10k, full split: solved 120/1800, memorised 22

Solves by witness length (solved/tasks):

| solver | 1-10 | 11-20 | 21-40 | 41-80 | 81+ |
|---|---:|---:|---:|---:|---:|
| astra low | 358/360 | 193/240 | 230/360 | 140/520 | 74/320 |
| BFS 10k | 120/360 | 0/240 | 0/360 | 0/520 | 0/320 |

Authorship at matched witness length (solved/tasks; behaviours in parentheses):

| solver | bucket | hand-written (v3) | sol-written |
|---|---|---:|---:|
| astra low | 1-10 | 279/280 (7) | 79/80 (2) |
| astra low | 11-20 | — (0) | 193/240 (6) |
| astra low | 21-40 | 39/40 (1) | 191/320 (8) |
| astra low | 41-80 | — (0) | 140/520 (13) |
| astra low | 81+ | — (0) | 74/320 (8) |
| astra low | all | 318/320 (8) | 677/1480 (37) |
| BFS 10k | 1-10 | 40/280 (7) | 80/80 (2) |
| BFS 10k | 11-20 | — (0) | 0/240 (6) |
| BFS 10k | 21-40 | 0/40 (1) | 0/320 (8) |
| BFS 10k | 41-80 | — (0) | 0/520 (13) |
| BFS 10k | 81+ | — (0) | 0/320 (8) |
| BFS 10k | all | 40/320 (8) | 80/1480 (37) |

Every eval behaviour, astra low then BFS 10k (solved/40, memorised):

| behaviour | ops | author | astra solved | astra memorised | BFS solved |
|---|---:|---|---:|---:|---:|
| `push_back` | 4 | sol | 39 | 0 | 40 |
| `increment_each` | 5 | v3 | 40 | 0 | 0 |
| `keep_negative` | 5 | v3 | 40 | 0 | 0 |
| `keep_zeros` | 5 | v3 | 40 | 0 | 0 |
| `reverse` | 5 | v3 | 40 | 0 | 40 |
| `rotate_left` | 5 | sol | 40 | 0 | 40 |
| `keep_below` | 6 | v3 | 39 | 1 | 0 |
| `negate_each` | 6 | v3 | 40 | 0 | 0 |
| `n_minus_each` | 7 | v3 | 40 | 0 | 0 |
| `drop_last` | 12 | sol | 38 | 2 | 0 |
| `min_pair` | 12 | sol | 40 | 0 | 0 |
| `cartesian_pair_sums` | 15 | sol | 40 | 0 | 0 |
| `sum_negative` | 16 | sol | 40 | 0 | 0 |
| `all_positive` | 18 | sol | 9 | 31 | 0 |
| `capped_sum` | 20 | sol | 26 | 14 | 0 |
| `replace_value` | 22 | sol | 39 | 1 | 0 |
| `prefix_sums` | 25 | v3 | 39 | 0 | 0 |
| `concatenate_shorter_first` | 26 | sol | 35 | 5 | 0 |
| `shared_value_occurrences` | 31 | sol | 17 | 21 | 0 |
| `aligned_matches` | 39 | sol | 20 | 18 | 0 |
| `nearest_interval_endpoint_each` | 39 | sol | 20 | 14 | 0 |
| `positive_majority_three` | 39 | sol | 5 | 33 | 0 |
| `count_in_range` | 40 | sol | 31 | 9 | 0 |
| `largest_inside_interval` | 40 | sol | 24 | 16 | 0 |
| `nondecreasing_indicator` | 43 | sol | 5 | 34 | 0 |
| `minimum_cross_list_gap` | 44 | sol | 6 | 32 | 0 |
| `adjacent_sign_change_count` | 47 | sol | 6 | 30 | 0 |
| `positive_balance_prefix_count` | 49 | sol | 5 | 35 | 0 |
| `stable_distinct_shared_values` | 49 | sol | 19 | 17 | 0 |
| `farther_choice_from_target` | 54 | sol | 1 | 38 | 0 |
| `farthest_from_target_last_tie` | 59 | sol | 11 | 27 | 0 |
| `aligned_maxima` | 69 | sol | 30 | 7 | 0 |
| `pair_floor_averages` | 69 | sol | 26 | 3 | 0 |
| `distinct_count_three` | 70 | sol | 9 | 31 | 0 |
| `span_inside_interval` | 76 | sol | 4 | 34 | 0 |
| `furthest_squared_radius` | 78 | sol | 8 | 30 | 0 |
| `keep_ascending_pairs` | 80 | sol | 10 | 13 | 0 |
| `second_largest` | 83 | sol | 12 | 26 | 0 |
| `widest_pair` | 84 | sol | 11 | 27 | 0 |
| `axis_point_count` | 86 | sol | 0 | 38 | 0 |
| `bounds_cover_list_flag` | 92 | sol | 5 | 35 | 0 |
| `manhattan_point_radii` | 94 | sol | 25 | 5 | 0 |
| `lexicographic_order` | 124 | sol | 3 | 36 | 0 |
| `polyline_manhattan_length` | 188 | sol | 1 | 33 | 0 |
| `median_floor` | 253 | sol | 17 | 18 | 0 |

Longest behaviours astra solved at least once, with one solving program each:

  median_floor (253 ops, 17/40): `nil [ over over quote [ lt? ] cat filter over cons [ quote [ lt? false eq? ] cat filter ] dip cat ] fold dup dup 0 [ drop 1 add ] fold 1 1 add div [ dup ] dip quote swap quote swap cat [ drop i [ dup 0 eq? ] [ ] [ 1 sub [ uncons drop ] dip ] ifte quote swap quote swap cat ] fold i drop uncons swap drop [ dup 0 [ drop 1 add ] fold 1 sub 1 1 add div [ dup ] dip quote swap quote swap cat [ drop i [ dup 0 eq? ] [ ] [ 1 sub [ uncons drop ] dip ] ifte quote swap quote swap cat ] fold i drop uncons swap drop ] dip add 1 1 add div`
  polyline_manhattan_length (188 ops, 1/40): `uncons [ uncons ] dip swap 0 quote [ quote ] dip cat [ quote ] dip cat [ swap i [ [ over swap sub [ 0 lt? ] [ 0 swap sub ] [ ] ifte ] dip ] dip [ swap ] dip add [ swap ] dip quote [ quote ] dip cat [ quote ] dip cat ] fold i [ drop drop ] dip`
  lexicographic_order (124 ops, 3/40): `[ [ over null? ] [ drop drop [ null? ] [ drop 0 ] [ drop 1 ] ifte ] [ [ [ over null? ] dip swap ] [ drop drop drop 0 1 sub ] [ [ [ uncons ] dip uncons ] dip [ [ swap ] dip ] dip [ [ over over eq? ] dip swap ] [ [ drop drop ] dip dup i ] [ drop lt? [ drop drop ] dip [ ] [ drop 0 1 sub ] [ drop 1 ] ifte ] ifte ] ifte ] ifte ] dup i`
  manhattan_point_radii (94 ops, 25/40): `nil 0 cons [ [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte swap uncons [ 0 eq? ] [ drop swap cons 1 cons ] [ drop uncons swap [ add ] dip swap cons 0 cons ] ifte ] fold uncons drop nil [ cons ] fold`
  bounds_cover_list_flag (92 ops, 5/40): `[ drop drop ] dip nil swap cons [ 1 1 add dup add 1 add eq? ] filter 0 [ drop 1 add ] fold`
  widest_pair (84 ops, 11/40): `[ swap [ dup null? ] [ drop drop 0 ] [ uncons swap uncons swap [ sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] dip [ swap ] dip swap dup i [ over over lt? ] [ swap drop ] [ drop ] ifte ] ifte ] dup i`
  second_largest (83 ops, 12/40): `dup uncons [ [ lt? ] [ swap drop ] [ drop ] ifte ] fold quote [ lt? ] cat filter uncons [ [ lt? ] [ swap drop ] [ drop ] ifte ] fold`
  keep_ascending_pairs (80 ops, 10/40): `[ nil nil ] [ swap i [ dup null? ] [ drop swap quote ] [ uncons [ drop ] dip [ swap ] dip swap [ over over lt? ] [ quote swap quote swap cat cat ] [ drop drop ] ifte nil ] ifte quote swap quote swap cat ] fold i drop`
  furthest_squared_radius (78 ops, 8/40): `nil nil cons 0 cons [ dup mul swap uncons swap uncons [ drop ] dip [ dup null? ] [ drop swap quote ] [ uncons [ drop ] dip [ swap ] dip add [ over over lt? ] [ swap drop ] [ drop ] ifte nil ] ifte nil swap cons swap cons ] fold uncons [ drop ] dip`
  span_inside_interval (76 ops, 4/40): `quote [ swap lt? false eq? ] cat swap [ filter ] dip quote [ lt? false eq? ] cat filter dup [ uncons [ [ over over lt? ] [ drop ] [ swap drop ] ifte ] fold ] dip uncons [ [ over over lt? ] [ swap drop ] [ drop ] ifte ] fold swap sub`
  distinct_count_three (70 ops, 9/40): `[ over eq? ] [ drop drop drop 1 dup add ] [ [ [ swap ] dip eq? ] [ drop drop drop 1 dup add ] [ [ drop eq? ] [ drop drop drop 1 dup add ] [ drop drop drop 1 dup add 1 add ] ifte ] ifte ] ifte`
  aligned_maxima (69 ops, 30/40): `nil nil cons swap cons [ swap uncons dup null? quote [ cons swap drop ] [ uncons [ [ swap ] dip swap ] dip over over lt? quote [ swap drop ] [ drop ] ifte [ swap uncons swap drop ] dip cons nil swap cons swap cons ] ifte ] fold uncons drop uncons swap drop nil [ cons ] fold`
  pair_floor_averages (69 ops, 26/40): `nil nil cons nil cons [ [ uncons swap uncons swap drop ] dip [ over null? ] [ [ drop ] dip quote ] [ [ uncons swap drop ] dip add 1 1 add div cons nil ] ifte nil swap cons swap cons ] fold uncons swap drop nil [ cons ] fold`
  farthest_from_target_last_tie (59 ops, 11/40): `quote [ dup ] swap cat [ sub dup mul [ nil swap cons ] dip cons ] cat map uncons [ [ over uncons swap drop over uncons swap drop swap lt? ] [ drop ] [ swap drop ] ifte ] fold uncons drop uncons swap drop`
  farther_choice_from_target (54 ops, 1/40): `[ swap drop ] dip [ over dup mul over dup mul lt? ] [ swap drop ] [ drop ] ifte`
  stable_distinct_shared_values (49 ops, 19/40): `quote [ swap quote [ eq? ] cat filter null? false eq? ] cat filter nil [ dup quote [ eq? false eq? ] cat swap [ filter ] dip cons ] fold nil [ cons ] fold`
  positive_balance_prefix_count (49 ops, 5/40): `nil 0 cons 0 cons [ [ uncons swap uncons swap drop ] dip add [ dup 0 swap lt? ] [ swap 1 add swap ] [ ] ifte nil swap cons swap cons ] fold uncons swap drop`
  adjacent_sign_change_count (47 ops, 6/40): `nil 0 cons 0 cons [ [ uncons swap uncons swap drop ] dip [ swap ] dip dup [ mul nil swap cons [ 0 lt? ] filter 0 [ drop 1 add ] fold add ] dip swap nil swap cons swap cons ] fold uncons drop uncons swap drop`
  minimum_cross_list_gap (44 ops, 6/40): `swap quote [ swap quote [ sub [ dup 0 lt? ] [ 0 swap sub ] [ ] ifte ] cat map uncons [ [ over over lt? ] [ drop ] [ swap drop ] ifte ] fold ] cat map uncons [ [ over over lt? ] [ drop ] [ swap drop ] ifte ] fold`
  nondecreasing_indicator (43 ops, 5/40): `uncons nil swap cons 1 cons [ [ uncons swap uncons [ drop ] dip ] dip [ over over swap lt? ] [ [ drop drop 0 ] dip ] [ swap drop ] ifte nil swap cons swap cons ] fold uncons swap drop`
