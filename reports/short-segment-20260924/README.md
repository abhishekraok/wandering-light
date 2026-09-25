# short-segment-campaign (#168): the cohort searches

**Reading: NEITHER.** No cell exceeds `BEST`'s 16 verified behaviours by more than 2 under any search. The capped arms verify 12-14 of 43, below `BEST`. The segment-reset beam adds at most 1 over flat-48 in any cell. No segment-reset cell verifies a behaviour outside `BEST`'s flat-48 set. Every new behaviour in the campaign came from the flat-widening control, which is a wider beam, not a reset. Seed 0 only; the owner cut seeds 1-2.

`search.py report --reruns ../short-segment-rerun` over `checkpoints/short-segment-search`, `report.md` and `report.json` here. Registration: [#168](https://github.com/abhishekraok/wandering-light/issues/168).

## Verified behaviours of 43 (#101 cohort, 300 tasks, seed 0)

| arm / encoding | flat-48 | segment-reset | flat-widening (matched fuel) | registered prediction, flat / segment |
|---|---:|---:|---:|---|
| staged / current (`BEST`) | **16** (reproduces #158, 300/300 tasks identical) | 15 (-all_positive) | 17 (+pair_floor_averages, +prefix_sums, -keep_zeros) | 16-18 / 17-21 |
| capped / current (machine) | 13 | 14 (+all_positive) | 12 (-keep_zeros) | 16-18 / 18-23 |
| capped / comparing (difference) | 13 | 14 (+all_positive) | 12 (+shared_value_occurrences, -reverse, -rotate_left) | 17-19 / **21-27** |
| capped / difference, local scoring, K=64 (exploratory) | | 13 | 14 (+prefix_sums, +shared_value_occurrences, -sum_negative) | within +2 of K=16 (advisor) |

- **Fused and staged / comparing were not searched.** The owner cut those families at 13:25 PDT on 2026-09-24.
- **Graded is not verified.** Graded behaviours run 2-4 above verified in every cell. `bounds_cover_list_flag`, `largest_inside_interval` and `aligned_matches` are the usual shortcuts. `all_positive` flips between its genuine program and its 22-op shortcut with cohort order, as in #158.
- **Task level is not the registered measure.** capped / difference segment-reset solves 167 tasks against flat-48's 161. That is the same 16 graded behaviours.
- **Cost.** Segment-reset spends about 20x flat-48's fuel (879M against 43M on capped / difference; 1.54B against 38M for `BEST`). Its 1x-cost curve point is at or below flat-48 in every cell (`report.md`, per-checkpoint table).
- **max_segments.** capped / difference segment-reset ends 101 of 300 tasks on `max_segments`: the ranker kept roots that led nowhere in 12 segments.

## The clock-capped tasks

RERUN_LINES

## Segment-capped yardstick, raw nats per op (registered)

Median per program over the 25 yardstick programs (`per_op_segment_capped.json`), seeds 0 / 1 / 2. Clears are raw `ln 64` per segment.

| policy | registered (advisor / mine) | measured | segmented clears of 25 |
|---|---|---|---|
| capped / current | 0.6-0.8 / 1.0-1.5 | 0.73 / 0.70 / 0.60 | 2 / 3 / 2 |
| capped / difference | 0.45-0.6 / 0.8-1.2 | 0.69 / 0.65 / 0.60 | 2 / 4 / 4 |
| staged / current | 2.3 | 3.05 / 2.72 / 2.88 | 0 |
| staged / difference | 1.9 | 2.89 / 2.68 / 2.93 | 0 |
| fused / current | 4.7 | 5.48 / 5.27 / 5.15 | 0 |
| fused / difference | 4.0 | 5.60 / 5.49 / 5.42 | 0 |

- **The style is learned; it does not solve.** The capped policies pay 0.6-0.7 nats per op on held-out capped programs, near the advisor's prediction. That is about a quarter of `staged`'s cost and an eighth of `fused`'s. On the v4 authored witnesses they pay 6.9-8.3. Even so, raw segment-reset clears only 2-4 of the 25 programs, and the cohort shows it: no capped cell reaches `BEST`.
- **The comparing encoder adds nothing here.** capped / difference and capped / current are within 0.05 per op, and 13 = 13, 14 = 14 on the cohort.
- `per_op_body_capped.json` is the same scoring on #167's body-capped programs: capped 0.81-0.88, staged 2.1-2.6, fused 4.6-4.9.
- The yardstick programs are `check.py`-verified (500 fresh samples), not 10,000-draw verified.

## Ranking: the local K diagnostic

Top-K share of segments with the witness's closed state among the merged closed states (capped / difference s0, depth 8, witness forced back in as a root):

| scoring | set | top 16 | top 64 |
|---|---|---:|---:|
| cumulative | capped train (37 programs) | 0.19 | 0.32 |
| local, scored from local roots | capped train | 0.32 | 0.50 |
| cumulative | yardstick (25) | 0.03 | 0.10 |
| local, scored from local roots | yardstick | 0.13 | 0.33 |

- **This replaces the 0.46 / 0.27 local figures.** Those ranked cumulatively chosen roots.
- **Local scoring ranks the witness far higher and solves nothing more.** On the cohort the local K=64 arm verifies 13 against K=16's 14. The advisor's rule for this arm (3 or more over K=16 means the ranker is the bottleneck) is not met. The loss is in what the policy produces, not how the roots are ranked.

## Flat-widening control: a limit

The control runs at the segment beam's horizon, 12 x depth: 96 for capped, 192 for staged. That is 2-4x flat-48's depth.
- On capped it is worse than flat-48 (12 against 13). It spends its budget at depth, not width. Of the 143 capped / difference tasks it ends by budget, 95 completed width 256 and 47 never completed width 64. It loses short behaviours flat-48 keeps: `reverse`, `rotate_left` and `keep_zeros`.
- It is still the only search that verified behaviours `BEST` does not: `prefix_sums`, `pair_floor_averages` and `shared_value_occurrences`, three of #167's 25.

## Scope cuts and conditions

- **Seeds.** Seeds 1-2 were cut by the owner at about 13:05 PDT on 2026-09-24, so no column has a seed spread.
- **Cells.** The fused and difference-staged families were cut at 13:25 PDT. The capped / machine eval archive backfill was skipped.
- **Hardware.** CPU, 8 threads, 2 lanes. Other jobs' CPU windows are in `checkpoints/short-segment-search/CONTENTION.md`.
- **Reruns.** GPU, 1 thread, 1800 s clock. Each rerun holds only the capped positions, so its library starts empty. Those tasks reached search in the full cell, so replay had not fitted them there.
- **Frozen per cell.** K, depth, the fit check and the merge rule are in `report.md`, under "Rules, one line each".
