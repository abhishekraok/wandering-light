# short-segment-campaign (#168): the cohort searches

**Reading: NEITHER, on the cells searched (seed 0; the fused and staged / difference families were not searched).** No cell exceeds `BEST`'s 16 verified behaviours by more than 2 under any search. The capped arms verify 12-14 of 43, below `BEST`. The segment-reset beam adds at most 1 over flat-48 in any cell. No segment-reset cell verifies a behaviour outside `BEST`'s flat-48 set. Every new behaviour in the campaign came from the flat-widening control, a wider beam at a longer horizon, not a reset. Seed 0 only; the owner cut seeds 1-2.

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
- **Cost.** Segment-reset spends 20x flat-48's fuel on capped / difference (879M against 43M), 40x for `BEST` (1.54B against 38M), and 34x for the local K=64 arm. Within flat-48's own per-task fuel (the 1x curve point), segment-reset solves fewer graded tasks in every cell, though capped / difference keeps its 14 verified behaviours there (`report.md`, per-checkpoint table).
- **max_segments.** capped / difference segment-reset ends 101 of 300 tasks on `max_segments`, with no shown fit after 12 segments.

## The clock-capped tasks

Every segment task that hit the 120 s CPU clock was rerun alone on GPU under an 1800 s clock. **None was solved, so no verified count moves.**

| cell | capped tasks | rerun ended | median s | solved | verified |
|---|---:|---|---:|---:|---|
| capped / difference, K=16 | 6 | 6 max_segments | 37 | 0 | 14 -> 14 |
| capped / machine, K=16 | 6 | 6 max_segments | 46 | 0 | 14 -> 14 |
| staged / machine (`BEST`), K=16 | 36 | 36 max_segments | 66 | 0 | 15 -> 15 |
| capped / difference, local K=64 | 98 | 93 max_segments, 5 shown fits | 114 | 0 | 13 -> 13 |

- **The five shown fits fail the held-out examples.** They are `capped_sum`, `shared_value_occurrences`, `largest_inside_interval`, `axis_point_count` and `nondecreasing_indicator`.
- **The clock cost the segment cells nothing.** At the longer clock there are no new solves: 141 tasks ended at `max_segments`, and five stopped at shown fits that failed held-out. This covers the segment cells only; `BEST`'s widening cell still has 21 tasks capped at its 300 s clock.
- The local arm's 98 were run as three parts (`checkpoints/short-segment-search/lanes-rerun.sh`).

## Segment-capped yardstick, raw nats per op (registered)

Median per program over the 25 yardstick programs (`per_op_segment_capped.json`), seeds 0 / 1 / 2. Clears are raw `ln 64` per segment.

| policy | registered (advisor / mine) | measured | segmented clears of 25 |
|---|---|---|---|
| capped / current | 0.6-0.8 / 1.0-1.5 | 0.73 / 0.69 / 0.60 | 2 / 3 / 2 |
| capped / difference | 0.45-0.6 / 0.8-1.2 | 0.69 / 0.65 / 0.60 | 2 / 4 / 4 |
| staged / current | 2.3 | 3.05 / 2.72 / 2.88 | 0 |
| staged / difference | 1.9 | 2.89 / 2.68 / 2.93 | 0 |
| fused / current | 4.7 | 5.48 / 5.27 / 5.15 | 0 |
| fused / difference | 4.0 | 5.60 / 5.49 / 5.42 | 0 |

- **The style is learned; it does not solve.** The capped policies pay 0.6-0.7 nats per op on held-out capped programs, near the advisor's prediction. That is about a quarter of `staged`'s cost and an eighth of `fused`'s. On the v4 authored witnesses they pay 6.9-8.3. Even so, raw segment-reset clears only 2-4 of the 25 programs, and the cohort shows it: no capped cell reaches `BEST`.
- **Where the cost sits.** Pooled over programs, capped / difference s0 pays 1.35 nats per content op against 0.18 per structural op (`[`, `]` and the consumers).
- **The comparing encoder adds nothing here.** It has 659,932 parameters against the current encoder's 655,836. capped / difference and capped / current are within 0.05 per op, and 13 = 13, 14 = 14 on the cohort.
- `per_op_body_capped.json` is the same scoring on #167's 22 genuine body-capped programs: capped 0.81-0.88, staged 2.1-2.6, fused 4.6-4.9.
- The yardstick programs are `check.py`-verified (500 fresh samples), not 10,000-draw verified.

## Ranking: the local K diagnostic

Top-K share of segments with the witness's closed state among the merged closed states (capped / difference s0, depth 8, witness forced back in as a root):

| scoring | set | top 16 | top 64 |
|---|---|---:|---:|
| cumulative | capped train (37 programs) | 0.19 | 0.32 |
| local, scored from local roots | capped train | 0.32 | 0.50 |
| cumulative | yardstick (25) | 0.03 | 0.10 |
| local, scored from local roots | yardstick | 0.13 | 0.33 |

- **This replaces the earlier local top-16 figures of 0.46 and 0.27** (the K diagnostic comment on #168), which were ranked on cumulatively chosen roots. The same top-16 shares under a true reset are 0.32 and 0.13.
- **The registered K criterion was missed.** The registration asked K to cover the witness's rank in most segments, and predicted 60-80% (mine) or at least 80% (the advisor's) in the top 16 for capped / comparing. Cumulative measured 19% on train and 3% on the yardstick. K=16 was kept as registered.
- **Production and ranking both leak.** Of 268 train segments, the witness's closed state is produced in 235. Of those, 134 rank in the top 64 locally and 101 fall below. On the yardstick, 171 of 230 are produced, 77 rank in the top 64 and 94 fall below. These are forced-root diagnostics, not a decomposition of the solver's failures.
- **The tested ranking change did not help on the cohort.** The local K=64 arm verifies 13 against K=16's 14, so the advisor's trigger (3 or more over K=16) is not met. It changes K and scoring together, and it hit the 120 s clock on 98 of 300 tasks, against 6 for K=16. Rerun uncapped, none of the 98 is solved. A learned value was outside the registration, and this does not rule it out.

## Flat-widening control: width at a longer horizon

The control runs at the segment beam's horizon, 12 x depth: 96 for capped, 192 for staged. That is 2-4x flat-48's depth.
- On capped it is worse than flat-48 (12 against 13). It spends its budget at depth, not width. Of the 143 capped / difference tasks it ends by budget, 95 completed width 256 and 47 never completed width 64. It loses short behaviours flat-48 keeps: `reverse`, `rotate_left` and `keep_zeros`.
- It is still the only search that verified behaviours `BEST` does not: `prefix_sums`, `pair_floor_averages` and `shared_value_occurrences`, three of #167's 25. The gain belongs to width at 2-4x the depth, not to width alone; a widening at depth 48 would separate the two.

## Scope cuts and conditions

- **Seeds.** Seeds 1-2 were cut by the owner at about 13:05 PDT on 2026-09-24, so no column has a seed spread.
- **Cells.** The fused and difference-staged families were cut at 13:25 PDT. The capped / machine eval archive backfill was skipped.
- **Hardware.** CPU, 8 threads, 2 lanes. Other jobs' CPU windows are in `checkpoints/short-segment-search/CONTENTION.md`.
- **Reruns.** GPU, 1 thread, 1800 s clock. Each rerun holds only the capped positions, so its library starts empty. Those tasks reached search in the full cell, so replay had not fitted them there.
- **Frozen per cell.** K, depth, the fit check and the merge rule are in `report.md`, under "Rules, one line each".

## Codex review of the flat-widening control

[CONTROL_REVIEW.md](CONTROL_REVIEW.md). The fuel budget, same horizon and package-not-reset caveat were adopted. Not adopted, and open for anyone reusing `segment_beam_solve` or the control: check every complete candidate in both searches, not only at closed states; report paired solve-against-budget curves at preset budgets; freeze the merge-representative rule.
