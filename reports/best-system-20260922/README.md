# One command, the pinned system, and what width buys it (best-system-command #158)

`uv run python -m wandering_light.solve TASKS` runs `solve.BEST` on a task file
and prints, per task, whether the program found passes the task's held-out
examples and the wall clock charged for it, then the pair `docs/metrics.md`
defines: coverage and total wall clock over every task attempted, solved or
not. This report is that command, run as cells that differ from `BEST` in
**one** setting each.

`BEST` is `staged-s0-e3` — the staged arm of staged-witness-reauthoring (#155),
seed 0, epoch 3, published at
`abhishekraok/wandering-light-stack-policy@e5a0b1391c5268c3ae04a49e6862f19a30779bb2`
— at beam 64 × depth 48, guard `both`, the run's own solved-program library
replayed before each search, and a 16,384-atom state cap
(beam-state-size-unbounded, #157).

Cohort: the compute-optimal-mark-28 (#101) 300-task / 43-behaviour eval cohort
of `stack_pbe_v4`, in cohort order, one host (`UltraDesk`, CPU, 8 threads),
driver `experiments/best_system_cells.py`, run dir
`checkpoints/best-system-20260922`. `timed` is the sum of the per-task clocks,
which cover library replay, search and policy inference; grading runs after the
last clock stops, so no held-out example warms the executor's memo table for a
later search. **Verified** behaviours are the union of 10,000-draw `genuine`
verdicts and catalogue-witness-equal programs
(`experiments/step3_verify_solves.py`), the convention
`docs/RESULTS_DETAILED.md` uses; **graded** counts a behaviour whose program
merely passed held-out grading.

## Cells

TABLE

## What the cells say

**The command reproduces #155's checkpoint and adds the library's tasks, not
its behaviours.** At beam 64 the pinned system grades the same 18 behaviours
and verifies the same 16 as #155's own CUDA cell for this checkpoint, and
solves 180 of 300 tasks against that cell's 143. The extra 37 tasks are library
replays. That is solved-program-cache (#151)'s "a cache moves tasks, not
behaviours", reproduced on a different checkpoint, a different device and a
different driver.

**The state cap is free here.** With and without it, all 300 rows agree on
program, executions (34,785,672), fuel, expansions, depth and source; the clock
moves 150 s → 153 s, the cost of counting atoms on every retained successor.
The cap is 6× the largest state along any v4 witness (124 atoms) and 100× below
the runaway of #157 (peak 1,998,856). It does **not** make the runaway task
cheap: `cartesian_pair_sums` still costs 33 s at beam 512, the slowest task in
the campaign. The cap bounds memory and hashing, not the search.

**Width buys graded behaviours and almost no verified ones.** Beam 64 → 512 is
8× the width and 9.2× the clock for **+1 verified behaviour**, while graded
behaviours go 18 → 22 and tasks 180 → 190. Four of the five behaviours width
adds are shortcuts that pass held-out grading and fail draw-testing.

**And width is not monotone in verified behaviours.** `all_positive` is
verified at beam 64 and 128 and **lost** at 256 and 512: the wider beam reaches
a shortcut program (`[ 0 lt? ] filter [ 0 eq? ] map [ null? ] [ drop 1 ] [ drop
0 ] ifte`) before the genuine one, and a solve is a solve to the search. Beam
512 nets `+pair_floor_averages`, `+prefix_sums`, `−all_positive`.

This is the opposite of what solved-program-cache (#151) measured on
`witness-s0-e4`, the v4-only checkpoint, where the same width ladder took
verified behaviours 12 → 16 and lost none. Two checkpoints, two answers. What
differs is where they start: the staged checkpoint already verifies 16 at beam
64, which is where the fused one **ends** at beam 512, and the behaviours it
has not got are not ones a wider beam finds. Read the #151 row as a result
about a checkpoint with room, not as a general property of width.

**Cost to clear is a cost of failure.** 96% of the clock is spent on tasks that
are never solved, at every width, and the median task costs 1 ms because the
library answers it. `docs/metrics.md` charges failures deliberately, and this
is what that choice looks like in practice: driving the cost down means
failing faster or failing less often, not solving the easy tasks better.

## What this does not say

One seed, one host, one checkpoint, one run per cell; no error bar on the
clock. Verified counts are behaviours, not tasks, and the cohort's 43
behaviours are not v4's 220. Nothing here compares the staged checkpoint with
the fused one under matched conditions — #155 did that on CUDA without a
library, and the numbers in this report are not comparable with it beyond the
behaviour sets quoted above.
