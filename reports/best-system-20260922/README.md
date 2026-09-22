# One command, the pinned system, and what width and the library buy it (best-system-command #158)

`uv run python -m wandering_light.solve TASKS` runs `solve.BEST` on a task file
and prints, per task, whether the program found passes the task's held-out
examples and the wall clock charged for it, then the pair `docs/metrics.md`
defines: coverage and total wall clock over every task attempted, solved or
not.

`BEST` is `staged-s0-e3` — the staged arm of staged-witness-reauthoring (#155),
seed 0, epoch 3, at
`abhishekraok/wandering-light-stack-policy@e5a0b1391c5268c3ae04a49e6862f19a30779bb2`
— beam 64 × depth 48, guard `both`, the run's own solved-program library, and a
16,384-atom state cap (beam-state-size-unbounded, #157). Every cell below is
that system with **one** setting changed.

Cohort: the compute-optimal-mark-28 (#101) 300-task / 43-behaviour eval cohort
of `stack_pbe_v4`, in cohort order, one host (`UltraDesk`, CPU, torch's default
8 threads), **one process per cell**, driver `experiments/best_system_cells.py`,
run dir `checkpoints/best-system-20260922`. `timed` is the sum of the per-task
clocks, which charge library replay, search and policy inference; grading runs
after the last clock stops, and `run_body`'s memo is cleared before each cohort
so no cell inherits work from another. (An earlier single-process run of these
cells had clocks flattered by 1–2% for exactly that reason; executions were
identical to the digit, so only the clock was ever at risk.) **Verified**
behaviours are 10,000-draw `genuine` verdicts plus catalogue-witness-equal
programs; **graded** counts a behaviour any of whose programs passed held-out
grading.

## Cells

| cell | beam | library | tasks /300 | search-solved | graded /43 | **verified /43** | timed (s) | executions |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `b64` (**BEST**) | 64 | on | 180 | 19 | 18 | **16** | 154 | 34,785,672 |
| `b128` | 128 | on | 181 | 17 | 18 | **16** | 330 | 65,233,317 |
| `b256` | 256 | on | 183 | 19 | 20 | **16** | 706 | 120,034,579 |
| `b512` | 512 | on | 190 | 20 | 22 | **17** | 1,397 | 214,205,105 |
| `nocap-b64` | 64 | on | 180 | 19 | 18 | **16** | 153 | 34,785,672 |
| `nolib-b64` | 64 | off | 143 | 143 | 18 | **16** | 209 | 49,078,770 |
| `nolib-b128` | 128 | off | 158 | 158 | 17 | **16** | 399 | 82,869,294 |
| `nolib-b256` | 256 | off | 168 | 168 | 19 | **18** | 823 | 147,323,518 |
| `nolib-b512` | 512 | off | 177 | 177 | 21 | **17** | 1,661 | 268,820,179 |

## The library suppresses the retries that find genuine programs

At beam 64 the library is a clear win on the objective as `docs/metrics.md`
states it: **180 tasks against 143, in 154 s against 209 s**, with 29% fewer
executions. `nolib-b64`'s 143 tasks reproduce #155's own cell for this
checkpoint exactly — an independent driver, on a different device, agreeing to
the task.

But at beam 256 the library **costs two verified behaviours**, 18 → 16, and one
behaviour shows the mechanism. `all_positive` has 9 tasks in the cohort:

| cell | solved | how |
|---|---|---|
| `nolib-b256` | 2 of 9 | one search finds a 22-op **shortcut**, another finds the 23-op **genuine** program → behaviour **verified** |
| `b256` | 6 of 9 | one search finds the 22-op shortcut; the other five tasks replay it from the library → graded, **not verified** |
| `nolib-b512` | 2 of 9 | both searches find the shortcut → not verified at this width either way |

A library turns k independent attempts at a behaviour into one shared answer.
When the first answer is a shortcut, the attempts that would have found a
genuine program never happen. That is the sharp form of a caveat belonging on
every library-on cell: `b64` solves 180 tasks but only **19** by search, so its
verified count rests on 19 independent searches and 161 replays.
solved-program-cache (#151) found that a cache "moves tasks, not behaviours";
on this checkpoint at beam 256 it moves verified behaviours **down** by two, and
#151's cells could not see it — they were run on a checkpoint that had no
genuine program to lose at that width.

## Width is not monotone in verified behaviours, with or without the library

Library-off, verified goes **16 → 16 → 18 → 17** across beam 64 → 512.
Library-on, **16 → 16 → 16 → 17**. Graded is not monotone either:
`nondecreasing_indicator` is graded at beam 64 and lost at 128 in both arms.

The beam takes the **first** fitting successor it meets while expanding a
layer (`stack_beam.py:334`), which is before the layer is scored at all
(`:345`), so the order is shallowest depth, then the parent's position in the
beam, then op index — the current layer's ranking has no say. Widening
therefore enlarges the fitting set at every depth and can only add candidates,
shortcuts included. Monotonicity in *verified* behaviours
was never a property of this search, and `all_positive` is the case that shows
it: a 22-op shortcut sits one level shallower than the 23-op genuine program and
displaces it once the beam is wide enough to reach it.

Beam 64 → 512 is 8× the width and 9.1× the clock for **+1** verified behaviour
in each arm, while graded behaviours go 18 → 22 and tasks 180 → 190. Width adds
four graded behaviours library-on (`aligned_matches`, `pair_floor_averages`,
`prefix_sums`, `stable_distinct_shared_values`) and loses none; **two of the
four are shortcuts**, and the other two are exactly the +1 net verified, since
`all_positive` goes the other way. Library-off it adds three, one of them a
shortcut.

This is the opposite of what solved-program-cache (#151) measured on
`witness-s0-e4`, the v4-only checkpoint, where the same ladder took verified
behaviours 12 → 16 and lost none. What differs is where the two checkpoints
start: the staged one already verifies 16 at beam 64, which is where the fused
one *ends* at beam 512, and the behaviours it has not got are not ones a wider
beam finds. Read #151's row as a result about a checkpoint with room, not as a
property of width.

## The state cap is free, and is not a speed fix

With and without the 16,384-atom cap, all 300 rows agree on program,
executions, fuel, expansions, depth and source; the clock moves 153.4 s →
154.2 s, **0.46%**, the cost of counting atoms on retained successors. The cap
is **132×** the largest state along any v4 witness (124 atoms) and 6× the
largest along any world-v3 witness (2,642, `codearc-695`), and 122× below the
#157 runaway's peak of 1,998,856.
`cartesian_pair_sums` is the slowest task in **every** cell and still costs 33 s
at beam 512:
the cap bounds hashing and memory, not the search.

## Cost to clear is a cost of failure

96% of the clock goes to tasks that are never solved (`b64` 0.967, `b512`
0.961); library-off it is 86.7–89.2%. The median library-on task costs **1 ms**
because the library answers it, against 596 ms library-off at the same width.
`docs/metrics.md` charges failures deliberately, and this is what that choice
looks like: driving the cost down means failing faster or less often, not
solving the easy tasks better.

## What this does not say

One seed, one host, one checkpoint, one run per cell, no interval on the clock.
Verified counts are behaviours of the cohort's 43, not of v4's 220. Nothing here
compares the staged checkpoint with the fused one under matched conditions.
Coverage in `docs/metrics.md` is held-out **graded**, so the pair reported for
the objective is the graded one; the verified column sits beside it because
graded measures the sampler's distribution and verified predicts transfer to
tasks drawn differently — and 180 → 190 tasks at 9× the cost is the former.
