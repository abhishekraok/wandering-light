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

## Running fewer searches costs verified behaviours, and the library is how these cells run fewer

At beam 64 the library is a clear win on the objective as `docs/metrics.md`
states it: **180 tasks against 143, in 154 s against 209 s**, with 29% fewer
executions. `nolib-b64`'s 143 tasks match the task *count* of #155's own cell
for this checkpoint (143 graded, from an independent driver on a different
device); the two runs' task sets were not compared.

At beam 256 the library arm verifies **16** behaviours against 18. That gap is
exactly the searches that did not happen. `b256` searched 131 of 300 tasks;
scoring `nolib-b256` on **only those 131 tasks** gives 16 verified behaviours
and the identical set, losing the same two. On every task both arms searched,
they return the same program at the same depth. The library changes which tasks
are searched and nothing about the search, so this is a result about search
count, not about libraries: any cache, memo or deduplication that skipped the
same 169 searches would cost the same two behaviours.

Both losses turn on cohort order:

| behaviour | shortcut found | the task it was replayed to | what that task's own search finds |
|---|---|---|---|
| `all_positive` | cohort position 134, 22 ops | 154 (and 156, 164, 172, 174) | the 23-op **genuine** program |
| `prefix_sums` | cohort position 232, 16 ops | 233 (and 239) | the 18-op **genuine** program |

In each case the shortcut simply came first. Had 154 preceded 134, the library
would have held the genuine program and replayed *that*. Cross-behaviour replay
transfer is exactly zero (#151), so a behaviour's library entry is decided by
whichever of its search-succeeding tasks comes first; under a random cohort
order each of these is lost about half the time. #151 checked order-robustness
over 100 task orders **on tasks**; nothing has checked it on verified
behaviours, and this is n=1 in order.

**The metric is one-sided here.** A replay can only duplicate a program some
search already found, so it can never *add* a verified behaviour and can remove
one. Counted in verified **tasks** the library wins at every width, this cell
included: 176 / 178 / 171 / 178 against 139 / 155 / 165 / 168. The caveat that
survives is therefore not "a library costs behaviours" but: **a verified count
rests only on the searches that ran.** `b64` solves 180 tasks but only 19 by
search, so its 16 verified behaviours rest on 19 independent attempts and 161
replays.

(`prefix_sums`'s shortcut verdict is the weaker of the two: its 16-op program
fails 0 of 10,000 draws from the behaviour's own sampler and errs only on the
empty list, which that sampler never produces. It is a shortcut under the
verified rule, which asks for transfer to tasks drawn differently, but not one
the behaviour's own distribution can see.)

## Width replaces genuine programs with shortcuts

The beam takes the **first** fitting successor it meets while expanding a layer
(`stack_beam.py:334`), which is before the layer is scored at all (`:345`), so
the order is shallowest depth, then the parent's position in the beam, then op
index — the current layer's ranking has no say. A wider beam therefore reaches
fitting programs a narrow one never sees, shortcuts included. It is not a
superset operation: the visited set is shared across depths (`:364`), so a
product a wide beam retains early becomes a duplicate later, which is how
`nondecreasing_indicator` is *lost* at beam 128 in both arms.

The direct evidence is per task, and monotone over four widths. Library-off,
`all_positive`'s 9 cohort tasks are solved by the **genuine** program on 6, 3, 1
and 0 of them as the beam goes 64 → 128 → 256 → 512, while the 22-op shortcut
appears on 0, 0, 1 and 2. Memorised rows — a program that fits the shown
examples and fails the held-out ones — rise over the same ladder, 17 → 30 → 39 →
43 library-off and 12 → 20 → 28 → 33 library-on. Width buys fitting programs,
and an increasing share of them are wrong.

Verified behaviours therefore need not rise with width, and library-off they run
**16 / 16 / 18 / 17**. Read that ladder as an existence proof, not a trend.
Verified *tasks* over the same cells run 139 / 155 / 165 / 168 and **are**
monotone, and beam 256's 18 rests on three behaviours verified by a single task
each (`all_positive`, `prefix_sums`, `pair_floor_averages`). One checkpoint, one
policy seed, one cohort, one cohort order; the cells are deterministic given a
checkpoint and a task, so there is no run-to-run variation to appeal to in
either direction, and equally no estimate of the variance that governs the
claim.

Beam 64 → 512 is 8× the width, and 9.1× the clock library-on (8.0× library-off),
for **+1** verified behaviour in each arm while graded behaviours go 18 → 22 and
tasks 180 → 190. Width adds four graded behaviours library-on and loses none;
**two of the four are shortcuts**, and the other two (`pair_floor_averages`,
`prefix_sums`) are exactly the +1 net verified. Library-off it adds three, one a
shortcut.

solved-program-cache (#151) ran the same on/off comparison at all four widths on
`witness-s0-e4`, the v4-only checkpoint, and its verified column did not move
(12 / 12 / 14 / 16 in both arms) while rising with width. The two checkpoints
differ; why is not measured. What can be said is where they start: the staged
one already verifies 16 at beam 64, where the fused one ends at beam 512. Width
still adds two verified behaviours here, and also costs `all_positive`, which
the fused checkpoint never had.

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

96% of the clock goes to tasks the cell does not solve (`b64` 0.967, `b512`
0.961); library-off it is 86.7–89.2%. The median library-on task costs **1 ms**
because the library answers it, against 596 ms library-off at the same width.
`docs/metrics.md` charges failures deliberately, and this is what that choice
looks like: driving the cost down means failing faster or less often, not
solving the easy tasks better.

## What this does not say

One checkpoint, one policy seed, one host, one cohort and one cohort order. The
search is deterministic given a checkpoint and a task, so "one run per cell" is
not a sampling caveat -- there is nothing to resample -- but it also means these
cells carry no estimate of the variance that governs a 2-behaviour swing.
Verified counts are behaviours of the cohort's 43, not of v4's 220. Nothing here
compares the staged checkpoint with the fused one under matched conditions.
Coverage in `docs/metrics.md` is held-out **graded**, so the pair reported for
the objective is the graded one; the verified column sits beside it because
graded measures the sampler's distribution and verified measures transfer to
tasks drawn differently — and 180 → 190 tasks at 9× the cost is the former.
