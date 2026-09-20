# witness-trajectory-shape (#150): cycles of grow-and-run, or write once and run once?

Static, no model, no search. `uv run python experiments/witness_trajectory_shape.py --out reports/witness-trajectory-shape-20260919`,
seconds on CPU. Tables in [`tables.md`](tables.md), per-witness rows in `shape.json`.

**Method.** Every distinct witness of `stack_pbe_v4` (220) and `stack_pbe_world_v3` (526) is replayed on
its first shown input and each step labelled **W** (taken under an open mark, or the `[` opening one:
the step adds program text and the data below the mark does not move), **D** (a top-level op: one
visible transition) or **X** (a top-level `i` / `dip` / `map` / `filter` / `fold` / `ifte`: a quotation
runs inside one step). Labels depend on the program alone; fuel and state size are read off that one
input. A policy row exists per step and every v4 behaviour has 40 tasks, so on v4 shares of witness ops
are shares of training rows.

## Findings (2026-09-19)

- **v4 witnesses are written, then run.** 84% of all witness ops are W (9,054 of 10,723); per witness the
  W share is 0.60 / 0.83 / 0.93 at p10 / p50 / p90 and does not depend on split (train 0.83, eval 0.84).
  **127 of 220 witnesses execute exactly once**, 7 never; mean 1.57 executions per witness. 60% of all
  executions fall in the last decile of the program. 131 of 220 are at least 80% W with at most two
  executions.
- **The one execution is most of the program.** The largest quotation consumed is a median 57% of the
  witness's ops (64% past 40 ops), and a median 57% of machine fuel is spent inside combinators, where no
  trajectory state exists. The input is first touched a median 71% of the way through; peak state size
  is 5.1x the larger of input and output (7.7x past 40 ops), so the stack grows once and collapses once.
- **It gets more so with length, not less.** 41+ ops: W share 0.85, one execution at the median, p90 3.
  `largest_absolute_prefix_sum` (57 ops) is `D5 W46 X1 D5`; `manhattan_point_radii` (94) `D7 W78 X1 D4 W3 X1`.
- **World-v3 is more cyclic, from macro expansion, with a monolithic tail.** W is 88% of ops, but
  executions are 1 / 2 / 8 (mean 3.63) and spread evenly over the program (last decile 36%), because each
  macro call expands to its own write-run cycle: `leetcode-sum-of-digits-in-base-k` (222 ops) is 22
  repeats of `D3 W6 X1`. Still 176 of 526 are >=80% W with <=2 executions, and the longest witness is
  `W1510 X1`. Hidden fuel is higher than v4's, median 75%.
- **So most policy rows are next-token-of-program-text rows.** On ~84% of rows the state differs from
  its predecessor by one appended symbol and the input has not moved; the Markov framing buys nothing
  there that a sequence model over the program text would not have.
- **Sub-goal relabelling has little to name.** A relabelled target is only meaningful at a state with no
  open mark after an execution; v4 has 1.57 of those per witness (1 / 1 / 3), and for 127 witnesses the
  only one is at or next to the final state.

**Two causes.** Teacher style: a fused `[dup mul 2 mul 4 add] map` where three `map`s would leave two
visible intermediates. And the machine: iteration exists only inside combinators, so a loop body is
written before any of it runs and the loop is one opaque step; a fold with real state cannot be staged
without changing the algorithm.

**Not measured:** whether staged witnesses are easier to learn or search. This gives the metric a staged
re-authoring would be accepted against (executions per witness, largest-body share, first touch).
