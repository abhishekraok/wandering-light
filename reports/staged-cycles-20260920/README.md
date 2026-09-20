# staged-witness-reauthoring (#155): per-cycle learnability with the next closed state as the goal

`experiments/staged_cycles.py`, run dir `checkpoints/staged-cycles-20260920` (UltraDesk, RTX 4090, CUDA). Full
tables in [`cycles.md`](cycles.md). Driver and tests written by a Claude Opus 5 sub-agent to a spec from this
session; run and analysed here.

**Setup.** Two arms of the #101 1m recipe (3 seeds, epoch 3, 313,560 task-balanced rows per epoch), trained
with each row's target set to the **nearest closed state after it** on its witness: `staged-subgoal` on the
staged witnesses, `fused-subgoal` on the v4 witnesses. Evaluation is on the **validation-skeleton holdout**, 14
behaviours that are in no arm's fit set and have a staged witness, 5 tasks each. Each witness is cut at its
closed states; for every transition one beam search (width 64, depth = cycle length + 4, guard `both`) starts
at the closed state with the **next closed state as its target**, an oracle sub-goal. A transition is solved
when the returned program replays the start stacks onto the target stacks on every shown example. 750 staged
and 500 fused transitions per seed; none hit the 10 s wall-clock cap.

## Result (2026-09-20)

| witness | transitions of 2+ ops solved | 2-5 ops | 6-12 ops | 13+ ops | **whole witnesses walked end to end** |
|---|---|---|---|---|---|
| fused | 0.445 (0.51 / 0.39 / 0.44) | 0.867 | 0.756 | 0.308 | **0.152** (0.23 / 0.11 / 0.11) |
| staged | **0.914** (0.92 / 0.90 / 0.92) | 0.997 | 0.856 | 1.000 | **0.795** (0.83 / 0.77 / 0.79) |

The length buckets are thin and not behaviour-matched: a task's five instances repeat the same cycle text, so
fused 2-5 is 3 distinct cycles from 2 behaviours, fused 6-12 is 3 distinct from 3, and staged 13+ is 4 distinct
cycles. Read the 2+-op and whole-witness columns; the buckets say only that fused's failures sit in its long cycle.

The overall shares (0.878 fused, 0.965 staged) are not the result: `beam_solve` tests for a hit while it
expands, so every one-op transition is solved by exhaustive one-step search whatever the policy says, and both
witness kinds are mostly one-op transitions.

- **With the right sub-goals handed to it, the policy walks 80% of held-out staged witnesses from input to
  output, and 15% of fused ones.** That is the proposer closed-state-value-search (#156) needs, measured on
  behaviours the policy never trained on.
- **The mechanism is vocabulary, not skill at writing cycles.** Splitting the 2+-op cycles by whether the
  cycle's exact op text occurs inside some *fit* witness of the same style:

  | witness | cycles whose text is in the fit witnesses | solved | cycles whose text is new | solved |
  |---|---|---|---|---|
  | fused | 45 of 110 (41%) | 0.86 (0.96 / 0.76 / 0.87) | 65 | **0.16** (0.20 / 0.14 / 0.14) |
  | staged | 275 of 305 (90%) | 0.99 (0.99 / 0.98 / 1.00) | 30 | **0.20** (0.27 / 0.17 / 0.17) |

  These are rows, not independent cycles: the new-text rows are **6 distinct staged cycles** (4 behaviours) and
  **13 distinct fused cycles** (12 behaviours), of which a seed solves 1-2 and 3-4. So: a cycle the policy has
  never seen is rarely written in either style, and on this evidence staging does not make the policy better
  at writing a new one; six cycles cannot say more than that. It makes a held-out program consist, nine parts in ten, of
  cycles the training text already contains, against four in ten for fused witnesses, whose one long
  quotation is almost always new. Of the solved multi-op staged cycles, 87% come back as exactly the witness's ops (95% over all solved
  cycles, one-op ones included); that is equality of the returned program, a stricter thing than the
  containment in the table.
- **The same fact read from the search side** is the arms result (`reports/staged-arms-20260920`): the staged
  policy retrieves known cycles, and a token-level beam does not chain them.

## Against the registered prediction

Prediction 2 said staged at least 0.5 of transitions and fused under 0.1. As specified, "share of transitions"
was the wrong quantity (see above). On the quantities that mean something, staged clears 0.5 easily (0.914 of
2+-op transitions, 0.795 of witnesses); fused is **not** under 0.1 (0.445 of 2+-op transitions), because 41%
of its multi-op cycles are short setup idioms that recur across witnesses. End to end the prediction's
direction holds, 0.795 against 0.152.

## Limits

14 behaviours, 5 tasks each; per-behaviour figures in `cycles.md` rest on 5-105 transitions. The sub-goal is
an oracle: nothing here says a learned ranker can pick it. The novel-cycle rate (0.16-0.20) rests on 6
distinct staged and 13 distinct fused cycle texts (30 and 65 rows per seed). `largest_adjacent_sum` is the one behaviour where staged is *worse* per
transition (0.78 against 0.92 overall; 0.60 against 0.56 on 2+ ops).
