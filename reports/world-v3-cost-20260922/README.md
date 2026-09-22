# The pinned system on world data: 1 of 44, and the wall is depth (clear-world-v3)

`goals/g1.md` asks for cost to clear on **v4 + world data**. Every number this
repository holds is v4. This is the world half, measured with the same command
the owner will use:

```
uv run python -m wandering_light.solve world_v3_eval.jsonl.gz
```

`solve.BEST` = `staged-s0-e3` (staged-witness-reauthoring (#155)) at beam 64 ×
depth 48, guard `both`, library on, 16,384-atom state cap; the 44 eval tasks of
`stack_pbe_world_v3`, one per behaviour; `UltraDesk`, CPU. Rows in `rows.json`.

## Result

| | |
|---|---|
| solved | **1 of 44** (coverage **0.023**) |
| wall clock | **71.1 s** over every task attempted |
| executions | 16,312,648 |
| memorised (shown-fit, held-out-fail) | **0** |
| from the library | 0 |
| termination | `max_depth` on **43**, `fit` on 1 |

The one solve is `mbpp-436`, whose witness is 8 ops; the search returned the
same 8-op program, `[ 0 lt? ] filter uncons swap drop`, in 0.166 s.

For comparison, the same system on the v4 cohort solves 180 of 300 (0.600).

## Depth is not the lever: 8x the depth, 25.6x the clock, zero extra solves

The obvious first move was to give the search room, since the witnesses are
longer than it. It buys nothing. Same checkpoint, same beam, same 44 tasks,
only `max_depth` changed:

| depth | solved /44 | wall clock (s) | executions | slowest task (s) | termination |
|---:|---:|---:|---:|---:|---|
| 48 | **1** | 69.7 | 16,312,648 | 3.0 | `max_depth` 43, `fit` 1 |
| 96 | **1** | 196.9 | 29,942,403 | 7.9 | `max_depth` 43, `fit` 1 |
| 192 | **1** | 641.7 | 53,831,092 | 20.0 | `max_depth` 43, `fit` 1 |
| 384 | **1** | 1,787.8 | 99,594,772 | 50.7 | `max_depth` 43, `fit` 1 |

Every unsolved task still ends at `max_depth` at every setting: the beam never
exhausts its frontier, it simply walks further without getting closer. Run dir
`checkpoints/world-depth-20260922`, driver `experiments/best_system_cells.py`
with `--tasks`.

## Why, in the repository's own arithmetic

`docs/STATUS.md` already carries the law this obeys. A beam of width `W` ranked
on cumulative log-probability tolerates about `ln W` nats of accumulated
deficit **over the whole witness**, so a witness of length `L` must hold its
per-step deficit under `ln W / L`. At `W` = 64 that is 4.159 nats total:

| witness length | per-step budget (nats) |
|---:|---:|
| 8 (the one solved task) | 0.520 |
| 48 (the depth limit) | 0.087 |
| 150 (this split's median) | **0.028** |
| 827 (the longest) | 0.005 |

Against that, fused-policy transfer (staged-witness-reauthoring (#155)) measured
a median per-step deficit of **0.075** nats on witnesses the policy was *trained
on*, and **0.535** on unseen content in a familiar style. So at this split's
median length the budget is 0.028 and the best deficit ever measured for this
family of policies is 0.075 — nearly three times over, on training data, before
any of the difficulty of unseen behaviours.

That is the whole result. Depth cannot help because depth is not what is
spent; width cannot help because doubling `W` adds 0.69 nats *in total*, which
at `L` = 150 is 0.005 nats per step. The only term on the left that can move is
`L`. It also explains the pattern exactly: the one solve has an 8-op witness
and a 0.520-nat budget, comfortably above the measured deficits, and three of
the four tasks inside the depth limit still fail at budgets of 0.087 and below.

## The wall is the depth limit, not the beam and not the policy

The 44 eval witnesses run 8 to 827 ops, median **150**. Only **4 of 44** are
within the search's depth of 48, and the system solved 1 of those 4. Every
other task ended at `max_depth`: the search did not run out of beam or of
budget, it ran out of program.

That is a necessary condition for the failure, and the ladder above shows it is
not the operative one: given 384 steps the policy still cannot walk a 150-op
path. best-system-command (#158) measured the width axis on v4 and found 8× the
beam buys +1 verified behaviour; model-size-at-fixed-epoch (#159) found the size
axis flat on v4 and costing 3.9× the clock per node. Neither transfers here, and
the arithmetic above says why none of the three can.

**The honest form of the claim.** Witness length is an upper bound, not a
certified distance, so "40 of 44 witnesses exceed the depth" does not prove no
short program exists for those behaviours — only that the one we know of is too
long and the search found nothing shorter within **384** steps. What is certain is
that the protocol as published cannot clear this split, and that raising
coverage here needs either more depth or a vocabulary that makes the programs
shorter.

Which is the case the highways programme (#96) was opened to make, and what
macro-compression-check (#144) measured: world-v3's witnesses are a median 28.5
tokens in the authoring library's macro form against a median 136 raw `mark-28`
ops. The library that compresses them exists as authoring plumbing; it has
never been given to a solver as actions.

## Zero memorised is worth noticing

On the v4 cohort this system memorises 12 to 33 tasks depending on width —
programs that fit the shown examples and fail the held-out ones. Here it
memorises **none**. With 6 shown examples and a median 21 held-out per task,
and behaviours drawn from MBPP, CodeARC and LeetCode rather than from a
generator, there is no sampler regularity cheap enough to fit. That makes this
split a harder and cleaner test than v4's, and it is the one the owner's brief
names.

## Caveats

One run, one host, one checkpoint, one split of 44 tasks. Coverage is held-out
graded, as `docs/metrics.md` defines it. Below full coverage the 71.1 s is what
was spent without clearing the set, not a cost to clear.
