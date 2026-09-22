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

## Why — measured on this split, after a first answer that was wrong

`docs/STATUS.md` carries a budget law: a beam of width `W` tolerates about
`ln W` nats of accumulated deficit **over the whole witness**, so a witness of
length `L` must hold its per-step deficit under `ln W / L`. At `W` = 64 that is
4.159 nats, which is 0.028 per step at this split's median length of 150 ops.

My first reading of the ladder paired that budget with the **0.075** nats per
step that staged-witness-reauthoring (#155) measured, and concluded the gap was
a factor of three that shortening `L` would close. That was wrong: 0.075 is a
number for witnesses the policy was **trained on**. `experiments/witness_deficit.py`
measures it where the law is being applied — the pinned policy walked along each
eval witness, recording at every step how many nats behind its top legal choice
the witness's own op is:

| corpus, eval split | witnesses | median `L` | **median per-step deficit** | budget at median `L` | over by | within the whole budget |
|---|---:|---:|---:|---:|---:|---:|
| `stack_pbe_v4` | 45 | 40 | **1.84** | 0.104 | 17.7× | 5 of 45 |
| `stack_pbe_world_v3` | 44 | 150 | **4.19** | 0.028 | **150.7×** | 1 of 44 |

The median **survival depth** — the step at which accumulated deficit first
exceeds the budget, so the step a beam of 64 would drop the witness — is **2**
on both corpora. The policy's own top legal choice is the witness's next op on
33% of world-v3 steps and 50% of v4 eval steps.

**So shortening `L` does not close this gap either.** At the macro form's median
of 28.5 tokens (macro-compression-check, #144) the budget would be 0.146 nats
per step against a measured 4.19 — still **28.7×** over. At the measured
deficit, a witness would have to be **under one op** on world-v3, or 2.3 on v4
eval, to fit inside `ln 64`. No reachable beam width helps either: carrying
world-v3's median witness would need `ln W` ≥ 627.

## What that argument can and cannot carry

It cannot carry a conclusion about behaviours, in either direction, and that is
the lesson rather than a caveat. The same system solves **180 of 300** v4
cohort tasks and grades 18 behaviours while only **5 of 45** v4 eval witnesses
are within budget. The search plainly does not follow the authored witness: it
finds other, easier programs. `docs/STATUS.md` already records the law as
conservative — 83.7% accurate with **0 false positives and 7 false negatives** —
and these numbers are that same conservatism at a larger scale.

What the deficit measurement does establish is narrow and worth having: along
the programs we know solve these behaviours, this policy is nowhere close, on
both corpora, and three times further away on world data. What it does **not**
establish is that a shorter vocabulary would or would not help, because the
quantity it measures is the deficit along *a particular program*, and a macro
library changes which programs exist.

The empirical result above — 8× the depth, 25.6× the clock, zero extra solves —
stands on its own and does not depend on any of this.

## The wall is the depth limit, not the beam and not the policy## The wall is the depth limit, not the beam and not the policy

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
