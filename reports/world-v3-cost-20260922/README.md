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

## The wall is the depth limit, not the beam and not the policy

The 44 eval witnesses run 8 to 827 ops, median **150**. Only **4 of 44** are
within the search's depth of 48, and the system solved 1 of those 4. Every
other task ended at `max_depth`: the search did not run out of beam or of
budget, it ran out of program.

That bounds what the other levers can do here. best-system-command (#158)
measured the width axis on v4 and found 8× the beam buys +1 verified behaviour;
on world data width cannot help at all while the depth limit binds, because a
wider beam explores more programs of the *same* length.
model-size-at-fixed-epoch (#159) found the size axis flat on v4 and costing 3.9×
the clock per node; a better policy cannot emit a 150-op program in 48 steps
either.

**The honest form of the claim.** Witness length is an upper bound, not a
certified distance, so "40 of 44 witnesses exceed the depth" does not prove no
short program exists for those behaviours — only that the one we know of is too
long and the search found nothing shorter within 48 steps. What is certain is
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
