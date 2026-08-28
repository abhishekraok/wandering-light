# `mark-28` reproduction

Date: 2026-08-27 · solver-build-quotations (#60) · branch point `06d8a93`

All arrows below are **published target -> observed**. Every measurement
reproduced at the published precision.

## Baselines

| basis | ops | `b_eff` | bits/step | growth/depth | result |
|---|---:|---:|---:|---:|---|
| `executable-26` | 26 -> 26 | 13.38 -> 13.377956 | 3.74 -> 3.741786 | 7.3x -> 7.288767x | **REPRODUCED** |
| `full-52` | 52 -> 52 | 37.33 -> 37.328793 | 5.22 -> 5.222217 | 32.4x -> 32.398463x | **REPRODUCED** |

STATUS records the older 13.32 and 37.28 measurements. The bounded protocol
gives the near-match above; it does not give those older figures exactly.

## New basis

| basis | ops | `b_eff` | bits/step | growth/depth | result |
|---|---:|---:|---:|---:|---|
| `mark-28` | 28 -> 28 | 23.11 -> 23.109761 | 4.53 -> 4.530430 | 22.5x -> 22.471228x | **REPRODUCED** |

`b_eff` is pooled legal edges / expanded states through depth 4 over the four
existing roots. Growth is the mean of the final two adjacent new-frontier
ratios on the one-list root; `full-52` stops at depth 4 and the other bases at
depth 5.

Reproduce:

```bash
uv run python experiments/stack_machine_bfs.py --summary --basis executable-26
uv run python experiments/stack_machine_bfs.py --summary --basis full-52
uv run python experiments/stack_machine_bfs.py --summary --basis mark-28
```

## Program costs

| program | `full-52` | `mark-28` | observed answer | result |
|---|---:|---:|---:|---|
| even-square sum | 37 -> 37 ops; fuel 118 -> 118 | 25 -> 25 ops; fuel 106 -> 106 | `56` | **REPRODUCED** |
| 12-item predicate build | 25 -> 25 ops | 14 -> 14 ops | `(dup 1 1 add div 1 1 add mul sub 0 eq?)` | **REPRODUCED** |
| recursive list sum | 53 -> 53 ops; fuel 149 -> 149 | 33 -> 33 ops; fuel 129 -> 129 | `10` | **REPRODUCED** |

The even-square input was `(3 4 1 2 7 6)`; recursive sum used
`(0 1 2 3 4)`.

The exact six programs, inputs, outputs, op counts, and recursive fuel are
pinned by `test_published_mark_28_program_costs_and_outputs`:

```bash
uv run pytest tests/test_stack_machine_quotations.py \
  -k published_mark_28_program_costs_and_outputs
```

## State-space split

These are newly discovered frontier states, not cumulative states.

| depth | all states | quote-closed | closed % | result |
|---:|---:|---:|---:|---|
| 3 | 1,842 -> 1,842 | 721 -> 721 | 39.1% -> 39.142% | **REPRODUCED** |
| 4 | 37,312 -> 37,312 | 6,298 -> 6,298 | 16.9% -> 16.879% | **REPRODUCED** |

## Verification

| check | result |
|---|---|
| required stack-machine tests | **PASS** — 78 passed, 1 skipped |
| stack-machine plus non-slow corpus tests | **PASS** — 99 passed, 1 skipped |
| changed-file Ruff check and format check | **PASS** |
| `./lint.sh check` | **BLOCKED / NOT CLEAN** — the same seven findings occur on clean `main`: two unused test bindings and five findings under `wandering_light/training/`. The task forbids touching training; this change adds no finding. |

## Verdict

Yes. With `MARK` implemented as a distinct structural value tagged
`("mark",)`, `mark-28` reproduces every branching, program-cost, fuel, and
state-split measurement requested here. It makes quotation construction
representable at 28 ops while preserving state-key sufficiency. The 16.9%
quote-closed share at depth 4 also reproduces the stated limitation: this
measurement does not establish tractability or justify adoption without the
separate corpus study.
