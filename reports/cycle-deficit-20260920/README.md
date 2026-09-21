# Where a staged witness's deficit sits: inside recurring cycles, or between them?

[mine-v4-library (#148)](../../../issues/148). Analysis only: nothing trained, searched or selected.

**Question.** The 35 most ops-saving cycles of the staged v4 corpus compress staged witnesses 1.7x in length (#148 comment). A beam pays in summed per-step deficit (`log p(top legal op) - log p(authored op)`, the `search_proxy.py` quantity), not in length. How much of that would a cycle-macro library remove?

**Protocol.** The staged arm of [staged-witness-reauthoring (#155)](../../../issues/155), seeds 0 / 1 / 2, epoch 3, `checkpoints/staged-arms-20260920/` (local to `UltraDesk`). 137 treated behaviours (staged, not contrived), 4 train tasks each, replayed along the staged witness on CPU; 123 fit behaviours and the 14 validation-skeleton held-out ones reported apart. A **cycle** is a maximal run of adjacent top-level quotations immediately followed by a combinator; anything else at top level, including ops between a `]` and its combinator, is **glue**. The library is the top 35 cycles by ops saved, selected on fit behaviours only. Per occurrence a macro saves at most the cycle's summed deficit (`upper`: the macro is always ranked first) and, if choosing which cycle stays as hard as the cycle's hardest step, `sum - max` (`tail`).

    uv run python experiments/cycle_deficit.py --run-dir checkpoints/staged-arms-20260920 --out reports/cycle-deficit-20260920
    uv run python experiments/cycle_recurrence.py --out reports/cycle-deficit-20260920   # the recurrence counts, `recurrence.json`

## Result

| | fit (123 behaviours) | held out (14) |
|---|---:|---:|
| library cycles, share of steps | 51.7% | 57.2% |
| library cycles, **share of deficit** | **15.1%** | **17.2%** |
| other cycles, share of steps / deficit | 20.0% / 18.2% | 21.6% / 22.0% |
| glue, share of steps / **deficit** | 28.3% / **66.7%** | 21.2% / **60.7%** |
| nats per step: library / other cycle / glue | 0.07 / 0.23 / 0.59 | 0.18 / 0.62 / 1.75 |
| compression in steps | 1.83x | 1.98x |
| **compression in deficit**, tail / upper | **1.01x** / 1.18x | **1.02x** / 1.21x |
| median summed deficit per witness (budget ln 64 = 4.16) | 7.7 | 17.6 |
| behaviours inside the budget: raw / minus tail / minus upper | 37 / 38 / 44 | 1 / 1 / 2 |

**A cycle library halves the length of a staged witness and leaves what the beam pays unchanged.** Library cycles are over half the steps and about a sixth of the deficit; glue is a quarter of the steps and two thirds of the deficit. On held-out behaviours a glue step costs ten times a step inside a library cycle.

Within a cycle the deficit is one spike at the first body token, where the policy commits to which cycle it is writing, and zero after it. Mean nats by position, most used library cycles:

Fit:

- `[` 0.02 `drop` 0.10 `1` 0.00 `add` 0.00 `]` 0.00 `fold` 0.00 (528 scores)
- `[` 0.04 `add` 0.49 `]` 0.00 `fold` 0.00 (360 scores)
- `[` 0.12 `[` 0.31 `lt?` 0.00 `]` 0.00 `[` 0.00 `swap` 0.15 `]` 0.00 `[` 0.00 `]` 0.00 `ifte` 0.00 `drop` 0.00 `]` 0.00 `fold` 0.00 (324 scores)
- `[` 0.17 `cons` 0.03 `]` 0.00 `fold` 0.00 (264 scores)
- `[` 0.12 `swap` 1.10 `]` 0.13 `dip` 0.00 (252 scores)

Held out:

- `[` 0.07 `add` 1.32 `]` 0.00 `fold` 0.00 (84 scores)
- `[` 0.05 `0` 3.91 `[` 0.00 `drop` 0.02 `1` 0.00 `add` 0.00 `]` 0.00 `fold` 0.00 `]` 0.00 `map` 0.33 (36 scores)
- `[` 0.24 `over` 0.02 `uncons` 0.00 `swap` 0.00 `drop` 0.00 `swap` 0.00 `cons` 0.01 `cons` 0.00 `]` 0.00 `fold` 0.00 (36 scores)
- `[` 0.25 `cons` 0.00 `]` 0.00 `fold` 0.00 (36 scores)
- `[` 0.89 `0` 0.48 `swap` 0.59 `lt?` 0.06 `]` 0.00 `filter` 0.00 (36 scores)

A macro keeps the spike (which cycle becomes which macro) and removes the tail, and the tail is already zero. Hence 1.01-1.02x.

Where the glue deficit goes, by op (share of glue deficit, nats per step):

- fit: `nil` 13% (1.15), `dup` 13% (1.28), `swap` 13% (0.42), `uncons` 10% (0.72), `quote` 8% (1.99), `0` 8% (0.61), `over` 8% (2.11)
- held out: `quote` 22% (4.54), `mul` 11% (10.88), `0` 10% (1.37), `dup` 8% (1.73), `[` 7% (2.72), `swap` 6% (0.66), `filter` 6% (6.17)

That is the set-up for the next cycle: the accumulator seed (`nil`, `0`), what to copy (`dup`, `over`), how to pack (`quote`), and the rare top-level arithmetic op (`mul`, 10.9 nats a step held out).

## Reading

- **For a learned policy, a cycle library is cosmetic.** This agrees with #155's retrieval finding from the other side: the recurring cycles are the text the staged policy has memorised, so they are its cheapest steps.
- **The cost to clear sits in the glue**: deciding what the next cycle should consume and produce. That is the sub-goal choice of [closed-state-value-search (#156)](../../../issues/156), not a vocabulary problem.
- A library still helps a solver with no policy (BFS pays `log b` on every step, memorised or not). That case is not measured here.

## Caveats

- Staged witnesses exist only for v4 **train** behaviours. The held-out column is 14 behaviours the policy did not fit, not the 43-behaviour eval cohort.
- The deficit is measured against the greedy-best op at the witness's own state, a heuristic for what a beam tolerates (#151: 5/5 precision, 7 false negatives of 43).
- `tail` and `upper` bracket a macro head's deficit without training one. The bracket is 1.01-1.21x, against the 6.7x median the unsolved cohort needs.
- Ops between a `]` and its combinator count as glue, so a quotation run late is not a cycle here; 20-22% of steps are in cycles outside the library.
