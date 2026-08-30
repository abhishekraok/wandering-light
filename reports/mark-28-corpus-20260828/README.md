# `mark-28` corpus probe — the cheaper branching bought nothing

Date: 2026-08-28 · [solver-build-quotations (#60)](../../issues/60) step 2 · branch point `5bf9ab7`

#60's own plan gates everything downstream on this step: *regenerate a corpus
against `mark-28` and compare certified depth and the share of tasks that use a
combinator.* Its stated bar — "if the corpus is not more expressive, the cheaper
branching bought nothing."

**The gate is not met.**

## Like-for-like at depth 5

Same generator, same seed, same `--per-depth 40`, one 16-core desktop.

| | `executable-26` | `mark-28` |
|---|---:|---:|
| tasks | 1,417 | 1,417 |
| generation seconds | 130 | **1,364** |
| uses a combinator | 95 (6.7%) | 99 (**7.0%**) |
| runs code at all | 175 (12.4%) | 176 (12.4%) |
| builds a quotation | 0 (0.0%) | 61 (4.3%) |

The combinator share moves **0.3 points**. The share that runs code at all does
not move. Against #59's 5.7% figure on the committed depth-6 corpus, neither
basis clears it by anything worth the 10.5x in generation cost.

## At depths 2, 3 and 4 the corpora are identical

Not similar — **identical**. Same task counts (175 / 558 / 977), same programs,
zero uses of `[` or `]`, at up to 4.4x the cost. `mark-28` first produces a
different corpus at depth 5, which is the first depth at which `[ x ] map`
(five ops) can exist at all.

## The 61 quotation builders are degenerate

| | |
|---|---:|
| execute what they built | **2 of 61 (3.3%)** |
| body length 1 op | 34 |
| body length 2 ops | 27 |
| body length 3+ | 0 |

Every body is an op fragment — `add nil`, `filter quote`, `swap mul`,
`uncons uncons`. These are not program bodies. `mark-28` is being used to put a
**symbol on the stack as data**, which `executable-26` cannot do at all, and
then `cons` or `cat` it. Fifty-nine of sixty-one never run what they build.

## Why, and it is not the basis's fault

Tasks are certified **shortest**, and the families **hand quotations over in the
input**. A built quotation can therefore never win against a supplied one:
`[ 1 add ] map` is five ops where `(1 add) map` is one. So the generator selects
against the very construction `mark-28` was added to enable, at every depth,
regardless of basis.

This is [stack-corpus-no-higher-order (#59)](../../issues/59)'s circularity one
level up. The basis decides what is *reachable*; the generator decides what is
*selected*. Enriching the basis while the generator still supplies quotations
and certifies shortest programs cannot produce higher-order tasks.

**The lever is the generator, not the op set.** A corpus that exercises
quotation construction needs families that stop supplying quotations — and that
is a change with its own cost, since #59 records that the families supply them
"precisely so the combinators are reachable at all".

## Cost, for the record

`mark-28` certifies one depth shallower for the same budget. Depth 5 costs
1,364 s against 130 s; at its measured 22.5x growth per depth, depth 6 projects
to roughly eight hours against fifty minutes. The pre-registered estimate from
the published growth rates was a loss of one to two depths; the measured loss is
one.

## Reproduce

```bash
uv run python experiments/mark28_corpus_probe.py --depth 5 --dir corpora
```

Generation is ~25 minutes for the `mark-28` arm. The script skips a corpus that
already exists, so the comparison re-runs in seconds.

## What this does not say

`mark-28` itself behaves exactly as #60 claims — that was verified separately in
`reports/mark-28-20260827/` and independently reproduced. Nothing here argues it
should be reverted; it is a strictly additive basis behind a flag, and it is the
only basis in the repo that *can* express quotation construction. What this says
is narrower: **adopting it as the corpus basis, on the current generator, buys
0.3 points of combinator share for 10.5x the generation cost.**
