# Inner frames E2, first gate: does a policy shown the frame pay less inside bodies? (#160)

**WEAK, as registered.**
- **Clears barely move.** Showing a policy the frame its combinator will feed a body lowers calibrated defined-body deficit by 14% on v4 eval and 13% on world-v3. v4 clears go from 12 to 13 at `s0-e5`. Per cell, frames beats world by a median of one clear (10 cells up, 3 down, sign test p = 0.09). FAIL missed by one clear.
- **One open level is the clearest positive.** There the frames arm pays 0.62 of the world arm's deficit on v4 (0.68 without one outlier witness) and 0.84 on world-v3. Deeper, it pays 0.87-0.93.
- **The label alone hurts on world-v3 only**, and more with nesting. On v4 its cost is that one witness.
- **Not identified on v4:** how much of the drop is the frame acting inside the body, rather than a policy trained on frame rows being better at everything.

[Registration](https://github.com/abhishekraok/wandering-light/issues/160#issuecomment-5784663532) and [amendment](https://github.com/abhishekraok/wandering-light/issues/160#issuecomment-5784964005) were posted before any frame-arm cell was measured.

## Setup

`experiments/frame_gate.py`; rows from `experiments/frame_rows.py`.

**Recipe.** The three arms share clear-world-v3 (#161)'s recipe:
- v4 train plus world-v3 train: 429,078 fit rows and 26,040 validation rows, the same for every arm;
- the `1m` architecture (655,836 parameters), LR 1.25e-4, batch 256, six epochs, seeds 0-2, every epoch snapshotted, GPU.

**Arms.** At body steps where E1's capture-aware walk defines an open level's frame:
- **world** (the machine state, already trained) sees nothing added;
- **tags** adds a label after a separator: consumer, `ifte` role, capture count;
- **frames** adds the label plus the frame after the body prefix.

**Measure.** Every cell is measured on one witness per eval behaviour: 45 on v4, 44 on world-v3.
- **Deficit.** Each op's deficit is its gap to the best legal op, divided by the checkpoint's own temperature. The temperature is fitted on its validation rows, in its own encoding, under the same legal mask.
- **Ratio.** A ratio is of calibrated deficit totals, taken as the median over the 18 matched (seed, epoch) pairs.
- **Clears.** Both decompositions are used, and each `[` is charged the out-of-sample surprisal of its consumer.

No temperature sat at the edge of its search. The frames arm's temperature, over the world arm's in the same cell, has median 1.006 and range 0.93-1.03, so calibration does not drive the ratios.

## Against the registered predictions

| | predicted | measured, median of 18 [range] |
|---|---:|---:|
| **v4** frames/world, defined bodies | 0.85 (0.70-1.00) | **0.862** [0.74-0.93] |
| v4 tags/world | 1.00 | 1.067 [0.93-1.19]; 0.968 without `bounds_cover_list_flag` |
| v4 frames/tags | 0.85 | 0.792 [0.74-0.88] |
| v4 top-1 gain, frames over world | +0.01 to +0.04 | +0.030 |
| v4 clears at `s0-e5`, frames | 12-14 (world 12) | **13** (world 12, tags 12) |
| v4 top-level ops, frames/world | within ±10% | 0.876 [0.75-1.01] |
| v4 untouched body ops, frames/world | within ±10% | 0.915 [0.75-1.41] |
| **world-v3** frames/world | 0.92 (0.75-1.05) | 0.867 [0.79-0.96] |
| world-v3 tags/world | 1.00 | 1.111 [1.01-1.16] |
| world-v3 frames/tags | 0.92 | 0.789 [0.75-0.83] |
| world-v3 top-1 gain | ~0 | +0.018 |
| world-v3 clears at `s0-e5`, frames | 2 (world 2) | 1 (world 2) |
| world-v3 top-level ops, frames/world | within ±10% | 0.922 [0.84-1.01] |
| world-v3 untouched body ops, frames/world | within ±10% | 1.201 [0.87-1.40] |

**Gate.** The gate reads clears 13 and ratio 0.862.
- PASS needs clears ≥ 18 and ratio < 0.6.
- FAIL needs clears ≤ 12 and ratio ≥ 0.85, so it missed by one clear.

**Held:** the frames/world ratio on both corpora, the v4 top-1 gain, v4 clears, and v4 untouched ops.

**Missed:**
- **v4 top-level ops**, 12% against ±10%.
- **tags/world**: 1.07 on v4, which is one witness (0.97 without it), and 1.11 on world-v3.
- **world-v3 untouched ops**, 1.20: 148 ops, 140 of them in `leetcode-count-square-sum-triples`.
- **world-v3 clears**: 1 against 2.
- **frames/tags** came in lower than predicted on both corpora, because the label cost more than predicted.

## What the numbers say

**Depth is the clearest split.** Defined body ops by number of open levels:

| open levels | v4 ops | tags/world | frames/world | world-v3 ops | tags/world | frames/world |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 384 | 0.80 | **0.62** | 2,181 | 0.99 | **0.84** |
| 1, without `bounds_cover_list_flag` | 361 | 0.83 | **0.68** | | | |
| 2 or more | 976 | 1.15 | 0.93 | 4,144 | 1.23 | 0.89 |
| 2 or more, without it | 917 | 1.01 | 0.87 | | | |
| 1, tags state unclipped | | | | 1,882 | 0.98 | 0.85 |
| 2 or more, tags state unclipped | | | | 2,720 | 1.17 | 0.89 |

- **The frame helps at every depth, and most where one level is open.** On v4 that is 0.68 against 0.87 without the outlier. On world-v3 it is 0.84 against 0.89.
- **The label goes from helping to hurting as levels stack.** On world-v3 it moves from 0.98 to 1.17, on states the cap does not clip. On v4, without the outlier, its gain fades from 0.83 to 1.01.
- **Clipping adds to this but is not the mechanism.** On world-v3, tags-clipped defined ops read 1.24.
- **Encoding or depth?** The split alone cannot say. Deeper bodies are different bodies, further from the spec, and a frame may help them less for that reason alone. What points at the encoding is that correct information should not raise deficit, yet the stacked label does on world-v3. An encoding that shows only the innermost level would separate the two, since the same deeper ops are then measured under both.
- **This is not a gate reading.** 0.62 is on 384 of the 1,360 v4 defined ops.

**One v4 witness carries the label's v4 cost.**
- `bounds_cover_list_flag` is 114 of the world arm's 575 v4 defined-body nats. It nests a min/max idiom, `[ [ over lt? ] [ swap drop ] [ drop ] ifte ] fold`, under two and three `dip`s. Those chains are rare in the training rows: 144 and 36 fit steps.
- Its 10 fold-body ops cost a median 8.0 nats on the world arm, 47.8 on tags and 34.4 on frames. Across the other 22 v4 witnesses with fold bodies, no arm differs from the world arm by more than 0.63 nats (`fold_check.py`).
- Without it, v4 reads frames/world 0.818, tags/world 0.968 and frames/tags 0.834.

**By consumer.** The label does the work on `filter`; the frame adds most over the label on `ifte`. v4 without the outlier, and world-v3:

| consumer | v4 ops | frames/world | tags/world | frames/tags | world-v3 ops | frames/world | tags/world | frames/tags |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ifte` | 805 | 0.79 | 0.98 | **0.81** | 3,492 | 0.88 | 1.13 | 0.79 |
| `dip` | 203 | 0.96 | 0.96 | 1.02 | 1,436 | 0.86 | 1.07 | 0.80 |
| `fold` | 196 | 0.98 | 0.96 | 1.02 | 803 | 0.88 | 1.31 | 0.70 |
| `filter` | 44 | 0.48 | **0.53** | 0.93 | 142 | 0.65 | 0.79 | 0.80 |
| `map` | 30 | 0.81 | 0.88 | 0.84 | 452 | 0.89 | 1.03 | 0.86 |

**Top-level ops, and what share of the drop is in-body.**
- On the frames arm, top-level ops are 12% cheaper on v4 (0.876) and 8% on world-v3 (0.922), though their input is the same in every arm.
- On v4 most of that is at the `[` and consumer ops around bodies (0.70), which the label alone also cuts (0.82). Direct top-level ops read 0.92 for frames and 1.04 for tags, so the frame improves ops it is never shown at. Frame rows are extra state-to-op supervision.
- The design therefore cannot separate the frame as an input inside a body from the frame as a training signal.
- **On v4 the in-body share is not identified.** Dividing the defined-body ratio by a same-input control, cell by cell, gives 0.94 against all top-level ops, 0.84 against direct ones and 0.93 against untouched body ops. Net of a same-input control, then, the in-body reduction lies anywhere from 6% to 16%, depending on the control.
- **On world-v3 it is about 6%** against both top-level controls (0.94). Untouched ops there moved the other way (1.20, one witness).

**Commitment still binds.**
- The commitment charge is the same in every arm.
- With it charged, even free *defined-body* ops clear only 23 of 45 v4 witnesses on the world arm. Re-encoded and untouched bodies stay charged.

## What this licenses

**WEAK is the owner's call.** The registered PASS reading, a committing search with a consumer ranker, is not licensed.

**Proposed next campaign**, one GPU campaign, with its own registration. This is the advisor session's design and mine. It has four arms:
- **Innermost-only encoding, tags and frames.** Show only the innermost defined level's label and frame, and leave ancestors as the machine shows them.
  - Predicted: frames/world on v4 defined ops at 2+ levels about 0.75, and depth 1 stays near 0.62. If depth 1 moves too, something else changed.
  - The discriminating tags reading is world-v3's unclipped 2+ levels (1.17 stacked). v4's is already about 1.0 without the outlier.
- **An exit-frame oracle**, on the innermost-only encoding: frames plus the frame the body must produce, taken from the authored witness. It licenses nothing directly. It answers whether a body lacks a *specification* or whether the policy is the bottleneck, and that question is coarse enough that the encoding cannot flip it.
  - Under 0.5 of the world arm, with 18 or more clears charged, points to skeleton-first search: choose the consumer and the hole, derive the body's sub-spec, solve it under its own `ln W`.
  - Above 0.8 means specification is not the lever.
  - Report the position split, since the exit frame makes a body's last op nearly free by construction.
- **A learned commitment ranker**, trained on the innermost-only rows: a head that predicts the label (consumer, role, captures) from state and spec at each `[`, instead of taking it as input. The shape-only prior charges 1.24 nats per `[`. At current bodies, charged against uncharged clears are 12 against 18, so a good ranker is worth up to 6 v4 clears. On `filter`, the label alone halves the world arm's deficit.

**The innermost-only arm cannot turn E2 into a PASS.** On the world arm's exchange rate, 0.75 gives about 14 clears, and even 0.60 everywhere gives about 16; PASS needs 18 and under 0.6. So that arm re-reads how big the frame's effect is, and keeps the oracle on the better encoding. The decision is carried by the oracle and the ranker.

**After that: macros under a mark**, macro-action-space (#162), which needs `run_body` to take an ops table (#149). This is the lever the registration named on FAIL.
- It shortens bodies instead of cheapening steps.
- mine-v4-library (#148) is a prior against it: cycle macros bought 1.72x length compression but only 1.02x deficit.
- It targets world-v3, which the owner is replacing with a smaller eval, so it is best decided on that eval.

## Caveats

- **One witness moves v4.** `bounds_cover_list_flag` is a fifth of v4's defined-body deficit, so every v4 ratio is given with and without it where it matters.
- **World-v3 is reported, not gated.** The state cap clips 40.7% of its frames-encoded under-mark states, against 28.9% flat. In 3,244 of them it cuts the innermost label (`truncation.json`).
- **Authored witnesses.** Deficit is measured along one authored witness per behaviour, which is not a behaviour's difficulty.

## Files

- **`report.json`**: the verdict, medians, per-pair rows, per-class controls (`controls`) and truncation.
- **`cells.json`**: every measured cell of the three arms (18 each), without per-op rows.
- **`prediction.json`**: the world arm's own numbers that the thresholds were read from.
- **`probe.json`**: the one-epoch learning-rate probe.
- **`truncation.json`**: which end the state cap cuts.
- **`splits.py`**, **`splits.json`**: ratios on op subsets (open levels, clipping, the outlier, consumers, top-level kinds) and per-cell quotients against the same-input controls.
- **`fold_check.py`**: which witnesses carry the defined `fold` deficit.

Checkpoints are in `checkpoints/frame-arm-20260922/{frames,tags}/policies/1m-s{0,1,2}-e{1..6}` (gitignored, 163 MB of per-op rows under `measure/`).

Reproduce:

```
uv run python experiments/frame_gate.py train --arm frames --device cuda
uv run python experiments/frame_gate.py train --arm tags --device cuda
uv run python experiments/frame_gate.py measure --arm frames --cells all
uv run python experiments/frame_gate.py measure --arm tags --cells all
uv run python experiments/frame_gate.py truncation
uv run python experiments/frame_gate.py report
```

The world arm's 18 cells come from the same `measure --arm world --cells all`, on `checkpoints/world-arm-20260922`.
