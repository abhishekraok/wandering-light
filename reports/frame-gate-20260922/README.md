# Inner frames E2, first gate: does a policy shown the frame pay less inside bodies? (#160)

**WEAK, as registered.** Showing a policy the frame its combinator will feed a body lowers calibrated defined-body deficit by 14% on v4 eval. It does not move clears: 13 of 45 at `s0-e5`, against the world arm's 12. Most of the drop is not the frame. Top-level ops, whose input is identical in every arm, drop 12% too. Divided out cell by cell, what the frame adds inside bodies is about 6%.

[Registration](https://github.com/abhishekraok/wandering-light/issues/160#issuecomment-5784663532) and [amendment](https://github.com/abhishekraok/wandering-light/issues/160#issuecomment-5784964005) were posted before any frame-arm cell was measured.

## Setup

`experiments/frame_gate.py`; rows from `experiments/frame_rows.py`.

**Recipe.** The three arms share clear-world-v3 (#161)'s recipe:
- v4 train plus world-v3 train: 429,078 fit rows and 26,040 validation rows, the same for every arm;
- 1m parameters, LR 1.25e-4, batch 256, six epochs, seeds 0-2, every epoch snapshotted, GPU.

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
| v4 tags/world | 1.00 | **1.067** [0.93-1.19] |
| v4 frames/tags | 0.85 | 0.792 [0.74-0.88] |
| v4 top-1 gain, frames over world | +0.01 to +0.04 | +0.030 |
| v4 clears at `s0-e5`, frames | 12-14 (world 12) | **13** (world 12, tags 12) |
| v4 top-level ops, frames/world | within ±10% | **0.876** [0.75-1.01] |
| **world-v3** frames/world | 0.92 (0.75-1.05) | 0.867 [0.79-0.96] |
| world-v3 frames/tags | 0.92 | 0.789 [0.75-0.83] |
| world-v3 top-1 gain | ~0 | +0.018 |
| world-v3 clears at `s0-e5`, frames | 2 (world 2) | 1 (world 2) |
| world-v3 top-level ops, frames/world | within ±10% | 0.922 [0.84-1.01] |

**Gate.** The gate reads clears 13 and ratio 0.862.
- PASS needs clears ≥ 18 and ratio < 0.6.
- FAIL needs clears ≤ 12 and ratio ≥ 0.85, so it missed by one clear.
- Clears across all 18 cells: frames median 12.5 (range 5-17), world 12 (5-14), tags 10.5 (7-13).

**Two registered predictions failed:**
- Top-level ops on v4 moved 12%, against the predicted ±10%.
- The tags control *raised* defined-body deficit by 7%, against a predicted 0%.

## What the numbers say

**Most of the drop is not the frame.**
- Top-level ops see the same input in every arm. Yet the policy trained on frame rows is 12% cheaper on them on v4, and 8% on world-v3.
- Divide the defined-body ratio by the top-level ratio, cell by cell. The body-specific part is then **0.939** on v4 (range 0.79-1.16) and **0.944** on world-v3 (range 0.82-1.06).
- Training on the frame rows made a better policy everywhere, and the frame itself adds about 6% inside bodies.

**The label alone hurts.**
- Tags raises defined-body deficit by 7% on v4 and 11% on world-v3.
- On world-v3 it raises re-encoded body ops by 33%.
- So frames/tags (0.79) overstates what executing the body adds. The fairer baselines are the world arm (0.86) and the top-level control (0.94).

**Where the frame does help**, frames/world on defined bodies by consumer:

| consumer | v4 ops | v4 | world-v3 ops | world-v3 |
|---|---:|---:|---:|---:|
| `filter` | 44 | **0.48** | 142 | **0.65** |
| `dip` | 228 | 0.79 | 1,436 | 0.86 |
| `ifte` | 852 | 0.81 | 3,492 | 0.88 |
| `map` | 30 | 0.81 | 452 | 0.89 |
| `fold` | 206 | **2.43** | 803 | 0.88 |

- `filter` bodies are where the frame shows something the written stack does not.
- v4 `fold` is the exception. The world arm pays a median 22 nats over its 206 fold-body ops, against 61 for tags and 48 for frames. Both arms that carry a label make v4 fold bodies dearer. This is 26-39 nats against a defined-body total of about 575.

**Commitment still binds.** The commitment charge is the same in every arm. With it charged, even free bodies clear 23 of 45 on the world arm. A 6% body-specific gain is far from moving that.

## What this licenses

**WEAK is the owner's call.** The registered reading of PASS, a committing search with a consumer ranker, is not licensed. The frame's gain inside bodies is real but small, and it does not change what clears. The registration named macros under a mark as the next lever on FAIL: macro-action-space (#162), which needs `run_body` to take an ops table (#149). That lever shortens bodies instead of making each step cheaper. Commitment binds under either one.

## Caveats

- **The top-level control carries the whole ops mix of an arm's training.** "About 6%" assumes the general gain applies to body ops in the same proportion as to top-level ops. The per-cell range (0.79-1.16 on v4) is wide.
- **World-v3 is reported, not gated.** The state cap clips 40.7% of its frames-encoded under-mark states, against 28.9% flat. In 3,244 of them it cuts the innermost label (`truncation.json`).
- **Authored witnesses.** Deficit is measured along one authored witness per behaviour, which is not a behaviour's difficulty.

## Files

- **`report.json`**: the verdict, medians, per-pair rows, per-class controls (`controls`) and truncation.
- **`cells.json`**: every measured cell of the three arms (18 each), without per-op rows.
- **`prediction.json`**: the world arm's own numbers that the thresholds were read from.
- **`probe.json`**: the one-epoch learning-rate probe.
- **`truncation.json`**: which end the state cap cuts.

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
