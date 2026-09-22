# A clean v4 + full world-v3 policy: the data lever is real, is saturating, and moves no coverage (clear-world-v3 #161)

clear-world-v3 (#161) found `solve.BEST` — trained on `stack_pbe_v4` alone —
sitting a median **4.19 nats per step** behind its own top choice along
world-v3's eval witnesses, and the two world-trained checkpoints that happened
to exist, the `random` and `ablated` arms of keep-below-ablation (#136), at
**0.76** and **1.10**. Both were built for a different ablation and each dropped
159 world behaviours from training, so the effect was measured and its ceiling
was not. This is the arm that should have been the baseline: v4 plus the
**whole** world-v3 train split.

`experiments/world_arm.py`, run dir `checkpoints/world-arm-20260922`. #101's 1m
recipe — LR 1.25e-4, batch 256, six epochs, three seeds, every epoch
snapshotted — over 429,078 fit rows and 26,040 validation rows across 786
behaviours, validation pinned to the **v4** holdout as world-plus-v4-policy
(#134) pinned it, dropping the three behaviours #134 found sharing a witness
with a v4 eval behaviour. The row counts reproduce #134's mixed arm exactly.

## The data lever, measured against everything before it

Median per-step deficit along the same 44 world-v3 eval witnesses:

| policy | trained on | **deficit (nats)** | over budget | top-1 share |
|---|---|---:|---:|---:|
| `staged-s0-e3` (**BEST**) | v4 staged | 4.19 | 150.7× | 0.333 |
| `fused-s0-e3` | v4 fused | 3.79 | 136.4× | 0.405 |
| `refused-s0-e3` | v4 re-fused | 3.29 | 118.3× | 0.361 |
| #136 `ablated` | v4 + world − 159 | 1.10 | 39.6× | 0.582 |
| #136 `random` | v4 + world − 159 | 0.76 | 27.2× | 0.696 |
| **this arm** | v4 + **full** world-v3 | **0.66** | **23.9×** | 0.745 |

Seeing world data is worth a factor of **5 to 6**. That is the largest single
effect measured on this split.

## But it is saturating, and that is the number that matters

Adding the 159 behaviours the #136 arms were missing moves the deficit from
**0.76 to 0.66** — **13%** for a third more world training data. The deficit is
also flat in training length: the per-epoch means run 0.664 / 0.668 / 0.723 /
0.676 / 0.685 / 0.715 over epochs 1–6, a spread of 0.06 against a seed spread of
up to 0.14 within a single epoch, so no epoch is distinguishable from another.

Top-1 share does rise with training — 0.645 at epoch 1 to 0.745 at epoch 6 — so
the policy becomes more often exactly right while the cost of the steps it gets
wrong does not fall. On a beam that spends its budget on the wrong steps, the
second quantity is the one that decides.

The remaining gap is **23.9×**: 0.66 nats per step against a budget of 0.0278 at
this split's median 150-op witness. At the observed rate of return — a third
more data for 13% — closing that with more world behaviours is not a plausible
route. training-data-sources (#118) has roughly 127 left in the
sampler-passing pool, which on this curve would be worth a few per cent.

## Coverage does not move at all

The acceptance test, the same search `solve.BEST` runs, seed 0:

| epoch | deficit | solved /44 | coverage | wall clock | memorised |
|---|---:|---:|---:|---:|---:|
| 1 | 0.635 | **0** | 0.000 | 79.9 s | 0 |
| 3 | 0.776 | **1** | 0.023 | 81.0 s | 0 |

Against `BEST`'s 1 of 44. **A 6.3× reduction in per-step deficit produces no
change in coverage**, which is the sharpest form of the methodological point
this issue has been making: at this range the metric that steers and the metric
that accepts are decoupled, and a campaign judged on coverage alone would have
reported this arm as a total failure indistinguishable from doing nothing.

It is also why the earlier depth ladder could burn 25.6× the clock and learn
nothing: coverage was the only thing being watched.

## Reading

The data lever works and will not finish the job. Three axes were already flat
on v4 — width (best-system-command, #158), size (model-size-at-fixed-epoch,
#159), training length (epoch-selection-in-behaviours, #154) — and depth is flat
on world data (#161). Data is the first lever that moves anything here, by a
factor of five to six, and it is visibly saturating a third of the way through
its own corpus.

What is left is the term the budget law says nobody has touched: the **length**
of the programs. `ln W / L` at this split's median `L` of 150 is 0.0278 nats;
at the authoring library's median macro form of 28.5 tokens it is 0.146.
Composing that with this arm's 0.66 gives 18.8 nats against a 4.159 budget —
**4.5×** over rather than 150×. That is arithmetic on today's numbers and not a
forecast: a policy over macro actions has its own deficits and nobody has
measured them. It is the only remaining combination inside an order of
magnitude, and measuring it is the next thing worth a campaign.

## Caveats

Three seeds, one corpus, one recipe, one 44-task split. The epoch with the
lowest deficit is chosen on the eval split, which is selection on eval; the
deficit is flat across epochs, so this changes nothing, and both acceptance runs
are reported rather than the better one. Deficit is measured along **authored
witnesses**, which is not the difficulty of a behaviour — the same search finds
other programs, which is why coverage and deficit can disagree, and is the
caveat the Superseded entry in `docs/STATUS.md` records.
