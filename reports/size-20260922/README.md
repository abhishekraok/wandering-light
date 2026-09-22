# Policy size at a fixed epoch: flat (model-size-at-fixed-epoch #159)

compute-optimal-mark-28 (#101) found the size axis flat and selected every arm
on validation cross-entropy, which stopped them at epoch 1 or 2.
epoch-selection-in-behaviours (#154) then showed that rule picks 6.00 verified
behaviours where a fixed later epoch reaches 11.00, and `docs/STATUS.md`
retracted the selection rule. So the one conclusion this repository holds about
model size rested on a rule it has since abandoned. This campaign asks the
question again with the checkpoint allowed to train.

**The answer is the same: flat, and on the objective it is worse.** At the
pre-registered epoch 3, going from 655,836 to 6,288,412 parameters — **9.6×** —
moves verified behaviours from **15.67 to 16.33 of 43**. That is the prediction
registered on the issue before the run ("all three sizes within one verified
behaviour of each other at epoch 3"), which holds, and it is weaker than it
looks: pooled over all twelve cells per size the arms are **15.58 ± 0.64 /
16.17 ± 0.69 / 16.17 ± 0.90**, so 2m and 10m are identical and the between-size
gap is smaller than the seed spread it sits inside (median 1 behaviour, up to
2, within a single size and epoch). The 10m arm also produced the campaign's
single worst cell, 14.

And it costs. The larger policy expands **fewer** nodes — 1,744 / 1,703 / 1,669
per task, pooled — while costing **1.19 / 2.20 / 4.69 ms per expansion**, so it
buys no reduction in search and pays about **3.9× the wall clock for the same
search**. Under `docs/metrics.md`, which compares coverage first and then cost,
a 10m policy is therefore not an improvement on a 1m one here: its coverage is
indistinguishable and its cost is multiples higher.

## What ran

`experiments/size_at_fixed_epoch.py`. The staged arm of
staged-witness-reauthoring (#155) — `arm_rows("staged")`, draw seed 20260920,
the 6 contrived rows excluded, **123 treated behaviours, 4,920 of 6,360 fit
tasks, 313,560 rows per epoch** — trained at 1m / 2m / 10m, seeds 0 / 1 / 2, 6
epochs, every epoch snapshotted, at #101's probe learning rates (1.25e-4 /
6.25e-5 / 3.125e-5), batch 256, validating on the fused holdout. Then the #101
300-task / 43-behaviour cohort at beam 64 × depth 48, guard `both`, library
off, with the 16,384-atom state cap of beam-state-size-unbounded (#157), for
epochs 3–6 of every seed. Every distinct solved program is draw-tested through
`experiments/step3_verify_solves.py`; **verified** is `genuine` or
catalogue-witness-equal.

**The gate passed bitwise.** It was registered as "validation CE equal to four
decimals", weakened from bitwise because CUDA's index kernels are
nondeterministic without `torch.use_deterministic_algorithms(True)` and #155
trained on CUDA. In the event the retrained 1m seed-0 epoch-3 checkpoint is
**byte-identical** to the published `staged-s0-e3`: same sha256, max |Δw| =
0.0, CE 2.6756505558933896 either way. This arm is #155's staged arm continued,
not a lookalike, and the 1m epoch-3 row below reproduces #155's published
staged figure of 15.67 with the same per-seed triple.

## Verified behaviours of 43, mean over three seeds

| epoch | 1m (655,836) | 2m (2,045,660) | 10m (6,288,412) |
|---|---:|---:|---:|
| **3** (pre-registered) | **15.67** (16 / 16 / 15) | **16.00** (16 / 16 / 16) | **16.33** (16 / 17 / 16) |
| 4 | 16.00 (16 / 16 / 16) | 16.67 (17 / 17 / 16) | 16.67 (17 / 17 / 16) |
| 5 | 15.67 (15 / 17 / 15) | 16.33 (17 / 16 / 16) | 16.67 (17 / 16 / 17) |
| 6 | 15.00 (15 / 15 / 15) | 15.67 (17 / 15 / 15) | 15.00 (14 / 16 / 15) |

Graded tasks of 300, mean over seeds: 139.7 / 141.3 / 141.0 at epoch 3, and
110.3 / 114.3 / 125.3 by epoch 6. Epoch 4 is the best epoch for every size, but
for 10m it only **ties** epoch 5 at 16.67. The arm names are nominal: they are
0.66M, 2.0M and 6.3M parameters, a 9.6× span rather than the 10× the labels
suggest.

## Cross-entropy moves and behaviours do not

| size | parameters | validation CE at epoch 3 (seed 0) | verified behaviours at epoch 3 |
|---|---:|---:|---:|
| 1m | 655,836 | 2.676 | 15.67 |
| 2m | 2,045,660 | 2.384 | 16.00 |
| 10m | 6,288,412 | 2.247 | 16.33 |

**0.43 nats of held-out cross-entropy buys 0.67 behaviours.** The larger policy
is a better model of the witness text by a wide margin and barely a better
solver, which is epoch-selection-in-behaviours (#154)'s finding restated on the
size axis rather than the epoch axis: cross-entropy on witness text is not the
quantity that decides whether a search clears a behaviour.

By witness-length bucket at epoch 3 (1–5 / 6–10 / 11–20 / 21–40 / 41+), mean
over seeds: 1m 6.00 / 3.00 / 5.00 / 1.67 / 0.00, 2m 5.67 / 3.00 / 5.00 / 2.33 /
0.00, 10m 6.00 / 3.00 / 5.00 / 2.00 / 0.33. Everything size buys sits in the
21–40 bucket and one 41+ solve at 10m; the short buckets are identical. Over
the union of three seeds at epoch 3 each larger arm adds exactly one behaviour
the 1m arm never gets and loses none — but **not the same one**: 2m adds
`prefix_sums` and 10m adds `pair_floor_averages`. That is one more reason to
read the difference as seed-level noise rather than a capacity ladder.

## Reading

Size is closed as a lever at fixed epochs, as it was at CE-selected ones, and
now for reasons the earlier campaign could not give: the checkpoints were
allowed to train, the epoch was pre-registered rather than chosen on the eval
cohort, the gain is under one behaviour for an order of magnitude of parameters
and is not separated from initialisation noise, and the larger policy is
**strictly worse on the objective** once its per-node cost is charged. Taken with staged-witness-reauthoring (#155), which found the
staged arm's gain is **retrieval** of cycles occurring verbatim in its training
text, and with best-system-command (#158), which finds the width axis nearly
exhausted on this checkpoint (8× the beam for +1 verified behaviour), three
axes that could have bound — parameters, search width, training length — are
all flat at roughly the same place. What is left is the data: 175 distinct
training programs, which is what compute-optimal-mark-28 (#101) concluded
binds.

## What this does not say

Three seeds, one cohort, one corpus, one recipe. The comparison is at a
pre-registered epoch and at each size's best epoch; both give the same answer,
but "best epoch" is read off the eval cohort and is an oracle over four values,
so only the epoch-3 row is a deployable comparison. Learning rates are #101's
probe choices for each size and were not re-probed for the staged rows, so a
larger arm could in principle be under-tuned rather than saturated. Cells ran in three parallel
lanes by seed on an 8-core host, so every absolute second here is inflated by
about 3× oversubscription and no absolute timing figure is quoted. The lanes
moved through the sizes in lockstep, so the contention is roughly uniform
across sizes; that is what makes the *relative* per-expansion cost above
comparable while absolute throughput is not. One 1m cell (seed 1, epoch 5) has
a 133.9 s slowest task against a 6–10 s norm elsewhere, which is a scheduling
artefact rather than search behaviour. Verified counts are behaviours
of the cohort's 43, not of v4's 220.
