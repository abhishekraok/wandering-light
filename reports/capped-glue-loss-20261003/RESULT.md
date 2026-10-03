# Capped-policy-glue-loss (#178): completed bounded diagnostic

The registered TOP_LOSS_ROUTE prediction is falsified on its breadth criterion,
despite positive mean matched top-minus-body NLL in every seed. Both encodings
have a median26 of45 contributing behaviours with positive delta (57.78%),
below the frozen60% requirement. Median mean deltas are +0.640927 nats
(machine) and +0.477030 (difference), above the0.25 requirement.
Do not read the terminal NO_MATCHED_MIDSPAN_TOP_EXCESS as zero average excess.

This is a descriptive train-distribution exact-op mid-span association. It
does not test omitted boundary glue or establish causal runtime/frame effects,
generalization, search clock or solver coverage. No tuning, rerun, ablation or
BEST promotion follows this failed registered gate.

## Paired primary measurements

All occurrences per side/cell, cells equally within behaviour, behaviours
equally within checkpoint; seeds are replicates of one fixed panel.

| encoding | seed | top NLL | body NLL | top minus body | positive behaviours |
|---|---:|---:|---:|---:|---:|
| machine | 0 | 1.526734 | 0.736558 | +0.790177 | 26/45 |
| machine | 1 | 1.322938 | 0.682011 | +0.640927 | 29/45 |
| machine | 2 | 1.192635 | 0.772006 | +0.420630 | 23/45 |
| difference | 0 | 1.352757 | 0.758248 | +0.594509 | 27/45 |
| difference | 1 | 1.231286 | 0.754256 | +0.477030 | 26/45 |
| difference | 2 | 1.164750 | 0.849252 | +0.315499 | 19/45 |

Both encodings pass the median-delta and two-positive-seeds criteria, and both
miss median positive-behaviour fraction. Numerical gates are unchanged.

## Support, execution and independent verification

64 original-skeleton groups selected without refill from134 eligible groups.
45 groups/behaviours contribute108 exact-op mid-span cells; support passes.
4112 sealed steps include137 matched top and278 matched body occurrences.
Six complete checkpoints supply24672 raw rows. No duplicated selected capped
texts; alternative first-refusal counts are19132 ordinary,0 fuel and0 depth.

The single trial ran from independently approved frozen62f4076, started
2026-10-03 13:42 PDT and ended13:43 PDT, log exit0/complete=true.
RuntimeMaxSec1800/Memory4G, cooperative deadline900, CPU/thread1/batch32.
This is not comparative clock evidence. Input/train/side/checkpoint pins and
the pre-trial source/report-hash receipt remain unchanged.

Independent verifier source5211680 ended exit0, with no model deserialization:
separate original-skeleton/hash selection, every shown replay/target/Mark/owner/span,
cell membership/support, every alternative mask/reason scope, typed encoder
exposure, all raw metrics and full summaries, exact row/checkpoint ordering and
both-encoding reading were rederived. Thirty-four synthetic verifier tests and
full lint/format passed before real verification. High-effort independent
other-family result/code review is still pending; this is not landing approval.

[Complete seed/cell/behaviour measurements and artifact hashes](measurement.json).
[Independent verification receipt](independent-verification.json).
[Preregistration and Amendment A](preregistration.md), [pre-trial review](pretrial-review.md).
Raw per-step logits and full secondary summaries remain under main
checkpoints/capped-glue-loss-20261003/trial; paths/hashes are in measurement.json.
Before drawing further descriptive conclusions, inspect the required secondary
diagnostics; none changes the registered gate.
