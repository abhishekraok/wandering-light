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
full lint/format passed before real verification. Independent high-effort Claude Opus result/code review **PASS** on frozen
`bcd20fe456271a16cf0756654e13d3294da2b6a5` is retained in [result-review.md](result-review.md).
The reviewer independently recomputed the entire selection/replay/mask/exposure
and raw-logit result, reproduced 153 relevant tests and obtained byte-identical
no-model reverification. It found no blocker; its scoped nonblocking limitations
remain recorded. The complete code, evidence and reporting are landed on main.

Final development validation including both verifier suites completed exit 0
at 2026-10-03 14:15 PDT: 2892 passed, 40 skipped, 7 deselected, 1 xfailed and
8 existing warnings. Whole-tree lint/format and publication exact-copy/hash,
denominator and typed-width checks passed. This is validation, not a new trial.

[Complete seed/cell/behaviour measurements and artifact hashes](measurement.json).
[Independent verification receipt](independent-verification.json).
[Preregistration and Amendment A](preregistration.md), [pre-trial review](pretrial-review.md).
Raw per-step logits remain under main checkpoints/capped-glue-loss-20261003/trial;
paths/hashes are in measurement.json. Required secondary summaries are published below.

## Required secondary reporting

These are descriptive summaries of the same complete sealed panel. No selection,
inference, threshold, gate or trial has changed. [SECONDARY.json](SECONDARY.json)
retains the exact independently checked all/common joint label summaries
(side, content/structural, span position, exact op and depth), every behaviour's
token/loss/deficit shares, and the assembly-excluded cell/behaviour results.
The publication script is retained beside this report as `publish_secondary.py`.

### Common support and omitted exposure

| side | all content | common support | omitted content | structural |
|---|---:|---:|---:|---:|
| top | 491 | 137 | 354 | 1167 |
| body | 1207 | 278 | 929 | 1247 |

All common-support content is middle of a combinator-delimited span.
The omitted content includes unmatched middle as well as boundary tokens;
structural tokens are excluded from the primary. The full joint breakdown is
in SECONDARY.json. Top-content flags refer to any later top-owned opener or an
earlier depth-one closer followed by a later run in the same span; they can overlap:
358 occurrences before an opener and
46 between a closer and combinator.

### Legal support and loss decomposition

The following means weight common-support **occurrences**, unlike the primary
equal-cell/then-equal-behaviour means. Each row satisfies unmasked NLL = legal
NLL + negative log legal probability mass. Rank counts strictly greater legal
logits; ties share rank. Deficit is best legal logit minus witness logit.

| encoding | seed | side | rows | unmasked NLL | legal NLL | -log legal mass | deficit | rank | legal size |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| machine | 0 | top | 137 | 1.568812 | 1.561642 | 0.007171 | 1.221952 | 2.408759 | 15.321168 |
| machine | 0 | body | 278 | 0.855445 | 0.855445 | 0.000000 | 0.496519 | 1.654676 | 28.000000 |
| machine | 1 | top | 137 | 1.299340 | 1.292584 | 0.006756 | 0.942602 | 2.116788 | 15.321168 |
| machine | 1 | body | 278 | 0.654108 | 0.654108 | 0.000000 | 0.290224 | 1.420863 | 28.000000 |
| machine | 2 | top | 137 | 1.179603 | 1.167731 | 0.011872 | 0.761169 | 2.021898 | 15.321168 |
| machine | 2 | body | 278 | 0.901350 | 0.901350 | 0.000000 | 0.518682 | 1.762590 | 28.000000 |
| difference | 0 | top | 137 | 1.433511 | 1.427149 | 0.006362 | 1.105418 | 2.299270 | 15.321168 |
| difference | 0 | body | 278 | 0.900584 | 0.900584 | 0.000000 | 0.525807 | 1.636691 | 28.000000 |
| difference | 1 | top | 137 | 1.282140 | 1.270753 | 0.011387 | 0.878685 | 2.065693 | 15.321168 |
| difference | 1 | body | 278 | 0.814832 | 0.814832 | 0.000000 | 0.440918 | 1.593525 | 28.000000 |
| difference | 2 | top | 137 | 1.207536 | 1.192348 | 0.015188 | 0.770150 | 2.080292 | 15.321168 |
| difference | 2 | body | 278 | 0.966302 | 0.966302 | 0.000000 | 0.570492 | 1.751799 | 28.000000 |

Legal sets depend on the shown execution state and the bounded interpreter.
Under Mark, emitting the same op name has different semantics from executing it.
These secondaries do not identify a causal quotation/frame effect.

### Token, loss and deficit shares

All 64 selected behaviours contribute. This table is the arithmetic mean of each
behaviour's top share across **all tokens**, including structural tokens, without
pooling seeds or encodings. The complete per-behaviour shares remain in SECONDARY.json.
An undefined zero-total metric share is retained as null and excluded from that
metric's mean; token share is always defined.

| encoding | seed | top token share | top unmasked-loss share | top legal-deficit share | defined deficit behaviours |
|---|---:|---:|---:|---:|---:|
| machine | 0 | 0.427206 | 0.453287 | 0.483679 | 62 |
| machine | 1 | 0.427206 | 0.472631 | 0.487593 | 64 |
| machine | 2 | 0.427206 | 0.490276 | 0.492447 | 63 |
| difference | 0 | 0.427206 | 0.439058 | 0.471943 | 63 |
| difference | 1 | 0.427206 | 0.459817 | 0.452781 | 64 |
| difference | 2 | 0.427206 | 0.453966 | 0.455657 | 64 |

### Typed encoder width and truncation

Widths count typed `encode_stack` tokens **before** caps; these are shown-example
states at witness prefixes, not independent tasks. A row is flagged if any shown
example truncates. State cap is 160 and target cap is 64 in both encodings.

| population | rows | example states | state width min/median/max | target width min/median/max | state truncation rows/examples | target truncation rows/examples |
|---|---:|---:|---|---|---|---|
| all | 4112 | 19504 | 1/21.0/208 | 1/1.0/14 | 73/87 | 0/0 |
| common_support | 415 | 1921 | 1/19/193 | 1/1/14 | 5/5 | 0/0 |

Body-depth counts are {1: 1893, 2: 561} across all rows and
{1: 211, 2: 67} in common support. Full width histograms
are retained in SECONDARY.json. Training task-pool membership does not establish
that these rows appeared in the with-replacement training sample; exact multiplicity
was not reconstructed and remains unavailable.

### Assembly exclusion and resource scope

`quote`, `cons` and `cat` cells are flagged assembly cells. Removing them changes
the descriptive population below; it supplies no new gate or inference.

| encoding | seed | primary delta | delta without assembly | remaining cells | remaining behaviours |
|---|---:|---:|---:|---:|---:|
| machine | 0 | +0.790177 | +0.903063 | 77 | 44 |
| machine | 1 | +0.640927 | +0.709147 | 77 | 44 |
| machine | 2 | +0.420630 | +0.534913 | 77 | 44 |
| difference | 0 | +0.594509 | +0.732710 | 77 | 44 |
| difference | 1 | +0.477030 | +0.535369 | 77 | 44 |
| difference | 2 | +0.315499 | +0.400010 | 77 | 44 |

There were no duplicate selected capped-program texts. Alternative legality checks
had 19132 ordinary first refusals,
0 fuel and 0 depth refusals;
no scored row is resource-affected. Refusal reasons record only the first refusing
shown example/op prefix. Ordinary refusal means illegal under this bounded mask;
resource refusal, if present, would not establish semantic impossibility.
Witness replay succeeded on every shown example. Per-example/op fuel was 100,000,
initial depth 0 and maximum quotation depth 32. This diagnostic supplies neither
comparative solve timing nor a claim about omitted boundaries or held-out tasks.
