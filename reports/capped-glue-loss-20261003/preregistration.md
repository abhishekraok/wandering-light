# Capped-policy loss: top-level versus quoted emissions

D1, 2026-10-03, before new model inference or loss analysis.
[capped-policy-glue-loss (#178)](https://github.com/abhishekraok/wandering-light/issues/178). Independent design and pre-trial code review must pass
before this bounded diagnostic runs. Source audit only has occurred.

## Question and what the source permits

Within a capped policy's own capped training style, are top-level content
emissions harder to predict than the same op names emitted inside quotations,
matching behaviour and position in a combinator-delimited span?

`short_body/transfer.py` pools `frame_gate.walk` into content/structural and
span-position marginals, discarding the traces required here. Its committed
per-op JSONs cannot answer the joint matched question. `frame_gate.step_classes`
uses virtual-frame attribution; it is not the lexical ownership used by
`segment_composite.levels`. We will not equate those labels.

Under a mark, an op name emits code; at top level it executes a transition.
Legal action sets and semantics differ. Matching names therefore estimates an
association in next-token prediction, not a causal nesting effect or execution
difficulty. `quote`/`cons`/`cat` can construct code that runs later: lexical
ownership is not ownership of runtime work. A positive result motivates only
design of a separate train-only ablation, not a loss/frame intervention here.

## Frozen population

`inputs.json` pins train bytes, capped side-program bytes and the six existing
epoch-3 capped checkpoints (machine/difference, seeds 0/1/2), including weight,
config and training-manifest hashes. Only the v4 TRAIN payload is opened.
No eval tasks, hidden examples, yardstick or BEST outcomes participate.
These checkpoints have trained on this population: this is an in-distribution
diagnostic, not generalization evidence.

Eligible behaviours occur in the exact capped side file and are outside
`validation_skeleton(task)` using their ORIGINAL fused skeleton (fallback to
behaviour as the existing helper does). Original skeletons define groups.
Within each group choose the behaviour with smallest SHA256 of UTF-8
`capped-glue-D1-behaviour:` plus its name, lexical tie break. Select up to 64
groups by smallest SHA256 of UTF-8 `capped-glue-D1-group:` plus group key,
lexical tie break. Use the first original-file task for its representative,
all shown examples and original target; substitute only its capped program.
Retain original task/skeleton identity. Fewer than 64 eligible groups means a
smaller panel, not refill. Untreated fused fallbacks and validation groups stay
outside the panel.

Seal selected identities/programs and all support counts before any logits.
Selection and support depend only on metadata, syntax and shown replay,
never on losses. No outcome-selected subset or cap retry.

## Lexical ownership and matching

Use `segment_composite.levels`' convention: `[` belongs to its parent level,
`]` to the body it closes, and another token to the innermost open quotation
or top level if none. Record opener and depth. Unmatched brackets are errors.
For every non-bracket content emission, cross-check lexical depth against
the replayed Mark stack on every shown example. No consumer/frame inference
is needed to decide this label.

Structural ops are fixed to `[`, `]`, `map`, `filter`, `fold`, `ifte`, `dip`,
`i`, as in `transfer.py`. All other mark-28 ops are content. Structural tokens
appear in descriptive totals, never primary matching.

Spans end after each top-level executed combinator, exactly `step_kinds`' run
label and `transfer.positions`. They are combinator-delimited spans, not
guaranteed genuinely closed states. The final pending tail remains a span.
Position is first/middle/last/only in the full span, including body tokens.

A common-support cell is (behaviour, exact op name, span-position label) with
at least one top and one body content occurrence within the same chosen task
and capped program. All occurrences contribute to their side's mean; do not
multiply Cartesian pairs, trim difficult occurrences or choose a favourable
depth. Report body depth separately without using it to change this panel.

Support gate: at least 20 independent original-skeleton groups contribute at
least 40 such cells. Gate failure is INSUFFICIENT_SUPPORT for this construction,
without scoring the models or changing selection.

## Primary, controls and falsifier

Primary is unmasked next-token NLL over all 28 ops, from stable logsumexp of
`PolicyScorer.logits` in each checkpoint's own encoding. It matches training
cross-entropy; legal support changes do not change its normalizer.

For cell c, delta_c = mean NLL(top) - mean NLL(body). Average cells uniformly
within behaviour, then contributing behaviours uniformly within checkpoint.
Report every cell/behaviour/seed, never pool encodings or treat seeds as
independent tasks. No long-witness exposure weighting in the primary.

Prediction TOP_LOSS_ROUTE: in EACH encoding, median over its three seed deltas
is at least +0.25 nats; at least two seed deltas are positive; and median over
seeds of the fraction of contributing behaviours with delta_b > 0 is at least
0.60. This is a practical diagnostic threshold, not a significance test or
solver-improvement criterion. If support passes and either encoding misses
any requirement, the prediction is falsified: NO_MATCHED_TOP_EXCESS. Retain
exact/null/reversed deltas and encoder disagreement, not only the label.
Any provenance, replay, ownership or logit error gives INVALID, not a negative.

Secondary diagnostics: best-legal-logit minus witness logit, legal-masked NLL
and rank; totals/counts by level x content/structural x span position and exact
op; per-behaviour top share of loss/deficit alongside top share of tokens.
Show common-support and all-row totals and each side's omitted exposure.
Record legal-set size and encoder truncation indicators where recoverable
without changing the scorer. No calibration, oracle scaling, predicted-clears
headline, metric substitution or added interpretation gate.

## Bounded execution and verification

Implementation/tests first, on synthetic rows and bracket/Mark fixtures, then
independent pre-trial code/design PASS before any checkpoint inference. Freeze
committed run source. Output is under main `checkpoints/capped-glue-loss-20261003/`
with exact revision, hashes, sealed panel, limits and complete per-step records.

CPU, one torch thread, fixed batches of 32, six checkpoints once in inputs.json
order. Use existing `hindsight_subtargets.replay_states` and
`stack_beam.step_all_quiet` semantics: DEFAULT_FUEL_LIMIT=100,000 per example/op, initial depth 0 and
MAX_QUOTATION_DEPTH=32 (refusal above 32). Pin these in run provenance. Check rewritten
witness ends at the original shown target. Refusal, illegal witness op,
incorrect Mark depth, nonfinite logits and missing/hash-mismatched input are
errors, never skipped rows or zero losses. Resource refusals are distinct.

Global deadline 900 seconds includes selection/support, replay and scoring.
If any task fails or the deadline hits, save partial INVALID output; no partial
task/checkpoint contributes to a reading. Persistent RuntimeMaxSec=1800 and
MemoryMax=4G include import/serialization overhead. Kill is no result, and only
exit 0 plus complete valid output supplies a reading. Record exact unit/command/
commit in work/runs.md; inspect others' jobs and disk before launch; no launch
below 100 GB free. This is not comparative search timing or cost to clear.

Independent result verification re-derives selection and lexical ownership from
pinned inputs and recomputes metrics from stored raw logits. Passing other-family
high-effort result review is required before landing. Numbers to RESULTS_DETAILED,
state to STATUS, result/argument to this issue. No BEST promotion, solver search,
training or held-out measure. A positive result permits designing a NEW train-only
loss/frame ablation; a subsequent solver improvement needs three paired seeds and
verified coverage/total wall clock. Negative/unsupported findings are recorded
without enlarging this panel until favourable.

## Related work and owner thinking

SketchAdapt learns a sketch/completion split and compares against pure generation
and pure synthesis. Its list tests train at length 3 and test lengths 3/4 using
enumeration budgets. Training uses completion-likelihood bounds instead of
timing each sketch. That supports testing a split, not presuming bodies are the
hard steps here. [Paper](https://proceedings.mlr.press/v97/nye19a/nye19a.pdf).

At [frozen master 687d6b6](https://github.com/mtensor/neural_sketch/tree/687d6b6c68ad534de32285398f54317b7f2edceb),
`make_holey_deepcoder` chooses holes using grammar scores; evaluation sends
generated sketches to a separate enumerator. The timeout branch thresholds
negative log weights, not measured wall time. README recommends working-mnye,
so this code read scopes master, not every experiment revision. No reproduction.

Abhishek's note `Learning efficiency alongside correctness` motivates separating
loss/deficit associations from elapsed solve cost. A later solve comparison must
charge inference and failed attempts too.
