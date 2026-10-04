# Train-only score-gap/rank/depth feasibility — D1

The program-synthesis janthu (e-den, Codex) wrote this, 2026-10-03 PDT.
score-gap-feasibility (#180), issue-first design before task selection, search, model loading or threshold
selection. Design and pre-run exact-code review must pass before data work.
Synthetic implementation/tests may proceed during design review. This is one
bounded reference-trace diagnostic, not a real band search or clock comparison.

## Question and falsifier

Can one preregistered absolute band retain known positive reference-search
lineages better than a width32 prefix, with no more retained frontier slots on
those SAME fixed width64 snapshots? Rank/depth/gap profiles only motivate a later
allocator test. Positive continuation labels are shown-fit upper bounds; no
other row becomes a negative or a shortest-distance label. A failed construction
kills this route for this panel/limits, not all adaptive search.

D1 predicts a common band from the frozen grid will retain at least95% of the
informative successful reference lineages, improve on the width32 snapshot
prefix by at least10 percentage points, and use no more mean normalized frontier
slots, in EACH of three trained seeds on a skeleton-separated confirmation half.
Failure of any conjunction component falsifies D1. Insufficient support,
incomplete/corrupt provenance or a killed attempt gives a separate non-pass.
No retuning, panel refill, enlarged cap or favourable rerun follows.

## Frozen inputs and selection

inputs.json reuses committed byte pins from capped-policy-glue-loss (#178): v4
TRAIN source, capped side-program file and difference capped-s0/s1/s2-e3 policies.
Only the difference family is used, because the earlier source audit identifies
it as the intended capped candidate; this is not selection from current data.
All three already trained on this population: confirmation tests profile
transfer across original skeletons inside familiar training data, not unseen
behaviour generalization. No BEST/staged arm, checkpoint tuning or retraining.

Eligible original TRAIN tasks have a behaviour present in the capped side file
and are outside validation_skeleton computed from their ORIGINAL skeleton,
fallback behaviour. Group by that same original key. Within group choose one
behaviour by SHA256('score-gap-D1-behaviour:' + name), lexical tie-break; use the
first original-file task for it, all shown examples and original targets. No
capped-program substitution into the search Task. The side file only determines
eligibility; its witness is not a label. Select first48 groups ordered by
SHA256('score-gap-D1-group:' + group_key), lexical tie-break. Alternating positions
in that order form24 selection and24 confirmation groups. Fewer than48 yields
INSUFFICIENT_SELECTION; do not refill/change hashes. No outcome, program length,
score, eval archive or hidden-example access informs selection. Whitelist only
identity/skeleton/program/shown input/output fields from the train payload;
hidden fields are discarded without inspection or grading.

Reserve a unique attempt BEFORE reading payloads. Check full source bytes and
seal task identities, group/half/order, shown typed inputs/outputs and pin
checks BEFORE models. Execute groups in fixed selected order, seeds0/1/2 per
group; retain half labels but compute no band metrics until every cell is sealed.
The reference never sees witness metadata, split labels or hidden examples.

## Reference trace and upfront resources

Imported production beam_solve only, ordinary width64/depth48/guardboth,
max_state_atoms16384, no library/restarts, CPU, uncalibrated PolicyScorer in
its own difference encoding. Torch/OMP/MKL/OpenBLAS one thread; record host and
versions. Model loading excluded from solver seconds but INCLUDED in the global
diagnostic deadline. No quiet-clock/speed estimate. All three input/config/
weight/manifest hashes and actual module paths/bytes checked before/after. Load the three models
once, then reuse their independent scorer instances in the fixed cell order.

Each task-seed solve has200000 successful per-example executions and10seconds
from immediately before beam_solve. These caps are fixed upfront, never matched
to observed spend; ordinary budget/wall-clock/max-depth/exhausted returns are
completed bounded reference outcomes, not successful lineages. Default machine
fuel/depth per op remains unchanged. Interpreter refusal is not a negative
continuation. Record every stop/counter, partial last layer, shown fit and calls.

Global deadline2700seconds includes hashing/selection/replay/model loading/search/
recording/analysis/serialization. At most2GiB serialized trace payload and4GiB
process memory; external RuntimeMaxSec5400. Limit1,000,000 attempted-op observer
records over the run: reaching it stops the diagnostic as resource-incomplete,
without synthesizing a solver terminal or silently sampling rows. Production
expansions charge on parent entry even without op0 callback; retain actual
BeamResult counters and complete callbacks, never derive expansions from
attempts alone. Cap/completeness/provenance checks precede primary computation.
Single launch only; immutable attempt/final/failure artifacts under main
checkpoints/score-gap-feasibility-20261003. A hard kill/no exit0 is no result.

## Known positive lineage and reference snapshots

Use complete observer layer_started/scored/retained data and final program.
Independently replay all returned programs on all shown examples; any failed
returned-program replay is invalid input/evidence, not an excluded success.
Reconstruct each retained ancestor of the exact returned program using recorded
parents plus actual typed product states. Confirm its suffix reaches all shown
targets. Do not infer ancestors merely from visually printed stacks, authored
witness prefixes or latest state occurrence. Every ancestor must agree with
actual retained beam order, program and key. No genuine-closure requirement:
open quotation states are legitimate beam nodes and the saved suffix is replayed
with their complete state. It is a known continuation, not a value ordering.

At a completed scored layer, C is the ACTUAL ordered ordinary retained beam,
after guard/atom/seen filtering, at most64. Record every C state's cumulative
score, stable rank, depth, legal mask, typed key and ancestor membership. We do
not classify the unread ranked tail, change normalization, call the #179 band
candidate, or label alternatives. First-fit terminal layers have no policy
scores and stay explicitly unscored. Reference snapshots are immutable; pure
profile evaluation does not rerun search for each threshold.

A successful cell is informative only if its exact returned program has at
least two scored retained ancestor layers and at least one such layer has more
than32 states in C. One-op/short/tiny-beam fits are separate uninformative fits,
not counted as zero-loss evidence. Other complete bounded outcomes have no
lineage label. Keep all fixed selected groups in the denominator reports.
Require at least12 informative groups per half in EACH seed. The same groups
are selected upfront; do not intersect/filter to an outcome-favourable shared
support or pool seeds as tasks. Support failure is INSUFFICIENT_LINEAGE_SUPPORT.

## Band profiles, rank control and weighting

Frozen grid in natural-log score units:0.25,0.5,1,2,4,8,16,32. No scale fitting,
relative band, per-depth schedule, parent cap, temperature or entropy arm.
For C with finite best scorea, retain the initial prefix with score>=a-tau,
the exact inclusive #179 contract. All-minus-infinity C falls back to all C.
NaN/+infinity at any executable non-fit score invalidates the cell; illegal
matrix entries are ignored, as in #179. Explicit minus-infinity is retained in
profiles and yields an infinite finite-anchor gap, never an arbitrary cap.
The fixed rank control retains the first min(32,|C|) of the SAME C. Neither is
an actual re-executed narrower beam or a deployable retrospective budget rule.

Lineage protection for an informative cell is1 iff its labelled ancestor is
retained at EVERY scored ancestor layer, otherwise0. Average equally across
its informative selected groups within half/seed. The fixed-prefix protection
uses the identical ancestor population. Report each task/depth/rank/gap, plus
all uninformative fits/nonfit stop reasons and their full denominators.

For every completed cell, slot fraction is sum of profile retained sizes over
ALL its scored C divided by sum|C|; cells with no scored layers are assigned1
for BOTH profiles and explicitly counted. These are prospective frontier slots,
INCLUDING the last unexpanded scored layer, not expansions/executions or saved
clock. Average cell fractions equally over ALL24 selected groups within half/
seed. Every group contributes once; long traces do not dominate this mean.
Also retain raw slot totals, per-depth and occurrence-weighted descriptive
means with denominators. Positive-lineage protection is conditional; slot means
include failures. Neither is verified behavioural coverage.

On the SELECTION half only, a grid band is feasible iff in EACH seed protection
>=0.95, protection-minus-fixedprefix>=0.10 and slotmean<=fixedprefix slotmean.
Select the SMALLEST feasible tau, or terminal NO_SELECTION_PROFILE; retain all
eight profiles, including failures. Lock chosen tau/selection receipt BEFORE
opening confirmation-profile summaries. Then evaluate exactly that tau on the
confirmation half against the identical conjunction in each seed; no second
choice. PASS_PROFILE_FEASIBILITY iff all pass and all prerequisites complete.
Selected tau is a train diagnostic proposal, not a deployed solver threshold.
Confirmation failure is NO_CONFIRMATION_PROFILE, with exact failed components.
This is deterministic descriptive selection, not a p-value/inferential claim.

## Why a pass is insufficient

Band cuts change future support and retained-only cross-depth seen. States
absent from the reference prefix may re-enter an executed band's search, change
anchors/first fits/library reuse. Even preserving every saved ancestor on these
snapshots does not guarantee recovering the reference program. Unscored first
fits and CPU batching also break a direct slot-to-clock inference. The reference
is library-off; this cannot forecast BEST's library-on prequential coverage.

After a reviewed pass, propose a NEW issue-first experiment: actual broad fixed
beam, actual narrower fixed-width beam and candidate with one upfront shared
resource policy/horizon/guards/state cap/library+order/host/threads, three paired
trained seeds, verified coverage AND all-attempt total CPU clock including
failures/scoring/restarts/library. Register equal coverage/lower clock and higher
coverage/no extra clock separately; no retrospective per-task spend oracle,
held-out tuning, settled width/depth ladder repeat or promotion outside#158.

## Validation and permission boundaries

Before the single data launch: independent high-effort design and exact-version
code review, synthetic tests of selection/projection/typed ancestor lineage,
first-fit-without-score, last-layer completeness, stopped partial traces, ties/
minus-infinity/fallback, equal-group/no-score weighting, exact split/seed gate
logic, exclusive attempt and strict atomic complete serialization. Verifier
must independently reconstruct typed inputs/attempts/lineage/scores/metrics from
low-level primitives and frozen definitions, not call profile implementation
again. Seal all reference traces before independent replay/analysis; verify
hashes and provenance before logits/summary access. After run, independently
verify complete outputs and obtain fresh exact-version high-effort result
review before landing. Record numbers in RESULTS_DETAILED, state in STATUS,
argument and all nonpasses on the issue. No model/corpus change or BEST move.
