# Score-band retention: synthetic semantics gate A

The program-synthesis janthu (e-den, Codex) wrote this, 2026-10-03 PDT.
Design only: no implementation, synthetic gate execution, corpus selection,
checkpoint loading, inference, timing or coverage measurement has occurred.

## Question and choice

Can a layer-synchronous flat beam add an absolute cumulative-log-score band
with an explicit retention/seen-state contract and an unchanged disabled path?
Choose score-band retention; defer score-prioritized expansion. The existing
solver tests every executable successor for shown fit before scoring and keeps
one seen set across depths. Reordering expansions adds a separate first-fit,
shortcut/library and batching change. This arm changes only retention; future
layers and their first fits may still change. Passing this engineering gate
would establish that a later train-only allocation test has a specified object,
not that score confidence identifies useful branches or improves coverage/clock.

Motivation: short-segment-campaign (#168)'s longer-horizon widening sometimes
moved verified behaviours at much larger recorded spend. Its different horizon,
retrospective per-task budgets, restart rules and host contention do not isolate
width allocation. See the committed historical report
`reports/short-segment-20260924/report.md`; no raw eval/hidden input was read.
Capped-policy-glue-loss (#178) settled a different descriptive loss question;
its failed breadth gate is not evidence that this retention rule works.

## Closest literature and the gap

[Freitag and Al-Onaizan (2017), section 4 equation 2](https://aclanthology.org/W17-3207.pdf)
apply an absolute score cutoff after reducing to the ordinary bounded beam;
section 5 selects pruning thresholds on a separate selection set. Their
translation-quality observation does not assert identical outputs. This design
uses a gap in cumulative legal-normalized log scores and keeps ties at the
boundary; no threshold or speedup is transferred. The scoped search recovered
no public code for their in-house implementation; source reproduction is absent.

[Meister, Vieira and Cotterell (2020)](https://aclanthology.org/2020.tacl-1.51.pdf)
prove best-first beam equivalence under their monotonic scoring and completed
hypothesis rules. Our fit-before-ranking and cross-depth state merging differ.
Author code `rycolab/bfbs` at `0f2a4de58407499692000a44e55a59d56e3a3eae`
was source-read in the preceding audit, not executed. This design makes no
best-first equivalence, shortest-program or score-optimality claim.

## Frozen implementation scope

Base/reference: main `f9ae90bec5edab1537b35f6595d5dbfb7174c867`,
`wandering_light/stack_beam.py::_beam_solve`, its existing public wrapper and
low-level typed machine/guard/encoder primitives. Implementation, if approved,
is an **experiment-local** flat search in `experiments/score_band_search.py`.
Do not alter production `stack_beam.py`, `solve.py`, `BEST`, widening or segment
search, PolicyScorer, interpreter semantics or the debugger. A later solver
integration or cohort campaign needs its own preregistration/review; this gate
cannot authorize promotion. A bounded copy is deliberate to isolate the new
arm; disabled equivalence is independently tested against the frozen ordinary
beam, not against a second copy of the candidate. Preserve the acyclic-GC
wrapper behaviour and restore collection on normal and exceptional exits.

Inputs are manually constructed typed stack tasks and finite toy state graphs;
scripted scorers only. No v4/world/capped files, archive rows, saved checkpoint,
Hub/W&B data, held-out grading, training or model creation is permitted. The
word "seed" below means a fixed synthetic scorer permutation, not a trained
checkpoint or research replicate. Existing production tests remain validation.

## Retention definition

Parameters: the ordinary positive integer width W, horizon H and existing
caps/guard/resource settings; optional `score_band_nats=None` or a finite real
tau >= 0 (bool, negative, NaN and either infinity invalid). No finite real-task
threshold, horizon, checkpoint or panel is selected here. Synthetic arithmetic
uses tau in {0, 1, 2} and None; values illustrate boundary/tie cases only.

Each layer executes the ordinary beam in parent order and OPS order, with the
same per-attempt budget/deadline checks, counters, early refusal scope, legal
mask, pre-ranking shown-fit return and one batched score call. Score is parent
cumulative log probability plus its current legal-normalized op log probability;
no length normalization, temperature fitting, entropy or local-only score.

For a layer that did not return a fit, stable-sort executable successors in
nonincreasing score, preserving enumeration order on ties. Eligibility is the
ordinary close/type guard on the first shown stack, the total-product atom cap,
and absence from the global seen set plus this layer's earlier accepted states.
Let C be the first at most W distinct eligible candidates obtained with the
**pre-layer** seen set. If empty, ordinary frontier exhaustion applies. Let a
be C[0]'s score. With finite tau, retain exactly the prefix of C whose score
satisfies score >= a - tau. Equality is included; tau=0 retains all best-score
ties up to W. An inadmissible/seen high-scoring candidate cannot anchor the band.

Only final retained states are inserted into the persistent seen set. Eligible
states dropped by the band or width are not remembered as visited. Therefore
future layers can revisit a state pruned earlier, and banded search is not a
subset of the disabled run. No guarantee of monotone work, coverage or first
fit across tau/width is made. Structural product equality is the existing typed
Value equality, never printed text, encoded tokens or a program-prefix key.

C is a mathematical definition, not a requirement to scan the entire ranked
layer: a single stable scan may stop at W retained states, or at the first
otherwise eligible below-band candidate after a valid anchor. It never adds a
band-rejected state to seen. All predecessors were already executed/scored;
any savings start with a later layer, not with the cutoff layer's attempts.

None must follow the ordinary path: result fields except wall clock, ordered
frontiers, retained parents, masks, normalized scores, counters, termination,
and collector restoration match the reference. Finite mode rejects NaN/+inf
successor cumulative scores before retention/seen mutation; -inf is permitted
only as zero-probability support. A finite anchor excludes -inf; if every
otherwise eligible score is -inf, the band is uninformative and retains the
ordinary C unchanged, avoiding -inf-minus-minus-inf arithmetic. This convention
must be visible in telemetry. The disabled path keeps reference scorer behaviour.

Use experiment-local diagnostic recording; do not reuse a recorder that labels
all unread successors as width cuts. Record ordered retained identities and
parents, anchor, tau, per-layer stop cause (`width`, `score_band`, `exhausted`),
the first band-rejected eligible candidate, all-inf fallback, and `pruned`.
`pruned` stays conservative for width exactly as reference and is true on an
eligible band exclusion; no-pruning is not a probability-mass certificate.
Untimed diagnostic recording is separate from any future primary clock run.

## Gate A prediction, falsifier and fixtures (registered before execution)

Prediction: the experiment-local arm can satisfy every case below and exactly
reproduce disabled reference traces on all registered cases, while demonstrating
an actual next-layer expansion reduction on a constructed safe-pruning case.
Falsifier: any mismatch, wrong fit order/anchor/boundary/seen state, unpriced
attempt, incomplete fixture manifest, or no constructed reduction. A passing
fixture is software feasibility, never evidence for real coverage or clock.

1. **Disabled equivalence.** Fixed synthetic scorer permutations s0/s1/s2,
W in {1, 4, 16}, H in {1, 4, 8}, guards off/close/both on manually typed tasks.
Compare reference and candidate result/counter/order/mask/score traces, excluding
seconds; include ordinary refusal, no legal successor and no remaining eligible
successor. Width/horizon choices are fixture coverage, not a performance ladder.
2. **Fit before score/guards.** Two one-step shown fits (Int(2),Int(2))->Int(4)
using add/mul: return the first op in actual OPS order even if the scripted scorer
prefers the other, and do not call the scorer. Also preserve a shown fit whose
output exceeds the candidate state cap, and zero budget/past deadline before fit.
3. **Anchor after filtering.** Highest score is separately close/type-invalid,
size-invalid or already seen. A lower eligible state anchors; no poisoned empty
frontier. Same-layer duplicates use the first ranked occurrence only.
4. **Ties and arithmetic.** Exact boundary retained, next below boundary dropped,
tau=0 top ties retain enumeration order up to W; negative parents use cumulative
rather than local ordering. None/effectively inactive finite band agree when no
candidate is excluded. All -inf fallback and NaN/+inf/parameter failure are explicit.
5. **Seen-state consequences.** A band-excluded state reached later is still
eligible. A previously retained state is rejected across depths. Include two
prefixes producing equal typed states and distinct typed states whose capped
encoder blocks agree, establishing state identity independent of encoding.
6. **Useful and adverse constructions.** Fixed toy graph A prunes dead branches,
retains the exact reference first fit, and strictly lowers total expanded nodes
by the next layer (all spent attempts retained in counters). Fixed graph B has
its only successful continuation outside the band: reference fits, finite arm
fails. Fixed graph C diverges in first-fit program via later seen-state/path
changes. These cases demonstrate both possible savings and explicit risk;
none may be dropped because it is adverse.
7. **Boundaries and observation.** Budget0, overshoot at one multi-example attempt,
already-past deadline, H exhaustion, no successors, guard/atom refusals, observer
on/off equality of result/counters, exceptions restoring GC. The finite recorder
must identify a band cutoff rather than silently call it a width cutoff.

An independent expected-retention function uses plain fixture tables and typed
identity keys, without importing the candidate's retention code. Complete fixed
fixture table and expected outcomes must be committed before the first gate
execution. The table may specify logical outcomes/order rather than precomputed
policy-model values; use exactly representable toy scores for boundary tests.
Report every fixture identity and outcome, with source hashes and expected versus
actual counts/order. All-or-none gate; no subset result or tuning after a failure.
Bug fixes to the synthetic implementation remain development, with a new explicit
version/result and review before claiming gate PASS; they authorize no real-data
retry. No real task/model invocation is in Gate A.

Run CPU/thread1, <=300 seconds whole synthetic gate, <=1 GB outputs, one foreground
command once code/fixture table/tests pass independent high-effort pre-gate review.
Timeout or incomplete output means NO_GATE_RESULT, not a partial pass. Persist an
atomic complete JSON manifest and no timing comparison. Ordinary unit tests can
be rerun during development; each declared complete gate invocation is logged.
After a complete PASS, independent high-effort code/result review is required
before landing this new arm/evidence. Design-only review now authorizes only
implementation on synthetic fixtures, not Gate A execution or a cohort trial.

## Later work, explicitly unregistered

Only after Gate A passes and is reviewed: select a train-only disjoint design /
validation panel by original skeleton, choose/freeze one threshold and resource
policy without eval/hidden data, then preregister three paired trained seeds,
all attempted tasks and verified coverage plus total CPU wall clock. Hold the
checkpoint/horizon/caps/guard/library/order/host/threads fixed across the paired
arms, and price failure, scoring, restart and library overhead. Shared-library
effects require full ordered runs, not successful-task-only rescoring. Register
coverage gain at no extra clock separately from equal coverage at less clock.
No current Gate A result licenses these predictions, real-task calls or BEST.
