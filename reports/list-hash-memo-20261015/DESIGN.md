Written by the program-synthesis janthu (e-den).

## Prediction and falsifier, before implementation or data

Can lazy memoization of an immutable `Lst` object's existing structural hash reduce actual solver process clock while preserving its exact search result under a common upfront execution budget?

Prediction: for every one of three paired deterministic repeats, candidate total child-process wall clock over the fixed24 TRAIN tasks is <=9/10 of control, with exact per-task program/termination/depth/executions/fuel/expansions/pruned equality. Check ratios using Fraction of recorded binary floats. Otherwise park this fixed lazy-Lst-hash construction without panel, cap, key, threshold or implementation tuning on these outcomes. Incomplete/killed execution is no scientific result and permits no favourable retry. This is a timing/compatibility pilot, not verified coverage or BEST promotion.

## Source and closest evidence

At production main `ce7662de0b632ce2e97f4f2d1bfb37cf08d8bff1`, `_beam_solve` uses structural products for seen membership and insertion. `Lst` is a frozen slotted dataclass with sole field `items`, so its generated hash recomputes `hash((self.items,))`. Retained objects can be hashed again, including shared child values. Existing `run_body` memoization does not cache these value hashes. Cost/frequency/savings are unmeasured.

The earlier optimization commit `438d4b4` removed repeated value_key construction and unobserved refusal strings; it did not memoize value hashes. A bounded source/history search found no such current implementation, not proof of novelty. CrossBeam's [paper](https://arxiv.org/pdf/2203.10452) and author [value](https://github.com/google-research/crossbeam/blob/master/crossbeam/dsl/value.py)/[search](https://github.com/google-research/crossbeam/blob/master/crossbeam/algorithm/synthesis.py) code retain semantic intermediates and cache a value representation used for hashing. Its wall-clock appendix also cautions against transferring candidate-count gains to runtime. No reproduction or speed transfer is claimed.

This is not an alternative width/key/capacity retry of policy-item-reuse-screen (#184), whose negative construction is parked. It caches no embedding, score or interpreter outcome, and leaves the exact semantic key/hash integer unchanged.

## Fixed candidate and software gate

Only `Lst` gains a lazy per-object private hash slot, inherited from a private non-dataclass base so it stays outside dataclass fields. On a miss return/store EXACTLY the old `hash((self.items,))`; on a hit return that same integer. Other value types, equality, field schema, repr, constructor, value_key, parser/formatter, primitive behavior, scorer and search stay unchanged. No global table, eager pre-hashing or interpreter/model cache. Failed hash computation must not publish a memo. Scope is the existing closed, immutable typed Value union; no new acceptance of arbitrary mutable foreign fields.

[CPython 3.12.8 tuple](https://github.com/python/cpython/blob/v3.12.8/Objects/tupleobject.c#L294) and [dataclass](https://github.com/python/cpython/blob/v3.12.8/Lib/dataclasses.py#L621) sources motivate this boundary. Salted Sym hashes must never persist across processes. Require exact hash/equality/partition agreement against the original definition, fresh/equal-but-distinct/nested/marked/mixed typed values, repeated calls, collisions, invalid hash inputs, normal frozen behavior, unchanged fields/asdict/astuple/replace/repr/pattern interface, and pickle/copy/deepcopy across different PYTHONHASHSEEDs and old/new source directions. Lazy hash memo must be absent from serialized field state. Test actual imported production search and a small untrained scorer with manually typed tasks; compare every substantive result field/logits/features/legal mask. These are software checks, not real provenance.

Implement only in fresh own worktrees. Freeze two real sources: control = production ce7662d plus experiment-only runner/tests; candidate = that control plus the sole Lst change. Both sources need high-effort other-family code/design review BEFORE real task/model access. Pin exact heads/bytes/origins of both and verify the only production difference. Do not simulate control by patching hash in an already enlarged candidate class. No source under a running job may change.

## Fixed inputs, resources and order

Original stack_pbe_v4 TRAIN only, revision `00c2a9996d32df8d1894b45f4246a66b6df46c25`, payloadSHA `13450717aadfce2ec16e890f3e9509514cf4c82bdff3d1199006854fe80c683b`. First source row per nonempty skeleton; rank by SHA256 of UTF8 `list-hash-memo-v1` + one NUL byte + skeleton, tie by skeleton text; first24. No refill/filter/outcome selection. Children receive only source_index and original shown pairs, no witness/behaviour/hidden/reference fields. Save exact identities/projection/pins; no eval/world payload.

Same fixed staged-s0-e3 checkpoint `e5a0b1391c5268c3ae04a49e6862f19a30779bb2`. Fresh CPU float32 processes, numerical/interopthreads1, PYTHONHASHSEED0, flat production beam64/depth48/guardboth/atom16384/libraryOFF, fixed upfront100000 successful-example executions per task, NO task deadline. Global enforced job limit is not a solver deadline. Clear run_body memo at each process start; ordinary object lifetimes/shared constants and existing body-cache lifetime otherwise unchanged. Repeats17/29/43 are labels, not trained seeds. Fixed cell order: control17,candidate17,candidate29,control29,control43,candidate43. One host, no concurrent own compute/timing/review jobs; record external point activity.

Primary clocks are parent-measured subprocess launch-to-return, including imports/load/search/output and all24 attempted tasks, not a favourable subset or just model-forward time. Preserve per-task solve clocks, executions/fuel/expansions/terminals, per-process/whole-phase clocks and peak resource data descriptively. No warmup/timed reruns. All six timing cells complete BEFORE shown-only typed integrity replay and paired analysis; no hidden/reference grading. Every paired substantive result must match exactly, including failures. No retrospective allocation or omission of slow tasks.

One expected10min CPU attempt in a persistent unit, hard RuntimeMaxSec1200/MemoryMax12G/MemorySwapMax0. Main checkpoints outputs/cache/logs, verified log parent and committed frozen sources before launch. Record exact launcher, actual cgroup/threads/module origins, host/source/model/input/raw before/after pins, wrapperexit0 and completeallcells; no end-of-limit partial result. Post-run generated-program checks/review also enforce12GiB/no swap. Reconstruct selection, metrics/equality/gates and shown replay independently; no giant trace/codec/verifier framework or whole-search-proof claim.

## Interpretation

The conjunction is exact result compatibility AND >=10% process-clock reduction in EVERY repeat. Failure parks this construction without tuning/retry. PASS earns only a separately registered release-cohort verified-coverage/all-attempt-clock qualification under existing repository rules; it does not authorize landing the production hash change, claiming held-out speed, changing BEST or relaxing owner gates. Test-only/review repairs before real execution are not data eligibility. No trained-model authoring, training, paid API or new dataset generation.

Status: the one registered pilot completed normally; exact compatibility passed and every fixed clock gate failed. Construction parked; final result review passed. The contract above is unchanged.
