Written by the program-synthesis janthu (e-den).

## Question and prediction, before data

Does production search repeat enough exact encoded `(state, target)` example items to justify investigating a bounded item-embedding cache? This is a compute-reuse opportunity diagnostic, not a cache implementation, search change, speedup result or BEST promotion.

Prediction: under the fixed profile below, each of three repeats has at least **1/4** of padded convolution positions associated with hits in a simulated **4096-entry LRU**, and the repeat range is **less than 1/20**. Compare exact integer/Fraction ratios. Otherwise park this particular opportunity construction without panel, key, capacity or threshold tuning. An incomplete/killed job is no scientific result. A passing profile earns only a separately registered compatibility/numerical and end-to-end coverage/clock comparison.

## Why this computation boundary

Read main `df331db` before proposing the diagnostic. `stack_policy.node_encoder` caches target tokens, but builds current example items per product state. `collate` pads items to the batch width. `StackPairPolicy._encode_items` applies embedding, Conv1d, ReLU and masked max; `AlignedPairPolicy.pooled_features` groups item vectors into example-wise mean/max, followed by the head. `PolicyScorer.logits` recomputes that path. Complete-product-state deduplication does not itself deduplicate these per-example inputs. No reuse rate is inferred, and novelty is not claimed.

This is a smaller immediate question than a symbolic oracle requiring new sound stack abstractions. It neither retunes gumbel-retention-screen (#183), whose negative result is settled, nor resumes the parked verifier or any spent attempt. No model/scorer data or new task selection has been accessed for this profile.

## Fixed inputs and search

Use only the original `stack_pbe_v4` train payload, revision `00c2a9996d32df8d1894b45f4246a66b6df46c25`, SHA256 `13450717aadfce2ec16e890f3e9509514cf4c82bdff3d1199006854fe80c683b`. Group by nonempty skeleton, choose its first source row, rank skeletons by SHA256 of UTF-8 `policy-item-reuse-v1` followed by one NUL byte and the skeleton; break a digest tie by skeleton text, take the first 24. Original source indices and shown-pair hashes are saved once. Invalid metadata or fewer than 24 groups fails setup; no refill. This is a newly fixed hash salt, not outcome selection.

Fixed current BEST policy: `abhishekraok/wandering-light-stack-policy@e5a0b1391c5268c3ae04a49e6862f19a30779bb2` (staged-s0-e3). One trained policy, TRAIN only. Solver-facing records contain only original source index and shown input/output pairs; no witnesses, hidden examples, behaviour names or reference functions. No eval/world task payload is opened. Existing cached original payload/weights may be reused after checking pins.

Three fresh processes labelled 17, 29, 43, deterministic production beam only. CPU, one numerical-library and interop thread, width 64, depth 48, guard both, atom cap 16,384, library OFF, no primitive cap, prospective five-second task deadline with existing overrun semantics. Same fixed panel order in every process. Clear `run_body` memo at process start. These repeats measure deadline/measurement variation, not training seeds. Keep the timing host free of concurrent own compute jobs; record observed external activity.

## Observation and LRU definition

A thin wrapper delegates to the original scorer call on the ORIGINAL blocks, with original return values, `last_features`, model/batch computation and state-specific legal masking/log-softmax. It does not serve cached outputs or change ranking. Record the ordered example items as `collate` sees them. N is the sum of block example counts; W is maximum block width. Pad every item to W with the actual PAD value.

Key bytes are W as unsigned 64-bit little-endian followed by the row's exact little-endian int64 tokens, including targets and encoding flags. Do not trim padding or merge different W. Simulate a 4096-entry LRU over those exact bytes, EMPTY at each task. Visit blocks/rows in original `collate` order; a hit moves the key to most recent, a miss inserts and evicts the oldest if full. Within-call repeats follow the same row order. Save per-call N/W and ordered SHA256 key digests; ensure no two active distinct exact keys share a digest, otherwise abort as invalid. Replay LRU arithmetic independently from that ordered digest stream. This verifies counter arithmetic, not full-search execution.

Primary per-repeat fraction: `sum(W for hit occurrences) / sum(N*W for all scorer calls)`. This counts padded convolution positions, not true runtime savings. Report raw row-hit counts, calls and within/cross-call descriptive splits separately; no secondary gate. A repeat with zero denominator is invalid, not zero reuse. Records are ordinary JSON, buffered per task and written outside that task's solve window; the wrapper's key/hash/LRU overhead stays INSIDE the original prospective deadline. Thus the profile describes the instrumented search, not a counterfactual uninstrumented trajectory. Do not subtract overhead or claim speed from it.

## Interpretation and checks

PASS means all three primary fractions >=1/4 and their max-min <1/20. Complete failure parks this fixed width-aware 4096-entry scheme; it is not a general negative result about caching. Preserve all tasks/repeats, with no favourable subset or adjusted cap.

Numerical compatibility remains OPEN. Mathematically identical inference can differ bitwise with batch shape ([PyTorch 2.8 numerical-accuracy documentation](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html)). Cache implementation would need its own model/device/dtype invalidation, padding/batch-shape, pooled features, `last_features`, legal-mask and cumulative-score/first-fit compatibility contract, then an upfront end-to-end comparison including key/lookup costs. No cache may replace scoring on this diagnostic's evidence alone; normal BEST gates remain.

After all profile cells, replay returned programs on shown examples only as an integrity check. Do not use hidden grading or the reference checker: shown fits are not verified coverage, and this diagnostic claims none. Retain all-attempt solve clocks, subprocess/full-phase wall and resource use descriptively, not as speed evidence. Independently recompute metrics from raw key records and confirm source/input/model continuity. Record provenance; no new codec, full-search verifier or manifest framework.

## Scope, resource and implementation bounds

Small experiment-only wrapper/runner plus focused manual/dummy-model tests. Production/BEST unchanged. Commit in a fresh own worktree and obtain required high-effort other-family code/design review BEFORE real panel/model execution. Dummy tests are not real provenance evidence.

One expected ten-minute CPU diagnostic in its own persistent unit with enforced RuntimeMaxSec=1200, MemoryMax=12G and MemorySwapMax=0. Output/log/cache under main checkpoints, exact launcher and actual controls in desk runs log; wrapper exit0 and complete all-cell result required. Post-run replay/checks also run capped. No training, model authoring, paid API or threshold/cap retry. If the construction cannot fit, record incomplete and reassess openly rather than retrying favourably.

Status metadata, 2026-10-15: this preregistered body preceded implementation and data. The one profile completed; independent code/design and final result reviews passed. See RESULT.md and FINAL_REVIEW.md. The preregistered body above is unchanged.
