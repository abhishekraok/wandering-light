# Versioned basis libraries and usage-driven pilot

Date: 2026-08-15

## Executive result

This work adds immutable, content-addressed basis-function sets and tests the first usage-driven challenger, `wl-pilot-compressed-v1`, against a matched `wl-core-v1` control.

The challenger passed the synthetic held-out pilot:

| Metric | Control | Challenger | Paired change |
|---|---:|---:|---:|
| Test tasks solved | 5,789 / 12,000 (48.2417%) | 5,823 / 12,000 (48.5250%) | **+34 solves, +0.2833 percentage points** |
| Solve-rate 95% paired-bootstrap CI | — | — | **[+0.0417, +0.5250] percentage points** |
| Common-success path length | 1.97788 | 1.95961 | **−0.01826 (−0.923%)** |
| Common-success execution-weighted proxy | 33.99833 | 33.81058 | **−0.18775 (−0.552%)** |
| Active functions | 118 | 116 | **−2** |
| Measured throughput | 115.21 tasks/s | 118.96 tasks/s | +3.26% directional |

There were 5,695 tasks solved by both arms, 6,083 failed by both, 128 challenger-only solves, and 94 control-only solves. The exact two-sided McNemar test gives `p = 0.02655`. The common-success path-length bootstrap CI was `[−0.02335, −0.01335]`, and the execution-weighted CI was `[−0.26918, −0.10724]`.

The original unadapted `rl-6k-with-lp` checkpoint solved 5,076 / 12,000 (42.3000%) on the same test split and protocol. This anchor separates the large benefit of matched adaptation from the smaller basis-library improvement.

**Decision:** preserve the challenger as the accepted basis for the next iteration, but do not move the `default` alias yet. The evidence is positive and paired, but it comes from one checkpoint, one training seed, and a synthetic random-walk distribution. Promotion should follow a real/replay benchmark and repeated-seed confirmation.

## Basis registry

Basis libraries are ordered JSON manifests under `wandering_light/basis_sets`. An immutable basis ID maps to an exact function sequence and an index-pinned SHA-256 digest. Each function also carries a stable readable ID and a fingerprint of its name, types, code, and metadata.

| ID | Alias | Functions | Digest | Purpose |
|---|---|---:|---|---|
| `wl-core-pyhash-v1` | `checkpoint-rl-6k-with-lp` | 118 | `sha256:21959c1255caf37b9b2665bf0d83e5c24fc7925e49b2377ddd44e9f2c7d6d4b6` | Exact checkpoint-era palette, including Python-hash behavior |
| `wl-core-v1` | `default` | 118 | `sha256:ffc602fb24db249df7eb4f6b0d5ba38d5a5070b2c68c6c5f94c7f387de682494` | Current deterministic core |
| `wl-pilot-compressed-v1` | `pilot-compressed` | 116 | `sha256:60d6ebe054b600410f58b25ff1d3f9cb62b234f3825fdb1c4f71c47eb24fd1c0` | First measured challenger |

Aliases are conveniences, not artifact identities. Checkpoints, usage files, corpora, and evaluation outputs persist the immutable ID and digest.

```python
from wandering_light.basis_set import load_basis_set

basis = load_basis_set("wl-pilot-compressed-v1")
function_defs = basis.as_function_set()

print(basis.identity_dict())
# {
#   "basis_set_id": "wl-pilot-compressed-v1",
#   "basis_set_digest": "sha256:60d6...fd1c0",
# }
```

Loading enforces the manifest schema, function fingerprints, manifest digest, and index-pinned digest. Execution snapshots the definitions and rejects mutation, conflicting definitions, and mixed-basis provenance. The historical Python-hash palette additionally requires a process started with a fixed `PYTHONHASHSEED`; its original training seed is unknown, so exact historical hash outputs cannot be guaranteed.

Usage tracking is now basis-aware and keyed by stable function IDs. Its primary `invocation_count` is total occurrence, including repeated calls within a solution; `solution_count` measures task coverage. Recording occurs only after the complete trajectory has executed and matched the requested output. Search candidates and random-solver activity do not contaminate the signal.

## Refining the proposed loop

The proposed solve–measure–modify–repeat loop is sound, with one important refinement: frequency should generate candidates, not decide deletion by itself.

The implemented loop is:

1. Freeze a stratified corpus and its discovery, validation, and untouched test splits.
2. Solve discovery and validation with the incumbent checkpoint.
3. Count total occurrences only in independently verified solutions.
4. Mine repeated verified subsequences and identify semantic redundancy.
5. Express changes as a new immutable child manifest; deprecate by omission, never by rewriting history.
6. Counterfactually rewrite and execute incumbent solutions under the child basis.
7. Retrain a matched control and challenger because changing the available vocabulary changes the solver's action space.
8. Compare both arms on identical held-out task IDs with paired statistics and a cost-aware objective.
9. Promote only after accuracy, cost, regression, and reproducibility gates pass.

This differs from “remove the least-used functions” in a useful way. Low use can indicate redundancy, but it can also identify a rare specialist or a region the current policy fails to reach. Conversely, a frequently used identity can still be pure waste. This pilot therefore retained low-frequency specialists and removed only functions with a semantic proof or an equivalent retained replacement.

A practical future objective is to minimize expected verified execution cost plus a weighted library-description cost, subject to a solve-rate non-inferiority constraint. Total occurrence remains the requested primary usage signal and an effective way to prioritize what to inspect.

## Relation to prior work

The design follows the solve–compress–retrain pattern in [DreamCoder](https://arxiv.org/abs/2006.08381), which alternates library extension with recognition-model training and replay. [Stitch](https://arxiv.org/abs/2211.16605) motivates mining recurring program structure and judging abstractions by compression rather than occurrence alone. [babble](https://arxiv.org/abs/2212.04596) shows why equivalence-aware mining can outperform purely syntactic matching, a natural next step for redundant or differently spelled trajectories. [LILO](https://arxiv.org/abs/2310.19791) reinforces the value of an iterative synthesize–compress–document loop and interpretable names.

Changing a basis also changes the policy's action space. [Growing Action Spaces](https://proceedings.mlr.press/v119/farquhar20a.html) provides a related reinforcement-learning perspective: curricula and transfer are needed when action spaces change. Finally, the library-size and path-size accounting is motivated by Rissanen's [minimum-description-length principle](https://doi.org/10.1016/0005-1098(78)90005-5). This pilot reports MDL-like telemetry but does not yet optimize a calibrated end-to-end MDL objective.

## Corpus and incumbent measurement

The fixed corpus contains 84,000 unique random-walk `TrajectorySpec` records, stratified across 12 input types and witness lengths 1–5:

- discovery: 60,000
- validation: 12,000
- untouched test: 12,000
- manifest digest: `sha256:40fa81a3376fb45b41e93dc95dcc23e60a666c024aca5f40b184da891e15025c`

The incumbent was the personal checkpoint [`abhishekraok/induction-basicfns-opt125m-sft434k-rl-6k-with-lp`](https://huggingface.co/abhishekraok/induction-basicfns-opt125m-sft434k-rl-6k-with-lp) at revision `0c4ea07bfa618321b8dc5ce956ce5a86560d99a7`, with lineage in [W&B run `dp8ylg8y`](https://wandb.ai/abhishekraok-na/wandering-light-rl_induction/runs/dp8ylg8y).

On discovery plus validation, deterministic budget-1 inference solved 30,758 / 72,000 tasks (42.7194%) with mean successful path length 2.0023. The run is [W&B `3fefq3zb`](https://wandb.ai/abhishekraok-na/wandering-light-basis/runs/3fefq3zb).

The checkpoint is historically associated with `wl-core-pyhash-v1`; this experiment evaluated it with the deterministic `wl-core-v1`. Those palettes have the same size and differ only in the three deterministic replacements recorded by the child manifest, but this remains a compatibility limitation rather than an exact historical replay.

On the untouched test split, the same checkpoint solved 5,076 / 12,000 (42.3000%) in [W&B `su6962pc`](https://wandb.ai/abhishekraok-na/wandering-light-basis/runs/su6962pc). All 5,076 successes were independently re-executed.

## Candidate selection

`wl-pilot-compressed-v1` removes three definitions and adds one:

| Change | Discovery + validation evidence | Rationale |
|---|---:|---|
| Omit `identity_int` | 306 occurrences | Semantic no-op; every use can be deleted |
| Omit `bool_identity` | 25 occurrences | Semantic no-op; every use can be deleted |
| Replace `duplicate` with `repeat` | 231 occurrences vs. 328 for `repeat` | Equivalent string operation; retain one spelling |
| Add `bytearray_is_empty` | `bytearray_to_bytes → bytes_is_empty` occurred 357 times on 357 tasks | Highest-frequency useful verified contiguous pair; collapses two calls to one |

This is intentionally not a blind bottom-of-the-ranking cull. For example, rare specialists such as `contains_space` and `endswith_z` remain because low use may be a task-distribution or policy-reachability effect.

Counterfactual rewriting of all 30,758 verified discovery-plus-validation solutions changed 796 programs and reduced 61,586 steps to 60,898: 688 fewer steps (−1.117%), with zero execution mismatches.

## Matched adaptation

Both arms started from the same incumbent checkpoint and used the same 25,709 verified discovery successes, selected by the identical task-ID digest:

`sha256:66ae20fa37876befc7043bdb1aff639bccb05cba62b833b9d1c4708a5db5b48b`

The control retained source solutions. The challenger rewrote solutions using manifest-declared rules, reexecuted them against frozen outputs, and then trained on the rewritten labels. Its 667 changed training programs contained 774 rewrite events because some programs needed more than one rewrite:

- 301 macro collapses
- 271 `identity_int` deletions
- 20 `bool_identity` deletions
- 182 `duplicate → repeat` replacements

| Training field | Control | Challenger |
|---|---:|---:|
| Records | 25,709 | 25,709 |
| Label steps | 51,406 | 50,814 |
| Training tokens | 2,295,361 | 2,291,736 |
| Steps / epochs | 804 / 1 | 804 / 1 |
| Batch / learning rate | 32 / `2e-5` | 32 / `2e-5` |
| Precision / seed | bf16 / 42 | bf16 / 42 |
| Final training loss | 0.047527 | 0.048252 |
| `model.safetensors` SHA-256 | `4a1064d082c8f812dd84b867a0999d00c49d9016a06369843970ef4e7dfdaa1c` | `79ffc10a75d56b68ce3d3398b2a93fcbde9eee2f8662d696c9275a85c763c230` |

The checkpoints are weights-only in the project sense: model, tokenizer, configuration, trainer metadata, basis manifest, and `wandb_run.url` are retained; optimizer, scheduler, and RNG state are not.

Both training manifests captured the same source snapshot: Git commit `0f843bb06e73a05e8a6b051556715e55f410fbb6`, tracked-diff digest `sha256:04ee3e23160955cc742fd5f94fee7d5a8b7ff4d72c14d0c808814a5cdad91e01`, and untracked-file digest `sha256:33bcb5a772f7de3145daad6db386f4095d40141e8a91fc2d0d41eef279e76dc2`.

Training runs:

- [Control: W&B `p2u6towk`](https://wandb.ai/abhishekraok-na/wandering-light-basis/runs/p2u6towk)
- [Challenger: W&B `j6mfdye2`](https://wandb.ai/abhishekraok-na/wandering-light-basis/runs/j6mfdye2)

## Held-out evaluation

The arms were evaluated sequentially on the exact same 12,000 untouched test records with deterministic budget-1 decoding, seed 4242, batch size 64, and `max_new_tokens=128`.

Both runs used `cuda:0` on the same NVIDIA GeForce RTX 4090 and have hardware fingerprint:

`sha256:00c076557cbef08110cc4e66f6573ee3238c742387622eea4c5e9621b24d17c2`

Evaluation runs:

- [Control: W&B `ks5ahtrp`](https://wandb.ai/abhishekraok-na/wandering-light-basis/runs/ks5ahtrp)
- [Challenger: W&B `by7fwvjc`](https://wandb.ai/abhishekraok-na/wandering-light-basis/runs/by7fwvjc)

No input type regressed by more than 0.3 percentage points; the worst was `dict` at −0.3 points. The execution-weighted metric is a transparent static AST proxy—dispatch plus selected code operations—with no measured per-function overrides. It is useful for relative comparison, but it is not a calibrated runtime or energy model.

The challenger emitted `bytearray_is_empty` on 66 successful test tasks, while the control emitted the source pair on 73. Bytearray solve count was effectively flat (683 versus 684). The total solve improvement was diffuse, so the +34 solves must not be attributed causally to the new macro.

String alias consolidation behaved cleanly: the control emitted `repeat` plus `duplicate` 120 times (79 + 41), while the challenger emitted `repeat` 119 times. Removed identities account for 56 control occurrences on successful test programs.

The measured throughput gain is also directional: the runs shared hardware and protocol, but there was only one sequential timing measurement per arm.

Library-size telemetry moved as follows:

| Representation | Control | Challenger | Change |
|---|---:|---:|---:|
| Functions | 118 | 116 | −2 |
| Function-code UTF-8 bytes | 2,883 | 2,873 | −10 |
| Canonical function-definition bytes | 31,238 | 30,765 | −473 |
| Canonical full-manifest bytes | 31,773 | 32,528 | +755 |

The full manifest grew because it preserves deprecation, replacement, corpus, and selection provenance. That is acceptable operational metadata, but a future formal MDL score should define whether provenance bytes belong in the model-description penalty.

The canonical paired analysis is [`paired-test-analysis.json`](./paired-test-analysis.json), internal analysis digest:

`sha256:320527659a86065b1e1c290043bb78081b13ff6313ad76f0efe2274745e21c72`

## Artifacts

All remote artifacts are in personal entities.

- Code: branch [`codex/basis-library-lifecycle`](https://github.com/abhishekraok/wandering-light/tree/codex/basis-library-lifecycle); draft PR link will be added after publication.
- Control model: [`abhishekraok/wandering-light-basis-v1-control-verified-sft1@f6926e5`](https://huggingface.co/abhishekraok/wandering-light-basis-v1-control-verified-sft1/tree/f6926e575e5820083c92bb7f29d8b90d71450d3c), tagged `wl-core-v1`.
- Challenger model: [`abhishekraok/wandering-light-basis-pilot-compressed-v1-verified-sft1@e1c63c3`](https://huggingface.co/abhishekraok/wandering-light-basis-pilot-compressed-v1-verified-sft1/tree/e1c63c3891a1274d12ec1be5eea73fef2b326a67), tagged `wl-pilot-compressed-v1`.
- Corpus, compact evaluations, manifests, and report: [`abhishekraok/wandering-light-basis-pilot-20260815@e51eb87`](https://huggingface.co/datasets/abhishekraok/wandering-light-basis-pilot-20260815/tree/e51eb87f5feb2f42399ece5bfb1aeea2fe3656c5), tagged `pilot-20260815`.
- W&B project: [`abhishekraok-na/wandering-light-basis`](https://wandb.ai/abhishekraok-na/wandering-light-basis).

## Validation

- `PYTHONHASHSEED=0 pytest -q`: **481 passed, 2 skipped** in 197.09 seconds.
- Ruff lint passed on every changed and new Python file.
- `git diff --check` passed.
- Regenerating all built-in basis manifests was byte-for-byte idempotent.
- A clean wheel build, isolated installation, manifest load, and imports of the evaluation and training packages passed.
- `ruff format --check` reports two formatter-only deltas in `wandering_light/evals/create_data.py` and `experiments/train_basis_challenger.py`. They were not mechanically rewritten after training so the committed executable source remains identical to the source snapshot captured by both model manifests.

All experiment artifacts were separately checked for exact task IDs, corpus and basis digests, model-tree digests, matched training source, protocol, and hardware. Every reported challenger success and every original-checkpoint anchor success was independently re-executed.

## Limitations

- The evaluation distribution is synthetic and generated from the current DSL; it does not substitute for real tasks or an external benchmark.
- This is one incumbent checkpoint, one corpus seed, one adaptation seed, and one held-out test.
- Validation informed candidate selection; only the test split remained untouched.
- Budget-1 deterministic decoding measures the chosen operating point, not broader search robustness.
- The static execution-weighted proxy is not measured compute cost.
- The current solver prompt does not enumerate the available basis definitions, so a renamed or expanded vocabulary requires matched adaptation and may remain hard for the model to discover.
- The exact Python hash seed used by the historical checkpoint basis is unknown.
- The current train/eval prompt serializers differ by one versus two trailing newlines. Both arms use the same serializer within each phase, so the paired comparison remains controlled, but the discrepancy should be removed.
- Candidate changes are bundled. The paired result validates the bundle, not the individual causal contribution of each edit.

## Recommended next iteration

1. Keep `wl-core-v1` as `default`; retain `wl-pilot-compressed-v1` as the registered next-iteration challenger.
2. Add a frozen replay set of real or hand-authored tasks before changing the default alias.
3. Repeat training and paired evaluation across several corpus and optimization seeds.
4. Add explicit per-function measured compute costs and define a constrained MDL objective.
5. Mine verified shortest programs for larger recurring subsequences; add semantic or equality-aware canonicalization before counting patterns.
6. Use difficulty and solver-surprise sampling after establishing the fixed random-walk baseline.
7. Condition the solver explicitly on basis ID or available definitions so library changes are first-class model inputs.
8. Continue deprecating through child manifests. Never delete or mutate historical manifests needed to load older checkpoints.
