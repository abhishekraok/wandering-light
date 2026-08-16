# Versioned basis libraries and default-solver usage measurement

Date: 2026-08-15

## Summary

This work did two useful things:

1. It added stable, versioned basis-function sets so checkpoints, corpora, and
   usage files can identify the exact functions they use.
2. It generated a large synthetic corpus, ran the existing
   `rl-6k-with-lp` solver, and counted function occurrences only in verified
   successful solutions.

In round numbers:

| Stage | Result |
|---|---:|
| Generated trajectories | 84K |
| Trajectories used for the main measurement | 72K |
| Verified solver successes | 31K |
| Function calls in successful solutions | 62K |
| Mean calls per successful solution | 2 |

The remaining 12K trajectories were held out. No newly trained model is part
of this report. The exploratory SFT comparison was deleted because the basis
change was too small to justify retraining.

## Basis identities

The registry contains immutable JSON manifests under
`wandering_light/basis_sets`. Artifacts store the resolved basis ID and digest,
not a mutable alias.

| Basis ID | Alias | Functions | Purpose |
|---|---|---:|---|
| `wl-core-pyhash-v1` | `checkpoint-rl-6k-with-lp` | 118 | Checkpoint-era definitions |
| `wl-core-v1` | `default` | 118 | Current deterministic definitions |
| `wl-pilot-compressed-v1` | `pilot-compressed` | 116 | Exploratory redundancy cleanup; not promoted |

`wl-core-v1` keeps the checkpoint-era names and ordering. It changes only
`set_to_list`, `str_hash`, and `set_hash` to deterministic implementations.

The loader verifies the manifest, per-function fingerprints, and pinned digest.
Execution rejects mutated or mixed-basis definitions. Usage files are keyed by
stable function IDs and bound to one basis ID and digest.

## Corpus and default-solver measurement

The corpus contains 84K unique random-walk `TrajectorySpec` records:

- discovery: 60K
- validation: 12K
- held-out test: 12K

It is balanced across 12 input types and requested walk lengths one through
five. It rejects execution failures, identity outputs, constant outputs, and
exact duplicate input/output tasks.

The default `rl-6k-with-lp` checkpoint solved about 31K of the 72K discovery
and validation tasks, or roughly 43%. Those verified solutions contained about
62K function calls, averaging two calls per solution.

This is much lower than the roughly 85% previously observed on
`random_inputs_500`. It is not a regression on the same benchmark. The new
generator deliberately uses broader inputs: boundary and large numbers,
Unicode and structured strings, binary values, larger containers, and stepped
ranges. It also removes easy identity and constant-output cases. On the old
benchmark, applying comparable filters still leaves a solve rate around 77%,
so most of the remaining gap is input-distribution shift. The current solver
appears substantially fitted to the old narrow input generator.

Runs retained in the personal W&B entity:

- [Main 72K measurement](https://wandb.ai/abhishekraok-na/wandering-light-basis/runs/3fefq3zb)
- [Original-checkpoint held-out anchor](https://wandb.ai/abhishekraok-na/wandering-light-basis/runs/su6962pc)

## What usage means

The tracker records only a complete solution that executes successfully and
matches the requested output. It counts total occurrences, including repeated
calls in one solution. Failed predictions and random search activity do not
count.

The measurement therefore answers:

> Which functions does this solver use on this generated input distribution?

It does not by itself answer:

> Which functions are necessary in a minimum basis?

Usage should weight demand after semantic redundancy has been established. It
should not directly decide deletion.

The most frequent functions in this run included boolean conversions,
bytearray-to-bytes conversion, bytearray length, and ASCII-byte detection.
Two specialist functions received no successful calls. That does not prove
they should be removed: zero usage can also mean the model failed to reach that
part of the task distribution.

## Trajectory collapse

Requested random-walk length is not semantic or shortest-path length. The
corpus filters removed obvious identities and constants, but substantial
collapse remains.

For walks of requested lengths two through five, a proper subsequence of the
generating walk produced the same output in roughly 22%, 50%, 71%, and 84% of
tasks. Among successful nominal length-four and length-five tasks, the solver
almost always found a shorter program. Most successful programs were only one
or two calls long.

This does not invalidate the recorded solver calls, but it makes requested
walk length a poor difficulty label. It also means the corpus does not yet
study genuinely long compositions.

There are two separate fixes:

1. Remove semantically redundant basis functions and aliases.
2. Generate trajectories whose intermediate steps add behavior, reject
   cancellation and idempotent collapse, and certify or bound the shortest
   equivalent path before assigning a length.

Removing redundant functions alone is not enough. Even an irredundant basis can
produce `reverse -> reverse`, inverse pairs, or idempotent repetitions.

## Redundancy findings

The first inspection found three clear deprecation candidates:

- `identity_int`: semantic no-op
- `bool_identity`: semantic no-op
- `duplicate`: exact alias of retained `repeat`

It also found a frequent two-call sequence,
`bytearray_to_bytes -> bytes_is_empty`, for which `bytearray_is_empty` is a
possible one-call macro.

These changes are preserved in the exploratory child
`wl-pilot-compressed-v1`, but that child is not the new default. It is a small
compression example, not a substantially more expressive basis.

## Better objective: a minimum-cost generating basis

The next iteration should build a typed derivability graph. For each function,
search for compositions of the remaining functions with equivalent semantics.
Exact aliases and no-ops can be removed immediately. Other functions should be
judged by the cost of their best replacement, not merely by whether a
replacement exists.

The target is a minimum-cost generating basis that balances:

- semantic coverage;
- basis size and description cost;
- shortest execution cost;
- solver search cost;
- measured demand.

New functions should be evaluated in two separate groups:

- **Compression functions:** already derivable, but make frequent expensive
  paths much shorter.
- **Expressiveness functions:** reach useful transformations outside the
  current basis closure.

A corpus generated only from the current basis cannot test expanded
expressiveness, because every target is already reachable. Expressiveness
candidates require tasks drawn from a larger function universe, real tasks, or
hand-specified transformations that the old basis cannot represent.

LLM retraining should happen only after a material vocabulary change and after
there are enough targeted examples for the new functions. It should not be
part of every small basis edit.

## Artifacts

All remote artifacts are in personal entities.

- Code: [draft PR #32](https://github.com/abhishekraok/wandering-light/pull/32)
- Corpus, usage results, basis manifests, and report:
  [`abhishekraok/wandering-light-basis-pilot-20260815`](https://huggingface.co/datasets/abhishekraok/wandering-light-basis-pilot-20260815)
- W&B project:
  [`abhishekraok-na/wandering-light-basis`](https://wandb.ai/abhishekraok-na/wandering-light-basis)

The two exploratory SFT model repositories, their W&B runs, and their paired
evaluation artifacts were deleted.

## Recommended next iteration

1. Compute the typed semantic-derivability graph for the current 118 functions.
2. Identify a Pareto frontier of smaller bases versus replacement path cost.
3. Propose a broader set of functions that expands semantic coverage, not just
   one derived macro.
4. Generate and shortest-path relabel tasks from the revised basis; reject
   collapsed long walks.
5. Run the default solver and use verified occurrence counts as demand weights.
6. Retrain only after the basis has changed enough for the comparison to be
   meaningful.
