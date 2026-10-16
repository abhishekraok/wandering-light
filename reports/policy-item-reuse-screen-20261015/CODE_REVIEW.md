CODE REVIEW PASS

**Reviewed HEAD:** `23c260752070d0401f16b4848586701bb2083178`, clean. DESIGN.md is byte-identical to its registration commit `ce9bd16`, which came before the implementation commit `8bc22d2`. Nothing under `wandering_light/` differs from `origin/main`; the branch adds only the 5 new files, so production and BEST are unchanged.

## Checks I ran
All ran inside `wl-item-reuse-review-code1-20261015.service`; I confirmed memory.max is 12 GiB and swap.max is 0.

- **Focused suite:** 52 passed. The tree was clean afterwards.
- **In-process check with an untrained synthetic `StackPairPolicy`:** both encodings (machine and difference), 5 typed tasks, real `beam_solve` at width 64, guard both, atom cap 16384.
  - The beam results matched except for the clock.
  - On every call, raw logits, `last_features` and legal-masked `log_probs` were bitwise equal.
  - The recorder's W, N and digests matched sha256 of `<Q W>` plus the int64 rows of production `collate`'s dense tensor.
  - The totals matched both an LRU I wrote independently (at capacity 4096 and at 3) and `replay_counts`.
- **Real phase with real child processes:** real `run_phase`, real `cell` subprocesses, provenance, host snapshots and `summarize`. Only `prepare_inputs` was replaced, with 24 typed synthetic tasks and an untrained checkpoint; this is not provenance evidence.
  - Exit 0, 24 s, 0.93 GB max RSS.
  - Threads and interop threads were 1, and the thread environment variables were 1.
  - All module origins resolved inside the frozen tree, the cgroup was recorded, and all three cells completed.
  - A second variant that ended in `max_depth` terminals also exited 0.
- **Source semantics:**
  - **Search settings:** these match production `solve_tasks` with library off: flat `beam_solve`, deadline `t0 + 5.0`, `run_body.cache_clear()` and `gc.freeze()`.
  - **Keys:** PAD is 0, blocks are int16 widened to `<i8`, and W is the per-call maximum, as in `collate`.
  - **LRU:** a hit moves the key to most recent and a miss evicts the oldest only when full. Rows are visited in collate order. Within-call versus cross-call is decided by the call that last touched the key.
  - **Digest collisions:** a collision between distinct active keys is caught by comparing exact bytes.
  - **Gate:** the conjunction uses exact `Fraction`s (`>= 1/4`, range `< 1/20`), and a zero or invalid denominator raises.
  - **No cache served:** `super().logits` is always called.
  - **Panel selection:** new salt, first original row per skeleton, no refill. The manifest pins match the design (revision `00c2a99…`, train SHA, 171 skeletons, at most 6 examples per task).
  - **Child inputs:** the child sees only `source_index` and the shown inputs and outputs.
  - **Ordering:** replay of returned programs on shown examples runs only after all three timing cells, with no hidden examples or reference checker.
  - **Continuity:** input, source and checkpoint pins are re-checked, the raw cell files are re-hashed, and `INCOMPLETE.json` is written on any failure.
  - **Deadline:** the observer runs inside the original deadline.
  - **Acquisition:** it is identical to the previously executed gumbel runner.

## Blockers
None.

## Non-blocking notes
1. At `experiments/policy_item_reuse.py:89` the eviction `pop` sits inside an `assert`, so under `python -O` it would vanish. The runner never uses `-O`.
2. The child's module origins are recorded but not compared with the parent's. The child checks on its own that they are under ROOT. `experiments` is a namespace package that also includes the main checkout, but both experiment modules used resolve to the frozen tree and are checked.
3. The weights are pinned only by the revision path. A cached checkpoint is reused without comparing its contents, though the file SHAs are recorded. That fits "record provenance, do not enforce it".
4. **Worth carrying into the result.** Because W is part of the key and W changes from layer to layer, a cross-call hit needs exactly the same W in two calls. Cross-call hits were 0 in every synthetic run. A PASS could therefore come almost entirely from duplicates within a call, which argues for deduplicating within a batch rather than for a persistent cache. Report the within/cross split next to the reading. The item encoder is mathematically width-invariant, so the width-aware key undercounts reuse; the design already registers it as conservative.
5. The observer costs about 4.5 ms per 384-row call, against about 83 ms for one untrained default-architecture forward pass. This is descriptive only, not a speed claim.

## Limitations
- This PASS covers source and design readiness only. It says nothing about real provenance, data, the reuse rate, numerical compatibility or speed.
- `prepare_inputs` (fetching the real corpus and checkpoint) was checked by reading the source only, as the pre-data limit requires.
- Synthetic results are not evidence.

## Ready for the registered one profile
Yes.

**Docs updates:** changing pending to passed in docs only (STATUS, RUN.md, the DESIGN status line with the preregistered body left alone) does not need a new review. The condition is that `git diff 23c2607 <new> -- experiments tests wandering_light` comes back empty. Note that such a commit changes the `head` that provenance records, so the run log should cite both commits. Any executable change needs a separate review.

The full review is in `/home/abhishekrao/repos/wandering-light/checkpoints/policy-item-reuse-screen-20261015/review-code1/REVIEW.md`, and the scratch scripts and synthetic run outputs are in `scratch/` next to it.
