# DESIGN review: score-band-retention (#179) — **CHANGES_REQUIRED**

Three blockers, each a small text repair to `DESIGN.md`; no redesign is needed. The core retention mechanism is coherent, and I could not break it.

**Revision:** `69e53c80d121796fbd765cfbdd13166be0d34e46`, clean tree. Its only change over the reference base `f9ae90b` is `DESIGN.md` (+189 lines).

**Inspected (source reading and reasoning only; nothing executed, no raw task, hidden, archive or checkpoint payloads opened):**
- `reports/score-band-contract-20261003/DESIGN.md`, in full.
- `wandering_light/stack_beam.py`: `BeamObserver`, `BeamResult.pruned`, `beam_solve`, `_beam_solve`, `step_all_*`, `atoms_exceed`, `key_of`.
- `wandering_light/stack_policy.py`: `logits`, `log_probs`, `node_encoder`, `encode_stack`.
- `wandering_light/quotation_guard.py`: `blocked`.
- `wandering_light/stack_debugger/trace.py`: the width-cut labelling in `layer_retained`.
- `tests/test_stack_beam_equivalence.py`.
- Grep hits only in `docs/metrics.md`, `reports/short-segment-20260924/report.md` and the long-horizon audit note.
- `AGENTS.md` was not opened; the conventions I applied are the ones in the project instructions.

## Blockers

### B1. "Finite toy state graphs" can make fixture 6 self-verifying
- **Problem:** production `_beam_solve` hard-wires `OPS`, `step_all_*` and `parse_stack`, so it cannot run on an abstract graph. Fixture 6 asserts reference outcomes ("reference fits, finite arm fails"; "retains the exact reference first fit").
- **Consequence:** if the candidate gets an injectable transition function or op set to run graphs A/B/C, the "reference" there can only be the candidate with `None`. That is the fork verifying itself, on a code path fixture 1 never touched.
- **Repair:**
  - Every fixture is a hand-written mark-28 `Task` on the real machine; "graph" means the reachable states induced by a scripted scorer.
  - The candidate has no transition or `OPS` injection.
  - Every reference outcome and trace comes from imported production `beam_solve` with a `BeamObserver`, with the `stack_beam.py` hash in the manifest.
  - State how the scripted scorer addresses a state (call index, row, op), since equal capped blocks cannot be told apart by content.

### B2. NaN/+inf "rejects" is undefined and collides with the production scorer
- **Verb is unspecified:** "rejects" could mean raise or drop the candidate. These give different frontiers, counters and terminations.
- **Scope is unspecified, and the sort depends on it:** `sorted` over keys containing NaN has no defined order. The check must therefore run before the sort, over every non-fit executable successor. Otherwise the early cutoff makes rejection depend on scan position.
- **Counterexample to a whole-matrix check:** `PolicyScorer.log_probs` returns an all-NaN row for a node with no legal op (`peak` becomes 0, `shifted` is −inf, `norm` is log 0, and −inf − (−inf) is NaN). The reference tolerates this because those entries are never indexed. A beam with one dead node and one live node would be falsely rejected by a matrix-wide finite check.
- **Same row breaks trace comparison:** fixture 1 compares "normalized scores", and NaN ≠ NaN under naive equality.
- **Repair:**
  - Finite mode raises `ValueError` after scoring and before sort or any seen mutation, iff a cumulative score at a successor `(node, op)` is NaN or +inf. Collector restored, counters as spent.
  - Entries at illegal positions are never inspected.
  - Trace comparison is over legal entries, or NaN-aware.
  - Invalid `score_band_nats` raises before any work, next to the existing `spent_in` unit check.
  - Add the dead-row-plus-live-row fixture.

### B3. Stop cause and the unread tail are not exact enough to freeze expected outcomes
- **Tie cases are unspecified:**
  - W fills on the last ranked successor. The reference gives `pruned=False` here.
  - W fills while the next unread candidate would be below the band.
  - An all −inf fallback layer.
- **Unread tail has no stated status:** the design avoids the debugger recorder, which labels every unreached successor as a width cut. It never says what the tail is instead.
- **Repair:**
  - `exhausted` iff every ranked successor was read.
  - `width` iff W was reached with successors unread; this is the only case that sets width-pruned, exactly as the reference.
  - `score_band` iff the scan stopped at an eligible below-band candidate before W.
  - Unread successors carry no verdict and no band or width count; their eligibility is unknown.
  - Record `width_pruned` and `band_pruned` separately. #168's `saturated` means "every wider beam runs the same search". That stays true when only the band pruned, so a single `pruned=True` would wrongly deny it.

## Assessment by question

1. **Single mechanism: coherent.**
   - Python's `sorted(..., reverse=True)` is stable, so "nonincreasing, enumeration order on ties" matches the reference.
   - The single-scan form equals the C-prefix definition. Scores are nonincreasing, so the first eligible below-band candidate ends the prefix.
   - A same-layer duplicate of a band-dropped state is necessarily below the band too.
   - The only new seen-set consequence is that band-dropped states are forgotten. Width-dropped states are already forgotten in the reference.
   - Current-layer fit order is untouched: fit is tested before scoring. Later-layer divergence is stated, and no library is in scope.
2. **Scores.**
   - The −inf handling is consistent: a finite anchor excludes −inf, and an all −inf layer retains C.
   - The stated rationale is wrong but harmless. `a − tau` with finite tau never forms −inf − (−inf), and IEEE `-inf >= -inf` already gives the fallback.
   - The NaN side is B2.
3. **Disabled equivalence.**
   - It is adequate once B1 is fixed.
   - Typed identity versus capped encoding is meaningful: `encode_stack` keeps only the top `cap` tokens, so distinct deep states do share blocks, and a candidate deduplicating on block bytes would be caught.
   - The refusal, budget, deadline, collector and observer checks are all testable against the reference's hooks.
4. **Telemetry.** The intent is right; exactness is B3.
5. **Gate A bounds.** They are bounded and practical. The grid is tiny and 300 s is ample. Useful and adverse constructions are both retained, expected outcomes are frozen first, and both reviews are named. I would add no process.
6. **Research value.**
   - Gate A is a unit-test suite with a registered manifest. Its "prediction" can only fail on a bug, and graph A's reduction holds by construction. It says nothing about width, horizon or spend, and the design admits this.
   - Retention over priority is the right choice. It keeps fit order, batching and layer structure, so the arm is a pure per-layer frontier reduction.
   - It does not reduce the #168 confound by itself. A band at W only ever narrows the frontier, so the informative control is a narrower fixed-width beam at matched spend, which the "Later work" section does not name.
   - An absolute tau on unnormalized cumulative scores may be near-greedy at depth or never bind. A train-only read of score gaps by rank and depth against witness rank should come before threshold selection, and may kill the arm.
   - The gate is worth building only because it is cheap and experiment-local.

## Nonblocking limits

- **Dtype:** `PolicyScorer` returns float32, and whether the cumulative score stays float32 depends on the NumPy version. Pin the band comparison to float64. Exactly representable toy scores hide this in Gate A.
- **Uncovered settings:** fixture 1 omits guard mode `body`, the `fuel` and `executions+fuel` budget units, and `max_state_atoms` on versus off. Cover them or make the candidate refuse them.
- **Normalization:** the solver uses whatever the scorer returns. Say that scripted scores need not be normalized.
- **Unpriced checks:** guard, size and seen checks on the cutoff candidate are not counted, same as the reference. The scan must keep reading past ineligible below-band candidates in order to set `band_pruned`.
- **Literature:** the Freitag and Meister citations and the `bfbs` SHA were not checked. No claim transfers from them, and the design says so.

A pass on the repaired design would authorize synthetic-fixture implementation only. Code review is still required before Gate A, and code and result review before landing. Nothing here authorizes real-data calls, a threshold, a panel or `BEST`.
