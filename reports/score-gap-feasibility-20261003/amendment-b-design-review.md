VERDICT: AMENDED DESIGN PASS

Review of frozen `7e82a74232c012deaa9616deaeff43057c66c242` for score-gap-feasibility (#180): Amendment B design plus synthetic foundations. B1-R is fixed and I found no new blocker. This pass accepts the amended design and synthetic foundations only, and permits finishing the implementation for a separate exact-code review. It does not authorize a data launch.

The tree was clean before and after. Nothing was written or committed, and no data, weights, trained scorer, `run_gate` or #179 entry point was touched. The permitted pytest run left three ignored `__pycache__` directories, which I removed.

## Audit of both prior CHANGES_REQUIRED reports

Both retained reports match their receipt hashes (`71255bbc…` for `design-review.md`, `2860c20c…` for `amended-design-review-report.md`), so I audited the actual texts.

- **B1-R (uint8 reason dictionary): resolved.**
  - `DESIGN.md:85-94` stores no reason; the record is 19-byte `<BBBQQ`.
  - `step_all_explained` (`stack_beam.py:267-283`) builds the reason from `step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)` and the loop index, which is the pinned re-derivation.
  - `stepped == len(parent)` means success and `stepped < len(parent)` means refusal at that index, so no string parsing is needed.
  - `run_body`'s `lru_cache` does not affect results: my replay with the cache cleared matched exactly.
- **B1 (cap and bytes): resolved.**
  - `len(OPS)` is 28; 144 × 83,244 = 11,987,136, below 12,000,000.
  - Attempts take 227,755,584 bytes; scores plus masks take 59,935,680.
  - Depth 0–47, node 0–63 and op 0–27 fit uint8, and the codec rejects out-of-range values, bools, floats and truncated or over-long payloads.
- **B1 (resource-stop mechanism): resolved.** An exception raised from the observer propagates through `beam_solve` with no terminal callback, so it cannot appear as `wall_clock_cap`. The 2 GiB typed-state bound is still labelled as enforced, not guaranteed.
- **B2: resolved.** `gate_components` compares `Fraction` against 19/20 and 1/10 (`score_gap_profiles.py:276-285`); the code is unchanged since `51e8514`.
- **B3: resolved.**
  - The whitelist at `DESIGN.md:51-53` now drops `program`, matching `project_shown`.
  - Hash order, minimum representative, first source row, even-is-selection, validation fallback, the zero-C rule, float64-only band test and source-parent legal rows are unchanged from the version the last review accepted.
- **B4: resolved.** `NO_HEADROOM_OVER_PREFIX32` is selection-half only with tau locked null, and `DESIGN.md:15-17` claims no route kill.
- **Frozen parts untouched.** The `51e8514..HEAD` diff of `DESIGN.md` touches only the whitelist, the record format, terminal precedence and the confirmation-failure report. Grid, conjunction, caps and selection are unchanged, and production `wandering_light` has no diff from `88444b4`.
- **Prior obligations now in the text:** terminal precedence (`DESIGN.md:145-148`), confirmation prefix protection beside a failed conjunction (`:204-205`), and full-matrix NaN/+inf validity (`:163-164`).

## Reproduced evidence

- **Scoped pytest:** 85 passed in 5.63 s with the permitted command (a test duration, not a measurement).
- **Ruff:** `ruff check` and `ruff format --check` with `--no-cache` are clean on all six files.
- **Synthetic probe** (in `/dev/shm`, removed): four manually typed tasks, a seeded random scorer, width 64, depth 7, guard both.
  - All 37,436 attempts round-tripped through the codec and replayed to the exact moved product, reason string, fuel and stepped, including 99 nested-prefix refusals and 142 refusals after at least one example had succeeded.
  - Summed record fuel and stepped equalled the `BeamResult` counters.
  - Not exercised: no fuel-exhaustion or quotation-depth refusal occurred.
- **Budget stop mid-layer:** a `budget=1501` run ended with 1,502 executions, 48 expansions and 82 started nodes. Expansions therefore cannot be derived from `layer_started` sizes, as `DESIGN.md:105-108` already says.

## Nonblocking scope notes

- **Codec test is not independent verification.** It uses the same `step_raw` loop as production, on two layers of one task. It shows the format is sufficient, nothing more; `DEVELOPMENT.md` says this.
- **Stale wording.** `AMENDMENT_A.md:10` still says 20-byte records and 239,742,720 bytes; `DESIGN.md` and Amendment B are operative. `STATUS.md` says the review "is being prepared", which will need updating with this verdict.
- **#172 not confirmed fixed.** Unretained successors are not bounded by `max_state_atoms`, so a 4 GiB kill during search or replay is possible; the design discloses it as a no-result.
- **Support is still borderline.** `INSUFFICIENT_LINEAGE_SUPPORT` and `NO_HEADROOM_OVER_PREFIX32` remain the likely terminals; the descriptive tables are the durable output.

## Points to pin in the exact-code review

None of these depends on an outcome, so pinning them then is still before data.

1. **Invalid-score cell.** `DESIGN.md:163` says NaN/+inf "invalidates the cell" but not the terminal. State whether that is a run-wide provenance non-pass (my reading of `:23-24`) or a cell kept in denominators without a label.
2. **Refusal classification rule.** `DESIGN.md:91-93` requires distinguishing ordinary, fuel and depth refusals but gives no rule. Pin a suffix match on the replayed reason (`out of fuel`, `quotation nested too deep`) after any `op: ` prefixes.
3. **Verifier and the 2700 s deadline.** `DESIGN.md:81` lists "replay" inside the global deadline, while `:234-236` puts independent verification after the run. Say explicitly that the in-run replay is returned programs only and the full attempt replay is outside the deadline. Search alone can take 1440 s.
4. **Record framing.** The 19-byte record has no cell identifier. The framing must give per-cell and per-layer boundaries, and the verifier must check that attempts form a lexicographic `(depth, node, op)` prefix with node below the layer's actual beam size. The codec only checks the global 0..63 range.
5. **Partial last layer.** On a `budget` or `wall_clock_cap` stop, a layer has a start beam and attempts but no scored or retained callback. The recorder must seal it as partial, and expansions must come from `BeamResult`.
6. **Start-beam storage.** Layer d+1's start beam is layer d's retained beam; store it once, or the 2 GiB budget is spent twice.

## Later implementation obligations (unchanged, none built)

- Bounded file recorder with global byte, attempt and deadline exceptions.
- Attempt reservation before any payload read.
- Atomic selection receipt, sealed before confirmation summaries are opened.
- Runner-level terminal precedence, with both-half support checked before selection.
- All eight profiles and the descriptive tables retained on every terminal.
- Full-matrix finite check.
- Independent serialized-result verifier that does not call `score_gap_profiles`.
- `SyntheticCapture` must not be the recorder.

## Scope

Reference-snapshot protection is not behavioural verification or survival in an executed band. The width-32 prefix is not an executed narrower search, and slots are not clock. Nothing here supports a BEST or local performance claim. This pass does not transfer to changed complete code or to the result checker.

Not checked: the side file, corpus, checkpoints and issue thread were not opened.
