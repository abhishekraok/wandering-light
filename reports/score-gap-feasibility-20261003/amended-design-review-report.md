VERDICT: CHANGES_REQUIRED

Review of frozen `51e85140c57e1fe53721f378de3fb70176c4dff3` for score-gap-feasibility (#180), amended design plus synthetic foundations. One blocker remains, in B1's record format; B2, B3, B4 and the nonblocking fixes from the original review are resolved. The tree was clean before and after, nothing was written or committed, and no data, weights, trained scorer, `run_gate` or #179 entry point was touched.

## Blocker

**B1-R. The `uint8` reason code cannot be shown to hold the reasons production emits, and overflow is undefined (`DESIGN.md:85-86`, `AMENDMENT_A.md:10`).**
- Production's reason is `f"example {stepped}: {outcome}"` (`wandering_light/stack_beam.py:280`), and `outcome` is an open set of strings.
- `run_body` prefixes the refusing op's name at each quotation level, up to depth 32 (`wandering_light/stack_machine.py:300-316`).
- Several reasons interpolate counts, e.g. `body must leave exactly one value, left {n}` (`stack_machine.py:327`, `500`, `524`, `539`).
- My synthetic probe (random typed stacks, all 28 ops, 1,120,000 `step_raw` calls) produced 304 distinct refusal strings, already over 255 with the example index stripped. That shows the field can overflow; it does not measure how likely that is on the real panel.
- The design gives no rule for the 256th reason. The recorder must then either abort, which burns the single launch like the original B1, or alias reasons, which is not faithful.
- Fix, any one of:
  - Widen the code to `uint16` and re-derive the bound. A 22-byte record gives 263,716,992 bytes, still well inside 2 GiB.
  - Keep `uint8` over a closed normalised taxonomy frozen in the design, with exact strings left to the verifier's `step_raw` re-derivation.
  - Drop the stored reason and state that the verifier re-derives it from parent state plus op.
- Whichever is chosen, state the overflow behaviour and that the example index comes from `stepped`, not the string.

## B1–B4 audit against the original review

- **B1, cap and bytes: resolved apart from B1-R.**
  - 28 ops confirmed; 144 × 83,244 = 11,987,136, below 12,000,000, so the attempt cap cannot bind.
  - Attempt records are 239,742,720 bytes; scores plus masks are 59,935,680 bytes, under 60 MB.
  - The float32 assumption holds: `PolicyScorer.log_probs` arithmetic stays float32 (`stack_policy.py:336-344`), and `0.0 + np.float32` is float32 under NumPy 2.4.4.
  - `depth`, `node` and `op` fit `uint8`; successors are deterministic from parent state plus `step_raw` at the default fuel limit.
  - The stop mechanism is sound as designed: an exception from the observer propagates through `beam_solve` without `search_terminated` firing, so it cannot appear as `wall_clock_cap`.
  - Typed-state bytes are honestly labelled as enforced, not guaranteed.
- **B2: resolved.** `gate_components` compares `Fraction` against 19/20 and 1/10 (`score_gap_profiles.py:276-285`). My probe reproduced the original 12 float disagreements, all at n = 20, and the exhaustive test covers n = 12–24.
- **B3: resolved.**
  - Hash order, minimum representative, first source row and even-is-selection match `DESIGN.md:39-47` (`score_gap_profiles.py:95-102`); I recomputed the literal test orderings independently.
  - The validation key matches production `validation_skeleton` (`stack_policy.py:394-398`), including the behaviour fallback.
  - ΣC = 0 gives 1 on both profiles (`score_gap_profiles.py:187`).
  - The band test alone casts to float64 and matches the #179 inclusive contract (`score_band_search.py:170-171`); −inf is excluded under a finite anchor and all-−inf falls back.
  - The lineage module walks recorded `parents`, full program and typed keys, checks score dtype and bits, and replays every suffix. It uses no last-occurrence or witness rank.
- **B4: resolved.** Prefix protection > 9/10 is exactly the unattainable-gain condition for every n in 12–24 (probe). `NO_HEADROOM_OVER_PREFIX32` locks tau as null; grid and conjunction are unchanged; `DESIGN.md:15-17` no longer claims a route kill.
- **Nonblocking items: all addressed in text.**
  - Citation: line 459 is the sort, in a 462-line file whose hash matches.
  - Memory limit: systemd `MemoryMax=4G`.
  - Timing: `wall_clock_cap` cells are counted separately.
  - Also covered: granularity, panel overlap, the width 1–64 prefix curve, and the exact-bits recorder.

## Reproduced evidence

- Scoped pytest command from this tree: 54 passed in 8.81 s (author saw 8.53 s; neither is an allocator measurement).
- `ruff check` and `ruff format --check` with `--no-cache`: clean on the four files.
- One labelled synthetic probe in `/dev/shm`, removed afterwards; its figures are the ones quoted above.

## Obligations the later runner and verifier still owe

None of these is done, and none blocks this design review.

- **Terminal precedence.** `choose_selection` checks support on the selection half only, and `confirm` reports confirmation support after tau is locked. The runner must check both halves first and write the order down: support (either half), then headroom, then selection.
- **Score validity.** `Snapshot` validates retained scores only. The NaN/+inf check at `DESIGN.md:152` covers every executable non-fit score, so it needs the full scored matrix.
- **Outputs.** `choose_selection` returns only the choice; retaining all eight profiles and the descriptive tables is the runner's job.
- **Confirmation headroom.** A confirmation half with prefix > 9/10 surfaces as `NO_CONFIRMATION_PROFILE` with gain failed. Report prefix protection beside it so it is not read as a band failure.
- **Whitelist wording.** `DESIGN.md:52` lists "program" as whitelisted, while the code and its trap test never read it. Reconcile towards the code.
- **Memory.** Unretained successors are not bounded by `max_state_atoms` before retention; a 4 GiB kill there is a no-result, as disclosed.
- **Not yet built.** Recorder, attempt reservation, atomic selection receipt, and the independent serialized-result verifier (attempt replay, ranking, verdicts, counters, digests). `SyntheticCapture` must not be the recorder.

## Scope

Once B1-R is fixed, a pass would accept only the amended design and synthetic foundations, and permit finishing the implementation for exact-code review. It would not authorize a data launch, would not transfer to a changed runner or result checker, and would support only a later preregistration, not a CPU-clock claim or a BEST move. Snapshot protection does not imply lineage survival in an executed band, as `DESIGN.md:198-203` states.

Not checked: the side file, corpus, checkpoints and issue thread were not opened, and I did not confirm whether stack-machine-quotation-growth (#172) is fixed on this tree.
