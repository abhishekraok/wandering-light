VERDICT: FINAL CODE/RESULT PASS

Reviewed `21d8b134a501876a39b70bf3aa415353214d5460` for score-band-retention (#179), read-only. I found no landing-blocking defect in the candidate, checker, verifier or raw evidence, and the claims and limits are honest. The review tree, frozen gate source `826a8bb` and frozen verifier tree `c63b72a` were git-clean before and after every command; the two older trees hold only ignored caches that predate this review.

## What I reproduced

- **Tests.** The permitted pytest command from this tree gave `762 passed in 25.10s`. Ruff lint and format pass on the eight score-band files (run with `--no-cache`).
- **Verifier rerun.** `verify_gate.py --phase complete` from this tree gave exit 0 with empty stderr.
  - The output matches the retained `independent-verification/result.json` on every key except `verifier_source_revision` and the five primitive paths, which now point at this tree with identical hashes.
  - Counts: 455 IDs, 396 search rows, 390 production comparisons, 10 typed tasks, 5 witnesses, 49 controls.
  - Verifier hash is `44525b58…458c` in both trees; `git diff c63b72a HEAD` touches only five report/evidence files.
- **Pinned hashes.** `final.json` is `a9d9fffb…a70a` (37,674,832 bytes) before and after; attempt `b3093334…`, gate log `a21ac993…`, verification result `7e4d1a41…`, receipt `d93bea7a…`, and stderr is the empty-file hash. The portable copies in the tree are byte-identical to the checkpoint ones.
- **Raw manifest probe.** All 455 rows are PASS, and all nine module origins sit in the frozen `826a8bb` tree. All 324 disabled rows have candidate result and event digest equal to the reference. The mandatory pairs match RESULTS_DETAILED:

  | Case | Production reference | Finite band |
  |---|---|---|
  | A-safe | `mul dup add`, 6 expansions, 78 executions, 80 fuel | `mul dup add`, 3 expansions, 36 executions, 36 fuel |
  | B-only-solution-cut | `mul dup add`, fit | no program, `max_depth`, 43 executions |
  | C-later-first-fit | `dup add mul`, 80 executions | `mul dup add`, 36 executions |

- **Early-add expectation.** `fit/enumeration` records 10 executions in the raw manifest. A hand count on `2 2` agrees: nine ops succeed, three refuse at zero cost, then `add`. The corrected test value of 10 is right.

## Candidate against production

`experiments/score_band_search.py` has an empty diff from `826a8bb`, and the five production primitives hash to the pinned values. Reading `_band_beam_solve` beside `stack_beam._beam_solve` (lines 361–483):

- The expansion loop, budget and deadline checks, charging, fit-before-score, legal mask, single `log_probs` call and sort key are identical.
- `_validate_band` runs before the GC toggle and any parsing, and the collector is restored in `finally`.
- The NaN/+inf check (lines 301–305) covers every successor after `layer_scored` and before `sorted`; illegal positions are never read.
- In `_retain`, the anchor is set only after the guard, size and seen filters. `seen.add` happens only on retention.
- The width/band/exhausted stop precedence matches DESIGN, and unread rows get no verdict.
- An all `-inf` anchor falls through to ordinary retention.
- `BeamResult.pruned` is width-only, `band_pruned` is separate, and nothing is wired into widening.

**Synthetic code probe (labelled, not result verification).** I ran 1,500 random scripted cases on the manual tasks through the candidate, the verifier's definition-form oracle and imported production. The cases varied width, horizon, guard, atom cap, budget and unit, band (including None), `-inf` and tie scores, NaN at illegal positions, and injected NaN/+inf.

- Candidate and oracle agreed on every result field, the full event stream, every layer field including ranked and old-seen digests, and scorer calls.
- Observed and unobserved runs agreed.
- Production equalled the oracle with the band off in every non-injected case, and the candidate with band None equalled production's trace (199 cases).
- Coverage: 312 band-pruned, 395 width-pruned, 71 all-`-inf` fallback layers, 27 invalid-score ValueErrors.

## Checker changes at `219a655`

The three gaps from the amended pre-gate review are closed:

- `verify_layers` now requires the band-callback depth sequence to equal the retained-event sequence with no duplicates. This catches a dropped final unexpanded layer.
- ValueError cases assert exactly one scored event and zero layers; arithmetic units assert `layer.retained` against a literal expectation.
- `manual_typed_expectation` is a separate literal table for all ten tasks, including both Boolean/Int examples of T-typed. A test substitutes a different input/output pair for which the witness still fits and confirms it is rejected.

The mutation tests corrupt observations, patch `_retain`, or substitute tasks, so they do not just mirror the implementation.

## Verifier audit

- **Independence.** It imports only the five `wandering_light` primitives. It has its own literal 455-ID catalog, task text, typed products and scorer formula. Attempts are rebuilt with `step_raw`, including first-refusal partial charges. Identity uses `value_key`. Retention is built as ordinary C first, then the band prefix. The parent expansion is charged before the op-0 cap check.
- **Production comparison.** Production `beam_solve` runs with its own `BeamObserver` and is compared separately; the candidate is never called.
- **Scope it states honestly.** GC, parameter, and exception rows are checked as saved snapshots only. Exception rows keep `expansions: null`.
- **Refusal order.** `failure.json`, then the pinned log and `exit 0`, then the pinned final, terminal header, attempt, receipt and provenance. Primitive paths must be in the verifier's own tree and are rehashed after replay.
- **Mutation probe.** I applied 63 in-memory mutations of real manifest rows and four malformed-JSON strings; all were rejected with the expected label. They covered counters, programs, flags, bool/int swaps, layer fields, digests, definitions, typed inputs, GC and exception rows, arithmetic and API units, terminal headers, duplicate keys and bare NaN/Infinity.

## Nonblocking notes

1. **STATUS wording.** `docs/STATUS.md:122` still carries the bold "**The declared complete Gate A has not run:**" mid-paragraph, and the same paragraph later says it ran. Rewrite it to current state in the landing commit, which also has to flip "final review pending" there, in RESULTS_DETAILED, GATE_A.md and `gate-a-summary.json`.
2. **Stale checkpoint summary.** `checkpoints/…/gate-a-summary.json` (`d472c158…`) still says independent verification is PENDING; the tree copy (`b5bb01e0…`) is the current one.
3. **Verifier leniency.**
   - Search rows are read by key, so an extra unknown key would pass; control rows are compared whole.
   - `same_json` accepts `36.0` for an integer counter.
   - Neither matters while the final is hash-pinned, and the real rows carry exactly the checked keys.
4. **Unobservable ordering.** That the invalid-score check precedes sorting and seen mutation rests on reading the source plus "one scored event, no retained event".
5. **Saved snapshots.** GC and exception side effects in the raw output are not replayed; I checked that source by reading and through the passing development tests.

## Scope

This pass permits landing the software-QA branch under the repository rules. A's saving and B/C's loss and divergence hold by construction and are not evidence of useful real allocation, coverage or clock. No threshold, panel, model, campaign or `BEST` change is supported; the train-only feasibility read and controlled comparison remain separately preregistered later work.

I did not call `run_gate` or the gate CLI, create an attempt or final, load models or corpora, use the network or other agents, or edit, commit or publish anything. The fuzz and mutation scripts and the reproduced verifier output lived under `/dev/shm` and were deleted.
