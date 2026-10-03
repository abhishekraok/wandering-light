Claude (Opus 5.5) wrote this — independent other-family review, high effort, read-only, of `git diff 2ef1113..d9101b7` for branch-and-check (#164) O1-A.

# Verdict: PASS

No landing-blocking findings. The audit reproduces exactly, the gate matches the preregistration, and the STATUS claim is no wider than the evidence. Five low-severity notes and a list of prerequisites for the future model follow.

## What I verified

- **Reproduction:** re-running `audit()` at `d9101b7` on the registered archive gives a result identical to the committed `archive_audit.json` in every key except `code_revision`. `o1_archive.py` is unchanged between `a2ef966` and `d9101b7`.
- **Source:** SHA256 is `e358fca1…ca78`, as registered.
- **Headline numbers:** all confirmed.

  | quantity | value |
  |---|---|
  | tasks / behaviours | 592 / 148 |
  | eligible tasks / behaviours | 584 / 146 |
  | candidate pairs | 36,050 |
  | kept / `on_witness` records | 466,241 / 1,960 |
  | genuinely closed `on_witness` records | 1,427 |
  | saved cuts not genuinely closed | 2,268 over 428 tasks |
  | shown-fit traces | 133 |
  | errors | 0 |

- **Training provenance (checked beyond what the audit does):** all 592 rows' `inputs`/`outputs` equal the live `stack_pbe_v4` train task at that `example_set`. All 592 witnesses equal `reports/short-segment-20260923/train_capped.jsonl`. The archive has one policy, one system and one corpus.
- **Product closure:** `closed_product` requires every example to be closed. No candidate state carries a mark, and none holds a `Sym` at any depth.
- **Repeated and terminal states:** 13 tasks repeat a product state, and latest-occurrence dedup handles them. The goal never occurs before the final step, and no start state equals the goal.
- **Saved cuts versus closure:** `closed_cuts` is "top-level combinator applied with no mark", which is not closure. The `remove_values_seen_in_second` example checks out: 3 saved cuts, 2 not closed, 0 of 4 flagged records closed.
- **False negatives:** the code never labels a kept record. In a 1% sample of unflagged kept records (4,722), none reaches a witness cut state, so the flags are not under-reporting.
- **Terminations:** the archive has no timeouts at all (`max_segments` 459, `fit` 133).
- **Gate versus preregistration:** `gate_passed = not errors and len(eligible) >= 100` at `o1_archive.py:159` covers each listed falsifier. The margin is 146 against 100.
- **Timing:** the preregistration comment is stamped 21:22:35 PDT, the code commit 21:25:29 and the result commit 21:30:15. The order is consistent; I cannot verify that no replay ran before the comment.
- **Tests and lint:** 12 tests pass and ruff is clean on the new files.
- **Docs:** STATUS, RESULTS_DETAILED and the report README say "recorded-suffix upper bounds, not optimal distance" consistently. STATUS carries state, RESULTS_DETAILED carries the row, with no duplication.

## Findings in this audit (all low, none blocking)

1. **Latest-occurrence is not quite the shortest recorded suffix when the path loops** — `experiments/branch_check/o1_archive.py:53-58`, `:105`.
   - Counterexample: `sign_histogram` set 3 (48 ops) has `states[35] == states[43]`. Candidate 34 is labelled 14, but splicing the loop gives 1 + 5 = 6. Candidates 41 and 42, inside the loop, are labelled 7 and 6.
   - Effect: the recorded order "34 farther than 41 and 42" is reversed or tied under the tighter bound from the same record. This is 2 of 36,050 pairs, in 1 task.
   - Minimal correction: label with the spliced distance (shortest path over state-equality), or drop pairs that straddle a repeated-state loop. This can be done in the trainer's candidate builder rather than here. The count and gate are unaffected.

2. **Provenance is self-attested** — `o1_archive.py:63`. The row's own `split` field and the live skeleton split are checked, but the shown examples and witness are not compared with the corpus. I checked independently and found 0 mismatches, so this is optional hardening only.

3. **`code_revision` has no dirty-tree marker, and the input is an absolute path to a gitignored checkpoint** — `o1_archive.py:147-149`. The SHA makes it checkable but only on this host. Optional: record `git status --porcelain` emptiness, and note whether the archive is on the Hub.

4. **Stale closure claims remain upstream.**
   - `wandering_light/stack_beam.py:607-609` says a cut "means `states[t + 1]` is closed".
   - `experiments/short_segment/archive_train.py:9-11` says the same of `witness_cuts`.
   - `goals/g1-handoff-20260925.md:19` tells the reader to rebuild witness positives from `witness_cuts`.
   - Out of scope for a no-solver-change branch, and the STATUS sentence does reach readers. A follow-up docstring fix and a one-line Superseded entry pointing at the handoff instruction would close it.

5. **One test is redundant** — `tests/test_o1_archive.py:53`. `test_timeout_does_not_supply_negatives` asserts the same row as the first test, and the archive has no timeout rows. Harmless.

## Prerequisites for a meaningful O1 comparison (future model, not bugs here)

- **Candidates are mostly states the search never ranks.** Of 5,967 candidates, 2,503 are combinator-cut states and 586 are the initial state. The remaining roughly 2,900 follow plain ops, which the segment search never stores as closed states. Only 8,377 of the 36,050 pairs lie within {initial, cut state}. Eligibility is unchanged under that restriction (584 tasks / 146 behaviours).
- **The reverse mismatch also holds.** 2,268 cut occurrences hold code, and the search will present those to a ranker trained only on genuinely closed states. Register which state set the ranker is trained and evaluated on, and report the cut-state subset separately.
- **Pairs are within a single path.** Every pair comes from one witness, but search compares states across different paths, most of them off-witness and unlabelled. A pair-ordering score cannot stand in for the search gate.
- **The independent unit is the behaviour or skeleton, not the task.** There are 148 behaviours with 4 example sets sharing one witness each, and per-task pair counts run from 0 to 378. Weight or cap per behaviour, and split by skeleton and lineage.
- **Goal-blind control.** Prefix position is an oracle, and state size or shape may track it. The goal-blind and shuffled-goal controls need identical state features, as the result note already says.
- **Single source.** The archive is one policy, one seed (`capped-s0-e3`). `remove_value` and `remove_values_seen_in_second` contribute nothing.

**PASS** — the evidence suffices for the narrow STATUS claim: the gate passed, saved cuts are not closure, labels are upper bounds, and nothing has been trained.
