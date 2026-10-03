# Independent result and landing review

Claude Opus, high effort, on exact `90e66ee2b9be57fa439f991012c7f5552c1c0518`. Personal-plan call with both paid API keys unset. Unit `wl-o1-inventory-result-review-20261003` completed 2026-10-03 11:19:36 PDT; log ends `exit 0`. The following is the full returned review.

**Verdict: PASS** for fast-forwarding the full branch at `90e66ee` to `main`. No blocking issues. This is a pass on the measured outcome and its verification only; it claims no headline or coverage change, does not close branch-and-check (#164), and approves no owner-gated work.

## What I checked

- **Hashes and revision.** Raw output is `b42f26d3…5ff2` and the archive is `e358fca1…ca78`, both matching `inventory_summary.json`. The output's `code_revision` is `c664a7c`, and the run tree is at that commit and clean.
- **No source drift.** `git diff c664a7c HEAD` touches only STATUS, RESULTS_DETAILED and the report directory; no file under `experiments/`, `tests/` or `wandering_light/` changed.
- **Exit and kill evidence.** The log ends `exit 0`. The user journal shows start 11:00:10 PDT and `Consumed 31.269s CPU time` at 11:00:45 PDT, with no runtime or memory kill line.
- **Verifier.** The suggested command printed PASS, and its output is byte-identical to the committed `inventory_summary.json`.
- **Tests and lint.** The 49 synthetic/replay/beam tests pass; ruff check and format are clean on the builder, test file and `verify.py`.
- **Fast-forward.** The local `origin/main` ref (`a03c9e4`) is an ancestor of `90e66ee`. I did not fetch, so this is relative to that ref.

## Independent probe of the raw output

`verify.py` trusts the recorded catalogue, the recorded exclusion flags and the recorded failed trials (`verify.py:96-116`). A zero-pair claim depends on there being no false negatives, so I replayed those myself with my own code, not the builder's helpers. Everything agreed.

- **Census.** 52 tasks: 33 complete, 9 candidates/task_calls, 8 matrix/task_calls, 1 roots/task_calls, 1 matrix/run_calls. The run-scoped stop is the last task (`abs_diff_pair`), and there are no errors.
- **Counters.** Per-task calls sum to exactly 2,000,000 and fuel to 10,817,925. All 18 task-capped tasks sit at exactly 50,000 calls. No fuel, atom, depth or deadline censoring occurred anywhere.
- **Catalogue.** I recomputed it for all 52 tasks with an independent shortest-path/lexicographic-tie routine; it matches in every task. None was truncated or over length.
- **Eligibility.** All 1,024 candidates in complete tasks were replayed. Product, cut membership, witness membership and each exclusion flag match: 494 code-bearing, 73 expanded, 558 excluded in total, 466 eligible.
- **Every trial.** All 4,512 trials in complete matrices were replayed: 3,705 ordinary refusals, 796 that ran but missed the goal, 11 successes. Each success flag and label matches.
- **Pairs.** Ten labelled candidates sit in five complete tasks, and no two share both segment and prefix length. Zero pairs therefore holds whatever the labels are.
- **Partial positive.** `first_maximum_index` has one candidate with label 69; the task was censored later, in candidate-prefix replay.

## Overclaim check

- README lines 3-7 and 30-37 and the ledger paragraph scope the negative to the exact sample, catalogue and caps. They treat unlabelled candidates as censored, carry the target-selection bias, and make no distance, goal-dependence, solver or clock claim.
- The timing caveat is at README lines 16-18.
- `docs/RESULTS.md` has no O1-B row, which is correct for a non-headline result.
- The STATUS "Current work" text is accurate, and that file is already identical on `main`.

## Non-blocking limitations

1. **The 33 complete tasks are cost-selected, not a random subset.** The per-task call cap removed 18 of 52. Censored tasks have longer witnesses (median 65 ops against 48) and larger catalogues. The largest complete task used 49,745 calls. One sentence in the README or ledger saying so would help.
2. **The registered breakdown is not in the committed prose.** `preregistration.md:60-61` asks for raw, code-bearing, expanded and eligible counts separately; they exist only in the raw output. For complete tasks:

   | Count | Value |
   |---|---|
   | Selected | 1,024 |
   | Code-bearing | 494 |
   | Expanded | 73 |
   | Eligible | 466 |
   | Eligible and on-witness | 2 |

   Eighty-two percent of trials are ordinary refusals.
3. **README lines 23-25 slightly overstate the verifier.** "Recomputed catalogue minima" is computed from recorded trial flags; the verifier does not rebuild the catalogue or re-check exclusions and failures. My probe closed that gap. Either soften the wording or note this review.
4. **STATUS wording will be stale after landing.** It says "on the review branch pending independent result review", and line 11 still reads "Last updated: 2026-10-02". Update both in the landing follow-up.
5. **README line 54 says the CLI "rejects" a source hash mismatch.** It actually records an error and writes a failed output (`o1_inventory.py:510-511, 550-551`).
6. **Not checked.** I did not open the issue comments linked from the ledger and STATUS, since that needs account access.
