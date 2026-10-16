RESULT REVIEW PASS

**Revisions.** I reviewed the frozen tree at `23a66142b1ca27d5eb50ad0ff6816a5f83655507` (clean). The run executed `0212640d68e39e5d0ddf7024ad0efc55264d8809`, from the tree `gumbel-retention-review2-20261015`, which is still at that HEAD and clean. `git diff 0212640..23a6614` touches only `docs/RESULTS_DETAILED.md`, `docs/STATUS.md` and `reports/gumbel-retention-screen-20261015/*`. Nothing changed under `experiments/`, `wandering_light/` or `tests/`. All 10 source files pinned in provenance have the same SHA256 at 0212640, in the executed tree and at 23a6614, so the code review carries over.

**Scope.** I reviewed code and completed results for gumbel-retention-screen (#183): the baseline feasibility phase and the registered six-cell comparison. Inputs were TRAIN only. I opened no eval tasks or world-v3 tasks, ran no search, inference or training, and made no paid or model calls. I modified no source and made no commits. My artifacts are only in `checkpoints/gumbel-retention-screen-20261015/final-review/`: `independent_check.py`, `.json`, `.err`, `REVIEW-NOTE.txt` and `pytest-tmp/`.

## Code review
- **The fork matches production.** `experiments/gumbel_retention.py:_search` is line-for-line the same as `stack_beam._beam_solve` except for the retention key:
  - With `rng=None`, it sorts the native `scores` list, stable and descending. That is the same order as production's lambda sort, including ties.
  - With an RNG, it adds one float64 Gumbel draw to each executable non-fit successor, after the fit check and before the guard/seen/width filters.
  - Retained nodes keep the native unperturbed score, so noise never accumulates along a path.
  - A nonfinite score raises an error.
- **Run cells match the design.** Each cell (`gumbel_retention_run.py:run_cell`) sets 1 numerical thread and 1 interop thread and refuses an environment that differs. It clears the `run_body` memo once and reads only `{source_index, inputs, outputs}`. The candidate builds its PCG64 `SeedSequence([seed, source_index])` after `t0`, so RNG work is inside the solve clock. The deadline is `t0 + 5.0`.
- **No grading between timed cells.** The `run_phase` loop only launches fresh subprocesses and reads their JSON. `summarize` (grade/check) runs after `host-after`, and the file mtimes agree: last cell 19:33:40, host-after 19:33:40.7, verification 19:36:18.
- **Gate code** compares exact Fractions against 11/10.
- **Tests:** the focused synthetic suite passed, 41/41, under the main `.venv`, frozen-tree `PYTHONPATH`, no bytecode, with a scratch basetemp. This is software evidence only.
- I found no correctness defect.

## Independent result checks
My own script ran under this unit's actual cgroup (`wl-gumbel-final-review-20261015.service`, MemoryMax=12884901888, MemorySwapMax=0), exit 0, 157 s. It does not use the runner's `select_panel`, `summarize` or gate helpers.
1. **Panel selection.** I parsed the original train gzip (SHA `134507…683b`, 7000 rows, 171 skeleton groups) as raw JSON. Taking the first row per skeleton and ranking by SHA256 of `gumbel-retention-v1\0`+skeleton gives exactly the recorded 24 source indices. There were no digest collisions. Skeletons, behaviours and `shown_sha256` all match. `shown.json` is byte-for-byte the train rows' shown pairs with only the allowed keys. Baseline and comparison selections are identical.
2. **Cells.** All 9 cells are complete, carry the right arm and seed, and have rows in panel order. A row has a program exactly when its termination is `fit`. Threads and interop are 1/1. Every module origin is in the executed tree. Host is UltraDesk.
3. **Shown replay and hidden grading.** I replayed every returned program with `run_program`, `parse_stack` and `value_key` on the shown pairs (all fit), then on `held_out_*` from the raw rows. Repository `grade()` agrees on all 17 distinct programs (0 mismatches).
4. **Verification.** I ran the registered `step3_verify_solves.check` (10k draws per mode, seed 0, dense, min_tested 500) on all 17 distinct (behaviour, program) pairs.
   - Verdicts: 6 equal a catalogue witness, 2 are draw-tested genuine (`maximum` and `remove_values_seen_in_second`, 0 failures in every mode, at least 9079 tested), 9 are shortcuts, 0 inconclusive.
   - My outputs equal the saved `verification.json` exactly for the baseline phase (12 programs) and the comparison phase (17 programs), tallies and counterexamples included.
5. **Verified sets.** Verified means hidden-graded AND (genuine OR witness-equal).
   - Every control cell: 12 shown fits, 10 hidden-graded, 8 verified (6 witness-equal + 2 draw-tested).
   - Candidate cells: 11 shown fits; 8, 8 and 9 hidden-graded; 7 verified each (5 + 2).
   - All four shortcut behaviours appear in every cell.
   - Sets, counts, `fsum` clocks, executions, fuel and terminations all equal the saved summaries.
6. **The lost behaviour.** `running_minima` (source 2400) ends in a `fit` at depth 36 in about 2.9 s in every control. In every candidate it ends at `max_depth` 48 in about 3.5 s. The loss comes from the search, not from the 5 s cap.
7. **Gate arithmetic.** Fraction of the saved binary floats:

   | Seed | Control verified | Candidate verified | Ratio |
   |---|---:|---:|---|
   | 17 | 8 | 7 | 145230657167/144518407022 |
   | 29 | 8 | 7 | 964861077917/940295079327 |
   | 43 | 8 | 7 | 1854102428414/1890114916035 |

   - Median net gain is −1, so `median ≥ 2` fails.
   - The minimum gain is −1, so `every gain ≥ 0` fails.
   - The median ratio is 145230657167/144518407022 ≈ 1.004928, so `≤ 11/10` passes.
   - The conjunction fails: NEGATIVE_TRAIN_SCREEN. Lost = {running_minima} and new = ∅ in every pair.
8. **Copies match the originals.** All 10 `RAW/*.json` files are byte-identical to the originals in `comparison/`. `BASELINE_SELECTION.json` matches `baseline/selection.json`. Every key of `COMPARISON.json` and `BASELINE.json` equals the original `summary.json`.
9. **Final pins, checked after inspection.** All 21 `COMPARISON_AUDIT.json` and 15 `BASELINE_AUDIT.json` file pins match. Checkpoint file hashes match in both phases. Provenance head is 0212640 with empty `git_status`. Both manifests hash `26fd…518b`.

## Run validity
- **Limits.** Both units' initial status records show MemoryMax=12884901888, MemorySwapMax=0, and RuntimeMaxUSec of 20 min (baseline) and 40 min (comparison). Both wrapper logs end with a complete summary and `exit 0`.
- **Order.** The registered order was B17, G17, G29, B29, B43, G43, each a fresh process, timed on the single host UltraDesk. All three controls were timed fresh. The historical baseline clocks are not used in the gate.
- **Failures charged.** All 24 attempts count in each clock total. The worst deadline overrun is 5.136 s, consistent with the documented production semantics.
- **Registration.** The registering comment `issuecomment-6206493716` was created 2026-10-15 19:12:46 PDT and never edited. That is after the baseline ended (19:07) and before the comparison launched (19:27:23). It carries the unchanged design and amendment.
- **Host evidence is point checks only.** The before/after snapshots show only the runner's own python. The GPU was idle (P8, 0% utilisation, about 3.7 GiB resident from another process). Cell load averages were 0.25–1.09. This does not prove continuous exclusive use of the host. The clock reading is only "no runtime penalty beyond 1.10×"; it supports no speedup claim. Baseline-phase clocks had recorded external CLI activity and remain separate.

## The uncapped parent audits
The parent baseline and comparison post-run audits ran without the required memory scope, so they must not be described as capped. The measured phases did run capped, including their in-run verification, and their initial unit properties are saved. Those audits only re-checked saved outputs. Every correctness-relevant check they made, I redid under an actual 12 GiB/no-swap cgroup, with identical results. The omission leaves no concrete result blocker. It is correctly documented as a deviation, and no telemetry is missing for the measured phases.

## What I did not reproduce
- I did not re-execute either search or regenerate candidate Gumbel draws. Whether the timed searches behaved as recorded rests on the reviewed code, saved per-row outputs and provenance pins, not on a replay.
- Whole-search equivalence of noise-disabled mode with production is supported by code inspection and the synthetic tests only.
- I did not re-derive the systemd properties; I read the saved initial status files.

**Scope of the claim.** This is one fixed policy (staged-s0-e3), 24 TRAIN skeletons, library off, CPU with one thread, width 64, depth 48, 5 s caps. The seeds vary search noise and timing, not training. The result parks this fixed fresh-layerwise heuristic. It shows nothing about held-out behaviours, training-seed variation, deployed BEST or other diversity methods, and it is not a universal null. No BEST change and no tuning follow, and the reports say so.

## Ready to land
Yes. The code and this bounded negative result are ready to land under repository rules, and this is the passing other-family high-effort review. One required docs-only touch first, which needs no re-review: these lines still say final review is pending or in progress, and should be changed to cite this review:
- `RESULT.md`: "Final independent result review is pending"
- `BASELINE.md`: "has not yet received final independent result review"
- the two `RESULTS_DETAILED.md` #183 sections: "final independent result review pending"
- `STATUS.md`: "result audit/review in progress … provisional"

Then merge `origin/main` (no rebase) and push `HEAD:main`.
