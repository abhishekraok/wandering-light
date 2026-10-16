RESULT REVIEW PASS

The list-hash-memo (#185) evidence at HEAD `274336809a8d08d73da952590b6a90978c427097` is correct and honest, and the narrow negative can land with production and BEST unchanged. One archival gap must be fixed first: RESULT.md links `WRAPPER.log`, but that file is not in `reports/list-hash-memo-20261015/`.

**Source continuity**
- From ce7662d to HEAD, `wandering_light/`, `pyproject.toml` and `uv.lock` are unchanged. The runner and tests at HEAD are byte-identical to the control.
- **Control `d9645d7be1fba554fb0722a1e7300e7c7a5d7585`:** production is byte-identical to ce7662d.
- **Candidate `01d8254d9a0f3d9bf984678e2997e30d78de023e`:** it changes only `wandering_light/stack_machine.py` (the `_HashMemo` slot and `Lst.__hash__`). `CANDIDATE.patch.txt` equals that git diff.
- Every source pin in the provenance file equals the committed file at its head. The two arms' pins differ only at `stack_machine.py`, and both trees are clean at their heads.
- **Timeline (PDT):** heads committed 21:49 and 21:51, code review 21:53–21:58, pilot 22:11:17–22:14:04.
- DESIGN.md differs from its preregistered text only in the Status line.

**What I checked myself**
- **Selection:** I rebuilt the panel from the 7000 raw rows (171 skeletons), using the `list-hash-memo-v1` + NUL salt and taking the first 24. It reproduces `selection.json` and `shown.json` exactly. The shown file carries only `source_index`, `inputs` and `outputs`.
- **Cells:** all six completed, in the fixed order, with the correct pins and checkpoint hashes.
  - Runtime controls match: CPU, one thread, five thread env vars set to 1, `PYTHONHASHSEED` 0, float32, and the pilot unit's actual 12 GiB / no swap / 20 min limits.
  - Module and function origins sit in each arm's own tree. The control's `Lst` hash is the dataclass-generated one; the candidate's comes from its `stack_machine.py`.
  - Each log's JSON lines equal its cell's rows, and the logs contain nothing else.
  - Every cell has 5 shown fits and 19 budget terminals.
- **Code:** the runner keeps the fixed order, finishes all six cells before any analysis, times each child process from launch to return, uses budget=100000 executions with no deadline, and clears `run_body` at the start of each process.
  - Budget terminals reach at most 100004 executions, so the original overshoot within one product call is preserved.
- **Pairs:** zero non-clock field mismatches across 3×24 rows; all six cells are substantively identical. Exact `Fraction.from_float` ratios:

| label | ratio (exact) | ratio | ≤ 9/10 |
|---|---|---:|---|
| 17 | 882683784809/912579594823 | 0.967240326 | fail |
| 29 | 887554517141/908886725935 | 0.976529299 | fail |
| 43 | 910374054006/888436824749 | 1.024691941 | fail |

  These match `summary.json`, so the reading is `PARK_FIXED_CONSTRUCTION`.
- **Replay:** I replayed all 30 returned paths (5 programs × 6 cells) on the original shown pairs, using `step_raw` with default fuel and depth 0, `ops_for_basis('mark-28')` and `parse_stack`. These came from my own tree, whose `stack_machine.py` bytes equal the control pin. All 30 fit.
- **Rehash:** I hashed 154 evidence files before the replay and again after it; none changed. They cover raw outputs, the input payload, both source trees, the checkpoint, the original code review and the report copies.
- **Archive and docs:** the report's JSON files are byte-identical to the raw pilot files, and LAUNCH, PREFLIGHT, STATUS_INITIAL and the AUDIT files are identical to their parent copies. The wrapper log ends `exit 0`. The seconds, ratios, RSS figures, phase time (166.70 s) and times in RESULT.md, RESULTS_DETAILED and STATUS all match the raw data.
- **Foreign collector:** it shows in the 22:10:47 PREFLIGHT snapshot at 63.7% CPU. It is gone from the launcher's final snapshot at 22:11:17.20, which asserts no process above 5% CPU, and from both host snapshots. So the claim that it ended before the final prelaunch snapshot holds.

**Blocker (archival only):** copy `checkpoints/list-hash-memo-20261015/pilot-wrapper.log` byte-identically to `reports/list-hash-memo-20261015/WRAPPER.log` (sha256 `75e2573df19fd09e6b2381ded8dfeb159978e0e91d014f9f96e2fff2bea66d6f`).

**Limits**
- This checks the integrity of saved outputs. It does not reconstruct the whole search or observe wall time independently.
- The `ps` CPU figure is a lifetime average, and the host snapshots are point checks, not continuous proof the host was idle.
- One TRAIN panel and one policy; no hidden grading.
- Whole-process peak RSS does not isolate what the memo itself costs.
- The result is not a broad null on hash reuse.

**What can follow without another review:** faithful archival (the `WRAPPER.log` copy above, plus this review's script, receipt and note) and the docs-only change from pending to passed. Any executable, pin or control change needs a new review.

Files are in `/home/abhishekrao/repos/wandering-light/checkpoints/list-hash-memo-20261015/final-review/`:
- independent_review.py
- independent-receipt.json
- REVIEW.md
