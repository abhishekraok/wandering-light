# Complete independent verification and final review

The program-synthesis janthu (e-den, Codex) wrote this, 2026-10-03 PDT.

The preserved completed Gate A output passed separate independent verification
from clean frozen `c63b72a541ad290d8f45c8582a9008d69605087f`: actual `exit 0`,
empty stderr and `INDEPENDENT_VERIFICATION_PASS_FINAL_REVIEW_PENDING`.
The historical verifier verdict remains unchanged in the retained output.
Fresh other-family high-effort final code/result review now passed on frozen
21d8b134; see final-review.md and its receipt. This permits landing the
software-QA evidence for score-band-retention (#179); no real allocation,
coverage, clock or BEST claim follows.

## Reconstruction and evidence

The verifier independently rebuilt all455 fixed IDs:396 complete search rows,
ten typed input/output tasks and49 observation/parameter/GC/arithmetic/API/source
controls. The390 nonexception search rows also matched separately imported
ordinary production beam_solve in every result and full event; the six fixed
invalid-score rows terminate before retention and have no production run in
the original gate. Every returned program was replayed against manually typed
shown examples, with no held-out data or model/weight/corpus access.

Reconstruction uses established step_raw/Value/guard/encoding primitives, its
own literal definitions and fixed scorer rules, typed-key identity and a
definition-form ordinary-C/band-prefix construction. It never imports or calls
candidate search/retention, fixture constructors, runner or development checkers.
Every parent/op attempt, first-refusing-example reason and partial charge,
budget/deadline boundary, legal mask, legal score, cumulative stable ordering,
guard/atom/seen eligibility, retained-only seen update, callback/parent order,
anchor, pruning/stop flag and terminal outcome is reconstructed. Complete
event/ranked/old_seen digests and every saved layer/result/expected value match.
Unread-tail eligibility remains absent. The API's all-illegal row is separately
scoped; it is not claimed reachable in a real expanded mark-28 parent.

Mandatory A saving, B solution loss and C changed first-fit all remain. Their
pair numbers are recorded in RESULTS_DETAILED and the retained verification
result JSON. No gate retry, favourable subset or changed fixture/threshold/cap
occurred. Original source826a8bb, attempt/final/log and pre-gate receipt stay
untouched. Actual frozen source and all five intended verifier primitive paths
and hashes were checked; verifier HEAD, cleanliness, file bytes and primitive
bytes were checked again after replay.

Original GC/exception controls record snapshots rather than full side-effect
traces. The verifier checks all saved snapshots/metadata against the fixed
registered assertions; it never re-executes the candidate. Source correctness
of those side effects was separately assessed in the final code review. No missing exception
expansion counter is invented: the original replay expectation records it null.

## Validation and command

The initial new unit run found one wrong test expectation: T-same's early add
fit costs10 executions/fuel, not12 (OPS index12 includes refusals). The preserved
raw manifest and independently imported production confirm10; only the added
test expectation changed. No original gate definition or output was modified.
Final verifier tests190 passed6.99s. Clean frozen-c63b72a combined verifier,
candidate, runner and existing equivalence/resource/observer suites passed
762 tests21.69s. Whole Ruff lint/format/diff checks passed. These seconds are
development validation, not an allocator clock comparison; the earlier full
repository suite predates the latest checker/verifier work.

Exact actual verification command, from the separate frozen verifier tree:

```sh
PYTHONPATH=/home/abhishekrao/eden/janthus/program-synthesis/work/trees/score-band-independent-verifier-20261003 \
UV_PROJECT_ENVIRONMENT=/home/abhishekrao/repos/wandering-light/.venv UV_NO_SYNC=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run python reports/score-band-contract-20261003/verify_gate.py \
  --run-root /home/abhishekrao/repos/wandering-light/checkpoints/score-band-contract-20261003 \
  --frozen-tree /home/abhishekrao/eden/janthus/program-synthesis/work/trees/score-band-amended-pre-gate-review-20261003 \
  --phase complete
```

Raw stdout/result, empty stderr, actual exit log and receipt are under main
checkpoints/score-band-contract-20261003/independent-verification. Portable
result/receipt copies are beside this report. The original partial foundation
receipt remains historical; it is not the full verifier verdict. Final review
explicitly assessed changes from original826a8bb through checker219a655,
foundation05029d4/d80446c and complete verifierc63b72a. No earlier PASS transfers
to these changed checking paths.
