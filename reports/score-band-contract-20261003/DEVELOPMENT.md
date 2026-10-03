# Synthetic development validation, before Gate A

The program-synthesis janthu (e-den, Codex) wrote this, 2026-10-03 PDT.

The actual amended independent high-effort Claude Opus review passed on frozen
`e8213dec2f4fe28f18b5e2269c5ead3f52ddcf5f`. Report and actual successful CLI
receipt are retained beside this file. N1–N4 were pinned in fixture version A.1
before implementation or fixture execution. No real task/corpus/checkpoint,
model creation/inference, held-out grading, threshold selection or BEST change
occurred. Production files are unchanged against `f9ae90b`.

## Implementation and checks

The first candidate source is `1709b89`; the fixed catalog is `fa64090`, and
independent development checks are `f08aa15`. Complete manifest runner source
is `e36b89f88da314b8d8a04950dbab033bbf9d6684`. The disabled grid compares the
candidate with imported ordinary production beam_solve and BeamObserver.
Independent checks reconstruct every real-machine attempt/refusal, charging,
legal masks and scores, then typed-state retention and seen changes without
calling candidate retention. All fixed inputs and scripts remain A.1.

Development commands use the shared installed environment without sync:
`UV_PROJECT_ENVIRONMENT=/home/abhishekrao/repos/wandering-light/.venv UV_NO_SYNC=1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run python -m pytest ... -q`.
Both candidate and wandering_light module origins were confirmed in the own
intended development worktree. The reference source SHA256 is
`dcf4558a6cfa11d8b8dde9aaa02ff21c9837ec16279a692f9b7e723c94ec616b`.

- Initial fixed development suite on `f08aa15`: **426 passed**, 10.70s.
- Candidate/runner plus existing equivalence, short-segment resource and debugger
  observer suites on `e36b89f`: **486 passed**, 21.87s.
- Full repository suite: **3324 passed,40 skipped,7 deselected,1 xfailed**, 8 existing
  multiprocessing-fork warnings,179.09s. It started from e36b89f; a state-only
  docs/STATUS merge776ec73 occurred during this foreground command, with all
  tested code unchanged. This is not a frozen comparative timing measurement.

The manually constructed A-safe case retained the same `mul dup add` first fit
and matched its expected six versus three expanded nodes. B retained the
registered solution loss, and C the registered changed first-fit program.
All324 disabled pairs matched imported production traces/results excluding
seconds. These constructions establish software feasibility only; no useful
real allocation/coverage/clock conclusion follows.

## Runner serialization failure and visible repair

Source inspection found that legal-mask bytes in attempt/scoring events were
not handled by the runner's strict-JSON hasher. A regression was committed at
`dd25e9e` BEFORE execution, keeping every fixture and expected outcome unchanged.
Actual regression output was **6 failed,6 passed** (5.16s): TypeError, bytes are
not JSON serializable. No declared gate attempt or manifest was created.
This was an implementation defect, not a failed fixture construction.

Repair `f05364c` serializes masks explicitly as typed bytes plus their exact hex
string. Regression cases cover A/B/C, an executable-NaN exception, all-inf support,
a representative disabled grid case and the root-i resource-body case. Post-repair
new/existing focused suites: **494 passed**,18.77s. Whole-repository Ruff lint,
format and diff whitespace checks passed. The full suite above predates this
small runner-only repair; no later full-suite claim is made.

Actual stdout/test logs are retained in the janthu's work directory:
score-band-development-tests.log, score-band-core-final-tests.log,
score-band-full-suite.log, score-band-runner-regression-before.log,
score-band-core-repaired-tests.log. No log's seconds support an allocator clock
claim. A dedicated declared complete Gate A has NOT executed.

## Review state and remaining permissions

Independent pre-gate Opus/high review runs from frozen clean **e36b89f**,
`wl-score-band-pre-gate-review-20261003`, launched15:47 PDT/expected15:57 PDT.
It reviews the original source and cannot authorize the later f05364c repair.
Actual verdict/usage remain pending. Retain its report and address all blockers;
then obtain a fresh exact-source PRE-GATE CODE PASS covering the byte repair
before any declared gate invocation. Neither a design PASS nor a PASS on an old
runner permits running the repaired source.

The runner reserves a new exclusive attempt, retains all455 frozen IDs and
complete expected/actual definitions, counters and typed retained order, links
actual review/source hashes, and publishes one atomic complete manifest. A
failed/incomplete/deadline invocation is NO_GATE_RESULT; no partial PASS.
After an actual complete PASS, separate independent code/result review remains
required before candidate/evidence landing. No real campaign is registered here.
