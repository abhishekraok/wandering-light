# Independent verifier foundation; full result verdict pending

The program-synthesis janthu (e-den, Codex) wrote this, 2026-10-03 PDT.

Separate verifier `verify_gate.py` has independently checked the preserved
completed output's terminal, provenance, complete ordered catalog and manual
inputs. It explicitly reports `PENDING_SEARCH_REPLAY_AND_CONTROLS` and cannot
issue a full independent Gate A PASS. Candidate/evidence landing and fresh
high-effort final code/result review remain pending.

## What this phase checks

Failed exit, failure-file or incomplete terminal headers are rejected before
the result payload is consulted. Strict JSON rejects duplicate keys, unnamed
nonfinite values and Boolean/integer aliases. Original final/attempt/log/review
receipt hashes are pinned in the verifier; the actual successful review log,
explicit first verdict line, source revision and metadata are checked. All15
frozen source files, the original9 module origins and the actual reference
source hash are recomputed, with clean frozen source `826a8bb` preserved.

The verifier has its own literal catalog implementation rather than importing
experiment fixture constructors: **455 ordered unique IDs**, **396 search
definitions**, each complete Task/scorer/settings definition, **ten** explicit
typed input/output tasks and **five** nonempty witnesses replayed directly via
production `step_raw`. Task text, separately written typed expectations, both
saved expected/actual copies, witness metadata and typed encoding are checked.
Neither candidate search, runner, fixture constructors nor development
checkers are imported or called. No attempt/final gate file is written.

## Actual invocation and origin repair

Foundation source05029d4 initially accepted byte-identical low-level primitives
resolving through main's shared editable environment when invoked as a file.
The first partial receipt is retained in the janthu work directory. Its numbers
were not wrong, but it did not enforce the intended verifier-tree origin.
Visible repair `d80446c8cbda597c09291527b66a93bdcf3f96c8` requires exact intended
paths as well as bytes. The corrected foreground invocation completed actual
exit0, with empty stderr and both primitive origins in this development tree:

```sh
PYTHONPATH=/home/abhishekrao/eden/janthus/program-synthesis/work/trees/score-band-contract-20261003 \
UV_PROJECT_ENVIRONMENT=/home/abhishekrao/repos/wandering-light/.venv UV_NO_SYNC=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run python reports/score-band-contract-20261003/verify_gate.py \
  --run-root /home/abhishekrao/repos/wandering-light/checkpoints/score-band-contract-20261003 \
  --frozen-tree /home/abhishekrao/eden/janthus/program-synthesis/work/trees/score-band-amended-pre-gate-review-20261003
```

`verifier-foundation-result.json` and `verifier-foundation-receipt.json` retain
the scoped outcome, original pins, verifier source/hash and intended primitive
origins. Later08867b4 changes only the portability of the origin regression;
verifier code is byte-identical to d80446c. This is a verification amendment,
not a retry of the declared synthetic gate or a change to any fixture.

## Validation and next step

Foundation tests covered synthetic failure/incomplete terminals, ID population
changes, substituted definitions/settings, all saved input copies, typed
aliases, malformed JSON, file hashes/limits, actual review failures/quoted
PASS, source paths/hashes and dirty/wrong frozen revisions. Initial129 tests
passed on05029d4, followed by130 with the origin regression. Clean08867b4
combined verifier/candidate/runner/production-equivalence/resource/observer
validation passed **702 tests**,20.46s. Whole lint/format/diff checks passed.
No new full-suite claim or comparative timing measurement is made.

Next independently reconstruct every machine attempt/refusal, legal mask,
scripted score, cumulative stable rank, guard/atom/seen decision, retained
callback and terminal counter. Saved event/ranked/old_seen hashes must match
independently generated full traces, including early returns and budget
boundaries. Compare imported production reference traces separately and
verify the remaining observation, parameter/GC, arithmetic/API and source
controls. Retain A's constructed saving and B/C's adverse outcomes. Only after
all these checks and fresh final review can the completed evidence land.

This phase establishes input/provenance validity only. No model, corpus,
held-out task, checkpoint weights, real threshold/campaign or BEST change
occurred. Original gate source/output remain frozen; no gate retry follows.
