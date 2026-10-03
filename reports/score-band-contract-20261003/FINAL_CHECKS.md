# Post-gate checker repairs; independent result verification pending

The program-synthesis janthu (e-den, Codex) wrote this, 2026-10-03 PDT.

Development checker revision `219a655242d1e50f5f9dfe669571db821675876d`
repairs the nonblocking gaps in the actual amended pre-gate review. It does
not replace the measured source, authorize another gate invocation, or
independently verify the completed result. Final review must explicitly cover
the differences from frozen gate source `826a8bbe3c0059032ca6c190a766cbc09226efc4`.

## Changes and evidence

`verify_layers` now requires the complete ordered band-callback depth sequence
to equal the retained-event depth sequence, with no duplicates. This catches
omitted final callbacks even when no subsequent layer expands that beam.
Regressions also cover omitted entire streams, duplicated/reordered callbacks,
wrong depths and duplicated/reordered retained events.

The runner now asserts exactly one scored event and zero retained callbacks
for the fixed executable-score ValueError cases, and checks the arithmetic
units' retained products/programs/scores before marking them PASS.

All ten manually registered input/output products and witness programs have
separate typed literal expectations in `manual_typed_expectation`. Both
examples of the Boolean/Int list task are checked. Parsing cannot supply its
own expected value; witness replay alone cannot establish the registered
definition. Tests substitute inputs, outputs and witnesses, including a
different pair of input/output examples whose original witness still fits.
The five nonempty registered witnesses are replayed by the existing low-level
production replay primitive.

Before repair, the targeted corruption regressions produced **17 failed,
29 passed,470 deselected**,8.29s: observations escaped detection or produced
the wrong exception. An initial dirty-tree development run gave **515 passed,
1 failed**,11.43s because the existing missing-receipt test correctly encountered
the runner's clean-source guard first. No attempt/output directory was created.
After committing the repair, the complete focused command on clean `219a655`
passed **572 tests**,19.03s:

```sh
UV_PROJECT_ENVIRONMENT=/home/abhishekrao/repos/wandering-light/.venv UV_NO_SYNC=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run python -m pytest tests/test_score_band_search.py tests/test_score_band_gate.py \
tests/test_stack_beam_equivalence.py tests/test_short_segment_search.py \
tests/test_stack_debugger_trace.py -q
```

Whole-repository Ruff lint and formatting checks, plus whitespace checks,
passed. No new full-suite claim or comparative timing claim is made. Actual
logs remain in the janthu work directory as `score-band-final-gap-{regression-before,
tests,focused}.log`.

## Preserved source and output

Candidate search, fixture definitions, machine/OPS/parser, ordinary production
search and documented solver source have an empty diff from frozen `826a8bb`.
Recomputed SHA256 values for the original final manifest, attempt, gate log
and successful review receipt match `gate-a-summary.json` exactly. The frozen
execution tree was not edited and the single declared gate was not rerun.

The completed raw output still awaits a separate verifier that independently
reconstructs definitions, machine attempts, scores, retention and counters;
the repaired development checker and its tests do not substitute for that.
Original pre-gate PASS does not transfer to this changed checker source.
Final independent high-effort code/result review and evidence landing remain
pending. This is constructed software QA only; no real allocation, coverage,
clock, corpus/model/cohort or BEST claim follows.
