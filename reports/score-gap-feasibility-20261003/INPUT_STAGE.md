# Reserved shown-input stage — SG-INPUT1

The program-synthesis janthu (e-den, Codex) wrote this, 2026-10-03 PDT.

This is synthetic software evidence, not a panel measurement or data eligibility.
The accepted design has not yet received COMPLETE exact-code PASS. No actual
train/side/checkpoint bytes, real selection, model load, inference or attempt
reservation have occurred. All test files are freshly made dummy fixtures.

`experiments/score_gap_inputs.py` supplies the pre-model boundary. It requires
an active reservation and its canonical on-disk attempt receipt, rejects an
existing input-stage directory before reopening pins, validates the entire
ordered three-difference-seed pin specification before opening pinned files,
then hashes all source/config/manifest/weight bytes without loading a model.
This stage is not an executable trial entrypoint. A future reviewed fixed-path
runner must enforce exact review/source eligibility before reserving the one
real attempt; a generic reservation alone does not grant that eligibility.

TRAIN rows retain only behaviour, original skeleton, split and shown text pairs.
The side reader accesses only behaviour identity: its program/length/outcomes
are neither accessed nor retained. Selection uses the existing frozen D1 hash
order, original validation groups, first original-file occurrence and fixed48
groups with even/odd halves. Source indices count nonblank original records.
Original outputs never change to the side witness's outputs. Every input and
checkpoint hash is checked again after streaming projection/selection. A source
mutation invalidates even an otherwise insufficient-selection outcome.

The exclusive input directory contains:

- `inputs.json`: schema/kind, attempt size/hash, resolved input/checkpoint pins,
  every checked file's size/hash, population counts, selected order/half/group,
  original whitelisted shown metadata, and contiguous typed-frame references.
- `shown.bin`: one SG-BEAM1 frame per representative, containing two zero-score,
  empty-program nodes for the complete shown product and original target. The
  existing codec preserves Bool/Int/List/Sym/Mark types and exact scalar bits.
- `input-seal.json`: atomic no-replace schema/kind and the full sizes/hashes of
  both fsynced streams. Every stream and this seal charge the shared payload
  budget. `INSUFFICIENT_SELECTION` seals empty typed/selected streams plus
  population/provenance, without a model load or a profile threshold lock.

The companion `load_inputs` hashes both entire streams before typed decoding,
checks closed metadata/population/pin schemas, fixed48 order/parity/validation
exclusion and source bounds, consumes every contiguous typed frame, compares
its exact re-encoding with original shown text, and checks on-read hashes.
It returns a solver object containing only inputs and outputs; identity/half/
source metadata stays outside imported production beam. This reader proves
structural projection consistency, not independent selection from the original
full input population. The after-run verifier must reconstruct that selection
and every attempted-op/search/profile stream independently.

Thirty new tests cover plain/gzip dummy sources; hidden/witness field traps;
first-source representative pairs; duplicate side membership; reservation and
duplicate-stage ordering; malformed complete pin populations; changed train,
side or weight bytes before projection and during reading; insufficient
selection; global resource failure without a seal; hash-before-decode; resealed
metadata/frame/provenance mutations; explicit typed lists; and a manually typed
one-op first fit through imported production with a metadata-free solver object.
Combined195 focused synthetic tests passed in9.17s; full lint and formatting
passed, and14 actual diagnostic/primitive module origins were in the intended
development tree. Times are validation duration, not solver-performance claims.

Remaining: fixed144-cell CPU model/reference runner, source/review/module/
version/host/thread checks, full global provenance and final schemas, complete
independently authored serialized-output verifier, fresh COMPLETE code PASS,
ONE frozen real attempt, after-run independent verification and final review.
Production solver, machine, corpus and BEST remain unchanged.
