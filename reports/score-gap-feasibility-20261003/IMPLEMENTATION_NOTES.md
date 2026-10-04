# Pre-data implementation pins — storage version SG-CELL1

The program-synthesis janthu (e-den, Codex) wrote this, 2026-10-03 PDT.
Frozen7e82a74 Amendment B review passed DESIGN/synthetic foundations only.
Its full report/receipt are preserved. These implementation definitions are
fixed before data and must be included in the eventual COMPLETE exact-code
review. No task-file selection, model/weight access or attempt has begun.
They change no panel, grid, conjunction, solver or resource limits.

1. Any NaN/+inf cumulative score at an executable non-fit successor is a
   RUN-WIDE provenance failure, not an unlabeled cell retained in denominators.
   Check the whole legal matrix BEFORE retention, including future guarded,
   duplicate and unread rows. Illegal entries are ignored. Preserve production
   scalar accumulation and explicit -inf. Dedicated DiagnosticProvenanceError
   stops the run; an incomplete/failure terminal cannot become a profile pass.
2. Refusal classification comes from exact independently re-derived strings:
   None = success; exact `out of fuel` or suffix `: out of fuel` = fuel;
   exact `quotation nested too deep` or corresponding colon-prefixed suffix
   = depth; all others ordinary. Unknown-op names quoting those phrases are
   ordinary. stepped, not strings, identifies the first refusing example.
3. The in-run replay within global2700s is returned programs and their known
   retained suffixes on all shown examples only. FULL independent attempted-op
   replay and ranking/state/result verification are AFTER the sealed run,
   outside that deadline, without model loading. This does not authorize retry.
4. Per-cell attempts.bin is fixed19-byte <BBBQQ. events.jsonl records cell
   identity, every start/scored/retained/terminal/result callback in order;
   started contains attempt_offset in that file, terminal contains per-last-
   layer and whole-cell attempt counts. Check exact depth/node/op lexicographic
   prefix with node below the ACTUAL source beam size. Cell seal pins all four
   streams by size and SHA. Identity/search/input/provenance requirements still
   belong to the future runner and full verifier, not generic codec metadata.
5. Budget/wall-clock stops have a started partial layer, possibly zero attempts,
   with no scored/retained event. The terminal copies ACTUAL BeamResult
   expansions and all counters; it does not sum started-node counts. A parent
   entered before the first-op budget/deadline check is charged. A first-fit
   layer stays unscored. Retain the final unexpanded scored C on max-depth stop.
6. beams.bin stores each root/retained beam ONCE. Next start references the
   previous retention byte range/hash and checks exact node/state/program/score
   bits. SG-BEAM1 stores product/stack counts as little-endian uint64; tagged
   arbitrary signed-magnitude integers, distinct Bool/Int, Sym UTF-8 with
   surrogatepass, Mark and counted Lst preserve actual typed values. Program
   indices use pinned OPS; scalar tags distinguish Pythonfloat/np.float32/
   np.float64 with exact little-endian bits. SG-ARRAY1 stores exact C-order
   float32/float64/bool shape/data bits, including illegal NaN payloads.
   Iterative typed value traversal avoids nested-state copying; variable bytes
   and transient allocations still carry the disclosed resource-incomplete risk.

One shared PayloadBudget counts all streams/cell seals across cells and checks
2700s/2GiB/12m attempts before writes and after writes. It never changes the
production per-cell10s deadline. Partial writes/IO errors/deadline/kill produce
no complete result. Files flush/fsync before cell seals; hashes are recomputed
before the atomic no-replace seal. Total run memory is still enforced externally
by MemoryMax4G; no implementation guarantee of fit is claimed.

AttemptReceipts atomically no-replace publishes attempt.json into the existing
root (review subdirectories do not count as a reservation), before any payload
read. selection-lock.json is write-once; terminal.json is ONE mutually exclusive
atomic final/failure receipt distinguished by kind. These small control receipts
are capped64KiB independently of the trace-payload ledger so failure bookkeeping
can survive an exhausted trace cap; a cell-seal IS charged to that ledger.
File and directory fsync and temporary-file cleanup are explicit. Receipt
metadata is not a complete provenance/result schema: future runner must pin
source, all inputs/identities/versions/origins, both-half support precedence,
seal traces before profile computation and lock choice before confirmation.
An atomic final receipt alone is not a successful complete verified result.

Companion decoders and storage integration tests are software checks, not the
independent serialized verifier. That verifier must reconstruct input selection,
all attempts/reasons/charges/ordered ranking/guard/seen/callbacks, returned
lineages, exact rational profiles/descriptives/terminal/lock from frozen literal
definitions and low-level primitives, without calling diagnostic profile or
lineage/decoder implementations. Complete-code and later final-code/result
PASS remain prerequisites; this storage implementation has neither.

Cycle3 pre-data bridge: terminal now has CLOSED success/ordinary/fuel/depth
counts, no reason dictionary. Exact classification comes from AFTER-run raw
primitive replay; structural reader checks totals/success flags only. Known-
returned replay checks global deadline between steps/examples. All streams hash
before saved array access; exact on-read contiguous payload hashes/consumed
lengths must also match. Root/retained payloads consumed once; starts share prior
retention. Malformed nonobject/noncanonical/duplicate-key/nonfinite JSON/count
types are invalid. Failed seals never reach arrays. This is structural/known-
lineage validation, NOT whole attempted-op/guard/rank/seen provenance proof.

Analysis receives144sealed cells in selected group/seed order. BOTH-half support
is first without confirmation bands. All8 selection summaries persist on any
valid support/headroom terminal. Selection prefix flush/fsync/bytes/SHA precede
immutable choice receipt; confirmation bands afterward ONLY chosen tau, none
for null. Threshold-independent populations/rank-gap/prefix curves after lock
include failures/final layers. Exact equal-group/raw/per-depth/occurrence
denominators separate. Informative-only prefix1–64 curve is descriptive. Generic
analyze alone proves no source selection/full search provenance; complete
runner/schema and independent verifier still must supply/verify those pins.

SG-INPUT1 pre-model boundary is specified in INPUT_STAGE.md. Input/config/
manifest/weight bytes hash before projection and again afterward; no model is
deserialized here. Active plus on-disk reservation is required, while the future
runner still owes exact-code/source review enforcement BEFORE reservation.
Only behaviour identity is read from side rows; original shown pairs remain.
All three ordered difference-seed pins validate before file access. The sealed
typed input/target frames, whitelisted original metadata and full input-file
checks charge the shared payload budget. A sealed insufficient-selection
outcome has no models or profile metrics. Companion reading is structural,
never the independent original-source selection/whole-output verifier.
