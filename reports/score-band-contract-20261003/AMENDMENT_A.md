# Amendment A, before implementation or fixture execution

The program-synthesis janthu (e-den, Codex) wrote this, 2026-10-03 PDT.
The actual independent Claude Opus high-effort design review returned
CHANGES_REQUIRED on69e53c8, exit0; report and receipt are retained beside this
file. This amendment changes the specification only. No solver, fixture, gate,
model, checkpoint or corpus call has occurred; no result or trial threshold is
changed. The single chosen retention mechanism remains the same.

- B1: all end-to-end fixtures now use real hand-written mark-28 Tasks, the
unchanged machine/parser/OPS and imported production beam_solve/BeamObserver.
No transition/op injection or candidate(None) substitute for the reference.
Fixed scripts address call/row/op, explicitly including differing row counts.
FIXTURES.md supplies concrete useful/adverse paths and manual expected outcomes;
these are hypotheses about constructions, not observed results. Separate pure
arithmetic and scorer-API units carry no production solve/outcome claim.
- B2: invalid band parameters raise ValueError before all work. Finite-mode
NaN/+inf raises ValueError over EVERY non-fit executable successor's cumulative
score AFTER scoring and BEFORE sort/retention/seen changes, ignoring illegal
matrix positions. Disabled traces compare legal entries/NaN-aware matrices.
A fully expanded mark-28 parent cannot have an all-illegal row: literal0/1 are
unconditionally legal and each step_all_* attempt has fresh fuel100000. The
requested dead/live-row regression is therefore an explicitly separate scorer-
API unit with scripted logits/masks, no model, while end-to-end tests retain
real OPS and put NaNs only at illegal positions. This is a source-based scope
clarification, not an executed result.
- B3: exact exhausted/width/score_band stop precedence; W reached on last ranked
row means exhausted, W reached with unread rows means width, even if next row
would be below-band. Unread rows have no eligibility/verdict/count. Width and
band pruning are separate; wrapped BeamResult.pruned retains WIDTH meaning.
Experimental search is not integrated with widening/saturation.
- Nonblocking points: band comparison casts the already accumulated reference
scores/tau to float64; no ranking precision change. Scripted scores need not be
normalized. Add body guard, all three budget units, atom cap on/off, and adverse
fixtures. The later unregistered real campaign requires a narrower fixed-width
control and a train-only score-gap/rank/depth feasibility read before threshold
selection; no historical or held-out retrospective spend oracle.

Fixed fixture table and code will be committed before any development solver
fixture execution. DESIGN PASS on this amended text permits synthetic fixture
implementation/development testing only; separate pre-gate code review and
independent code/result review before candidate landing remain required.
No actual-data panel, threshold, policy, horizon, coverage/clock measurement,
BEST change or owner decision is authorized by this amendment.
