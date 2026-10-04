# Synthetic foundations — no data runner or trial

The program-synthesis janthu (e-den, Codex) wrote this, 2026-10-03 PDT.

Original design review CHANGES_REQUIRED is retained unchanged. Amendment A
addresses B1–B4 BEFORE any task/payload/model access. New code is pure profile/
selection logic and an in-memory synthetic observer/known-lineage replayer;
there is NO bounded artifact recorder, task-file reader, model loader, attempt
reservation, real runner, selection lock serializer or independent result
verifier yet. SyntheticCapture does not retain attempted-op records and MUST
NOT be used to launch the real study. No data attempt was reserved or spent.

experiments/score_gap_profiles.py projects solver-facing shown fields without
hidden/program reads, selects original groups/metadata hash/minimum-file row,
keeps saved score dtype until float64 band tests, handles empty/all-inf cases,
uses exact Fraction protection/gain/slot means and per-seed fixed populations.
Selection and confirmation entry points are separate, reject wrong-half access,
require identical24-group per-seed panels and a locked-grid choice. Actual receipt
atomicity and prerequisite all-half support/completeness belong in the later runner.
NO_HEADROOM_OVER_PREFIX32 records an unattainable +.10 gain without denying
same-protection slot savings; the conjunction and finite grid remain unchanged.

experiments/score_gap_lineage.py uses imported production observer signatures
and low-level typed step_raw replay. It validates exact start/scored/retained
callback order through the last layer, terminal/result agreement, next beam
identity, unique terminal parent, recorded parent indices/op/program/full typed
states and cumulative-score dtype/bits. Each known suffix is replayed to all
shown targets, with no shortest-distance or negative labels. It is a foundation,
not independent verification of complete serialized search outputs: full attempt
replay/eligibility/ranking/counters/digests and artifact resource controls remain.

Meaningful synthetic tests: whitelist trap values; literal SHA group/behaviour
order and minimum source index; original validation and fallback key; float32
round-down boundary versus float64 band comparison; inclusive ties/-inf;
empty/no-score slot denominator; equal-group versus occurrence weighting;
all integer protection/gain boundaries for n12–24; three-seed no-headroom,
support, fixed-choice confirmation/failed-component and missing/substituted-panel
cases. Manually typed ordinary-production tasks test unique actual lineage,
unscored first fit, last unexpanded-layer removal, parent/state/mask/score
mutations, Bool/Int identity, open typed quotations and budget0 parent charge.
No #179 fixture constructor, run_gate, declared entry point, corpus/checkpoint
or model was called. None of these development runs is a D1 measurement.

Validation: profile-only45 tests passed0.39s; combined54 tests passed8.53s.
These are test durations, not allocator CPU-clock comparisons. Whole repository
Ruff lint/format and git diff --check passed on stable files. Production
wandering_light code has no diff from main88444b4. A first lint check reported
pairwise-style errors and unformatted drafts; replaced zip with pairwise and
formatted before final stable tests. No test failure occurred.

Command from the development tree, explicitly binding imports:
PYTHONPATH=<own frozen source> UV_PROJECT_ENVIRONMENT=/home/abhishekrao/repos/wandering-light/.venv UV_NO_SYNC=1 uv run python -m pytest tests/test_score_gap_profiles.py tests/test_score_gap_lineage.py -q -p no:cacheprovider

Fresh amended design review and later exact-complete-code PASS remain required
before real selection/model access. Preserve the original refusal and subsequent
exact-version reports; do not transfer a design PASS into data-run eligibility.
