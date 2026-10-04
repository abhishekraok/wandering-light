# Amendment A — before any data or model access

The program-synthesis janthu (e-den, Codex) wrote this, 2026-10-03 PDT.
Original frozen59f7e7d design review returned actual CHANGES_REQUIRED, retained
in design-review.md and receipt. No task selection/payload/weight access,
model/scorer call or real attempt occurred. This amendment fixes B1–B4 before
execution; it is not a cap retry. DESIGN.md incorporates the operative rules.

B1: raise global callback cap to12million above fixed worst-case11,987,136;
20-byte attempt records need at most239,742,720 bytes, scores/masks under60MB.
Successors reconstruct from typed parent states rather than being stored per
attempt. Variable state bytes remain bounded by enforced2GiB, not a guaranteed
fit. Dedicated diagnostic exceptions enforce global caps; production per-cell
wall deadline stays separate. systemd MemoryMax=4G enforces process memory.

B2: compare exact integer/Fraction protection, gain and equal-group slot means;
1-18/20 is exactly1/10, not an IEEE subtraction. Tests cover all count boundaries.

B3: zero total retained slots contributes1 on both profiles, with separate
no-score/empty-beam flags; hashes use ascending lowercase UTF-8 SHA, minimum
representative and even-zero-based selection parity. Band tests alone promote
recorded scores to float64, never change accumulated score dtype. Recorded
minus-inf is excluded under finite anchors and falls back with all-minus-inf.
Legal rows refer to the source parent, not the retained state's next expansion.

B4: NO_HEADROOM_OVER_PREFIX32 distinguishes an arithmetically impossible gain;
no route-wide kill or rejection of equal-protection/lower-slot opportunity.
Conjunction thresholds/grid/selection are unchanged. Descriptive outputs retain
all states/stops/conditional populations even when support/headroom fail.

Also pin exact score bits/callback capture instead of debugger rounding,
disclose possible prior-panel overlap and deadline-dependent support, document
integer granularity and descriptive prefix curves; correct author sort citation
from nonexistent464 to459. Fresh amended design review remains required before
data, followed by exact-code review. No corpus/model/solver/BEST alteration.
