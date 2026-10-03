# Gate A fixed manual fixtures — version A, before execution

The program-synthesis janthu (e-den, Codex) wrote this, 2026-10-03 PDT.
These are hand-reasoned expectations, NOT observed solver/test results. No
fixture has been executed. A wrong construction is reported and versioned,
not silently replaced by a search for a favourable example. DESIGN.md Amendment
A governs. All search fixtures use unchanged production mark-28 OPS, parser,
guards and interpreter; no transition/op/parser injection or toy simulator.

## Inputs, addressing and independent checks

All Tasks are manually assembled, basis mark-28, synthetic/train labels, no
held-out examples, no archive/checkpoint/corpus data. Programs below are known
manual witnesses where stated. For pure unreachable-target probes the program
metadata is empty and witness_source is `synthetic-no-witness`; this is absent
witness metadata, not a claim that identity meets the target. The solver reads
inputs/outputs and the tests must not use that metadata as guidance.

| task id | input stacks | output stacks | known witness |
|---|---|---|---|
| T-pair | `2 3`; `4 5` | `12`; `40` | `mul dup add` |
| T-list | `(1 2)`; `(3 4)` | `(2 1)`; `(4 3)` | `nil [ cons ] fold` |
| T-typed | `(true 1) false`; `(false 2) true` | `(1 true) false`; `(2 false) true` | `swap nil [ cons ] fold swap` |
| T-single | `2 3` | `12` | `mul dup add` |
| T-same | `2 2` | `4` | `add` (also mul) |
| T-probe | `2 3` | `9999` | absent |
| T-equal-probe | `2 2` | `9999` | absent |
| T-empty | `[]` | `9999` | absent |
| T-open | `MARK 0` | `9999` | absent |
| T-long | 163 top-level integer zeros | `9999` | absent |

Assert parsed typed states, not visually inferred brackets; parenthesized lists
are Lst, `MARK` under basis mark-28 is Mark, and `[]` is the empty stack.
T-long's explicit input string is `' '.join(['0'] * 163)`; its fixed generation
is serialization of one synthetic state, not data selection.

A script assigns scores by (zero-based scorer-call number, row, actual op name),
not by block bytes or hidden state. Every unspecified executable score is -100
unless a case says otherwise; illegal positions can be -inf or NaN as specified.
There is no softmax requirement on these synthetic scores. Each reference and
candidate invocation gets a fresh identical script/call counter. Call/row rules
are fixed, including when pruning makes candidate row counts differ.

For the disabled grid, score(call,row,op,s) =
`-((7 * op_index + 11 * row + 13 * call + 17 * s) % 29)`, using the ACTUAL OPS
index, s in {0,1,2}. Set illegal entries to -inf. Grid: each of T-pair/T-list/
T-typed, every s, W in {1,4,16}, H in {1,4,8}, guard in off/body/close/both,
cap None, no budget/deadline, band None: **324 reference/candidate pairs**.
IDs are `disabled/{task}/s{s}/w{W}/h{H}/{guard}`. Compare all result fields except
seconds, ordered frontiers/parents, legal masks and legal scores, refusal/counter
traces and pruned; band_pruned must be false. Reference is imported production
beam_solve/BeamObserver, never candidate(None). Pin its source hash in manifest.

The independent retention oracle consumes the actual ordered successor/score
fixture table, old seen identities and guard/size outcomes recorded at a layer.
It uses typed value_key tuples and stable plain-table sorting to form ordinary
C up to W, then includes C scores >= float64(C[0].score)-float64(tau). It imports
no candidate selection/retention code. The oracle labels only read rows; unread
eligibility stays unknown. Compare every observed finite layer to this oracle,
not only total counts. Independent low-level replay of recorded attempts checks
states/refusal scope and the reference's executions/fuel charging convention;
it is not a claim that the existing counter prices refused combinator-body work.

## Useful and adverse real-machine constructions (mandatory)

All three: T-single, W4/H3, guard off, no atom/resource cap, tau1; unspecified
scores -100. Special initial-call priorities are exact finite values.

| id | call0 named scores | call1 named scores | manual expected result |
|---|---|---|---|
| A-safe | mul0, drop-2, quote-3, dup-4 | row0 dup0; other rows unspecified | both `mul dup add`; reference expands6 nodes, finite3 |
| B-only-solution-cut | drop0, mul-2, quote-3, dup-4 | row0 dup-10; row1 dup0; others unspecified | reference `mul dup add`, finite max_depth/no program |
| C-later-first-fit | mul0, dup-2, drop-3, quote-4 | row0 dup-10; row1 add0; others unspecified | reference `dup add mul`, finite `mul dup add` |

Call2 is normally never scored because these third-layer fits happen during
expansion; B's finite nonfit call2 uses all unspecified scores. A's reference
root retains mul/drop/quote/dup; depth2 expands all four without fitting12, and
its next first parent is the mul-dup state. Finite retains only mul, then dup:
the cutoff layer's executions/scoring were still paid. B's finite path is drop,
then dup, so the final input is (2,2), from which no one op yields12. The reference
still expands mul at depth2 and gives its dup successor the top score. C keeps
mul's path in the finite arm but the reference's retained dup path scores its add
successor above mul's dup state, changing the enumerated first fit at depth3.

These are constructions, not evidence of allocator quality. If the manual counts
or programs fail on the real interpreter/reference, the fixture gate fails as
specified and the error is reported. Do not substitute a graph simulator or
infer raw timing benefit from reduced nodes. All failed/adverse cases are retained.

## Finite semantic fixture IDs

Each search case has a fresh production reference invocation and a finite
candidate invocation; expected finite retention is checked independently. Unless
stated otherwise: guard off, W4, no budget/atom cap, tau0, H1; use T-probe.
No claim of reference equality is made when finite pruning changes its frontier.

- **fit/enumeration**: T-same, H1; score mul0/add-100 but a scorer that raises if
called. Both must return add (actual OPS add precedes mul), with zero score calls.
- **fit/before-size**: same with cap0. Fit is still add; output4 exceeds cap.
- **anchor/seen**: T-equal-probe, swap0/add-2/mul-3; swap returns the seen input,
so add's4 anchors at-2; mul's4 is duplicate and does not enter seen again.
- **anchor/close**: T-empty, guard close/H1, open-mark0/literal0 score-2; the
open Mark is close-invalid with zero remaining ops, so literal0 anchors at-2.
- **anchor/type**: T-open, guard body/H1, null?0/close-mark-2. Captured null? after
literal0 makes an unrunnable open body and is type-invalid; close-mark emits
Lst(Int0), anchors at-2. No false use of a list display as an open Mark.
- **anchor/size**: T-probe, cap2, dup0/mul-2. Dup's3 atoms exceed cap; mul's6 anchors.
- **duplicate/same-layer**: T-equal-probe, add0/mul0. Add's4 retained, mul's4
same typed state duplicated; equal scores preserve OPS order.
- **seen/reentry**: T-probe, H2/tau1, call0 swap0/mul-2/drop-3/quote-4; call1
row0 mul0. The finite swap-mul state6 is retained though root-mul was band-cut.
The reference recorded6 at root and labels the same later swap-mul state duplicate.
- **identity/encoding**: T-long, H1, dup0/drop0; cap200. Dup164zeros and drop162zeros
are both retained and distinct from root163zeros, while their capped machine
encoder blocks agree. Full typed states, not encoded bytes, decide identity.
- **boundary/included**: T-probe, tau1, mul0/drop-1/quote-2. Mul6 and drop2 retained;
quote is first eligible below-band. Exact boundary is included.
- **boundary/top-ties**: T-probe, tau0, mul0/drop0/quote-1. Mul/drop retain OPS
order (drop precedes mul), quote is first eligible band cut. W1 separately tests
width precedence with an unread tail, even if that tail would be below-band.
- **ordering/cumulative**: C's real-machine second layer demonstrates fixed
negative parent scores + local scores, with the same independent table oracle.
- **band/inactive**: T-probe, H2/tau2, every returned score0 (including executable
positions) and W4; finite and None traces/results agree, band_pruned false.
- **score/all-inf**: T-probe, all executable scores -inf, H1. Retain ordinary C
in stable order; all_inf_fallback true, band_pruned false. Unread tail unknown.
- **score/illegal-nan**: T-probe, executable scores fixed finite0, all illegal
positions NaN; H2. Legal-entry comparisons/retention succeed, illegal NaN ignored.
- **score/executable-nan** and **score/executable-posinf**: T-probe, make one
executable score (including a late row such as quote) invalid. Finite raises
ValueError AFTER scoring but BEFORE sorting/seen/retention, regardless of cutoff.
Repeat with invalid dup at cap2 and invalid swap on T-equal-probe: still raises.
- **empty/eligible**: T-empty, cap0, H1. No retained state, frontier_exhausted;
anchor null, band_pruned false, width_pruned false, stop exhausted.
- **stop/width-before-unread-guard**: T-empty, cap1/guard close/H1, W5, tau2, every score0.
Five literals alone are admissible. Open Mark is guard-refused; it is read after
fifth literal unless an earlier width break stops. EXPECTATION follows reference
scanner: with W5 on the fifth literal and unread Mark, stop width/pruned true.
This is intentionally NOT exhausted-at-width. For genuine last-ranked width
completion use a separate exact table arithmetic unit, described below, because
mark-28's unread guard rows make the real case different. The manifest keeps these as separate real-search versus arithmetic-unit IDs.

## Resource/exception fixture IDs

- `resource/{unit}/{cap}/{budget}/{band}`: T-pair, W4/H3, guards both, units
executions/fuel/executions+fuel, caps None or64, budgets0/1/7, bandNone or1.
Stop budget. Budget0: executions0/fuel0/expansions1. Budget1: first literal
executes both examples, executions2/fuel2. Budget7: executions/fuel units stop
at4 literals, executions8/fuel8; combined unit stops at2 literals, executions4/
fuel4. All stop before scoring. A single op's overshoot is retained/charged.
- `resource/body/{unit}/{cap}`: T-list with H6/W4 and scripts favouring successive
nil/[ /cons/]/fold, budgets100, units all3/capNone or64. Compare disabled reference
trace/counters and low-level replay, covering combinator costs or cap interruption.
- `resource/past-deadline`: T-same deadline0; no attempt/scoring, executions0/fuel0,
expansions1. Parameter/H exhaustion and collector behaviour remain explicit.
- `parameter/{negative,nan,posinf,neginf,bool,string}`: band values -1/NaN/+inf/-inf/
True/"1". ValueError before any parsing/attempt/scorer; reference has no band API.
- `exception/scorer`: T-probe scripted scorer raises a distinct RuntimeError;
original GC enabled and disabled each restored. No suppression/partial gate pass.
- `observation/{case}`: every finite semantic and A/B/C case is repeated without
observer; same program/termination/counters/depth/width+band flags, seconds ignored.
All disabled grid calls use a capturing observer on BOTH paths. Reference case
may be observed without pretending its width-only recorder understands bands.

## Separate arithmetic/scorer API units (no solver-outcome claim)

No transition injection. These are ordinary unit inputs used to check boundary
arithmetic/diagnostic behaviour, not state graphs labelled as production solves.

- `unit/last-ranked-width`: table of exactly W distinct eligible rows, including
all-inf table, expected stop exhausted/width_pruned false on final-row completion.
- `unit/width-before-band`: W retained while more rows are unread, expected width/
width_pruned true, no claim about unread eligibility and band_pruned false.
- `unit/scorer-dead-and-live`: existing PolicyScorer.log_probs unbound method on
an object returning scripted zero logits, legal mask with one all-false row and
one literal0-only row. Expected all-NaN first row, finite0 at live legal position,
-inf at live illegal positions; use NaN-aware comparison. No model is made/run.
This validates the API caveat; it is NOT a reachable mark-28 parent. Every fresh
per-op expansion allocation has fuel100000 and literal0/1 have no type guard,
so a fully expanded real parent always contributes legal literal successors.

A complete manifest lists the fixed expanded IDs, actual OPS order, source hashes,
all Task/scorer/setting definitions and every outcome; summarize no selected subset.
Scorer call/row addressing, the empty witness placeholders, arithmetic/API unit
scope and absence of real coverage/clock remain explicit limitations.
