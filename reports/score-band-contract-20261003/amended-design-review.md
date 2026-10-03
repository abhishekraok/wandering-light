# AMENDED DESIGN review: score-band-retention (#179) — **AMENDED DESIGN PASS**

**Revision:** `e8213dec2f4fe28f18b5e2269c5ead3f52ddcf5f`, clean tree. `wandering_light/`, `tests/` and `experiments/` are byte-identical to the reference base `f9ae90b`. Reference `stack_beam.py` SHA256 is `dcf4558a6cfa11d8b8dde9aaa02ff21c9837ec16279a692f9b7e723c94ec616b`.

B1, B2 and B3 are repaired. No blockers remain. Four small items (N1–N4 below) should be pinned in a versioned `FIXTURES.md` before any fixture executes; none needs another design round.

## What I inspected and what I did not run

- **Read in full:** `AGENTS.md`, `DESIGN.md`, `AMENDMENT_A.md`, `FIXTURES.md`, `design-review.md`, `design-review-receipt.json`, `wandering_light/stack_beam.py`, `wandering_light/quotation_guard.py`.
- **Read in part:**
  - `stack_machine.py` lines 54–853: values, `Op`, `step_raw`, `run_body`, `_literal`, all op tables, `MARK_OPS`, `parse_stack`.
  - `stack_policy.py` lines 60–345: encoder, `encode_stack`, `PolicyScorer.log_probs`.
  - `stack_pbe.py`: `BASIS`, `OPS` and the `Task` dataclass.
  - `tests/test_stack_beam_equivalence.py`: the `_Deterministic` scorer stub.
- **Commands run:** only `git rev-parse`, `git status`, `git diff`, `sha256sum`, `grep`, `wc`, `ls`.
- **Not run:** no tests, solver, interpreter, parser, model, scorer, corpus, checkpoint, archive, download, other agent or CLI, publication or file edit. Every fixture claim below is hand-traced from source.
- **Not checked:** the literature citations and the `bfbs` SHA.

## B1: real machine, frozen reference — repaired

- `_beam_solve` reads only `task.inputs` and `task.outputs`, so an empty `program` with `witness_source="synthetic-no-witness"` cannot guide the search.
- A scripted scorer without an `encoder` attribute falls back to machine blocks, as the existing `_Deterministic` stub does. No injection is needed.
- The design states the reference is imported `beam_solve` plus `BeamObserver`, never candidate(None), with the hash in the manifest. Arithmetic and scorer-API units are labelled separately with no solve claim.
- Scripts address (call, row, op name), and a fresh counter per invocation is stated.

The mark-28 order I used for all traces is: `0 1 nil true false dup swap drop over cons uncons null? add sub mul div eq? lt? dip quote cat i map filter fold ifte [ ]`.

**Fixtures A/B/C, hand-traced on T-single (`2 3` → `12`, W4, H3, guard off, tau 1).** The root has 17 executable successors and none fits. No depth-2 successor of `(6)`, `(2)`, `(2 (3))` or `(2 3 3)` equals `(12)`. I checked `i` and `dip` on `(2 (3))`, which give `(2 3)` and `(3 2)`.

| Fixture | Reference (traced) | Finite arm (traced) | Matches table |
|---|---|---|---|
| A-safe | keeps mul, drop, quote, dup; next beam `(6 6) (6 0) (6 1) (6 nil)`; `add` on `(6 6)` fits; 1+4+1 = 6 expansions | keeps mul only, then `(6 6)` only; fits at 3 expansions | yes: both `mul dup add`, 6 vs 3 |
| B-only-solution-cut | beam `(2) (6) …`; `(6 6)` at −2 outranks `(2 2)` at −10; fits `mul dup add` | drop anchors and mul at −2 is cut; `(2 2)` has no one-op route to 12; ends `max_depth` | yes |
| C-later-first-fit | `(2 6)` at −2 outranks `(6 6)` at −10; `mul` (index 14) fits after add and sub miss: `dup add mul` | mul only, then `(6 6)`, then add: `mul dup add` | yes |

The other semantic cases also trace as written:

- **anchor/close:** `(MARK)` has one mark and zero ops remaining.
- **anchor/type:** `body_effect` on `(0 null?)` returns unrunnable, and `]` yields `Lst(Int 0)`.
- **anchor/size, anchor/seen, duplicate/same-layer:** add (index 12) precedes mul (14).
- **seen/reentry:** as written; the band-cut `(6)` is re-admitted in the finite arm and is a duplicate in the reference.
- **boundary/top-ties:** drop (index 7) precedes mul.
- **identity/encoding:** 162, 163 and 164 zeros all cap to the same 160 tokens.
- **empty/eligible:** six successors, all over cap 0, ending `frontier_exhausted` at depth 1.
- **stop/width-before-unread-guard:** `nil` is one atom, W5 fills at rank 4 of 6, reference `pruned=True`.
- **Resource counts:** 0/0/1, 2/2, 8/8 and 4/4 follow from the pre-attempt check and two examples per literal.
- **Witnesses:** T-list and T-typed are valid; the fold builds `(2 1)` and `(1 true)`.

## B2: invalid scores — repaired

- The check is defined as a raise, after scoring and before sort or seen mutation, over every non-fit executable successor's cumulative score. Illegal positions are never read. Parameter errors come before all work, beside `spent_in`.
- **The unreachable-dead-row clarification is correct.** `_literal` and `[` have a null guard and no capture, and `step_raw` gets a fresh `DEFAULT_FUEL_LIMIT` per attempt. A fully expanded node therefore always has at least six legal successors; a budget or deadline stop returns before scoring.
- **It hides no solver case.** The solver-relevant property is that illegal positions are never inspected, and score/illegal-nan covers that end to end at H2.
- **The API unit's expectation is right.** For the all-false row, `peak` becomes 0, `shifted` is −inf, `norm` is log 0, and the row is NaN. The live row gives 0 and −inf.
- The −inf fallback is well defined: ranking is nonincreasing, so a −inf anchor means every later row is −inf. Float64 is used only for the band comparison, which leaves ranking untouched.

## B3: stop cause and flags — repaired, one corner to pin (N1)

- Width on the last ranked row is `exhausted`, matching the reference's `rank + 1 < len(ranked)`.
- Width with an unread guard-refused or below-band tail is `width`, matching the reference's conservative `pruned`.
- Unread rows carry no verdict. `BeamResult.pruned` stays width-only, `band_pruned` is separate, and there is no widening, debugger or `BEST` wiring.

## Nonblocking items to pin before fixture execution

- **N1. Stop-cause overlap.** If the first eligible below-band row is the last ranked row, both `exhausted` and `score_band` hold as written. This wording came from my own earlier repair text. `band_pruned` is unambiguous (true). Add one sentence giving `score_band` precedence, and one arithmetic unit for it.
- **N2. Last-ranked width is realisable on the real machine.** T-empty with W6, guard off, H1 has exactly six eligible successors, so the reference gives `pruned=False`. `FIXTURES.md` says the real case differs and uses only a table unit. Add this as a real fixture, for None and finite. Otherwise nothing in the grid is likely to check the disabled path's `pruned` edge against production.
- **N3. `resource/body` is underspecified and under-reaches.**
  - "Scripts favouring successive nil/[/cons/]/fold" has no exact scores, and the guard mode is not stated.
  - At budget 100 with two examples and W4, the search stops in layer 2. That reaches only `i` at the root, never the fold body.
  - Pin the scores, and either raise the budget or narrow the claim to `i`. The comparison is equality with the reference, so no expectation can be biased.
- **N4. NumPy warnings.** `unit/scorer-dead-and-live` will emit divide and invalid RuntimeWarnings from `log_probs`; expect or suppress them in the test, not in production.

## Remaining limitations

- Gate A is software QA. A's reduction holds by construction, and B and C show loss and divergence by construction. Nothing here supports a useful-policy, coverage or clock claim.
- The handwritten expectations are source-traced by me, not executed. A mismatch on the real interpreter is a reportable fixture failure, to be versioned, as the design says.
- The independent oracle consumes recorded eligibility outcomes, so it checks retention arithmetic, not the guards. Guard correctness rests on the disabled grid against production.
- The later study is correctly marked unregistered. It names the narrower fixed-width control, the train-only score-gap/rank/depth screen before threshold selection, and upfront shared horizon and caps with no held-out or retrospective oracle.

## What this PASS authorizes

Synthetic implementation and development tests on the fixed manual fixtures only. A separate high-effort pre-gate code review must approve the code and fixture table before the declared Gate A invocation, and an independent code and result review must precede landing. No real threshold, panel, checkpoint, horizon, campaign, data call or `BEST` change follows.
