**Verdict: PASS** to run the registered O1-B diagnostic at `c664a7c` (code `42c67f5`). No blocking defects. This is not a result, and not approval of any inventory claim or BEST promotion.

## What I reviewed
- `preregistration.md` with Amendment A, and `design-review.md`.
- The final `experiments/branch_check/o1_inventory.py` and `tests/test_o1_inventory.py`, in full.
- Targeted source: `step_raw`/`run_body`/combinator runners (`stack_machine.py:273-322, 508-628`), `closed_cuts`, `_run_segment`, `_next_roots`, `_segment_beam_solve` (`stack_beam.py:604-617, 685-715, 768-778, 876-898`), `archive_record` (`search.py:431-515`), `archive_train.py:113-135`, and `closed_product`.
- The suggested suite: 49 passed. I also ran inline synthetic probes; no archive was loaded and no counts were computed.

## The five blockers: all resolved
1. **Whole-task censoring.** Fuel and depth refusals are matched by suffix (`:185-186`); atoms are checked at source and after every outer op (`:174`, `:195`). Both propagate through `required_path` (`:205`) and the matrix (`:421-422`) to clear the task's pairs (`:432-436`). No combinator rewrites an inner `out of fuel`, so the suffix match is complete.
2. **Complete matrices, positive-only labels.** The label is the minimum length over successes only (`:423-426`). Unlabelled candidates and ties form no pair (`:287-289`), and the gate reads only `complete` tasks (`:470-472`).
3. **Charging, limits, persistence.** Every path replays from its source with no memoization (`:169-200`). A refused `runs_code` op charges its allocation and a guard refusal charges 1 (`:182`). All four caps and the deadline are checked before every primitive call, with reduced allocation (`:85-105`). The only `step_raw` call site is `:180`. Partial trials and the unprocessed manifest are written (`:414-420`, `:535-546`).
4. **Fatal provenance.** Task and input errors both land in `errors` (`:527-528`, `:540-543`), and `passed` requires `not errors` (`:481`). A repeated product within a segment is an error (`:383-384`).
5. **Pairs and groups.** Unordered product pairs count once per task, first admissible by segment then digest, with repeats counted separately (`:278-302`). Union-find runs over all validated train rows of every example set (`:444-466`, `:522-529`), and a group counts only through a completed task with pairs (`:474`).

## The other checks
- **Graph splicing and ties:** vertices are products, the goal is absorbing (`:221`), and ties resolve by reverse BFS then greedy smallest token (`:227-249`). Every continuation is replayed to the goal (`:259`).
- **Closure and membership:** closure is tested on every example (`:238`, `:398`); witness membership is recomputed against all witness states (`:373`, `:392`).
- **Root eligibility:** the expanded set is the root union through the current segment (`:350-370`, `:399`), matching `stack_beam.py:878/898`.
- **Selection:** digest order, 64 tasks and 32 records fixed before replay, no refill (`:136-166`).
- **Leakage:** only shown `inputs`/`outputs` are read; import opens no data files.
- **Claims:** the docstring and the `label` field disclaim optimal distance; nothing asserts goal dependence.

## Scoped limitations (non-blocking)
- **The caps may bind, as registered.** A shape refusal of a `runs_code` op charges the full 2,000. In my 6-candidate probe, 12 such refusals were 24,000 of 24,457 charged fuel. A 32×32 matrix of them exceeds the 2M task cap. This is faithful to Amendment A.3; the result must report it as budget censoring, not absence.
- **Unexpected exceptions lose the whole run.** `AttributeError` (a null `witness` or `program`, `:336`, `:459`) and `MemoryError` are uncaught, so nothing is written. This fails closed. Optional hardening: write the partial result and exit nonzero.
- **A row failing input validation is dropped before selection** (`:517-531`), so a 65th row would fill the slot. The gate is already false in that case.
- **`gate.passed` can be true while `stop` is set or `complete` is false** (`:481`, `:547`). The preregistration permits this only as a bounded existence result. The printed summary omits `stop` (`:577`).
- **Missing run metadata.** There are no wall-clock start/end timestamps and no dirty-tree flag (`:498`, `:554`); take them from the unit.

## Test sufficiency
Adequate for the material faults. The gaps, with my probe results:
- **Distance filter (`:248`):** the tie test never has the smaller token lead farther. My probe with witness `1 drop swap add` returned `('swap','add')`, which is correct.
- **End-to-end pair:** no test produces a pair through `analyze_task`. My probe gave one pair with labels 4/12, rejected the both-on-witness pair, and counted one repeat.
- **Depth censoring:** `quotation nested too deep` is untested. My probe with 36 nested `i` was censored.
- **Record digest order** is not pinned to a literal.
- **Code inside a list:** `parse_stack` flattened `[ [ 1 1 ] 0 ]` to three ints in my probe, so the test input `[ [ add ] ]` is probably a bare symbol, not a nested list.

## What the result review must still check
- The unit logged `exit 0` with no MemoryMax or RuntimeMaxSec kill; `sha256` matches; `code_revision` is the reviewed commit on a clean tree.
- `complete`, `stop` and `errors` are read before `gate.passed`, including each error reason, since a code fault would also appear there as a provenance error.
- Censored tasks are tabulated by phase and reason and reported as budget censoring.
- The 64-key manifest is re-derived independently from the digest rule.
- A sample of pairs is re-replayed from the saved programs, confirming distinct products, equal prefix length, the same segment and at least one off-witness member.
- The group count is recomputed from `group_map`; both-off-witness pairs are reported separately.
- The write-up carries the catalogue's target-selection bias and makes no distance-ordering, goal-dependence or prevalence claim.
