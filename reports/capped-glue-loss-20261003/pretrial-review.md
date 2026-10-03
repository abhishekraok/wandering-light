The program-synthesis janthu (e-den, Codex) retained this independent Claude Opus high-effort review.

Exact source62f40764bbe7b4ccf52fc3708a0eda22d850828a; unit wl-capped-glue-pretrial-review-20261003 completed exit0. PASS licenses only the registered trial; result/code review still required before landing.

# Pre-trial review: capped-policy-glue-loss (#178), D1 + Amendment A

**Verdict: PASS** for source `62f40764bbe7b4ccf52fc3708a0eda22d850828a`, for the one registered bounded trial only. This is not a result or landing claim. I wrote no receipt and launched nothing.

## What I ran
- **Tests:** the four `tests/test_capped_glue_*.py` files, 119 passed, with `wandering_light` and `experiments` resolving to this checkout. The tree was clean afterwards.
- **Pins:** all 2 input and 18 checkpoint pins hash-match `inputs.json`. This was bytes-only hashing through `verify_provenance`; nothing was parsed or deserialized.
- **Checkpoint metadata:** the six `policy_config.json` files give encoding as pinned, `ordered_actions == OP_NAMES`, mark-28, caps 160/64. The six training manifests are arm `capped`, 3 epochs, and carry the same train and side hashes.
- **Campaign directory:** `trial/` is absent and there is no receipt.
- **End-to-end synthetic probe:** `run_diagnostic` with the real `default_loader` on six freshly saved random policies exited 0 (30 groups, 73 cells, 508 rows, 153 truncated, multi-example, nested quotes).
  - Runner logits matched an independent `witness_rows` batch-of-1 construction to 4.8e-7.
  - Unmasked NLL matched `torch` cross-entropy.
- **Terminal-state probes:**

  | Case | Exit | Reading | Stage A sealed | Models loaded |
  |---|---|---|---|---|
  | Missed shown target | 2 | INVALID | no | 0 |
  | Refused witness op | 2 | INVALID | no | 0 |
  | Deadline during Stage A | 2 | INVALID | no | 0 |
  | 19 groups | 0 | INSUFFICIENT_SUPPORT | yes | 0 |

  The 19-group case made zero `legal_alternatives` and zero `encoder_diagnostics` calls.

## Earlier blockers
- **B1 (degenerate position match): resolved.** The preregistration scopes the primary to exact-op mid-span emissions. `matched_cells` raises if common support falls outside `middle` (`capped_glue_loss.py:265`). Index, span length and the two top-context flags are recorded per row.
- **B2 (alternative refusals): resolved.** Any witness refusal is INVALID (`capped_glue_loss.py:121`, `:371`). Alternatives go through `step_all_explained`, are classified by suffix under the example/op prefixes, and are scoped to the first refusing example.
- **B3 (staging and rerun): resolved.** Stage A is sealed and re-read by hash before the first `loader` call (`capped_glue_run.py:352-360`). Amendment A's no-rerun rule is enforced by the atomic `attempt.json` reservation plus the artifact pre-check (`:301-312`).

## Blocking defects
None found.

## Checked against the contract
- **Selection:** holdout and grouping use the original skeleton, with separate salts, the first original-file row, only the program substituted, and no refill. It mirrors `staged_arms.arm_rows`.
- **Ownership and spans:** lexical ownership follows `segment_composite.levels`. The Mark cross-check runs on every token and every shown example, and the shown target is verified. Spans equal `transfer.positions`.
- **Aggregation and support:** occurrences are averaged per side, then cells, then behaviours. Support needs at least 20 groups and at least 40 cells in total.
- **Losses and gate:** the legal-mass decomposition is exact, and the gate is applied per encoding as registered.
- **Round-trip:** sealed state text re-parses exactly, because literals are not captured under a mark and `[`/`]` are never captured.
- **Hidden and eval data:** hidden fields are dropped at load and no eval consumer is imported.
- **Final manifest:** it is written last and atomically. Summaries and the reading are rederived from the persisted raw vectors.

## Nonblocking limitations
1. **Launch condition.** The shared venv has an editable install of the main checkout (currently `f776d17`). Only `python -m experiments.capped_glue_run` with the working directory set to the frozen tree resolves imports to the reviewed source. The clean/HEAD guard checks `REPO` but not where `wandering_light` came from, so a script-path invocation would mix trees.
2. **Untested paths.** Runner tests use mock scorers only; the real scorer path is covered by my probe, not the suite. `main()`'s clean, HEAD and receipt guards have no test.
3. **Exit codes.** An argparse rejection exits 2, the same as a cooperative INVALID, and pre-reservation refusals exit 1. The launcher should distinguish them by whether `attempt.json` and `final.json` exist.
4. **One-shot fragility, as registered.** A single selected witness that refuses or misses its target makes the whole trial INVALID. An alternative combinator run on a doubling body is not bounded by fuel (the stack-machine-quotation-growth issue, #172) and could hit the 4G limit, giving NO RESULT.
5. **Seal completeness.** `validate_stage_a` rederives cells, support, counts and encoder diagnostics from the sealed rows. It does not rederive selection, states or legal masks from the pinned inputs; that remains for the independent result verification.
6. **Derived-only diagnostics.** Per-side omitted exposure is derivable from rows but not stored. The without-assembly primary is per checkpoint only.
7. **Smaller points.**
   - A support-only run still hashes checkpoint bytes.
   - Hidden fields are parsed transiently by `json.loads` before being dropped.
   - Only intra-op threads are pinned.
   - The hard-kill test is an in-process `KeyboardInterrupt`, not a real signal.
8. **Interpretation.** Matching is a descriptive exact-op mid-span association on training-distribution rows. It supports no causal, generalization, boundary-glue or solver claim.
