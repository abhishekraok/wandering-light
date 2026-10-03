# Independent D1 design review

Claude Opus, high effort on exact `da6f99a98ccc6a0c0ad92fc77431fd340e4e1dfc`. Personal-plan call with both paid API keys unset; unit wl-capped-glue-design-review-20261003 ended exit 0. The following is the full returned review.

**Verdict: CHANGES_REQUIRED** — three blocking fixes, all text amendments to `preregistration.md`. None changes the registered population or selection rule.

Review was source, compact manifests and published reports only. I hashed the two pinned inputs without opening them: both match `inputs.json`, and all six training manifests record the same train and side hashes.

## Blocking

**B1. The span-position match is degenerate: common support can only be `middle`.**
- A span starts right after a top-level `run`, where no mark is open (`step_kinds` only labels `run` when not under a mark). Its first token is therefore top-level content or `[`, never body content.
- A span's last token is the top-level combinator, or the program's final token, which is top-level or `]`. So body content is never `last` or `only` either.
- Consequences:
  - Position adds no control beyond the op name.
  - Every top-level `first`/`last`/`only` content token is excluded from the primary. That includes the first glue op after a combinator, which is where the "glue between cycles" hypothesis points.
- Fix:
  - State this in the preregistration and drop the "matching position" wording from the question.
  - Scope `NO_MATCHED_TOP_EXCESS` to mid-span tokens only; it must not read as "no top-level excess".
  - Record each occurrence's index in span, span length, and whether a top token sits before `[` or between `]` and the combinator. This is descriptive only, not a new gate.
  - If you would rather change the position scheme, that is a population change and needs re-review.

**B2. Resource censoring of non-witness ops in the legal mask is undefined, and it is one-sided.**
- Under a mark, every op is legal and nothing executes: 26 ops are captured, and `[` and `]` are always legal there. So body masked NLL equals body unmasked NLL, and body rows can never be censored.
- At top level, alternative `i`/`dip`/`map`/`filter`/`fold`/`ifte` actually run code. "Out of fuel" and "quotation nested too deep" make them illegal, on top rows only.
- "Refusal … are errors" and "Resource refusals are distinct" can only be about the witness op; for alternatives, refusal is the norm.
- `step_all_quiet` returns no reason, so the named semantics cannot make resource refusals distinct.
- Fix:
  - Witness op refused for any reason → INVALID.
  - Alternative op refused for resource reasons → treated as illegal (as search and the #168 marginals do), but classified with a reason-bearing call such as `step_all_explained` and counted per row.
  - Note that the reason comes from the first refusing example only, and that reason strings carry op-name and example prefixes; cover that with a fixture.
  - Report the secondaries with affected rows flagged.

**B3. Staging and rerun rules leave the primary hostage to the secondary and leave INVALID open-ended.**
- Legal-set enumeration is independent of the model, and it is the only step that can run away. A single `step_raw` cannot be interrupted by a cooperative deadline, and quotation growth is not bounded by fuel (the stack-machine-quotation-growth issue, #172).
- As written, a blow-up there gives a kill or INVALID after some checkpoints have been scored, with no stated rerun rule.
- Fix:
  - Stage A is everything independent of the model: selection, ownership, spans, support, legal masks, censor counts. Complete it, hash it and seal it before any checkpoint is loaded.
  - A rerun after a deadline, kill or OOM must use the identical sealed stage A and committed source. Any limit change is a reviewed amendment made before partial logits are inspected.
  - Add a table of the four terminal states (`TOP_LOSS_ROUTE`, `NO_MATCHED_TOP_EXCESS`, `INSUFFICIENT_SUPPORT`, `INVALID`) with exit code and completeness marker for each. `INSUFFICIENT_SUPPORT` should be exit 0 with complete output and no checkpoint loaded.
  - Completeness means row count equals sealed rows × 6, with the final manifest written last and atomically.

## Checked and acceptable

- **Falsifiability:** thresholds, weighting, and the both-encodings conjunction are fixed. The estimand is frozen once B1 is fixed.
- **Unmasked NLL matches training:** training is plain cross-entropy over the 28-way head, and `PolicyScorer.logits` is unmasked.
- **Grouping:** skeleton is a per-behaviour field, with 40 tasks per behaviour. Training applies `validation_skeleton` to the original task before the program swap, and the design mirrors that.
- **Mark cross-check:** `[` and `]` are never captured, bodies cannot contain marks, and executed bodies cannot push one. Lexical depth therefore equals the top-level Mark count on every example.
- **Scope:** in-distribution only, no held-out tuning or promotion, and a positive result licenses only designing a new train-only ablation.

## Nonblocking limitations

- **Illegal mass:** top-level unmasked NLL includes probability on illegal ops; body NLL cannot. Report `−log(legal mass)` per side so a positive result can be read as "illegal mass" versus "wrong legal op".
- **Redundant criterion:** "at least two seed deltas positive" is implied by median ≥ +0.25.
- **Support gate wording:** say explicitly "≥20 groups and ≥40 cells in total".
- **Row exposure:** `balanced` draws a fixed with-replacement sample (313,560 of 342,080 distinct rows), so some panel rows were never trained on. Per-row multiplicity is recomputable from `SAMPLE_SEED`; record it as a covariate.
- **Panel size:** 64 tasks out of 5,360 treated tasks (134 behaviours). Three seeds share one panel, so they are replicates, not independent tasks.
- **Body is not always code:** "body" means "under a mark" and includes data-list literals that are never executed. Top-level `quote`/`cons`/`cat` build code; flag those cells and show the primary with and without them, without gating on it.
- **Truncation:** the encoder keeps the top 160 state tokens, which cuts differently by side. Pre-cap width is always computable, so make the indicator mandatory.
- **Aliases:** 171 skeletons cover 175 train behaviours, and distinct original skeletons can still share a capped program. Report duplicate capped program texts among selected groups.
- **Split check:** assert `split == "train"` on every selected row.
- **Batching:** add a fixture that batch-of-1 and batch-of-32 logits agree; earlier scorers ran one row at a time.
- **Imports:** the runner must not call `frame_gate.eval_witnesses` or `validation_rows`, which read eval data.
- **Relation to #168:** its per-op marginals are masked deficits with virtual-frame classes, so they are not comparable to this primary.
- **Related work:** the summary is a scoped source read, as stated.
