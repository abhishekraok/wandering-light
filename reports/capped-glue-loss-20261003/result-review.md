# Result/code review: capped-policy-glue-loss (#178), D1 + Amendment A

**Verdict: PASS** for frozen `bcd20fe456271a16cf0756654e13d3294da2b6a5`. No blocking defect in the code, the registered result, or its interpretation.

## Blockers
None.

## What I ran
- **Six registered suites:** 153 passed in 22.6 s. `wandering_light`, `experiments.capped_glue_run`, `capped_glue_loss` and `stack_beam` all resolved to this checkout.
- **Reverification from this checkout:** exit 0 in 9.7 s. `result-review/reverification.json` is byte-identical to `verification/verification.json` and to the committed `independent-verification.json`.
- **My own recomputation:** a throwaway script that imports neither `experiments.capped_glue_*` nor `verify*.py`, only interpreter and encoder primitives. It did not deserialize a checkpoint or run inference.
- **Eight synthetic tamper probes** against both verifiers, all rejected as they should be:
  - a panel that includes a validation-holdout group;
  - a panel selected with the wrong group salt;
  - reordered checkpoints, five checkpoints, and a load-attempt count of 7;
  - `retry_allowed=true`, a log whose last line is `exit 2`, and a flipped reading.
- **Spent attempt unchanged:** sha256 of every file under `trial/`, `verification/`, `pretrial-review/` and `diagnostic-run.log` is the same before and after. The checkout is clean; my temporary files are removed.
- **Not run:** the full repository suite (2892 passed per `IMPLEMENTATION.md`) and `publish_secondary.py`.

## Provenance and terminal
- **Pins:** both inputs and all 18 checkpoint files hash-match `inputs.json`, which is identical in the source checkout and this one.
- **Source:** the execution checkout is clean at `62f4076`. Nothing under `wandering_light/` or `experiments/` differs between `62f4076` and `bcd20fe`, so the verifier runs the same interpreter semantics as the trial.
- **Terminal:** the log ends `exit 0`; `final.json` has `complete=true`, `exit_code=0`, `retry_allowed=false`. The trial directory holds exactly `attempt.json`, `stage-a.json`, `final.json` and six raw/summary pairs.
- **Verifier independence:** neither `verify_stage.py` nor `verify.py` imports a diagnostic module. They rebuild selection, replay, cells, masks and widths from the pinned inputs, not from the seals. Unsuccessful terminals are rejected before any raw file is opened, and a test enforces this.

## Selection, ownership and support (reconstructed)
- **Selection:** 7,000 train rows, 175 behaviours, 148 side programs, 134 eligible original-skeleton groups, 64 selected with no refill.
  - Holdout and grouping use the original skeleton before substitution; no held-out group is in the panel.
  - Salted UTF-8 SHA256 ordering with lexical tie-break, first original row, shown examples and target only.
  - Panel identity, original programs and capped programs match the pinned bytes exactly.
- **Replay:** all 4,112 steps match the sealed rows on Mark depth for every shown example, owner, kind, span index and length, position, product text and target. Every witness reaches its shown target.
- **Legal masks:** all 28 ops × 4,112 rows match. First-refusal counts are 19,132 ordinary, 0 fuel, 0 depth. The largest successful single call used 236 fuel against the 100,000 limit.
- **Support:** 108 cells, 45 behaviours, 45 groups, all mid-span; 137 top and 278 body occurrences. Body content is never first, last or only. No duplicate capped programs; 31 assembly cells.

## Primary, recomputed from stored logits
All 24,672 rows have 28 finite logits in exact sealed order per checkpoint.

| encoding | seed | top NLL | body NLL | delta | positive behaviours |
|---|---:|---:|---:|---:|---:|
| machine | 0 | 1.526734 | 0.736558 | +0.790177 | 26/45 |
| machine | 1 | 1.322938 | 0.682011 | +0.640927 | 29/45 |
| machine | 2 | 1.192635 | 0.772006 | +0.420630 | 23/45 |
| difference | 0 | 1.352757 | 0.758248 | +0.594509 | 27/45 |
| difference | 1 | 1.231286 | 0.754256 | +0.477030 | 26/45 |
| difference | 2 | 1.164750 | 0.849252 | +0.315499 | 19/45 |

- **Gate:** both encodings pass the median delta (+0.640927 and +0.477030, against 0.25) and have three positive seeds. Both fail only breadth: median 26/45 = 0.5778 against 0.60.
- **Reading:** `NO_MATCHED_MIDSPAN_TOP_EXCESS` is correct, and `RESULT.md` names the failing component and keeps the positive means.
- **Margin:** the breadth miss is one behaviour (27/45 would pass) in each encoding. The smallest behaviour delta in magnitude is 2.5e-4, far above the 4.8e-7 batch noise the pre-trial review measured, so no sign is in doubt.
- **Every cell and behaviour mean** in the six summaries matches my recomputation to 1e-9.

## Secondaries
- **Legal decomposition table:** all 12 rows match on NLL, legal NLL, legal mass, deficit, rank and legal size. Body legal size is 28 with zero mass loss; top is 15.32. No witness logit ties another legal logit, so rank ties are moot.
- **Shares:** top token share is 0.427206; loss and deficit shares and the defined-deficit denominators (62/64/63/63/64/64) match. Nulls are kept and excluded from that mean.
- **Exposure:** content is 491 top and 1,207 body, with 354 and 929 omitted; structural is 1,167 and 1,247. Context flags are 358 before an opener and 46 between closer and combinator.
- **Widths:** state 1/21/208 and target 1/1/14 over 19,504 example states; 73 rows (87 examples) truncate state, none truncate target. Common support has 5 rows truncated. Body depth is 1,893 at depth 1 and 561 at depth 2.
- **Without assembly:** all six deltas match (77 cells, 44 behaviours), presented as description with no replacement gate.
- **`SECONDARY.json`:** every copied block equals its hash-pinned source summary, and the label totals sum to 4,112 (all) and 415 (common).
- **`measurement.json`:** primary and without-assembly blocks equal the summaries, with cell group keys replaced by their SHA256 as the file declares.
- **Training-row multiplicity** is labelled unavailable, as registered.

## Interpretation
`RESULT.md`, `RESULTS_DETAILED.md` and `STATUS.md` keep the claim to a descriptive train-distribution exact-op mid-span association. None makes a causal, frame, generalization, search-clock, coverage or `BEST` claim; boundary glue is stated as untested, and no ablation or rerun is authorized.

## Nonblocking limitations
1. **One-ulp copy difference.** `measurement.json` takes its `encodings` block from the verifier, so difference seed 0 reads `0.5945088107442946` where `final.json` has `…947`.
2. **Import origin depends on invocation.** A script run from outside the checkout picks up the main checkout's editable install. `verify.py` guards against this by inserting its own repo root; `publish_secondary.py` needs no package imports.
3. **Verifier test gaps.** The committed independent-stage tests never include a validation-holdout group or a hash tie; my probes and the real reconstruction cover the holdout case.
4. **Path trust.** `verify.py` does not check that the stage seal path is inside the trial directory or that `--repo` is at the expected revision in git. I checked both by hand.
5. **Publisher is not idempotent-safe.** `publish_secondary.py` rewrites `RESULT.md` by string split and uses bare `assert`; it has no test.
6. **Cosmetic.** `RESULT.md` and `RESULTS_DETAILED.md` drop spaces before numbers ("median26 of45", "frozen60%").
7. **Scope of the numbers.** The legal-decomposition table is occurrence-weighted, so it is not comparable to the primary. Depth 1 and 2 bodies are pooled. Refusal reasons cover only the first refusing example.
