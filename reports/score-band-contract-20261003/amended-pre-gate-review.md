VERDICT: PRE-GATE CODE PASS

Reviewed revision `826a8bbe3c0059032ca6c190a766cbc09226efc4` (detached, clean before and after every command) for score-band-retention (#179). No blockers remain. B1 and N1–N7 are repaired, with a few small verifier gaps listed below that cannot change a gate outcome on this exact source.

## What I verified

- **Production unchanged.** `git diff f9ae90b HEAD` touches only `docs/STATUS.md`, the four `experiments/score_band_*.py`, the two tests and the report directory. `stack_beam.py` SHA256 is `dcf4558a…616b`, matching the pin.
- **Fixtures and candidate unchanged since e36b89f.** `score_band_search.py`, `score_band_fixtures.py`, DESIGN, AMENDMENT_A and FIXTURES have an empty diff; only checks, runner, tests, docs and STATUS changed.
- **Tests.** The permitted command gave `527 passed in 19.41s`.
- **Import origins.** All 9 modules in `SOURCE_MODULES` resolve inside this tree.
- **Candidate against `_beam_solve`, line by line.** The expansion loop, budget/deadline checks, charging, fit-before-score, single `log_probs` call and sort key are identical. In the candidate:
  - `_validate_band` runs before GC, parsing and `spent_in`.
  - The NaN/+inf check covers every successor after `layer_scored` and before `sorted`.
  - The anchor is set only after guard, size and seen filtering, and `seen.add` happens only on retention.
  - Only the band comparison casts to float64.
  - `BeamResult.pruned` is width-only, `band_pruned` is separate, and nothing is wired into widening.

## Independent retest of the core contract

I wrote my own definition-form oracle (ordinary C from the pre-layer seen set, then the band prefix, retained-only seen, typed `value_key` identity, `step_raw` attempts and its own budget checks). It shares no code with the candidate or with `score_band_checks`.

- **700 random configurations** on the fixed manual tasks (widths 1–16, horizons 1–6, all guards, caps, three budget units, float32/float64, −inf or NaN at illegal positions, some all-−inf):
  - Every run matched my oracle on program, termination, depth, executions, fuel, expansions, width flag, band flag and per-layer retained programs.
  - All 116 `None` runs equalled imported production in result and full event trace.
  - Of 584 finite runs, 135 band-pruned and 117 diverged from production; every finite run without a band exclusion equalled production.
  - Observed equalled unobserved everywhere, and `verify_attempts` and `verify_layers` passed everywhere.
  - In 499 runs I injected NaN or +inf at a random executable position of a random scored call; all raised the `cumulative` ValueError, observed and unobserved, with GC state restored.
- **Budget sweep (N4).** Budgets 0–450 × three units on the `resource/body` shape: 1,353 `None` runs equalled production and passed `verify_attempts` on both sides, and the same budgets at tau=1 matched my oracle. 19 of these stop exactly at a node boundary; none tripped.

## Repairs

- **B1.** `json_value` now emits `{"type":"bytes","hex":…}`. I ran `run_search_case` on all 29 semantic, all 43 resource and 48 disabled cases, plus all 29 non-search helpers. Everything serialized as strict JSON, about 1.9 MB total, largest case 118 KB, 6.8 s. The inactive-band script is still executable 0 / illegal −inf.
- **N1.** Only an exact first line grants permission. My parsing probes rejected trailing space, bold, block-quote, lowercase, CHANGES_REQUIRED first, `is_error`, wrong effort, family, receipt verdict, `pre_gate_code`, and report or log hash mismatches. All of this runs before `out.mkdir`. The test fixtures call `validate_review_receipt` only; the one `run_gate` test uses an absent receipt.
- **N2.** An owned temporary is removed on deadline and on a forced `os.replace` failure; a preexisting temporary raises `FileExistsError` and is left intact. No final is published in either case.
- **N3.** Each of the 9 modules must resolve to its intended path and the hash is taken from the actual `__file__`; origins are recorded in the final manifest.
- **N4.** The node-boundary rule is correct, including a next-layer parent with no attempt. The six regressions are extra tests; the gate's resource IDs and budgets 0/1/7/100 are unchanged.
- **N5.** Initial seen comes from the manual inputs, the retained event and the next started beam are both asserted, and the three registered mutations are rejected. My extra mutations (relabelled stop, wrong anchor, extra unread verdict, reordered ranking, stale `old_seen`, widened retention, dropped first layer) were also rejected, with one exception below.
- **N6.** `expected_search_result` uses the production reference for equivalence, resource and `None` cases, and frozen hand values plus recomputed counters and oracle layers for finite cases. The values I saw match FIXTURES:

  | Case | Reference | Finite arm |
  |---|---|---|
  | A-safe | `mul dup add`, 6 expansions, 78 executions | `mul dup add`, 3 expansions, 36 executions |
  | B-only-solution-cut | fits `mul dup add` | no program, `max_depth` |
  | C-later-first-fit | `dup add mul` | `mul dup add` |

  For every search case I ran, the manifest's expected layers equalled the actual layer entries.
- **N7.** STATUS records the implementation, the actual CHANGES_REQUIRED and that a fresh review is required. The 455-ID catalog is unique and in the documented order.

## Remaining gaps (not blocking; fix before the final code/result review)

1. **Dropped last layer is not rejected.** `verify_layers` never asserts one `BandLayer` per retained event, so removing the final layer from the capture passes. The candidate calls `band_retained` unconditionally and all 700 fuzz runs had equal counts, so this cannot hide anything on this source.
2. **Some `expected` values are recorded but not asserted.**
   - ValueError cases: `scored_layers` and `retained_layers`.
   - Arithmetic units: `expected.retained`.
   - `check_input`: expected is the same parsed object as actual; only T-empty, T-open, T-long and T-typed's first input have hand-typed assertions, the rest rely on witness replay.

   I compared these myself for all six ValueError cases and all four arithmetic units; they agree.
3. **`tau` prints as `1` in expected and `1.0` in actual** on 16 layers. Numerically equal.
4. **Transitive `wandering_light` imports are not hash-listed** (`executor`, `solver`, `corpus_hub` and others). They resolve in this tree and are covered by the clean-tree and revision check.
5. **STATUS still words the Amendment B repair as "next".** DEVELOPMENT.md and AMENDMENT_B record it as done.

## Scope and limitations

- This pass authorizes only the single bounded synthetic Gate A command in FIXTURES.md. The runner requires `HEAD` to equal the receipt's revision and a clean tree, so it must run from a checkout at `826a8bb`; committing this report first would make it refuse.
- It does not authorize landing the candidate or evidence, any real model, data, panel, threshold, horizon or campaign, or a `BEST` change. Independent final code/result review is still required.
- I did not call `run_gate`, create an attempt or manifest, or run the full catalog through the runner. The other 276 disabled cases ran through pytest only.
- That the invalid-score check precedes sorting and seen mutation rests on reading the source plus "scored, no retained event"; it is not directly observable.
- Gate A is software QA. A's saving and B/C's loss and divergence hold by construction and say nothing about usefulness, real coverage or clock.

## Commands run

- `git status`, `rev-parse`, `log`, `diff --stat` and `--name-only` against `e36b89f` and `f9ae90b`, and `sha256sum`.
- The permitted pytest command, with `-p no:cacheprovider` and `PYTHONDONTWRITEBYTECODE=1`.
- Three stdin Python probes: import origins; the oracle fuzz and budget sweep; runner helpers, tamper, receipt parsing and atomic-write failure. The last used a `/tmp` scratch directory, since removed.

Nothing else was run: no models, corpus, checkpoints, network, other agents, edits, commits or GitHub writes.
