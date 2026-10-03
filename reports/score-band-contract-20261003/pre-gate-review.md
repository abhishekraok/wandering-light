# Pre-gate code review: score-band-retention (#179) — **CHANGES_REQUIRED**

The retention code and fixtures implement the A/A.1 contract, but the gate runner cannot complete as written: it raises on the first catalog case. The single declared Gate A invocation would be spent on a `NO_GATE_RESULT`. No pass is granted; Gate A is not authorized on this revision.

**Revision reviewed:** `e36b89f88da314b8d8a04950dbab033bbf9d6684`, detached. The tree was clean before and after every command (`git status --porcelain --ignored` empty).

## Blocker

**B1. `capture_summary` cannot serialize any scored trace.**
- `Capture.layer_scored` (`experiments/score_band_fixtures.py:113`) stores `legal.tobytes()` in the event.
- `capture_summary` (`experiments/score_band_gate.py:130`) calls `digest(capture.events)`, and `json_value` has no `bytes` branch.
- I called `run_search_case` on `disabled/T-pair/s0/w1/h1/off` (the first catalog ID), `A-safe` and `score/executable-nan/late`. All three raised `TypeError: Object of type bytes is not JSON serializable`.
- The dev tests miss it because they duplicate the case logic and never call `run_search_case` or `capture_summary`.

**Fix:**
- Add a `bytes` branch to `json_value` (for example `value.hex()`), or record the mask as nested bools.
- Add a control test that runs `run_search_case` and strict `json.dumps(json_value(...), allow_nan=False)` on at least one case per category, plus `check_input`, `check_gc`, `check_arithmetic` and `check_api`.
- New revision, new pre-gate review.

With an in-memory `bytes→hex` patch (no file edit), the rest of the runner worked on a 19-case sample across all three search categories plus all 29 non-search checks:
- Everything serialized as strict JSON, with typed identity kept (`{"type":"Mark"}`, `Lst.items`) and non-finite values named.
- The largest case was about 0.27 MB and 0.5 s (W16/H8), so the full run should sit well under 100 MB and 300 s.
- GC state was restored afterwards.

## What I established independently

- **Production unchanged.** `git diff f9ae90b HEAD` touches only `docs/STATUS.md`, the four `experiments/score_band_*.py`, the two tests and the report directory. Nothing under `wandering_light/` or the existing tests changed. `stack_beam.py` SHA256 is `dcf4558a…616b`, matching the pinned value.
- **Import origins.** All four experiment modules and `wandering_light.{stack_beam,stack_machine,stack_pbe,stack_policy,quotation_guard}` resolve inside this tree, not the shared venv's checkout.
- **Tests.** The permitted pytest command gave `486 passed in 19.95s`.
- **Candidate against `_beam_solve`, line by line.**
  - The expansion loop is identical: budget/deadline checks, charging, legal mask, fit return before scoring, one `log_probs` call, same sort key.
  - The finite NaN/+inf check runs over every successor after `layer_scored` and before `sorted`; illegal positions are never read.
  - `_validate_band` runs before GC, parsing and `spent_in`.
  - Anchor is set only after guard, size and seen filtering. `seen.add` happens only on retention.
  - Stop precedence matches Amendment A B3, including last-row band rejection.
  - `BeamResult.pruned` is width-only, `band_pruned` is separate, and there is no widening wiring.
  - Only the band comparison casts to float64. The −inf anchor retains ordinary C and flags the fallback.
- **Independent fuzz (mine, not the author's fixtures).** 60 configurations × 6 band values = 360 runs on the fixed manual Tasks, with random tied scores, float32 and float64, −inf or NaN at illegal positions, and all guards, widths and caps.
  - Every `None` run equalled imported production in result and full event trace.
  - Every finite run passed `verify_attempts` and `verify_layers`, and observed equalled unobserved.
  - Every finite run without a band exclusion equalled production; 81 runs band-pruned and diverged.
  - My extra assertions held: depth-0 `old_seen` is exactly the inputs, `layer.retained` equals the next layer's actual beam and the `retained` event, and there is one scoring call per layer.
- **Mandatory A/B/C cases, as executed.**

  | Case | Reference | Finite arm |
  |---|---|---|
  | A-safe | `mul dup add`, 6 expansions, 78 executions | `mul dup add`, 3 expansions, 36 executions |
  | B-only-solution-cut | fits `mul dup add` | no program, `max_depth` |
  | C-later-first-fit | `dup add mul` | `mul dup add` |

- **Catalog.** 455 unique IDs in the documented order, 324 + 29 + 43 + 29 + 10 + 12 + 2 + 4 + 1 + 1, and the runner's result order matches `catalog_ids()`. Fixture definitions match FIXTURES.md A.1.
- **Runner controls, read from source.** The receipt, clean-tree, thread and hash checks all precede `out.mkdir` (exclusive), and `attempt.json` precedes any fixture. The final manifest is written only after the ID/PASS check, a hash and clean-tree recheck, and the 300 s deadline. `__pycache__` and `checkpoints` are gitignored, so the clean check will not trip on itself.

## Fix in the same revision (not blocking alone)

- **N1. Receipt check is a substring match.** `"PRE-GATE CODE PASS" not in review["result"]` also accepts a CHANGES_REQUIRED report that quotes the phrase, as this one does. Parse an explicit verdict line, or reject any result containing `CHANGES_REQUIRED`.
- **N2. Leftover temp on timeout.** If the deadline trips inside `atomic_json`, `final.json.tmp` stays on disk containing `SYNTHETIC_GATE_A_PASS`. Unlink it before raising.
- **N3. Origin check covers two modules.** Only `stack_beam` and the candidate are checked, and hashes are taken from repo paths, not `module.__file__`. Check every imported module in `SOURCE_PATHS`.
- **N4. `verify_attempts` expansions rule.** It fails falsely when a budget stops exactly at a node boundary. On the `resource/body` shape, budgets 193, 194, 339, 340, 395 and 396 trip it. The fixed budgets 0, 1, 7 and 100 are unaffected.
- **N5. Small oracle gaps.** `verify_layers` trusts the candidate's depth-0 `old_seen` and does not assert that `layer.retained` is the next beam. Both hold in my probes; add the two assertions.
- **N6. Manifest `expected` is a label.** Entries like `safe-6-vs-3` are not the expected values; hand expectations live only in assertions.
- **N7. Stale STATUS.** `docs/STATUS.md` on the branch still says no candidate code exists and the amended review is running.

## Limitations

- I did not call `run_gate` with any receipt, create an attempt or manifest, or run the full catalog through the runner's functions. Runner coverage is the sample above plus source reading.
- The ValueError cases show "scored, then no retained event"; that the check precedes sorting rests on reading the source.
- `band/inactive` returns −inf at illegal positions where FIXTURES.md says every returned score is 0. Illegal entries are ignored, so the outcome is unaffected.
- Gate A is software QA only. A's saving and B/C's loss and divergence hold by construction and say nothing about allocator usefulness, real coverage or clock.

## Commands run

- `git rev-parse`, `git status`, `git log`, `git diff --stat` and `--name-only` against `f9ae90b`, `sha256sum`, `grep`.
- The permitted pytest command, with `-p no:cacheprovider` and `PYTHONDONTWRITEBYTECODE=1`.
- Three stdin Python probes: import origins and runner helpers; runner helpers with the in-memory patch; the fuzz, A/B/C values and budget sweep.

Nothing else was run: no models, corpus, checkpoints, network, other agents, edits, commits or GitHub writes.
