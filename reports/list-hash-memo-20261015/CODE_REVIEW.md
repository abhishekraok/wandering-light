CODE REVIEW PASS

**Heads reviewed.** Both worktrees are clean and read-only; I made no edits or commits.
- Control: `d9645d7be1fba554fb0722a1e7300e7c7a5d7585`
- Candidate: `01d8254d9a0f3d9bf984678e2997e30d78de023e`
- Baseline: `ce7662de0b632ce2e97f4f2d1bfb37cf08d8bff1`

**Source claims, checked with git.**
- `ce7662d..d9645d7` touches nothing under `wandering_light/` and neither `pyproject.toml` nor `uv.lock`. It adds only the runner, tests, DESIGN, RUN and STATUS, so control production is byte-identical to the baseline.
- `d9645d7..01d8254` changes only `stack_machine.py` (+16/−1: the `_HashMemo` slot and `Lst.__hash__`) and STATUS. The last merge brought in a STATUS change only.
- The driver and test files are byte-identical between the two trees.

**Design and code findings.**
- **Hash, schema and copies:** `__hash__` stores exactly the old `hash((self.items,))` and treats only `None` as "not cached", so a cached 0 is a hit. Because the class defines `__hash__` itself, the dataclass decorator leaves it alone.
- **Pickle:** the frozen-slotted `__getstate__` saves only `[items]`. Unpickling, copying, deepcopying and `replace` all build fresh objects with no memo.
- **Staleness:** nothing in `wandering_light/` calls `object.__setattr__`, `__new__` or `__setstate__` on an existing `Lst`, so a memo cannot go stale.
- **No global state:** there is no global table and no eager hashing.
- **Solver path:** the flat beam uses `seen` only for membership checks and never iterates it. The budget check before each op, the overshoot within one product, and budget-before-deadline order are untouched.
- **Runner:**
  - The six cells run in the fixed order, and all of them finish before the shown-only replay and the analysis.
  - The parent times each subprocess from launch to return.
  - `run_body` cache is cleared at process start, and `gc.freeze()` is called as in #183/#184.
  - The gate checks every pair with `Fraction` against ≤9/10 and requires exact equality on all non-clock fields.
  - Invalid, missing, non-finite or partial data raises instead of passing, and the parent rehashes sources, model, input and raw outputs after the replay.
  - The panel selector matches the design: train split, first row per skeleton, `list-hash-memo-v1\0` salt, first 24, no refill.

**Independent checks.** All machine execution ran under real 12 GiB/no-swap systemd scopes or services, with the main `.venv` Python 3.12.8 and `PYTHONDONTWRITEBYTECODE=1`.
- **Suggested pytest set** (candidate, `LIST_HASH_CONTROL_ROOT` pointing at the real control): 100 passed, 1 skipped (the existing guarded empty-list case). The control's own `test_list_hash_memo.py`: 43 passed.
- **Lint:** ruff check and format-check pass on the three changed files. I did not rerun the full lint.
- **Real source against real source:** 3005 random nested typed values, under `PYTHONHASHSEED` 0, 17, 29 and 4242.
  - Identical: hash integers; equality partitions, which also match `value_key`; repr, asdict, astuple, fields and match args; and pickle bytes for protocols 0–5, with the candidate pickling after its memos were filled.
  - Cross-source, cross-seed loads in both directions are equal and recompute the hash for their own seed, with no memo carried over.
  - Holding: `Int(0)` and `Bool(False)` stay distinct; unequal values with colliding hashes stay distinct; a failed hash raises twice and caches nothing at any level.
  - The only differences are the extra base class in the MRO, `__hash__`'s source location, and instance size growing from 24 to 32 bytes.
- **Untrained search:** the real production `beam_solve` (width 64, depth 48, guard both, atom cap 16384, execution budget, no deadline) with a seeded tiny untrained policy on 5 hand-typed tasks.
  - Each source ran in its own process under hash seeds 0 and 29.
  - Every result field and the digests of every layer's log-probs and legal masks are identical.
  - The 3001-executions-at-a-3000-budget overshoot is preserved.
- **Runner's own readers, inside real user services** (RuntimeMaxSec 1200, 12 GiB, no swap):
  - `runtime_limits()` returns `20min`, `12884901888` and `0`, and these match the cgroup files.
  - Every module and unwrapped callable resolves inside its intended root.
  - Control's `Lst.__hash__` comes from `<string>` (dataclass-generated); the candidate's comes from its own `stack_machine.py`.

**Blockers:** none.

**Notes that don't block the pilot:**
1. `validate_sources` checks only which production file changed and that the driver bytes match. The other shared modules are equal only because the heads are pinned, so the heads, paths and pins must not change after this PASS.
2. RUN.md says caches go under the main checkpoints directory, but the runner does not set cache locations itself. The checkpoint bytes are pinned either way; record the actual cache paths at launch.
3. The primary clock also counts `git status` and the checkpoint hashing in every cell. This is the same in both arms and pulls the ratio toward 1, which makes the gate harder to pass, not easier.

**Limits.**
- Everything I ran used synthetic values, hand-typed tasks and an untrained model.
- I did not open or fetch any real TRAIN, eval or world panel or the pretrained checkpoint, and did not run the real phase or cell or reserve a timing attempt.
- The service probe is not registered provenance.
- I made no claim about speed.

**Readiness:** both sources and the design are ready for the one registered timing/compatibility pilot at exactly these two heads. A passing pilot still would not authorize landing the production hash change, a BEST change, or any held-out claim.

**What can follow without another review:** archiving this review faithfully, and changing the docs metadata from pending to passed. Any executable change, or any change to heads, pins or controls, needs a separate review.

Everything is in `/home/abhishekrao/repos/wandering-light/checkpoints/list-hash-memo-20261015/review-code1/claude/`:
- REVIEW.md
- runtime_probe.py
- xsource.py
- xsource.sh
- search_probe.py
- out/
