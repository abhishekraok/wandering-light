RESULT REVIEW PASS

**Reviewed:** final HEAD `df1acd0d3eb68bb4f70f0d9e3954bff742bedefd`, with a clean tree. The measured executable source is `23c260752070d0401f16b4848586701bb2083178`. Nothing under `experiments/`, `tests/` or `wandering_light/` changed between the two commits; only the docs and the report files did.

The script, receipt and log are in `checkpoints/policy-item-reuse-screen-20261015/final-review/`: `independent_check.py` (sha256 `3c3187f2…d757d8a3`), `independent-receipt.json`, `independent-check.log` and `REVIEW.md`. The script exited 0 with no failures in 15.3 s, peaking at 0.92 GB, inside this review's unit with 12 GiB and no swap enforced.

## What I checked myself
I used only the raw files, git and low-level primitives (`parse_stack`, `step_raw`, `ops_for_basis`, `node_encoder`). I did not import the diagnostic modules, use the parent audit, run search or the model, or read held-out or reference fields.

- **Version continuity:**
  - `DESIGN.md` has not changed since the registration commit `ce9bd16`, which comes before the implementation commit `8bc22d2`.
  - The frozen profile tree is at `23c2607` and clean.
  - All 12 source pins match, and those files are byte-identical at the final HEAD.
  - The training payload, manifest and Hub revision pins match. The four weight files match their pins and equal the accepted pins from the gumbel-retention-screen (#183) run.
- **Selection:** rebuilt from the raw JSONL (7,000 rows, 171 skeletons): first row per skeleton, the `policy-item-reuse-v1` + NUL salt hash, tie by skeleton text, first 24. No two skeletons share a hash. Both `selection.json` and the shown-only projection match exactly.
- **LRU and counters:** I recomputed each task's 4096-entry LRU two ways, an ordinary LRU and the definition "hit if fewer than 4096 distinct other keys since last use". The two agree on every occurrence, and every per-task counter, within/cross split and total matches.
  - Each call's row count is beam size × examples, and the first call scores a single node.
  - Call counts match each task's depth and how it ended: 36 fits, 29 at max depth, 7 at the time cap. No task has zero calls.
  - No digest ever appears at two different widths.
- **Gate (exact fractions):** 758,653/9,321,733, 762,584/9,351,685 and 765,440/9,371,269, i.e. 8.1385%, 8.1545% and 8.1679%. None reaches 1/4, so the reuse gate fails. The range is 25,685,966,863/87,356,467,489,177 (about 0.0294 points), well under 1/20, so stability passes. Reading: NOT_JUSTIFIED.
- **Within/cross split:** within-call 5.8568–5.8745% and cross-call 2.2763–2.2977% of all positions.
- **Reports match the raw files:** PROFILE, PROVENANCE, SELECTION, AUDIT and the audit script are byte-identical to the raw files. CELLS.json matches every raw field apart from the per-call key arrays. The numbers in RESULT, STATUS and RESULTS_DETAILED match what I recomputed.
- **Cell metadata:** all three cells ran 1 thread and 1 interop thread on CPU in float32, in the profile unit's cgroup. All 10 modules loaded from the frozen tree, and each cell log equals its rows.
- **Extra key check:** I added this as a reviewer check; it does not widen the registered claim. The 366 first-call keys equal SHA256 of the 8-byte width plus the int64 padded rows that production's encoder produces from the shown inputs. Every intermediate state along all 36 fit programs appears as a complete block in that layer's recorded keys (684 of 684). So the saved key streams have the registered byte format and came from real search over real encoded items.
- **Shown replay:** I replayed every returned program with `step_raw` using the mark-28 op set (same names and order as production) and fresh default fuel per op. All 36 fits reproduce, and the same 12 tasks fit under each label. The parent's failed first audit came from the older `OP_BY_NAME` table, which has no `[`. Its corrected second run differs only in that op table and is correctly scoped. Both attempts are preserved.
- **Rehash after replay:** I rehashed 60 source, weight, input, raw, report and review-evidence files after the replay; all are unchanged.

## Blockers
None.

## Non-blocking
1. **Stale wording:** the #183 paragraph in `docs/STATUS.md` still says the item-reuse diagnostic "has not run", and DESIGN.md's status line says "proposed design only". These are docs-only fixes.
2. **Parent audit bug:** `audit.py` expects the reading `WORTH_COMPATIBILITY_STUDY` on a pass, but the code emits `REUSE_OPPORTUNITY`. It would only have fired on a passing result, so it cannot affect this one.
3. **Exclusivity:** the host checks before and after show only the profile's own process above 1% CPU. They are single snapshots, not continuous proof of exclusive use, which the report already says.

## Limits
- This is a counter replay plus spot checks of the key format. It is not a re-run of the whole search and does not reconstruct every key.
- It covers one policy, a TRAIN-only panel of 24 tasks, library off, and deadline-dependent instrumented search.
- It says nothing about numerical cache compatibility, speed, verified coverage or BEST.
- It parks this fixed scheme only; it is not evidence that caching in general cannot work.

## What may follow without another review
Only faithful archival of this review and docs metadata changes (pending → passed, plus fixing item 1) may follow without another review. Before landing, `git diff 23c2607 <new> -- experiments tests wandering_light` must still be empty. Any executable change needs a new review. Production and BEST are unchanged.
