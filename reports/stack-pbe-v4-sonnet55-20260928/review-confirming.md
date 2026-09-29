Claude Opus 5.5, high effort

**PASS**

All four findings from the first review are resolved or declined for a sound reason. The confirming checks found no new material issue.

**Earlier findings**
- **P2, provenance labels (resolved).** `validation.json` now uses `analysis_python`, `analysis_claude`, `analysis_source_sha256` and `benchmark_base_commit`. The report README says these values are captured at analysis time and that per-call runtime versions were not recorded.
- **P3, refused-call turn count (declined, reasonably).** The saved field adds `num_turns - 1` to `permission_denials`, so the exact turn count can't be recovered from it. The README and the ledger both describe it as a turn/permission proxy and not an execution trace, which is enough.
- **Timing nit (resolved).** `RESULTS_DETAILED.md` labels the Sonnet timings as summed call seconds. The values 528 / 617 / 1,306 / 1,957 / 12,229 match `validation.json`.
- **`analyse.py` writing partial files (resolved).** The `--efforts` subset option is gone, so every run audits all five arms. A missing `claude` CLI now records `null` instead of crashing.

**Confirming checks**
- **Cost arithmetic.** $/task = (1300·2 + mean_out·10)/1e6:
  - low 1,112.225 → $0.01372225 ✓
  - max 40,035.7 → $0.402957 ✓
  - The effort-sweep `analyse.py` gives the same numbers from its per-call mean, and the Sonnet rows in both `summary.json` files agree.
- **Ledger rows.** Solved, memorised, shown-only, illegal ("do not run") and $/task agree with the artifacts for all five arms.
- **Length buckets in `table.md`**, recomputed from `by_behaviour` witness lengths:
  - low 8/8, 4/8
  - medium 7/8, 6/8
  - max 7/8, 6/12, 3/12
  - All ✓.
- **Refusals.** There are four, all in max (tasks 1466, 1347, 1729, 897). Each has:
  - `stop_reason: refusal`, an empty candidate and a `failed` verdict;
  - a proxy count of 2 (8 in total) and `text_tool_call: false`;
  - 129,229–130,903 output tokens, which are priced;
  - 958–1,011 s, under the 1,800 s deadline.

  `analyse.py` asserts that a nonzero proxy occurs only on refusal errors. The plot shows `Mx*` with a side note, and the table column is headed "tool/turn proxy".
- **Headline claims.** Sonnet xhigh scores 26 at $0.0587 and GPT-6 sol max scores 26 at $0.02186, which match the summaries. The top-cluster list in the effort-sweep README is consistent with `table.md`.

**Optional nits (non-blocking)**
- In `docs/RESULTS_DETAILED.md`, the Sonnet low row says "0 tool calls". "Proxy 0" would match the max row's wording.
- The rows alternate between "untooled" and "tools-disabled" for the same command.
