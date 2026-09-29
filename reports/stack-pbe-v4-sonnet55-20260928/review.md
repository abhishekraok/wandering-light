Claude Opus 5.5, high effort

# Review: Sonnet 5.5 on the v4 40-task sample (#176)

**Verdict: PASS.** Nothing blocks merging. Two low-priority fixes and two nits are listed below.

## What I checked, and it holds

**Protocol**
- The run uses the unchanged `stack_pbe_baseline.py` with `--tasks 40 --split eval --seed 17` and one sample per task.
- `ClaudeExecTokenGenerator` runs with `--tools ""`, strict MCP, no settings sources and no session persistence.
- The preflight asserts that the task indices match Opus 5.5 low's.
- `analyse.py` recomputes the seed-17 shuffle, asserts all five arms against it, and re-grades all 200 candidates.
- `validation.json` `task_indices` matches the first index in the JSONs (361).
- `.gitattributes` pins `run.sh` to LF, which is correct for the Windows-worktree/WSL setup.

**Counts**
- In every arm, solved + memorised = shown_fit: low 17+4=21, medium 18+6=24, high 23+11=34, xhigh 26+11=37, max 24+11=35.
- Max `by_behaviour` sums to 24 solved and 11 memorised. Low has 17 `solved` verdicts.
- Illegal counts are 10 / 8 / 3 / 1 / 5, which match "do not run" in `RESULTS_DETAILED.md`.
- Max's 5 illegal are the 4 refusals plus task 359.

**Cost arithmetic**
- The formula is `(1300·2 + mean_out·10)/1e6`.
- Low's output sum is 44,489, so the mean is 1,112.225 and the cost is $0.01372225. ✓
- Max's output sum is 1,601,428, so the mean is 40,035.7 and the cost is $0.402957. ✓
- Medium, high and xhigh ($0.01632225, $0.03800225, $0.0587025) all reproduce from their means.
- The effort-sweep `summary.json` Sonnet rows are identical to these.

**Claims**
- Every `RESULTS_DETAILED.md` Sonnet row agrees with the artifacts: solved, memorised, shown-only, illegal, $/task, 0 tool calls outside max, and 8 proxy counts on max.
- The `aligned_matches` shortcut (task 1634, `drop uncons swap drop nil swap cons`) is solved in all five arms, so "includes 1 known shortcut" is correct.
- The GPT-6 sol max figure of $0.02186 matches 0.0218555.

**Plot integration**
- Pricing is $2/$10 in `PRICE`.
- The Sonnet special case requires 40 model-confirmed calls with nonzero usage, and all 5 arms qualify.
- `refusals` is derived from `stop_reason == "refusal"` and equals 4 for max. It drives the `Mx*` marker and the side note.
- The ARMS stems are present, and the column header was renamed to "tool/turn proxy".

**Refusals and the proxy**
- All four refused calls show `stop_reason: refusal`, an empty candidate, a `RuntimeError`, a `failed` verdict, `tool_calls: 2` and `text_tool_call: false`.
- Their outputs are 129,229 to 130,903 tokens, matching "129–131k".
- `analyse.py` asserts that a nonzero proxy occurs only on refusal errors.
- The README correctly describes the field as `max(num_turns-1,0)+permission_denials` and not an execution trace.

## Findings

**P2: `validation.json` records analysis-time versions, not run-time versions**
- Location: `reports/stack-pbe-v4-sonnet55-20260928/analyse.py` L127–129 (`"source_commit"`, `"python"`, `"claude"`) and the `source_sha256` block.
- `claude --version`, `platform.python_version()` and the source hashes are captured when `analyse.py` runs. `source_commit` is a hardcoded literal.
- The README says the file "records … source hashes, runtime versions". If the CLI auto-updated during the two-day run, or the analysis ran from a different checkout, the record is wrong. The CLI version matters here because it governs the turn/refusal behaviour.
- Fix, either of:
  - Capture `claude --version` and the source hashes in `preflight.py` (it already writes `probe.json`), or write them per arm in `run.sh`.
  - Or rename the fields to `analysis_*` and soften the README sentence.

**P3: the refusal evidence points to three model requests per refused call**
- Location: `sonnet55_max_v4_sample40.json`, tasks 1466, 1347, 1729 and 897.
- These calls report `input_tokens` of about 3.5k and `cache_creation_input_tokens` of about 3.5k. Every other call reports about 1.1–1.2k, so the refused calls used about 3× the input.
- That is consistent with `num_turns = 3`: two CLI-initiated continuation turns, with no tools executed. The ~130k output is therefore summed over those turns.
- The README's wording is accurate but understated. Consider adding one sentence: "input usage (~3× a single prompt) indicates three model turns within one `claude -p` invocation; no tool was available".
- The cost effect of pricing these calls at 1.3k input instead of ~3.5k is about $0.0004/task on $0.403. That is immaterial, and it follows the plot's stated fixed-prompt estimate.

**Nit: the "wall (s)" column holds summed call seconds for the Sonnet rows**
- Location: `docs/RESULTS_DETAILED.md`, the Sonnet rows.
- These values are summed per-call `seconds` from `validation.json`, not the run wall clock.
- Since each arm ran its tasks sequentially, the two are nearly equal. Either say "call seconds" in the notes or use the log's START/FINISH timestamps.

**Nit: `analyse.py` overwrites `validation.json` and `summary.json` wholesale**
- Running it with `--efforts` on a subset silently drops the other arms from both files, and running it without the `claude` CLI crashes.
- Consider refusing to write provenance unless all five efforts are present.

## Not verifiable from the evidence

I could not check `accuracy_vs_cost.png` itself or the effort-sweep `table.md` and README. The code that generates them is correct, and the reproduce block regenerates both.
