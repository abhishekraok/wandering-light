# Frontier models on `stack_pbe_v4` through `codex exec`: an effort curve and an anchor (phase 1)

Date: 2026-09-10 · [codex-exec-frontier-effort-curve (#123)](../../issues/123)
· corpus `stack_pbe_v4` (Hub revision `00c2a999`), seed-17 40-task eval sample,
the same sample as every v4 frontier and local-arm row. Phase 2 (the full
1,800-task split, for the authorship split at matched length) has not run.

## Protocol

Same prompt, parser and grader as the luna API row in
[`stack-pbe-v4-baseline-20260901`](../stack-pbe-v4-baseline-20260901/README.md):
a solver sees only the 3–6 shown pairs, one sample per task, graded on 8
held-out pairs. The model is reached through `codex exec` (CLI 0.153.4) on a
Pro subscription instead of the Responses API, via
`CodexExecTokenGenerator` (`--provider codex`), trimmed to the API protocol:

| configuration, measured on one task | input tokens |
|---|---:|
| `codex exec` as installed | 15,856 |
| private `CODEX_HOME` holding only `auth.json` (no global AGENTS.md, skills, plugins) | 15,368 |
| + every tool feature disabled (`--disable shell_tool`, `unified_exec`, `code_mode_host`, `multi_agent`, …) | 11,691 |
| + `model_instructions_file` replacing the agent instructions with one line | 7,594 |
| the task prompt itself | ~400 |

What remains above the prompt is tool schema Codex sends regardless (the
code-mode `exec`/`wait` pair, `request_user_input`, the `collaboration`
namespace; confirmed in a request trace). With `code_mode_host` off they fail
closed: asked to run a command, the model answered `NO_TOOLS` and the event
stream shows the call erroring. Every call runs from an empty directory,
`--ephemeral`, `--strict-config`. The `--json` event stream's tool attempts
and token usage are recorded per task under `generation` in each JSON.
**No arm attempted a tool call on any task**, so every row below is one
untooled sample per task, the API protocol.

Arms: `gpt-5.6-luna` high as the anchor against the API row (same model,
effort, sample, seed); `gpt-5.6-terra` high; `gpt-6-astra` at low, medium and
high, the effort curve. Five arms ran concurrently, 200 calls, 49 minutes wall.

## Results

| arm | solved | memorised | shown-only | `aligned_matches` shortcut | illegal | reasoning tok/task, median (max) | wall | artifact |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `gpt-5.6-luna` high, API row (2026-09-01) | 15 (37.5%) | 3 | 45.0% | 1 → 14 real | 18 | — | 2,286 s | `../stack-pbe-v4-baseline-20260901/luna_v4_sample40.json` |
| `gpt-5.6-luna` high | **12 (30.0%)** | 6 | 45.0% | 1 → 11 real | 17 | 2,540 (11,622) | 2,936 s | `luna_high_v4_sample40.json` |
| `gpt-5.6-terra` high | **16 (40.0%)** | 4 | 50.0% | 1 → 15 real | 12 | 840 (5,949) | 1,344 s | `terra_high_v4_sample40.json` |
| `gpt-6-astra` low | **24 (60.0%)** | 16 | 100.0% | 1 → 23 real | 1 | 256 (1,311) | 606 s | `astra_low_v4_sample40.json` |
| `gpt-6-astra` medium | **24 (60.0%)** | 16 | 100.0% | 1 → 23 real | 0 | 278 (1,447) | 661 s | `astra_medium_v4_sample40.json` |
| `gpt-6-astra` high | **23 (57.5%)** | 15 | 95.0% | 1 → 22 real | 1 | 396 (2,787) | 840 s | `astra_high_v4_sample40.json` |

"Shown-only" is what a grader without held-out examples would have said.
"Illegal" counts candidates that do not run. Wall is the sum of per-call
seconds for the five concurrent arms. Every arm's one `aligned_matches` solve
is the head-of-first-list sampler shortcut recorded in the baseline report.

### Solves by witness length, and by who wrote the behaviour

| arm | 1–10 ops | 11–20 | 21–40 | 41+ | hand-written (v3's 40) | `gpt-5.6-sol`-written |
|---|---:|---:|---:|---:|---:|---:|
| luna high, API row | 6/8 | 5/8 | 4/12 | 0/12 | 5/8 | 10/32 |
| luna high | 6/8 | 4/8 | 2/12 | 0/12 | 6/8 | 6/32 |
| terra high | 5/8 | 5/8 | 5/12 | 1/12 | 6/8 | 10/32 |
| astra low | 8/8 | 8/8 | 7/12 | 1/12 | 8/8 | 16/32 |
| astra medium | 8/8 | 8/8 | 7/12 | 1/12 | 8/8 | 16/32 |
| astra high | 8/8 | 7/8 | 7/12 | 1/12 | 8/8 | 15/32 |

The authorship columns are **confounded with length on this sample**: seven
of its eight hand-written tasks are 5–7-op witnesses (`keep_below`,
`keep_negative`, `keep_zeros` ×2, `n_minus_each`, `negate_each`, `reverse`)
and the eighth is `prefix_sums` at 25, while only one sol-written task
(`rotate_left`, 5) is under 11 ops. Phase 2 exists to make that comparison at
matched length over all 45 eval behaviours.

## What the numbers say

**The anchor holds within one-sample noise.** Luna through `codex exec`
scores 12/40 against the API row's 15/40. Paired by task (same seed, same
order) the two agree on 33 of 40; both solve 10, the API row alone 5, codex
alone 2. The seven flips are short behaviours: `rotate_left` (5 ops),
`keep_below` (6), `min_pair` (12), `replace_value` (22, twice),
`concatenate_shorter_first` (26), `largest_inside_interval` (40). That is the
temperature-1 flip pattern the local-arm report saw between two runs of the
same model, not a length-dependent shift. One sample per task cannot separate
a 3-task harness residual from sampling noise, and a replicate through the
API is what it would cost to try; the residual, if any, is smaller than the
gaps the rest of the table reports.

**Astra fits the shown examples on every task, and the held-out set does all
the grading.** All three astra arms return a program that runs and matches
every shown pair on 38–40 of 40 tasks; luna manages 18. Fifteen or sixteen of
those fits are memorisations: arithmetic coincidences assembled from
`1 1 add` constants that reproduce the 3–6 shown outputs and fail the 8
hidden ones (`widest_pair` → a constant; `bounds_cover_list_flag` → a
polynomial in the list head; `adjacent_sign_change_count` → count of
negatives). Shown-only grading would put astra at **100%**; held-out grading
puts it at 60%. The overstatement is **40 points**, against 7.5 for the luna
API row and 0.7–2.7 for BFS and terra on v3. The stronger the solver, the more
the held-out protocol is worth; this is the largest gap the project has
measured.

**The effort curve is flat, and the reason is that astra did not use the
knob.** 24 / 24 / 23 at low / medium / high. Median reasoning per task went
256 → 278 → 396 tokens and never exceeded 2,787; terra at high reasons 840
median, luna 2,540. On this prompt astra low and astra high are the same
solver, so the row does not say whether more thinking would buy longer
programs; it says the effort setting does not make astra think more here.
A curve that varies compute would need a model that responds to the setting
(luna and terra do) or a prompt that asks for it.

**The length wall does not move.** Every arm solves 0 or 1 of the 12 tasks
past 40 ops. The three long solves are each a different behaviour on a
different arm: terra's 54-op `farther_choice_from_target`, astra low's 59-op
`farthest_from_target_last_tie` (astra high memorised the same task), and
astra medium and high's 49-op `positive_balance_prefix_count` with a
two-accumulator fold. All are held-out-surviving; none has had the sampler
audit that #101's long solves failed, so they are quoted as solves under that
caveat. Below the wall the gradation is clear: 21–40 ops goes 2–4 (luna),
5 (terra), 7 (astra) of 12.

**Terra is not ahead of luna on v4.** 16 against 12–15, with 4 memorised
against 3–6. On v3 the gap was 82.5 against 75.0 on the same sample. v4's
branch-bearing 40-op median closes it.

## Cost

200 calls, ~1.55M input tokens (of which ~7k per call is the unremovable tool
schema), 272k output tokens, on the subscription. Wall clock is the
per-arm sum above; the five arms ran concurrently in 49 minutes. Luna is the
slow arm at 52 s median, 216 s max per task.

## Caveats carried with every row

180 of 220 behaviours were written by `gpt-5.6-sol`, so a frontier number
here is corpus calibration, not a model ranking; median 8-op window overlap
between witnesses is 79.8%, so success is composition; the sample's median
witness is 26 ops against the split's 40; one sample per task at the model's
default temperature, and same-task verdicts flip between runs.

## Files

`run_phase1.sh` launched the five arms; `analyse.py` produced the tables
above from the five JSONs and the API row; `*.log` carry per-task progress.
