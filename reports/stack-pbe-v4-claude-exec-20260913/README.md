# Claude Opus 5 on `stack_pbe_v4` through `claude -p`: the effort curve `codex exec` could not measure (phase 1)

Date: 2026-09-13 · [opus-effort-curve (#137)](../../issues/137)
· corpus `stack_pbe_v4` (Hub revision `00c2a999`), seed-17 40-task eval sample,
the same sample as every v4 frontier and local-arm row. Phase 2 (the full
1,800-task split) is not started; the last section says what it would cost.

## Why

[codex-exec-frontier-effort-curve (#123)](../../issues/123) asked what a
frontier model's solve rate does as reasoning effort grows and got a flat
curve for a reason that says nothing about the corpus: `gpt-6-astra` did not
spend the knob (median reasoning 256 → 278 → 396 tokens per task, solves
24 / 24 / 23 of 40). Claude Opus 5 does spend it, so the same protocol on Opus
is the measurement #123 set out to make. It is also the first row on v4 from a
family other than the corpus author's: 180 of 220 behaviours were written by
`gpt-5.6-sol`.

## Protocol

Same prompt, parser and grader as the luna API row in
[`stack-pbe-v4-baseline-20260901`](../stack-pbe-v4-baseline-20260901/README.md):
a solver sees only the 3–6 shown pairs, one sample per task, graded on 8
held-out pairs. The model is reached through `claude -p` (Claude Code 2.1.265)
on the Claude subscription, via `ClaudeExecTokenGenerator`
(`--provider claude`), trimmed the way the `codex exec` rows were:

- `--tools ""` (no built-in tools), `--strict-mcp-config` (no MCP servers),
  `--setting-sources ""` (no user, project or local settings),
  `--no-session-persistence`, an empty working directory **outside the
  repository** so no `CLAUDE.md` or git context is discovered, and
  `--system-prompt` replacing Claude Code's system prompt with two lines:
  *"Follow the instructions in the message. You have no tools and cannot run
  commands or code; answer from the message alone."* `--bare` would strip
  more but disables OAuth, and the subscription is the point. No
  `--fallback-model`: a fallback would answer as another model. The child's
  environment drops `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`,
  `ANTHROPIC_BASE_URL` and the Bedrock / Vertex / Foundry switches, which
  would re-route the call away from the subscription without a trace (found
  in review after the runs; none was set when they ran).
- `--output-format json` returns the answer with input, cache-creation,
  cache-read, output and thinking tokens, the canonical model that served the
  call, the turn count and permission denials. All of it is under
  `generation` in each JSON, with the codex key names so one analysis reads
  both (`reasoning_output_tokens` is thinking tokens; `input_tokens` counts
  cached tokens too).
- Residual above the prompt: **~0.6k input tokens** (a 12-token probe cost
  622; the task prompts cost a median 1,325 against Codex's 7.6k residual of
  tool schema). The issue body quotes 3.4k; that was measured from a directory
  inside this repository, where Claude Code adds git context, and is what the
  empty directory outside the repo removes.
- **Every call was served by `claude-opus-5`** (the `opus` alias, recorded per
  call), every call ended `end_turn` in one turn, and **no call attempted a
  tool or wrote a tool call into its answer** (0 of 120). The probe for #137
  had seen Opus at effort high answer a task with a Bash call written as text
  when the system prompt did not mention tools; the line saying there are
  none was enough. `parse_result` still flags the shape, so a row where the
  count is nonzero would say so.

Arms: `claude-opus-5` at effort **low / high / xhigh** (no medium, by the
owner's choice; Claude Code's own default is xhigh). Three arms ran
concurrently, each sequential over its 40 tasks, 120 calls, **59.5 minutes
wall**. Reference rows are `gpt-6-astra` low and high from
[`stack-pbe-v4-codex-exec-20260910`](../stack-pbe-v4-codex-exec-20260910/README.md),
same tasks in the same order.

## Results

| arm | solved | memorised | shown-only | `aligned_matches` shortcut | do not run | thinking tok/task, median (max) | out tok, arm | per-task s, median (max) | artifact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gpt-6-astra` low | 24 (60.0%) | 16 | 100.0% | 1 → 23 real | 1 | 256 (1,311) | 14,437 | — (606 s arm) | `../stack-pbe-v4-codex-exec-20260910/astra_low_v4_sample40.json` |
| `gpt-6-astra` high | 23 (57.5%) | 15 | 95.0% | 1 → 22 real | 1 | 396 (2,787) | 22,480 | — (840 s arm) | `../stack-pbe-v4-codex-exec-20260910/astra_high_v4_sample40.json` |
| `claude-opus-5` low | **17 (42.5%)** | 6 | 57.5% | 1 → 16 real | 13 | 526 (15,747) | 95,070 | 11 (214) | `opus_low_v4_sample40.json` |
| `claude-opus-5` high | **20 (50.0%)** | 12 | 80.0% | 1 → 19 real | 5 | 1,400 (27,006) | 163,852 | 23 (341) | `opus_high_v4_sample40.json` |
| `claude-opus-5` xhigh | **25 (62.5%)** | 13 | 95.0% | 1 → 24 real | 1 | 1,962 (119,941) | 280,596 | 28 (1,457) | `opus_xhigh_v4_sample40.json` |

"Shown-only" is what a grader without held-out examples would have said. "Do
not run" counts candidates the machine rejects. The sample holds two
`aligned_matches` tasks; the shortcut column counts the solve that never
compares elements (no `eq?`), the head-of-first-list program the baseline
report recorded. Opus high and xhigh each solved the *other* `aligned_matches`
task with a zip-and-compare fold, which counts as real. `analyse.py` prints
these tables and the pairings below.

### Solves by witness length, and by who wrote the behaviour

| arm | 1–10 ops | 11–20 | 21–40 | 41+ | hand-written (v3's 40) | `gpt-5.6-sol`-written |
|---|---:|---:|---:|---:|---:|---:|
| `gpt-6-astra` low | 8/8 | 8/8 | 7/12 | 1/12 | 8/8 | 16/32 |
| `gpt-6-astra` high | 8/8 | 7/8 | 7/12 | 1/12 | 8/8 | 15/32 |
| `claude-opus-5` low | 8/8 | 6/8 | 3/12 | 0/12 | 8/8 | 9/32 |
| `claude-opus-5` high | 8/8 | 6/8 | 5/12 | 1/12 | 8/8 | 12/32 |
| `claude-opus-5` xhigh | 8/8 | 8/8 | 8/12 | 1/12 | 8/8 | 17/32 |

The authorship columns are confounded with length on this sample, as in
#123: seven of the eight hand-written tasks are 5–7-op witnesses.

## What the numbers say

**The knob is spent, and the curve rises.** Paired over the same 40 tasks,
high thinks more than low on 38 of 40 (median ratio 2.27×) and xhigh more than
high on 29 of 40 (1.28×); medians 526 → 1,400 → 1,962 tokens, against astra's
256 → 396. Solves go **17 → 20 → 25**, and the sets are nested: nothing solved
at low is lost at xhigh, nothing solved at high is lost at xhigh, and the one
flip (`drop_last`, 12 ops, solved at low, failed at high, solved at xhigh) is
the one-sample noise every row carries. So this is the measurement #123
wanted: on this corpus, more thinking buys more solves.

**What it buys is the 12–49-op band; the sample's long tasks do not move.**
All eight xhigh-over-low gains are `aligned_matches` (39 ops), `replace_value`
×3 (22), `positive_balance_prefix_count` (49), `concatenate_shorter_first`
(26), `drop_last` (12) and `cartesian_pair_sums` (15): 21–40 ops goes 3 → 5 →
8 of 12. Past 40 ops every arm solves 0 or 1 of 12, and the 1 is the same
49-op two-accumulator fold astra medium and high solved. On those twelve
tasks what effort does is convert answers that fail into answers that fit the
shown pairs: failed **8 → 3 → 1**, memorised **4 → 8 → 10**, solved 0 → 1 → 1
(over the whole sample: failed 17 → 8 → 2, do-not-run 13 → 5 → 1, memorised
6 → 12 → 13). The most expensive call of the study, `median_floor` (253-op
witness) at **119,941 thinking tokens and 24 minutes**, produced a
memorisation: a filter-and-count that reproduces the shown outputs and fails
the held-out set. At xhigh Opus fits 11 of the twelve and computes one
(`keep_ascending_pairs`, 80 ops, still fails); astra low fits all twelve and
computes one. Phase 2 of #123 showed those twelve came from behaviours astra
rarely solves while it solves 214 of the split's 840 long tasks, so whether
Opus reaches astra's long solves is a full-split question, not answered here.

**At the top of its range Opus solves the same tasks as astra.** xhigh against
astra high: agree on **38 of 40**, both solve 23, Opus alone 2 (`drop_last`,
where astra memorised, and the real `aligned_matches` solve, where astra
failed), astra alone 0. Against astra low: 37 of 40, Opus alone the two above,
astra alone the 59-op `farthest_from_target_last_tie` that Opus memorised. It
gets there on **12.5× the output tokens** (280,596 against astra high's 22,480
for the arm) and 4× the wall. Two families, one of them the corpus author's,
land on the same 23–25 of 40 once each spends what it will spend; on this
sample the difficulty is in the tasks. That is the cross-family point the
issue asked for, and it is one sample of 40 at the easy end of the split.

**Shown-only overstatement grows with effort.** 15 → 30 → 32.5 points at low /
high / xhigh (57.5 − 42.5, 80 − 50, 95 − 62.5), against astra's 40. At low,
Opus mostly fails outright (13 of 40 do not run, the luna pattern); at xhigh
it fits the shown pairs on 38 of 40 and 13 of those fits are coincidences of
the astra kind: `[ 0 1 sub lt? ] filter 0 [ drop 1 add ] fold` (count the
negatives) for `adjacent_sign_change_count`, `[ 0 eq? ] filter 0 [ drop 1 add
] fold` (count the zeros) for `axis_point_count`, `swap drop swap drop 1 sub`
for `largest_inside_interval`. The held-out protocol is again worth more the
harder the solver tries.

## Cost

120 calls, 161,811 input tokens (of which ~0.6k per call is Claude Code's
residual; no cache reads, each call is a fresh process below the cacheable
minimum), 539,518 output tokens of which 533,444 are thinking, on the
subscription. Per-arm wall 1,424 / 2,220 / 3,568 s (sums of per-call seconds);
the three ran concurrently in 59.5 minutes. The xhigh arm's per-task time is
28 s median with one 24-minute call.

## Caveats carried with every row

180 of 220 behaviours were written by `gpt-5.6-sol`, so a frontier number here
is corpus calibration, not a model ranking; median 8-op window overlap between
witnesses is 79.8%, so success is composition; the sample's median witness is
26 ops against the split's 40; one sample per task at the CLI's default
temperature, and same-task verdicts flip between runs (one flip in three arms
here, seven between two luna runs in #123).

## Phase 2, not started

The full 1,800-task split at one effort would say whether the same-set finding
holds across all 45 eval behaviours and, in particular, whether Opus reaches
the 214 long solves astra low made. At the phase-1 rates: xhigh is ~89 s per
task, so ~7.5 h wall on six shards and ~12.6M output tokens; high is ~56 s per
task, ~4.6 h and ~7.4M, but five solves behind on the sample. Whether either is
worth the plan's weekly share is the owner's call; `run_phase2.sh` in the codex
report directory is the template.

**Declined.** On closing #137 the owner said phase 2 will not run: benchmarking frontier models is not this project's goal. Do not propose a full-split frontier run unless asked.
