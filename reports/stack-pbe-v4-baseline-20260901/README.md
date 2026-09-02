# Baselines on `stack_pbe_v4`: what solves the 220-behaviour corpus

Date: 2026-09-01 · [rerun-baselines-v4 (#85)](../../issues/85) · corpus
`stack_pbe_v4` (Hub revision `00c2a999`), 1,800-task eval split over 45
behaviours and 44 skeletons. Every number here is a **v4** number; the v3 rows
in `docs/STATUS.md` stay as they are.

## The prediction, recorded before the numbers

From the issue, verbatim: *"On v3, `gpt-5.6-terra` failed only on
branch-bearing, multi-stage behaviours while every 5-op elementwise behaviour
was at or near 40/40. v4 is 76% branch-bearing with a median witness of 43 ops.
If 85.7% does not fall substantially, that is evidence the catalogue is still
too easy, and the finding is about the corpus rather than the solver."*

One change to the protocol, made before any run: the frontier model is
**`gpt-5.6-luna`, not `gpt-5.6-terra`**, on cost. A luna-on-v4 number cannot be
differenced against terra's v3 85.7%, so luna was also run on v3's 40-task
seed-17 sample as a matched anchor. The prediction is tested luna against luna.

## The caveat that sits beside every number

On `mark-28` a witness's median share of 8-op windows that *another* witness
also contains is **79.8%**, and eval witnesses share **76.8%** of theirs with
train witnesses (`docs/STATUS.md`, "What the basis costs in shared code"). A
solver can assemble much of an unseen witness from fragments of seen ones.
**Success on v4 is evidence of composition, not invention.**

## The protocol

Unchanged from the v3 report. A solver sees **only the shown examples** of a
task (3–6 pairs) and is graded on its **8 held-out** pairs by
`wandering_light/stack_pbe_eval.py`. Two numbers: **solved** (right on data it
was not given — the only one worth quoting) and **shown-only** (what a grader
without held-out examples would have said). One sample per task.

## The 40-task sample is easier than the split

The seed-17 sample of 40 eval tasks covers 27 behaviours. Its median witness is
**26 ops** against the full split's **40** (min 5, max 253 in both). Sample
numbers here overstate what the full split would give, and the BFS rows below
show by how much where both were run.

## BFS against a node budget, two arms

Breadth-first over the product behaviour of the shown examples, stopping at the
first program that fits them; budgets are expansions; `max_depth` 8.

Two arms. **Unpruned** is the machine as it stands: under an open `[` every
one of the 28 ops is legal, because capture records a symbol without checking
it. **With guard** applies the search-side predicate from
[prune-open-mark-states (#70)](../../issues/70) to each successor state —
`blocked(stack, remaining, "both")` from `wandering_light/quotation_guard.py`,
with `remaining` the ops left under the depth cap — and drops states whose
open body can never run: a body that is ill-typed as a program ("type"), or a
state with more open marks than ops left to close them ("close"; sound only
because every corpus target is mark-free, which the script asserts). A third
check on how deep a body reaches was found unsound during #70's review — `cat`
widens a list in one op — and removed before merge. The guard never prunes a
witness prefix (0 of 26,013 on v4) and never loses the *shortest* fit, which
is what a first-fit solver needs; it can drop longer dead-code variants, so it
is not the tool for an exhaustiveness count. It changes what a node budget
*reaches*, never which programs exist at a depth. Same budgets, same seed,
same machine; guarded wall clock is reported as measured, since the predicate
taxes every successor.

| budget | tasks | unpruned | shown-only | memorised | wall clock | with guard | shown-only | memorised | wall clock |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | full 1,800 | **80/1800 (4.4%)** | 4.9% | 8 | 2,116 s | **80/1800 (4.4%)** | 4.9% | 8 | 2,154 s |
| 10,000 | full 1,800 | **120/1800 (6.7%)** | 7.9% | 22 | 22,803 s | TBD_G10K | | | |
| 100,000 | sample 40 | **6/40 (15.0%)** | 20.0% | 2 | 3,917 s | **6/40 (15.0%)** | 20.0% | 2 | 3,624 s |

**Pre-registered before the guarded arm ran.** At 1,000 nodes the unpruned
arm solves exactly the two first-order 5-op behaviours and 0/40 on the four
5-op behaviours that open a `[` (next section). The guard should flip those
four at the same budget — that was the prediction as first written. The #70
session's measurement on v3 (states per task, depth 4: body check 1.30x,
close 4.29x, both 4.43x; depth 6 both 17.0x) says it probably will not: at
body length 1–2 the prune rate is 36% or less, and if the four do flip it will
be the close check doing it. The prediction stays as written; the miss, if it is one,
is the result.

**Outcome: a miss, and a clean one.** At 1,000 nodes the guarded arm returns
the *same candidate with the same verdict on all 1,800 tasks* — 80 solved, 8
memorised, the same two first-order behaviours, the four quotation builders
still 0/40 — and costs 1.8% more wall clock for it. A budget of 1,000
expansions never reaches a body long enough for the type check to fire, and
never reaches the last ply where the close check bites. The guard does not
change what 1,000 nodes buy on this corpus.

**At 100,000 on the sample, the same again:** identical candidate and verdict
on all 40 tasks, the three 6–7-op behaviours (`keep_below`, `negate_each`,
`n_minus_each`) still 0/1 each. Wall clock 3,624 s against 3,917 s, but the
unpruned run shared the machine with five other jobs and this one with two,
so nothing is attributed to the guard there. The arithmetic says why the
outcome does not move: BFS finds a 6-op program when it pops that program's
depth-5 ancestor, and #70's counts put the depth-5 ball at ~980,000 states
unpruned and ~105,000 guarded (v3 roots; v4's data stacks differ but not by
an order of magnitude). A 100,000-expansion budget is at the edge of
finishing depth 5 even guarded. So the guard's first *visible* step on this
corpus lies between 100,000 and roughly 1,000,000 expansions — a budget
sweep, not run here. Within the v3 budgets the two arms are the same floor.
TBD_G10K_NARRATIVE

**At 1,000 nodes the 80 solves are exactly two behaviours at 40/40** —
`push_back` (4 ops, `nil swap cons cat`) and `rotate_left` (5 ops,
`uncons nil swap cons cat`) — and nothing else. Four other behaviours have
5-op witnesses and score **0/40**: `increment_each` (`[ 1 add ] map`),
`keep_negative` (`[ 0 lt? ] filter`), `keep_zeros` (`[ 0 eq? ] filter`),
`reverse` (`nil [ cons ] fold`). The difference is not length. The two that
solve are first-order; the four that do not open a `[`, and under an open mark
every one of the 28 ops is legal (the measurement in
[prune-open-mark-states (#70)](../../issues/70)). A 5-op quotation-building
program costs more nodes than a 5-op first-order one, and 1,000 is below the
line.

Memorisations at 1,000 nodes: `concatenate_shorter_first` four times, always
`cat` or `swap cat` — shown examples that happened to keep the shorter list on
one side; `median_floor` once via `uncons swap drop`, the second element. The
v4 analogue of v3's `uncons drop i` on `last`.

At 100,000 nodes on the sample: **5/5 at ≤5 ops, 0/9 at 6–15, 1/26 at 16
and above** — the v3 step function again, wall between 5 and 6 ops, and the
100x budget over 1,000 bought the four quotation-building 5-op behaviours
(`keep_negative`, `keep_zeros`, `reverse`, and `rotate_left`'s 3-op fit
`uncons quote cat`) and nothing longer. Wall clock 3,917 s against v3's 3,326.

**The one solve past 16 ops is a corpus defect, not a search result.**
`aligned_matches` carries a 39-op witness and BFS solved one of its two tasks
with `drop uncons swap drop quote` — head of the first list, as a singleton —
which passed all 8 held-out examples. Look at the inputs: in **every** example
of both tasks the two lists agree at position 0 and the second list's
position 1 is `-9`. The sampler, `_partly_matching_lists` at
`wandering_light/stack_pbe.py:370`, sets `right[0] = left[0]` to guarantee a
match and `right[1] = next(v for v in SMALL if v != left[1])` to guarantee a
miss — and that `next` is deterministic, the first value of `SMALL` that is
not `left[1]`, so it is `-9` almost every time. No other behaviour uses this sampler. The
behaviour collapses to
"first element" on any example where nothing later coincides, and on one
task's 13 examples nothing did. Eight
hidden examples do not help when the sampler bakes the shortcut into all of
them. This is the class of defect the held-out protocol exists to surface, and
it wants a sweep over the other 219 behaviours' samplers — filed on the issue,
not fixed here.

At 10,000 nodes on the full split — the like-for-like against v3's 8.9% at
the same budget — **6.7%**, and the 40 solves gained over 1,000 nodes are one
behaviour: `reverse`, `nil [ cons ] fold`, at 40/40. It is the first
quotation-building behaviour any search has solved on v4. The other three
5-op quotation builders (`increment_each`, `keep_negative`, `keep_zeros`)
stay at **0/40** at 10x the budget; both first-order 5-op behaviours were
already at 40/40 at 1,000. So on the full split: **120/240 at ≤5 ops, 0/1,560
at 6 and above.** Wall clock 22,803 s, 6.3 h, against 2,116 s at 1,000 — the
10x budget costs 10.8x because almost every task now exhausts it.

Memorisations rose from 8 to **22** with the budget, fourteen of them on
`positive_majority_three`, whose output is 0 or 1: `mul div dup mul`,
`div over add div`, `add div dup mul` and so on, arithmetic that lands on the
right bit for 3–6 shown examples and not for 8 hidden ones. A bigger budget
buys more coincidence. At 10,000 nodes the shown-only grader would overstate
by 1.2 points; at 1,000 by 0.4.

## `qwen3:30b-a3b`, the free local arm

One RTX 4090, 300 s per-request deadline — wall clock, not a token cap, per
the v3 report's finding that a 4,096-token cap truncates this model mid-thought
— one sample per task.

| corpus | solved | shown-only | memorised | wall clock |
|---|---:|---:|---:|---:|
| v3 sample (v3 report) | 10/40 (25.0%) unbounded; 9/40 (22.5%) under the deadline | 27.5% | 1 | ~32 min |
| **v4 sample** | **6/40 (15.0%)** | 15.0% | 0 | 3,381 s |

Solves: `keep_negative`, `reverse`, `negate_each` at 1/1, `keep_zeros` 1/2,
`drop_last` 2/3 — **4/8 at ≤7 ops, 2/6 at 8–15, 0/26 at 16 ops and above.**
No request hit the deadline; the wall clock (85 s per task against ~48 on v3)
is the model thinking longer and still failing. 30 of the 34 failures are
programs that do not run. No memorisation and no overstatement: what it gets,
it gets outright.

## `gpt-5.6-luna`, and the prediction

### Anchor: luna on v3's forty tasks

Same forty tasks, same seed, as the terra sample in the v3 report.

| solver | solved | shown-only | overstatement | memorised | wall clock |
|---|---:|---:|---:|---:|---:|
| `gpt-5.6-terra` (v3 report) | 33/40 (82.5%) | 90.0% | +7.5 pts | 3 | — |
| `gpt-5.6-luna` | **30/40 (75.0%)** | 85.0% | +10.0 pts | 4 | 622 s |

| behaviour | ops | terra | luna |
|---|---:|---:|---:|
| `double_each` | 5 | 4/4 | 4/4 |
| `sum` | 5 | 3/3 | 3/3 |
| `keep_negative` | 5 | 3/3 | 3/3 |
| `scale_by` | 6 | 2/2 | 1/2 |
| `sum_both` | 6 | 5/7 | 6/7 |
| `last` | 6 | 1/2 | 1/2 |
| `keep_above` | 7 | 1/3 | 2/3 |
| `count_negative` | 12 | 4/4 | 2/4 |
| `maximum` | 16 | 4/6 | 2/6 |
| `abs_each` | 16 | 4/4 | 4/4 |
| `minimum` | 16 | 2/2 | 2/2 |

Luna is 3 tasks behind terra on this sample, all of it on `maximum` and
`count_negative`; it writes the nested-`ifte` 16-op programs (`abs_each`,
`minimum`) as readily as terra does. **75.0% is the v3 number the v4 result is
measured against.**

### luna on v4's forty tasks

| solver | corpus | solved | shown-only | overstatement | memorised | wall clock |
|---|---|---:|---:|---:|---:|---:|
| `gpt-5.6-luna` | v3 sample | 30/40 (75.0%) | 85.0% | +10.0 pts | 4 | 622 s |
| `gpt-5.6-luna` | **v4 sample** | **15/40 (37.5%)** | 45.0% | +7.5 pts | 3 | 2,286 s |

**The prediction's null branch is ruled out.** Luna against luna, same
protocol, same seed: **75.0% → 37.5%**, halved. The catalogue moved. And the
sample is the *easy* end of the split (median witness 26 against 40), so the
full-split number would be lower still.

Where the solves land, by witness length:

| witness ops | solved | detail |
|---:|---:|---|
| ≤7 | 6/8 | `keep_negative`, `keep_zeros`, `reverse`, `rotate_left`, `negate_each` all solved; `keep_below`, `n_minus_each` 0/1 |
| 8–15 | 3/6 | `min_pair` 2/2, `drop_last` 1/3, `cartesian_pair_sums` 0/1 |
| 16–39 | 5/12 | `sum_negative` 2/2, `replace_value` 2/3, `aligned_matches` 1/2; `prefix_sums`, `concatenate_shorter_first`, `nearest_interval_endpoint_each`, `positive_majority_three` all 0 |
| 40–99 | 1/13 | `largest_inside_interval` 1/2; eight other behaviours 0 |
| 253 | 0/1 | `median_floor` |

The model's wall is near **40 ops**: 14/26 below it, 1/14 at or above. And
one of the 15 is not a solve at all: luna's `aligned_matches` answer is
`drop uncons swap drop nil swap cons` — the same head-of-first-list shortcut
BFS found (see the corpus defect under BFS above). Counting only behaviours
actually computed, **14/40 (35.0%)**. v3's
longest witness was 35 ops and luna wrote its 16-op nested-`ifte` programs
freely; this sample has 14 tasks past anything v3 contained and the model gets
one of them.

Memorisations, 3. `axis_point_count` and `adjacent_sign_change_count` each
fell to a filter-and-count that matched the shown examples by coincidence.
`keep_ascending_pairs` is the purest case in the project so far: the candidate
branches on the input's sum and **pushes a hard-coded list** matching one shown
output (`nil 1 1 add 1 add ... cons`). A shown-only grader scores it as solved.

How it fails also changed. Of the 22 non-solves, **18 are programs that do not
run** — a guard refuses on at least one input — and only 4 run and compute the
wrong thing. On v3 the split was 5 of 10. On the longer witnesses the model is
losing the stack, not the algorithm.

Wall clock 2,286 s against 622 s on v3 — 3.7x per task at the same effort
setting, so the model is also reasoning longer on the harder corpus.

## Reproduce

Corpus payloads are not in git; the tracked manifests pin the digests.

```bash
uv run python -c "from wandering_light.corpus_hub import fetch_corpus; fetch_corpus('wandering_light/evals/data/stack_pbe_v4/manifest.json')"
uv run python -c "from wandering_light.corpus_hub import fetch_corpus; fetch_corpus('wandering_light/evals/data/stack_pbe_v3/manifest.json')"

R=reports/stack-pbe-v4-baseline-20260901
uv run python experiments/stack_pbe_baseline.py --backend bfs --budget 1000   --split eval            --out $R/bfs_v4_1k_full.json
uv run python experiments/stack_pbe_baseline.py --backend bfs --budget 10000  --split eval            --out $R/bfs_v4_10k_full.json
uv run python experiments/stack_pbe_baseline.py --backend bfs --budget 100000 --split eval --tasks 40 --out $R/bfs_v4_100k_sample40.json
# Guarded arm, run 2026-09-02 against main at b142b3e (guard-open-quotation-states #97 merged)
uv run python experiments/stack_pbe_baseline.py --backend bfs --budget 1000   --split eval --guard both            --out $R/bfs_v4_1k_full_guard.json
uv run python experiments/stack_pbe_baseline.py --backend bfs --budget 10000  --split eval --guard both            --out $R/bfs_v4_10k_full_guard.json
uv run python experiments/stack_pbe_baseline.py --backend bfs --budget 100000 --split eval --guard both --tasks 40 --out $R/bfs_v4_100k_sample40_guard.json
OLLAMA_BASE_URL=http://localhost:11434 \
uv run python experiments/stack_pbe_baseline.py --backend llm --provider ollama --model qwen3:30b-a3b --tasks 40 --split eval --out $R/qwen3_v4_sample40.json
uv run python experiments/stack_pbe_baseline.py --backend llm --provider openai --model gpt-5.6-luna --effort high --tasks 40 --split eval --out $R/luna_v4_sample40.json
uv run python experiments/stack_pbe_baseline.py --backend llm --provider openai --model gpt-5.6-luna --effort high --tasks 40 --split eval --corpus wandering_light/evals/data/stack_pbe_v3 --out $R/luna_v3_sample40.json
```

Seed 17 throughout (the script default). All runs on one machine: unpruned
BFS, ollama and luna on 2026-09-01, the guarded BFS arm on 2026-09-02 once the
guard was on `main`. BFS on CPU, ollama on one RTX 4090, luna through the
OpenAI API. Raw JSONs
with every candidate and verdict sit beside this file.

**One trap, since removed at its source.** When these ran, the shell exported
`OLLAMA_BASE_URL` pointing at an unreachable host (the export has since been
deleted), and `solve_with_llm` scores a `ConnectionError` as an unsolved task
rather than failing the run — so the first ollama attempt returned a
clean-looking 0/40 made entirely of connection failures. The override in the
command above is what the run actually used and is harmless on a clean shell.
The durable lesson is the second half: a zero from the ollama arm should be
checked against stderr before it is believed.
