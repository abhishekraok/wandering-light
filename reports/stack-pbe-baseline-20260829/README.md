# First solver numbers on `stack_pbe_v3`, and what held-out examples change

Date: 2026-08-29 · [pbe-generator-behaviour-first (#67)](../../issues/67) · corpus `stack_pbe_v3.jsonl.gz`

Two things are being measured. What solves the corpus, and what the corpus's
hidden examples catch that the old protocol could not.

## The protocol

A solver sees **only the shown examples** of a task — 3 to 6 input/output pairs.
Whatever it returns is graded on that task's **8 held-out** pairs, which it never
saw. `experiments/stack_pbe_baseline.py` never passes the held-out set to a
backend, and `tests/test_stack_pbe_baseline.py` pins that by re-running BFS on a
task with its held-out fields stripped and requiring the same answer.

Two numbers come out of `wandering_light/stack_pbe_eval.py`:

* **solved** — right on data it was not given. The only number worth quoting.
* **shown-only** — what a grader without held-out examples would have reported.

`stack_pbe_v2` could not report the second at all. Its data came from a
seventeen-row table, so there was nothing left over to hold out, and every
stack-side solver number in the project is a shown-only number.

## BFS against a node budget

Breadth-first over the product behaviour of the shown examples, stopping at the
first program that fits them. Budgets are expansions, so 10,000 is comparable to
the figure `stack_pbe_v2` was quoted against.

The same forty eval tasks at three budgets, so the rows compare.

| budget | solved | shown-only | memorised | wall clock |
|---:|---:|---:|---:|---:|
| 1,000 | **0/40 (0.0%)** | 2.5% | 1 | 30 s |
| 10,000 | **3/40 (7.5%)** | 10.0% | 1 | 369 s |
| 100,000 | **11/40 (27.5%)** | 30.0% | 1 | 3,326 s |

Each 10x of budget costs 10x the wall clock and roughly triples the solves. The
memorisation count does not move: it is the same single task at every budget.

At 100,000 nodes — a hundred times the budget v2's 75.2% was quoted at — BFS
solves **every** five-op behaviour in the sample and almost nothing else:

| behaviour | witness ops | solved |
|---|---:|---:|
| `double_each` | 5 | 4/4 |
| `sum` | 5 | 3/3 |
| `keep_negative` | 5 | 3/3 |
| `last` | 6 | 1/2 |
| `scale_by` | 6 | 0/2 |
| `sum_both` | 6 | 0/7 |
| `keep_above` | 7 | 0/3 |
| `count_negative` | 12 | 0/4 |
| `maximum` | 16 | 0/6 |
| `abs_each` | 16 | 0/4 |
| `minimum` | 16 | 0/2 |

The same step function `deep-corpus-v1` shows: everything up to the depth the
budget reaches, nothing past it. The wall sits between 5 and 6 ops, and a 100x
budget increase moved it by one op. At a measured ~19–24x branching per step,
buying the next op costs roughly another 20x, and `span` at 35 ops is not on the
same scale as anything purchasable.

This is the property the old corpus lacked. Its 75.2% described search over an
enumeration of what search already reaches.

## A language model against the same forty tasks

`gpt-5.6-terra`, reasoning effort high, one sample per task, the same forty eval
tasks the BFS sweep used.

| solver | budget / effort | solved | shown-only | memorised |
|---|---|---:|---:|---:|
| BFS | 1,000 nodes | 0/40 (0.0%) | 2.5% | 1 |
| BFS | 10,000 nodes | 3/40 (7.5%) | 10.0% | 1 |
| BFS | 100,000 nodes | 11/40 (27.5%) | 30.0% | 1 |
| `gpt-5.6-terra` | effort high | **33/40 (82.5%)** | 90.0% | **3** |

Where they differ is the whole point. BFS stops dead between 5 and 6 ops; the
model does not notice the boundary:

| behaviour | witness ops | BFS @100k | `gpt-5.6-terra` |
|---|---:|---:|---:|
| `double_each` | 5 | 4/4 | 4/4 |
| `sum` | 5 | 3/3 | 3/3 |
| `keep_negative` | 5 | 3/3 | 3/3 |
| `scale_by` | 6 | 0/2 | 2/2 |
| `sum_both` | 6 | 0/7 | 5/7 |
| `last` | 6 | 1/2 | 1/2 |
| `keep_above` | 7 | 0/3 | 1/3 |
| `count_negative` | 12 | 0/4 | 4/4 |
| `maximum` | 16 | 0/6 | 4/6 |
| `abs_each` | 16 | 0/4 | 4/4 |
| `minimum` | 16 | 0/2 | 2/2 |

`abs_each` and `minimum` are 16-op programs with a nested `ifte` inside a
combinator body, at a search depth of roughly 10^20 nodes. The model writes them
directly. Whatever it is doing is not search over this basis.

## The whole eval split

Forty tasks is a small sample, and one of its numbers did not survive contact
with the full 440. All 440 eval tasks:

| solver | solved | shown-only | overstatement | memorised |
|---|---:|---:|---:|---:|
| BFS @ 10,000 nodes | 39/440 (8.9%) | 9.5% | +0.7 pts | 3 |
| `qwen3:30b-a3b` (40-task sample) | 10/40 (25.0%) | 27.5% | +2.5 pts | 1 |
| `gpt-5.6-terra`, effort high | **377/440 (85.7%)** | 88.4% | **+2.7 pts** | 12 |

**Correction to the 40-task figure.** On the sample the model's overstatement
looked like 7.5 points against BFS's 2.5. On 11x the data it is **2.7 against
0.7** — the ratio holds at roughly 4x, the magnitude does not. Three
memorisations out of forty was too thin to quote, and the 7.5 was noise around a
smaller true value. The direction is what survives: **the stronger solver
overstates about four times as much**, so holding examples back is worth more as
solvers improve, not less.

**Where the memorisations concentrate** is the more interesting half. All twelve
land on three behaviours — `maximum` 6, `minimum` 4, `last` 2 — and the first two
are exactly the `ifte`-inside-a-`fold` shape, the hardest structure in the
catalogue. The model's near-misses are not spread evenly; they cluster where the
program has a branch whose behaviour only shows up on particular data.

**A 30B local model lands on top of a 100,000-node search.** `qwen3:30b-a3b`
scores 25.0% against BFS's 27.5% on the same forty tasks, and like BFS it stops
early — 5- and 6-op behaviours only, nothing at 7+. The frontier model is about
3.4x either of them and shows no such wall. Whatever separates them is not
model size alone.

**One artifact worth naming.** At a 10,000-node budget BFS solves `sum` 39/40 and
`double_each` 0/40, though both witnesses are five ops. The budget does not
complete depth 5, so which five-op programs get found depends on the op ordering,
not on depth. The 100,000-node column does not have this problem; read the 10,000
row as order-dependent.

**What this says about the catalogue.** 85.7% is too high for this to work as an
LLM benchmark. That is a fact about the forty behaviours, not the generator, and
it is the number that should aim the v4 catalogue. The behaviours that survive
are specific: `maximum` 27/40, `minimum` 29/40, `keep_above` 29/40,
`count_negative` 31/40 — branch-bearing and multi-stage — while every 5-op
elementwise behaviour is at or near 40/40. v4 wants many more of the former.

## What the held-out examples caught

One memorisation at every budget, and it is the same task. BFS returns
`uncons drop i` for a `last` task whose three shown examples are all
**two-element** lists:

```
shown            [ (9 1) ]  -> [ 1 ]      [ (-2 -2) ] -> [ -2 ]   [ (4 -7) ] -> [ -7 ]
held out         [ (-9 -4 -2 -3 7 6) ] -> want [ 6 ]   got [ -4 -2 -3 7 6 ]
```

The program drops the head and *runs the one-element tail as a program*, which
returns the last element only when the list has length two. On a six-element list
it leaves the tail. Under the old protocol this is a solve; here it is a failure,
which is what it is.

The overstatement is small on this corpus — 2.5 points at both budgets — but it
is not zero, and it appeared in the first forty tasks anyone graded.

**Part of it is a generator hole, not solver weakness.** 24 of the 1,480
list-bearing tasks (1.6%) show only one list length across all their shown
examples, almost all of them three-example tasks. Requiring at least two distinct
list lengths among the shown examples would close it, and belongs in the next
corpus rather than a silent regeneration of this one.

## Not measured

**No language model has been run.** No API credentials are present on this
machine and no local model is pulled, so the arm that would say whether these
behaviours are *hard* or merely *unfamiliar to search* is missing. The backend is
written and tested — response parsing has its own tests — and needs only a key.
Until then the only claim here is about search.

`solver.py` has OpenAI, Gemini and Ollama generators and **no Anthropic one**.

## Reproduce

The corpus payload is not in git. Fetch it first; the tracked manifest pins the
digests the download is checked against.

```bash
uv run python -c "from wandering_light.corpus_hub import fetch_corpus; fetch_corpus('wandering_light/evals/data/stack_pbe_v3/manifest.json')"

uv run python experiments/stack_pbe_baseline.py --backend bfs --budget 10000 --split eval
uv run python experiments/stack_pbe_baseline.py --backend bfs --budget 100000 --split eval --tasks 40
uv run python experiments/stack_pbe_baseline.py --backend llm --provider gemini --model gemini-2.5-flash --tasks 40
```
