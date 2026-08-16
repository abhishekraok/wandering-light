# Proposal: type-masked policy/value encoder over BFS

Detailed design for #33. The issue holds the decision; this file holds the execution detail.
Written to be followed by an agent end to end.

## 1. The baseline that reframes everything

Measured on `random_inputs_500_shortest_v1` (480 certified specs), current `wl-core-v1` basis, BFS
guarded against executor exceptions:

| depth | solved | ms/task |
|------:|-------:|--------:|
| 1 | 183/480 = 38.12% | 0.3 |
| 2 | 384/480 = 80.00% | 1.3 |
| 3 | 471/480 = 98.12% | 3.4 |
| 4 | **480/480 = 100.00%** | 5.0 |

Certified distances in that file: 183 at 1, 201 at 2, 87 at 3, 9 at 4, none beyond.

The `rl-6k-with-lp` LLM scores 83.96% at budget 1 and 85.42% at budget 16 on the same file.
**Exhaustive search beats it outright, exactly, at ~5 ms per task with no model.** Any learned
solver evaluated on this corpus is competing against a free, perfect opponent.

This explains two earlier observations. Best-of-16 was flat (+1.46 points for 16x compute) because
the model is not sampling-limited, it is simply worse than search. And the ~43% on the broad-input
corpus versus ~77% on filtered `random_inputs_500` is distribution overfitting, which search does
not do at all.

## 2. Where a learned policy can actually pay

BFS cost grows ~5.8x per depth level. Full expansion from a single `int` root, measured:

| depth | states | wall clock |
|------:|-------:|-----------:|
| 3 | 513 | 0.1 s |
| 4 | 2,794 | 0.5 s |
| 5 | 16,015 | 2.7 s |
| 6 | 93,776 | 14.9 s |
| 7 | ~550k (extrapolated) | ~90 s |

A *failed* search pays full expansion cost, so a task at distance 7 costs BFS roughly 90 seconds.
That is the regime where a policy that prunes the frontier is worth having — and where the value
head has a real job, because it can order the frontier.

**The corpus has zero tasks past distance 4.** So the prerequisite for this whole experiment is a
task distribution with long shortest paths. That is work item 0, and it is not optional: without it
every arm below ties at 100% and the experiment measures nothing.

## 3. Blockers to clear first

**Bug: BFS crashes on runtime executor failures.** `wandering_light/solver.py:289` calls
`executor.execute(func, current_list)` inside `_bfs_search` with no exception handling. Type
compatibility is checked at the container level (`item_type == func.input_type_cls()`), but a
function can accept `builtins.list` and still raise on the contents — `list_median` on a list of
strings raises `TypeError` and kills the entire search. This makes BFS unusable at depth >= 2 on
heterogeneous inputs, which is why the table in section 1 required a guarded reimplementation.
Fix: treat a raising expansion as an absent transition, matching `TrajectoryGraph.expand`.

**Budget semantics.** See #25. Define budget as **node expansions** for every arm and report it,
so learned and search solvers are comparable on one axis.

## 4. Work item 0: a corpus with long shortest paths

Required properties:

- Certified or bounded shortest distance per task, not requested random-walk length. Nominal length
  is a bad label: 84% of nominal length-5 walks admit a shorter subsequence (see #22 and the
  collapse section of `reports/basis-library-20260815/README.md`).
- Non-trivial mass at distances 5 through 8.
- Broad input distribution, following `experiments/basis_library_pilot.py` (boundary values,
  Unicode, binary, larger containers, stepped ranges), with identity and constant outputs rejected.

Method. Generating long *certified* tasks by relabeling random walks will not work — collapse
removes almost all of them. Generate forward instead: expand a state graph breadth-first to depth
`d` from many roots using `TrajectoryGraph.expand` (which already returns
`ExpansionResult.certified_depth`), then emit `(root, node_at_depth_k)` pairs. A node first reached
at BFS depth `k` in a complete expansion is *certified* at distance `k` by construction, with no
extra search. This is the cheapest correct route to the data and it reuses existing code.

Budget note: full expansion to depth 6 is ~94k states and ~15 s per root, so a few hundred roots is
an overnight job on the desktop and gives a large corpus. Depth 7+ needs sampling rather than
exhaustive expansion.

## 5. Model design

**Input.** `[CLS] serialize(src) [SEP] serialize(dst) [SEP]` using `TypedList.to_string()`, which is
already canonical JSON: `{"type": "builtins.int", "items": [1, 2, 3]}`.

**Tokenizer.** Byte-level or character-level, trained from scratch. Do **not** use pretrained BERT
weights or WordPiece: the pretraining encodes English semantics that are irrelevant here, and
WordPiece shreds numerals. What the BERT *architecture* contributes is segment embeddings for the
src/dst split, which is a genuine fit.

**Heads.**
- *Policy*: `|basis|`-way logits from `[CLS]`. Type-incompatible entries masked to `-inf` **during
  training as well as inference**, so no gradient is spent on illegal actions.
- *Value*: `(K+1)`-way softmax over distance-to-target rather than scalar regression. The entropy of
  that distribution gives the README's confidently-solvable / confidently-far / unsure trichotomy
  directly.

**Size budget.** Under 10M parameters. If it needs a GPU to run inference, the efficiency argument
is already lost.

**Why type masking matters.** Measured on `wl-core-v1` and the 118,730 certified paths:

| quantity | value |
|---|---|
| unmasked action space | 118 = 6.88 bits/choice |
| mean legal actions after source-type masking | 9.8 = 3.30 bits |
| removed exactly, for free | 3.58 bits, 92% of the action space |
| `(src_type -> dst_type)` pairs with one candidate | 27 of 45 (60%) |
| 1-step tasks decided by type lookup alone | 10,234/26,614 = 38.5% |

`BFSPredictor._is_type_compatible` already implements the predicate.

## 6. Arms

All evaluated at a **fixed node-expansion budget**, on the long-path corpus from section 4.

| arm | description | isolates |
|---|---|---|
| **B0** | BFS to exhaustion at the budget | the real baseline |
| **B1** | existing `rl-6k-with-lp` LLM | historical context |
| **A1** | no learning: unique `(src_type -> dst_type)` lookup, else fail | how much is free |
| **A2** | encoder classifier, single shot, no mask | does the architecture work at all |
| **A3** | A2 + type-masked logits | value of the mask |
| **A4** | A3 + closed-loop execution | value of closed loop |
| **A5** | A4 + value-ordered best-first search | value of the value head |
| **H** | **hybrid**: at each step run BFS to depth 3; if it hits, take it; else let the policy pick one step and recurse | the proposed system |

**H is the headline arm.** Depth-3 BFS costs ~3.4 ms and covers everything within 3 of the current
state, so the policy is only ever asked to make progress on the part search cannot reach cheaply.
That is the correct division of labour: search handles the tail, the network handles the horizon.

## 7. Data and labels

**Step-level examples.** Expand each certified path by executing prefixes: a path of length L yields
L examples of `(current_state, target, next_function, remaining_distance)`. On
`induction_shortest_v1` that is 118,730 paths x mean 2.04 = 242,647 examples — useful for a smoke
test, but too short-pathed for the real experiment. The section 4 corpus is the training set.

**Ties are the correctness trap.** Certified data gives *a* shortest path, not the set of optimal
first actions, and this DSL converges heavily (`double` and `square` agree on `[2,2,2]`). Training
cross-entropy against only the recorded action penalizes correct alternatives.

- Compute the optimal action set: action `a` is optimal at distance `d` iff
  `dist(f_a(s), target) == d - 1`. In a complete forward expansion these distances are already
  known, so this is a lookup rather than a search.
- If a single-label loss is used for speed, **evaluation must still give any-optimal-action credit**,
  and both numbers reported.

## 8. Metrics

- **Solve rate vs node-expansion budget**, as a curve, per arm. Not a single number — the whole
  question is where the learned arms cross B0.
- **Solve rate split by certified distance** (1..8). Closed loop and search only help at distance
  >= 2; averaging over a distance-1-heavy set hides the effect.
- Mean solution length against certified optimum.
- Parameter count and median wall-clock per solve.
- Value head: **ECE and a reliability diagram** on held-out certified distances. Calibration, not
  accuracy. Rare opportunity: the ground truth is proven, not estimated.

## 9. Acceptance and kill criteria

Pre-register before running.

- **H beats B0 at equal node-expansion budget on distance >= 6 tasks.** This is the experiment. If
  it fails, the learned policy adds nothing over search and the direction is closed.
- **A3 >= A2** by a clear margin, or type masking is not the win the numbers suggest.
- **A4 > A3** on distance >= 2 tasks specifically.
- **Value head ECE < 0.05** held out, reliability diagram attached.
- **Kill criterion 1:** if A1 (zero learning) lands within 5 points of B1 anywhere, report that
  benchmark weakness first and stop.
- **Kill criterion 2:** if the section 4 corpus cannot be built with meaningful mass at distance
  >= 6, stop and report that. It means the DSL cannot pose the question, which is a finding in
  itself and is consistent with the retirement reasoning in #27.

## 10. Implementation pointers

- New predictor subclasses `FunctionPredictor` and implements
  `predict_functions_batch(problems, available_functions) -> list[FunctionDefList]`. The closed loop
  and the interleaved BFS both live **inside** that method — construct an
  `Executor(available_functions)` and step. No change to `TrajectorySolver` or the eval harness.
- Register in `get_solver_by_name` (`wandering_light/solver.py:686`) so
  `run_evaluation.py --solver_names=[...]` works unchanged.
- Load the basis via `basis_set.load_basis_set("default")` -> `wl-core-v1`, then `.as_function_set()`.
  Record the resolved basis ID and digest in every artifact; the loader fails closed on mixed
  provenance.
- Action indices must be **stable function IDs** (`basis_set.basis_function_id`), never list
  positions, so a basis edit cannot silently permute the label space.
- Reuse `TrajectoryGraph.expand` / `ExpansionResult` from `wandering_light/proposer_pilot/graph.py`
  for corpus generation; it already does structural state dedup and tracks certified depth.

## 11. Scope

Mean certified path length on the existing corpus is 2.04 and BFS solves it perfectly in 5 ms, so
nothing here revives library learning or the self-play program — see #27 and the basis report's own
conclusion that the compression pilot found three no-ops/aliases and one macro.

What this can produce is one clean result: **a sub-10M-parameter policy that beats exhaustive search
at equal compute in the depth regime where search becomes expensive.** That is a real claim about
learned search guidance, it is falsifiable at small scale, and it fits on the available hardware.
Anything beyond that is out of scope.
