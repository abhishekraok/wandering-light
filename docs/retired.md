# Retired: the TypedList eval harness and the proposer

Everything new is written against the stack machine (#42). This records what is
retired, what is deliberately kept, and why — so the distinction does not have
to be rediscovered.

## Retired

Not maintained, not extended, and not a basis for new work.

| | |
|---|---|
| `wandering_light/evals/` | `run_evaluation`, `evaluate_solver`, `create_data`, and the committed corpora under `evals/data/` other than `stack_pbe_v2.jsonl.gz` |
| `wandering_light/training/` | SFT and GRPO training for the `TypedList` solver, including the dual proposer/solver loop |
| `wandering_light/solver.py` | `TrajectorySolver`, `create_bfs_solver`, `create_token_solver`, `TrainedLLMTokenGenerator` |
| `wandering_light/proposer_pilot/solve_rater.py` | scores tasks with a `TypedList` solver |

Closed with this: #22 (difficulty from shortest path), #25 (comparable budgets),
#23, #24, #28 (proposer diversity and the curriculum loop). #27 is the decision
to drop the proposer; it stays open because the graph-plus-value-function
direction that replaces it has not been built.

### Why

**The corpora were solved by search.** BFS reached 100% on
`random_inputs_500_shortest_v1` at depth 4 in about 5 ms per task. That is not
a fixable flaw. A corpus built by expanding to depth *d* is by construction one
that search reaches at depth *d*, so forward generation cannot produce problems
that defeat forward search. Every learned solver measured against those sets was
losing to a free exact opponent.

**The replacements exist and are stronger.** `stack_pbe_v2.jsonl.gz` carries
1,857 tasks certified shortest to depth 6, records the basis each label was
certified against, and ships with a measured baseline in
`reports/stack-machine-basis/bfs-baseline.md`. The metric is expansions per
solved task, because unlimited-budget solve rate saturates at 100% for the
reason above.

## Kept

Still imported, still tested, still correct. Do not delete these while removing
the above.

| | why |
|---|---|
| `typed_list.py`, `executor.py` | the `wl-core-v1` comparison runs on them |
| `basis_set.py`, `function_def.py`, `common_functions.py` | load and execute the 118-function basis |
| `proposer_pilot/graph.py` | `TrajectoryGraph.expand`, used for exhaustive old-basis expansion |

`experiments/stack_machine_monoid.py`, `stack_machine_cost.py` and
`stack_machine_bfs.py` all import these. They produce the numbers behind #42's
conclusions — that the two bases generate new behaviour at 5.77× and 5.87× per
op, and that programs are a median 2.00× longer on the stack machine. Those
claims are only re-checkable while the old basis still runs.

## If you are removing the retired code

It is severable but not small: 23 of 48 test files reference
`evals`/`training`/`solver`, and five scripts outside them import from those
modules —

- `experiments/basis_library_pilot.py`
- `experiments/relabel_shortest_data.py`
- `experiments/generate_deep_corpus.py`
- `wandering_light/proposer_pilot/solve_rater.py`

Delete modules together with their tests in one change, and expect the suite to
lose roughly half its files. Do it as its own PR, not folded into feature work.
