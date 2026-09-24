# Meeting pairs through the 10-draw gate (#169, steps 1 and 2)

When the beam reaches a closed state it already kept by another route, it
holds two programs to one state: `first`, kept, and `second`, discarded, never
shorter than `first` (seen-set-collisions, #164). Walked back to their shared
prefix, the two give a next-op label at the divergence state with no solver
call: `op_first` reached the met state in no more ops than `op_second`. The
label is worth something only if the two programs compute the same function;
on the k shown examples they can agree by accident (`[ over lt? ]` vs
`[ swap lt? ]`, #164), and that pair taught as redundancy is a shortcut
(stack-debugger, #115).

`experiments/seen_set_collisions.py` now writes every closed cross-path
meeting as a pair with its divergence walk and tail shape (`commutation`: the
same ops in another order; `insertion`: `second` is `first` with ops
inserted; `other`). `experiments/meeting_pairs_gate.py` replays each pair on
10 fresh draws from the behaviour's sampler, shown inputs excluded, and keeps
it where the two stacks agree on every draw both programs accept (`pass`);
one disagreement is `fail`; fewer than 5 informative draws is `inconclusive`.

Cell: the world arm `1m-s0-e5` (`checkpoints/world-arm-20260922/policies/1m-s0-e5`)
on the #101 cohort (300 tasks, 43 behaviours), `BEST`'s search (beam 64 x
depth 48, guard both, state cap 16,384), library off, CPU, 4 threads,
09:49-10:08 PDT 2026-09-24, code at `d6771c5`. The search is deterministic and
reproduced the #164 cell to the count (5,127 closed cross-path meetings in 288
tasks; [search-summary.md](search-summary.md)). Per-pair verdicts in
[gate.jsonl.gz](gate.jsonl.gz), the summary in [gate.md](gate.md) / [gate.json](gate.json).

## Results

| pairs | n | pass | fail | pass rate of judged |
|---|---:|---:|---:|---:|
| all | 5,127 | 4,440 | 686 | **86.6%** |
| same-layer (equal length, no preference) | 1,172 | 1,086 | 86 | 92.7% |
| earlier-layer (second longer) | 3,955 | 3,354 | 600 | **84.8%** |
| earlier-layer / insertion | 1,317 | 987 | 329 | 75.0% |
| earlier-layer / other | 2,638 | 2,367 | 271 | 89.7% |

One pair was inconclusive. By the length of `second` at the meeting: 1-5 ops
98.4% pass (2,411 pairs), 6-9 89.0% (292), 10-19 82.5% (623), 20-47 68.7%
(1,426), 48 84.0% (374). Insertions that pass insert a median of 6 ops; those
that fail, 18: a long inserted chunk is a function that happens to agree on
the shown inputs, not a no-op.

Among the 3,354 passing earlier-layer pairs the shorter route saves a median
of 2 ops (mean 7.5; 73.5% save 5 or fewer; 5.2% save 40 or more, a long
route to a state one or two ops reach). They diverge at 1,224 distinct states
in 288 tasks and give 2,767 distinct (state, preferred op, dispreferred op)
labels; 2,147 of the pairs diverge at the root, so 288 of the labelled states
are task roots. 75 states (6%) carry contradictory labels (a over b to one met
state, b over a to another): the pair label is "shorter to *this* state",
which a policy conditioned on the task's goal cannot see.

Failures by behaviour: `sum_negative` 102, `capped_sum` 57, `n_minus_each`
55, `widest_pair` 50, `keep_below` 43, then a tail over 32 others. The
example failing pair `dup div` vs `drop 1` (x/x = 1 unless x = 0) is the
mechanism in one line.

## Against the registered predictions

- **>= 80% of closed earlier-layer meetings pass**: 84.8%. Passed, by less
  than the identities-and-commutations picture predicted; the deep meetings
  are a third aliasing.
- **Median 2-4 ops saved**: 2. Passed.
- The training prediction (step 3) is untested.

## Not measured, and a caveat

The gate draws from the task's own sampler, as #151's did. A pair that
aliases on the sampler's whole support (two programs equal on non-empty lists
of positive ints, say) passes it and would be a shortcut on wider inputs;
#134's perturbed and dense draws would catch some of those. No gate was run
on `BEST` or on world-v3 (no samplers there). Whether the labels change what
a policy writes is step 3.
