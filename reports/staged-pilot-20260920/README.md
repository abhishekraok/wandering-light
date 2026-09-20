# staged-witness-reauthoring (#155): pilot, 20 train behaviours

Four Claude Opus 5 authors, 5 behaviours each, brief in [`BRIEF.md`](BRIEF.md), sample in `pilot.tsv`
(fused witness of 12+ ops, 5 / 5 / 10 over 12-20 / 21-40 / 41+, seed 17). Programs in
`programs_<author>.jsonl` with the author's notes; `assessed.jsonl` is every program re-run through
`experiments/staged_witnesses/check.py --file` after the authors returned. 26 to 34 minutes of agent
time per author.

**Result: 18 of 20 staged, 20 of 20 verified; gate (13 of 20, median length ratio <= 2.5) passed.**

| | fused | staged |
|---|---:|---:|
| total ops over the 20 | 1,029 | 921 |
| share of ops written under a mark | 83.2% | 70.1% |
| closed states, share of ops | 15.0% | 26.7% |
| executions per witness, median | 1 | 4.5 |
| largest quotation, atoms, median | 24.5 | 7 |

Length ratio staged / fused, median: 12-20 ops **2.07** (5 of 5 staged), 21-40 **1.19** (4 of 5),
41+ **0.77** (9 of 10); all 20, 1.13. Short fused witnesses double; long ones shrink, because the
authors replaced the algorithm rather than splitting it (`keep_positive_sum_pairs` 98 -> 49 ops,
`aligned_match_count` 61 -> 31, `squared_point_norms` 72 -> 39).

**Not staged.** `running_maxima`: a max-combine is `[ [ lt? ] [ swap ] [ ] ifte drop ]`, 8 atoms on
its own, so any running extremum needs a 10-atom body; rule 2 only. `kth_order_statistic`: selection
needs a 26-atom insertion body; verified at 75 ops against the fused 174, six closed stages, breaks
rules 2 and 3.

**What the authors report.** Rule 2 (8 atoms) binds, never rule 1. Unpacking a two-slot list costs 5
atoms, leaving 3 for work. Reusable idioms that fit the cap: rotate-fold as a zip
(`[ uncons ] dip <op> quote cat`, 6 atoms), slot-swapping split, type-separated accumulator (append
`Bool`s to an int list and `[ true eq? ] filter`), `0 [ lt? ] [ mul ] [ drop ] ifte` as `max(v, 0)`.

**Caveats for the full run.**

- The `+2 post-execution states` bar pushes short witnesses into contrivance. `mean` already executes
  twice fused, its natural staging (sum, count, divide) gains 1, and the author reached +2 with a
  shifted mean; `compare_each_to_mean` uses `d // (d*d + 1)` as a branch-free sign. Proposed
  classifier: staged iff within rules 1-3 and (gain >= 2 **or** the fused witness broke a rule).
- Some staged programs trade a small body for a large intermediate: `running_maxima` materialises
  every prefix (peak 40 atoms against the fused 30). Verified within fuel on v4's short lists;
  quadratic in list length.
- Stack still ramps within a cycle: a cycle is `[ body ] combinator`, so most of its ops are writes by
  construction, and W share only falls to 70%. What changed is the size of each ramp (24.5 -> 7 atoms)
  and what sits between them.
