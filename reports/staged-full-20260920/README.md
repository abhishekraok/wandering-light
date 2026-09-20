# staged-witness-reauthoring (#155): full run over the v4 train behaviours

27 Claude Opus 5 authors of ~5 behaviours each (8 to 29 minutes of agent time per author), brief in
[`BRIEF.md`](BRIEF.md), plus two who wrote re-fused twins for the pilot's 20
([`BRIEF_REFUSE.md`](BRIEF_REFUSE.md)). Author files `programs_*.jsonl` carry each author's notes.
`assessed.jsonl` is every file re-run through `experiments/staged_witnesses/check.py --file` after the
authors returned. The deliverable is **`experiments/staged_witnesses/v4.jsonl`**, rebuilt from the
author files by `build_side_file.py`, which re-assesses every row and applies the overrides and flags
listed at its top.

**Scope.** 175 train behaviours. 22 have a fused witness under 12 ops and keep it (*already short*).
The other 153 were rewritten: the pilot's 20 and 133 here. Eval behaviours are untouched; nothing at
eval reads a witness. No staged witness collides with an eval skeleton.

## Result

**153 of 153 staged programs and 153 of 153 re-fused twins verify** (500 fresh samples against the
Python reference, and every shown and held-out pair of all 40 tasks).

| verdict | behaviours |
|---|---:|
| staged | **143** (82% of train behaviours; 88% of fused training rows, one row per witness op, get a new witness) |
| not staged | 8 |
| fused already within rules, nothing honest to add | 2 (`list_totals`, `positive_part`) |
| already short, not attempted | 22 |

Staged by fused-length bucket: 12-20 ops 24 of 26, 21-40 26 of 28, 41+ 93 of 99. Of the
validation-skeleton holdout's 16 behaviours, 14 were in scope and all 14 staged.

Over the 143 staged behaviours, one witness each, replayed on the first shown input:

| | fused (v4) | staged | re-fused twin |
|---|---:|---:|---:|
| total ops | 7,470 | 6,425 | 6,790 |
| ops written under a mark | 84.1% | 70.4% | 75.4% |
| closed states, share of ops | 14.4% | 27.5% | 22.5% |
| executions per witness, median | 1 | 4 | 2 |
| largest quotation, atoms, median | 29 | 7 | 16 |
| longest write run, steps, median | 36 | 11 | 21 |

**Length.** Staged / fused, median 0.88; 86 of 143 staged witnesses are shorter than the fused one.
By bucket: 12-20 ops **1.67**, 21-40 **1.05**, 41+ **0.74** (75% shorter). Two exceed the 2.5x
guideline (`evaluate_polynomial_horner` 2.85, `run_length_encoding` 2.69). Long fused witnesses shrink
because the authors replaced a packed-accumulator fold by a pipeline, not because they split it.

**How staged was reached.** 110 by +2 or more post-execution states; 17 with +1 and 16 with 0 or
fewer, those 33 through the clause "the fused witness itself broke a rule". A non-positive gain is
usually a good rewrite that *removed* executions (`three_number_range`: four `ifte`s became a max fold
and a min fold, gain -2; `point_centroid_floor` 0.45x, gain -4), so executions under-count staging;
largest quotation and longest write run are the steadier shape measures. A handful are compression
only: `remove_values_seen_in_second` and `zero_base_by_minimum` keep the fused shape with a smaller
body.

**The re-fused control.** The twin executes fewer times than the staged program for 136 of 143; 6
twins are the staged program unchanged because nothing merges. It is a partial control: the twin's
median largest quotation is 16 atoms against the fused 29, and 19 twins are themselves within the
staging rules. Twins are a median 1.04x the staged length, so at a fixed algorithm **staged is not
longer than fused**: merging folds costs accumulator packing the split gets free. A few twins
necessarily land on the v4 witness's shape (`dot_product`), and `sort_three`'s is the staged program
wrapped in one quotation and run with `i`, since top-level `ifte` cycles have no combinator to merge.

## Not staged (8)

- **Sorting and selection (4):** `sort_ascending` (36-atom insert), `stable_sort_flat_pairs` (43),
  `merge_sorted_lists` (27), `kth_order_statistic` (26). The insert must see the whole accumulator and
  the item together, and the number of passes is data-driven. All four best attempts verify and are
  0.31x to 0.51x the fused length.
- **The extremum of every prefix (4):** `running_maxima`, `running_minima`, `running_ranges`,
  `running_peak_shortfall`. A scalar extremum fits (the combine `[ lt? ] [ swap ] [ ] ifte drop` is 7
  atoms), and so does any *question* about records (`record_high_sum`, `record_low_count`,
  `record_extreme_count` stage by asking, per element, which earlier values block it). Computing the
  extremum of every prefix does not, whether it is emitted as a list or reduced afterwards
  (`running_peak_shortfall` returns a scalar): reducing a prefix inside a `map` is 10 atoms, a scan that carries
  the list is 12.

## Reviewer overrides and flags (in `v4.jsonl`, field `flags`)

- **Override, 1.** `running_minima`: the author's program was a prefix-minimum by doubling, correct
  only for lists of at most 8 elements (the sampler draws 2-6). An approximation is not a witness;
  replaced by the exact scan the author named, which breaks rule 2, so the behaviour is *not staged*.
- **Contrived, 6.** Within the rules only through a trick a person would not give as the
  decomposition: `mean` (shifted mean), `compare_each_to_mean` (sign as `d // (d*d+1)`) and `max_pair`
  (selection as the blend `a + s*(b-a)`), all three from the pilot's stricter classifier; `sorted_within_pairs` (pair sort as `(s+g)/2`, 141 ops, 14
  executions); `largest_zero_delimited_sum` (data turned into programs and run with `i`, so its
  intermediates hold code); `evaluate_polynomial_horner` (Horner, one honest fold, replaced by an
  explicit power sum). Authors from w17 on were asked to mark such rows and marked none.
- **Precondition, 7.** Correct on the sampler's domain and not beyond, as the fused witness is:
  `lo < hi` (4), `count >= 0`, at least two elements, non-empty list.

## What bound

Rule 2 (8 atoms) forced every algorithm change; rule 3 (12 write steps) forced most of the plumbing.
An `ifte`-in-`map` body such as bool-to-int is exactly 8 atoms and **13** write steps, one over, and
authors spent effort on `nil` as the condition, consing the literal in, or splitting with a `cons`. If
a rule is ever relaxed, 12 to 14 write steps would remove more contrivance than a larger atom cap.
Idioms are collected in the brief and the author notes; `experiments/world_behaviours/REFERENCE.md`'s
packing convention was backwards (`nil swap cons swap cons` on `a b` gives `( a b )`) and is fixed.
