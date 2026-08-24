# Cost recertification of `stack_pbe_v2`

## Result

**0 of the 101 requested loop-witness labels change under Kt weighting.** The
largest discrepancy is therefore **0 bits**.

A label changes only when another program has strictly smaller

```text
Kt = |program| + log2(fuel).
```

An equal-weight alternate witness is a tie, not a changed label. The comparison
is exact rather than floating point: the search orders the integer

```text
score = fuel * 2**|program| = 2**Kt.
```

For a PBE task, `fuel` is the sum of the cumulative machine fuel used to run the
program on every example. The program is encoded once. This is the work done by
the product-behaviour transition; recording the convention matters because 54
of the 101 stored witnesses use different fuel on different examples.

| family | tasks checked | labels changed |
|---|---:|---:|
| `with_filter` | 40 | 0 |
| `with_fold` | 37 | 0 |
| `with_map` | 18 | 0 |
| `bool_int` | 3 | 0 |
| `lm_order2` | 2 | 0 |
| `int_pair` | 1 | 0 |
| **total** | **101** | **0** |

This meets the issue's kill criterion: at this basis size, uniform edge weight
is an adequate approximation for the requested loop-witness slice. No corpus
labels need rewriting.

## Scope correction

The committed artifact contains 1,857 tasks. Exactly 101 stored programs
contain `map`, `fold` or `filter`, but they are not exactly the three named
families:

- the families `with_map`, `with_fold` and `with_filter` contain 522 tasks;
- only 95 of the requested 101 loop witnesses are in those families;
- the other six are three `bool_int`, two `lm_order2` and one `int_pair` task.

The experiment therefore selects by stored-program membership and asserts that
the result is 101. It does not claim that these are every variable-cost witness
in the corpus: `dip`, `i` and `ifte` also execute quotations, but are outside
the issue's explicit experiment scope.

## Certificate

The 101 tasks share eight distinct input tuples. For each tuple, the experiment:

1. replays every stored witness to recover its per-example fuel and exact score;
2. uses the largest stored score as an incumbent bound;
3. runs `expand_cost` once for the root, ordered by exact Kt score;
4. retains every nondominated `(length, per-example fuel vector)` arrival at a
   behaviour; and
5. compares each target's best score with its stored witness's score.

One first arrival per behaviour would not certify Kt. A longer low-fuel prefix
can be cheaper at an intermediate behaviour while a shorter high-fuel prefix
becomes cheaper after an expensive shared suffix. Componentwise Pareto
dominance is safe: a discarded path uses at least as many ops and at least as
much fuel on every example, so it cannot beat the retained path after any
suffix.

The incumbent also gives an exhaustive depth bound. With `n` examples, every
depth-`d` program uses at least `n*d` fuel, so its score is at least
`n*d*2**d`. Once that exceeds the largest incumbent score, no deeper path can
improve any requested label. All eight roots were bounded at depth six or
less.

## Reproduction

From the repository root:

```bash
uv run python -m experiments.recertify_stack_machine_cost
```

The recorded run completed all eight roots, reached at most 815,665 distinct
behaviours for one root, and produced `changed: 0`, `unchanged: 101`, and
`largest_discrepancy_bits: 0.0`. The compressed corpus used for the run has
SHA-256 `7c2f90073c8bf1a69bfb902e25feccd90eb5e522c73c51ab7de7e63c6484fed0`.
