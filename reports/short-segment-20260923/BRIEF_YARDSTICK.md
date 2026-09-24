# Brief: a segment-capped yardstick program (short-segment-campaign #168)

You are one of several authors. Your behaviours are listed in your assignment. Work from the repo root.

## What this is for

**Analysis only: these programs are never training data.** They are a yardstick: the policies trained in #168 are scored on how many raw nats per op they pay along them, and whether a segment-reset beam can follow them. The behaviours are the 25 eval behaviours `BEST` does not solve. short-body-authoring (#167) already wrote each one with no quotation over 4 atoms. Its longest segments run 8-35 ops. The target style is:
- **short segments:** a closed state at least every 8 ops;
- **short bodies:** no quotation over 4 atoms (the #167 program already meets this).

Start from the #167 program given in your assignment, and restructure it so no segment runs over 8 ops.

## The machine

Read **"The machine"** in `experiments/world_behaviours/REFERENCE.md`.
- Primitives only: `0 1 nil true false dup swap drop over add sub mul div eq? lt? cons uncons null? quote cat [ ] i dip map filter fold ifte`.
- The only integer literals are `0` and `1`.

## Commands (always `PYTHONPATH=. uv run python ...`)

```bash
PYTHONPATH=. uv run python experiments/short_body/check.py --show <behaviour>
grep '"<behaviour>"' reports/short-body-20260923/programs_*.jsonl   # #167's programs
PYTHONPATH=. uv run python experiments/short_body/check.py --try <behaviour> "<program>" --segment-cap 8
PYTHONPATH=. uv run python experiments/short_body/check.py --file reports/short-segment-20260923/yardstick_<name>.jsonl --segment-cap 8
```

`--try` verifies against the Python reference on 500 fresh samples and on every task's shown and held-out pairs. It prints:
- `within_cap`: no executed quotation over 4 atoms, nested counted;
- `within_segment_cap`: `longest_segment <= 8`. A segment ends after each top-level combinator application: `map`, `filter`, `fold`, `ifte`, `dip` or `i` run with no `[` open;


## Rules

1. **Bodies:** no executed quotation over 4 atoms, nested counted.
2. **Segments:** at most 8 ops between closed states.
3. **General programs only.**
   - The program must be correct for inputs of **any** length, not just the sampler's.
   - **No unrolling** to a maximum length.
   - **No tricks that rely on bounded values**, such as dividing by 2^64 to test equality.
   - Closures built by `map` and run with `i` inside a fold body are fine.
4. **Prefer the decomposition a person would describe:** `map`, then `filter`, then `fold`, each stage with a nameable intermediate.
5. **Correctness first, then rule 1, then rule 2, then length.**

If you cannot meet both caps with a general program, return your best **general** verified program, say which cap it breaks, and say what blocked it. Never return an unverified program.

## Examples: train programs within both caps (#155's staged witnesses, and #168's train authoring)

- `quotient_remainder`: `over over div dup [ mul ] dip [ sub ] dip [ quote ] dip cons`
- `count_negative`: `[ 0 lt? ] map [ ] filter [ drop 1 ] map 0 [ add ] fold`
- `endpoints`: `dup 0 [ swap drop ] fold [ uncons swap drop ] dip quote [ quote ] dip cat`
- `count_zeros`: `[ 0 eq? ] map [ true eq? ] filter [ drop 1 ] map 0 [ add ] fold`
- `count_equal`: `[ sub ] swap cons map [ 0 eq? ] map [ ] filter 0 [ drop 1 add ] fold`
- `sum_inside_interval`: `[ swap ] dip [ lt? ] swap cons filter swap [ swap lt? ] swap cons filter 0 [ add ] fold`
- `rotate_right`: `dup 0 [ swap drop ] fold quote [ quote cat ] fold nil [ cons ] fold uncons drop nil [ cons ] fold`
- `sign_balance`: `dup [ 0 swap lt? ] filter 0 [ drop 1 add ] fold swap [ 0 lt? ] filter 0 [ drop 1 add ] fold sub`
- `distance_from_target_total`: `[ sub ] swap cons map dup [ 0 swap lt? ] filter 0 [ add ] fold swap [ 0 lt? ] filter 0 [ add ] fold sub`
- `sum_absolute_values`: `dup [ 0 lt? ] filter [ 0 swap sub ] map 0 [ add ] fold swap [ 0 swap lt? ] filter 0 [ add ] fold add`
- `total_distance_outside_interval`: `[ over ] dip [ sub ] swap cons map swap [ swap ] dip [ swap sub ] swap cons map cat [ 0 swap lt? ] filter 0 [ add ] fold`
- `positive_part`: `quote [ 0 swap lt? ] filter 0 [ add ] fold`
- `product_inside_interval`: `[ 1 sub swap ] dip 1 add [ lt? ] swap cons filter swap [ swap lt? ] swap cons filter 1 [ mul ] fold`
- `merge_sorted_lists` (insertion sort run as closures): `cat [ quote dup cat ] map [ [ lt? ] quote cat ] map [ [ swap cons ] cat ] map [ [ cons ] quote cat ] map [ [ dip over ] cat ] map [ [ over false ] cat ] map [ [ quote ] cat ] map [ [ eq? ] quote cat ] map [ [ cat cat ] cat ] map [ [ filter ] cat ] map [ [ filter ] quote cat ] map nil [ i dip cat ] fold`
- `squared_deviation_total`: `dup [ drop 1 ] map 0 [ add ] fold over 0 [ add ] fold swap div [ sub ] swap cons map [ dup mul ] map 0 [ add ] fold`

## Idioms that fit in 4 atoms

- **Counting trues:** `[ ] filter`, then `0 [ drop 1 add ] fold`.
- **Bind an outer value into a short body:** `[ lt? ] swap cons filter`. The consed value counts as an atom.
- **Prefix sums without a long body:** map each x to an adder `[ [ add ] swap cons ] map`, then fold with body `map 0 cons`, which shifts the earlier entries and pushes a fresh 0.
- **Maximum without a conditional body:** map each x to the filter `[ lt? false eq? ]` bound to x, then fold `filter` over the list.
- **Split long scans across stages.** Every top-level combinator application ends a segment, so long glue between two combinators breaks rule 2. Move work into the next `map` or `filter`.

## Output

Write `reports/short-segment-20260923/yardstick_<name>.jsonl`, one line per behaviour:

```json
{"behaviour": "...", "program": "...", "status": "capped | best_attempt | unsolved", "breaks": "none | body | segment | both", "notes": "one or two sentences"}
```

Finish by running `--file` with `--segment-cap 8` and fixing anything that does not verify.
- **Budget:** about 15 minutes per behaviour.
- **Do not** edit any other file, commit, push, post to GitHub, or delegate.

## Idioms the train authors found

- **Build long bodies as closures:** a chain of short `map`s, each `cat`-ing a few atoms onto per-element quotations, then one fold that runs them with `i`. Every map is its own segment.
- **Zip two lists:** turn one list into closures `[ uncons x ... ]` with maps, then a rotate-fold `[ [ uncons ] dip i ] fold` runs them against the other list.
- **Slot-swap split into alternate elements:** a fold of `(uncons z cons swap cons)` closures deals a list into two piles.
