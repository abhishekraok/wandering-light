# Brief: write a short-body program for a hard eval behaviour (short-body-authoring, #167)

You are one of several authors. Your behaviours are listed in the assignment you were given. Work from the repo root.

## Why, and what this is for

The solver (`BEST`) solves 18 of 43 cohort behaviours. The 25 it does not solve have authored witnesses whose longest quotation body is a median of 40 ops. On the behaviours it does solve, the solver's own programs are pipelines of **short** bodies, where the authors nested `ifte` inside a long `fold` (solver-program-rescoring, #165).

The question: **does a program with only short quotation bodies exist for each unsolved behaviour, and how long is it?**

These programs are **analysis artefacts only**: they are scored, never trained on. Correctness comes first, the body cap second, and length third.

## The machine

Read the section **"The machine"** of `experiments/world_behaviours/REFERENCE.md`: ops, `ifte`, closed bodies, packing accumulators.
- **Primitives only, no macros:** `0 1 nil true false dup swap drop over add sub mul div eq? lt? cons uncons null? quote cat [ ] i dip map filter fold ifte`.
- **The only integer literals are `0` and `1`.**

## Commands (always `PYTHONPATH=. uv run python ...`)

```bash
PYTHONPATH=. uv run python experiments/short_body/check.py --show <behaviour>
PYTHONPATH=. uv run python experiments/short_body/check.py --try <behaviour> "<program>"
PYTHONPATH=. uv run python experiments/short_body/check.py --try <behaviour> "<program>" --cap 8
PYTHONPATH=. uv run python experiments/short_body/check.py --file reports/short-body-20260923/programs_<name>.jsonl
```

- **`--show`** prints the summary, the Python reference, sample input/output stacks and the authored witness.
- **`--try`** verifies your program against the reference on 500 fresh samples and on every task's shown and held-out pairs. It then prints:
  - `within_cap`: no quotation it executes holds more than the cap in atoms, nested quotations counted;
  - its metrics beside the authored witness's: `longest_body`, `segments`, `longest_segment`.

## The rule

**No quotation over 4 atoms, nested quotations counted** (`largest_quotation <= 4`). Nothing else is restricted:
- Length may grow.
- Top-level plumbing (`swap`, `over`, `dip`, `cons`, `uncons`, `quote`, `cat`) is free.
- Change the algorithm freely.

**Prefer many short stages.** `map`, then `filter`, then `fold`, each with a short body, is the target shape. Between stages the stack should hold an intermediate a person could name.

**If you cannot meet 4 atoms**, find the smallest cap you can: try `--cap 8`. Return your best verified program with its `largest_quotation`, and say in `notes` what forced a bigger body.

**Never return an unverified program.** If nothing verifies, return `status: unsolved` with notes.

## What the solver writes: its own verified programs on behaviours it solves

These are the style that clears. They are verified, and all but two were found by `BEST`'s beam:

- `all_positive`: `[ 0 swap lt? ] map [ false eq? ] filter [ null? ] [ drop 1 ] [ drop 0 ] ifte`
- `capped_sum`: `[ [ lt? ] [ ] [ swap ] ifte drop ] swap cons map 0 [ add ] fold`
- `count_in_range`: `[ over ] dip [ lt? ] swap cons filter swap [ swap lt? ] swap cons filter 0 [ drop 1 add ] fold swap drop`
- `count_in_range`: `swap [ [ lt? ] swap cons filter ] dip [ swap lt? ] swap cons filter 0 [ drop 1 add ] fold`
- `drop_last`: `nil [ cons ] fold uncons swap nil [ cons ] fold swap drop`
- `largest_inside_interval`: `[ lt? ] swap cons swap [ [ lt? ] [ swap ] [ ] ifte drop ] swap cons [ swap ] dip map swap filter uncons [ [ lt? ] [ swap ] [ ] ifte drop ] fold`
- `largest_inside_interval`: `swap [ [ lt? ] swap cons filter ] dip [ [ lt? ] [ swap ] [ ] ifte drop ] fold`
- `min_pair`: `[ lt? ] [ ] [ swap ] ifte drop`
- `min_pair`: `[ lt? ] [ drop ] [ swap drop ] ifte`
- `min_pair`: `[ over lt? ] [ swap ] [ ] ifte drop`
- `negate_each`: `[ dup 0 1 add div dup add sub ] map`
- `negate_each`: `[ dup 0 swap sub quote swap cons ] map [ uncons drop uncons swap drop ] map`
- `negate_each`: `[ dup 0 swap sub quote swap cons ] map nil [ cat ] fold nil [ cons ] fold nil nil cons nil cons [ [ uncons ] dip cons quote cat ] fold uncons swap drop`
- `negate_each`: `nil [ cons ] fold [ dup 0 swap sub quote swap cons ] map nil [ swap cat ] fold nil [ cons ] fold nil nil cons nil cons [ [ uncons ] dip cons quote cat ] fold uncons swap drop`
- `pair_floor_averages`: `nil [ cons ] fold nil quote nil cons [ [ uncons ] dip cons quote cat ] fold uncons swap uncons swap drop [ [ uncons ] dip add quote cat ] fold [ 1 1 add div ] map`
- `prefix_sums`: `0 quote [ over uncons swap drop add cons ] fold nil [ cons ] fold uncons drop`
- `push_back`: `quote cat`
- `rotate_left`: `uncons quote cat`
- `sum_negative`: `[ 0 lt? ] filter 0 [ add ] fold`

## Idioms that fit in 4 atoms

- **Counting trues:** `[ ] filter` then `0 [ drop 1 add ] fold`. The fold body is 3 atoms.
- **Take a list's head element:** `uncons swap drop` at the top level, not in a body.
- **Bind an outer value into a short body:** `[ lt? ] swap cons filter`, from `count_in_range` above. The consed value becomes an atom of the body, so count it.
- **Carry state across stages as a list**, then unpack it at the top level between stages. Unpacking in a body costs atoms.
- **Fission:** a fold that does three things becomes three passes, e.g. `map`, `map`, `fold`.
- **Zip-like pairing:** build pairs with a short `map` over a consed list, then operate per pair.

## Output

Write `reports/short-body-20260923/programs_<name>.jsonl`, one line per behaviour:

```json
{"behaviour": "...", "program": "...", "cap": 4, "status": "cap4 | cap8 | best_attempt | unsolved", "notes": "one or two sentences: the stages you chose, or what blocked the cap"}
```

- **`cap`** is the smallest cap your program is within.
- **`status`:**
  - `cap4` or `cap8`: verified and within that cap;
  - `best_attempt`: verified, but over 8 atoms;
  - `unsolved`: nothing verified.

Finish by running `--file` on it and fixing anything that does not verify.
- **Budget:** about 15-20 minutes per behaviour.
- **Do not** edit any other file, commit, push, post to GitHub, or delegate to other agents.
