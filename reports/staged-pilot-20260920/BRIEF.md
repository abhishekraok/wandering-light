# Brief: rewrite a fused `mark-28` witness as a staged one

staged-witness-reauthoring (#155), pilot. You are one of four authors. Your behaviours are listed,
one per line, in the `assign_<letter>.txt` you were given. Work from the repo root.

## Why

witness-trajectory-shape (#150) found that the existing witnesses are written, then run: a long
quotation is assembled under an open `[` while the input sits untouched, then one `fold` or `map`
does everything inside a single opaque step. We want the **same functions** written as several short
write-run cycles, so that between cycles the stack holds an intermediate a person could name.

`x -> 2x^2 + 4` over a list. Fused: `[ dup mul dup add 1 add 1 add 1 add 1 add ] map`.
Staged: `[ dup mul ] map [ dup add ] map [ 1 add 1 add 1 add 1 add ] map`: three cycles, and
after each the stack is "the squares", "the doubled squares", "the answer".

## The machine

Read the section **"The machine"** of `experiments/world_behaviours/REFERENCE.md` (ops, `ifte`,
closed bodies, packing accumulators). **Macros are not available here**: write primitives only
(`0 1 nil true false dup swap drop over add sub mul div eq? lt? cons uncons null? quote cat
[ ] i dip map filter fold ifte`, and whatever else the fused witness itself uses). The only integer
literals are `0` and `1`.

## Commands (always `PYTHONPATH=. uv run python ...`)

```bash
PYTHONPATH=. uv run python experiments/staged_witnesses/check.py --show <behaviour>
PYTHONPATH=. uv run python experiments/staged_witnesses/check.py --try <behaviour> "<program>"
PYTHONPATH=. uv run python experiments/staged_witnesses/check.py --file reports/staged-pilot-20260920/programs_<letter>.jsonl
```

`--show` prints the summary, the Python reference, six sample input/output stacks, the fused
witness and its metrics. `--try` verifies your program against the reference on 500 fresh samples
and on every task's shown and held-out pairs, then prints its metrics beside the fused witness's
and a verdict.

## Rules

1. **Run a quotation promptly.** Once its `]` closes, at most 4 top-level ops may pass before the
   combinator that consumes it (`longest_park <= 4`). Those ops are for what closed bodies force:
   `swap cons` to bind an outer value into the body, or opening the next quotation of an `ifte`.
   Do not build a quotation and park it while other work happens.
2. **No quotation over 8 atoms**, nested quotations counted (`largest_quotation <= 8`). Nest only
   for `ifte` branches.
3. **No write run over 12 steps** (`longest_write_run <= 12`).
4. **Every closed state should be nameable.** A closed state has no open `[` and no finished
   quotation waiting on the stack. Aim for intermediates like "the squares", "the running
   minimum", "the count so far", not half-assembled plumbing.
5. **Prefer fission**: `map`, then `map`, then `fold`, over one fold doing all three. Changing the
   algorithm to make it stageable is wanted. Length may grow; try to stay under 2.5x the fused
   length, but correctness and rules 1-3 come first.

The verdict `staged` needs rules 1-3 to hold **and** at least 2 more post-execution closed states
than the fused witness (`post_execution_gain >= 2`). That is the target.

**Unstageable is a result, not a failure.** If a behaviour cannot meet the rules (typically a fold
whose state genuinely needs a big body), return your best verified attempt, the one that gets
closest, and say in `notes` what blocked it. Never return an unverified program.

## Output

Write `reports/staged-pilot-20260920/programs_<letter>.jsonl`, one line per behaviour:

```json
{"behaviour": "...", "program": "...", "status": "staged | best_attempt | unsolved", "notes": "one or two sentences: the intermediates you chose, or what blocked staging"}
```

Finish by running `--file` on it and fixing anything that does not verify. Budget about 10-15
minutes per behaviour. Do not edit any other file, do not commit, do not push, do not post to
GitHub, and do not delegate to other agents.
