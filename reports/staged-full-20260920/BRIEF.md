# Brief: rewrite fused `mark-28` witnesses as staged ones, and re-fuse them (full run)

staged-witness-reauthoring (#155), full run. You are one of 27 authors. Your behaviours are listed,
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
PYTHONPATH=. uv run python experiments/staged_witnesses/check.py --file reports/staged-full-20260920/programs_<letter>.jsonl
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

The verdict `staged` needs rules 1-3 to hold **and** either at least 2 more post-execution closed
states than the fused witness (`post_execution_gain >= 2`) **or** a fused witness that itself broke a
rule. So do not contrive extra stages to reach +2: if the fused witness has a 20-atom quotation and
your natural decomposition (say sum, count, divide) is within the rules, that is staged. Prefer the
decomposition a person would describe over a clever trick.

**Unstageable is a result, not a failure.** If a behaviour cannot meet the rules (typically a fold
whose state genuinely needs a big body), return your best verified attempt, the one that gets
closest, and say in `notes` what blocked it. Never return an unverified program.

## Idioms the pilot authors found (all fit the 8-atom cap)

- **Unpack a two-slot list** costs 5 atoms (`uncons swap uncons swap drop`), leaving 3 for work.
  Carry a pair per element through several maps when a chain would lose the original value.
- **Rotate-fold as a zip**, 6 atoms: fold list A with list B as the accumulator, body
  `[ uncons ] dip <op> quote cat`: take B's head, append the combined value at the tail. When the
  lengths match, the accumulator ends as the elementwise result in order.
- **Walk two lists in lockstep**, 8 atoms: `[ swap uncons [ swap ] dip <op> quote cat ]`.
- **Slot-swapping split**, 6 atoms: accumulator `( A B )`, body `[ uncons ] dip cons quote cat`
  pushes the item into the head slot and swaps the slots, separating even and odd positions.
- **`max(v, 0)`** is `0 [ lt? ] [ mul ] [ drop ] ifte`, exactly 8 atoms.
- **Type-separated accumulator**: `eq?` is structural across types, so `Bool`s appended to a list
  of ints can be pulled back out with `[ true eq? ] filter`.
- **Defer the comparison out of the scan.** A min/max combine is 8 atoms by itself
  (`[ lt? ] [ swap ] [ ] ifte drop`), so a running extremum does not fit in one fold body. Build the
  intermediates first (prefixes, pairwise margins via a nested consed `map`), aggregate after.
- Counting trues: `[ ] filter` then `0 [ drop 1 add ] fold`.

Known walls: a running maximum needs a 10-atom body; selection / sorting needs ~26. Return those as
`best_attempt`.

## The re-fused control

For every behaviour you stage, also write `refused_program`: **the same algorithm as your staged
program**, with consecutive cycles merged back into as few quotations as you reasonably can (`map`
after `map` becomes one `map`; a `map` feeding a `fold` becomes one `fold`). No rules apply to it; it
only has to verify, and it should execute fewer times than your staged program. Do not go back to the
original fused witness's algorithm: the point is a twin that differs from your staged program in
shape only. If nothing merges, repeat the staged program and say so in `notes`. Check it with
`--try <behaviour> "<refused program>"` (ignore the verdict; it must say `"verified": true`).

## Output

Write `reports/staged-full-20260920/programs_<letter>.jsonl`, one line per behaviour:

```json
{"behaviour": "...", "program": "...", "refused_program": "...", "status": "staged | best_attempt | unsolved", "notes": "one or two sentences: the intermediates you chose, or what blocked staging"}
```

Finish by running `--file` on it and fixing anything that does not verify. Budget about 10-15
minutes per behaviour. Do not edit any other file, do not commit, do not push, do not post to
GitHub, and do not delegate to other agents.
