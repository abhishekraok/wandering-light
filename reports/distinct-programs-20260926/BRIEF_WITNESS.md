# Brief: staged `mark-28` witnesses for new behaviours (distinct-programs, #174)

You are one author among several. Your behaviour ids are listed one per line in the
`assign_<letter>.txt` you were given. Work from the repository root. Do not commit, push, edit
any tracked file, post to GitHub, or delegate to other agents.

## Why

These behaviours are new training programs for a small policy that learns to write `mark-28`
stack programs. It learns best from **staged** programs, written as a few short write-run cycles.
After each cycle the stack holds an intermediate a person could name.

`x -> 2x^2 + 4` over a list. Fused: `[ dup mul dup add 1 add 1 add 1 add 1 add ] map`.
Staged: `[ dup mul ] map [ dup add ] map [ 1 add 1 add 1 add 1 add ] map`. That is three cycles,
and after them the stack holds "the squares", "the doubled squares", then "the answer".

## Stage only where the pieces are real

Break a program into cycles only where it has logically separate pieces: a transform then a
filter, a count then a divide, the squares then their sum. **Do not invent stages to satisfy a
rule or to look staged.** No identity maps, no splitting one natural operation across two
quotations, no stashing values just to create a closed state.

A behaviour whose natural program is a single `map`, `filter` or `fold`, or a few stack ops, is
written that way and is a good result. The verdict `plain` is as welcome as `staged`. A contrived
stage is worse than none: a person reading the program should see why each cycle exists.

## The machine

Read the section **"The machine"** of `experiments/world_behaviours/REFERENCE.md`. **Macros are
not available here.** Write primitives only, the 28 ops of `mark-28`:

`0 1 nil true false dup swap drop over add sub mul div eq? lt? cons uncons null? quote cat [ ] i dip map filter fold ifte`

The only integer literals are `0` and `1`; build others arithmetically. The result must be the
only value left on the stack, an int or a flat list of ints. Convert a bool on top of the stack with
`[ ] [ drop 1 ] [ drop 0 ] ifte`.

## Commands

```bash
B=reports/distinct-programs-20260926/behaviours.jsonl
uv run python experiments/distinct_programs/check.py $B --show <id>
uv run python experiments/distinct_programs/check.py $B --try <id> "<program>"
uv run python experiments/distinct_programs/check.py $B --file reports/distinct-programs-20260926/programs_<letter>.jsonl
```

`--show` prints the description, the Python reference and every test. `--try` runs your program on
every test, shown and held-out. If it verifies, it prints the shape metrics and a verdict:
`staged` (rules hold, two or more cycles), `plain` (rules hold, one cycle) or `rule_break`.

**The reference is the behaviour.** Some tests were drawn to break properties the others share.
Whatever the reference returns there is the answer.

## Rules (from staged-witness-reauthoring, #155)

1. **Run a quotation promptly.** Once its `]` closes, at most 4 top-level ops may pass before the
   combinator that consumes it (`longest_park <= 4`).
2. **No quotation over 8 atoms**, nested quotations counted (`largest_quotation <= 8`).
3. **No write run over 12 steps** (`longest_write_run <= 12`).
4. **Every closed state between cycles should be nameable**, like "the squares" or "the count so
   far", not half-assembled plumbing.

If a behaviour cannot meet rules 1-3 (a fold whose state genuinely needs a large body, or sorting),
return your best verified program. Its verdict will be `rule_break`; say what blocked it in `note`.

## Idioms (all fit the 8-atom cap)

- Bind an outer value into a body: `lst x [ add ] swap cons map`.
- Unpack a two-slot list: `uncons swap uncons swap drop` (5 atoms).
- Rotate-fold as a zip: fold list A with list B as the accumulator, body `[ uncons ] dip <op> quote cat`.
- `max(v, 0)`: `0 [ lt? ] [ mul ] [ drop ] ifte`.
- Count the trues: `[ ] filter` then `0 [ drop 1 add ] fold`.
- A running extremum or a sort does not fit 8 atoms. Build intermediates first, or accept `rule_break`.

## Output

Write `reports/distinct-programs-20260926/programs_<letter>.jsonl`, one line per assigned id:

```json
{"id": "lf-c001", "program": "...", "status": "solved", "note": "cycles: the tail after two drops, then its head; or why it is one cycle"}
{"id": "dc-foo", "program": "", "status": "unsolved", "note": "why: e.g. fuel, or the depth cap on a loop"}
```

`status: solved` only when `--try` says `"verified": true`. Budget about 8 minutes per behaviour.
An honest `unsolved` with the reason is a valid result. Finish by running `--file` on your output
and fixing anything that does not verify.
