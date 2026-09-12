# Authoring brief: wave 5 of world-authoring-500 (#127)

You are writing `mark-28` programs (witnesses) for world-authored behaviours:
MBPP problems and CodeARC functions, all
restricted to ints and flat int lists. Run every command from the repository
root you were started in. Do not commit, push, or edit any tracked file.

Read `experiments/world_behaviours/REFERENCE.md` first: machine semantics,
idioms, the two machine limits (nesting depth 32, fuel 100,000 per test), and
the canonical macro library `experiments/world_behaviours/macros.txt`.

## The rule

**The reference solution is the behaviour.** Each behaviour's tests include
inputs drawn to break properties its supplied examples happened to share (a
negative number where all were positive, a duplicate, an index outside the
list). Whatever the reference answers on those is the answer, including its
quirks (Python's negative indexing, truncating division, an early-return
branch). A witness must reproduce the reference on every test, shown and
held-out, or it is not a witness. Read the reference, not the description.

## Commands

```bash
B=reports/world-authoring-20260911-w5/behaviours.jsonl
uv run python experiments/world_behaviours/verify.py $B --id mbpp-266 --show              # description, reference, every test
uv run python experiments/world_behaviours/verify.py $B --id mbpp-266 --program "sum"     # pass count, first failures, expansion
uv run python experiments/world_behaviours/verify.py $B --programs <your output file>     # verify the whole file at the end
```

## Macros

Use the canonical names in `macros.txt`. You may add your own macros in
`experiments/world_behaviours/macros_K.txt` (K = your letter), one
`name: body` per line, names prefixed `kK-` (e.g. `ka-argmax`). Names must not
collide with basis ops or canonical macros; every `macros*.txt` in the
directory is loaded by the verifier. Add a macro when you need a word twice
or it makes a program readable; a merge pass folds useful ones into the
canonical library afterwards.

## Output

One JSON object per assigned id, one per line, to the output file named in
your instructions:

```json
{"id": "mbpp-266", "program": "<macro form that passes every test>", "status": "solved", "note": "shape: fold with packed accumulator; replays the reference's -1 on empty input"}
{"id": "codearc-12", "program": "", "status": "unsolved", "note": "why: e.g. needs a histogram over 10^5 values, fuel"}
```

`status: solved` only if `--program` reports passed == total. Partial
programs are recorded as `unsolved` with the best pass count in the note.
Budget about 15 minutes per behaviour; an honest `unsolved` with the reason is
a valid result and is expected for some of these. Do not fake, narrow, or
skip ids. When done, run the whole-file verification, then print the output
file's contents and your macro file's contents.
