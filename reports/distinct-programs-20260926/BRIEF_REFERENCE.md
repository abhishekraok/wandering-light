# Brief: write Python references for list-function behaviours (distinct-programs, #174)

You are one author among several. Your behaviour ids are listed one per line in the
`assign_<letter>.txt` you were given. Work from the repository root. Do not commit, push, edit
any tracked file, post to GitHub, or delegate to other agents.

## Task

Each behaviour comes from a public benchmark: BIG-bench `list_functions`, DreamCoder or
LambdaBeam. It has a description (sometimes only a name) and a set of given input/output tests.
It has no reference implementation. Write one: a Python function named `f`, taking the test's
`args` in order and returning the output.

The reference defines the behaviour on inputs the tests never show. A later step draws new inputs
that break properties the given tests happen to share: an empty list, a negative number, a
duplicate, an index past the end. Whatever `f` returns there becomes the answer. So:

- **Write the function the description names**, in its most natural reading. Where the
  description is silent, generalise the way a person would. Do not overfit the given tests with
  lookups or special cases.
- **Be total on the domain.** Arguments are ints and flat lists of ints. `f` must return an int
  or a flat list of ints (never a bool, None or a nested list), and must not raise on empty lists,
  negatives or out-of-range indices. Choose the natural convention and say which in `note`:
  "element 3 of a short list" returns `[]`, a max of the empty list returns 0, and so on.
- **Keep it cheap.** No loops over more than about 10^4 steps.

## Commands

```bash
B=reports/distinct-programs-20260926/pool.jsonl
uv run python experiments/distinct_programs/check.py $B --show <id>               # description and every given test
uv run python experiments/distinct_programs/check.py $B --reference <id> ref.py   # does f in ref.py reproduce every given test?
```

Put scratch files under `reports/distinct-programs-20260926/scratch_<letter>/`.

## Output

Write `reports/distinct-programs-20260926/refs_<letter>.jsonl`, one line per assigned id:

```json
{"id": "lf-c001", "solution": "def f(xs):\n    return xs[2:3]\n", "status": "ok", "note": "element 3 is index 2; short lists give []"}
{"id": "dc-foo", "solution": "", "status": "skip", "note": "why: e.g. the given tests contradict each other, or the function needs a value outside ints"}
```

`status: ok` only when `--reference` reports `"passes": true`. Budget about 3 minutes per
behaviour. A reasoned `skip` is a valid result.
