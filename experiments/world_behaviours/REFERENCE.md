# Authoring `mark-28` programs for world behaviours

Task: for each behaviour id assigned to you, write a `mark-28` program that
passes **every** test, using the macro library. Record what you wrote and
whether it verified. The probe question is *what fraction of world-authored
behaviours can be given a witness, and how long the witnesses are*, so an
honest "unsolved" with a reason is a valid result. Do not fake or narrow.

## Commands (run from the repo root, always via `uv run`)

```bash
B=reports/world-behaviours-probe-20260909/behaviours.jsonl
# read a behaviour: description, reference Python, and every test
uv run python experiments/world_behaviours/verify.py $B --id mbpp-970 --show
# try a program (macro form allowed); prints pass count, first failures, expansion
uv run python experiments/world_behaviours/verify.py $B --id mbpp-970 --program "min2"
# verify your whole output file at the end
uv run python experiments/world_behaviours/verify.py $B --programs reports/world-behaviours-probe-20260909/programs_K.jsonl
```

Output file, one JSON object per line, one per assigned id:

```json
{"id": "mbpp-970", "program": "min2", "status": "solved", "note": ""}
{"id": "mbpp-152", "program": "", "status": "unsolved", "note": "needs sort; O(n^2) selection with pair accumulator exceeded my budget"}
```

`program` is the **macro form** (what you typed). The verifier expands it.

## The machine

A state is a stack of values: `Int` (unbounded), `Bool`, `Lst` (`nil` is the
empty list), and quotations, which are just lists of op names. **Inputs are the
task's arguments, first argument deepest, last on top.** The result must be the
**only** value left on the stack, and it is an int or a list of ints, never a
bool: convert with `bool>int`. Extra values left on the stack fail the test.

Literals: only `0 1 nil true false`. Other constants come from macros (`2` … `10`,
`100`, `1000`, `-1`) or arithmetic. `div` floors like Python `//`; `mod`
(macro) matches Python `%`. Division by zero is a refusal.

| op | effect | notes |
|---|---|---|
| `dup swap drop over` | usual | `over: ( a b -- a b a )` |
| `add sub mul div` | `( a b -- a op b )` | ints only |
| `eq?` | `( a b -- bool )` | structural, works on lists |
| `lt?` | `( a b -- a<b )` | ints only; `gt? le? ge?` are macros |
| `cons` | `( lst a -- (a . lst) )` | **prepends** |
| `uncons` | `( lst -- tail head )` | refuses on empty list |
| `null?` | `( lst -- bool )` | |
| `quote` | `( a -- (a) )` | |
| `cat` | `( (a) (b) -- (a b) )` | list append |
| `[ … ]` | build a quotation | ops inside are captured, not run |
| `i` | `( q -- … )` | run the quotation on the stack below |
| `dip` | `( x q -- … x )` | run `q` under the top value |
| `map` | `( lst q -- lst' )` | body sees **only the item**; must leave exactly one value |
| `filter` | `( lst q -- lst' )` | body sees only the item; must leave one `Bool` |
| `fold` | `( lst acc q -- x )` | body sees `( acc item )`, must leave one value |
| `ifte` | `( qc qt qf -- … )` | see below |

`ifte`: the condition runs on a *copy* of the stack below the three quotations
and must leave a `Bool` on top; then the chosen branch runs on the **original**
stack (the condition's consumption is discarded). So `[ dup 0 lt? ] [ neg ] [ ] ifte`
is `abs`, and in `[ over lt? ] [ drop ] [ swap drop ] ifte` the branches still
see both numbers.

**Bodies are closed.** A `map`/`filter`/`fold` body cannot see the outer stack.
To use an outer value inside a body, cons it into the quotation first:
`x [ add ] swap cons` builds the quotation `( x add )`, so
`lst x [ add ] swap cons map` adds `x` to every element. Two values:
`lst a b [ body ] swap cons swap cons` gives `( a b body )` — inner-most pushed
last, so the body sees `item a b`. Ints inside a quotation push themselves.

**Several accumulators in a fold**: pack them in a list.
`nil swap cons swap cons` packs `a b` into `( b a )`; inside the body
`[ uncons swap uncons swap drop ] dip` unpacks the accumulator under the item;
finish with `uncons drop uncons swap drop` (or similar) to pull the field you
want. See v4 witnesses such as `maximum_subarray_sum` for the full pattern.

**Loops over integers.** `range` turns `n` into `( 0 … n-1 )`, `range1` into
`( 1 … n )`; then `fold`/`filter`/`map`. A while-loop is a self-applying
quotation: with `state Q` on the stack, `dup i` runs `Q` with `Q` still on the
stack, so `Q`'s else-branch can end in `dup i` to recurse.

**Warning, found by every author in the first probe:** `MAX_QUOTATION_DEPTH`
is 32 and each iteration of a self-applying loop nests about four levels, so
`range` refuses past **n = 7** ("quotation nested too deep") and any `dup i`
loop dies after ~8 iterations, sooner inside a `map`/`fold` body. Build index
lists without recursion instead: a literal cons chain, or doubling with
`dup dup len [ add ] swap cons map cat` (`k1-range`, `k2-range`, `k3-range`,
`k4-iota` in the author macro files all do this; up to 1,024 elements for
~10k fuel), then `fold` over the list with a packed state. Fuel is 100,000
steps per test; a fold costs ~3 per element, a `10` constant inside a body
costs 8 ops per use, and nested ranges over thousands exhaust it.

**Recursion on lists** has the same depth cap; use `fold` and rebuild.

## Macros

`experiments/world_behaviours/macros.txt` is shared. It has constants, `neg
inc dec abs max2 min2 sq mod not gt? le? ge? bool>int even? odd?`, and list
words `len sum product reverse head tail last max min range range1 take drop-n
nth contains`. Read it: bodies are short and show the idioms.

Add your own macros to **your** file `experiments/world_behaviours/macros_K.txt`
(K = your author number), same `name: body` format, one per line. Names must
not collide with basis ops or existing macros; prefix with `kK-` if in doubt.
Every `macros*.txt` in the directory is loaded. Macros are the point of the
probe as much as programs: a word you needed three times is a highway
candidate.

## Working method

1. `--show` the behaviour. Read the reference Python: the tests define the
   function, including edge cases you would not guess from the description.
2. Decide the shape: scalar arithmetic; `map`/`filter`; single fold; fold with
   a packed accumulator; `range` + fold; while-loop. Prefer the simplest.
3. Write it in macro form, verify, read the first failures, fix.
4. Budget about 10–15 minutes per behaviour. If it needs sorting, nested loops
   over ranges of 1000, or bit operations you cannot afford, record `unsolved`
   with the reason and move on. Partial programs are not recorded as solved.
5. Never edit `behaviours.jsonl`, `macros.txt`, `verify.py`, or another
   author's files. Do not commit or push.
