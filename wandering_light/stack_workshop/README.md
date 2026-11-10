# Signal House

A local workshop that teaches the `mark-28` stack machine, one lesson at a
time. This is the Python service; a browser frontend comes later.

```bash
uv run python -m wandering_light.stack_workshop.server --port 8018
```

The server binds 127.0.0.1, takes `POST` with a JSON object, and refuses a
request whose `Origin` is not `127.0.0.1` or `localhost`.

```bash
curl -s 127.0.0.1:8018/api/verify -d '{"lesson": "echo", "program": "dup add"}'
```

## What is where

| module | what it holds |
|---|---|
| `engine.py` | `compile_program`, `stack_from`, `snapshot`, `execute`, `inspect_op` |
| `curriculum.py` | `LESSONS`, 14 authored lessons, and `BY_ID` |
| `model.py` | `ADVISER`, an optional `stack_policy` checkpoint that ranks the next op |
| `server.py` | `api(route, data)`, `Handler`, `main()` |

**The machine is observed, not restated.** Every step is one call to
`stack_machine.step_raw`. What happens inside a `map` or an `ifte` is read off
the machine's own nested `step_raw` calls with `sys.setprofile`, after clearing
the machine's quotation cache so a warm cache cannot hide them.

**Macros** are read from `experiments/world_behaviours/macros.txt` and expand
to primitives before anything runs. Each token keeps the source span of the
word the learner wrote.

**Expected outputs** in a lesson come from a Python reference for the rule
(`curriculum.REFERENCES`), not from the lesson's solution. A lesson shows three
examples and holds out three; `/api/verify` runs both and says which the
verdict rests on.

**The core needs only the standard library.** torch is imported when a
checkpoint is loaded (`POST /api/model`), and the workshop runs without one.
`/api/context` also lists world-corpus problems for the sandbox when that
payload is already on disk; it never fetches.

## Routes

| route | body | reply |
|---|---|---|
| `/api/context` | `{}` | lessons (no solutions, hints or extra tests), ops, macros, adviser status, world shelf |
| `/api/evaluate` | `lesson`, `program` | `execute` on the shown examples, every frame |
| `/api/verify` | `lesson`, `program` | `passed`, counts, the first three failing rows, `evidence` |
| `/api/inspect` | `lesson`, `program`, `step`, `example` | the nested ops token `step` ran |
| `/api/hint` | `lesson`, `index` | one hint |
| `/api/reveal` | `lesson` | the authored solution |
| `/api/suggest` | `lesson`, `program` | the adviser's next ops, with previews |
| `/api/model` | `checkpoint`, `download` | load a checkpoint; replies with the status |

`"lesson": "sandbox"` runs `examples` from the request instead, with no extra
tests. Errors are `400 {"error": ...}`.
