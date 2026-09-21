# Signal House

A local workbench for learning the repo's **mark-28** stack machine.
You repair a quiet coastal relay by writing programs, predicting their effects,
and inspecting execution. The trained policy is an optional adviser.

## Start on Windows

Double-click **`Start Signal House.cmd`** at the worktree root. It opens
<http://127.0.0.1:8018>. Keep the terminal open while using the app; Ctrl+C stops
that app only. Starting the launcher again reuses an already running workshop.

The first launch builds the frontend if necessary. Requirements:

- Python **3.12+**, matching the repository's minimum.
- Node **22.12+** and npm to install/build the frontend. Node is unnecessary
  after the build, unless you change the frontend.
- The launcher can locate the desktop app's bundled Python and Node runtimes
  when they are present on this machine.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start-stack-workshop.ps1
# Rebuild after a frontend edit:
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start-stack-workshop.ps1 -Rebuild
# Install optional CPU inference dependencies into a dedicated environment:
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start-stack-workshop.ps1 -InstallPolicy
```

Core teaching and the HTTP service use only Python's standard library. The app
binds to loopback, is not hosted publicly, and sends no example data to a model
service. The frontend assets, policy, and downloaded corpus are local.

## Start on Linux or WSL

There is no launcher; build the frontend once, then run the server.

```bash
cd web/stack-workshop && npm ci && npm run build
cd ../.. && python3 -m wandering_light.stack_workshop.server
```

Open <http://127.0.0.1:8018>; under WSL the Windows browser reaches the same
address. `static/` is untracked, so a fresh clone or worktree needs the build
step, and any frontend edit needs it again. The server itself needs no virtual
environment; the optional adviser does. Pass `--port` if another workshop is
already running.

## What is in the workshop

- **14 repairs across five chapters**: operand order and duplication; quotations
  and map/filter; accumulation, dip and branching; counterexamples and transfer;
  then the canonical macro vocabulary.
- **Vim-enabled CodeMirror editor**, instruction/macro completion, stack-effect
  help, previews across current examples, comments and source highlighting.
- **Live example stacks and reversible timeline**. Rewinding selects recorded
  states. Enter an instruction to observe the actual nested machine calls.
- **Color, motion, and optional sound**: number colors persist across views,
  short transitions reveal state changes, and stepping produces a stable pitch
  from the top number. Quotations use a different timbre. A small signal plot
  compares the current top numeric value/list to the target. Sound and animation
  are optional, and no information depends on either.
- **Custom PBE experiments** with editable input/output pairs. The outer JSON
  array is a stack, deepest first; a list value is nested inside it. For exact
  large outputs, use stack notation as a string, e.g. `"[ 1234567890123456789 ]"`.
- **Canonical macro dictionary**, read directly from
  `experiments/world_behaviours/macros.txt`, with bodies, expanded primitives,
  previews and matching source locations in the editor.
- **World-task shelf** using the pinned `stack_pbe_world_v2` manifest. It loads
  the published 183-task corpus descended from #118/#120/#127 and keeps source
  labels and extra tests. These tasks are optional later material.
- **Progress, per-repair programs, predictions, notes and saved fragments** in
  browser local storage, with JSON export/import. A fresh browser profile or
  different port has separate storage. Export before moving profiles.

## Learning design

The progression starts with partially worked repairs, fades to blank programs,
then recombines familiar ideas in unfamiliar tasks. Every lesson has a concrete
goal, visible examples, independent Python references for extra tests, a hint
ladder, and a short explanation of what the repair establishes. Later lessons
prompt retrieval of selected earlier repairs. The learner can revisit or skip
any lesson; there are no lives, timers, penalties, or forced unlock gates.

Hints progress from conceptual direction to a concrete next step. Revealing a
reference preserves the learner's attempt in the notebook. Completing a repair
means passing its finite tests; it does not certify equivalence on all inputs,
optimality, or mastery. Custom PBE tasks are explicitly labeled example fits.
Learning effectiveness and playthrough duration have not been measured.

## Trained adviser

The app uses the existing `stack_policy.load_policy` and `PolicyScorer`, with
the existing token encoder and action order. It never invents model scores or
substitutes heuristics when weights are missing. Loading is lazy: the curriculum
works before the model is ready. Install the optional dependencies above first.

In **Guide → Connect a checkpoint**, choose a folder containing
`policy_config.json`, `model.safetensors`, and the checkpoint metadata, or
download the archived `mark28-policy-2m-s1:v0` artifact. W&B authentication may
be required; sign in locally with the dedicated environment's `python -m wandb
login`, or provide an already downloaded checkpoint. Do not paste credentials
into the app. `STACK_POLICY_CHECKPOINT` and the launcher's `-Checkpoint` are
also supported. The archive defaults to the same personal W&B entity as the
existing stack debugger; no training runs or optimizer states are created.

The adviser ranks legal next instructions from the selected timeline state,
using **shown examples only**. Weights are policy preferences, not success
probabilities. Natural-language instruction explanations and campaign hints
are authored; they are not claims about the model's internal reasoning.

## Keyboard

| Action | Shortcut |
|---|---|
| Search commands, repairs and words | Ctrl+K |
| Run / check repair | Ctrl+Enter / Ctrl+Shift+Enter |
| Step backward / forward | Alt+Left / Alt+Right |
| Rewind | Alt+Home |
| Enter current instruction | Alt+I |
| Hint | Alt+H |
| Code completion | Ctrl+Space |
| Focus repair log / editor / examples / tools | Alt+1 / 2 / 3 / 4 |
| Vim pane focus | Ctrl+W then h/j/k/l, where the browser permits it |
| Vim commands | `:run`, `:step`, `:back`, `:hint`, `:check`, `:next` |
| Keyboard help outside editor | `?` |

All controls also participate in normal Tab navigation. Browser-reserved keys
can override Ctrl+W; the Alt+number bindings avoid that conflict. Ordinary Vim
motions, insert/visual modes, undo, search, and editing come from
`@replit/codemirror-vim`, not a home-grown imitation.

## Development and validation

```powershell
python -m wandering_light.stack_workshop.server
cd web/stack-workshop
npm ci
npm run dev
npm run build
npm test
```

Vite uses port 5178 and proxies `/api` to port 8018. Its build writes to
`wandering_light/stack_workshop/static/`. The compiled assets are intentionally
untracked; the lockfile and Windows launcher make the source build repeatable.

```powershell
python -m pytest tests/test_stack_workshop.py -q
```

Python tests check every campaign witness against independently authored
expected results, primitive-prefix parity with the canonical machine, macro
spans, cached nested execution, counterexamples, invalid data, and the HTTP
contract. DOM tests run the actual frontend against a temporary Python server
to exercise keyboard commands, lesson switching, custom examples, instruction
inspection and saving. They test behavior, not browser layout or sound quality.

Execution has bounded fuel, input sizes and expanded-program length. The
machine's `run_body` cache is cleared before instruction inspection, under a
shared lock, so cached results do not hide nested calls. Inspection records the
first 350 completed calls and labels truncation. Large stacks and nested
quotations show explicit display omissions while computation retains every
value. The 183 published reference programs were checked through the app's
verifier, including their extra tests. Arbitrary user programs can still exceed
the interactive fuel/size limits; these are app limits, not expressibility claims.

Where the host supports the proposed WebMCP document API, the app registers
`read_stack_workshop` and `stage_stack_program`. They use the same state and
actions as the interface. Their registration contract is covered by the DOM
tests; a native host WebMCP context has not been verified.

Design references: Bret Victor's [Learnable Programming](https://worrydream.com/LearnableProgramming/),
[CodeMirror completion](https://codemirror.net/examples/autocompletion/), and
[CodeMirror Vim](https://github.com/replit/codemirror-vim).
