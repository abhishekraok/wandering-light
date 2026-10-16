# Project conventions

- Keep saved checkpoints weights-only unless training resumption is explicitly planned. Preserve model/tokenizer/config, trainer metadata, and `wandb_run.url`; optimizer, scheduler, and RNG state are normally disposable.
- `wandb_run.url` should point to the checkpoint's run in the personal W&B entity so model lineage remains recoverable.
- **Record provenance; do not enforce it.** Hobby project. Record commit, dirty
  flag, config, data digests, seed and W&B URL in every artifact. Never add
  checks that refuse to run or load on a mismatch; report it as a caveat. In
  review, an experiment driver much larger than the module it drives is a
  finding.
- `README.md` is for a first-time human visitor (what, why, how to run); agent
  conventions go here.
- Interactive tools (debuggers, explorers) must feel instant: navigate
  client-side over precomputed data, keep models warm in one long-lived
  process, no page reloads.

## Working with other agents

Several agents and sessions touch this repo. Work in a worktree so your work 
doesn't collide. Don't put the worktree under /tmp. "The main checkout" is
UltraDesk WSL `~/repos/wandering-light`, which holds `checkpoints/`, `.drafts/`
and `wandb/`.

- **`docs/RESULTS.md` records numbers; `docs/STATUS.md` records state; issues
  record argument.** A run adds a row to `docs/RESULTS_DETAILED.md`, to `RESULTS.md`
  only if it is a headline, and to STATUS if it changes a claim. Apart from the
  headline copy, never the same content in two places. If
  you establish something, add a row there — a claim that lives only in a
  comment gets contradicted by the next comment. Update it as part of the
  work, not afterwards.
- **Close an issue only when its evidence is on `main`.** Citing a report that
  exists only on an unmerged branch means the justification disappears when
  the branch is pruned.
- **Whoever opens a PR owns its lifecycle.** Review other people's, comment on
  them, do not merge or close them. Another agent may still be writing to it.
- **Post results on the issue, not in the PR.** PRs die on merge and results
  should outlive the code review that carried them. A PR body says what the
  code *does*; the issue accumulates what was *learned*, across runs. One
  issue per question.

When you are wrong about something that has been acted on, correct it in
`docs/STATUS.md` under Superseded as well as in the thread. A correction in a
comment does not reach someone reading the file.

## How much process a change needs

Not every task needs an issue and a PR. Match the ceremony to what the
change settles:

- **A measurement, a new corpus, a solver change, anything another agent
  will argue with:** issue first, results on the issue, one PR or one
  reviewed branch per run (see below).
- **Small and self-evident** (docs wording, a lint fix, a test for existing
  behaviour, a convention note like this one): commit straight to `main`
  from a worktree, with a message that says what changed. No issue, no PR.
- In between, prefer less. The point of the conventions above is that
  results outlive the code review that carried them, not that every change
  gets a review.

## Landing work and asking the owner

- **Pushing a branch needs no confirmation.**
- **Fast-forwarding a branch to `main` is pre-authorized** once it has a
  passing high-effort review from the other agent family and its results are
  posted on the issue. No PR is needed for that; a PR is for code another
  agent should argue with before it lands. Rewriting `main`'s history or
  force-pushing still needs the owner.
- **Merge, then review, then push.** Fetch and merge `origin/main` into the
  branch (do not rebase; a rebase needs a force-push) before the final checks
  and the other-family review. Review again anything substantive changed after
  that. Land with `git push origin HEAD:main` from the worktree, which leaves
  the main checkout alone. If the push is rejected, fetch, merge and repeat
  the affected checks; never force.
- **Lint with `uv run ./lint.sh ci`**, never bare `./lint.sh`: the pyenv ruff
  is older than CI's, and the bare form reformats instead of checking.
- **New worktree: `uv sync --extra dev --extra experiments`** before
  `uv run pytest`, or a global pytest runs and fails on imports.
- **Seven `I001` hits on `wandb` imports in the main checkout are local** (its
  `wandb/` directory). CI is green; never `--fix` them.
- **Commit messages and `gh` bodies go through a file** (`-F`, `--body-file`).
  Backticks in a double-quoted `-m` run as a command.
- **Write run output into a directory in the main checkout from launch**
  (`checkpoints/<run>/`), not the worktree: closing a session can delete its
  worktree. Before landing a report, check every path it cites exists there.
- **Metered API keys (OpenAI, Anthropic, Gemini) need the owner's go-ahead
  per run.** A peer agent's brief does not authorize it. Plan-billed
  `claude -p` / `codex exec` with those keys unset needs none.
- **The sample-40 model sweeps** (new-models-sample40 (#173) and its
  extensions) are for the owner's own view of how agents are progressing, not a
  project result: keep them cheap, and do not run Fable at max.
- **Do not tune against the 44 world-v3 tasks** (owner, 2026-09-22); a smaller
  world eval may replace them.
- **When a campaign ends, do not block on the owner.** Post the report, start
  the recommended next step that needs no permission, and ask any gated
  question as a message to the owner, or to the session the owner has named
  to answer for them, never as a blocking prompt. Keep the ungated work going
  until the answer arrives.
- **When a result lands, put the figure (the PNG) and a few lines in front of
  the owner**; report markdown is for agents. If the work has an explainer
  (listed in `docs/STATUS.md` § Where things live), update it in place; never
  publish a second copy.

## Measurement traps

- **Seeds.** Run extra seeds when a registered reading depends on the spread.
  Do not run them to fill idle time once the seeds agree; the next question
  is the better use of the GPU.
- `run_body`'s memo is process-global: run one process per timed cell.
- Validate a search predicate by witness replay on v4, not v3 alone (v3 has
  no closures).
- Compare data arms at matched epochs; early stopping on validation CE
  confounds the comparison with training length.
- When a readout needs an open gate at baseline, check the gate is open there
  before registering a contrast on it. For a thresholded readout, use a
  same-seed replicate as the noise floor, or cite an existing one (a GPU rerun
  is not the same run).
- Until stack-machine-quotation-growth (#172) is fixed, cap any job that
  executes generated stack programs (policy rollouts, random walks, search):
  `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 -- <command>`.
  A run killed by the cap is incomplete: never retry it uncapped or score its
  missing output as zero.
- The beam and the grader charge fuel differently: `stack_beam` gives every op a
  fresh `DEFAULT_FUEL_LIMIT`, while `stack_pbe.run_program` charges the whole
  program against one. A program the beam accepts can fail grading on fuel.
- `tests/test_generate_deep_corpus.py::test_resumed_serial_and_parallel_payloads_are_byte_identical`
  has failed under CPU contention. Rerun it alone; if it fails alone, it is a
  real failure.

Whether the project is active, paused, or being worked in parallel with
something else is the owner's call and is not recorded here.
