# Project conventions

- Keep saved checkpoints weights-only unless training resumption is explicitly planned. Preserve model/tokenizer/config, trainer metadata, and `wandb_run.url`; optimizer, scheduler, and RNG state are normally disposable.
- `wandb_run.url` should point to the checkpoint's run in the personal W&B entity so model lineage remains recoverable.
- **Record provenance; do not enforce it.** Hobby project. 
  
## Working with other agents

Several agents and sessions touch this repo. Work in a worktree so your work 
doesn't collide. Don't put the worktree under /tmp.

- **`docs/RESULTS.md` records numbers; `docs/STATUS.md` records state; issues
  record argument.** A run adds a row to `RESULTS_DETAILED.md`, to `RESULTS.md`
  only if it is a headline, and to STATUS if it changes a claim. Never the same content in two places. If
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
  code *does*; the issue accumulates what was *learned*, across runs. One PR
  per run, one issue per question.

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

- **Fast-forwarding a branch to `main` is pre-authorized** once it has a
  passing high-effort review from the other agent family and its results are
  posted on the issue. No PR is needed for that; a PR is for code another
  agent should argue with before it lands. Rewriting `main`'s history or
  force-pushing still needs the owner.
- **Review first, merge once clear.** Merge `origin/main` into the branch, do
  not rebase (a rebase needs a force-push). Land with
  `git push origin HEAD:main` from the worktree, which leaves the main
  checkout alone.
- **Lint with `uv run ./lint.sh ci`**, never bare `./lint.sh`: the pyenv ruff
  is older than CI's, and the bare form reformats instead of checking.
- **New worktree: `uv sync --extra dev --extra experiments`** before
  `uv run pytest`, or a global pytest runs and fails on imports.
- **Seven `I001` hits on `wandb` imports in the main checkout are local** (its
  `wandb/` directory). CI is green; never `--fix` them.
- **Commit messages and `gh` bodies go through a file** (`-F`, `--body-file`).
  Backticks in a double-quoted `-m` run as a command.
- **Copy run directories the reports cite into the main checkout** before the
  worktree goes.
- **Paid API spend needs the owner's go-ahead per run.** A peer agent's brief
  does not authorize it.
- **Extra seeds and extra checks are not the "next step that needs no
  permission".** The owner, 2026-09-26: do not run work that is not needed.

## Measurement traps

- `run_body`'s memo is process-global: run one process per timed cell.
- Validate a search predicate by witness replay on v4, not v3 alone (v3 has
  no closures).
- Compare data arms at matched epochs; early stopping on validation CE
  confounds the comparison with training length.
- Run the baseline through a registered gate before registering a contrast on
  it, and take one same-seed replicate as the noise floor of a thresholded
  readout (a GPU rerun is not the same run).
- Until stack-machine-quotation-growth (#172) is fixed, cap any job that rolls
  out policy-proposed programs:
  `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`.
- `tests/test_generate_deep_corpus.py::test_resumed_serial_and_parallel_payloads_are_byte_identical`
  fails under CPU contention and passes alone.
- **When a campaign ends, do not block on the owner.** Post the report, start
  the recommended next step that needs no permission, and ask any gated
  question as a message to the owner, or to the session the owner has named
  to answer for them, never as a blocking prompt. Keep the ungated work going
  until the answer arrives.

Whether the project is active, paused, or being worked in parallel with
something else is the owner's call and is not recorded here.
