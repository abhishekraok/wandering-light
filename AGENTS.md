# Project conventions

- Keep saved checkpoints weights-only unless training resumption is explicitly planned. Preserve model/tokenizer/config, trainer metadata, and `wandb_run.url`; optimizer, scheduler, and RNG state are normally disposable.
- `wandb_run.url` should point to the checkpoint's run in the personal W&B entity so model lineage remains recoverable.

## Working with other agents

Several agents and sessions touch this repo. 

- **`docs/STATUS.md` records state; issues record argument.** Never both. If
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
