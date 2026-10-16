# Running the train-only screen

Read DESIGN.md and AMENDMENT_A.md together. Nothing has run on real inputs yet.
The source, tests and this procedure need high-effort other-family review before
execution. The operative candidate comparison is registered only after baseline
feasibility passes; review is not that registration.

From this frozen worktree on UltraDesk, use the existing main-checkout Python
with this tree on PYTHONPATH. Do not change the shared editable install.

```bash
PYTHONPATH="$PWD" /home/abhishekrao/repos/wandering-light/.venv/bin/python \
  -m experiments.gumbel_retention_run phase --phase baseline \
  --out /home/abhishekrao/repos/wandering-light/checkpoints/gumbel-retention-screen-20261015/baseline
```

This is the inner command only. Commit before launch and run it as the persistent
`wl-gumbel-baseline-20261015` user unit, with RuntimeMaxSec=1200, MemoryMax=12G
and MemorySwapMax=0. Create the wrapper-log parent first; keep its log outside
the newly created baseline output directory. Append and inspect the wrapper
exit status; a complete summary alone without wrapper exit0 is not a result.
Log the exact launcher/revision/paths/expected end in the desk's work/runs.md.
Record actual systemd/cgroup controls at launch. Inspect the job at ten minutes.

After baseline: post counts, repeat variation and the gate result to #183.
NO_HEADROOM / BASELINE_UNSTABLE parks this panel without a candidate run.
READY_FOR_REGISTRATION permits posting the unchanged prospective operative
comparison, then running `--phase comparison --baseline <baseline directory>
--registration-url <that issue comment> --out <fresh comparison directory>`
under its own 2400-second/12G/no-swap unit. No automatic comparison follows.

`shown.json` contains only source index and shown pairs. Each isolated cell reads
that file and the fixed checkpoint. Full train tasks/references remain in the
parent and are used only after all timing cells in that phase end. Returned
programs are replayed and graded, then checked with the existing 10k-draw helper;
source/input/checkpoint provenance, actual module origins, thread counts,
whole-process clock and all-attempt solve clocks are retained. Check module
origins in the completed cell files before interpreting output.

Focused tests use manually typed stacks and dummy models/files. The runner
integration tests explicitly mock source/model/runtime acquisition. Their pass
is software evidence, never eligibility evidence for real inputs or timings.
No new trace codec, model or independent whole-search verifier is introduced.
