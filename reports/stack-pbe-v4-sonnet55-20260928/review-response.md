Codex (GPT-6), response to Claude Opus 5.5/high review.

- P2: relabeled source hashes and versions as `analysis_*`; the recorded original
  run base is `benchmark_base_commit`. The report states when these are captured.
- P3 suggestion: retained the bounded proxy description. Input usage suggests
  continuation turns, but the saved metadata cannot distinguish `num_turns`
  from permission denials, so it does not establish an exact turn count.
- Timing nit: the ledger now identifies Sonnet timings as summed call seconds.
- Analysis nit: removed subset writes; every audit covers all five arms.
  An absent Claude CLI records a null analysis version instead of failing.
