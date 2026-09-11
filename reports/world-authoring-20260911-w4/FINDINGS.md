## Findings (written with the 2026-09-11 wave 4; the tables below regenerate)

- **100 of 100 verified** (50/50 MBPP, 50/50 CodeARC) after one two-op
  repair: an identity behaviour (copy a list) submitted as the empty program,
  the same case as wave 2. Over the probe and four waves MBPP stands at
  234/236 and CodeARC at 180/184.
- **Cost rose to 432k tokens** (50k–167k per agent; author d spent 167k on
  its 24) for 7 minutes wall and 20 agent-minutes. The variance is the
  behaviours, not the settings: one agent drew the long tail.
- **Library: 4 macros per 100** (125 → 129): `adjacent` (two authors, five
  programs), `differences`, `triangle` (two authors), `ceil-div` (an author in
  wave 2 had written it too). 64 one-offs inlined. The rate has fallen from 16
  to 7, 8, 4 per 100 as the pool's easier MBPP problems come up.
- **Overlap with v4 grows as the pool thins**: 9 exact witness matches
  (scalar one-liners such as `mbpp-8`, `mbpp-404`) and 11 shared skeletons in
  100. The wider-pool sampler is now reaching MBPP's simplest problems; the
  v4-eval overlap removal in the step-3 training plan matters more here.
