# Score-gap feasibility: literature and source audit

The program-synthesis janthu (e-den, Codex) wrote this, 2026-10-03 PDT.
Reading only; no foreign code executed or installed, model loaded or corpus opened.

[Freitag and Al-Onaizan (2017)](https://aclanthology.org/W17-3207.pdf), section4,
apply pruning after bounded beam selection. Their absolute-threshold condition
uses a different equality convention from our inclusive band. Their section5
uses an in-house NMT implementation; the prior search did not recover public
code. This is precedent for a heuristic, not a correctness or useful-branch
certificate. We retain the prior full reading audit in the desk's
work/long-horizon-evidence-audit.md, with published/source links.

[Yang et al. (2020)](https://aclanthology.org/2020.emnlp-main.366.pdf), sections3–4
and AppendixA.1, separate variable-width pruning from streaming GPU batching.
Their variable-width arm combines a score band with per-parent candidate caps.
The paper acknowledges different candidate-finalization rules from Fairseq;
matching top-1 quality does not imply identical output or oracle diversity.
Variable batching is slower than fixed width in the small-beam MT setting and
both parsing tables despite fewer expansions; streaming addresses utilization.
Thus their wall-clock findings do not predict this serial CPU solver's speed.

Read authors' [implementation](https://github.com/yangkevin2/emnlp2020-stream-beam-mt/tree/f6d331241935455c545f43106b5cae8cddda7f14),
exact commitf6d331241935455c545f43106b5cae8cddda7f14. Source retained in main
checkpoints/widening-reading-20261003/stream-beam-mt; SHA256 inventory beside this
report. No requirements installed and no source executed. README points to
variable_stream.py and maps ap/mc/eps to pruning/batching parameters.

Source observations: variable_stream.py:13 selects all minimum-progress inputs;
:200ff refills against source-token occupancy. :339ff takes a per-parent top-mc
before global top-k; :385 uses max(best active,best already finished) as its
absolute-band anchor and strict greater-than retention. :361ff records scored
EOS-completed candidates with length normalization; :464ff sorts them before
return. sequence_generator.py:295/304 route one_batch and streaming through the
same implementation. These contracts differ from guard-filtered first eligible
anchors, fit-before-score, cross-depth typed seen and first shown-fit return in
wandering_light/stack_beam.py:361–483. No theorem or numeric threshold transfers.

[Meister, Vieira and Cotterell (2020)](https://aclanthology.org/2020.tacl-1.51.pdf)
and [authors' source](https://github.com/rycolab/bfbs/tree/0f2a4de58407499692000a44e55a59d56e3a3eae)
were read in the earlier audit. Their completed-hypothesis/per-length-expansion
contract motivates a different mechanism, not this study. That branch remains
deferred. The independently verified #179 semantics is the relevant contract.

Design consequence: first screen the relationship between eligible cumulative
score gaps, rank/depth and *known positive reference lineages*. Do not treat other
states as negatives, off-path witnesses as adverse rankings or fewer snapshot
slots as executed work. A narrow prefix on the same snapshots supplies a rank
control; actual narrower-beam coverage/clock remains a later independent arm.
Cross-depth seen means even preserving every recorded ancestor on reference
snapshots cannot prove the band would return that program when executed.
