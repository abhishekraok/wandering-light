# Fused-policy transfer to staged text (staged-witness-reauthoring #155, clear-stack-pbe-v4 #99)

**This is a transfer measurement, not a difficulty measurement.** The checkpoint
(`witness-s0-e4`, the 1m v4-only arm of #101) was trained only on `stack_pbe_v4`
witnesses, which are write-then-run. A staged witness closes a short body every
few ops and runs it at once, so the policy has never seen a `]` four tokens in
followed by a combinator and another `[`. Low probability there is unfamiliarity
as much as difficulty, and a fused-trained policy cannot separate the two. The
`L x deficit` figures below are therefore **out of distribution** and do **not**
estimate the deficit a staged-trained policy would show, which is what the
`ln W` break-even argument needs. The real test needs the full staged corpus;
18 witnesses cannot train a policy.

Driver `experiments/staged_transfer.py`; staged programs from the #155 pilot
(`pilot_assessed.jsonl`, 18 of 20 behaviours staged). Each behaviour is scored
twice against the same v4 train task -- same inputs, same final target -- once
on its fused witness and once on its staged rewrite. Tolerance is `ln 64` =
4.159 nats, spent once over the whole witness.

## Headline

| | median length | median per-step deficit | median accumulated `L x d` | within `ln W` |
|---|---:|---:|---:|---:|
| fused (these 18) | 43.5 | 0.086 | 3.5 | 11 / 18 |
| **staged (these 18)** | 39.5 | **3.392** | **162.1** | **0 / 18** |

Staged text costs about **39x the per-step deficit** of its fused twin under
this policy, and no staged witness fits inside the beam's tolerance. Staged is
also slightly *shorter* at the median, so this is not a length effect.

## The familiarity control, which is why the headline is not a verdict

The pilot behaviours are v4 **train** behaviours, so their fused witnesses are
in this policy's training set. Against 40 random fused witnesses per split at
the same length profile (12-98 ops):

| | median per-step deficit | median `L x d` | within `ln W` |
|---|---:|---:|---:|
| fused, train split (policy saw these) | 0.075 | 3.2 | 23 / 40 |
| fused, eval split (unseen content, familiar style) | 0.535 | 24.6 | 4 / 40 |
| staged (unseen content, unseen style) | 3.392 | 162.1 | 0 / 18 |

Unseen *content* in the familiar style costs about **7x** the deficit. Unseen
*style* costs about **45x**, roughly **6x** more again. So the staged penalty is
far larger than held-out-ness alone explains -- but "unseen style" is exactly
what a fused-trained policy is guaranteed to punish, so the excess is not
evidence that staged programs are intrinsically harder.

## The step-kind split, which is the part that was meant to be readable

The diagnostic registered in advance: if deficit concentrates at the cycle
boundary (`]`, the combinator, the next `[`) while tokens *inside* a body stay
cheap, that is weak evidence short bodies are cheaper steps -- the mechanism
staging is supposed to exploit. If body tokens are expensive too, it is only OOD.

| step kind | fused steps | fused mean deficit | staged steps | staged mean deficit |
|---|---:|---:|---:|---:|
| `body` (under an open mark) | 400 | 0.107 | 298 | **5.302** |
| `open` (`[`) | 133 | 0.091 | 131 | 0.966 |
| `close` (`]`) | 133 | 0.025 | 131 | 2.557 |
| `run` (combinator or `i`) | 28 | 0.106 | 92 | 2.617 |
| `direct` (top-level op) | 125 | 0.356 | 151 | 2.764 |

**The diagnostic returns a clean negative.** Body tokens are the *most*
expensive staged step kind, not the cheapest -- 5.302 nats against 2.5-2.8 at
the boundary. One `body` step alone exceeds the whole `ln 64` budget. So this
run gives **no** evidence that short bodies are cheaper steps for the policy,
and the pattern is the one predicted for plain OOD.

A detail worth keeping: `open` is cheap even in staged text (0.966). The policy
is happy to *start* a quotation anywhere; what it cannot predict is a short
body's contents, having been trained that `[` is followed by a long one.

## What this does and does not license

- It does **not** say staging raises deficit. It says a fused-trained policy
  cannot read staged text, which was expected.
- It does **not** move the break-even arithmetic of #99. The question there --
  whether a staged-trained policy has a lower per-step deficit, so that a
  roughly equal-length staged witness fits inside `ln W` -- is untouched.
- It **does** remove the cheap optimistic reading. Had body tokens come out
  cheap, that would have been weak early support for the staging mechanism.
  They did not, so there is no early support to have.
- It **does** price one cost of the plan: any staged corpus must be trained on,
  not merely authored. Reusing an existing fused-trained checkpoint on staged
  text is worthless, and this run bounds how worthless.
