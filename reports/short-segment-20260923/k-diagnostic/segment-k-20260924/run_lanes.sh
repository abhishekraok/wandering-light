#!/usr/bin/env bash
# K diagnostic (#168), s0 of each policy, 40 programs per set, GPU. Code: claude/campaign-code (uncommitted segment_k_diagnostic.py).
cd /home/abhishekrao/repos/wandering-light/.claude/worktrees/campaign-code
k() { uv run python experiments/segment_k_diagnostic.py --checkpoint $1 --set $2 --limit 40 --device cuda --out /home/abhishekrao/repos/wandering-light/checkpoints/segment-k-20260924/$3 >> /home/abhishekrao/repos/wandering-light/checkpoints/segment-k-20260924/$3-$2.out 2>&1; }
(k /home/abhishekrao/repos/wandering-light/checkpoints/short-segment-difference-20260923/policies/capped-s0-e3 capped-train capped-difference; k /home/abhishekrao/repos/wandering-light/checkpoints/short-segment-difference-20260923/policies/capped-s0-e3 yardstick-167 capped-difference) &
(k /home/abhishekrao/repos/wandering-light/checkpoints/short-segment-machine-20260923/policies/capped-s0-e3 capped-train capped-machine; k /home/abhishekrao/repos/wandering-light/checkpoints/short-segment-machine-20260923/policies/capped-s0-e3 yardstick-167 capped-machine) &
(k /home/abhishekrao/repos/wandering-light/checkpoints/staged-arms-20260920/policies/fused-s0-e3 fused-train fused-machine; k best yardstick-167 best) &
wait
touch /home/abhishekrao/repos/wandering-light/checkpoints/segment-k-20260924/LANES_DONE
