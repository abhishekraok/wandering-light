"""BEST with the library off, and the #155 fused-s0-e3 checkpoint under BEST's search
(library on and off), on the seed-17 40-task v4 eval sample. CPU."""

from dataclasses import replace
from pathlib import Path

from wandering_light.solve import BEST, solve_file

L = Path("reports/stack-pbe-v4-effort-sweep-20260927/local")
tasks = L / "sample40_seed17_v4_eval.jsonl.gz"
FUSED = Path(
    "/home/abhishekrao/repos/wandering-light/checkpoints/staged-arms-20260920/policies/fused-s0-e3"
)
cells = [
    ("best_staged_s0_e3_nolib", replace(BEST, library=False), None),
    ("fused_s0_e3_lib", replace(BEST, name="fused-s0-e3"), FUSED),
    ("fused_s0_e3_nolib", replace(BEST, name="fused-s0-e3", library=False), FUSED),
]
for name, system, ckpt in cells:
    with (
        open(L / f"{name}_sample40.txt", "w") as out,
        open(L / f"{name}_sample40.log", "w") as log,
    ):
        s = solve_file(
            tasks,
            system=system,
            checkpoint=ckpt,
            json_path=L / f"{name}_sample40.json",
            out=out,
            log=log,
        )
    print(name, s.line(), flush=True)
