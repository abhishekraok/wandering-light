"""Write the seed-17 40-task eval sample of stack_pbe_v4 exactly as
experiments/stack_pbe_baseline.py draws it (--tasks 40 --split eval --seed 17)."""

import gzip
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

from wandering_light.stack_pbe import load

corpus = "wandering_light/evals/data/stack_pbe_v4"
tasks = load(corpus)
index_of = {id(t): i for i, t in enumerate(tasks)}
tasks = [t for t in tasks if t.split == "eval"]
random.Random(17).shuffle(tasks)
tasks = tasks[:40]
out = Path(sys.argv[1])
with gzip.open(out, "wt") as h:
    for t in tasks:
        h.write(json.dumps(asdict(t), sort_keys=True) + "\n")
print([index_of[id(t)] for t in tasks][:5], [t.behaviour for t in tasks][:5])
