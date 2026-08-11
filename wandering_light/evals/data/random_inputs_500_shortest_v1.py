"""Certified non-identity relabeling of ``random_inputs_500.py``."""

from pathlib import Path

from wandering_light.shortest_path_data import (
    certified_specs,
    read_jsonl_gz,
)

_records = read_jsonl_gz(Path(__file__).with_suffix(".jsonl.gz"))
eval_trajectory_specs = certified_specs(_records)
