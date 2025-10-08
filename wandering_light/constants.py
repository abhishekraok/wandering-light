"""
Common constants used across the wandering_light package.
"""

from enum import StrEnum

# Evaluation file path
DEFAULT_EVAL_FILE = "wandering_light/evals/data/random_inputs.py"
DEFAULT_SOLVER_CHECKPOINT = "abhishekraok/induction-basicfns-opt125m-longsft"


class Task(StrEnum):
    """Supported training tasks."""

    INDUCTION = "induction"
    PROPOSER = "proposer"
    DUAL = "dual"
