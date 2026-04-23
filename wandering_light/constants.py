"""
Common constants used across the wandering_light package.
"""

from enum import StrEnum

# Evaluation file paths.
# DEFAULT_EVAL_FILE is the small (100 spec) set used for online evals during
# training where fast feedback matters more than statistical power.
# STANDALONE_EVAL_FILE is the larger (500 spec) set used for standalone CLI
# evaluations where comparability across solver/proposer runs matters.
DEFAULT_EVAL_FILE = "wandering_light/evals/data/random_inputs.py"
STANDALONE_EVAL_FILE = "wandering_light/evals/data/random_inputs_500.py"
DEFAULT_SOLVER_CHECKPOINT = "abhishekraok/induction-basicfns-opt125m-sft434k-rl-6k-with-lp"


class Task(StrEnum):
    """Supported training tasks."""

    INDUCTION = "induction"
    PROPOSER = "proposer"
    DUAL = "dual"
