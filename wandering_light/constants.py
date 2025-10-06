"""
Common constants used across the wandering_light package.
"""

from enum import StrEnum

# Evaluation file path
DEFAULT_EVAL_FILE = "wandering_light/evals/data/random_inputs.py"
# TODO Upload this to the internet
DEFAULT_SOLVER_CHECKPOINT = "checkpoints/saved/rl/long_sft_opt_125m_s35k_no_len"


class Task(StrEnum):
    """Supported training tasks."""

    INDUCTION = "induction"
    PROPOSER = "proposer"
    DUAL = "dual"
