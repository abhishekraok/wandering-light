"""Evaluation helpers with heavyweight model imports resolved on demand."""

from typing import Any

__all__ = ["evaluate_model_checkpoint_with_trajectories"]


def __getattr__(name: str) -> Any:
    if name == "evaluate_model_checkpoint_with_trajectories":
        from .model_eval import evaluate_model_checkpoint_with_trajectories

        globals()[name] = evaluate_model_checkpoint_with_trajectories
        return evaluate_model_checkpoint_with_trajectories
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(__all__)
