"""Task-graph utilities for the proposer pilot.

``graph`` is pure Python, while ``solve_rater`` reaches the solver stack and
therefore torch and the LLM clients.  The public names are resolved lazily so
that importing the graph — for corpus generation or shortest-path work — does
not pay for the model dependencies.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wandering_light.proposer_pilot.graph import (
        ExpansionResult,
        Node,
        Task,
        TrajectoryGraph,
    )
    from wandering_light.proposer_pilot.solve_rater import SolveRater, SolveResult

__all__ = [
    "ExpansionResult",
    "Node",
    "SolveRater",
    "SolveResult",
    "Task",
    "TrajectoryGraph",
]

_MODULE_BY_EXPORT = {
    "ExpansionResult": "graph",
    "Node": "graph",
    "Task": "graph",
    "TrajectoryGraph": "graph",
    "SolveRater": "solve_rater",
    "SolveResult": "solve_rater",
}


def __getattr__(name: str) -> object:
    module_name = _MODULE_BY_EXPORT.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(
        import_module(f"wandering_light.proposer_pilot.{module_name}"), name
    )
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
