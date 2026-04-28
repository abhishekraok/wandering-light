"""Compute solve-rates for (src, dst) tasks using a TrajectorySolver."""

from dataclasses import dataclass

from wandering_light.common_functions import basic_fns
from wandering_light.function_def import FunctionDefSet
from wandering_light.proposer_pilot.graph import Task
from wandering_light.solver import TrajectorySolver, create_bfs_solver
from wandering_light.typed_list import TypedList


@dataclass(frozen=True)
class SolveResult:
    src_tl: TypedList
    dst_tl: TypedList
    n_attempts: int
    n_solved: int

    @property
    def rate(self) -> float:
        return self.n_solved / self.n_attempts if self.n_attempts else 0.0


class SolveRater:
    def __init__(
        self,
        solver: TrajectorySolver | None = None,
        functions: FunctionDefSet | None = None,
        n_attempts: int = 1,
    ):
        self.solver = solver if solver is not None else create_bfs_solver()
        self.functions = functions if functions is not None else basic_fns
        self.n_attempts = n_attempts

    def rate(
        self,
        src_tl: TypedList,
        dst_tl: TypedList,
        n_attempts: int | None = None,
    ) -> SolveResult:
        n = n_attempts if n_attempts is not None else self.n_attempts
        problems = [(src_tl, dst_tl)] * n
        results = self.solver.solve_batch(problems, self.functions)
        n_solved = sum(1 for r in results if r.success)
        return SolveResult(
            src_tl=src_tl, dst_tl=dst_tl, n_attempts=n, n_solved=n_solved
        )

    def rate_tasks(
        self, tasks: list[Task], n_attempts: int | None = None
    ) -> list[SolveResult]:
        n = n_attempts if n_attempts is not None else self.n_attempts
        problems: list[tuple[TypedList, TypedList]] = []
        for t in tasks:
            problems.extend([(t.src_tl, t.dst_tl)] * n)
        results = self.solver.solve_batch(problems, self.functions)
        out: list[SolveResult] = []
        for i, t in enumerate(tasks):
            chunk = results[i * n : (i + 1) * n]
            n_solved = sum(1 for r in chunk if r.success)
            out.append(
                SolveResult(
                    src_tl=t.src_tl,
                    dst_tl=t.dst_tl,
                    n_attempts=n,
                    n_solved=n_solved,
                )
            )
        return out
