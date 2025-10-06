from dataclasses import dataclass
from typing import Optional

from wandering_light.function_def import FunctionDef, FunctionDefSet
from wandering_light.typed_list import TypedList


@dataclass
class TrajectoryResult:
    """Result of executing a trajectory - can be success or failure."""

    success: bool
    trajectory: Optional["Trajectory"] = None
    error_msg: str | None = None
    failed_at_step: int | None = None  # Which function step failed (0-indexed)

    @classmethod
    def success_result(cls, trajectory: "Trajectory") -> "TrajectoryResult":
        return cls(success=True, trajectory=trajectory)

    @classmethod
    def failure_result(
        cls, error_msg: str, failed_at_step: int | None = None
    ) -> "TrajectoryResult":
        return cls(success=False, error_msg=error_msg, failed_at_step=failed_at_step)


class Executor:
    def __init__(self, available_functions: FunctionDefSet | list[FunctionDef]):
        if isinstance(available_functions, list):
            self.available_functions = available_functions
        else:
            self.available_functions = available_functions.functions

    def execute(self, fn_def: FunctionDef, inputs: TypedList) -> TypedList:
        # 1. Type‑check the incoming list
        if inputs.item_type is not fn_def.input_type_cls():
            raise TypeError(f"Expected {fn_def.input_type}, got {inputs.item_type}")

        # 2. Build a tiny wrapper around the expression
        wrapper = fn_def.executable_code()

        # 3. Exec in a sandbox
        safe_globals = {
            "__builtins__": {
                **__builtins__,
                "open": lambda *a, **k: (_ for _ in ()).throw(OSError("I/O disabled")),
            },
            "TypedList": TypedList,
        }
        loc = {}
        exec(wrapper, safe_globals, loc)
        fn = loc[fn_def.name]

        # 4. Run it!
        result = TypedList(
            [fn(x) for x in inputs.items], item_type=fn_def.output_type_cls()
        )
        if not isinstance(result, TypedList):
            raise TypeError("Must return a TypedList")
        if result.item_type is not fn_def.output_type_cls():
            raise TypeError(
                f"Expected output_type {fn_def.output_type}, got {result.item_type}"
            )

        # 5. Track usage
        fn_def.increment_usage()

        return result

    def execute_trajectory(self, spec: "TrajectorySpec") -> "TrajectoryResult":
        """
        Execute a TrajectorySpec and return a result indicating success or failure.

        Type mismatches and function execution errors are expected and handled gracefully.

        Args:
            spec: The trajectory specification to execute

        Returns:
            TrajectoryResult with either the successful trajectory or error details
        """
        # Avoid circular import
        from wandering_light.trajectory import Trajectory

        if not spec.function_defs:
            # Empty trajectory - always succeeds
            trajectory = Trajectory(spec, spec.input)
            return TrajectoryResult.success_result(trajectory)

        current = spec.input
        for i, fn in enumerate(spec.function_defs):
            try:
                current = self.execute(fn, current)
            except TypeError as e:
                return TrajectoryResult.failure_result(
                    f"Type mismatch at function '{fn.name}': {e}", failed_at_step=i
                )
            except Exception as e:
                return TrajectoryResult.failure_result(
                    f"Function '{fn.name}' failed: {e}", failed_at_step=i
                )

        trajectory = Trajectory(spec, current)
        return TrajectoryResult.success_result(trajectory)
