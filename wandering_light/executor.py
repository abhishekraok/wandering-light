import builtins
from dataclasses import dataclass
from types import FunctionType
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
    def __init__(
        self,
        available_functions: FunctionDefSet | list[FunctionDef],
        *,
        enforce_membership: bool = True,
    ):
        if isinstance(available_functions, list):
            functions = available_functions
        else:
            functions = available_functions.functions
        # Membership is a construction-time snapshot. Runtime counters on each
        # FunctionDef may change, but the executable palette itself may not.
        self.available_functions = tuple(functions)
        self._available_by_name: dict[str, FunctionDef] = {}
        self._registered_signatures: dict[str, tuple[object, ...]] = {}
        self._compiled: dict[tuple[str, str, str, str], FunctionType] = {}
        self.enforce_membership = enforce_membership
        for function in self.available_functions:
            existing = self._available_by_name.get(function.name)
            if existing is not None and self._definition_signature(
                existing
            ) != self._definition_signature(function):
                raise ValueError(
                    f"Conflicting executable definitions for {function.name!r}"
                )
            self._available_by_name[function.name] = function
            self._registered_signatures[function.name] = self._definition_signature(
                function
            )

    @staticmethod
    def _definition_signature(function: FunctionDef) -> tuple[object, ...]:
        metadata = function.metadata or {}
        return (
            function.name,
            function.input_type,
            function.output_type,
            function.code,
            metadata.get("basis_function_id"),
            metadata.get("basis_function_fingerprint"),
            metadata.get("basis_set_id"),
            metadata.get("basis_set_digest"),
        )

    def _registered_function(self, fn_def: FunctionDef) -> FunctionDef:
        if not self.enforce_membership:
            return fn_def
        registered = self._available_by_name.get(fn_def.name)
        if registered is None:
            raise ValueError(
                f"Function {fn_def.name!r} is not in this executor's basis set"
            )
        if registered != fn_def:
            raise ValueError(
                f"Function {fn_def.name!r} does not match the registered definition"
            )
        if (
            self._definition_signature(registered)
            != self._registered_signatures[registered.name]
        ):
            raise ValueError(
                f"Registered function {registered.name!r} was mutated after "
                "executor construction"
            )
        return registered

    def _compile(self, fn_def: FunctionDef) -> FunctionType:
        cache_key = (fn_def.name, fn_def.input_type, fn_def.output_type, fn_def.code)
        cached = self._compiled.get(cache_key)
        if cached is not None:
            return cached

        safe_globals = {
            "__builtins__": {
                **vars(builtins),
                "open": lambda *a, **k: (_ for _ in ()).throw(OSError("I/O disabled")),
            },
            "TypedList": TypedList,
        }
        local_namespace: dict[str, object] = {}
        exec(fn_def.executable_code(), safe_globals, local_namespace)
        function = local_namespace[fn_def.name]
        if not isinstance(function, FunctionType):
            raise TypeError(f"Compiled {fn_def.name!r} is not a function")
        self._compiled[cache_key] = function
        return function

    def execute(self, fn_def: FunctionDef, inputs: TypedList) -> TypedList:
        fn_def = self._registered_function(fn_def)

        # 1. Type‑check the incoming list
        if inputs.item_type is not fn_def.input_type_cls():
            raise TypeError(f"Expected {fn_def.input_type}, got {inputs.item_type}")

        # 2. Compile each immutable definition once per executor.
        fn = self._compile(fn_def)

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
        from wandering_light.function_def import FunctionDefList
        from wandering_light.trajectory import Trajectory, TrajectorySpec

        if not spec.function_defs:
            # Empty trajectory - always succeeds
            trajectory = Trajectory(spec, spec.input)
            return TrajectoryResult.success_result(trajectory)

        current = spec.input
        canonical_functions: list[FunctionDef] = []
        for i, fn in enumerate(spec.function_defs):
            try:
                registered = self._registered_function(fn)
                current = self.execute(registered, current)
                canonical_functions.append(registered)
            except TypeError as e:
                return TrajectoryResult.failure_result(
                    f"Type mismatch at function '{fn.name}': {e}", failed_at_step=i
                )
            except Exception as e:
                return TrajectoryResult.failure_result(
                    f"Function '{fn.name}' failed: {e}", failed_at_step=i
                )

        # A predictor may hand us an equal clone whose metadata is not the
        # registered basis provenance.  Execution already used the registered
        # object; return that same canonical sequence so downstream usage and
        # artifact tracking can never persist metadata supplied by the clone.
        canonical_spec = TrajectorySpec(
            spec.input,
            FunctionDefList(canonical_functions),
        )
        trajectory = Trajectory(canonical_spec, current)
        return TrajectoryResult.success_result(trajectory)
