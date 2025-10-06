import os
import random
from datetime import datetime

from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefList, FunctionDefSet
from wandering_light.typed_list import TypedList


class TrajectorySpec:
    """
    Plan of functions to apply: holds input list and function_defs without executing.
    """

    def __init__(self, input_list: TypedList, function_defs: FunctionDefList):
        self.input = input_list
        self.function_defs = function_defs

    def __eq__(self, other):
        return (
            isinstance(other, TrajectorySpec)
            and self.input == other.input
            and self.function_defs == other.function_defs
        )

    def __repr__(self):
        function_defs_str = self.function_defs.to_string()
        return f"TrajectorySpec({self.input} -> {function_defs_str})"

    @staticmethod
    def create_random_walk(
        input_list: TypedList, path_length: int, available_functions: list[FunctionDef]
    ) -> "TrajectorySpec":
        """Generate a random walk spec by selecting compatible fns (no execution)."""
        functions: list[FunctionDef] = []
        # start with the input type
        current_type = input_list.item_type
        for _ in range(path_length):
            sig = f"{current_type.__module__}.{current_type.__qualname__}"
            candidates = [f for f in available_functions if f.input_type == sig]
            if not candidates:
                break
            fn = random.choice(candidates)
            functions.append(fn)
            # update type only, no execution
            current_type = fn.output_type_cls()
        return TrajectorySpec(input_list, FunctionDefList(functions))

    @staticmethod
    def parse_from_string(
        repr_string: str, available_functions: FunctionDefSet
    ) -> "TrajectorySpec":
        """
        Parse a TrajectorySpec from its string representation.

        Args:
            repr_string: String like "TrajectorySpec(TL<int>([1, 2, 3]) -> add_one, double)"
            available_functions: FunctionDefSet to look up function names

        Returns:
            TrajectorySpec parsed from the string

        Raises:
            ValueError: If the string format is invalid or functions are not found
        """
        # Basic format validation
        if not repr_string.startswith("TrajectorySpec(") or not repr_string.endswith(
            ")"
        ):
            raise ValueError(f"Invalid TrajectorySpec format: {repr_string}")

        # Remove the TrajectorySpec( and ) wrapper
        content = repr_string[len("TrajectorySpec(") : -1]

        # Split on " -> " to separate input and functions
        parts = content.split(" -> ")
        if len(parts) != 2:
            raise ValueError(f"Expected format 'input -> functions', got: {content}")

        input_part, function_part = parts

        # Parse the input part (TypedList)
        # Format: TL<int>([1, 2, 3])
        try:
            input_list = TypedList.parse_from_repr(input_part.strip())
        except Exception as e:
            raise ValueError(
                f"Failed to parse input TypedList '{input_part}': {e}"
            ) from e

        # Parse the function names using available_functions
        try:
            parsed_functions = available_functions.parse_string(function_part.strip())
            if len(parsed_functions) == 0 and function_part.strip():
                raise ValueError(f"No functions found for: {function_part}")
        except Exception as e:
            raise ValueError(f"Failed to parse functions '{function_part}': {e}") from e

        return TrajectorySpec(input_list, parsed_functions)


class TrajectorySpecList:
    """
    A collection of TrajectorySpec objects with pretty printing capabilities.
    """

    def __init__(self, trajectory_specs: list[TrajectorySpec] | None = None):
        self.specs = trajectory_specs or []

    def append(self, spec: TrajectorySpec):
        """Add a TrajectorySpec to the list."""
        self.specs.append(spec)

    def extend(self, specs: list[TrajectorySpec]):
        """Add multiple TrajectorySpecs to the list."""
        self.specs.extend(specs)

    def shuffle(self, seed: int | None = None):
        """Shuffle the underlying list of specs for random ordering."""
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(self.specs)
        else:
            random.shuffle(self.specs)

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, index):
        return self.specs[index]

    def __iter__(self):
        return iter(self.specs)

    def __repr__(self):
        if not self.specs:
            return "TrajectorySpecList([])"

        lines = ["TrajectorySpecList(["]
        for i, spec in enumerate(self.specs):
            # Get function chain
            funcs = spec.function_defs.to_string()
            # Format input nicely
            input_repr = f"{spec.input.item_type.__name__}({spec.input.items})"
            lines.append(f"  [{i:2d}] {input_repr} -> {funcs}")
        lines.append("])")
        return "\n".join(lines)

    def to_py_file(self, file_path: str, variable_name: str = "trajectory_specs"):
        """
        Serialize TrajectorySpecList to a .py file that can be imported later.

        Args:
            file_path: Path to save the .py file
            variable_name: Name of the variable in the .py file
        """
        with open(file_path, "w") as f:
            f.write('"""\n')
            f.write("Generated trajectory specs data\n")
            f.write(f"Created on: {datetime.now().isoformat()}\n")
            f.write(f"Total specs: {len(self.specs)}\n")
            f.write('"""\n\n')

            f.write("from wandering_light.typed_list import TypedList\n")
            f.write(
                "from wandering_light.function_def import FunctionDef, FunctionDefList\n"
            )
            f.write(
                "from wandering_light.trajectory import TrajectorySpec, TrajectorySpecList\n\n"
            )

            # Generate unique function definitions

            all_functions_list = FunctionDefList()
            for spec in self.specs:
                all_functions_list.extend(spec.function_defs.functions)
            all_functions = {func.name: func for func in all_functions_list}

            # Write function definitions
            f.write("# Function definitions\n")
            for func_name, func_def in all_functions.items():
                f.write(f"{func_name}_func = FunctionDef(\n")
                f.write(f'    name="{func_def.name}",\n')
                f.write(f'    input_type="{func_def.input_type}",\n')
                f.write(f'    output_type="{func_def.output_type}",\n')
                f.write(f'    code="""{func_def.code}""",\n')
                f.write(f"    usage_count={func_def.usage_count},\n")
                f.write(f"    metadata={func_def.metadata!r}\n")
                f.write(")\n\n")

            # Write trajectory specs
            f.write("# Trajectory specifications\n")
            f.write("_specs = [\n")
            for spec in self.specs:
                # Write input TypedList
                repr(spec.input.items)

                # Write function list
                func_names = [f"{func.name}_func" for func in spec.function_defs]
                func_list = "FunctionDefList([" + ", ".join(func_names) + "])"

                f.write("    TrajectorySpec(\n")
                f.write(
                    f'        input_list=TypedList.from_str("""{spec.input.to_string()}"""),\n'
                )
                f.write(f"        function_defs={func_list}\n")
                f.write("    ),\n")
            f.write("]\n\n")

            f.write(f"{variable_name} = TrajectorySpecList(_specs)\n")

    @staticmethod
    def from_py_file(
        file_path: str, variable_name: str = "eval_trajectory_specs"
    ) -> "TrajectorySpecList":
        """
        Load TrajectorySpecList from a .py file.

        Args:
            file_path: Path to the .py file
            variable_name: Name of the variable in the .py file

        Returns:
            TrajectorySpecList loaded from the file
        """
        # Import the module dynamically
        import importlib.util
        import sys

        # Get absolute path and module name
        abs_path = os.path.abspath(file_path)
        module_name = os.path.splitext(os.path.basename(file_path))[0]

        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, abs_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Get the variable
        if not hasattr(module, variable_name):
            raise AttributeError(
                f"Module {module_name} does not have variable '{variable_name}'"
            )

        return getattr(module, variable_name)


class Trajectory:
    """
    Holds a spec plus its computed output without executing internally.
    """

    def __init__(self, spec: TrajectorySpec, output: TypedList):
        self.input = spec.input
        self.function_defs = spec.function_defs
        self.output = output

    def __eq__(self, other):
        return (
            isinstance(other, Trajectory)
            and self.input == other.input
            and self.function_defs == other.function_defs
            and self.output == other.output
        )

    def __repr__(self):
        function_defs_str = self.function_defs.to_string()
        return f"Trajectory({self.input} -> {function_defs_str} -> {self.output})"

    def to_spec(self) -> TrajectorySpec:
        return TrajectorySpec(self.input, self.function_defs)


class TrajectoryList:
    """
    A collection of Trajectory objects with pre-computed outputs.
    """

    def __init__(self, trajectories: list[Trajectory] | None = None):
        self.trajectories = trajectories or []

    def append(self, trajectory: Trajectory):
        """Add a Trajectory to the list."""
        self.trajectories.append(trajectory)

    def extend(self, trajectories: list[Trajectory]):
        """Add multiple Trajectories to the list."""
        self.trajectories.extend(trajectories)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        return self.trajectories[index]

    def __iter__(self):
        return iter(self.trajectories)

    def __repr__(self):
        if not self.trajectories:
            return "TrajectoryList([])"

        lines = ["TrajectoryList(["]
        for i, traj in enumerate(self.trajectories):
            # Get function chain
            funcs = traj.function_defs.to_string()
            # Format input and output nicely
            input_repr = f"{traj.input.item_type.__name__}({traj.input.items})"
            output_repr = f"{traj.output.item_type.__name__}({traj.output.items})"
            lines.append(f"  [{i:2d}] {input_repr} -> {funcs} -> {output_repr}")
        lines.append("])")
        return "\n".join(lines)

    @classmethod
    def from_trajectory_specs(
        cls, trajectory_specs: TrajectorySpecList, available_functions: FunctionDefSet
    ) -> "TrajectoryList":
        """
        Create a TrajectoryList by executing all trajectory specs.

        Args:
            trajectory_specs: TrajectorySpecList to execute
            available_functions: Functions available for execution

        Returns:
            TrajectoryList with pre-computed outputs
        """

        executor = Executor(available_functions)
        trajectories = []

        for spec in trajectory_specs.specs:
            result = executor.execute_trajectory(spec)
            if result.success:
                trajectories.append(result.trajectory)
            else:
                print(f"Warning: Failed to execute spec {spec}: {result.error_msg}")

        return cls(trajectories)

    @classmethod
    def from_file(cls, file_path: str) -> "TrajectoryList":
        """
        Load a TrajectoryList from a file.
        """
        trajectory_specs = TrajectorySpecList.from_py_file(file_path)
        available_functions = FunctionDefSet()
        for spec in trajectory_specs.specs:
            available_functions.extend(spec.function_defs.functions)
        return cls.from_trajectory_specs(trajectory_specs, available_functions)

    def to_spec_list(self) -> TrajectorySpecList:
        return TrajectorySpecList([traj.to_spec() for traj in self.trajectories])


if __name__ == "__main__":
    increment_fn = FunctionDef(
        name="increment",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
    )
    double_fn = FunctionDef(
        name="double",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x * 2",
    )
    tl = TypedList([1, 2, 3])
    available_functions = [increment_fn, double_fn]
    available_functions_set = FunctionDefSet(available_functions)
    executor = Executor(available_functions_set)
    spec = TrajectorySpec.create_random_walk(
        tl,
        path_length=3,
        available_functions=available_functions,
    )
    result = executor.execute_trajectory(spec)
    if result.success:
        print(result.trajectory)
    else:
        print(f"Execution failed: {result.error_msg}")
