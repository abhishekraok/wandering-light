import importlib

from pydantic import BaseModel


# Note: Currently only supports single-argument functions
class FunctionDef(BaseModel):
    name: str
    input_type: str  # e.g. "builtins.int" or "my_module.MyClass"
    output_type: str
    code: str  # a Python function without the signature and using `x`, e.g. "return x + 1"
    usage_count: int = 0
    metadata: dict = {}

    def increment_usage(self):
        self.usage_count += 1

    def _resolve_type(self, type_str: str) -> type:
        module_name, _, class_name = type_str.rpartition(".")
        mod = importlib.import_module(module_name)
        return getattr(mod, class_name)

    def input_type_cls(self) -> type:
        return self._resolve_type(self.input_type)

    def output_type_cls(self) -> type:
        return self._resolve_type(self.output_type)

    def executable_code(self) -> str:
        return f"""
def {self.name}(x):
    {self.code.strip()}
"""

    def __hash__(self):
        """Enable usage with sets by implementing hash."""
        return hash((self.name, self.input_type, self.output_type, self.code))

    def __eq__(self, other):
        """Enable usage with sets by implementing equality."""
        if not isinstance(other, FunctionDef):
            return False
        return (
            self.name == other.name
            and self.input_type == other.input_type
            and self.output_type == other.output_type
            and self.code == other.code
        )


class FunctionDefSet:
    """
    A set of unique FunctionDef objects for representing available functions.
    Automatically handles duplicates by ignoring them when adding functions.
    Used for storing collections of available functions where duplicates don't make sense.
    """

    def __init__(self, function_defs: list[FunctionDef] | None = None):
        self.functions = []
        self.name_to_function = {}  # Track seen functions for deduplication
        if function_defs:
            self.extend(function_defs)

    def append(self, func_def: FunctionDef):
        """Add a FunctionDef to the set, ignoring duplicates."""
        if func_def.name not in self.name_to_function:
            self.name_to_function[func_def.name] = func_def
            self.functions.append(func_def)

    def extend(self, func_defs: list[FunctionDef]):
        """Add multiple FunctionDefs to the set, ignoring duplicates."""
        for func_def in func_defs:
            self.append(func_def)

    def __len__(self):
        return len(self.functions)

    def __getitem__(self, index):
        return self.functions[index]

    def __iter__(self):
        return iter(self.functions)

    def __repr__(self):
        if not self.functions:
            return "FunctionDefSet([])"

        function_names = [func_def.name for func_def in self.functions]
        return f"FunctionDefSet([{', '.join(function_names)}])"

    def to_string(self) -> str:
        """
        Convert the FunctionDefSet to a comma-separated string of function names.

        Returns:
            Comma-separated string of function names (e.g., "inc, dec, double")
        """
        if not self.functions:
            return ""

        return ", ".join(func_def.name for func_def in self.functions)

    def parse_string(self, repr_string: str) -> "FunctionDefList":
        """
        Parse a string representation into a FunctionDefList (execution sequence).
        The string should be a comma-separated list of function names.
        Will look up each function name in self.functions and create a sequence.
        If a function is not found it will return an empty FunctionDefList.

        Args:
            repr_string: Comma-separated string of function names (e.g., "inc, dec, double")

        Returns:
            FunctionDefList containing the parsed function sequence (preserves duplicates)
        """
        if not repr_string.strip():
            return FunctionDefList()

        # Split by comma and clean up whitespace
        function_names = [
            name.strip() for name in repr_string.split(",") if name.strip()
        ]

        result_functions = []
        # Look up each function by name from our available functions
        for name in function_names:
            if name not in self.name_to_function:
                return FunctionDefList()  # Return empty if any function not found
            result_functions.append(self.name_to_function[name])

        return FunctionDefList(result_functions)


class FunctionDefList:
    """
    A sequence of FunctionDef objects for representing execution sequences.
    Preserves duplicates and order - used for function execution sequences
    where the same function can appear multiple times.
    """

    def __init__(self, function_defs: list[FunctionDef] | None = None):
        self.functions = function_defs if function_defs is not None else []

    def append(self, func_def: FunctionDef):
        """Add a FunctionDef to the sequence, preserving duplicates."""
        self.functions.append(func_def)

    def extend(self, func_defs: list[FunctionDef]):
        """Add multiple FunctionDefs to the sequence, preserving duplicates."""
        self.functions.extend(func_defs)

    def __len__(self):
        return len(self.functions)

    def __getitem__(self, index):
        return self.functions[index]

    def __iter__(self):
        return iter(self.functions)

    def __repr__(self):
        if not self.functions:
            return "FunctionDefList([])"

        function_names = [func_def.name for func_def in self.functions]
        return f"FunctionDefList([{', '.join(function_names)}])"

    def to_string(self) -> str:
        """
        Convert the FunctionDefList to a comma-separated string of function names.

        Returns:
            Comma-separated string of function names (e.g., "inc, dec, double")
        """
        if not self.functions:
            return ""

        return ", ".join(func_def.name for func_def in self.functions)

    def __eq__(self, other):
        """Enable usage with sets by implementing equality."""
        if not isinstance(other, FunctionDefList):
            return False
        return self.functions == other.functions
