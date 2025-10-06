from pydantic import BaseModel

from wandering_light.function_def import FunctionDef


class ExistingFunctions(BaseModel):
    functions: list[FunctionDef]


class NewFunction(BaseModel):
    function: FunctionDef


def create_different_function(existing_functions: list[FunctionDef]) -> FunctionDef:
    """Creates a new function different from existing ones.

    The previous implementation relied on the ``object_generator`` package which
    has been removed. This function is currently not implemented.
    """

    raise NotImplementedError("object_generator support has been removed")


if __name__ == "__main__":
    print(
        "Function synthesis is not available because object_generator has been removed."
    )
