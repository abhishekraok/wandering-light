import pytest

from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef, FunctionDefList
from wandering_light.trajectory import TrajectorySpec
from wandering_light.typed_list import TypedList


@pytest.fixture
def setup_exec():
    return FunctionDefList([])


def test_increment_function_def(setup_exec):
    available_functions = setup_exec
    code = "return x + 1"
    fn = FunctionDef(
        name="increment",
        input_type="builtins.int",
        output_type="builtins.int",
        code=code,
    )
    available_functions.append(fn)
    ex = Executor(available_functions)
    inp = TypedList([1, 2, 3])
    out = ex.execute(fn, inp)
    assert out == TypedList([2, 3, 4])
    # usage_count incremented
    assert fn.usage_count == 1


def test_executor_rejects_function_outside_palette():
    allowed = FunctionDef(
        name="increment",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
    )
    unregistered = FunctionDef(
        name="double",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x * 2",
    )
    executor = Executor([allowed])

    with pytest.raises(ValueError, match="not in this executor's basis set"):
        executor.execute(unregistered, TypedList([1]))


def test_executor_rejects_same_name_with_different_definition():
    allowed = FunctionDef(
        name="transform",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
    )
    impostor = FunctionDef(
        name="transform",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x - 1",
    )
    executor = Executor([allowed])

    with pytest.raises(ValueError, match="does not match the registered definition"):
        executor.execute(impostor, TypedList([1]))


def test_executor_rejects_list_with_mixed_basis_provenance():
    first = FunctionDef(
        name="increment",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
        metadata={
            "basis_function_id": "bf:increment:shared",
            "basis_function_fingerprint": "fingerprint-shared",
            "basis_set_id": "basis-a",
            "basis_set_digest": "digest-a",
        },
    )
    second = first.model_copy(deep=True)
    second.metadata.update({"basis_set_id": "basis-b", "basis_set_digest": "digest-b"})

    with pytest.raises(ValueError, match="Conflicting executable definitions"):
        Executor([first, second])


def test_executor_compiles_definition_once(monkeypatch):
    fn = FunctionDef(
        name="increment",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
    )
    executor = Executor([fn])
    original_compile = executor._compile
    compiled_ids = []

    def observing_compile(function):
        compiled = original_compile(function)
        compiled_ids.append(id(compiled))
        return compiled

    monkeypatch.setattr(executor, "_compile", observing_compile)
    executor.execute(fn, TypedList([1]))
    executor.execute(fn, TypedList([2]))

    assert compiled_ids[0] == compiled_ids[1]
    assert len(executor._compiled) == 1


def test_executor_rejects_mutated_registered_definition():
    fn = FunctionDef(
        name="increment",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
        metadata={
            "basis_function_id": "increment-v1",
            "basis_function_fingerprint": "fingerprint-v1",
            "basis_set_id": "basis-v1",
            "basis_set_digest": "digest-v1",
        },
    )
    executor = Executor([fn])
    fn.code = "return x + 100"

    with pytest.raises(ValueError, match="mutated after executor construction"):
        executor.execute(fn, TypedList([1]))


def test_trajectory_result_canonicalizes_equal_clone_to_registered_provenance():
    registered = FunctionDef(
        name="increment",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
        metadata={
            "basis_function_id": "bf:increment:registered",
            "basis_function_fingerprint": "fingerprint-registered",
            "basis_set_id": "basis-registered",
            "basis_set_digest": "digest-registered",
        },
    )
    forged = registered.model_copy(deep=True)
    forged.metadata.update(
        {
            "basis_function_id": "bf:increment:forged",
            "basis_set_id": "basis-forged",
            "basis_set_digest": "digest-forged",
        }
    )
    executor = Executor([registered])

    result = executor.execute_trajectory(
        TrajectorySpec(TypedList([1]), FunctionDefList([forged]))
    )

    assert result.success
    assert result.trajectory is not None
    assert result.trajectory.function_defs[0] is registered
    assert result.trajectory.function_defs[0].metadata["basis_function_id"] == (
        "bf:increment:registered"
    )
