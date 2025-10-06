import os
import tempfile

import pytest

from wandering_light.function_def import FunctionDef, FunctionDefList
from wandering_light.trajectory import TrajectorySpec, TrajectorySpecList
from wandering_light.typed_list import TypedList


class TestTrajectorySpecListSerialization:
    """Test suite for TrajectorySpecList serialization functionality."""

    def test_empty_list_serialization(self):
        """Test serialization of empty TrajectorySpecList."""
        empty_list = TrajectorySpecList()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            temp_path = f.name

        try:
            empty_list.to_py_file(temp_path, "test_empty")
            loaded = TrajectorySpecList.from_py_file(temp_path, "test_empty")

            assert len(loaded) == 0
            assert len(loaded.specs) == 0
        finally:
            os.unlink(temp_path)

    def test_single_spec_serialization(self):
        """Test serialization of TrajectorySpecList with one spec."""
        # Create a simple function
        inc_func = FunctionDef(
            name="increment",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
            usage_count=5,
            metadata={"test": "value"},
        )

        # Create a spec
        input_list = TypedList([1, 2, 3])
        spec = TrajectorySpec(input_list, FunctionDefList([inc_func]))
        spec_list = TrajectorySpecList([spec])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            temp_path = f.name

        try:
            spec_list.to_py_file(temp_path, "test_single")
            loaded = TrajectorySpecList.from_py_file(temp_path, "test_single")

            assert len(loaded) == 1
            assert loaded[0].input.items == [1, 2, 3]
            assert loaded[0].input.item_type == int
            assert len(loaded[0].function_defs) == 1
            assert loaded[0].function_defs[0].name == "increment"
            assert loaded[0].function_defs[0].usage_count == 5
            assert loaded[0].function_defs[0].metadata == {"test": "value"}
        finally:
            os.unlink(temp_path)

    def test_complex_types_serialization(self):
        """Test serialization with complex Python types."""
        # Test various types
        test_cases = [
            (TypedList([b"hello", b"world"]), bytes),
            (TypedList([bytearray(b"test")]), bytearray),
            (TypedList([{1, 2, 3}, {4, 5}]), set),
            (TypedList([(1, 2), (3, 4)]), tuple),
            (TypedList([range(5), range(1, 4)]), range),
            (TypedList([1 + 2j, 3 - 4j]), complex),
        ]

        for input_list, expected_type in test_cases:
            # Create a dummy function for this type
            func = FunctionDef(
                name=f"test_{expected_type.__name__}",
                input_type=f"builtins.{expected_type.__name__}",
                output_type=f"builtins.{expected_type.__name__}",
                code="return x",
            )

            spec = TrajectorySpec(input_list, FunctionDefList([func]))
            spec_list = TrajectorySpecList([spec])

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                temp_path = f.name

            try:
                spec_list.to_py_file(temp_path, f"test_{expected_type.__name__}")
                loaded = TrajectorySpecList.from_py_file(
                    temp_path, f"test_{expected_type.__name__}"
                )

                assert len(loaded) == 1
                assert loaded[0].input.item_type == expected_type
                assert loaded[0].input.items == input_list.items

                # Verify type preservation
                for original, loaded_item in zip(
                    input_list.items, loaded[0].input.items, strict=False
                ):
                    assert type(original) == type(loaded_item)
                    assert original == loaded_item
            finally:
                os.unlink(temp_path)

    def test_multiple_specs_serialization(self):
        """Test serialization of multiple specs with different functions."""
        # Create multiple functions
        inc_func = FunctionDef(
            name="inc",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
        )

        double_func = FunctionDef(
            name="double",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x * 2",
        )

        # Create multiple specs
        specs = [
            TrajectorySpec(TypedList([1, 2]), FunctionDefList([inc_func])),
            TrajectorySpec(TypedList([3, 4]), FunctionDefList([double_func])),
            TrajectorySpec(TypedList([5, 6]), FunctionDefList([inc_func, double_func])),
        ]

        spec_list = TrajectorySpecList(specs)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            temp_path = f.name

        try:
            spec_list.to_py_file(temp_path, "test_multiple")
            loaded = TrajectorySpecList.from_py_file(temp_path, "test_multiple")

            assert len(loaded) == 3

            # Check first spec
            assert loaded[0].input.items == [1, 2]
            assert len(loaded[0].function_defs) == 1
            assert loaded[0].function_defs[0].name == "inc"

            # Check second spec
            assert loaded[1].input.items == [3, 4]
            assert len(loaded[1].function_defs) == 1
            assert loaded[1].function_defs[0].name == "double"

            # Check third spec (multiple functions)
            assert loaded[2].input.items == [5, 6]
            assert len(loaded[2].function_defs) == 2
            assert loaded[2].function_defs[0].name == "inc"
            assert loaded[2].function_defs[1].name == "double"

        finally:
            os.unlink(temp_path)

    def test_file_error_handling(self):
        """Test error handling for file operations."""
        spec_list = TrajectorySpecList()

        # Test loading non-existent file
        with pytest.raises(FileNotFoundError):
            TrajectorySpecList.from_py_file("non_existent_file.py")

        # Test loading file with wrong variable name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            temp_path = f.name

        try:
            spec_list.to_py_file(temp_path, "correct_name")

            with pytest.raises(AttributeError):
                TrajectorySpecList.from_py_file(temp_path, "wrong_name")
        finally:
            os.unlink(temp_path)

    def test_generated_file_structure(self):
        """Test that generated .py files have proper structure."""
        inc_func = FunctionDef(
            name="test_func",
            input_type="builtins.int",
            output_type="builtins.int",
            code="return x + 1",
        )

        spec = TrajectorySpec(TypedList([1, 2]), FunctionDefList([inc_func]))
        spec_list = TrajectorySpecList([spec])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            temp_path = f.name

        try:
            spec_list.to_py_file(temp_path, "test_structure")

            # Read the generated file
            with open(temp_path) as f:
                content = f.read()

            # Check that it contains expected elements
            assert "from wandering_light.typed_list import TypedList" in content
            assert (
                "from wandering_light.function_def import FunctionDef, FunctionDefList"
                in content
            )
            assert (
                "from wandering_light.trajectory import TrajectorySpec, TrajectorySpecList"
                in content
            )
            assert "test_func_func = FunctionDef(" in content
            assert "test_structure = TrajectorySpecList(_specs)" in content
            assert "Created on:" in content
            assert "Total specs: 1" in content

        finally:
            os.unlink(temp_path)
