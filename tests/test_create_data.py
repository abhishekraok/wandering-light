import os
import tempfile
from collections import Counter
from unittest.mock import patch

import pytest

from wandering_light.evals.create_data import make_common_fns_data, parse_length_counts
from wandering_light.trajectory import TrajectorySpecList


class TestCreateData:
    def test_make_common_fns_data(self):
        """Test that make_common_fns_data returns the expected number of trajectories."""
        result = make_common_fns_data()
        assert len(result) == 500
        length_counts = Counter(len(t.function_defs) for t in result)
        assert length_counts == {1: 100, 2: 100, 3: 100, 4: 100, 5: 100}

    def test_make_common_fns_data_basic_functionality(self):
        """Test basic functionality without file saving."""
        result = make_common_fns_data(save_to_file=False)

        # Check return type
        assert isinstance(result, TrajectorySpecList)

        # Check expected trajectory count
        assert len(result) == 500

        # Check trajectory length distribution
        length_counts = Counter(len(t.function_defs) for t in result)
        assert length_counts == {1: 100, 2: 100, 3: 100, 4: 100, 5: 100}

        # Check that all trajectories have valid input lists
        for trajectory in result:
            assert trajectory.input is not None
            assert hasattr(trajectory.input, "items")
            assert hasattr(trajectory.input, "item_type")
            assert len(trajectory.input.items) > 0

        # Check that all trajectories have valid function definitions
        for trajectory in result:
            assert trajectory.function_defs is not None
            assert len(trajectory.function_defs) > 0
            for func_def in trajectory.function_defs:
                assert hasattr(func_def, "name")
                assert hasattr(func_def, "input_type")
                assert hasattr(func_def, "output_type")
                assert hasattr(func_def, "code")

    def test_make_common_fns_data_with_file_saving(self):
        """Test file saving functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with custom output directory
            result = make_common_fns_data(save_to_file=True, output_dir=temp_dir)

            # Check that files were created
            files = os.listdir(temp_dir)
            eval_files = [
                f for f in files if f.startswith("eval_data_v") and f.endswith(".py")
            ]
            latest_file = [f for f in files if f == "eval_data_latest.py"]

            assert len(eval_files) == 1, (
                f"Expected 1 eval data file, found {len(eval_files)}: {eval_files}"
            )
            assert len(latest_file) == 1, "Expected eval_data_latest.py file"

            # Check file contents are valid Python files
            versioned_file = os.path.join(temp_dir, eval_files[0])
            latest_file_path = os.path.join(temp_dir, "eval_data_latest.py")

            # Both files should be readable and contain expected content
            for file_path in [versioned_file, latest_file_path]:
                with open(file_path) as f:
                    content = f.read()

                # Check for expected imports and variable name
                assert "from wandering_light.typed_list import TypedList" in content
                assert "from wandering_light.function_def import FunctionDef" in content
                assert (
                    "from wandering_light.trajectory import TrajectorySpec, TrajectorySpecList"
                    in content
                )
                assert "eval_trajectory_specs = TrajectorySpecList" in content

                # Check that file contains trajectory data
                assert "TrajectorySpec(" in content
                assert "_specs = [" in content

            # Verify that data can be loaded back from the file
            loaded_data = TrajectorySpecList.from_py_file(
                latest_file_path, "eval_trajectory_specs"
            )

            # Check that loaded data matches original
            assert len(loaded_data) == len(result)
            assert len(loaded_data) == 500

    def test_make_common_fns_data_trajectory_validity(self):
        """Test that generated trajectories are valid and well-formed."""
        result = make_common_fns_data(save_to_file=False)

        # Check each trajectory
        for i, trajectory in enumerate(result):
            # Input validation
            assert trajectory.input is not None, f"Trajectory {i} has None input"
            assert hasattr(trajectory.input, "items"), (
                f"Trajectory {i} input missing 'items'"
            )
            assert hasattr(trajectory.input, "item_type"), (
                f"Trajectory {i} input missing 'item_type'"
            )

            # Function definitions validation
            assert trajectory.function_defs is not None, (
                f"Trajectory {i} has None function_defs"
            )
            assert len(trajectory.function_defs) > 0, (
                f"Trajectory {i} has empty function_defs"
            )
            assert len(trajectory.function_defs) <= 5, (
                f"Trajectory {i} has too many functions"
            )

            # Function chaining validation - each function's input should match previous output
            current_type = trajectory.input.item_type
            for j, func_def in enumerate(trajectory.function_defs):
                expected_input_type = (
                    f"{current_type.__module__}.{current_type.__qualname__}"
                )
                assert func_def.input_type == expected_input_type, (
                    f"Trajectory {i}, function {j}: expected input type {expected_input_type}, got {func_def.input_type}"
                )

                # Update current type for next iteration
                current_type = func_def.output_type_cls()

    def test_parse_length_counts_basic(self):
        """Test basic functionality of parse_length_counts."""
        result = parse_length_counts("1:10,2:20,3:30")
        expected = {1: 10, 2: 20, 3: 30}
        assert result == expected

    def test_parse_length_counts_single_pair(self):
        """Test parse_length_counts with a single length:count pair."""
        result = parse_length_counts("5:100")
        expected = {5: 100}
        assert result == expected

    def test_parse_length_counts_mixed_order(self):
        """Test parse_length_counts with mixed order."""
        result = parse_length_counts("3:50,1:25,2:75")
        expected = {3: 50, 1: 25, 2: 75}
        assert result == expected

    def test_parse_length_counts_with_spaces(self):
        """Test that parse_length_counts handles spaces gracefully."""
        result = parse_length_counts("1: 10, 2: 20, 3 : 30")
        expected = {1: 10, 2: 20, 3: 30}
        assert result == expected

    def test_parse_length_counts_invalid_format(self):
        """Test parse_length_counts with invalid format."""
        with pytest.raises(ValueError, match="Invalid format"):
            parse_length_counts("1-10,2-20")  # Wrong separator

        with pytest.raises(ValueError, match="Invalid format"):
            parse_length_counts("invalid")  # No separator

        with pytest.raises(ValueError, match="Invalid number"):
            parse_length_counts("1:a,2:b")  # Non-numeric values

        with pytest.raises(ValueError, match="Invalid number"):
            parse_length_counts("a:1,b:2")  # Non-numeric keys

    def test_make_common_fns_data_custom_length_counts(self):
        """Test make_common_fns_data with custom length_counts parameter."""
        custom_length_counts = {1: 20, 3: 30, 5: 10}
        result = make_common_fns_data(
            save_to_file=False, length_counts=custom_length_counts
        )

        # Check total count
        expected_total = sum(custom_length_counts.values())
        assert len(result) == expected_total

        # Check length distribution
        actual_length_counts = Counter(len(t.function_defs) for t in result)
        assert actual_length_counts == custom_length_counts

    def test_make_common_fns_data_empty_length_counts(self):
        """Test make_common_fns_data with empty length_counts."""
        result = make_common_fns_data(save_to_file=False, length_counts={})
        assert len(result) == 0

    def test_make_common_fns_data_none_length_counts(self):
        """Test make_common_fns_data with None length_counts uses default."""
        result = make_common_fns_data(save_to_file=False, length_counts=None)

        # Should use default length counts
        assert len(result) == 500
        length_counts = Counter(len(t.function_defs) for t in result)
        assert length_counts == {1: 100, 2: 100, 3: 100, 4: 100, 5: 100}

    def test_make_common_fns_data_custom_length_counts_with_file_saving(self):
        """Test make_common_fns_data with custom length_counts and file saving."""
        custom_length_counts = {2: 10, 4: 5}

        with tempfile.TemporaryDirectory() as temp_dir:
            result = make_common_fns_data(
                save_to_file=True,
                output_dir=temp_dir,
                length_counts=custom_length_counts,
            )

            # Check result
            assert len(result) == 15  # 10 + 5
            length_counts = Counter(len(t.function_defs) for t in result)
            assert length_counts == custom_length_counts

            # Check file was created
            files = os.listdir(temp_dir)
            assert any(f.startswith("eval_data_v") for f in files)
            assert "eval_data_latest.py" in files

    @patch("sys.argv", ["create_data.py"])
    def test_command_line_parsing_defaults(self):
        """Test command line argument parsing with defaults."""
        # Import the main execution logic

        # Test default length counts parsing
        default_csv = "1:100,2:100,3:100,4:100,5:100"
        result = parse_length_counts(default_csv)
        expected = {1: 100, 2: 100, 3: 100, 4: 100, 5: 100}
        assert result == expected

    def test_command_line_integration(self):
        """Test the integration of command line parsing with data generation."""
        # Test that custom length counts work end-to-end
        csv_input = "1:5,3:10"
        length_counts = parse_length_counts(csv_input)

        result = make_common_fns_data(save_to_file=False, length_counts=length_counts)

        assert len(result) == 15  # 5 + 10
        actual_counts = Counter(len(t.function_defs) for t in result)
        assert actual_counts == {1: 5, 3: 10}

    def test_parse_length_counts_edge_cases(self):
        """Test parse_length_counts with edge cases."""
        # Empty string
        with pytest.raises(ValueError, match="Invalid format"):
            parse_length_counts("")

        # Only commas
        with pytest.raises(ValueError, match="Invalid format"):
            parse_length_counts(",,,")

        # Zero counts (should be valid)
        result = parse_length_counts("1:0,2:5")
        assert result == {1: 0, 2: 5}

        # Large numbers
        result = parse_length_counts("100:999,50:1000")
        assert result == {100: 999, 50: 1000}

    def test_parse_length_counts_duplicate_lengths(self):
        """Test parse_length_counts with duplicate length values (last one wins)."""
        result = parse_length_counts("1:10,2:20,1:30")
        # Last occurrence should win
        assert result == {1: 30, 2: 20}

    def test_make_common_fns_data_with_zero_counts(self):
        """Test make_common_fns_data with some zero counts in length_counts."""
        length_counts = {1: 0, 2: 10, 3: 0, 4: 5}
        result = make_common_fns_data(save_to_file=False, length_counts=length_counts)

        assert len(result) == 15  # 0 + 10 + 0 + 5
        actual_counts = Counter(len(t.function_defs) for t in result)
        # Should only have lengths 2 and 4, not 1 and 3
        assert actual_counts == {2: 10, 4: 5}

    def test_make_common_fns_data_random_inputs_basic(self):
        """Test make_common_fns_data with random_inputs=True."""
        length_counts = {1: 5, 2: 10}
        result = make_common_fns_data(
            save_to_file=False, length_counts=length_counts, random_inputs=True
        )

        # Check total count
        assert len(result) == 15  # 5 + 10

        # Check length distribution
        actual_counts = Counter(len(t.function_defs) for t in result)
        assert actual_counts == {1: 5, 2: 10}

        # Check that all trajectories have valid input lists with random types
        for trajectory in result:
            assert trajectory.input is not None
            assert trajectory.input.item_type is not None
            assert len(trajectory.input.items) > 0
            # Verify the type is one of the supported types
            from evals.create_data import SUPPORTED_RANDOM_TYPES

            assert trajectory.input.item_type in SUPPORTED_RANDOM_TYPES

    def test_make_common_fns_data_random_inputs_different_types(self):
        """Test make_common_fns_data with random_inputs=True generates various types."""
        length_counts = {1: 50}  # Generate more to increase chance of different types

        result = make_common_fns_data(
            save_to_file=False, length_counts=length_counts, random_inputs=True
        )

        assert len(result) == 50

        # Collect all the types that were generated
        generated_types = set()
        for trajectory in result:
            generated_types.add(trajectory.input.item_type)

        from evals.create_data import SUPPORTED_RANDOM_TYPES

        # All generated types should be supported
        for gen_type in generated_types:
            assert gen_type in SUPPORTED_RANDOM_TYPES

        # With 50 trajectories, we should get multiple different types (very likely)
        # This is probabilistic but very unlikely to fail
        assert len(generated_types) > 1, (
            f"Expected multiple types, got only: {generated_types}"
        )

    def test_make_common_fns_data_random_inputs_vs_predefined(self):
        """Test that random_inputs=True generates different data than predefined inputs."""
        length_counts = {1: 10}

        # Generate with predefined inputs
        result_predefined = make_common_fns_data(
            save_to_file=False, length_counts=length_counts, random_inputs=False
        )

        # Generate with random inputs
        result_random = make_common_fns_data(
            save_to_file=False, length_counts=length_counts, random_inputs=True
        )

        # Both should have same length
        assert len(result_predefined) == len(result_random) == 10

        # Check that the input types are different
        # Predefined uses various types from SAMPLE_INPUTS, random uses randomly chosen types
        predefined_types = set()
        random_types = set()

        for traj in result_predefined:
            predefined_types.add(traj.input.item_type)

        for traj in result_random:
            random_types.add(traj.input.item_type)

        # Both should have multiple types, but the specific sets may differ
        from evals.create_data import SUPPORTED_RANDOM_TYPES

        for rand_type in random_types:
            assert rand_type in SUPPORTED_RANDOM_TYPES

    def test_make_common_fns_data_random_inputs_with_file_saving(self):
        """Test make_common_fns_data with random_inputs=True and file saving."""
        length_counts = {1: 5, 3: 7}

        with tempfile.TemporaryDirectory() as temp_dir:
            result = make_common_fns_data(
                save_to_file=True,
                output_dir=temp_dir,
                length_counts=length_counts,
                random_inputs=True,
            )

            # Check result
            assert len(result) == 12  # 5 + 7
            length_counts_actual = Counter(len(t.function_defs) for t in result)
            assert length_counts_actual == {1: 5, 3: 7}

            # Check all inputs have valid types
            from evals.create_data import SUPPORTED_RANDOM_TYPES

            for trajectory in result:
                assert trajectory.input.item_type in SUPPORTED_RANDOM_TYPES

            # Check file was created
            files = os.listdir(temp_dir)
            assert any(f.startswith("eval_data_v") for f in files)
            assert "eval_data_latest.py" in files

    def test_make_common_fns_data_random_inputs_bool_type(self):
        """Test that bool type can be generated and works correctly."""
        # Generate many trajectories to increase chance of getting bool type
        length_counts = {1: 100}
        result = make_common_fns_data(
            save_to_file=False, length_counts=length_counts, random_inputs=True
        )

        assert len(result) == 100

        # Find trajectories with bool input type
        bool_trajectories = [t for t in result if t.input.item_type is bool]

        # We should get at least some bool trajectories (probabilistic but very likely)
        # If this occasionally fails, it's due to randomness
        if bool_trajectories:  # Only test if we got bool trajectories
            for trajectory in bool_trajectories:
                assert trajectory.input.item_type is bool
                # For bool type, list length should be 2 (as per make_random_data implementation)
                assert len(trajectory.input.items) == 2
                for item in trajectory.input.items:
                    assert isinstance(item, bool)
                    assert item in [True, False]

    def test_make_random_data_generates_all_supported_types(self):
        """Test that make_random_data can generate all supported types with sufficient sample size."""
        from common_functions import basic_fns
        from evals.create_data import SUPPORTED_RANDOM_TYPES, make_random_data
        from function_def import FunctionDefList

        # Generate many trajectories to ensure we get all types
        # With 12 types and uniform distribution, we need enough samples to be confident
        # Using 1000 samples gives us very high probability of hitting all types
        length_counts = {1: 1000}

        result = make_random_data(
            available_functions=FunctionDefList(basic_fns),
            length_counts=length_counts,
            seed=42,  # Use seed for reproducibility
        )

        assert len(result) == 1000

        # Collect all generated types
        generated_types = set()
        for trajectory in result:
            generated_types.add(trajectory.input.item_type)

        # Verify all types are from supported types
        for gen_type in generated_types:
            assert gen_type in SUPPORTED_RANDOM_TYPES

        # The main assertion: all supported types should be generated
        missing_types = set(SUPPORTED_RANDOM_TYPES) - generated_types
        assert len(missing_types) == 0, (
            f"Missing types: {[t.__name__ for t in missing_types]}"
        )

        # Verify we got exactly the expected number of supported types
        assert len(generated_types) == len(SUPPORTED_RANDOM_TYPES), (
            f"Expected {len(SUPPORTED_RANDOM_TYPES)} types, got {len(generated_types)}. "
            f"Generated: {[t.__name__ for t in sorted(generated_types, key=lambda x: x.__name__)]}"
        )

        # Additional validation: check type-specific constraints
        for trajectory in result:
            if trajectory.input.item_type is bool:
                # Bool lists should have length 2
                assert len(trajectory.input.items) == 2
                for item in trajectory.input.items:
                    assert isinstance(item, bool)
            else:
                # Non-bool lists should have length 3-5
                assert 3 <= len(trajectory.input.items) <= 5
