import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add the web_ui/backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "web_ui", "backend"))

from wandering_light.executor import Executor
from wandering_light.function_def import FunctionDef
from wandering_light.typed_list import TypedList
from wandering_light.web_ui.backend.graph_executor import GraphExecutor


@pytest.fixture
def sample_functions():
    """Create sample functions for testing."""
    increment_func = FunctionDef(
        name="increment",
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

    return [increment_func, double_func]


@pytest.fixture
def mock_executor():
    """Create a mock executor for testing."""
    executor = Mock(spec=Executor)
    return executor


@pytest.fixture
def graph_executor(sample_functions, mock_executor):
    """Create a GraphExecutor instance for testing."""
    return GraphExecutor(sample_functions, mock_executor)


class TestGraphExecutorInit:
    def test_init(self, sample_functions, mock_executor):
        """Test GraphExecutor initialization."""
        ge = GraphExecutor(sample_functions, mock_executor)
        assert ge.available_functions == sample_functions
        assert ge.executor == mock_executor


class TestTopologicalSort:
    def test_simple_linear_graph(self, graph_executor):
        """Test topological sort with a simple linear graph."""
        nodes = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
        edges = [{"source": "A", "target": "B"}, {"source": "B", "target": "C"}]

        result = graph_executor.topological_sort(nodes, edges)
        assert result == ["A", "B", "C"]

    def test_parallel_branches(self, graph_executor):
        """Test topological sort with parallel branches."""
        nodes = [{"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}]
        edges = [
            {"source": "A", "target": "B"},
            {"source": "A", "target": "C"},
            {"source": "B", "target": "D"},
            {"source": "C", "target": "D"},
        ]

        result = graph_executor.topological_sort(nodes, edges)
        # A should come first, D should come last
        assert result.index("A") < result.index("B")
        assert result.index("A") < result.index("C")
        assert result.index("B") < result.index("D")
        assert result.index("C") < result.index("D")

    def test_single_node(self, graph_executor):
        """Test topological sort with a single node."""
        nodes = [{"id": "A"}]
        edges = []

        result = graph_executor.topological_sort(nodes, edges)
        assert result == ["A"]

    def test_no_nodes(self, graph_executor):
        """Test topological sort with no nodes."""
        nodes = []
        edges = []

        result = graph_executor.topological_sort(nodes, edges)
        assert result == []

    def test_cycle_detection(self, graph_executor):
        """Test that cycles are detected and raise an error."""
        nodes = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
        edges = [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
            {"source": "C", "target": "A"},  # Creates a cycle
        ]

        with pytest.raises(ValueError, match="Graph contains a cycle"):
            graph_executor.topological_sort(nodes, edges)

    def test_self_cycle(self, graph_executor):
        """Test detection of self-referencing cycles."""
        nodes = [{"id": "A"}]
        edges = [{"source": "A", "target": "A"}]

        with pytest.raises(ValueError, match="Graph contains a cycle"):
            graph_executor.topological_sort(nodes, edges)

    def test_complex_graph(self, graph_executor):
        """Test topological sort with a more complex graph."""
        nodes = [
            {"id": "A"},
            {"id": "B"},
            {"id": "C"},
            {"id": "D"},
            {"id": "E"},
            {"id": "F"},
        ]
        edges = [
            {"source": "A", "target": "C"},
            {"source": "B", "target": "C"},
            {"source": "C", "target": "D"},
            {"source": "D", "target": "E"},
            {"source": "D", "target": "F"},
        ]

        result = graph_executor.topological_sort(nodes, edges)

        # Verify dependencies are respected
        assert result.index("A") < result.index("C")
        assert result.index("B") < result.index("C")
        assert result.index("C") < result.index("D")
        assert result.index("D") < result.index("E")
        assert result.index("D") < result.index("F")


class TestGetInputNodes:
    def test_single_input(self, graph_executor):
        """Test getting input nodes with a single input."""
        nodes = [{"id": "A"}, {"id": "B"}]
        edges = [{"source": "A", "target": "B", "sourceHandle": "output"}]

        result = graph_executor.get_input_nodes("B", nodes, edges)
        assert result == [("A", "output")]

    def test_multiple_inputs(self, graph_executor):
        """Test getting input nodes with multiple inputs."""
        nodes = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
        edges = [
            {"source": "A", "target": "C", "sourceHandle": "output1"},
            {"source": "B", "target": "C", "sourceHandle": "output2"},
        ]

        result = graph_executor.get_input_nodes("C", nodes, edges)
        expected = [("A", "output1"), ("B", "output2")]
        assert sorted(result) == sorted(expected)

    def test_no_inputs(self, graph_executor):
        """Test getting input nodes when there are no inputs."""
        nodes = [{"id": "A"}, {"id": "B"}]
        edges = [{"source": "A", "target": "B"}]

        result = graph_executor.get_input_nodes("A", nodes, edges)
        assert result == []

    def test_default_handle(self, graph_executor):
        """Test getting input nodes with default handle."""
        nodes = [{"id": "A"}, {"id": "B"}]
        edges = [{"source": "A", "target": "B"}]  # No sourceHandle specified

        result = graph_executor.get_input_nodes("B", nodes, edges)
        assert result == [("A", "default")]


class TestExecuteGraph:
    def test_simple_typed_list_node(self, graph_executor):
        """Test execution of a simple typed list node."""
        graph_data = {
            "nodes": [
                {
                    "id": "node1",
                    "type": "typed_list",
                    "data": {"item_type": "builtins.int", "items": [1, 2, 3]},
                }
            ],
            "edges": [],
        }

        result = graph_executor.execute_graph(graph_data)

        assert result["status"] == "success"
        assert len(result["results"]) == 1
        assert result["results"][0]["node_id"] == "node1"
        assert result["results"][0]["result"] == [1, 2, 3]
        assert result["results"][0]["result_type"] == "builtins.int"

    def test_function_node_with_input(self, graph_executor):
        """Test execution of a function node with input."""
        # Mock the executor to return a known result
        mock_result = TypedList([2, 3, 4], item_type=int)
        graph_executor.executor.execute.return_value = mock_result

        graph_data = {
            "nodes": [
                {
                    "id": "input_node",
                    "type": "typed_list",
                    "data": {"item_type": "builtins.int", "items": [1, 2, 3]},
                },
                {
                    "id": "func_node",
                    "type": "function_def",
                    "data": {"name": "increment"},
                },
            ],
            "edges": [{"source": "input_node", "target": "func_node"}],
        }

        result = graph_executor.execute_graph(graph_data)

        assert result["status"] == "success"
        assert len(result["results"]) == 2

        # Check that executor was called with correct parameters
        graph_executor.executor.execute.assert_called_once()

    def test_function_node_without_input(self, graph_executor):
        """Test execution of a function node without input connections."""
        graph_data = {
            "nodes": [
                {
                    "id": "func_node",
                    "type": "function_def",
                    "data": {"name": "increment"},
                }
            ],
            "edges": [],
        }

        result = graph_executor.execute_graph(graph_data)

        assert result["status"] == "error"
        assert "has no input connections" in result["error"]

    def test_function_not_found(self, graph_executor):
        """Test execution with a function that doesn't exist."""
        graph_data = {
            "nodes": [
                {
                    "id": "input_node",
                    "type": "typed_list",
                    "data": {"item_type": "builtins.int", "items": [1, 2, 3]},
                },
                {
                    "id": "func_node",
                    "type": "function_def",
                    "data": {"name": "nonexistent_function"},
                },
            ],
            "edges": [{"source": "input_node", "target": "func_node"}],
        }

        result = graph_executor.execute_graph(graph_data)

        assert result["status"] == "error"
        assert "not found in available functions" in result["error"]

    def test_input_node_not_executed(self, graph_executor):
        """Test error when input node hasn't been executed yet."""
        # This is a bit tricky to test since topological sort should prevent this
        # We'll mock the topological sort to return wrong order
        with patch.object(
            graph_executor, "topological_sort", return_value=["func_node", "input_node"]
        ):
            graph_data = {
                "nodes": [
                    {
                        "id": "input_node",
                        "type": "typed_list",
                        "data": {"item_type": "builtins.int", "items": [1, 2, 3]},
                    },
                    {
                        "id": "func_node",
                        "type": "function_def",
                        "data": {"name": "increment"},
                    },
                ],
                "edges": [{"source": "input_node", "target": "func_node"}],
            }

            result = graph_executor.execute_graph(graph_data)

            assert result["status"] == "error"
            assert "has not been executed yet" in result["error"]

    @patch("importlib.import_module")
    def test_typed_list_import_error(self, mock_import, graph_executor):
        """Test error handling when importing typed list item type fails."""
        mock_import.side_effect = ImportError("Module not found")

        graph_data = {
            "nodes": [
                {
                    "id": "node1",
                    "type": "typed_list",
                    "data": {
                        "item_type": "nonexistent.module.Type",
                        "items": [1, 2, 3],
                    },
                }
            ],
            "edges": [],
        }

        result = graph_executor.execute_graph(graph_data)

        assert result["status"] == "error"
        assert "Error executing node node1" in result["error"]

    def test_unknown_node_type(self, graph_executor):
        """Test that unknown node types are skipped."""
        graph_data = {
            "nodes": [
                {"id": "unknown_node", "type": "unknown_type", "data": {}},
                {
                    "id": "typed_list_node",
                    "type": "typed_list",
                    "data": {"item_type": "builtins.int", "items": [1, 2, 3]},
                },
            ],
            "edges": [],
        }

        result = graph_executor.execute_graph(graph_data)

        assert result["status"] == "success"
        # Only the typed_list node should be in results
        assert len(result["results"]) == 1
        assert result["results"][0]["node_id"] == "typed_list_node"

    def test_complex_graph_execution(self, graph_executor):
        """Test execution of a more complex graph."""
        # Mock executor results
        mock_result1 = TypedList([2, 3, 4], item_type=int)
        mock_result2 = TypedList([4, 6, 8], item_type=int)
        graph_executor.executor.execute.side_effect = [mock_result1, mock_result2]

        graph_data = {
            "nodes": [
                {
                    "id": "input1",
                    "type": "typed_list",
                    "data": {"item_type": "builtins.int", "items": [1, 2, 3]},
                },
                {"id": "func1", "type": "function_def", "data": {"name": "increment"}},
                {"id": "func2", "type": "function_def", "data": {"name": "double"}},
            ],
            "edges": [
                {"source": "input1", "target": "func1"},
                {"source": "func1", "target": "func2"},
            ],
        }

        result = graph_executor.execute_graph(graph_data)

        assert result["status"] == "success"
        assert len(result["results"]) == 3  # input1, func1, func2

        # Verify executor was called twice (once for each function)
        assert graph_executor.executor.execute.call_count == 2

    def test_executor_exception(self, graph_executor):
        """Test handling of executor exceptions."""
        # Mock executor to raise an exception
        graph_executor.executor.execute.side_effect = Exception("Execution failed")

        graph_data = {
            "nodes": [
                {
                    "id": "input_node",
                    "type": "typed_list",
                    "data": {"item_type": "builtins.int", "items": [1, 2, 3]},
                },
                {
                    "id": "func_node",
                    "type": "function_def",
                    "data": {"name": "increment"},
                },
            ],
            "edges": [{"source": "input_node", "target": "func_node"}],
        }

        result = graph_executor.execute_graph(graph_data)

        assert result["status"] == "error"
        assert "Execution failed" in result["error"]


class TestEdgeCases:
    def test_empty_graph(self, graph_executor):
        """Test execution of an empty graph."""
        graph_data = {"nodes": [], "edges": []}

        result = graph_executor.execute_graph(graph_data)

        assert result["status"] == "success"
        assert result["results"] == []

    def test_graph_with_only_edges(self, graph_executor):
        """Test graph with edges but no corresponding nodes."""
        graph_data = {"nodes": [], "edges": [{"source": "A", "target": "B"}]}

        # This should raise an error due to edges referencing non-existent nodes
        with pytest.raises(KeyError):
            graph_executor.execute_graph(graph_data)

    def test_disconnected_graph_components(self, graph_executor):
        """Test graph with disconnected components."""
        graph_data = {
            "nodes": [
                {
                    "id": "node1",
                    "type": "typed_list",
                    "data": {"item_type": "builtins.int", "items": [1, 2, 3]},
                },
                {
                    "id": "node2",
                    "type": "typed_list",
                    "data": {"item_type": "builtins.int", "items": [4, 5, 6]},
                },
            ],
            "edges": [],  # No connections between nodes
        }

        result = graph_executor.execute_graph(graph_data)

        assert result["status"] == "success"
        assert len(result["results"]) == 2


class TestResultFormatting:
    def test_typed_list_result_formatting(self, graph_executor):
        """Test that TypedList results are properly formatted."""
        graph_data = {
            "nodes": [
                {
                    "id": "node1",
                    "type": "typed_list",
                    "data": {"item_type": "builtins.str", "items": ["hello", "world"]},
                }
            ],
            "edges": [],
        }

        result = graph_executor.execute_graph(graph_data)

        assert result["status"] == "success"
        assert len(result["results"]) == 1

        formatted_result = result["results"][0]
        assert formatted_result["node_id"] == "node1"
        assert formatted_result["result"] == ["hello", "world"]
        assert formatted_result["result_type"] == "builtins.str"

    def test_non_typed_list_result_handling(self, graph_executor, sample_functions):
        """Test handling of results that are not TypedList instances."""
        # This tests the case where executor returns something other than TypedList
        # which shouldn't happen in normal operation, but good to test robustness

        # Mock executor to return a non-TypedList result
        graph_executor.executor.execute.return_value = "not a typed list"

        graph_data = {
            "nodes": [
                {
                    "id": "input_node",
                    "type": "typed_list",
                    "data": {"item_type": "builtins.int", "items": [1, 2, 3]},
                },
                {
                    "id": "func_node",
                    "type": "function_def",
                    "data": {"name": "increment"},
                },
            ],
            "edges": [{"source": "input_node", "target": "func_node"}],
        }

        result = graph_executor.execute_graph(graph_data)

        # Should still succeed, but func_node won't appear in formatted results
        assert result["status"] == "success"
        # Only the input_node (TypedList) should be in results
        assert len(result["results"]) == 1
        assert result["results"][0]["node_id"] == "input_node"
