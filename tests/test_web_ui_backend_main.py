import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from wandering_light.web_ui.backend.main import (
    app,
    available_functions,
)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_function():
    """Create a sample function for testing."""
    return {
        "name": "test_increment",
        "input_type": "builtins.int",
        "output_type": "builtins.int",
        "code": "return x + 1",
        "metadata": {"description": "Increment function"},
    }


@pytest.fixture
def sample_typed_list():
    """Create a sample typed list for testing."""
    return {"item_type": "builtins.int", "items": [1, 2, 3]}


@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    return {
        "name": "test_graph",
        "nodes": [
            {
                "id": "node1",
                "type": "typed_list",
                "data": {"item_type": "builtins.int", "items": [1, 2, 3]},
                "position": {"x": 0, "y": 0},
            },
            {
                "id": "node2",
                "type": "function_def",
                "data": {"name": "test_increment"},
                "position": {"x": 100, "y": 0},
            },
        ],
        "edges": [
            {
                "id": "edge1",
                "source": "node1",
                "target": "node2",
                "sourceHandle": "output",
                "targetHandle": "input",
            }
        ],
    }


class TestRootEndpoint:
    def test_read_root(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Wandering Light Graph Editor API"}


class TestFunctionEndpoints:
    def test_get_functions_empty(self, client):
        """Test getting functions when list is empty."""
        # Clear the available functions
        available_functions.clear()
        response = client.get("/functions")
        assert response.status_code == 200
        assert response.json() == []

    def test_create_function_success(self, client, sample_function):
        """Test creating a function successfully."""
        # Clear the available functions first
        available_functions.clear()

        response = client.post("/functions", json=sample_function)
        assert response.status_code == 200

        response_data = response.json()
        assert response_data["name"] == sample_function["name"]
        assert response_data["input_type"] == sample_function["input_type"]
        assert response_data["output_type"] == sample_function["output_type"]
        assert response_data["code"] == sample_function["code"]

    def test_create_function_duplicate(self, client, sample_function):
        """Test creating a duplicate function raises error."""
        # Clear and add one function
        available_functions.clear()
        client.post("/functions", json=sample_function)

        # Try to create the same function again
        response = client.post("/functions", json=sample_function)
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]

    def test_get_function_success(self, client, sample_function):
        """Test getting a specific function successfully."""
        # Clear and add a function
        available_functions.clear()
        client.post("/functions", json=sample_function)

        response = client.get(f"/functions/{sample_function['name']}")
        assert response.status_code == 200
        assert response.json()["name"] == sample_function["name"]

    def test_get_function_not_found(self, client):
        """Test getting a non-existent function."""
        # Clear the available functions
        available_functions.clear()

        response = client.get("/functions/nonexistent")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_get_functions_with_data(self, client, sample_function):
        """Test getting functions when list has data."""
        # Clear and add a function
        available_functions.clear()
        client.post("/functions", json=sample_function)

        response = client.get("/functions")
        assert response.status_code == 200
        functions = response.json()
        assert len(functions) == 1
        assert functions[0]["name"] == sample_function["name"]


class TestExecuteEndpoint:
    def test_execute_function_success(self, client, sample_function, sample_typed_list):
        """Test executing a function successfully."""
        # Clear and add a function
        available_functions.clear()
        client.post("/functions", json=sample_function)

        response = client.post(
            f"/execute?function_name={sample_function['name']}", json=sample_typed_list
        )
        assert response.status_code == 200

        result = response.json()
        assert "items" in result
        assert "item_type" in result
        # The increment function should add 1 to each item
        assert result["items"] == [2, 3, 4]

    def test_execute_function_not_found(self, client, sample_typed_list):
        """Test executing a non-existent function."""
        # Clear the available functions
        available_functions.clear()

        response = client.post(
            "/execute?function_name=nonexistent", json=sample_typed_list
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @patch("importlib.import_module")
    def test_execute_function_import_error(self, mock_import, client, sample_function):
        """Test executing a function with import error."""
        # Clear and add a function
        available_functions.clear()
        client.post("/functions", json=sample_function)

        # Mock import error
        mock_import.side_effect = ImportError("Module not found")

        sample_typed_list = {"item_type": "nonexistent.module.Type", "items": [1, 2, 3]}

        response = client.post(
            f"/execute?function_name={sample_function['name']}", json=sample_typed_list
        )
        assert response.status_code == 400


class TestWebSocketEndpoint:
    @patch("wandering_light.web_ui.backend.main.graph_executor")
    def test_websocket_execute_graph_success(
        self, mock_graph_executor, client, sample_graph
    ):
        """Test WebSocket graph execution success."""
        # Mock successful execution
        mock_graph_executor.execute_graph.return_value = {
            "status": "success",
            "results": [],
        }

        with client.websocket_connect("/ws/execute-graph") as websocket:
            # Send graph data
            websocket.send_text(json.dumps(sample_graph))

            # Receive response
            data = websocket.receive_json()
            assert data["status"] == "success"

    @patch("wandering_light.web_ui.backend.main.graph_executor")
    def test_websocket_execute_graph_error(
        self, mock_graph_executor, client, sample_graph
    ):
        """Test WebSocket graph execution error."""
        # Mock execution error
        mock_graph_executor.execute_graph.side_effect = Exception("Execution failed")

        with client.websocket_connect("/ws/execute-graph") as websocket:
            # Send graph data
            websocket.send_text(json.dumps(sample_graph))

            # Receive response
            data = websocket.receive_json()
            assert data["status"] == "error"
            assert "Execution failed" in data["error"]

    def test_websocket_disconnect(self, client):
        """Test WebSocket disconnection handling."""
        # This test verifies that the websocket endpoint can be connected to
        # and disconnected from without errors
        with client.websocket_connect("/ws/execute-graph"):
            # Just test that we can connect and disconnect
            pass


class TestDataModels:
    def test_function_def_create_model(self, sample_function):
        """Test FunctionDefCreate model validation."""
        from wandering_light.web_ui.backend.main import FunctionDefCreate

        # Test valid data
        func = FunctionDefCreate(**sample_function)
        assert func.name == sample_function["name"]
        assert func.input_type == sample_function["input_type"]
        assert func.output_type == sample_function["output_type"]
        assert func.code == sample_function["code"]
        assert func.metadata == sample_function["metadata"]

    def test_typed_list_create_model(self, sample_typed_list):
        """Test TypedListCreate model validation."""
        from wandering_light.web_ui.backend.main import TypedListCreate

        typed_list = TypedListCreate(**sample_typed_list)
        assert typed_list.item_type == sample_typed_list["item_type"]
        assert typed_list.items == sample_typed_list["items"]

    def test_graph_create_model(self, sample_graph):
        """Test GraphCreate model validation."""
        from wandering_light.web_ui.backend.main import GraphCreate

        graph = GraphCreate(**sample_graph)
        assert graph.name == sample_graph["name"]
        assert len(graph.nodes) == len(sample_graph["nodes"])
        assert len(graph.edges) == len(sample_graph["edges"])

    def test_execution_result_model(self):
        """Test ExecutionResult model validation."""
        from wandering_light.web_ui.backend.main import ExecutionResult

        result_data = {
            "node_id": "test_node",
            "result": [1, 2, 3],
            "result_type": "builtins.int",
        }

        result = ExecutionResult(**result_data)
        assert result.node_id == result_data["node_id"]
        assert result.result == result_data["result"]
        assert result.result_type == result_data["result_type"]


class TestCORSMiddleware:
    def test_cors_headers(self, client):
        """Test that CORS headers are properly set."""
        response = client.get("/")
        # Basic test that the response is successful
        # Full CORS testing would require a more complex setup
        assert response.status_code == 200


class TestErrorHandling:
    def test_function_creation_with_invalid_code(self, client):
        """Test function creation with invalid code."""
        invalid_function = {
            "name": "invalid_func",
            "input_type": "builtins.int",
            "output_type": "builtins.int",
            "code": "invalid python code syntax $$",
            "metadata": {},
        }

        # Clear functions first
        available_functions.clear()

        response = client.post("/functions", json=invalid_function)
        # The function should be created (syntax error will occur during execution)
        assert response.status_code == 200

    def test_execute_with_invalid_type(self, client, sample_function):
        """Test execution with invalid type specification."""
        # Clear and add a function
        available_functions.clear()
        client.post("/functions", json=sample_function)

        invalid_typed_list = {"item_type": "nonexistent.Type", "items": [1, 2, 3]}

        response = client.post(
            f"/execute?function_name={sample_function['name']}", json=invalid_typed_list
        )
        assert response.status_code == 400


# Integration test
class TestEndToEndWorkflow:
    def test_complete_workflow(self, client, sample_function, sample_typed_list):
        """Test a complete workflow from function creation to execution."""
        # Clear functions
        available_functions.clear()

        # 1. Create function
        response = client.post("/functions", json=sample_function)
        assert response.status_code == 200

        # 2. Get all functions
        response = client.get("/functions")
        assert response.status_code == 200
        assert len(response.json()) == 1

        # 3. Get specific function
        response = client.get(f"/functions/{sample_function['name']}")
        assert response.status_code == 200

        # 4. Execute function
        response = client.post(
            f"/execute?function_name={sample_function['name']}", json=sample_typed_list
        )
        assert response.status_code == 200
        result = response.json()
        assert result["items"] == [2, 3, 4]  # incremented values
