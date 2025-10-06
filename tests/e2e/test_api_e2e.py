import json
import time

import pytest
from playwright.sync_api import APIRequestContext, sync_playwright


class TestAPIE2E:
    """End-to-end tests for the API endpoints."""

    @pytest.fixture
    def api_request_context(self, backend_server) -> APIRequestContext:
        """Create an API request context for testing backend endpoints."""

        with sync_playwright() as p:
            request_context = p.request.new_context(
                base_url="http://localhost:8000",
                extra_http_headers={"Content-Type": "application/json"},
            )
            yield request_context
            request_context.dispose()

    def _get_unique_name(self, base_name: str) -> str:
        """Generate a unique function name to avoid conflicts between tests."""
        timestamp = str(int(time.time() * 1000000))[-8:]
        return f"{base_name}_{timestamp}"

    def test_root_endpoint(self, api_request_context: APIRequestContext):
        """Test the root API endpoint."""
        response = api_request_context.get("/")
        assert response.status == 200

        data = response.json()
        assert "message" in data
        assert data["message"] == "Wandering Light Graph Editor API"

    def test_get_functions_empty_or_with_existing(
        self, api_request_context: APIRequestContext
    ):
        """Test getting functions endpoint returns a list."""
        response = api_request_context.get("/functions")
        assert response.status == 200

        data = response.json()
        assert isinstance(data, list)

    def test_create_function_valid(self, api_request_context: APIRequestContext):
        """Test creating a valid function."""
        function_data = {
            "name": self._get_unique_name("test_function"),
            "input_type": "builtins.int",
            "output_type": "builtins.int",
            "code": "return x + 1",
            "metadata": {"description": "Adds 1 to input"},
        }

        response = api_request_context.post(
            "/functions", data=json.dumps(function_data)
        )
        assert response.status == 200

        data = response.json()
        assert data["name"] == function_data["name"]
        assert data["input_type"] == function_data["input_type"]
        assert data["output_type"] == function_data["output_type"]
        assert data["code"] == function_data["code"]

    def test_create_function_duplicate_name(
        self, api_request_context: APIRequestContext
    ):
        """Test creating a function with duplicate name fails."""
        function_name = self._get_unique_name("duplicate_function")
        function_data = {
            "name": function_name,
            "input_type": "builtins.int",
            "output_type": "builtins.int",
            "code": "return x + 1",
        }

        # Create first function
        response1 = api_request_context.post(
            "/functions", data=json.dumps(function_data)
        )
        assert response1.status == 200

        # Try to create duplicate
        response2 = api_request_context.post(
            "/functions", data=json.dumps(function_data)
        )
        assert response2.status == 400

        error_data = response2.json()
        assert "already exists" in error_data["detail"]

    def test_get_function_by_name(self, api_request_context: APIRequestContext):
        """Test getting a specific function by name."""
        # First create a function
        function_data = {
            "name": self._get_unique_name("get_test_function"),
            "input_type": "builtins.int",
            "output_type": "builtins.int",
            "code": "return x * 2",
        }

        create_response = api_request_context.post(
            "/functions", data=json.dumps(function_data)
        )
        assert create_response.status == 200

        # Now get it by name
        get_response = api_request_context.get(f"/functions/{function_data['name']}")
        assert get_response.status == 200

        data = get_response.json()
        assert data["name"] == function_data["name"]

    def test_get_nonexistent_function(self, api_request_context: APIRequestContext):
        """Test getting a function that doesn't exist."""
        response = api_request_context.get("/functions/nonexistent_function_12345")
        assert response.status == 404

        error_data = response.json()
        assert "not found" in error_data["detail"]

    def test_execute_function_valid(self, api_request_context: APIRequestContext):
        """Test executing a valid function."""
        # First create a function
        function_data = {
            "name": self._get_unique_name("execute_test_function"),
            "input_type": "builtins.int",
            "output_type": "builtins.int",
            "code": "return x * 3",
        }

        create_response = api_request_context.post(
            "/functions", data=json.dumps(function_data)
        )
        assert create_response.status == 200

        # Execute the function
        execution_data = {"item_type": "builtins.int", "items": [1, 2, 3, 4, 5]}

        execute_response = api_request_context.post(
            f"/execute?function_name={function_data['name']}",
            data=json.dumps(execution_data),
        )
        assert execute_response.status == 200

        result = execute_response.json()
        assert "items" in result
        assert result["items"] == [3, 6, 9, 12, 15]  # Each input multiplied by 3

    def test_execute_nonexistent_function(self, api_request_context: APIRequestContext):
        """Test executing a function that doesn't exist."""
        execution_data = {"item_type": "builtins.int", "items": [1, 2, 3]}

        response = api_request_context.post(
            "/execute?function_name=nonexistent_function_12345",
            data=json.dumps(execution_data),
        )
        # Backend returns 400 for execution errors, which is reasonable
        assert response.status in [400, 404]

        error_data = response.json()
        assert "detail" in error_data

    def test_function_with_invalid_code(self, api_request_context: APIRequestContext):
        """Test creating a function with invalid Python code."""
        function_data = {
            "name": self._get_unique_name("invalid_code_function"),
            "input_type": "builtins.int",
            "output_type": "builtins.int",
            "code": "invalid python syntax @#$%^&*()",
        }

        response = api_request_context.post(
            "/functions", data=json.dumps(function_data)
        )
        # The API accepts the function but execution will fail
        # This tests that the API doesn't validate Python syntax at creation time
        assert response.status == 200

    def test_basic_connectivity(self, api_request_context: APIRequestContext):
        """Test basic API connectivity and response format."""
        response = api_request_context.get("/")
        assert response.status == 200

        # Verify basic response structure
        data = response.json()
        assert isinstance(data, dict)

    def test_function_list_after_creation(self, api_request_context: APIRequestContext):
        """Test that function list includes newly created functions."""
        # Get initial count
        initial_response = api_request_context.get("/functions")
        initial_count = len(initial_response.json())

        # Create a new function
        function_data = {
            "name": self._get_unique_name("list_test_function"),
            "input_type": "builtins.str",
            "output_type": "builtins.str",
            "code": "return x.upper()",
        }

        create_response = api_request_context.post(
            "/functions", data=json.dumps(function_data)
        )
        assert create_response.status == 200

        # Check updated list
        updated_response = api_request_context.get("/functions")
        updated_functions = updated_response.json()
        assert len(updated_functions) == initial_count + 1

        # Verify our function is in the list
        function_names = [f["name"] for f in updated_functions]
        assert function_data["name"] in function_names

    def test_function_metadata_preservation(
        self, api_request_context: APIRequestContext
    ):
        """Test that function metadata is preserved correctly."""
        function_data = {
            "name": self._get_unique_name("metadata_test_function"),
            "input_type": "builtins.int",
            "output_type": "builtins.int",
            "code": "return x + 10",
            "metadata": {
                "description": "Adds 10 to the input",
                "author": "test_user",
                "version": "1.0.0",
                "tags": ["math", "simple"],
            },
        }

        create_response = api_request_context.post(
            "/functions", data=json.dumps(function_data)
        )
        assert create_response.status == 200

        # Get the function back
        get_response = api_request_context.get(f"/functions/{function_data['name']}")
        retrieved_function = get_response.json()

        # Verify metadata is preserved
        assert retrieved_function["metadata"] == function_data["metadata"]

    def test_execute_with_different_data_types(
        self, api_request_context: APIRequestContext
    ):
        """Test executing functions with different data types."""
        # String function
        string_function = {
            "name": self._get_unique_name("string_test_function"),
            "input_type": "builtins.str",
            "output_type": "builtins.str",
            "code": "return x + '_processed'",
        }

        api_request_context.post("/functions", data=json.dumps(string_function))

        # Execute with string data
        string_execution = {
            "item_type": "builtins.str",
            "items": ["hello", "world", "test"],
        }

        response = api_request_context.post(
            f"/execute?function_name={string_function['name']}",
            data=json.dumps(string_execution),
        )
        assert response.status == 200

        result = response.json()
        expected = ["hello_processed", "world_processed", "test_processed"]
        assert result["items"] == expected
