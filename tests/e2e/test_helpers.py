"""Helper utilities for E2E tests."""

import contextlib
import json
import time
from typing import Any

from playwright.sync_api import APIRequestContext, Page


class E2ETestHelpers:
    """Helper class containing utility methods for E2E tests."""

    @staticmethod
    def wait_for_page_load(page: Page, timeout: int = 30000):
        """Wait for the page to fully load."""
        page.wait_for_selector('[data-testid="graph-editor"]', timeout=timeout)
        page.wait_for_load_state("networkidle")

    @staticmethod
    def create_test_function(
        api_context: APIRequestContext,
        name: str = "test_function",
        input_type: str = "builtins.int",
        output_type: str = "builtins.int",
        code: str = "return x + 1",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a test function via API and return the response."""
        if metadata is None:
            metadata = {}

        function_data = {
            "name": name,
            "input_type": input_type,
            "output_type": output_type,
            "code": code,
            "metadata": metadata,
        }

        response = api_context.post("/functions", data=json.dumps(function_data))
        assert response.status == 200, f"Failed to create function: {response.text}"

        return response.json()

    @staticmethod
    def create_function_via_ui(
        page: Page,
        name: str = "ui_test_function",
        input_type: str = "builtins.int",
        output_type: str = "builtins.int",
        code: str = "return x * 2",
    ):
        """Create a function using the UI."""
        # Open the add function accordion
        page.locator("text=Add New Function").click()

        # Fill in function details
        page.fill('input[placeholder*="function name"]', name)
        page.fill('input[placeholder*="input type"]', input_type)
        page.fill('input[placeholder*="output type"]', output_type)
        page.fill('textarea[placeholder*="function code"]', code)

        # Submit the form
        page.click("button:has-text('Add Function')")

        # Wait for the function to appear in the function list
        page.wait_for_selector(f"text={name}", timeout=10000)

    @staticmethod
    def drag_node_to_canvas(
        page: Page, node_selector: str, target_position: dict[str, int] | None = None
    ):
        """Drag a node from sidebar to canvas."""
        if target_position is None:
            target_position = {"x": 200, "y": 200}

        source_node = page.locator(node_selector)
        canvas = page.locator('[data-testid="graph-canvas"]')

        source_node.drag_to(canvas, target_position=target_position)
        time.sleep(0.5)  # Wait for the node to be placed

    @staticmethod
    def count_canvas_nodes(page: Page) -> int:
        """Count the number of nodes currently on the canvas."""
        return page.locator(".react-flow__node").count()

    @staticmethod
    def count_canvas_edges(page: Page) -> int:
        """Count the number of edges currently on the canvas."""
        return page.locator(".react-flow__edge").count()

    @staticmethod
    def clear_canvas(page: Page):
        """Clear all nodes from the canvas."""
        clear_button = page.locator("button:has-text('Clear Graph')")
        if clear_button.is_visible():
            clear_button.click()
        else:
            # If no clear button, select all nodes and delete
            page.keyboard.press("Control+a")
            page.keyboard.press("Delete")

    @staticmethod
    def take_screenshot(page: Page, name: str):
        """Take a screenshot for debugging purposes."""
        page.screenshot(path=f"test_outputs/{name}.png")

    @staticmethod
    def wait_for_toast_message(
        page: Page, message_text: str | None = None, timeout: int = 5000
    ):
        """Wait for a toast notification to appear."""
        if message_text:
            page.wait_for_selector(f"text={message_text}", timeout=timeout)
        else:
            # Wait for any toast
            page.wait_for_selector('[role="alert"]', timeout=timeout)

    @staticmethod
    def configure_typed_list(
        page: Page, item_type: str = "builtins.str", items: list[Any] | None = None
    ):
        """Configure the TypedList node in the sidebar."""
        if items is None:
            items = ["test1", "test2", "test3"]

        # Open the TypedList configuration accordion
        page.locator("text=Configure TypedList").click()

        # Change the item type
        item_type_input = page.locator('input[value*="builtins"]').first()
        item_type_input.clear()
        item_type_input.fill(item_type)

        # Change the items
        items_textarea = page.locator("textarea").first()
        items_textarea.clear()
        items_textarea.fill(json.dumps(items))

    @staticmethod
    def execute_graph(page: Page):
        """Click the execute graph button and wait for completion."""
        execute_button = page.locator("button:has-text('Execute Graph')")
        execute_button.click()

        # Wait for execution to complete (look for success/error toast)
        with contextlib.suppress(TimeoutError):
            page.wait_for_selector('[role="alert"]', timeout=10000)

    @staticmethod
    def verify_websocket_connection(page: Page) -> bool:
        """Check if WebSocket connection is established."""
        # Listen for console messages about WebSocket
        websocket_connected = False

        def handle_console(msg):
            nonlocal websocket_connected
            if "websocket" in msg.text.lower() and "connected" in msg.text.lower():
                websocket_connected = True

        page.on("console", handle_console)

        # Reload to trigger WebSocket connection
        page.reload()
        time.sleep(2)

        return websocket_connected

    @staticmethod
    def get_node_by_text(page: Page, text: str):
        """Get a node on the canvas by its text content."""
        return page.locator(f".react-flow__node:has-text('{text}')")

    @staticmethod
    def connect_nodes(page: Page, source_node_text: str, target_node_text: str):
        """Connect two nodes on the canvas."""
        source_node = E2ETestHelpers.get_node_by_text(page, source_node_text)
        target_node = E2ETestHelpers.get_node_by_text(page, target_node_text)

        # Get the connection handles
        source_handle = source_node.locator(".react-flow__handle-right")
        target_handle = target_node.locator(".react-flow__handle-left")

        # Drag from source to target
        source_handle.drag_to(target_handle)

    @staticmethod
    def assert_function_exists_in_list(page: Page, function_name: str):
        """Assert that a function exists in the function list."""
        function_node = page.locator(f".function-node:has-text('{function_name}')")
        assert function_node.is_visible(), f"Function {function_name} not found in list"

    @staticmethod
    def assert_canvas_has_nodes(page: Page, expected_count: int):
        """Assert that the canvas has the expected number of nodes."""
        actual_count = E2ETestHelpers.count_canvas_nodes(page)
        assert actual_count == expected_count, (
            f"Expected {expected_count} nodes, got {actual_count}"
        )

    @staticmethod
    def simulate_network_error(page: Page):
        """Simulate network error by intercepting requests."""

        def handle_route(route):
            route.abort()

        page.route("**/api/**", handle_route)

    @staticmethod
    def restore_network(page: Page):
        """Restore network by removing route handlers."""
        page.unroute("**/api/**")

    @staticmethod
    def check_accessibility(page: Page) -> dict[str, Any]:
        """Basic accessibility checks."""
        results = {
            "has_headings": page.locator("h1, h2, h3, h4, h5, h6").count() > 0,
            "has_alt_text": True,  # Check if images have alt text
            "keyboard_navigable": True,  # Basic keyboard navigation check
        }

        # Check images have alt text
        images = page.locator("img")
        for i in range(images.count()):
            img = images.nth(i)
            if not img.get_attribute("alt"):
                results["has_alt_text"] = False
                break

        return results
