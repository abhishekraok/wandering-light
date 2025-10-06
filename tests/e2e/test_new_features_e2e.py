import time

from playwright.sync_api import Page, expect


class TestNewFeaturesE2E:
    """Test suite for new features in the graph editor."""

    def test_drag_function_node_to_canvas(self, app_page: Page):
        """Test dragging a function node from sidebar to canvas."""
        # Wait for functions to load
        app_page.wait_for_selector("text=inc", timeout=10000)

        # Get the inc function node and canvas
        inc_node = app_page.locator(".function-node").filter(has_text="inc")
        canvas = app_page.locator('[data-testid="graph-canvas"]')

        # Perform drag and drop
        inc_node.drag_to(canvas)

        # Wait for the node to appear on canvas
        time.sleep(1)

        # Verify a function node was added to the canvas
        canvas_nodes = app_page.locator(".react-flow__node")
        expect(canvas_nodes).to_have_count(1)

        # Verify the node contains the function name
        expect(canvas_nodes.first).to_contain_text("inc")

    def test_keyboard_node_deletion(self, app_page: Page):
        """Test deleting nodes using keyboard shortcuts."""
        # First, add a node to the canvas
        inc_node = app_page.locator(".function-node").filter(has_text="inc")
        canvas = app_page.locator('[data-testid="graph-canvas"]')
        inc_node.drag_to(canvas)

        # Wait for node to appear
        time.sleep(1)
        canvas_node = app_page.locator(".react-flow__node").first
        expect(canvas_node).to_be_visible()

        # Select the node by clicking on it
        canvas_node.click()

        # Press Delete key
        app_page.keyboard.press("Delete")

        # Verify the node is deleted
        expect(app_page.locator(".react-flow__node")).to_have_count(0)

    def test_backspace_node_deletion(self, app_page: Page):
        """Test deleting nodes using Backspace key."""
        # Add a node to the canvas
        inc_node = app_page.locator(".function-node").filter(has_text="double")
        canvas = app_page.locator('[data-testid="graph-canvas"]')
        inc_node.drag_to(canvas)

        # Wait for node to appear
        time.sleep(1)
        canvas_node = app_page.locator(".react-flow__node").first
        expect(canvas_node).to_be_visible()

        # Select the node
        canvas_node.click()

        # Press Backspace key
        app_page.keyboard.press("Backspace")

        # Verify the node is deleted
        expect(app_page.locator(".react-flow__node")).to_have_count(0)

    def test_ui_button_node_deletion(self, app_page: Page):
        """Test deleting nodes using the UI delete button."""
        # Add a function node to the canvas
        inc_node = app_page.locator(".function-node").filter(has_text="inc")
        canvas = app_page.locator('[data-testid="graph-canvas"]')
        inc_node.drag_to(canvas)

        # Wait for node to appear
        time.sleep(1)
        canvas_node = app_page.locator(".react-flow__node").first
        expect(canvas_node).to_be_visible()

        # Select the node to show the delete button
        canvas_node.click()

        # Wait for delete button to appear and click it
        delete_button = app_page.locator('[aria-label="Delete node"]')
        expect(delete_button).to_be_visible()
        delete_button.click()

        # Verify the node is deleted
        expect(app_page.locator(".react-flow__node")).to_have_count(0)

    def test_multi_node_selection_and_deletion(self, app_page: Page):
        """Test selecting and deleting multiple nodes."""
        # Add two nodes to the canvas
        inc_node = app_page.locator(".function-node").filter(has_text="inc")
        double_node = app_page.locator(".function-node").filter(has_text="double")
        canvas = app_page.locator('[data-testid="graph-canvas"]')

        inc_node.drag_to(canvas)
        time.sleep(0.5)
        double_node.drag_to(canvas)
        time.sleep(1)

        # Verify both nodes are on canvas
        expect(app_page.locator(".react-flow__node")).to_have_count(2)

        # Select first node
        app_page.locator(".react-flow__node").first.click()

        # Hold Ctrl and select second node
        app_page.keyboard.down("Control")
        app_page.locator(".react-flow__node").nth(1).click()
        app_page.keyboard.up("Control")

        # Delete selected nodes
        app_page.keyboard.press("Delete")

        # Verify both nodes are deleted
        expect(app_page.locator(".react-flow__node")).to_have_count(0)

    def test_typed_list_node_displays_parsed_items(self, app_page: Page):
        """Test that TypedList node preview shows correctly parsed items."""
        # Configure TypedList with custom items
        app_page.locator("text=Configure TypedList").click()
        items_textarea = app_page.locator(
            "textarea[placeholder=\"[1, 2, 3] or ['a', 'b', 'c']\"]"
        )
        items_textarea.clear()
        items_textarea.fill("[100, 200, 300]")

        # Check that the TypedList preview updates
        typed_list_preview = app_page.locator(".typed-list-node")
        expect(typed_list_preview).to_contain_text("[100, 200, 300]")

        # Test with string items
        items_textarea.clear()
        items_textarea.fill('["test", "items"]')
        expect(typed_list_preview).to_contain_text("[test, items]")

    def test_automatic_execution_on_connection(self, app_page: Page):
        """Test that connecting nodes automatically triggers execution."""
        # Add a TypedList node to canvas
        typed_list_node = app_page.locator(".typed-list-node")
        canvas = app_page.locator('[data-testid="graph-canvas"]')
        typed_list_node.drag_to(canvas, target_position={"x": 100, "y": 100})

        # Add an inc function node to canvas
        inc_node = app_page.locator(".function-node").filter(has_text="inc")
        inc_node.drag_to(canvas, target_position={"x": 300, "y": 100})

        time.sleep(1)

        # Verify both nodes are on canvas
        expect(app_page.locator(".react-flow__node")).to_have_count(2)

        # Connect the nodes by dragging from TypedList output to inc input
        source_handle = app_page.locator(".react-flow__node").first.locator(
            ".react-flow__handle-right"
        )
        target_handle = (
            app_page.locator(".react-flow__node")
            .nth(1)
            .locator(".react-flow__handle-left")
        )

        source_handle.drag_to(target_handle)
        time.sleep(2)

        # Verify connection was made
        expect(app_page.locator(".react-flow__edge")).to_have_count(1)

        # Verify automatic execution occurred (result should be visible)
        # The inc function should show results in the node
        function_node = app_page.locator(".react-flow__node").nth(1)
        expect(function_node).to_contain_text("Result")

    def test_execution_results_display_in_nodes(self, app_page: Page):
        """Test that execution results are displayed within nodes."""
        # Add a TypedList node to canvas
        typed_list_node = app_page.locator(".typed-list-node")
        canvas = app_page.locator('[data-testid="graph-canvas"]')
        typed_list_node.drag_to(canvas, target_position={"x": 100, "y": 100})

        # Add a double function node to canvas
        double_node = app_page.locator(".function-node").filter(has_text="double")
        double_node.drag_to(canvas, target_position={"x": 300, "y": 100})

        time.sleep(1)

        # Connect the nodes
        source_handle = app_page.locator(".react-flow__node").first.locator(
            ".react-flow__handle-right"
        )
        target_handle = (
            app_page.locator(".react-flow__node")
            .nth(1)
            .locator(".react-flow__handle-left")
        )

        source_handle.drag_to(target_handle)
        time.sleep(2)

        # Verify the result is displayed in the function node
        function_node = app_page.locator(".react-flow__node").nth(1)
        expect(function_node).to_contain_text("Result")

        # Check that the result shows the doubled values
        # TypedList default is [1, 2, 3], so double should give [2, 4, 6]
        expect(function_node).to_contain_text("2")
        expect(function_node).to_contain_text("4")
        expect(function_node).to_contain_text("6")

    def test_websocket_connection_for_auto_execution(self, app_page: Page):
        """Test that WebSocket connection is established for auto execution."""
        # Add a TypedList node to canvas
        typed_list_node = app_page.locator(".typed-list-node")
        canvas = app_page.locator('[data-testid="graph-canvas"]')
        typed_list_node.drag_to(canvas, target_position={"x": 100, "y": 100})

        # Verify WebSocket connection status indicator
        # This depends on the specific implementation, but we can check for connection status
        time.sleep(2)

        # The connection should be established automatically
        # We can verify this by checking that auto-execution works
        inc_node = app_page.locator(".function-node").filter(has_text="inc")
        inc_node.drag_to(canvas, target_position={"x": 300, "y": 100})

        time.sleep(1)

        # Connect nodes and verify automatic execution
        source_handle = app_page.locator(".react-flow__node").first.locator(
            ".react-flow__handle-right"
        )
        target_handle = (
            app_page.locator(".react-flow__node")
            .nth(1)
            .locator(".react-flow__handle-left")
        )

        source_handle.drag_to(target_handle)
        time.sleep(2)

        # If WebSocket is working, we should see results
        function_node = app_page.locator(".react-flow__node").nth(1)
        expect(function_node).to_contain_text("Result")

    def test_delete_button_appears_only_when_selected(self, app_page: Page):
        """Test that delete button only appears when nodes are selected."""
        # Add a node to canvas
        inc_node = app_page.locator(".function-node").filter(has_text="inc")
        canvas = app_page.locator('[data-testid="graph-canvas"]')
        inc_node.drag_to(canvas)

        time.sleep(1)

        # Initially, delete button should not be visible
        delete_button = app_page.locator('[aria-label="Delete node"]')
        expect(delete_button).not_to_be_visible()

        # Select the node
        canvas_node = app_page.locator(".react-flow__node").first
        canvas_node.click()

        # Now delete button should be visible
        expect(delete_button).to_be_visible()

        # Click somewhere else to deselect
        canvas.click(position={"x": 500, "y": 500})
        time.sleep(0.5)

        # Delete button should be hidden again
        expect(delete_button).not_to_be_visible()
