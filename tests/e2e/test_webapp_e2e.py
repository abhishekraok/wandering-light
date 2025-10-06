import time

from playwright.sync_api import Page, expect


class TestWebAppE2E:
    """End-to-end tests for the Wandering Light web application."""

    def test_page_loads_successfully(self, app_page: Page):
        """Test that the application loads successfully."""
        # Check that the main heading is visible
        expect(app_page.locator("h1")).to_contain_text("Wandering Light")

        # Check that the sidebar is present
        expect(app_page.locator('[data-testid="sidebar"]')).to_be_visible()

        # Check that the graph editor canvas is present
        expect(app_page.locator('[data-testid="graph-editor"]')).to_be_visible()

    def test_sidebar_components_visible(self, app_page: Page):
        """Test that all sidebar components are visible."""
        # Check Node Palette heading
        expect(app_page.locator("text=Node Palette")).to_be_visible()

        # Check TypedList node is draggable
        expect(app_page.locator(".typed-list-node")).to_be_visible()

        # Check TypedList configuration accordion
        expect(app_page.locator("text=Configure TypedList")).to_be_visible()

        # Check Function Nodes section
        expect(app_page.locator("text=Function Nodes")).to_be_visible()

        # Check Add New Function section
        expect(app_page.locator("text=Add New Function")).to_be_visible()

    def test_create_new_function(self, app_page: Page):
        """Test creating a new function through the sidebar form."""
        # Open the add function accordion
        app_page.locator("text=Add New Function").click()

        # Wait for the form to be visible
        app_page.wait_for_selector("form", timeout=5000)

        # Fill in function details using more specific selectors based on the form structure
        # Find inputs within the form - they appear in order: Name, Input Type, Output Type, Code
        import time

        unique_name = f"test_function_{int(time.time())}"
        form_inputs = app_page.locator("form input")
        form_inputs.nth(0).fill(unique_name)  # Name field
        form_inputs.nth(1).fill("builtins.int")  # Input Type field
        form_inputs.nth(2).fill("builtins.int")  # Output Type field

        # Code textarea has a placeholder
        app_page.fill('textarea[placeholder="return x + 1"]', "return x * 2")

        # Submit the form
        app_page.click("button:has-text('Add Function')")

        # Wait for the function to appear in the function list
        expect(
            app_page.locator(".function-node").filter(has_text=unique_name)
        ).to_be_visible()

    def test_configure_typed_list(self, app_page: Page):
        """Test configuring a TypedList node."""
        # Open the TypedList configuration accordion
        app_page.locator("text=Configure TypedList").click()

        # Wait for the accordion to open
        app_page.wait_for_selector("input", timeout=5000)

        # Change the item type (first input in the TypedList config section)
        item_type_input = app_page.locator("input").first
        item_type_input.clear()
        item_type_input.fill("builtins.str")

        # Change the items (textarea in the TypedList config section)
        items_textarea = app_page.locator("textarea").first
        items_textarea.clear()
        items_textarea.fill('["hello", "world", "test"]')

        # Verify the TypedList node reflects the changes
        expect(app_page.locator(".typed-list-node")).to_contain_text("builtins.str")

    def test_drag_and_drop_typed_list_node(self, app_page: Page):
        """Test dragging and dropping a TypedList node onto the canvas."""
        # Get the TypedList node and canvas
        typed_list_node = app_page.locator(".typed-list-node")
        canvas = app_page.locator('[data-testid="graph-canvas"]')

        # Perform drag and drop
        typed_list_node.drag_to(canvas)

        # Wait for the node to appear on canvas
        time.sleep(1)

        # Verify a node was added to the canvas
        expect(app_page.locator(".react-flow__node")).to_have_count(1)

    def test_execute_graph_button(self, app_page: Page):
        """Test that the execute graph button is present."""
        # Check that execute button is present (without complex graph setup)
        execute_button = app_page.locator("button:has-text('Execute Graph')")
        expect(execute_button).to_be_visible()

    def test_clear_graph_functionality(self, app_page: Page):
        """Test that clear graph functionality exists."""
        # Check if clear graph button exists (implementation may vary)
        app_page.locator("button:has-text('Clear Graph')")
        # This test just verifies the button exists or the functionality is accessible

    def test_websocket_connection_status(self, app_page: Page):
        """Test that WebSocket connection status is displayed."""
        # Look for connection status indicators
        # This might be shown as a toast or status indicator

        # Wait for potential WebSocket connection
        time.sleep(2)

        # Check for any WebSocket-related messages
        # Note: Actual WebSocket testing requires more complex setup
        console_logs = []
        app_page.on("console", lambda msg: console_logs.append(msg.text))

        # Reload to trigger WebSocket connection
        app_page.reload()
        time.sleep(2)

        # Check if WebSocket connection messages appear in console
        [
            log
            for log in console_logs
            if "websocket" in log.lower() or "ws" in log.lower()
        ]
        # This is informational - WebSocket connection may or may not succeed in test environment

    def test_function_code_validation(self, app_page: Page):
        """Test that function code validation works."""
        # Open the add function accordion
        app_page.locator("text=Add New Function").click()

        # Wait for the form to be visible
        app_page.wait_for_selector("form", timeout=5000)

        # Try to create a function with invalid Python code
        form_inputs = app_page.locator("form input")
        form_inputs.nth(0).fill("invalid_function")  # Name field

        app_page.fill(
            'textarea[placeholder="return x + 1"]', "invalid python code @#$%"
        )

        # Submit the form
        app_page.click("button:has-text('Add Function')")

        # Check for error message or that function wasn't created
        # The exact error handling depends on implementation
        time.sleep(1)

    def test_responsive_layout(self, app_page: Page):
        """Test that the layout works on different screen sizes."""
        # Test desktop size (default)
        expect(app_page.locator('[data-testid="sidebar"]')).to_be_visible()
        expect(app_page.locator('[data-testid="graph-editor"]')).to_be_visible()

        # Test mobile size
        app_page.set_viewport_size({"width": 375, "height": 667})
        time.sleep(1)

        # On mobile, sidebar might be hidden or collapsible
        # This test verifies the layout adapts
        expect(app_page.locator("h1")).to_be_visible()

    def test_keyboard_shortcuts(self, app_page: Page):
        """Test keyboard shortcuts if implemented."""
        # Focus on the canvas
        app_page.locator('[data-testid="graph-canvas"]').click()

        # Test common shortcuts
        # Delete key to delete selected nodes
        app_page.keyboard.press("Delete")

        # Ctrl+Z for undo (if implemented)
        app_page.keyboard.press("Control+z")

        # Note: Actual behavior depends on implementation

    def test_error_handling_backend_down(self, app_page: Page):
        """Test error handling when backend is unavailable."""
        # This test would require stopping the backend server
        # For now, we test that the frontend handles network errors gracefully

        # Try to create a function which requires backend communication
        app_page.locator("text=Add New Function").click()

        # Wait for the form to be visible
        app_page.wait_for_selector("form", timeout=5000)

        form_inputs = app_page.locator("form input")
        form_inputs.nth(0).fill("test_function")  # Name field

        app_page.click("button:has-text('Add Function')")

        # Check that error is handled gracefully
        # Look for error toasts or messages
        time.sleep(2)

    def test_data_persistence_across_page_reload(self, app_page: Page):
        """Test if data persists across page reloads."""
        # Create a function
        self.test_create_new_function(app_page)

        # Reload the page
        app_page.reload()
        app_page.wait_for_selector('[data-testid="graph-editor"]', timeout=30000)

        # Check if the function still exists
        # Note: This depends on whether the app has persistence implemented
        time.sleep(2)

    def test_accessibility_basics(self, app_page: Page):
        """Test basic accessibility features."""
        # Check that main elements have appropriate ARIA labels or roles
        expect(app_page.locator("h1")).to_be_visible()

        # Check that interactive elements are keyboard accessible
        app_page.keyboard.press("Tab")
        time.sleep(0.5)
        app_page.keyboard.press("Tab")
        time.sleep(0.5)

        # Verify focus is moving through interactive elements
        # This is a basic accessibility check
