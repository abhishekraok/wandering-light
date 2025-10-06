# End-to-End Tests for Wandering Light Web Application

This directory contains comprehensive end-to-end (E2E) tests for the Wandering Light web application using Playwright for Python.

## Overview

The E2E test suite covers:

- **Web UI Testing**: User interactions, drag-and-drop functionality, form submissions
- **API Testing**: Backend REST API endpoints and data validation
- **WebSocket Testing**: Real-time communication and graph execution
- **Integration Testing**: Full user workflows from UI to backend
- **Error Handling**: Network errors, validation errors, and edge cases
- **Cross-browser Testing**: Chromium, Firefox, and Safari support

## Test Structure

```
tests/e2e/
├── __init__.py              # Package initialization
├── conftest.py              # Pytest fixtures and configuration
├── test_webapp_e2e.py       # Main UI interaction tests
├── test_api_e2e.py          # Backend API tests
├── test_websocket_e2e.py    # WebSocket functionality tests
├── test_helpers.py          # Utility functions and helpers
├── setup_e2e.py             # Setup script for test environment
├── pytest.ini              # Pytest configuration for E2E tests
└── README.md                # This documentation file
```

## Setup

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm

### Quick Setup

Run the setup script to install all dependencies and configure the environment:

```bash
python tests/e2e/setup_e2e.py
```

### Manual Setup

1. **Install Python dependencies**:
   ```bash
   pip install -e .
   ```

2. **Install Playwright browsers**:
   ```bash
   playwright install
   ```

3. **Install frontend dependencies**:
   ```bash
   cd web_ui/frontend
   npm install
   cd ../..
   ```

## Running Tests

### Run All E2E Tests

```bash
# Run all E2E tests (headless)
pytest tests/e2e/ -v

# Run with browser visible
pytest tests/e2e/ -v --headed

# Run with slow motion for debugging
pytest tests/e2e/ -v --headed --slowmo=1000
```

### Run Specific Test Files

```bash
# UI interaction tests
pytest tests/e2e/test_webapp_e2e.py -v

# API tests
pytest tests/e2e/test_api_e2e.py -v

# WebSocket tests
pytest tests/e2e/test_websocket_e2e.py -v
```

### Run Specific Test Categories

```bash
# Run only UI tests
pytest tests/e2e/ -v -m ui

# Run only API tests  
pytest tests/e2e/ -v -m api

# Run only WebSocket tests
pytest tests/e2e/ -v -m websocket

# Run slow tests
pytest tests/e2e/ -v -m slow
```

### Cross-Browser Testing

```bash
# Test with different browsers
pytest tests/e2e/ --browser=chromium
pytest tests/e2e/ --browser=firefox
pytest tests/e2e/ --browser=webkit
```

### Debugging Options

```bash
# Run with screenshots on failure
pytest tests/e2e/ -v --screenshot=only-on-failure

# Run with video recording
pytest tests/e2e/ -v --video=retain-on-failure

# Run with debug mode
pytest tests/e2e/ -v --headed --slowmo=2000 --screenshot=on
```

## Test Coverage

### UI Tests (`test_webapp_e2e.py`)

- ✅ Page loading and basic layout
- ✅ Sidebar component visibility and functionality
- ✅ Function creation through UI forms
- ✅ TypedList configuration
- ✅ Drag and drop functionality
- ✅ Graph creation with multiple nodes
- ✅ Execute graph button functionality
- ✅ Error handling and validation
- ✅ Responsive design testing
- ✅ Accessibility basics
- ✅ Keyboard navigation

### API Tests (`test_api_e2e.py`)

- ✅ REST API endpoints testing
- ✅ Function CRUD operations
- ✅ Data validation and error handling
- ✅ CORS headers verification
- ✅ Function execution with different data types
- ✅ Metadata preservation
- ✅ Duplicate function handling
- ✅ Large payload handling

### WebSocket Tests (`test_websocket_e2e.py`)

- ✅ WebSocket connection establishment
- ✅ Real-time graph execution
- ✅ Error handling over WebSocket
- ✅ Connection status indicators
- ✅ Reconnection on disconnect
- ✅ Large payload handling
- ✅ Concurrent operations
- ✅ Timeout handling
- ✅ Partial results streaming

## Test Helpers

The `test_helpers.py` file provides utility functions:

- `E2ETestHelpers.create_test_function()` - Create functions via API
- `E2ETestHelpers.create_function_via_ui()` - Create functions via UI
- `E2ETestHelpers.drag_node_to_canvas()` - Drag and drop operations
- `E2ETestHelpers.execute_graph()` - Execute graphs
- `E2ETestHelpers.wait_for_toast_message()` - Wait for notifications
- `E2ETestHelpers.take_screenshot()` - Debug screenshots
- `E2ETestHelpers.mock_websocket_response()` - Mock WebSocket responses

## Configuration

### Pytest Configuration

The E2E tests use a custom pytest configuration in `pytest.ini`:

- Default browser: Chromium
- Headless mode by default
- Screenshots on failure
- Video recording on failure
- 5-minute timeout per test
- Detailed logging

### Environment Variables

You can customize test behavior with environment variables:

```bash
# Set browser
export BROWSER=firefox

# Set headless mode
export HEADLESS=false

# Set slow motion delay
export SLOWMO=1000

# Set custom ports
export BACKEND_PORT=8001
export FRONTEND_PORT=3001
```

## Troubleshooting

### Common Issues

1. **Tests fail with "connection refused"**
   - Make sure both backend and frontend servers start properly
   - Check that ports 8000 (backend) and 3000 (frontend) are available
   - Verify firewall settings

2. **Drag and drop tests fail**
   - This is often due to timing issues
   - Try increasing wait times in test helpers
   - Run with `--slowmo=1000` to debug

3. **WebSocket tests are flaky**
   - WebSocket tests are inherently more complex
   - They often require mocking for reliability
   - Consider running WebSocket tests separately

4. **Browser doesn't launch**
   - Run `playwright install` to install browser binaries
   - Check if the browser executable is in PATH
   - Try running with `--headed` to see browser window

### Debug Mode

For debugging failing tests:

```bash
# Run single test with debug options
pytest tests/e2e/test_webapp_e2e.py::TestWebAppE2E::test_page_loads_successfully -v --headed --slowmo=2000 --screenshot=on
```

### Logs and Artifacts

Test artifacts are saved to:
- Screenshots: `test_outputs/screenshots/`
- Videos: `test_outputs/videos/`
- Logs: `test_outputs/logs/`

## CI/CD Integration

For continuous integration, use headless mode:

```yaml
# GitHub Actions example
- name: Run E2E Tests
  run: |
    python tests/e2e/setup_e2e.py
    pytest tests/e2e/ -v --browser=chromium --headed=false
```

## Contributing

When adding new E2E tests:

1. Use appropriate test markers (`@pytest.mark.ui`, `@pytest.mark.api`, etc.)
2. Follow the existing naming conventions
3. Use the helper functions from `test_helpers.py`
4. Add comprehensive docstrings
5. Test both happy path and error scenarios
6. Consider cross-browser compatibility

## Performance Considerations

- E2E tests are slower than unit tests
- Parallel execution can help but may cause port conflicts
- Consider using `pytest-xdist` for parallel execution
- Mock external dependencies when possible
- Use selective test execution during development

## Maintenance

- Update browser versions regularly with `playwright install`
- Keep frontend dependencies up to date
- Review and update selectors if UI changes
- Monitor test flakiness and add retries if needed
- Update test data when API changes 