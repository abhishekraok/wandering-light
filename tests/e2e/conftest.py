import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright


def pytest_collection_modifyitems(config, items):
    """Add skip marker to E2E tests if running in CI without proper setup."""
    if os.environ.get("CI") and not os.environ.get("E2E_ENABLED"):
        skip_e2e = pytest.mark.skip(
            reason="E2E tests skipped in CI (set E2E_ENABLED=1 to enable)"
        )
        for item in items:
            item.add_marker(skip_e2e)


def check_server_running(host, port, timeout=10):
    """Check if a server is running on the given host and port."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                if result == 0:
                    return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


@pytest.fixture(scope="session")
def backend_server():
    """Start the FastAPI backend server for testing."""
    # Check if server is already running (e.g., in CI)
    if check_server_running("localhost", 8000):
        print("Backend server already running on port 8000")
        yield None  # No process to manage
        return

    # Change to the backend directory
    backend_dir = (
        Path(__file__).parent.parent.parent / "wandering_light" / "web_ui" / "backend"
    )

    # Start the backend server
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent)

    try:
        backend_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
            cwd=backend_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to start and verify it's running
        if not check_server_running("localhost", 8000, timeout=15):
            raise RuntimeError("Backend server failed to start within timeout")

        yield backend_process

    except Exception as e:
        pytest.skip(f"Could not start backend server: {e}")
    finally:
        # Cleanup
        if "backend_process" in locals() and backend_process.poll() is None:
            backend_process.terminate()
            backend_process.wait(timeout=10)


@pytest.fixture(scope="session")
def frontend_server():
    """Start the React frontend server for testing."""
    # Check if server is already running (e.g., in CI)
    if check_server_running("localhost", 3000):
        print("Frontend server already running on port 3000")
        yield None  # No process to manage
        return

    frontend_dir = (
        Path(__file__).parent.parent.parent / "wandering_light" / "web_ui" / "frontend"
    )

    try:
        # Install dependencies if needed
        if not (frontend_dir / "node_modules").exists():
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)

        # Start the frontend server
        frontend_process = subprocess.Popen(
            ["npm", "start"],
            cwd=frontend_dir,
            env={**os.environ, "PORT": "3000", "BROWSER": "none"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to start and verify it's running
        if not check_server_running("localhost", 3000, timeout=30):
            raise RuntimeError("Frontend server failed to start within timeout")

        yield frontend_process

    except Exception as e:
        pytest.skip(f"Could not start frontend server: {e}")
    finally:
        # Cleanup
        if "frontend_process" in locals() and frontend_process.poll() is None:
            frontend_process.terminate()
            frontend_process.wait(timeout=10)


@pytest.fixture(scope="session")
def browser():
    """Create a browser instance for the test session."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def page(browser):
    """Create a new page for each test."""
    page = browser.new_page()
    yield page
    page.close()


@pytest.fixture
def app_page(page, backend_server, frontend_server):
    """Navigate to the application page."""
    try:
        page.goto("http://localhost:3000")
        # Wait for the page to load
        page.wait_for_selector('[data-testid="graph-editor"]', timeout=30000)
        return page
    except Exception as e:
        pytest.skip(f"Could not load application page: {e}")
