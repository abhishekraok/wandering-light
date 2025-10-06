#!/usr/bin/env python3
"""Setup script for E2E tests."""

import subprocess
import sys
from pathlib import Path


def check_requirements():
    """Check that all required tools are installed."""
    requirements = {
        "node": "Node.js is required for the frontend",
        "npm": "npm is required for frontend dependencies",
        "python": "Python is required for the backend",
    }

    missing = []
    for cmd, description in requirements.items():
        try:
            subprocess.run([cmd, "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(f"{cmd}: {description}")

    if missing:
        print("Missing requirements:")
        for req in missing:
            print(f"  - {req}")
        return False

    return True


def setup_frontend():
    """Set up the frontend dependencies."""
    frontend_dir = Path(__file__).parent.parent.parent / "web_ui" / "frontend"

    if not frontend_dir.exists():
        print(f"Frontend directory not found: {frontend_dir}")
        return False

    print("Installing frontend dependencies...")
    try:
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
        print("✓ Frontend dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install frontend dependencies: {e}")
        return False


def setup_backend():
    """Set up the backend dependencies."""
    root_dir = Path(__file__).parent.parent.parent

    print("Installing backend dependencies...")
    try:
        # Install the project in development mode
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=root_dir,
            check=True,
        )
        print("✓ Backend dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install backend dependencies: {e}")
        return False


def install_playwright():
    """Install Playwright browsers."""
    print("Installing Playwright browsers...")
    try:
        subprocess.run([sys.executable, "-m", "playwright", "install"], check=True)
        print("✓ Playwright browsers installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install Playwright browsers: {e}")
        return False


def create_test_outputs_dir():
    """Create directory for test outputs like screenshots."""
    test_outputs_dir = Path(__file__).parent.parent.parent / "test_outputs"
    test_outputs_dir.mkdir(exist_ok=True)
    print("✓ Test outputs directory created")


def verify_setup():
    """Verify that the setup is working."""
    print("\nVerifying setup...")

    # Check that we can import playwright
    try:
        import importlib.util

        playwright_spec = importlib.util.find_spec("playwright.sync_api")
        if playwright_spec is None:
            raise ImportError("Playwright not found")

        print("✓ Playwright import successful")
    except ImportError as e:
        print(f"✗ Playwright import failed: {e}")
        return False

    # Check that we can import fastapi
    try:
        import importlib.util

        fastapi_spec = importlib.util.find_spec("fastapi")
        if fastapi_spec is None:
            raise ImportError("FastAPI not found")

        print("✓ FastAPI import successful")
    except ImportError as e:
        print(f"✗ FastAPI import failed: {e}")
        return False

    # Check frontend package.json exists
    frontend_package = (
        Path(__file__).parent.parent.parent / "web_ui" / "frontend" / "package.json"
    )
    if frontend_package.exists():
        print("✓ Frontend package.json found")
    else:
        print("✗ Frontend package.json not found")
        return False

    return True


def main():
    """Main setup function."""
    print("Setting up E2E testing environment...")
    print("=" * 50)

    if not check_requirements():
        print("\nPlease install the missing requirements and try again.")
        sys.exit(1)

    success = True

    success &= setup_backend()
    success &= setup_frontend()
    success &= install_playwright()
    create_test_outputs_dir()
    success &= verify_setup()

    print("\n" + "=" * 50)
    if success:
        print("✓ E2E setup completed successfully!")
        print("\nYou can now run E2E tests with:")
        print("  pytest tests/e2e/ -v")
        print("\nOr run specific test files:")
        print("  pytest tests/e2e/test_webapp_e2e.py -v")
        print("  pytest tests/e2e/test_api_e2e.py -v")
        print("  pytest tests/e2e/test_websocket_e2e.py -v")

        print("\nTo run tests with browser visible (non-headless):")
        print("  pytest tests/e2e/ -v --headed")

        print("\nTo run tests with slow motion for debugging:")
        print("  pytest tests/e2e/ -v --headed --slowmo=1000")

    else:
        print("✗ E2E setup failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
