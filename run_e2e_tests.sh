#!/bin/bash

# E2E Test Runner for Wandering Light
# This script sets up the environment and runs E2E tests locally

set -e

echo "🚀 Setting up E2E test environment..."

# Install E2E dependencies if not already installed
echo "📦 Installing E2E dependencies..."
uv sync --extra e2e

# Install Playwright browsers
echo "🌐 Installing Playwright browsers..."
uv run playwright install chromium

# Set environment variable to enable E2E tests
export E2E_ENABLED=1

echo "🧪 Running E2E tests..."
uv run pytest tests/e2e/ -v

echo "✅ E2E tests completed!" 