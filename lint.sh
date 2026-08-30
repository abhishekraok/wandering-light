#!/bin/bash

# Simple script to run ruff linting and formatting
# Usage: ./lint.sh [check|fix|format|all]

set -e

case "${1:-all}" in
    "check")
        echo "🔍 Running ruff check..."
        ruff check .
        ;;
    "ci")
        # Verify only; never rewrite files. This is what CI runs.
        echo "🔍 Running ruff check..."
        ruff check .
        echo "🎨 Verifying formatting..."
        ruff format --check .
        ;;
    "fix")
        echo "🔧 Running ruff check with fixes..."
        ruff check . --fix
        ;;
    "format")
        echo "🎨 Running ruff format..."
        ruff format .
        ;;
    "all")
        echo "🔍 Running ruff check..."
        ruff check .
        echo ""
        echo "🎨 Running ruff format..."
        ruff format .
        echo ""
        echo "✅ Linting and formatting complete!"
        ;;
    *)
        echo "Usage: $0 [check|fix|format|ci|all]"
        echo "  check  - Run linter checks only"
        echo "  ci     - Verify lint and formatting without rewriting (CI)"
        echo "  fix    - Run linter checks and auto-fix issues"
        echo "  format - Run formatter only"
        echo "  all    - Run both linter and formatter (default)"
        exit 1
        ;;
esac 
