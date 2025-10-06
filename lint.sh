#!/bin/bash

# Simple script to run ruff linting and formatting
# Usage: ./lint.sh [check|fix|format|all]

set -e

case "${1:-all}" in
    "check")
        echo "🔍 Running ruff check..."
        ruff check wandering_light/ tests/
        ;;
    "fix")
        echo "🔧 Running ruff check with fixes..."
        ruff check wandering_light/ tests/ --fix
        ;;
    "format")
        echo "🎨 Running ruff format..."
        ruff format wandering_light/ tests/
        ;;
    "all")
        echo "🔍 Running ruff check..."
        ruff check wandering_light/ tests/
        echo ""
        echo "🎨 Running ruff format..."
        ruff format wandering_light/ tests/
        echo ""
        echo "✅ Linting and formatting complete!"
        ;;
    *)
        echo "Usage: $0 [check|fix|format|all]"
        echo "  check  - Run linter checks only"
        echo "  fix    - Run linter checks and auto-fix issues"
        echo "  format - Run formatter only"
        echo "  all    - Run both linter and formatter (default)"
        exit 1
        ;;
esac 
