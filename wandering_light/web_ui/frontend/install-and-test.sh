#!/bin/bash

# Installation and Testing Script for Wandering Light Frontend
# This script ensures the frontend tests run properly on different machines

set -e  # Exit on any error

echo "🚀 Setting up Wandering Light Frontend..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18.17.0 or higher."
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node --version | cut -d'v' -f2)
echo "📦 Node.js version: $NODE_VERSION"

# Check if nvm is available and use specified version
if command -v nvm &> /dev/null && [ -f .nvmrc ]; then
    echo "🔧 Using Node.js version from .nvmrc"
    nvm use
fi

# Install dependencies
echo "📥 Installing dependencies..."
if [ -f package-lock.json ]; then
    echo "   Using npm ci for exact dependency versions..."
    npm ci
else
    echo "   Using npm install..."
    npm install
fi

echo "✅ Dependencies installed successfully!"

# Run tests
echo "🧪 Running tests..."
npm run test:ci

echo "📊 Test Summary:"
echo "   - All tests completed"
echo "   - Note: React act() warnings are non-critical and indicate async state updates"
echo "   - Core functionality tests are passing"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Available commands:"
echo "  npm start          - Start development server"
echo "  npm test           - Run tests in watch mode"
echo "  npm run test:ci    - Run tests once (CI mode)"
echo "  npm run test:coverage - Run tests with coverage report"
echo "  npm run build      - Build for production" 