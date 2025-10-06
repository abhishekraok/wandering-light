# Wandering Light Graph Editor - Frontend

A React-based frontend for the Wandering Light Graph Editor built with TypeScript, Chakra UI, and ReactFlow.

## Prerequisites

- Node.js 18.17.0 or higher (see `.nvmrc`)
- npm or yarn package manager

## Installation

1. Install dependencies:
```bash
npm install
```

2. If using nvm (recommended):
```bash
nvm use
```

## Available Scripts

### Development
- `npm start` - Runs the app in development mode on http://localhost:3000
- `npm run build` - Builds the app for production

### Testing
- `npm test` - Runs tests in interactive watch mode
- `npm run test:watch` - Runs tests with watch mode enabled
- `npm run test:coverage` - Runs tests and generates coverage report
- `npm run test:ci` - Runs tests in CI mode (non-interactive, with timeout)

## Testing Setup

This project uses:
- **Jest** for test framework
- **React Testing Library** for component testing
- **@testing-library/user-event** for user interaction simulation
- **@testing-library/jest-dom** for custom Jest matchers

### Test Files
Tests are located alongside their corresponding components with `.test.tsx` extension:
- `src/App.test.tsx` - Main App component tests
- `src/components/GraphEditor.test.tsx` - Graph editor functionality
- `src/components/Sidebar.test.tsx` - Sidebar interactions and forms
- `src/components/nodes/FunctionNode.test.tsx` - Function node rendering
- `src/components/nodes/TypedListNode.test.tsx` - TypedList node functionality

### Running Tests on Different Machines

To ensure tests run consistently across different environments:

#### Method 1: Automated Setup (Recommended)
```bash
# Run the automated setup script
./install-and-test.sh
```

#### Method 2: Manual Setup
1. **Use the specified Node.js version**:
```bash
nvm install 18.17.0
nvm use 18.17.0
```

2. **Install exact dependencies**:
```bash
npm ci  # Uses package-lock.json for exact versions
```

3. **Run tests in CI mode**:
```bash
npm run test:ci
```

## Dependencies

### Production Dependencies
- React 18.2.0
- TypeScript 4.9.5
- Chakra UI 2.8.2
- ReactFlow 11.7.4
- Axios 1.4.0

### Development Dependencies
- Testing Library suite (Jest DOM, React, User Event)
- TypeScript type definitions
- Ark UI React (for advanced Chakra UI components)

## Test Coverage

The project is configured with coverage thresholds:
- Branches: 50%
- Functions: 50%
- Lines: 50%
- Statements: 50%

Run `npm run test:coverage` to see detailed coverage reports.

## Architecture

This frontend uses:
- **Component-based architecture** with React functional components
- **TypeScript** for type safety
- **Chakra UI** for consistent design system
- **ReactFlow** for graph visualization and interaction
- **Axios** for API communication
- **WebSocket** for real-time graph execution

## Troubleshooting

### Common Issues

1. **Module resolution errors**: Ensure you're using Node.js 18.17.0
2. **Dependency conflicts**: Delete `node_modules` and `package-lock.json`, then run `npm install`
3. **Test failures**: Run `npm run test:ci` to avoid watch mode issues
4. **Chakra UI import errors**: Ensure all Chakra UI dependencies are installed correctly

### Environment Variables

The frontend expects the backend API to be running on `http://localhost:8000`. This can be configured in the GraphEditor component if needed. 