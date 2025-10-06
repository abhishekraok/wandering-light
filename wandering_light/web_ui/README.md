# Wandering Light Graph Editor

A web-based visual programming interface for creating and executing graphs of FunctionDef and TypedList nodes.

## Overview

This application allows you to:
- Create TypedList nodes with various data types and values
- Create and register FunctionDef nodes
- Connect nodes to form computation graphs
- Execute the graphs and see the results in real-time

## Project Structure

```
web_ui/
├── backend/               # FastAPI backend
│   ├── main.py           # Main API endpoints
│   ├── graph_executor.py # Graph execution logic
│   └── requirements.txt  # Python dependencies
├── frontend/             # React frontend
│   ├── src/              # Source code
│   │   ├── components/   # React components
│   │   │   ├── nodes/    # Custom node components
│   │   │   ├── GraphEditor.tsx # Main graph editor
│   │   │   └── Sidebar.tsx     # Side panel with node palette
│   │   ├── App.tsx       # Main app component
│   │   ├── index.tsx     # Entry point
│   │   └── types.ts      # TypeScript definitions
│   └── package.json      # Node.js dependencies
└── README.md             # This file
```

## Setup & Installation

### Backend

1. Create a Python virtual environment (if not already created):
   ```bash
   cd /home/abhishekrao/repos/wandering_light
   python -m venv .venv
   ```

2. Install the backend dependencies:
   ```bash
   cd web_ui/backend
   ../.venv/bin/pip install -r requirements.txt
   ```

3. Run the backend server:
   ```bash
   ../.venv/bin/python main.py
   ```
   The API will be available at http://localhost:8000

### Frontend

1. Install the frontend dependencies:
   ```bash
   cd /home/abhishekrao/repos/wandering_light/web_ui/frontend
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```
   The UI will be available at http://localhost:3000
   
### Troubleshooting

- **Backend connection errors**: Make sure the backend server is running on port 8000 before starting the frontend
- **Missing module errors**: If you see errors about missing modules, make sure you've installed all dependencies with `npm install` in the frontend directory
- **TypeScript errors**: If you encounter TypeScript errors, try running `npm install` again to ensure all type definitions are properly installed

## Usage

1. **Create Nodes**: 
   - Drag TypedList nodes from the palette onto the canvas
   - Configure the TypedList with your desired data type and values
   - Create custom FunctionDef nodes or use existing ones from the registry

2. **Connect Nodes**:
   - Connect a TypedList output handle to a FunctionDef input handle
   - Make sure the types are compatible

3. **Execute Graph**:
   - Click the "Execute Graph" button to run the computations
   - View the results displayed on each node

## Example

To create a simple graph that increments each value in a list:

1. Drag a TypedList node onto the canvas
2. Set its type to "builtins.int" and values to [1, 2, 3]
3. Create a function named "increment" with:
   - Input type: builtins.int
   - Output type: builtins.int
   - Code: return x + 1
4. Drag the function onto the canvas
5. Connect the TypedList output to the function input
6. Click "Execute Graph"
7. The function node should show the result: [2, 3, 4]

## Testing

To run tests:

```bash
cd /home/abhishekrao/repos/wandering_light
pytest
```
