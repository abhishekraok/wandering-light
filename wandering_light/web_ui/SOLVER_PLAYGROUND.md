# Solver Playground

The Solver Playground is a web interface for experimenting with the Wandering Light solver and understanding model behavior.

## Features

- **Interactive Problem Definition**: Define input and output TypedLists with their types and values
- **Model Configuration**: Specify the checkpoint path for the TrainedLLMTokenGenerator
- **Real-time Solver Prediction**: See predicted functions and output from the solver model

## Usage

1. **Start the backend server**:
   ```bash
   cd wandering_light/web_ui/backend
   python main.py
   ```

2. **Start the frontend**:
   ```bash
   cd wandering_light/web_ui/frontend
   npm install  # First time only
   npm start
   ```

3. **Navigate to Solver Playground**: Click on "Solver Playground" in the navigation bar

4. **Create a problem**:
   - Enter input type (e.g., `builtins.int`)
   - Enter input items as JSON array (e.g., `[1, 2, 3]`)
   - Enter expected output type and items
   - Configure model checkpoint path
   - Click "Run Solver"

## API Endpoints

- `POST /solver/execute`: Execute solver to predict functions that transform input to expected output

## Example

Input: `[1, 2, 3]` (int)
Expected Output: `[2, 4, 6]` (int)

The solver will:
1. Use the trained model to predict which functions would transform the input to the expected output
2. Show the predicted function sequence and resulting output