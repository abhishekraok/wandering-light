# Evaluation Results Viewer

A standalone HTML page for browsing evaluation results from `run_evaluation.py` sample by sample.

## Features

- 📁 **Directory Selection**: Load evaluation results from any directory
- 📊 **Summary Overview**: View key metrics like success rate and average solution length  
- 🧭 **Sample Navigation**: Browse samples with Previous/Next buttons or direct number input
- 💬 **LLM Input/Output**: View the exact prompts sent to and responses from the LLM
- 📋 **Detailed Results**: See evaluation details including predicted vs golden functions

## Usage

1. **Open the HTML file**: Simply open `eval_viewer.html` in any modern web browser
2. **Select Directory**: Click "Choose Directory" and select a results directory like:
   ```
   evals/results/saved/Funcset_facebook_opt_125m_s76k/
   ```
3. **Browse Results**: Use the navigation controls to explore samples

## Expected Directory Structure

The viewer expects a directory containing:

```
result_directory/
├── summary.json                              # Overall evaluation summary
├── {SolverName}_{timestamp}.json             # Detailed results with sample-by-sample data
└── llm_input_output/
    └── {solver_name}/
        ├── 0_input.txt                       # LLM prompt for sample 0
        ├── 0_output.txt                      # LLM response for sample 0
        ├── 1_input.txt                       # LLM prompt for sample 1
        ├── 1_output.txt                      # LLM response for sample 1
        └── ...
```

## Sample Data Structure

Each sample displays:

- **Input**: The input data being transformed
- **Expected Output**: What the correct output should be
- **Actual Output**: What the solver actually produced
- **Success**: Whether the solver succeeded (✅/❌)
- **Solution Length**: Number of functions in the solution
- **Predicted Functions**: Functions the LLM predicted (blue tags)
- **Golden Functions**: Correct functions (green tags)
- **Error**: Any error message if the sample failed

## Keyboard Navigation

- **Left Arrow**: Previous sample
- **Right Arrow**: Next sample

## Browser Compatibility

Works with modern browsers that support:
- File API with directory selection (`webkitdirectory`)
- ES6+ JavaScript features
- CSS Grid

Tested on Chrome, Firefox, and Safari.

## Example Usage

1. Run evaluation:
   ```bash
   python evals/run_evaluation.py --eval_file=evals/data/eval_data.py --solver_names=["trained_local"] --num_samples=100
   ```

2. Open `eval_viewer.html` in browser

3. Select the generated results directory (e.g., `evals/results/20250626_172205/`)

4. Browse through samples to analyze LLM performance and errors 