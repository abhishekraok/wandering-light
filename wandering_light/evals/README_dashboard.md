# 📊 Evaluation Dashboard

A Streamlit-based dashboard for viewing and analyzing evaluation results across multiple runs.

## 🚀 Quick Start

1. **Install dependencies** (if not already done):
   ```bash
   pip install -e .
   ```

2. **Run some evaluations** (if you haven't already):
   ```bash
   python -m wandering_light.evals.run_evaluation path/to/eval_file.py --solver_names random bfs
   ```

3. **Launch the dashboard**:
   ```bash
   python wandering_light/evals/run_dashboard.py
   ```
   
   Or directly with Streamlit:
   ```bash
   streamlit run wandering_light/evals/dashboard.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 📈 Features

### Overview Metrics
- **Total Runs**: Number of evaluation runs matching current filters
- **Average Success Rate**: Mean success rate across selected runs  
- **Average Solution Length**: Mean solution length across successful runs
- **Total Samples**: Total number of samples evaluated

### Interactive Filters
- **Solver**: Filter by solver type (random, bfs, etc.)
- **Eval File**: Filter by evaluation dataset
- **Model Name**: Filter by model name extracted from command
- **Budget**: Filter by solver budget
- **Date Range**: Filter by evaluation run date

### Visualizations
- **Success Rate Over Time**: Line chart showing how success rates change over time
- **Success Rate by Solver**: Bar chart comparing average success rates by solver type
- **Solution Length Distribution**: Histogram of average solution lengths

### Results Table
- **Sortable columns**: Click headers to sort by any metric
- **Row selection**: Click on any row to see detailed run information
- **Comprehensive metrics**: All key evaluation metrics in one view

### Detailed Run View
When you click on a table row, see:
- Run configuration (solver, eval file, model name, budget, samples)
- Performance metrics (success rate, failures, solution length)
- Command used to generate the run
- Link to sample-by-sample viewer

## 🔄 Data Refresh

The dashboard automatically caches loaded data for performance. Click **"🔄 Refresh Data"** to reload after new evaluation runs.

## 📁 Data Structure

The dashboard searches for evaluation results in the root `results/` directory:

```
results/                              # Root level results directory
├── 20240101_120000/                  # Timestamped run directory
│   ├── summary.json                  # Run summary with metrics
│   ├── RandomSolve_*.json            # Detailed results per solver
│   └── llm_input_output/             # LLM interaction logs
├── 20240101_130000/
│   └── ...
└── saved/                            # Saved evaluation runs (nested)
    ├── ri_500/
    │   └── LEAD_sft_6k_rl_90k_opt125m/
    │       ├── summary.json
    │       └── ...
    └── ...
```

## 💡 Tips

- **Compare solvers**: Use filters to compare different solvers on the same eval file
- **Track progress**: Use the time-series chart to see improvement over time
- **Find outliers**: Sort by success rate or solution length to identify unusual runs
- **Drill down**: Click table rows for run details, then use `eval_viewer.html` for sample-level analysis
- **Model comparison**: Use the Model Name filter to compare different model versions

## 🔗 Related Tools

- **wandering_light.evals.run_evaluation**: Generate evaluation runs
- **wandering_light/evals/eval_viewer.html**: View individual runs sample-by-sample
- **wandering_light.evals.evaluate_solver**: Core evaluation logic 