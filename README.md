# Wandering Light

A Python library for discovering sequences of functions (trajectories) that transform input lists into desired output lists. 
The functions are pure (no side effects). The goal is to learn function usage, and also later synthesize new functions.
This follows the programming by example (PBE) paradigm.

## Features

- Define functions with `FunctionDef`.
- Represent planned sequences of functions without execution using `TrajectorySpec`.
- Execute trajectories and evaluate results with `Executor`.
- Several solvers included:
  - **RandomSolve**: tries random function sequences within a budget.
  - **BFSSolve**: performs breadth-first search up to a maximum depth.
  - **LLM solvers**: use OpenAI, Gemini, Ollama or local models to propose functions.
- Utilities for synthesizing functions and evaluating solvers.

## Installation

```bash
git clone https://github.com/abhishekraok/wandering_light.git
cd wandering_light
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

## Testing

Run the full test suite with:

```bash
pytest
```

## Project Structure

```
.
├── __init__.py         # Package initialization
├── executor.py         # Executes individual functions and trajectories
├── function_def.py     # Defines the FunctionDef class
├── common_functions.py # Built-in helper functions
├── trajectory.py       # Defines TrajectorySpec and Trajectory classes
├── solver.py           # Implements search-based solvers
├── synthesize.py       # Utilities for synthesizing new functions
├── typed_list.py       # Typed list container
├── evals/              # Evaluation scripts and workflows
├── tests/              # pytest test cases for core functionality
├── web_ui/
└── pyproject.toml      # Project configuration
```

## Usage

### Solving for an input output pair
You can search for a trajectory that maps a specific input to an output using one of the built-in solvers.
```python
from solver import get_solver_by_name

solver = get_solver_by_name("bfs", budget=3)
trajectory = solver.solve(input_list=<input_list>, output_list=<output_list>)
print("Found trajectory:", trajectory)
```

## Evaluation
Create an evaluation file 
```bash
python wandering_light/evals/create_data.py --save
```

Run using an LLM API
```bash
python wandering_light/evals/run_evaluation.py --eval_file=evals/data/eval_data_v20250831_160239.py --solver_names=["gemini"] --num_samples 100 --budget 1
```
See `solver.py` for additional solver names (openai, ollama, local models, etc.).


## Training SFT
Quick check, wihtout online evaluation
```bash
python wandering_light/training/sft.py --no-eval
```

Full dataset with 
```bash
python wandering_light/training/sft.py --full-run 
```
Suppose you store the output dir to SFT_OUTPUT_DIR, evalaute to ensure you have a decent success rate (between 30-70% ideally).
You can use the evaluation command below.

Next you can do RL.

## Training RL
```bash
python wandering_light/training/rl_grpo.py --batch-size 32 --model-name $SFT_OUTPUT_DIR --full-run --wandb-run-name $NAME
```

### Evaluate local trained LLM 
Assuming you have a checkpoint shown below.
```bash
python wandering_light/evals/run_evaluation.py --budget=1 --eval_file=wandering_light/evals/data/random_inputs_500.py  --solver_names=[trained_local] --budget 1 --model-name checkpoints/saved/rl/long_sft_opt_125m_s35k_no_len/
```

### Evaluation Dashboard
To see the results of all the past evalautions run
```bash
streamlit run wandering_light/evals/dashboard.py
```

## Proposer
The data generator.
First finetune it using SFT, using the `--task proposer` flag. Then evaluate it.

### Evaluate proposer
```bash
python -m wandering_light.evals.evaluate_proposer --model checkpoints/saved/sft/proposer_opt_125m_2k --solver-model checkpoints/saved/rl/long_sft_opt_125m_s35k_no_len/ 
```