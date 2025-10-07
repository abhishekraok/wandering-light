# Wandering Light

A Python library for learning function compositions from input-output examples.

## Motivation
Current LLMs do not know what they know and do not know. They lack a world model. See Sutton's [age of experience](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf). 
Instead of next token prediction we would like a training approach that lets models take action and learn from its outcome. 
We also want a task that is infinitely scalable for learning, bound only by computation.
Programming by Example (PBE) provides such an environment, where the model is tasked to find a series of functions that transforms a given inputs to outputs. 

### Capabilities 
Given any task the model should be able to classify it into one of these 3 possibilities. 

  1. Confidently say it can solve this
  2. Confidently say it cannot solve this
  3. Unsure

This is knowing what it knows and does not know. 
This lets it keep learning boundlessly in the future by exploring the 3 class. 
A value function in the RL paradigm should help with this.

### The tasks 
Let us assume each state represents a list of values (e.g. the integers [1,2,3]).
Using the naming convention of [Absolute Zero Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner)

*Induction*
We would like to train a solver model that can give us the shortest path between two states ([1,2,3] -> [3,5,7]) 
A path is defined as a DAG consisting of pure functions (e.g. double, plus1). 
Currently the functions take only one input and output only a single output.

*Propose*
Propose a new task for the solver, that is not too easy and not too hard. 


## Features

 - Scripts for SFT and RL on Induction and Propose tasks using the TRL library.
 - Wandb integration for monitoring and analyzing the metrics.
 - Evaluation scripts, website to visualize the evaluation metrics.
 - A web interface to visualize and interact with the various functions and data.
 - Clean code: 300+ unit tests, CI using Github actions.


## Installation
This project requires python 3.12 or later.
```bash
git clone https://github.com/abhishekraok/wandering-light.git
cd wandering-light
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Testing

Run the full test suite with:

```bash
pytest
```
# Code

## Concepts 

- Immutable functions defined in the class `FunctionDef`.
- Represent planned sequences of functions without execution using `TrajectorySpec`.
- Triplet of inputs, outputs and function sequences defined in the class `Trajectory`.
- Execute trajectories and evaluate results with `Executor` class.
- Several solvers included:
  - **RandomSolve**: tries random function sequences within a budget.
  - **BFSSolve**: performs breadth-first search up to a maximum depth.
  - **LLM solvers**: use OpenAI, Gemini, Ollama or local models to propose functions.

## Project Structure

```
.
├── wandering_light/        # Main package
│   ├── __init__.py         # Package initialization
│   ├── executor.py         # Executes individual functions and trajectories
│   ├── function_def.py     # Defines the FunctionDef class
│   ├── common_functions.py # Built-in helper functions
│   ├── trajectory.py       # Defines TrajectorySpec and Trajectory classes
│   ├── solver.py           # Implements search-based solvers
│   ├── synthesize.py       # Utilities for synthesizing new functions
│   ├── typed_list.py       # Typed list container
│   ├── llm_utils.py        # LLM integration utilities
│   ├── constants.py        # Project constants
│   ├── evals/              # Evaluation scripts and workflows
│   ├── training/           # Training scripts (SFT, RL)
│   └── web_ui/             # Web interface
├── tests/                  # pytest test cases for core functionality
├── pyproject.toml          # Project configuration and dependencies
├── pytest.ini              # pytest configuration
└── README.md               # This file
```

## Usage

### Solving for an input output pair
You can search for a trajectory that maps a specific input to an output using one of the built-in solvers.
```python
from wandering_light.solver import get_solver_by_name

solver = get_solver_by_name("bfs", budget=3)
trajectory = solver.solve(input_list=[1, 2, 3], output_list=[3, 5, 7])
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
Quick check, without online evaluation
```bash
python wandering_light/training/sft.py --no-eval
```

To train on the full dataset, use
```bash
python wandering_light/training/sft.py --full-run 
```
Suppose you store the output dir to SFT_OUTPUT_DIR, evaluate to ensure you have a decent success rate (between 30-70% ideally).
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
To see the results of all the past evaluations run
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