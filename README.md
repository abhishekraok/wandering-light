# Wandering Light

[![Python Package](https://github.com/abhishekraok/wandering-light/actions/workflows/python-package.yml/badge.svg)](https://github.com/abhishekraok/wandering-light/actions/workflows/python-package.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A Python library for tool call mastery through self play. 

Given a set of functions (tool calls), we generate input output examples from these and then train an LLM 
to predict the correct list of functions that can map the inputs to the outputs (AKA Induction task). 
The library also supports training the LLM for generating appropriately challenging input outputs (AKA proposal task). 


## Features

 - Scripts for SFT and RL on Induction and Proposal tasks using the TRL library.
 - Synthetic data generation using LLMs.
 - Wandb integration for monitoring and analyzing the metrics.
 - Evaluation scripts, website to visualize the evaluation metrics.
 - A web interface to visualize and interact with the data samples.
 - Clean code: 300+ unit tests, CI using Github actions.
 - Can train small models (0.1B) locally within a few hours.

## Motivation
Currently LLMs are trained to imitate text on the internet. 
As a result they do not know what they know and do not know, which causes hallucination. 
They lack a world model (See Sutton's [age of experience](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf)). 
Instead of next token prediction we would like a training approach that lets models take action and learn from its outcome. 
We call this self supervised tool-use learning (SSTL).
We can consider function calling or tool use as taking an action.
We also want a task that is infinitely scalable for learning, bound only by computation. 
Programming by Example (PBE) provides such an environment, where the model is tasked to find a series of functions that transforms a given inputs to outputs. 

We would like to develop some **meta cognition capabilities** in the model.
Given any task the model should be able to classify it into one of these 3 possibilities. 

  1. Confidently say it can solve this
  2. Confidently say it cannot solve this
  3. Unsure

This lets it keep learning boundlessly in the future by exploring the unsure tasks. 
A value function in the RL paradigm should help with this.

### The tasks 
Let us assume each state represents a list of values (e.g. the integers [1,2,3]).
Using the naming convention of [Absolute Zero Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner)

**Induction**
We would like to train a solver model that can give us the shortest path between two states ([1,2,3] -> [3,5,7]) 
A path is defined as a DAG consisting of pure functions (e.g. double, plus1). 
Currently the functions take only one input and output only a single output for simplicity.
In the future we would like to expand to multiple argument functions to make this library more practical.

**Propose**
Propose a new task for the solver, that is not too easy and not too hard. 


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

- `FunctionDef`: Class for immutable functions e.g. double.
- `TrajectorySpec`: Represents planned sequences of functions without execution e.g. [double, plus1].
- `Trajectory`: Class representing triplet of inputs, outputs and function sequences e.g. [(1,2,3), (3,5,7), (double, plus1)].
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
python wandering_light/evals/run_evaluation.py --budget=1 --eval_file=wandering_light/evals/data/random_inputs_500.py  --solver_names=[trained_local] --budget 1 --model-name abhishekraok/induction-basicfns-opt125m-longsft
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
 python wandering_light/evals/evaluate_proposer.py --model abhishekraok/proposer-basicfns-opt125m-sft2k --solver-model abhishekraok/induction-basicfns-opt125m-longsft
```
which should output
```python
EvalResult(parse_rate=0.96, avg_function_count=2.02, avg_function_count_ratio=1.38, solver_success_rate=0.15, num_samples=100, frac_non_zero_std=0.31)
```