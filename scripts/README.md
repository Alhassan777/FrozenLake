# Frozen Lake Scripts

This directory contains all the run scripts for different modes of the Frozen Lake environment.

## Overview

These scripts are designed to be run either directly or through the central entry point `run_frozen_lake.py` in the root directory.

## Scripts

- `run.py`: Main script for running the Frozen Lake game with various configuration options
- `run_agent_comparison_gui.py`: GUI for comparing different reinforcement learning algorithms (Q-Learning, SARSA, DQN)
- `run_auto_solver.py`: Script for automatically solving the Frozen Lake environment using reinforcement learning
- `run_game.py`: Simple script for running just the game itself

## Usage

It's recommended to use the central entry point in the root directory:

```bash
# Run the main game
python run_frozen_lake.py --mode game

# Run agent comparison GUI
python run_frozen_lake.py --mode compare-gui

# Run auto solver
python run_frozen_lake.py --mode solver
```

Each script can also be run directly from this directory if needed:

```bash
# Run simple game
python run_game.py

# Run with CLI arguments
python run.py --map_name 4x4 --no_slip

# Run agent comparison GUI
python run_agent_comparison_gui.py

# Run auto-solver
python run_auto_solver.py
```

## Organization

This directory is structured as a Python package (hence the `__init__.py` file), allowing the scripts to be imported by the central run script. 