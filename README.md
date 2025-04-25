# Frozen Lake GUI with Q-Learning

A graphical user interface for the Frozen Lake environment from Gymnasium, featuring Q-Learning based automatic solving capabilities. This implementation includes optimized parameters for solving both deterministic and stochastic (slippery) environments.


<img width="1470" alt="Screenshot 2025-04-25 at 4 56 59 PM" src="https://github.com/user-attachments/assets/eafd4143-ec11-4dff-9a12-f58014f1beea" />

## Project Structure

```
frozen_lake/
├── __init__.py                 # Package initialization
├── game.py                     # Main game class
├── game_controller.py          # Game controller with menu and Q-learning
├── menu.py                     # Menu system
├── custom_map.py               # Custom map functionality
└── agents/                     # Agent implementations
│   ├── __init__.py             # Package initialization
│   ├── base.py                 # Base agent class
│   ├── q_learning.py           # Q-Learning agent
│   ├── sarsa.py                # SARSA agent
│   ├── dqn.py                  # Deep Q-Network agent
│   └── trainer.py              # Agent trainer
└── utils/                      # Utility modules
    ├── __init__.py             # Package initialization
    ├── map_generator.py        # Map generation utilities
    ├── renderer.py             # Rendering utilities
    ├── sound.py                # Sound utilities
    └── agent_comparison.py     # Agent comparison utilities
```

## Running the Application

The simplest way to run the application is using the central entry point script `run_frozen_lake.py` in the root directory. This script provides access to all the functionality of the project.

```bash
# Run the main game
python run_frozen_lake.py --mode game

# Run agent comparison GUI
python run_frozen_lake.py --mode compare-gui

# Run auto solver
python run_frozen_lake.py --mode solver --algorithm q_learning
```

### Command Line Options

```bash
python run_frozen_lake.py --help
```

Available options for game mode:
- `--map_size`: Map size (4x4 or 8x8)
- `--no_slip`: Disable slippery ice (makes movement deterministic)

Additional options for solver mode:
- `--algorithm`: Choose the solving algorithm (q_learning, sarsa, dqn)

## Installation

### Option 1: Install from requirements

```bash
pip install -r requirements.txt
```

### Option 2: Install as a package

```bash
pip install -e .
```

This will install the package and create command-line entry points:
- `frozen-lake-gui`: Run the game interface
- `frozen-lake-auto-solver`: Run the auto-solver
- `frozen-lake-agent-comparison`: Run the agent comparison
- `frozen-lake-agent-comparison-gui`: Run the agent comparison GUI

## Game Mechanics

### Slippery vs. Non-Slippery Ice

The Frozen Lake environment has two modes of movement:

**Slippery Mode (Default):**
- In this mode, the ice is slippery, making movement stochastic
- When you choose an action (direction), there's only a 1/3 probability that the agent will move in that direction
- There's a 2/3 probability the agent will move perpendicular to the intended direction (either left or right of the intended direction)
- This adds a significant challenge as you can't perfectly control the agent's movement
- Example: If you press "right", the agent might move right, up, or down

**Non-Slippery Mode:**
- In this mode, the ice is not slippery, making movement deterministic
- When you choose an action, the agent will always move in the intended direction
- This makes the game much easier and allows for precise planning
- To play in non-slippery mode, use the `--no_slip` flag

The slippery nature of the environment mimics real-world ice physics and makes the game more challenging, requiring strategic planning that accounts for movement uncertainty.

## Reinforcement Learning Agents

This implementation includes multiple reinforcement learning agents that can learn to navigate the Frozen Lake environment autonomously:

### 1. Q-Learning Agent

The Q-Learning agent builds a Q-table that maps state-action pairs to expected rewards, allowing it to make intelligent decisions about which actions to take in each state.

- **Algorithm**: Off-policy temporal difference learning algorithm
- **Update Rule**: Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]

### 2. SARSA Agent

The SARSA (State-Action-Reward-State-Action) agent is an on-policy temporal difference learning algorithm.

- **Algorithm**: On-policy temporal difference learning algorithm
- **Update Rule**: Q(s,a) = Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
- **Key Difference**: Uses the actual next action (on-policy) instead of the maximum Q-value (off-policy)

### 3. Deep Q-Network (DQN) Agent

The DQN agent uses a neural network to approximate the Q-function, allowing it to generalize across states.

- **Algorithm**: Deep reinforcement learning algorithm
- **Key Features**: 
  - Neural network for Q-function approximation
  - Experience replay for stable learning
  - Target network for stable learning targets
  - One-hot encoding for discrete states

### Reinforcement Learning Parameters (Optimized)

- **Learning Rate (Alpha)**: 0.05 (Q-Learning/SARSA), 0.001 (DQN) - How quickly the agent learns from new experiences
- **Discount Factor (Gamma)**: 0.99 - How much the agent values future rewards compared to immediate rewards
- **Exploration Rate (Epsilon)**: 1.0 (initial) - Probability of taking a random action to explore the environment
- **Exploration Decay Rate**: 0.0005 - How quickly exploration decreases over time
- **Training Episodes**: 20000 (default) - Number of episodes to train the agent

These parameters have been optimized to work well with the stochastic (slippery) environment, which is more challenging for reinforcement learning. The agents use:
- Small random initialization of Q-values to break ties during early learning
- Random tie-breaking for actions with equal Q-values
- Gradual exploration decay to ensure sufficient exploration of the environment

### Auto-Solver Features

- **Training Mode**: Train the agent on the current map configuration
- **Auto-Play**: Let the trained agent automatically solve the puzzle
- **Q-Value Visualization**: See the agent's learned values for each action in each state
- **Training Progress Visualization**: Real-time progress bar and success rate during training
- **Performance Analysis**: View graphs of rewards, steps per episode, and success rates during training

## Agent Comparison

The Agent Comparison feature allows you to compare different reinforcement learning algorithms side-by-side, showing performance metrics in real-time.

### Comparison Features

- **Real-time Performance Metrics**: Watch as all agents learn simultaneously
- **Comparative Plots**:
  - Average reward per episode
  - Average steps per episode
  - Success rate (per 100 episodes)
  - Cumulative reward
- **Summary Statistics**:
  - Success rate
  - Average steps
  - Average reward
  - Training time
- **Side-by-Side Evaluation**: Test all agents on the same environment for direct comparison

### Running Agent Comparison

```bash
python run_frozen_lake.py --mode compare-gui
```

### Agent Comparison GUI

The graphical interface allows you to:
- Choose which agents to compare (Q-Learning, SARSA, DQN)
- Configure environment settings (map size, slippery/non-slippery)
- Set the number of training episodes
- Run the training in the background
- View comparative performance metrics in an easy-to-understand format

The GUI runs the training in the background and displays the results afterward, making it easier to compare different agent configurations without having to watch the training process.

## Requirements

- Python 3.6+
- Gymnasium
- Pygame
- NumPy
- Matplotlib
- tqdm
- PyTorch (for DQN agent)
- pandas
- seaborn
- IPython

## Custom Maps

You can create a custom map by running:

```bash
python -m frozen_lake.custom_map
```

Or by importing and calling the function in your own code:

```python
from frozen_lake.custom_map import run_custom_map

custom_map = [
    "SFFFF",
    "FHFHF",
    "FFFHF",
    "HFFFH",
    "HFFFG"
]

run_custom_map(custom_map, is_slippery=True)
```

## Controls

### Regular Game Controls
- Arrow keys: Move agent (Left, Right, Up, Down)
- R: Reset environment
- Q: Quit application
- H: Show/hide help overlay

## Performance Notes

- **Stochastic Environment**: In the slippery mode (default), the agent may take longer to learn a successful policy due to the randomness of the environment. The optimized parameters help overcome this challenge.
- **Algorithm Comparison**: Different algorithms perform differently depending on the environment:
  - Q-Learning: Often faster to converge, but may be less stable
  - SARSA: More stable learning, especially in stochastic environments
  - DQN: Higher memory usage but can generalize better to larger state spaces 
