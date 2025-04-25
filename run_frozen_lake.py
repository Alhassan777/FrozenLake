#!/usr/bin/env python3
# File: run_frozen_lake.py
# Directory: root

"""
Frozen Lake - Central Run Script

This script serves as the main entry point for all Frozen Lake functionality.
It provides a clean interface to run different modules:

1. Game Interface - Play the Frozen Lake game yourself
2. Game Solver - Auto-solve using Q-Learning, SARSA, or DQN algorithms
3. Training Comparison GUI - Compare the performance of different RL algorithms with a graphical interface

Usage:
    python run_frozen_lake.py --mode [game|solver|compare-gui]
    
Example Commands:
    # Play the game yourself
    python run_frozen_lake.py --mode game
    
    # Auto-solve using Q-Learning on a 4x4 grid (only 4x4 and 8x8 maps are supported)
    python run_frozen_lake.py --mode solver --algorithm q_learning --map_size 4x4
    
    # Auto-solve using SARSA on a 8x8 grid with non-slippery movement
    python run_frozen_lake.py --mode solver --algorithm sarsa --map_size 8x8 --no_slip
    
    # Compare training with GUI
    python run_frozen_lake.py --mode compare-gui
"""
import argparse
import sys
import os
import logging
import importlib

# Add scripts directory to Python path
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.insert(0, script_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to parse arguments and run the appropriate module
    """
    parser = argparse.ArgumentParser(description='Frozen Lake Central Runner')
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['game', 'solver', 'compare-gui'],
                        help='Which mode to run')
    
    # For solver mode
    parser.add_argument('--algorithm', type=str, choices=['q_learning', 'sarsa', 'dqn'],
                       help='Algorithm for auto-solver (for solver mode)')
    parser.add_argument('--map_size', type=str, choices=['4x4', '8x8'],
                       help='Map size (for game or solver modes)')
    parser.add_argument('--no_slip', action='store_true',
                       help='Disable slippery ice (for game or solver modes)')
    
    args = parser.parse_args()
    
    if args.mode == 'game':
        # Run the game interface for user play
        try:
            from frozen_lake.game_controller import GameController
            logger.info("Starting Frozen Lake game interface for user play")
            
            # Create controller
            controller = GameController()
            
            # Configure settings if provided
            if args.map_size:
                controller.menu.settings['map_name'] = args.map_size
            if args.no_slip is not None:
                controller.menu.settings['is_slippery'] = not args.no_slip
                
            # For game mode, disable RL explicitly
            controller.menu.settings['enable_rl'] = False
            
            # Show game menu and run
            controller.menu._show_game_menu()
            controller.run()
            return 0
        except ImportError:
            logger.error("Could not import the game controller module")
            return 1
            
    elif args.mode == 'solver':
        # Run the game solver with various algorithm options
        try:
            from frozen_lake.game_controller import GameController
            logger.info("Starting Frozen Lake auto-solver")
            
            # Create controller
            controller = GameController()
            
            # Configure settings
            if args.algorithm:
                controller.menu.settings['algorithm'] = args.algorithm
            if args.map_size:
                controller.menu.settings['map_name'] = args.map_size
            if args.no_slip is not None:
                controller.menu.settings['is_slippery'] = not args.no_slip
            
            # For solver mode, enable RL explicitly
            controller.menu.settings['enable_rl'] = True
            
            # Show auto-solver menu and run
            controller.menu._show_auto_solver_menu()
            controller.run()
            return 0
        except ImportError as e:
            logger.error(f"Could not import the auto-solver module: {e}")
            return 1
            
    elif args.mode == 'compare-gui':
        # Run the algorithm comparison with GUI
        try:
            from scripts import run_agent_comparison_gui
            logger.info("Starting Frozen Lake agent comparison GUI")
            gui = run_agent_comparison_gui.AgentComparisonGUI()
            gui.run()
            return 0
        except ImportError:
            logger.error("Could not import the agent comparison GUI module")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 