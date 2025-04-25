#!/usr/bin/env python3
# File: scripts/run_auto_solver.py
# Directory: scripts

"""
Auto-solver script for Frozen Lake

This script automatically solves the Frozen Lake environment using one of three
reinforcement learning algorithms: Q-Learning, SARSA, or DQN.

Usage:
    python run_auto_solver.py [--algorithm ALGORITHM] [--no_slip] [--map_size SIZE]

Where:
    ALGORITHM is one of: q_learning, sarsa, dqn
    SIZE is one of: 4x4, 8x8
"""
import os
import sys

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import argparse
import logging
from frozen_lake.game_controller import GameController
from frozen_lake.agents import QLearningAgent, SarsaAgent, DQNAgent
from frozen_lake.utils.logging_config import setup_logging

def main():
    """
    Main function to run the auto solver
    
    Returns:
        int: 0 for success, 1 for failure
    """
    try:
        # Set up logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Parse arguments
        parser = argparse.ArgumentParser(description='Frozen Lake Auto-Solver')
        parser.add_argument('--algorithm', type=str, default='q_learning', 
                            choices=['q_learning', 'sarsa', 'dqn'],
                            help='Algorithm to use: q_learning, sarsa, or dqn')
        parser.add_argument('--map_size', type=str, default='4x4', 
                            choices=['4x4', '8x8'],
                            help='Map size: 4x4 or 8x8 (only standard sizes supported by Gymnasium)')
        parser.add_argument('--no_slip', action='store_true', 
                            help='Disable slippery ice (makes movement deterministic)')
        parser.add_argument('--learning_rate', type=float, default=0.1,
                            help='Learning rate (alpha) for the agent')
        parser.add_argument('--discount_factor', type=float, default=0.99,
                            help='Discount factor (gamma) for the agent')
        parser.add_argument('--exploration_rate', type=float, default=1.0,
                            help='Initial exploration rate (epsilon) for the agent')
        parser.add_argument('--episodes', type=int, default=5000,
                            help='Number of episodes to train for')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='Batch size (for DQN only)')
        parser.add_argument('--update_target_every', type=int, default=100,
                            help='Update target network frequency (for DQN only)')
        
        args = parser.parse_args()
        
        # Log startup info
        logger.info(f"Starting Frozen Lake Auto-Solver with algorithm: {args.algorithm}")
        logger.info(f"Map size: {args.map_size}, Slippery: {not args.no_slip}")
        
        # Ensure necessary directories exist
        os.makedirs('assets', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Create game controller
        controller = GameController()
        
        # Configure basic settings
        controller.menu.settings['map_name'] = args.map_size
        controller.menu.settings['is_slippery'] = not args.no_slip
        controller.menu.settings['algorithm'] = args.algorithm
        
        # Configure algorithm-specific parameters
        if args.algorithm == 'q_learning':
            controller.menu.settings['learning_params']['q_learning']['learning_rate'] = args.learning_rate
            controller.menu.settings['learning_params']['q_learning']['discount_factor'] = args.discount_factor
            controller.menu.settings['learning_params']['q_learning']['exploration_rate'] = args.exploration_rate
            controller.menu.settings['learning_params']['q_learning']['training_episodes'] = args.episodes
        elif args.algorithm == 'sarsa':
            controller.menu.settings['learning_params']['sarsa']['learning_rate'] = args.learning_rate
            controller.menu.settings['learning_params']['sarsa']['discount_factor'] = args.discount_factor
            controller.menu.settings['learning_params']['sarsa']['exploration_rate'] = args.exploration_rate
            controller.menu.settings['learning_params']['sarsa']['training_episodes'] = args.episodes
        elif args.algorithm == 'dqn':
            controller.menu.settings['learning_params']['dqn']['learning_rate'] = args.learning_rate
            controller.menu.settings['learning_params']['dqn']['discount_factor'] = args.discount_factor
            controller.menu.settings['learning_params']['dqn']['exploration_rate'] = args.exploration_rate
            controller.menu.settings['learning_params']['dqn']['training_episodes'] = args.episodes
            controller.menu.settings['learning_params']['dqn']['batch_size'] = args.batch_size
            controller.menu.settings['learning_params']['dqn']['update_target_every'] = args.update_target_every
        
        # Start the auto-solver
        controller.start_auto_solver()
        
        return 0
        
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 