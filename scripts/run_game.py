#!/usr/bin/env python3
# File: scripts/run_game.py
# Directory: scripts

"""
Main entry point for the Frozen Lake game with Q-Learning

This script provides a user-friendly interface to start the Frozen Lake game 
with the menu system. It creates the necessary directories and handles
basic error conditions.

Usage:
    python run_game.py
"""
import os
import sys

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

def main():
    """
    Main entry point for the Frozen Lake GUI
    
    This function:
    1. Creates necessary directories for models and assets
    2. Initializes the game controller
    3. Runs the menu interface
    4. Handles any errors that might occur during startup
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('assets', exist_ok=True)
        
        # Import game controller
        from frozen_lake.game_controller import GameController
        
        # Create and run the game controller
        controller = GameController()
        
        # Explicitly trigger the game menu flow (not auto-solver)
        controller.menu._show_game_menu()
        controller.run()
        
        return 0
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure all required packages are installed.")
        print("Run: pip install -r requirements.txt")
        return 1
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 