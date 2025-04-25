#!/usr/bin/env python3
"""
Run a Frozen Lake game with a custom map
"""

from frozen_lake.game import FrozenLakeGame

def run_custom_map(custom_map, is_slippery=True, sound_dir='assets'):
    """
    Run a Frozen Lake game with a custom map
    
    Args:
        custom_map: List of strings representing the map
        is_slippery: Whether the ice is slippery
        sound_dir: Directory containing sound files
    """
    app = FrozenLakeGame(is_slippery=is_slippery, custom_map=custom_map, sound_dir=sound_dir)
    app.run()

if __name__ == "__main__":
    # Example custom map
    # S: Start position
    # F: Frozen surface (safe)
    # H: Hole (dangerous)
    # G: Goal position
    custom_map = [
        "SFFFF",
        "FHFHF",
        "FFFHF",
        "HFFFH",
        "HFFFG"
    ]
    
    # Run the game with the custom map
    run_custom_map(custom_map, is_slippery=True) 