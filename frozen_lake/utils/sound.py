"""
Sound utilities for Frozen Lake GUI
"""

import os
import pygame
import logging

def initialize_sounds(sound_dir='assets'):
    """Initialize sound effects
    
    Args:
        sound_dir: Directory containing sound files
        
    Returns:
        Dictionary of sound effects
    """
    sounds = {}
    logging.info(f"Initializing sounds from directory: {sound_dir}")
    
    try:
        # Move sound
        move_path = os.path.join(sound_dir, 'move.wav')
        if os.path.exists(move_path):
            move_sound = pygame.mixer.Sound(move_path)
            sounds['move'] = move_sound
            logging.info(f"Loaded move sound: {move_path}")
        else:
            logging.warning(f"Move sound file not found: {move_path}")
    except Exception as e:
        logging.error(f"Error loading move sound: {str(e)}")
        
    try:
        # Fall/Slip sound
        slip_path = os.path.join(sound_dir, 'slip.wav')
        if os.path.exists(slip_path):
            slip_sound = pygame.mixer.Sound(slip_path)
            sounds['fall'] = slip_sound  # Keep 'fall' as the key for backward compatibility
            logging.info(f"Loaded slip sound: {slip_path}")
        else:
            logging.warning(f"Slip sound file not found: {slip_path}")
    except Exception as e:
        logging.error(f"Error loading slip sound: {str(e)}")
        
    try:
        # Win sound
        win_path = os.path.join(sound_dir, 'win.wav')
        if os.path.exists(win_path):
            win_sound = pygame.mixer.Sound(win_path)
            sounds['win'] = win_sound
            # Set volume higher to ensure it's more noticeable
            win_sound.set_volume(0.8)
            logging.info(f"Loaded win sound: {win_path}")
        else:
            logging.warning(f"Win sound file not found: {win_path}")
    except Exception as e:
        logging.error(f"Error loading win sound: {str(e)}")
        
    return sounds 