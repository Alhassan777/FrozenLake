"""
Rendering utilities for Frozen Lake GUI
"""

import pygame
import numpy as np

def initialize_images(cell_size, colors):
    """Generate tile images for the Frozen Lake grid
    
    Args:
        cell_size: Size of each cell in pixels
        colors: Dictionary of colors for rendering
        
    Returns:
        Dictionary of images for each tile type
    """
    # Generate or load tile images
    images = {}
    
    # Start tile (S)
    start_img = pygame.Surface((cell_size, cell_size))
    start_img.fill(colors['light_blue'])
    # Draw frozen lake pattern
    for i in range(10):
        x1, y1 = np.random.randint(0, cell_size), np.random.randint(0, cell_size)
        x2, y2 = np.random.randint(0, cell_size), np.random.randint(0, cell_size)
        pygame.draw.line(start_img, (200, 230, 255), (x1, y1), (x2, y2), 2)
    # Add "Start" text
    font = pygame.font.SysFont('Arial', 16)
    text = font.render('START', True, colors['black'])
    text_rect = text.get_rect(center=(cell_size/2, cell_size/2))
    start_img.blit(text, text_rect)
    images['S'] = start_img
    
    # Frozen tile (F)
    frozen_img = pygame.Surface((cell_size, cell_size))
    frozen_img.fill((220, 240, 255))
    # Add ice pattern
    for i in range(15):
        x1, y1 = np.random.randint(0, cell_size), np.random.randint(0, cell_size)
        x2, y2 = np.random.randint(0, cell_size), np.random.randint(0, cell_size)
        pygame.draw.line(frozen_img, (255, 255, 255), (x1, y1), (x2, y2), 2)
    # Add shine effect
    pygame.draw.ellipse(frozen_img, (255, 255, 255, 128), 
                       (cell_size//4, cell_size//4, 
                        cell_size//2, cell_size//2), 0)
    images['F'] = frozen_img
    
    # Hole tile (H)
    hole_img = pygame.Surface((cell_size, cell_size))
    # Create ice background first
    hole_img.fill((220, 240, 255))
    for i in range(5):
        x1, y1 = np.random.randint(0, cell_size), np.random.randint(0, cell_size)
        x2, y2 = np.random.randint(0, cell_size), np.random.randint(0, cell_size)
        pygame.draw.line(hole_img, (255, 255, 255), (x1, y1), (x2, y2), 2)
    
    # Draw hole
    pygame.draw.circle(hole_img, (10, 20, 40), 
                      (cell_size//2, cell_size//2), 
                      cell_size//3)
    # Add crack lines around hole
    for i in range(8):
        angle = i * (np.pi / 4)
        length = np.random.randint(cell_size//5, cell_size//2)
        x1 = cell_size//2 + int(np.cos(angle) * cell_size//3)
        y1 = cell_size//2 + int(np.sin(angle) * cell_size//3)
        x2 = cell_size//2 + int(np.cos(angle) * (cell_size//3 + length))
        y2 = cell_size//2 + int(np.sin(angle) * (cell_size//3 + length))
        pygame.draw.line(hole_img, (50, 50, 70), (x1, y1), (x2, y2), 2)
    
    images['H'] = hole_img
    
    # Goal tile (G)
    goal_img = pygame.Surface((cell_size, cell_size))
    goal_img.fill(colors['green'])
    # Add some texture
    for i in range(10):
        x1, y1 = np.random.randint(0, cell_size), np.random.randint(0, cell_size)
        pygame.draw.circle(goal_img, (100, 255, 100), (x1, y1), 5)
    
    # Draw flag or treasure
    pygame.draw.rect(goal_img, (255, 215, 0), 
                    (cell_size//2 - 15, cell_size//2 - 20, 
                     30, 20))
    pygame.draw.rect(goal_img, (128, 0, 0), 
                    (cell_size//2 - 20, cell_size//2, 
                     40, 15))
    
    # Add "GOAL" text
    font = pygame.font.SysFont('Arial', 16)
    text = font.render('GOAL', True, colors['black'])
    text_rect = text.get_rect(center=(cell_size/2, cell_size/2 + 25))
    goal_img.blit(text, text_rect)
    
    images['G'] = goal_img
    
    return images

def initialize_player(cell_size, colors):
    """Create player sprites for each direction
    
    Args:
        cell_size: Size of each cell in pixels
        colors: Dictionary of colors for rendering
        
    Returns:
        List of player images for each direction
    """
    # Create player sprites facing different directions
    player_imgs = []
    
    # Right-facing player (default)
    right_img = pygame.Surface((cell_size//2, cell_size//2), pygame.SRCALPHA)
    # Draw body
    pygame.draw.circle(right_img, colors['player'], 
                     (cell_size//4, cell_size//4), 
                     cell_size//8)
    # Draw eyes
    pygame.draw.circle(right_img, colors['white'], 
                     (cell_size//4 + 5, cell_size//4 - 2), 
                     cell_size//16)
    player_imgs.append(right_img)
    
    # Down-facing player
    down_img = pygame.Surface((cell_size//2, cell_size//2), pygame.SRCALPHA)
    pygame.draw.circle(down_img, colors['player'], 
                     (cell_size//4, cell_size//4), 
                     cell_size//8)
    # Draw eyes
    pygame.draw.circle(down_img, colors['white'], 
                     (cell_size//4 - 3, cell_size//4 + 3), 
                     cell_size//16)
    pygame.draw.circle(down_img, colors['white'], 
                     (cell_size//4 + 3, cell_size//4 + 3), 
                     cell_size//16)
    player_imgs.append(down_img)
    
    # Left-facing player
    left_img = pygame.Surface((cell_size//2, cell_size//2), pygame.SRCALPHA)
    pygame.draw.circle(left_img, colors['player'], 
                     (cell_size//4, cell_size//4), 
                     cell_size//8)
    # Draw eyes
    pygame.draw.circle(left_img, colors['white'], 
                     (cell_size//4 - 5, cell_size//4 - 2), 
                     cell_size//16)
    player_imgs.append(left_img)
    
    # Up-facing player
    up_img = pygame.Surface((cell_size//2, cell_size//2), pygame.SRCALPHA)
    pygame.draw.circle(up_img, colors['player'], 
                     (cell_size//4, cell_size//4), 
                     cell_size//8)
    # Draw eyes
    pygame.draw.circle(up_img, colors['white'], 
                     (cell_size//4 - 3, cell_size//4 - 3), 
                     cell_size//16)
    pygame.draw.circle(up_img, colors['white'], 
                     (cell_size//4 + 3, cell_size//4 - 3), 
                     cell_size//16)
    player_imgs.append(up_img)
    
    return player_imgs 