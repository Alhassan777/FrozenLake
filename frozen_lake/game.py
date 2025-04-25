"""
Main game class for Frozen Lake GUI
"""

import gymnasium as gym
import pygame
import numpy as np
import sys
import os
from pygame.locals import *

from frozen_lake.utils.renderer import initialize_images, initialize_player
from frozen_lake.utils.sound import initialize_sounds

class FrozenLakeGame:
    """
    Main game class for the Frozen Lake GUI application
    
    This class implements a graphical interface for the Frozen Lake environment
    from Gymnasium. The agent must navigate from the starting position to the goal
    without falling into holes.
    
    The environment can be set to have slippery or non-slippery ice:
    - Slippery (default): The agent's movement is stochastic. When the agent chooses a direction,
      it only has a 1/3 probability of moving in that direction. There's a 1/3 probability
      of moving perpendicular (either left or right) to the intended direction.
    - Non-slippery: The agent's movement is deterministic. It will always move in the
      chosen direction.
    """
    def __init__(self, map_name="4x4", is_slippery=True, custom_map=None, sound_dir='assets'):
        # Initialize the environment
        if custom_map is not None:
            self.env = gym.make('FrozenLake-v1', desc=custom_map, is_slippery=is_slippery, render_mode=None)
        else:
            self.env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery, render_mode=None)
        
        # Store the is_slippery parameter since it's not accessible directly from the environment
        self.is_slippery = is_slippery
        
        self.action_space = self.env.action_space
        
        # Get map description
        self.desc = self.env.unwrapped.desc.astype(str)
        self.nrow, self.ncol = self.desc.shape
        
        # Reset the environment
        self.state, self.info = self.env.reset()
        
        # Initialize pygame
        pygame.init()
        
        # Define colors
        self.COLORS = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'blue': (0, 0, 255),
            'light_blue': (135, 206, 250),
            'green': (0, 255, 0),
            'red': (255, 0, 0),
            'brown': (139, 69, 19),
            'player': (255, 128, 0),
            'background': (240, 248, 255),
            'text_shadow': (50, 50, 50),
            'button': (135, 206, 250),
            'button_hover': (100, 149, 237),
            'button_text': (0, 0, 0),
            'border': (100, 100, 100)
        }
        
        # Set window size and title
        self.CELL_SIZE = 100
        self.WINDOW_SIZE = (self.ncol * self.CELL_SIZE, self.nrow * self.CELL_SIZE + 50)
        self.screen = pygame.display.set_mode(self.WINDOW_SIZE, pygame.RESIZABLE)
        pygame.display.set_caption("Frozen Lake")
        
        # Offset for centering
        self.grid_offset_x = 0
        self.grid_offset_y = 0
        
        # Create assets directory if it doesn't exist
        os.makedirs(sound_dir, exist_ok=True)
        
        # Generate or load images
        self.cell_images = initialize_images(self.CELL_SIZE, self.COLORS)
        
        # Initialize player images
        self.player_imgs = initialize_player(self.CELL_SIZE, self.COLORS)
        self.player_direction = 0
        
        # Animation variables
        self.animation_frames = 8
        self.current_frame = 0
        self.animating = False
        self.last_pos = self._state_to_position(self.state)
        self.target_pos = self.last_pos
        
        # Load sound effects if pygame mixer is available
        self.sounds = {}
        if pygame.mixer:
            pygame.mixer.init()
            self.sounds = initialize_sounds(sound_dir)
        
        # Font for displaying information
        self.font = pygame.font.SysFont('Arial', 20)
        
        # Game status
        self.done = False
        self.game_over = False
        self.won = False
        
        # Menu button
        self.menu_button = None
        self.return_to_menu = False
        
        # Main game loop
        self.clock = pygame.time.Clock()
    
    def _state_to_position(self, state):
        """Convert state integer to row, col position
        
        Args:
            state: Integer state from environment
            
        Returns:
            Tuple of (row, col)
        """
        # Convert state to row, col
        row = state // self.ncol
        col = state % self.ncol
        return row, col
    
    def draw_grid(self):
        """Draw the grid of tiles"""
        # Draw the grid
        self.screen.fill(self.COLORS['background'])
        
        for i in range(self.nrow):
            for j in range(self.ncol):
                cell_type = self.desc[i][j]
                cell_img = self.cell_images[cell_type]
                self.screen.blit(cell_img, (j * self.CELL_SIZE + self.grid_offset_x, 
                                         i * self.CELL_SIZE + self.grid_offset_y))
                
                # Draw grid lines
                pygame.draw.rect(self.screen, (100, 100, 100), 
                                (j * self.CELL_SIZE + self.grid_offset_x, 
                                i * self.CELL_SIZE + self.grid_offset_y, 
                                self.CELL_SIZE, self.CELL_SIZE), 1)
    
    def draw_agent(self):
        """Draw the player agent"""
        if self.animating:
            # Animate player movement
            progress = self.current_frame / self.animation_frames
            row, col = self.last_pos
            target_row, target_col = self.target_pos
            
            # Interpolate position
            current_row = row + (target_row - row) * progress
            current_col = col + (target_col - col) * progress
            
            # Draw player at interpolated position
            player_x = current_col * self.CELL_SIZE + self.CELL_SIZE//4 + self.grid_offset_x
            player_y = current_row * self.CELL_SIZE + self.CELL_SIZE//4 + self.grid_offset_y
            
            # Apply a simple bounce effect
            bounce_offset = 0
            if progress < 0.5:
                bounce_offset = -5 * np.sin(progress * np.pi)
            else:
                bounce_offset = -5 * np.sin(progress * np.pi)
                
            self.screen.blit(self.player_imgs[self.player_direction], 
                           (player_x, player_y + bounce_offset))
            
            # Update animation
            self.current_frame += 1
            if self.current_frame >= self.animation_frames:
                self.animating = False
                self.last_pos = self.target_pos
        else:
            # Draw player at current position
            row, col = self._state_to_position(self.state)
            player_x = col * self.CELL_SIZE + self.CELL_SIZE//4 + self.grid_offset_x
            player_y = row * self.CELL_SIZE + self.CELL_SIZE//4 + self.grid_offset_y
            
            self.screen.blit(self.player_imgs[self.player_direction], 
                           (player_x, player_y))
    
    def create_menu_button(self):
        """Create a menu button"""
        button_width = 150
        button_height = 30
        button_x = (self.WINDOW_SIZE[0] - button_width) // 2
        button_y = self.nrow * self.CELL_SIZE + self.grid_offset_y + 10
        
        return pygame.Rect(button_x, button_y, button_width, button_height)
    
    def draw_menu_button(self):
        """Draw the menu button if the game is won"""
        if self.game_over and self.won and self.menu_button:
            mouse_pos = pygame.mouse.get_pos()
            button_hover = self.menu_button.collidepoint(mouse_pos)
            
            # Draw button
            color = self.COLORS['button_hover'] if button_hover else self.COLORS['button']
            pygame.draw.rect(self.screen, color, self.menu_button, border_radius=5)
            pygame.draw.rect(self.screen, self.COLORS['border'], self.menu_button, 2, border_radius=5)
            
            # Draw button text
            text_surf = self.font.render("Return to Menu", True, self.COLORS['button_text'])
            text_rect = text_surf.get_rect(center=self.menu_button.center)
            self.screen.blit(text_surf, text_rect)
    
    def draw_status(self):
        """Draw status bar at the bottom of the screen"""
        # Calculate grid height
        grid_height = self.nrow * self.CELL_SIZE
        
        # Draw status bar at the bottom
        status_rect = pygame.Rect(0, grid_height + self.grid_offset_y, self.WINDOW_SIZE[0], 50)
        pygame.draw.rect(self.screen, (240, 240, 240), status_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), status_rect, 1)
        
        if self.game_over:
            if self.won:
                if not self.menu_button:
                    self.menu_button = self.create_menu_button()
                status_text = "You reached the goal! Click button to return to menu"
                text_color = self.COLORS['green']
            else:
                status_text = "You fell in a hole! Press R to restart."
                text_color = self.COLORS['red']
        else:
            status_text = "Use arrow keys to move. R: Reset, Q: Quit"
            text_color = self.COLORS['black']
        
        # Draw text with shadow for better readability
        shadow_text = self.font.render(status_text, True, self.COLORS['text_shadow'])
        text = self.font.render(status_text, True, text_color)
        
        text_rect = text.get_rect(center=(self.WINDOW_SIZE[0]//2, grid_height + self.grid_offset_y + 25))
        shadow_rect = shadow_text.get_rect(center=(self.WINDOW_SIZE[0]//2 + 1, grid_height + self.grid_offset_y + 26))
        
        self.screen.blit(shadow_text, shadow_rect)
        self.screen.blit(text, text_rect)
        
        # Draw the menu button
        self.draw_menu_button()
    
    def draw_slippery_indicator(self):
        """Draw indicator showing if the ice is slippery"""
        # Draw indicator showing if the ice is slippery
        if self.is_slippery:
            text = self.font.render("Slippery: ON", True, (50, 50, 200))
        else:
            text = self.font.render("Slippery: OFF", True, (50, 50, 50))
            
        self.screen.blit(text, (10, self.nrow * self.CELL_SIZE + self.grid_offset_y + 5))
    
    def take_action(self, action):
        """Process a player action
        
        Args:
            action: Integer action to take (0-3)
        """
        if not self.game_over and not self.animating:
            # Set player direction based on action
            self.player_direction = action
            
            # Record current position before taking action
            self.last_pos = self._state_to_position(self.state)
            
            # Take the action
            self.state, reward, terminated, truncated, self.info = self.env.step(action)
            
            # Record new position for animation
            self.target_pos = self._state_to_position(self.state)
            
            # Start animation if position changed
            if self.last_pos != self.target_pos:
                self.animating = True
                self.current_frame = 0
                
                # Play move sound
                if 'move' in self.sounds:
                    self.sounds['move'].play()
            
            # Check if game is over
            self.done = terminated or truncated
            if self.done:
                self.game_over = True
                self.won = reward > 0
                
                # Play appropriate sound
                if self.won and 'win' in self.sounds:
                    self.sounds['win'].play()
                elif not self.won and 'fall' in self.sounds:
                    self.sounds['fall'].play()
    
    def reset(self):
        """Reset the game to initial state"""
        self.state, self.info = self.env.reset()
        self.game_over = False
        self.won = False
        self.done = False
        self.animating = False
        self.last_pos = self._state_to_position(self.state)
        self.target_pos = self.last_pos
        self.menu_button = None
        self.return_to_menu = False
    
    def run(self):
        """Main game loop"""
        running = True
        
        # Force an initial resize to ensure proper display
        current_size = self.screen.get_size()
        self.resize_game(current_size[0], current_size[1])
        
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                
                elif event.type == VIDEORESIZE:
                    # Handle window resize
                    self.resize_game(event.w, event.h)
                
                elif event.type == KEYDOWN:
                    if event.key == K_q:
                        running = False
                    elif event.key == K_r:
                        self.reset()
                    elif not self.game_over and not self.animating:
                        if event.key == K_LEFT:
                            self.take_action(0)
                        elif event.key == K_DOWN:
                            self.take_action(1)
                        elif event.key == K_RIGHT:
                            self.take_action(2)
                        elif event.key == K_UP:
                            self.take_action(3)
                
                elif event.type == MOUSEBUTTONDOWN:
                    # Check if menu button was clicked
                    if self.menu_button and self.menu_button.collidepoint(event.pos):
                        self.return_to_menu = True
                        running = False
            
            # Draw the game
            self.draw_grid()
            self.draw_agent()
            self.draw_status()
            self.draw_slippery_indicator()
            
            pygame.display.update()
            self.clock.tick(60)
        
        # Return to menu instead of quitting if button was clicked
        if self.return_to_menu:
            return True
            
        pygame.quit()
        sys.exit()

    def resize_game(self, width, height):
        """Resize the game to fit the current window size"""
        # Save the window dimensions
        self.WINDOW_SIZE = (width, height)
        
        # Calculate available space for the grid
        available_height = height - 50  # Account for status bar
        
        # Calculate the maximum cell size that fits within the window
        max_cell_width = width // self.ncol
        max_cell_height = available_height // self.nrow
        
        # Use the smaller dimension to maintain square cells
        self.CELL_SIZE = min(max_cell_width, max_cell_height)
        
        # Calculate total grid size
        grid_width = self.CELL_SIZE * self.ncol
        grid_height = self.CELL_SIZE * self.nrow
        
        # Center the grid horizontally if needed
        self.grid_offset_x = (width - grid_width) // 2
        self.grid_offset_y = 0  # Keep aligned to top
        
        # Regenerate images with the new cell size
        self.cell_images = initialize_images(self.CELL_SIZE, self.COLORS)
        self.player_imgs = initialize_player(self.CELL_SIZE, self.COLORS) 