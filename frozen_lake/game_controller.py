"""
Game controller for Frozen Lake with integrated RL algorithm auto-solvers
"""
import os
import sys
import pygame
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive mode
import matplotlib.pyplot as plt
from pygame.locals import *

from frozen_lake.game import FrozenLakeGame
from frozen_lake.agents import QLearningAgent, SarsaAgent, DQNAgent, AgentTrainer
from frozen_lake.utils.map_generator import generate_random_map
from frozen_lake.utils.renderer import initialize_images, initialize_player
from frozen_lake.menu import MenuSystem

class GameController:
    """
    Controller for the Frozen Lake game with integrated RL algorithms
    """
    
    def __init__(self):
        """Initialize the game controller"""
        # Create the menu system
        self.menu = MenuSystem()
        
        # Game and agent instances
        self.game = None
        self.agent = None
        self.trainer = None
        
        # Create directories for saving models and assets
        self.model_dir = 'models'
        self.assets_dir = 'assets'
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.assets_dir, exist_ok=True)
        
        # Override menu methods
        self.menu._start_game = self.start_game
        self.menu._start_auto_solver = self.start_auto_solver
    
    def create_agent(self, algorithm, env_dims, learning_params):
        """
        Create an agent based on the selected algorithm
        
        Args:
            algorithm: The algorithm to use (q_learning, sarsa, dqn)
            env_dims: Environment dimensions (n_states, n_actions)
            learning_params: Dictionary of learning parameters
        
        Returns:
            An agent instance
        """
        n_states, n_actions = env_dims
        
        if algorithm == 'q_learning' or algorithm == 'Q-Learning':
            return QLearningAgent(
                state_space_size=n_states,
                action_space_size=n_actions,
                learning_rate=learning_params['q_learning']['learning_rate'],
                discount_factor=learning_params['q_learning']['discount_factor'],
                exploration_rate=learning_params['q_learning']['exploration_rate']
            )
        elif algorithm == 'sarsa' or algorithm == 'SARSA':
            return SarsaAgent(
                state_space_size=n_states,
                action_space_size=n_actions,
                learning_rate=learning_params['sarsa']['learning_rate'],
                discount_factor=learning_params['sarsa']['discount_factor'],
                exploration_rate=learning_params['sarsa']['exploration_rate']
            )
        elif algorithm == 'dqn' or algorithm == 'DQN':
            return DQNAgent(
                state_space_size=n_states,
                action_space_size=n_actions,
                learning_rate=learning_params['dqn']['learning_rate'],
                discount_factor=learning_params['dqn']['discount_factor'],
                exploration_rate=learning_params['dqn']['exploration_rate'],
                batch_size=learning_params['dqn']['batch_size'],
                update_target_every=learning_params['dqn']['update_target_every']
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def start_game(self):
        """Start the game with the current settings from the game menu"""
        try:
            # Get settings from menu
            settings = self.menu.settings
            
            # Start regular game with appropriate map size
            return self.start_regular_game(
                map_name=settings['map_name'],
                is_slippery=settings['is_slippery']
            )
        except Exception as e:
            import traceback
            print(f"Error in start_game: {e}")
            traceback.print_exc()
            return False
    
    def start_auto_solver(self):
        """Start the auto-solver with the current settings from the auto-solver menu"""
        try:
            # Get settings from menu
            settings = self.menu.settings
            
            # Get the algorithm (convert from display name to internal name if needed)
            algorithm = settings['algorithm']
            if algorithm == 'Q-Learning':
                algorithm = 'q_learning'
            elif algorithm == 'SARSA':
                algorithm = 'sarsa'
            elif algorithm == 'DQN':
                algorithm = 'dqn'
            
            # Get the training episodes based on the selected algorithm
            training_episodes = settings['learning_params'][algorithm]['training_episodes']
            
            # Start the game with the selected RL algorithm
            return self.start_game_with_rl(
                map_name=settings['map_name'],
                is_slippery=settings['is_slippery'],
                algorithm=algorithm,
                learning_params=settings['learning_params'],
                training_episodes=training_episodes,
                auto_solve=True
            )
        except Exception as e:
            import traceback
            print(f"Error in start_auto_solver: {e}")
            traceback.print_exc()
            return False
    
    def start_regular_game(self, map_name="4x4", is_slippery=True, custom_map=None):
        """
        Start a regular game without RL algorithms
        
        Args:
            map_name: Name of the map to use
            is_slippery: Whether the ice is slippery
            custom_map: Custom map to use
        
        Returns:
            True if the game returns to menu, False otherwise
        """
        self.game = EnhancedFrozenLakeGame(
            map_name=map_name,
            is_slippery=is_slippery,
            custom_map=custom_map,
            sound_dir=self.assets_dir
        )
        return self.game.run()
    
    def start_game_with_rl(self, map_name="4x4", is_slippery=True, custom_map=None,
                          algorithm='q_learning', learning_params=None, training_episodes=10000, 
                          auto_solve=False):
        """
        Start a game with a reinforcement learning algorithm
        
        Args:
            map_name: Name of the map to use
            is_slippery: Whether the ice is slippery
            custom_map: Custom map to use
            algorithm: The RL algorithm to use (q_learning, sarsa, dqn)
            learning_params: Dictionary of learning parameters
            training_episodes: Number of episodes to train for
            auto_solve: Whether to automatically solve the game
        
        Returns:
            True if the game returns to menu, False otherwise
        """
        self.game = EnhancedFrozenLakeGame(
            map_name=map_name,
            is_slippery=is_slippery,
            custom_map=custom_map,
            sound_dir=self.assets_dir,
            enable_rl=True,
            algorithm=algorithm,
            learning_params=learning_params,
            training_episodes=training_episodes,
            auto_solve=auto_solve
        )
        return self.game.run()
    
    def run(self):
        """Run the game controller"""
        # Run the menu loop
        running = True
        while running:
            # Display the menu and get settings
            settings = self.menu.run()
            
            # If the menu returns None, exit the game
            if settings is None:
                break
                
            # Start the appropriate game mode based on settings
            if settings.get('enable_rl', False):
                # Start auto-solver
                result = self.start_auto_solver()
            else:
                # Start regular game
                result = self.start_game()
                
            # If the game returns False, exit
            # If the game returns True, continue to show the menu
            if result is False:
                running = False
            # Do nothing for True (continue the loop)

        # Clean up
        pygame.quit()

class EnhancedFrozenLakeGame(FrozenLakeGame):
    """Enhanced version of FrozenLakeGame with RL algorithm integration"""
    
    def __init__(self, map_name="4x4", is_slippery=True, custom_map=None, sound_dir='assets',
                enable_rl=False, algorithm='q_learning', learning_params=None, training_episodes=20000, 
                model_path=None, auto_solve=False):
        """
        Initialize the enhanced Frozen Lake game
        
        Args:
            map_name: Name of the map to use
            is_slippery: Whether the ice is slippery
            custom_map: Custom map to use
            sound_dir: Directory containing sound files
            enable_rl: Whether to enable RL algorithms
            algorithm: The RL algorithm to use (q_learning, sarsa, dqn)
            learning_params: Dictionary of learning parameters
            training_episodes: Number of episodes to train for
            model_path: Path to a saved model to load
            auto_solve: Whether to automatically solve the game
        """
        print(f"Initializing EnhancedFrozenLakeGame with: map_name={map_name}, is_slippery={is_slippery}, enable_rl={enable_rl}, algorithm={algorithm}")
        
        # Store map parameters
        self.map_name = map_name
        self.is_slippery = is_slippery
        
        # Grid offset for centering
        self.grid_offset_x = 0
        self.grid_offset_y = 0
        
        # Initialize the base game
        super().__init__(map_name, is_slippery, custom_map, sound_dir)
        
        # RL settings
        self.enable_rl = enable_rl
        self.algorithm = algorithm
        self.learning_params = learning_params
        self.training_episodes = training_episodes
        self.auto_solve = auto_solve
        
        # RL components
        self.agent = None
        self.trainer = None
        
        # Trained flag
        self.trained = False
        
        # Auto-solve state
        self.auto_solving = False
        self.auto_step_delay = 300  # milliseconds
        self.last_auto_step = 0
        
        # Visualization flags
        self.show_q_values = False
        
        # Help overlay
        self.show_help = False
        
        # Add colors for Q-value visualization
        self.COLORS.update({
            'q_value_high': (0, 255, 0, 128),  # Green with alpha
            'q_value_medium': (255, 255, 0, 128),  # Yellow with alpha
            'q_value_low': (255, 0, 0, 128),  # Red with alpha
            'q_value_bg': (0, 0, 0, 64),  # Background for text
            'dqn_value_high': (0, 128, 255, 128),  # Blue with alpha (for DQN)
            'dqn_value_medium': (128, 0, 255, 128),  # Purple with alpha (for DQN)
            'dqn_value_low': (255, 0, 128, 128),  # Pink with alpha (for DQN)
        })
        
        # Add font for Q-values
        self.q_font = pygame.font.SysFont('Arial', 14)
        
        # Initialize RL if enabled
        if self.enable_rl:
            print(f"Initializing RL with algorithm: {algorithm}")
            self.initialize_rl(model_path)
            
        # Screen dimensions and scaling
        self.original_cell_size = self.CELL_SIZE
        self.resize_game(self.WINDOW_SIZE[0], self.WINDOW_SIZE[1])
        
        # Force a redraw to ensure the game displays properly
        self.draw_grid()
        self.draw_agent()
        pygame.display.flip()
        
        # Force a resize event to ensure proper sizing
        current_size = self.screen.get_size()
        pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, {'w': current_size[0], 'h': current_size[1], 'size': current_size}))
    
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
        
        # Update the q font size based on cell size
        q_font_size = max(10, int(self.CELL_SIZE / 8))
        self.q_font = pygame.font.SysFont('Arial', q_font_size)

    def initialize_rl(self, model_path=None):
        """
        Initialize RL components
        
        Args:
            model_path: Path to a saved model to load
        """
        # Create agent based on algorithm
        if self.algorithm == 'q_learning':
            self.agent = QLearningAgent(
                state_space_size=self.env.observation_space.n,
                action_space_size=self.env.action_space.n,
                learning_rate=self.learning_params['q_learning']['learning_rate'],
                discount_factor=self.learning_params['q_learning']['discount_factor'],
                exploration_rate=self.learning_params['q_learning']['exploration_rate']
            )
        elif self.algorithm == 'sarsa':
            self.agent = SarsaAgent(
                state_space_size=self.env.observation_space.n,
                action_space_size=self.env.action_space.n,
                learning_rate=self.learning_params['sarsa']['learning_rate'],
                discount_factor=self.learning_params['sarsa']['discount_factor'],
                exploration_rate=self.learning_params['sarsa']['exploration_rate']
            )
        elif self.algorithm == 'dqn':
            self.agent = DQNAgent(
                state_space_size=self.env.observation_space.n,
                action_space_size=self.env.action_space.n,
                learning_rate=self.learning_params['dqn']['learning_rate'],
                discount_factor=self.learning_params['dqn']['discount_factor'],
                exploration_rate=self.learning_params['dqn']['exploration_rate'],
                batch_size=self.learning_params['dqn']['batch_size'],
                update_target_every=self.learning_params['dqn']['update_target_every']
            )
            
        # Create trainer with the correct parameter order: agent first, then map_name and is_slippery
        print(f"Creating trainer with agent: {self.agent}, map_name: {self.map_name}, is_slippery: {self.is_slippery}")
        self.trainer = AgentTrainer(self.agent, map_name=self.map_name, is_slippery=self.is_slippery)
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            if self.algorithm == 'q_learning':
                self.agent.load_q_table(model_path)
            elif self.algorithm == 'sarsa':
                self.agent.load_q_table(model_path)
            elif self.algorithm == 'dqn':
                self.agent.load_model(model_path)
            self.trained = True
        elif self.auto_solve:
            # Train the agent if auto-solve is enabled
            self.train_agent()
    
    def train_agent(self):
        """Train the agent"""
        if not self.enable_rl or self.trained:
            return
            
        self.trained = True
        print(f"Training agent for {self.training_episodes} episodes...")
        
        try:
            # Show a message on the screen
            self._display_training_message("Training agent, please wait...")
            
            # Train with the specified number of episodes
            self.trainer.train(num_episodes=self.training_episodes, print_interval=1000)
            
            print("Training complete!")
            
            # Save the model
            model_dir = 'models'
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, f"{self.algorithm}_model.pkl")
            if self.algorithm == 'q_learning' or self.algorithm == 'sarsa':
                self.agent.save_q_table(model_path)
            elif self.algorithm == 'dqn':
                self.agent.save_model(model_path)
            print(f"Model saved to {model_path}")
            
            # Evaluate the agent
            success_rate = self.trainer.evaluate(num_episodes=100, render=False)
            print(f"Evaluation - Success Rate: {success_rate:.2f}%")
            
            # Create training plot
            self.training_plot = self.trainer.plot_training_results()
            
            # If auto-solve is enabled, start auto-solving
            if self.auto_solve:
                self.auto_solving = True
                self.agent.exploration_rate = 0  # Use exploitation only
                
        except Exception as e:
            print(f"Error during training: {e}")
        
        # Update screen to remove training message
        self.draw_grid()
        self.draw_agent()
        pygame.display.flip()
        
        self.trained = True
    
    def _display_training_message(self, message):
        """Display a message on the screen during training"""
        # First draw the game grid
        self.draw_grid()
        
        # Create a semi-transparent overlay
        overlay = pygame.Surface(self.WINDOW_SIZE, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))
        
        # Display the message
        font = pygame.font.SysFont('Arial', 24)
        text = font.render(message, True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.WINDOW_SIZE[0]//2, self.WINDOW_SIZE[1]//2))
        self.screen.blit(text, text_rect)
            
        # Update the display
        pygame.display.flip()
    
    def show_training_results(self):
        """Show the training results plot"""
        if not self.training_plot:
            return
            
        # Display the plot
        self.training_plot.show()
    
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
    
    def draw_q_values(self):
        """Draw the Q-values for each cell"""
        if not self.enable_rl or not self.show_q_values or not self.agent:
            return
            
        for i in range(self.nrow):
            for j in range(self.ncol):
                state = i * self.ncol + j
                q_values = self.agent.get_q_values(state)
                
                # Draw Q-value background
                q_bg_rect = pygame.Rect(j * self.CELL_SIZE + self.grid_offset_x, 
                                     i * self.CELL_SIZE + self.grid_offset_y, 
                                     self.CELL_SIZE, self.CELL_SIZE)
                q_bg_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                pygame.draw.rect(q_bg_surface, self.COLORS['q_value_bg'], 
                              q_bg_surface.get_rect())
                self.screen.blit(q_bg_surface, q_bg_rect)
                
                # Draw each Q-value as an arrow
                arrow_size = self.CELL_SIZE // 4
                
                # UP
                up_value = q_values[3]
                up_rect = pygame.Rect(j * self.CELL_SIZE + self.CELL_SIZE//2 - arrow_size//2 + self.grid_offset_x, 
                                    i * self.CELL_SIZE + 5 + self.grid_offset_y, 
                                    arrow_size, arrow_size)
                up_color = self._get_q_value_color(up_value)
                pygame.draw.polygon(self.screen, up_color, 
                                 [(up_rect.centerx, up_rect.top),
                                  (up_rect.right, up_rect.bottom),
                                  (up_rect.left, up_rect.bottom)])
                
                # RIGHT
                right_value = q_values[2]
                right_rect = pygame.Rect(j * self.CELL_SIZE + self.CELL_SIZE - arrow_size - 5 + self.grid_offset_x, 
                                       i * self.CELL_SIZE + self.CELL_SIZE//2 - arrow_size//2 + self.grid_offset_y, 
                                       arrow_size, arrow_size)
                right_color = self._get_q_value_color(right_value)
                pygame.draw.polygon(self.screen, right_color, 
                                 [(right_rect.right, right_rect.centery),
                                  (right_rect.left, right_rect.bottom),
                                  (right_rect.left, right_rect.top)])
                
                # DOWN
                down_value = q_values[1]
                down_rect = pygame.Rect(j * self.CELL_SIZE + self.CELL_SIZE//2 - arrow_size//2 + self.grid_offset_x, 
                                     i * self.CELL_SIZE + self.CELL_SIZE - arrow_size - 5 + self.grid_offset_y, 
                                     arrow_size, arrow_size)
                down_color = self._get_q_value_color(down_value)
                pygame.draw.polygon(self.screen, down_color, 
                                 [(down_rect.centerx, down_rect.bottom),
                                  (down_rect.left, down_rect.top),
                                  (down_rect.right, down_rect.top)])
                
                # LEFT
                left_value = q_values[0]
                left_rect = pygame.Rect(j * self.CELL_SIZE + 5 + self.grid_offset_x, 
                                     i * self.CELL_SIZE + self.CELL_SIZE//2 - arrow_size//2 + self.grid_offset_y, 
                                     arrow_size, arrow_size)
                left_color = self._get_q_value_color(left_value)
                pygame.draw.polygon(self.screen, left_color, 
                                 [(left_rect.left, left_rect.centery),
                                  (left_rect.right, left_rect.top),
                                  (left_rect.right, left_rect.bottom)])
    
    def _get_q_value_color(self, value):
        """Get color based on Q-value"""
        # Normalize the value (assuming most values will be between 0 and 1)
        # This can be adjusted based on your actual Q-value ranges
        if self.algorithm == 'dqn':
            if value > 0.6:
                return self.COLORS['dqn_value_high']
            elif value > 0.3:
                return self.COLORS['dqn_value_medium']
            else:
                return self.COLORS['dqn_value_low']
        else:  # q_learning or sarsa
            if value > 0.6:
                return self.COLORS['q_value_high']
            elif value > 0.3:
                return self.COLORS['q_value_medium']
            else:
                return self.COLORS['q_value_low']
    
    def auto_step(self):
        """Take a step using the trained agent"""
        if self.enable_rl and self.auto_solving and not self.game_over and not self.animating:
            current_time = pygame.time.get_ticks()
            if current_time - self.last_auto_step > self.auto_step_delay:
                state = self.state
                # Use exploitation only (no exploration)
                action = np.argmax(self.agent.get_q_values(state))
                self.take_action(action)
                self.last_auto_step = current_time
    
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
    
    def create_menu_button(self):
        """Create a menu button"""
        button_width = 150
        button_height = 30
        button_x = (self.WINDOW_SIZE[0] - button_width) // 2
        button_y = self.nrow * self.CELL_SIZE + self.grid_offset_y + 10
        
        return pygame.Rect(button_x, button_y, button_width, button_height)
    
    def draw_slippery_indicator(self):
        """Draw indicator showing if the ice is slippery"""
        # Draw indicator showing if the ice is slippery
        if self.is_slippery:
            text = self.font.render("Slippery: ON", True, (50, 50, 200))
        else:
            text = self.font.render("Slippery: OFF", True, (50, 50, 50))
            
        self.screen.blit(text, (10, self.nrow * self.CELL_SIZE + self.grid_offset_y + 5))
    
    def draw_extra_status(self):
        """Draw additional status information"""
        if not self.enable_rl:
            return
            
        # Draw auto-solving status
        if self.auto_solving:
            auto_text = self.font.render("Auto-Solving: ON", True, (0, 128, 0))
        else:
            auto_text = self.font.render("Auto-Solving: OFF", True, (128, 0, 0))
        self.screen.blit(auto_text, (self.WINDOW_SIZE[0] - 150, self.nrow * self.CELL_SIZE + self.grid_offset_y + 5))
        
        # Draw Q-value status
        if self.show_q_values:
            q_text = self.font.render("Q-Values: ON", True, (0, 128, 0))
        else:
            q_text = self.font.render("Q-Values: OFF", True, (128, 0, 0))
        self.screen.blit(q_text, (self.WINDOW_SIZE[0] - 150, self.nrow * self.CELL_SIZE + self.grid_offset_y + 25))
    
        # Draw algorithm name
        alg_text = self.font.render(f"Algorithm: {self.algorithm.upper()}", True, (0, 0, 128))
        self.screen.blit(alg_text, (10, self.nrow * self.CELL_SIZE + self.grid_offset_y + 25))
    
    def draw_help_overlay(self):
        """Draw help information overlay"""
        if not self.show_help:
            return
            
        # Create a semi-transparent overlay
        overlay = pygame.Surface(self.WINDOW_SIZE, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 192))  # Black with alpha
        self.screen.blit(overlay, (0, 0))
        
        # Header
        font_title = pygame.font.SysFont('Arial', 24, bold=True)
        title_text = font_title.render("Frozen Lake - Help & Controls", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(self.WINDOW_SIZE[0]//2, 50))
        self.screen.blit(title_text, title_rect)
        
        # Game controls
        font = pygame.font.SysFont('Arial', 18)
        commands = [
            ("Arrow Keys", "Move the player"),
            ("R", "Reset the game"),
            ("Q", "Quit the game"),
            ("H", "Toggle this help screen"),
            ("Space", "Pause/Unpause auto-solving")
        ]
        
        if self.enable_rl:
            rl_commands = [
                ("A", "Toggle auto-solving"),
                ("V", "Toggle Q-value display"),
                ("T", "Train the agent"),
                ("P", "Show training results"),
                ("+/-", "Speed up/slow down auto-play")
            ]
            commands.extend(rl_commands)
        
        y = 100
        for key, desc in commands:
            key_text = font.render(key, True, (255, 255, 0))
            desc_text = font.render(desc, True, (255, 255, 255))
            
            self.screen.blit(key_text, (self.WINDOW_SIZE[0]//2 - 150, y))
            self.screen.blit(desc_text, (self.WINDOW_SIZE[0]//2 - 30, y))
            
            y += 30
        
        # Instructions
        if y < self.WINDOW_SIZE[1] - 100:
            instructions = [
                "Goal: Reach the Goal (G) without falling into Holes (H)",
                "Slippery Ice: Your movement may not always go in the intended direction",
                "Press H again to close this help screen"
            ]
            
            y += 20
            for instruction in instructions:
                inst_text = font.render(instruction, True, (200, 200, 255))
                inst_rect = inst_text.get_rect(center=(self.WINDOW_SIZE[0]//2, y))
                self.screen.blit(inst_text, inst_rect)
                y += 30
    
    def run(self):
        """Run the game"""
        running = True
        clock = pygame.time.Clock()
        show_help = False
        
        # If auto-solve is enabled, train the agent and enable auto-solving
        if self.auto_solve and self.enable_rl:
            # Train the agent first
            self.train_agent()
            # Enable auto-solving and show Q-values
            self.auto_solving = True
            self.show_q_values = True
        
        # Force an initial resize to ensure proper display
        current_size = self.screen.get_size()
        self.resize_game(current_size[0], current_size[1])
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return
                
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize
                    self.resize_game(event.w, event.h)
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                        pygame.quit()
                        return
                    elif event.key == pygame.K_r:
                        # Reset the game
                        self.reset()
                    elif event.key == pygame.K_h:
                        # Toggle help overlay
                        self.show_help = not self.show_help
                    elif self.enable_rl and event.key == pygame.K_a:
                        # Toggle auto-solving
                        self.auto_solving = not self.auto_solving
                    elif self.enable_rl and event.key == pygame.K_v:
                        # Toggle Q-value display
                        self.show_q_values = not self.show_q_values
                    elif self.enable_rl and event.key == pygame.K_t:
                        # Train the agent
                        self.train_agent()
                    elif self.enable_rl and event.key == pygame.K_p:
                        # Show training results
                        if self.training_plot:
                            self.show_training_results()
                    elif self.enable_rl and (event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS):
                        # Speed up auto-solving
                        self.auto_step_delay = max(100, self.auto_step_delay - 100)
                    elif self.enable_rl and event.key == pygame.K_MINUS:
                        # Slow down auto-solving
                        self.auto_step_delay = min(2000, self.auto_step_delay + 100)
                    elif event.key == pygame.K_SPACE:
                        # Pause/Unpause auto-solving
                        if self.enable_rl:
                            self.auto_solving = not self.auto_solving
                    elif not self.game_over and not self.animating and (not self.enable_rl or not self.auto_solving):
                        if event.key == pygame.K_LEFT:
                            self.take_action(0)
                        elif event.key == pygame.K_DOWN:
                            self.take_action(1)
                        elif event.key == pygame.K_RIGHT:
                            self.take_action(2)
                        elif event.key == pygame.K_UP:
                            self.take_action(3)
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Check if menu button was clicked
                    if self.menu_button and self.menu_button.collidepoint(event.pos):
                        self.return_to_menu = True
                        running = False
            
            # Auto-step if auto-solving is enabled
            self.auto_step()
            
            # Draw game elements
            self.draw_grid()
            self.draw_q_values()
            self.draw_agent()
            self.draw_status()
            self.draw_slippery_indicator()
            self.draw_extra_status()
            
            # Draw help overlay
            self.draw_help_overlay()
            
            # Update the display
            pygame.display.flip()
            
            # Cap the frame rate
            clock.tick(60)
        
        # Return to menu if button was clicked
        if hasattr(self, 'return_to_menu') and self.return_to_menu:
            return True
            
        pygame.quit()
        sys.exit() 