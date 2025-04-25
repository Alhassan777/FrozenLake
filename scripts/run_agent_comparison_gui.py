#!/usr/bin/env python3
# File: scripts/run_agent_comparison_gui.py
# Directory: scripts

"""
GUI for Agent Comparison on Frozen Lake

This script provides a graphical interface to compare different reinforcement
learning algorithms (Q-Learning, SARSA, DQN) on the Frozen Lake environment.
It runs the training in the background and displays the results afterward.

Usage:
    python run_agent_comparison_gui.py
"""
import os
import sys

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import numpy as np
import gymnasium as gym
import pygame
import threading
import queue
import matplotlib
# Use Agg backend to avoid thread issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time
from frozen_lake.agents import QLearningAgent, SarsaAgent, DQNAgent
from frozen_lake.utils.agent_comparison import AgentComparison

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_BLUE = (173, 216, 230)
BLUE = (0, 0, 255)
GREEN = (0, 128, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

class Button:
    """Simple button class for GUI"""
    def __init__(self, x, y, width, height, text, color, hover_color, text_color=BLACK, font_size=20):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font_size = font_size
        self.hovered = False
        
    def draw(self, screen):
        """Draw the button"""
        color = self.hover_color if self.hovered else self.color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)  # Border
        
        font = pygame.font.SysFont(None, self.font_size)
        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)
        
    def is_hovered(self, pos):
        """Check if mouse is hovering over the button"""
        self.hovered = self.rect.collidepoint(pos)
        return self.hovered
        
    def is_clicked(self, pos, event):
        """Check if button is clicked"""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(pos)
        return False

class Checkbox:
    """Simple checkbox class for GUI"""
    def __init__(self, x, y, size, text, font_size=20, checked=False):
        self.rect = pygame.Rect(x, y, size, size)
        self.text = text
        self.font_size = font_size
        self.checked = checked
        
    def draw(self, screen):
        """Draw the checkbox"""
        pygame.draw.rect(screen, WHITE, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)  # Border
        
        if self.checked:
            pygame.draw.line(screen, BLACK, 
                            (self.rect.left + 4, self.rect.centery),
                            (self.rect.centerx - 2, self.rect.bottom - 4), 2)
            pygame.draw.line(screen, BLACK, 
                            (self.rect.centerx - 2, self.rect.bottom - 4),
                            (self.rect.right - 4, self.rect.top + 4), 2)
        
        font = pygame.font.SysFont(None, self.font_size)
        text_surf = font.render(self.text, True, BLACK)
        text_rect = text_surf.get_rect(midleft=(self.rect.right + 10, self.rect.centery))
        screen.blit(text_surf, text_rect)
        
    def is_clicked(self, pos, event):
        """Toggle checkbox state if clicked"""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(pos):
                self.checked = not self.checked
                return True
        return False

class Dropdown:
    """Simple dropdown menu for GUI"""
    def __init__(self, x, y, width, height, options, default_index=0, font_size=20):
        self.rect = pygame.Rect(x, y, width, height)
        self.options = options
        self.selected_index = default_index
        self.font_size = font_size
        self.open = False
        self.option_rects = []
        self.update_option_rects()
        
    def update_option_rects(self):
        """Update the rectangles for dropdown options"""
        self.option_rects = []
        for i in range(len(self.options)):
            self.option_rects.append(pygame.Rect(
                self.rect.x, self.rect.y + self.rect.height * (i + 1),
                self.rect.width, self.rect.height
            ))
        
    def draw(self, screen):
        """Draw the dropdown menu"""
        # Draw selected option
        pygame.draw.rect(screen, WHITE, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)  # Border
        
        font = pygame.font.SysFont(None, self.font_size)
        text = self.options[self.selected_index]
        text_surf = font.render(text, True, BLACK)
        text_rect = text_surf.get_rect(midleft=(self.rect.left + 10, self.rect.centery))
        screen.blit(text_surf, text_rect)
        
        # Draw dropdown arrow
        pygame.draw.polygon(screen, BLACK, [
            (self.rect.right - 20, self.rect.centery - 5),
            (self.rect.right - 10, self.rect.centery - 5),
            (self.rect.right - 15, self.rect.centery + 5)
        ])
        
        # Draw options if open
        if self.open:
            for i, option_rect in enumerate(self.option_rects):
                pygame.draw.rect(screen, WHITE, option_rect)
                pygame.draw.rect(screen, BLACK, option_rect, 2)  # Border
                
                text = self.options[i]
                text_surf = font.render(text, True, BLACK)
                text_rect = text_surf.get_rect(midleft=(option_rect.left + 10, option_rect.centery))
                screen.blit(text_surf, text_rect)
                
    def handle_event(self, pos, event):
        """Handle dropdown events"""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(pos):
                self.open = not self.open
                return True
            elif self.open:
                for i, option_rect in enumerate(self.option_rects):
                    if option_rect.collidepoint(pos):
                        self.selected_index = i
                        self.open = False
                        return True
                self.open = False
        return False

class AgentComparisonGUI:
    """GUI for comparing reinforcement learning agents"""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Agent Comparison for Frozen Lake")
        
        # Initialize variables
        self.map_size_dropdown = Dropdown(180, 100, 100, 30, ['4x4', '8x8'], 0)
        self.is_slippery_checkbox = Checkbox(180, 150, 20, "Slippery Environment", checked=True)
        self.episode_dropdown = Dropdown(180, 200, 150, 30, ['1000', '5000', '10000', '20000'], 1)
        self.run_button = Button(100, 250, 200, 40, "Run Comparison", LIGHT_BLUE, BLUE, text_color=BLACK)
        self.exit_button = Button(800, 20, 80, 30, "Exit", LIGHT_BLUE, BLUE, text_color=BLACK)
        
        # Agent configuration
        self.agent_checkboxes = [
            Checkbox(500, 100, 20, "Q-Learning", checked=True),
            Checkbox(500, 150, 20, "SARSA", checked=True),
            Checkbox(500, 200, 20, "DQN", checked=True)
        ]
        
        # Background training
        self.training_thread = None
        self.is_training = False
        self.training_complete = False
        self.progress = 0
        self.result_queue = queue.Queue()
        
        # Results
        self.results_df = None
        self.evaluation_df = None
        self.result_plot_surface = None
        self.saved_plot_path = None
        
        # Create save directory
        self.save_path = 'models/comparison'
        os.makedirs(self.save_path, exist_ok=True)
        
    def create_agents(self):
        """Create the selected agents"""
        agents = []
        
        # Get environment dimensions
        map_name = self.map_size_dropdown.options[self.map_size_dropdown.selected_index]
        is_slippery = self.is_slippery_checkbox.checked
        
        # Create environment to get state/action space sizes
        env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery)
        state_space_size = env.observation_space.n
        action_space_size = env.action_space.n
        env.close()
        
        # Add selected agents
        if self.agent_checkboxes[0].checked:  # Q-Learning
            q_learning = QLearningAgent(
                action_space_size=action_space_size,
                state_space_size=state_space_size,
                learning_rate=0.1,
                discount_factor=0.99,
                exploration_rate=1.0,
                min_exploration_rate=0.01,
                exploration_decay_rate=0.0005
            )
            agents.append({
                'name': 'Q-Learning',
                'agent': q_learning,
                'color': 'blue'
            })
        
        if self.agent_checkboxes[1].checked:  # SARSA
            sarsa = SarsaAgent(
                action_space_size=action_space_size,
                state_space_size=state_space_size,
                learning_rate=0.1,
                discount_factor=0.99,
                exploration_rate=1.0,
                min_exploration_rate=0.01,
                exploration_decay_rate=0.0005
            )
            agents.append({
                'name': 'SARSA',
                'agent': sarsa,
                'color': 'green'
            })
        
        if self.agent_checkboxes[2].checked:  # DQN
            dqn = DQNAgent(
                action_space_size=action_space_size,
                state_space_size=state_space_size,
                learning_rate=0.001,
                discount_factor=0.99,
                exploration_rate=1.0,
                min_exploration_rate=0.01,
                exploration_decay_rate=0.0005,
                batch_size=32,
                update_target_every=100
            )
            agents.append({
                'name': 'DQN',
                'agent': dqn,
                'color': 'red'
            })
        
        return agents
    
    def train_agents_thread(self):
        """Run the training in a separate thread"""
        try:
            map_name = self.map_size_dropdown.options[self.map_size_dropdown.selected_index]
            is_slippery = self.is_slippery_checkbox.checked
            num_episodes = int(self.episode_dropdown.options[self.episode_dropdown.selected_index])
            
            # Create agents
            agents = self.create_agents()
            
            if not agents:
                self.result_queue.put(("error", "No agents selected"))
                return
            
            # Create comparison object
            comparison = AgentComparison(
                agent_configs=agents,
                map_name=map_name,
                is_slippery=is_slippery
            )
            
            # Train agents (disable interactive plots in the comparison)
            results_df = comparison.train_agents(
                num_episodes=num_episodes,
                update_interval=0,  # Disable interactive updating
                save_path=self.save_path
            )
            
            # Evaluate agents (without rendering)
            evaluation_df = comparison.evaluate_agents(
                num_episodes=100,
                render=False
            )
            
            # Create result plot data for later rendering in the main thread
            agent_names = [agent['name'] for agent in agents]
            agent_colors = [agent['color'] for agent in agents]
            success_rates = [evaluation_df.loc[evaluation_df['Agent'] == name, 'Success Rate (%)'].values[0] for name in agent_names]
            avg_steps = [evaluation_df.loc[evaluation_df['Agent'] == name, 'Avg Steps'].values[0] for name in agent_names]
            avg_rewards = [evaluation_df.loc[evaluation_df['Agent'] == name, 'Avg Reward'].values[0] for name in agent_names]
            training_times = [results_df.loc[results_df['Agent'] == name, 'Training Time (s)'].values[0] for name in agent_names]
            
            plot_data = {
                'agents': agent_names,
                'colors': agent_colors,
                'success_rates': success_rates,
                'avg_steps': avg_steps,
                'avg_rewards': avg_rewards,
                'training_times': training_times,
                'map_name': map_name,
                'is_slippery': is_slippery
            }
            
            # Put results in queue for main thread to process
            self.result_queue.put(("success", (results_df, evaluation_df, plot_data)))
            
        except Exception as e:
            self.result_queue.put(("error", str(e)))
    
    def create_result_plot(self, plot_data):
        """Create the result plot in the main thread"""
        agent_names = plot_data['agents']
        agent_colors = plot_data['colors']
        success_rates = plot_data['success_rates']
        avg_steps = plot_data['avg_steps']
        avg_rewards = plot_data['avg_rewards']
        training_times = plot_data['training_times']
        map_name = plot_data['map_name']
        is_slippery = plot_data['is_slippery']
        
        # Create figure and axes
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f'Agent Comparison - {map_name} Map {"(Slippery)" if is_slippery else "(Non-Slippery)"}', fontsize=16)
        
        # Plot bar charts
        bar_width = 0.5
        x = np.arange(len(agent_names))
        
        # Plot success rate
        bars1 = axes[0, 0].bar(x, success_rates, bar_width, color=agent_colors)
        axes[0, 0].set_title('Success Rate (%)')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(agent_names)
        
        # Plot avg steps
        bars2 = axes[0, 1].bar(x, avg_steps, bar_width, color=agent_colors)
        axes[0, 1].set_title('Average Steps')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(agent_names)
        
        # Plot avg reward
        bars3 = axes[1, 0].bar(x, avg_rewards, bar_width, color=agent_colors)
        axes[1, 0].set_title('Average Reward')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(agent_names)
        
        # Plot training time
        bars4 = axes[1, 1].bar(x, training_times, bar_width, color=agent_colors)
        axes[1, 1].set_title('Training Time (s)')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(agent_names)
        
        # Add values on top of bars
        def add_labels(bars, ax):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'{height:.1f}', ha='center', va='bottom')
        
        add_labels(bars1, axes[0, 0])
        add_labels(bars2, axes[0, 1])
        add_labels(bars3, axes[1, 0])
        add_labels(bars4, axes[1, 1])
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the figure as a PNG file with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comparison_plots directory if it doesn't exist
        plots_dir = "comparison_plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        save_filename = f"agent_comparison_{map_name}_{timestamp}.png"
        save_path = os.path.join(plots_dir, save_filename)
        plt.savefig(save_path, dpi=300)
        self.saved_plot_path = os.path.join(plots_dir, save_filename)
        print(f"Comparison plot saved to: {save_path}")
        
        # Convert plot to pygame surface
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.buffer_rgba()
        size = canvas.get_width_height()
        
        # Create surface - convert memoryview to bytes
        self.result_plot_surface = pygame.image.fromstring(bytes(raw_data), size, "RGBA")
        
        # Clean up
        plt.close(fig)
    
    def run(self):
        """Main GUI loop"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # Check for training results
            if self.is_training and not self.result_queue.empty():
                status, data = self.result_queue.get()
                if status == "success":
                    self.results_df, self.evaluation_df, plot_data = data
                    self.create_result_plot(plot_data)
                    self.is_training = False
                    self.training_complete = True
                else:
                    print(f"Error during training: {data}")
                    self.is_training = False
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                mouse_pos = pygame.mouse.get_pos()
                
                # Handle button hover and clicks
                self.run_button.is_hovered(mouse_pos)
                self.exit_button.is_hovered(mouse_pos)
                
                if self.exit_button.is_clicked(mouse_pos, event):
                    running = False
                
                if not self.is_training and not self.training_complete and self.run_button.is_clicked(mouse_pos, event):
                    # Start training in a separate thread
                    self.is_training = True
                    self.training_complete = False
                    self.progress = 0
                    self.training_thread = threading.Thread(target=self.train_agents_thread)
                    self.training_thread.daemon = True
                    self.training_thread.start()
                
                # Handle dropdowns and checkboxes
                if not self.is_training and not self.training_complete:
                    self.map_size_dropdown.handle_event(mouse_pos, event)
                    self.episode_dropdown.handle_event(mouse_pos, event)
                    self.is_slippery_checkbox.is_clicked(mouse_pos, event)
                    
                    for checkbox in self.agent_checkboxes:
                        checkbox.is_clicked(mouse_pos, event)
                
                # Handle run again button in results view
                if self.training_complete:
                    run_again_button = Button(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT - 60, 200, 40, 
                                             "Run New Comparison", LIGHT_BLUE, BLUE)
                    if run_again_button.is_clicked(mouse_pos, event):
                        self.training_complete = False
            
            # Draw the screen
            self.screen.fill(WHITE)
            
            if not self.training_complete:
                # Draw UI elements
                self.draw_header()
                self.draw_config_section()
                
                if self.is_training:
                    self.draw_training_progress()
            else:
                # Draw results
                self.draw_results()
            
            # Update display
            pygame.display.flip()
            clock.tick(30)
        
        pygame.quit()
    
    def draw_header(self):
        """Draw the header section"""
        font = pygame.font.SysFont(None, 36)
        header = font.render("Agent Comparison for Frozen Lake", True, BLACK)
        self.screen.blit(header, (SCREEN_WIDTH // 2 - header.get_width() // 2, 30))
        
        # Draw exit button
        self.exit_button.draw(self.screen)
    
    def draw_config_section(self):
        """Draw the configuration section"""
        font = pygame.font.SysFont(None, 24)
        
        # Environment section
        env_title = font.render("Environment Configuration:", True, BLACK)
        self.screen.blit(env_title, (50, 70))
        
        map_label = font.render("Map Size:", True, BLACK)
        self.screen.blit(map_label, (70, 105))
        self.map_size_dropdown.draw(self.screen)
        
        self.is_slippery_checkbox.draw(self.screen)
        
        episodes_label = font.render("Episodes:", True, BLACK)
        self.screen.blit(episodes_label, (70, 205))
        self.episode_dropdown.draw(self.screen)
        
        # Agent section
        agents_title = font.render("Agents to Compare:", True, BLACK)
        self.screen.blit(agents_title, (400, 70))
        
        for checkbox in self.agent_checkboxes:
            checkbox.draw(self.screen)
        
        # Run button
        self.run_button.draw(self.screen)
    
    def draw_training_progress(self):
        """Draw training progress"""
        # Draw progress background
        progress_rect = pygame.Rect(100, 350, SCREEN_WIDTH - 200, 40)
        pygame.draw.rect(self.screen, GRAY, progress_rect)
        pygame.draw.rect(self.screen, BLACK, progress_rect, 2)
        
        # Update progress based on time (fake progress)
        if self.is_training:
            # Simulate progress based on time
            self.progress = min(100, self.progress + 0.1)
            
            # Draw progress bar
            if self.progress > 0:
                fill_width = int((progress_rect.width - 4) * (self.progress / 100))
                fill_rect = pygame.Rect(progress_rect.left + 2, progress_rect.top + 2, 
                                       fill_width, progress_rect.height - 4)
                pygame.draw.rect(self.screen, BLUE, fill_rect)
        
        # Draw progress text
        font = pygame.font.SysFont(None, 24)
        progress_text = font.render(f"Training in progress... {int(self.progress)}%", True, BLACK)
        self.screen.blit(progress_text, (SCREEN_WIDTH // 2 - progress_text.get_width() // 2, 400))
        
        font = pygame.font.SysFont(None, 20)
        info_text = font.render("This may take several minutes depending on the number of episodes.", True, BLACK)
        self.screen.blit(info_text, (SCREEN_WIDTH // 2 - info_text.get_width() // 2, 430))
    
    def draw_results(self):
        """Draw the results"""
        font = pygame.font.SysFont(None, 36)
        title = font.render("Agent Comparison Results", True, BLACK)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 30))
        
        # Draw result plot
        if self.result_plot_surface:
            plot_rect = self.result_plot_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
            self.screen.blit(self.result_plot_surface, plot_rect)
        
        # Display saved image path
        if self.saved_plot_path:
            font_small = pygame.font.SysFont(None, 20)
            path_text = font_small.render(f"Plot saved as: {self.saved_plot_path}", True, BLACK)
            self.screen.blit(path_text, (SCREEN_WIDTH // 2 - path_text.get_width() // 2, SCREEN_HEIGHT - 90))
        
        # Draw run again button
        run_again_button = Button(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT - 60, 200, 40, 
                                 "Run New Comparison", LIGHT_BLUE, BLUE)
        run_again_button.draw(self.screen)
        
        # Draw exit button
        self.exit_button.draw(self.screen)

def main():
    """Main function"""
    gui = AgentComparisonGUI()
    gui.run()
    return 0

if __name__ == "__main__":
    sys.exit(main()) 