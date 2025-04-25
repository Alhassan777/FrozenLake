"""
Agent Comparison module for Frozen Lake environment

This module provides tools for comparing different RL agents side-by-side,
showing performance metrics in real-time.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
import time
import matplotlib.animation as animation
from IPython.display import display, clear_output
import os
import torch

class AgentComparison:
    """
    Class to compare multiple agents on the same environment
    """
    
    def __init__(self, agent_configs, map_name="4x4", is_slippery=True, custom_map=None):
        """
        Initialize the comparison
        
        Args:
            agent_configs: List of dictionaries with agent configurations
                Each dict must have:
                - 'name': Display name for the agent
                - 'agent': Agent instance
                - 'color': Color for plots
            map_name: Name of the map to use
            is_slippery: Whether the ice is slippery
            custom_map: Custom map to use (overrides map_name)
        """
        self.agent_configs = agent_configs
        self.map_name = map_name
        self.is_slippery = is_slippery
        self.custom_map = custom_map
        
        # Metrics to track
        self.metrics = {
            'episode_rewards': defaultdict(list),
            'episode_steps': defaultdict(list),
            'success_rates': defaultdict(list),  # Per 100 episodes
            'cumulative_rewards': defaultdict(list),
            'running_success_rate': defaultdict(list),  # Moving window
            'training_time': defaultdict(float)
        }
        
        # Training metrics for real-time tracking
        self.current_metrics = {
            'success_count': defaultdict(int),
            'episode_rewards': defaultdict(list),
            'episode_steps': defaultdict(list),
            'window_size': 100
        }
        
    def train_agents(self, num_episodes=10000, max_steps=100, update_interval=100, save_path=None):
        """
        Train all agents and compare performance
        
        Args:
            num_episodes: Number of episodes to train for
            max_steps: Maximum number of steps per episode
            update_interval: How often to update the plots (0 to disable)
            save_path: Directory to save models and plots
            
        Returns:
            DataFrame: Comparison of final metrics
        """
        from frozen_lake.agents.trainer import AgentTrainer
        import gymnasium as gym
        
        # Create environment for each agent
        environments = []
        trainers = []
        
        for config in self.agent_configs:
            # Create environment
            if self.custom_map is not None:
                env = gym.make('FrozenLake-v1', desc=self.custom_map, is_slippery=self.is_slippery, render_mode=None)
            else:
                env = gym.make('FrozenLake-v1', map_name=self.map_name, is_slippery=self.is_slippery, render_mode=None)
            
            environments.append(env)
            
            # Create trainer without saving
            trainer = AgentTrainer(config['agent'], map_name=self.map_name, 
                                  is_slippery=self.is_slippery, 
                                  custom_map=self.custom_map)
            trainers.append(trainer)
        
        # Prepare for real-time plotting if enabled
        if update_interval > 0:
            plt.ion()
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Agent Comparison - Real-time Training Metrics', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for title
            
            # Initialize empty plots
            reward_lines = []
            step_lines = []
            success_lines = []
            cumulative_lines = []
            
            for config in self.agent_configs:
                name = config['name']
                color = config.get('color', None)
                
                # Rewards plot
                reward_line, = axes[0, 0].plot([], [], label=name, color=color)
                reward_lines.append(reward_line)
                
                # Steps plot
                step_line, = axes[0, 1].plot([], [], label=name, color=color)
                step_lines.append(step_line)
                
                # Success rate plot
                success_line, = axes[1, 0].plot([], [], label=name, color=color)
                success_lines.append(success_line)
                
                # Cumulative rewards plot
                cumulative_line, = axes[1, 1].plot([], [], label=name, color=color)
                cumulative_lines.append(cumulative_line)
            
            # Set up plot labels and legends
            axes[0, 0].set_title('Average Reward per Episode')
            axes[0, 0].set_xlabel('Episodes')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            axes[0, 1].set_title('Average Steps per Episode')
            axes[0, 1].set_xlabel('Episodes')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            axes[1, 0].set_title('Success Rate (per 100 episodes)')
            axes[1, 0].set_xlabel('Episodes')
            axes[1, 0].set_ylabel('Success Rate (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            axes[1, 1].set_title('Cumulative Reward')
            axes[1, 1].set_xlabel('Episodes')
            axes[1, 1].set_ylabel('Cumulative Reward')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # Training loop
        for episode in range(num_episodes):
            # Train each agent for one episode
            for i, config in enumerate(self.agent_configs):
                name = config['name']
                agent = config['agent']
                env = environments[i]
                
                # Start timer
                start_time = time.time()
                
                # Train for one episode
                state, _ = env.reset()
                episode_reward = 0
                episode_steps = 0
                
                for step in range(max_steps):
                    # Choose action
                    action = agent.get_action(state)
                    
                    # Take action
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    # Learn from the experience
                    agent.learn(state, action, reward, next_state, done)
                    
                    # Update state and metrics
                    state = next_state
                    episode_reward += reward
                    episode_steps += 1
                    
                    if done:
                        break
                
                # Update metrics
                self.metrics['episode_rewards'][name].append(episode_reward)
                self.metrics['episode_steps'][name].append(episode_steps)
                self.current_metrics['episode_rewards'][name].append(episode_reward)
                self.current_metrics['episode_steps'][name].append(episode_steps)
                
                # Check for success
                if episode_reward > 0:
                    self.current_metrics['success_count'][name] += 1
                
                # Update training time
                self.metrics['training_time'][name] += time.time() - start_time
                
                # Calculate metrics for plotting every 100 episodes
                if (episode + 1) % 100 == 0:
                    window_success_rate = self.current_metrics['success_count'][name]
                    self.metrics['success_rates'][name].append(window_success_rate)
                    
                    # Reset success count
                    self.current_metrics['success_count'][name] = 0
                
                # Calculate cumulative reward
                if len(self.metrics['cumulative_rewards'][name]) > 0:
                    last_cum_reward = self.metrics['cumulative_rewards'][name][-1]
                else:
                    last_cum_reward = 0
                self.metrics['cumulative_rewards'][name].append(last_cum_reward + episode_reward)
                
                # Calculate moving average success rate
                window_size = min(100, episode + 1)
                recent_rewards = self.metrics['episode_rewards'][name][-window_size:]
                success_count = sum(reward > 0 for reward in recent_rewards)
                success_rate = (success_count / window_size) * 100
                self.metrics['running_success_rate'][name].append(success_rate)
            
            # Update plots at specified intervals
            if update_interval > 0 and (episode + 1) % update_interval == 0:
                self._update_plots(episode, reward_lines, step_lines, success_lines, cumulative_lines)
                
                # Save intermediate results
                if save_path:
                    os.makedirs(save_path, exist_ok=True)
                    for i, config in enumerate(self.agent_configs):
                        name = config['name']
                        agent = config['agent']
                        
                        # Save model (use appropriate save method)
                        if hasattr(agent, 'save_q_table'):
                            agent.save_q_table(os.path.join(save_path, f"{name}_model.pkl"))
                        elif hasattr(agent, 'save_model'):
                            agent.save_model(os.path.join(save_path, f"{name}_model.pth"))
                    
                    # Save current comparison plot
                    if update_interval > 0:
                        plt.savefig(os.path.join(save_path, "comparison_plot.png"))
        
        # Turn off interactive mode and display the final plot if enabled
        if update_interval > 0:
            plt.ioff()
        
        # Create final comparison table
        comparison_data = []
        for config in self.agent_configs:
            name = config['name']
            
            # Calculate final metrics
            avg_reward = np.mean(self.metrics['episode_rewards'][name][-1000:])
            avg_steps = np.mean(self.metrics['episode_steps'][name][-1000:])
            final_success_rate = self.metrics['running_success_rate'][name][-1] if self.metrics['running_success_rate'][name] else 0
            total_time = self.metrics['training_time'][name]
            
            comparison_data.append({
                'Agent': name,
                'Avg Reward (last 1000)': avg_reward,
                'Avg Steps (last 1000)': avg_steps, 
                'Success Rate (last 100)': final_success_rate,
                'Training Time (s)': total_time
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save final results
        if save_path:
            # Save models
            os.makedirs(save_path, exist_ok=True)
            for i, config in enumerate(self.agent_configs):
                name = config['name']
                agent = config['agent']
                
                # Save model (use appropriate save method)
                if hasattr(agent, 'save_q_table'):
                    agent.save_q_table(os.path.join(save_path, f"{name}_model.pkl"))
                elif hasattr(agent, 'save_model'):
                    agent.save_model(os.path.join(save_path, f"{name}_model.pth"))
            
            # Save comparison table
            comparison_df.to_csv(os.path.join(save_path, "comparison_results.csv"), index=False)
            
            # Save final plot
            if update_interval > 0:
                fig, axes = self._create_final_plots()
                plt.savefig(os.path.join(save_path, "final_comparison_plot.png"))
                plt.close(fig)
        
        return comparison_df
    
    def _update_plots(self, episode, reward_lines, step_lines, success_lines, cumulative_lines):
        """
        Update the real-time plots
        
        Args:
            episode: Current episode number
            reward_lines: List of line objects for reward plot
            step_lines: List of line objects for step plot
            success_lines: List of line objects for success rate plot
            cumulative_lines: List of line objects for cumulative reward plot
        """
        # X-axis for all plots
        episodes = list(range(episode + 1))
        success_episodes = list(range(100, episode + 1, 100)) if episode >= 100 else []
        
        # Update each agent's plots
        for i, config in enumerate(self.agent_configs):
            name = config['name']
            
            # Update reward plot
            reward_data = self.metrics['episode_rewards'][name]
            if len(reward_data) > 0:
                # Use moving average for smoother plots
                window_size = min(100, len(reward_data))
                reward_avg = [np.mean(reward_data[max(0, j-window_size+1):j+1]) for j in range(len(reward_data))]
                reward_lines[i].set_data(episodes[:len(reward_avg)], reward_avg)
            
            # Update step plot
            step_data = self.metrics['episode_steps'][name]
            if len(step_data) > 0:
                window_size = min(100, len(step_data))
                step_avg = [np.mean(step_data[max(0, j-window_size+1):j+1]) for j in range(len(step_data))]
                step_lines[i].set_data(episodes[:len(step_avg)], step_avg)
            
            # Update success rate plot
            success_data = self.metrics['success_rates'][name]
            if len(success_data) > 0:
                success_lines[i].set_data(success_episodes[:len(success_data)], success_data)
            
            # Update cumulative reward plot
            cumulative_data = self.metrics['cumulative_rewards'][name]
            if len(cumulative_data) > 0:
                cumulative_lines[i].set_data(episodes[:len(cumulative_data)], cumulative_data)
        
        # Adjust axis limits for all plots
        for ax in plt.gcf().axes:
            ax.relim()
            ax.autoscale_view()
        
        plt.draw()
        plt.pause(0.01)
    
    def _create_final_plots(self):
        """
        Create final summary plots
        
        Returns:
            tuple: Figure and axes for the final plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Agent Comparison - Final Results', fontsize=16)
        
        # Get maximum lengths for x-axes
        max_episodes = max([len(self.metrics['episode_rewards'][config['name']]) 
                          for config in self.agent_configs])
        max_success_episodes = max([len(self.metrics['success_rates'][config['name']]) 
                                  for config in self.agent_configs])
        
        episodes = list(range(max_episodes))
        success_episodes = list(range(100, max_success_episodes * 100 + 1, 100)) if max_success_episodes > 0 else []
        
        # Plot for each agent
        for config in self.agent_configs:
            name = config['name']
            color = config.get('color', None)
            
            # Plot average reward (with smoothing)
            reward_data = self.metrics['episode_rewards'][name]
            if len(reward_data) > 0:
                window_size = min(100, len(reward_data))
                reward_avg = [np.mean(reward_data[max(0, j-window_size+1):j+1]) for j in range(len(reward_data))]
                axes[0, 0].plot(episodes[:len(reward_avg)], reward_avg, label=name, color=color)
            
            # Plot average steps (with smoothing)
            step_data = self.metrics['episode_steps'][name]
            if len(step_data) > 0:
                window_size = min(100, len(step_data))
                step_avg = [np.mean(step_data[max(0, j-window_size+1):j+1]) for j in range(len(step_data))]
                axes[0, 1].plot(episodes[:len(step_avg)], step_avg, label=name, color=color)
            
            # Plot success rate
            success_data = self.metrics['success_rates'][name]
            if len(success_data) > 0:
                axes[1, 0].plot(success_episodes[:len(success_data)], success_data, label=name, color=color)
            
            # Plot cumulative reward
            cumulative_data = self.metrics['cumulative_rewards'][name]
            if len(cumulative_data) > 0:
                axes[1, 1].plot(episodes[:len(cumulative_data)], cumulative_data, label=name, color=color)
        
        # Set plot labels and legends
        axes[0, 0].set_title('Average Reward per Episode')
        axes[0, 0].set_xlabel('Episodes')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].set_title('Average Steps per Episode')
        axes[0, 1].set_xlabel('Episodes')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].set_title('Success Rate (per 100 episodes)')
        axes[1, 0].set_xlabel('Episodes')
        axes[1, 0].set_ylabel('Success Rate (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].set_title('Cumulative Reward')
        axes[1, 1].set_xlabel('Episodes')
        axes[1, 1].set_ylabel('Cumulative Reward')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        return fig, axes
    
    def evaluate_agents(self, num_episodes=100, render=False, max_steps=100):
        """
        Evaluate all agents on the same environment
        
        Args:
            num_episodes: Number of episodes to evaluate for
            render: Whether to render the environment
            max_steps: Maximum number of steps per episode
            
        Returns:
            DataFrame: Evaluation results
        """
        import gymnasium as gym
        
        # Metrics to track
        eval_metrics = {
            'success_rate': {},
            'avg_steps': {},
            'avg_reward': {}
        }
        
        # Evaluate each agent
        for config in self.agent_configs:
            name = config['name']
            agent = config['agent']
            
            # Create evaluation environment
            if self.custom_map is not None:
                env = gym.make('FrozenLake-v1', desc=self.custom_map, is_slippery=self.is_slippery, render_mode='human' if render else None)
            else:
                env = gym.make('FrozenLake-v1', map_name=self.map_name, is_slippery=self.is_slippery, render_mode='human' if render else None)
            
            # Save original exploration rate and set to 0 for deterministic policy
            original_exploration = agent.exploration_rate
            agent.exploration_rate = 0
            
            # Evaluation loop
            success_count = 0
            total_steps = 0
            total_reward = 0
            
            for episode in range(num_episodes):
                state, _ = env.reset()
                episode_reward = 0
                steps = 0
                
                for step in range(max_steps):
                    # Choose action (exploitation only)
                    # For DQN, we need to use the policy network directly
                    if hasattr(agent, 'policy_net'):
                        # This is a DQN agent
                        with torch.no_grad():
                            state_tensor = agent.state_to_tensor(state)
                            q_values = agent.policy_net(state_tensor)
                            action = q_values.max(1)[1].item()
                    else:
                        # For Q-Learning and SARSA
                        action = np.argmax(agent.get_q_values(state))
                    
                    # Take action
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    # Update metrics
                    state = next_state
                    episode_reward += reward
                    steps += 1
                    
                    if done:
                        if reward > 0:
                            success_count += 1
                        break
                
                total_steps += steps
                total_reward += episode_reward
            
            # Restore original exploration rate
            agent.exploration_rate = original_exploration
            
            # Calculate evaluation metrics
            eval_metrics['success_rate'][name] = (success_count / num_episodes) * 100
            eval_metrics['avg_steps'][name] = total_steps / num_episodes
            eval_metrics['avg_reward'][name] = total_reward / num_episodes
        
        # Create evaluation dataframe
        eval_data = []
        for config in self.agent_configs:
            name = config['name']
            eval_data.append({
                'Agent': name,
                'Success Rate (%)': eval_metrics['success_rate'][name],
                'Avg Steps': eval_metrics['avg_steps'][name],
                'Avg Reward': eval_metrics['avg_reward'][name]
            })
        
        return pd.DataFrame(eval_data) 