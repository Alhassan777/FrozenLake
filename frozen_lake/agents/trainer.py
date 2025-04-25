"""
Trainer module for training agents on the Frozen Lake environment
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class AgentTrainer:
    """
    Trainer class for training agents on the Frozen Lake environment
    """
    
    def __init__(self, agent, map_name="4x4", is_slippery=True, custom_map=None, save_path=None):
        """
        Initialize the trainer
        
        Args:
            agent: The agent to train
            map_name: Name of the map to use
            is_slippery: Whether the ice is slippery
            custom_map: Custom map to use (overrides map_name)
            save_path: Path to save the trained model
        """
        self.agent = agent
        self.save_path = save_path
        
        # Initialize the environment
        if custom_map is not None:
            self.env = gym.make('FrozenLake-v1', desc=custom_map, is_slippery=is_slippery, render_mode=None)
        else:
            self.env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery, render_mode=None)
        
        # Performance tracking
        self.rewards_per_episode = []
        self.steps_per_episode = []
        self.success_rate_history = []
        self.cumulative_success_rate = []
    
    def train(self, num_episodes=10000, max_steps=100, print_interval=1000, save_interval=None, start_episode=0):
        """
        Train the agent on the environment
        
        Args:
            num_episodes: Number of episodes to train for
            max_steps: Maximum number of steps per episode
            print_interval: How often to print progress
            save_interval: How often to save the model
            start_episode: Episode to start from (for incremental training)
            
        Returns:
            dict: Training statistics
        """
        total_rewards = 0
        success_count = 0
        total_success_count = 0
        
        # Local tracking for this training batch
        local_rewards = []
        local_steps = []
        local_success_rates = []
        local_cumulative_rates = []
        
        # Training loop
        for episode in tqdm(range(num_episodes), desc="Training Progress"):
            state, _ = self.env.reset()
            episode_reward = 0
            step_count = 0
            
            # Episode loop
            for step in range(max_steps):
                # Choose action
                action = self.agent.get_action(state)
                
                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Learn from the experience
                self.agent.learn(state, action, reward, next_state, done)
                
                # Update state
                state = next_state
                episode_reward += reward
                step_count += 1
                
                if done:
                    break
            
            # Track performance
            self.rewards_per_episode.append(episode_reward)
            self.steps_per_episode.append(step_count)
            local_rewards.append(episode_reward)
            local_steps.append(step_count)
            total_rewards += episode_reward
            
            if episode_reward > 0:
                success_count += 1
                total_success_count += 1
            
            # Calculate running success rate (per 100 episodes)
            if (episode + 1) % 100 == 0:
                # Window success rate (last 100 episodes)
                window_success_rate = success_count
                self.success_rate_history.append(window_success_rate)
                local_success_rates.append(window_success_rate)
                
                # Cumulative success rate 
                cumulative_rate = (total_success_count / (episode + 1)) * 100
                self.cumulative_success_rate.append(cumulative_rate)
                local_cumulative_rates.append(cumulative_rate)
                
                # Only reset the window counter
                success_count = 0
            
            # Print progress
            if print_interval > 0 and (episode + 1) % print_interval == 0:
                avg_reward = total_rewards / print_interval
                current_cumulative_rate = (total_success_count / (episode + 1)) * 100
                print(f"Episode {start_episode + episode + 1}/{start_episode + num_episodes} - "
                      f"Avg Reward: {avg_reward:.4f}, "
                      f"Success Rate (last 100): {self.success_rate_history[-1]:.2f}%, "
                      f"Cumulative Rate: {current_cumulative_rate:.2f}%")
                total_rewards = 0
            
            # Save model
            if save_interval and (episode + 1) % save_interval == 0 and self.save_path:
                self.agent.save_q_table(self.save_path)
        
        # Final save
        if self.save_path:
            self.agent.save_q_table(self.save_path)
        
        return {
            'rewards_per_episode': local_rewards,
            'steps_per_episode': local_steps,
            'success_rate_history': local_success_rates,
            'cumulative_success_rate': local_cumulative_rates
        }
    
    def evaluate(self, num_episodes=100, render=False, max_steps=100):
        """
        Evaluate the agent on the environment
        
        Args:
            num_episodes: Number of episodes to evaluate for
            render: Whether to render the environment
            max_steps: Maximum number of steps per episode
            
        Returns:
            float: Success rate as a percentage
        """
        success_count = 0
        total_steps = 0
        
        # Set exploration to 0 to ensure deterministic policy evaluation
        original_exploration = self.agent.exploration_rate
        self.agent.exploration_rate = 0
        
        # Evaluation loop
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            steps = 0
            
            for step in range(max_steps):
                # Use exploitation only (no exploration)
                action = np.argmax(self.agent.get_q_values(state))
                
                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                if render:
                    self.env.render()
                
                state = next_state
                steps += 1
                
                if done:
                    total_steps += steps
                    if reward > 0:
                        success_count += 1
                    break
        
        # Restore original exploration rate
        self.agent.exploration_rate = original_exploration
        
        success_rate = (success_count / num_episodes) * 100
        avg_steps = total_steps / num_episodes if num_episodes > 0 else 0
        
        print(f"Evaluation - Success Rate: {success_rate:.2f}%, Avg Steps: {avg_steps:.2f}")
        return success_rate
    
    def plot_training_results(self):
        """
        Plot the training results
        """
        # Create figure with 4 subplots (added cumulative success rate)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20))
        
        # Plot rewards
        ax1.plot(self.rewards_per_episode)
        ax1.set_title('Rewards per Episode')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # Plot steps
        ax2.plot(self.steps_per_episode)
        ax2.set_title('Steps per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.grid(True)
        
        # Plot window success rate (per 100 episodes)
        if len(self.success_rate_history) > 0:
            episodes = np.arange(100, len(self.success_rate_history) * 100 + 1, 100)
            ax3.plot(episodes, self.success_rate_history)
            ax3.set_title('Success Rate (per 100 episodes)')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Success Rate (%)')
            ax3.grid(True)
        
        # Plot cumulative success rate
        if len(self.cumulative_success_rate) > 0:
            episodes = np.arange(100, len(self.cumulative_success_rate) * 100 + 1, 100)
            ax4.plot(episodes, self.cumulative_success_rate)
            ax4.set_title('Cumulative Success Rate')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Success Rate (%)')
            ax4.grid(True)
        
        plt.tight_layout()
        return fig 