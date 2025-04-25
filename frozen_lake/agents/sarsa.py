"""
SARSA Agent for the Frozen Lake environment
"""
import numpy as np
import os
import pickle
from frozen_lake.agents.base import Agent

class SarsaAgent(Agent):
    """
    SARSA Agent that learns to navigate the Frozen Lake environment
    
    This agent implements the SARSA algorithm, which is an on-policy temporal
    difference learning algorithm. It differs from Q-learning in that it updates
    using the action taken in the next state rather than the best action.
    """
    
    def __init__(self, action_space_size, state_space_size, 
                 learning_rate=0.05, discount_factor=0.99, 
                 exploration_rate=1.0, min_exploration_rate=0.01,
                 exploration_decay_rate=0.0005):
        """
        Initialize the SARSA agent
        
        Args:
            action_space_size: Number of possible actions
            state_space_size: Number of possible states
            learning_rate: Alpha - how quickly the agent learns
            discount_factor: Gamma - importance of future rewards
            exploration_rate: Epsilon - initial exploration rate
            min_exploration_rate: Minimum exploration rate
            exploration_decay_rate: Rate at which exploration decreases
        """
        self.action_space_size = action_space_size
        self.state_space_size = state_space_size
        
        # SARSA parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        
        # Initialize Q-table with small random values
        self.q_table = np.random.uniform(low=0, high=0.01, 
                                        size=(state_space_size, action_space_size))
        
        # Performance tracking
        self.training_episodes = 0
        self.successful_episodes = 0
        
        # Store the next action for SARSA update
        self.next_action = None
        
    def get_action(self, state):
        """
        Choose an action using epsilon-greedy policy
        
        Args:
            state: Current state
            
        Returns:
            int: The action to take (0-3)
        """
        # Exploration: choose a random action
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.action_space_size)
        
        # Exploitation: choose the best action according to Q-table
        best_actions = np.where(self.q_table[state] == np.max(self.q_table[state]))[0]
        return np.random.choice(best_actions)
    
    def learn(self, state, action, reward, next_state, done):
        """
        Update the Q-table based on the SARSA update rule
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Choose the next action using epsilon-greedy policy
        next_action = self.get_action(next_state) if not done else 0
        
        # SARSA update formula
        # Q(s,a) = Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
        td_target = reward + (1 - done) * self.discount_factor * self.q_table[next_state, next_action]
        td_error = td_target - self.q_table[state, action]
        
        self.q_table[state, action] += self.learning_rate * td_error
        
        # Update exploration rate
        if done:
            self.training_episodes += 1
            if reward > 0:  # Successfully reached the goal
                self.successful_episodes += 1
                
            # Decay exploration rate
            self.exploration_rate = self.min_exploration_rate + \
                                  (1.0 - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate * self.training_episodes)
    
    def reset(self):
        """
        Reset the agent for a new episode
        """
        # Nothing to reset for SARSA between episodes
        pass
    
    def get_success_rate(self):
        """
        Get the success rate of the agent during training
        
        Returns:
            float: Success rate as a percentage
        """
        if self.training_episodes == 0:
            return 0.0
        
        return (self.successful_episodes / self.training_episodes) * 100
    
    def save_q_table(self, filename='sarsa_q_table.pkl'):
        """
        Save the Q-table to a file
        
        Args:
            filename: Filename to save the Q-table to
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load_q_table(self, filename='sarsa_q_table.pkl'):
        """
        Load the Q-table from a file
        
        Args:
            filename: Filename to load the Q-table from
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        if not os.path.exists(filename):
            return False
        
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
        return True
    
    def get_q_values(self, state):
        """
        Get the Q-values for a given state
        
        Args:
            state: State to get Q-values for
            
        Returns:
            numpy.array: Array of Q-values for each action
        """
        return self.q_table[state] 