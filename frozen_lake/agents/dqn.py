"""
Deep Q-Network (DQN) Agent for the Frozen Lake environment
"""
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from frozen_lake.agents.base import Agent

class DQNNetwork(nn.Module):
    """
    Neural network for the DQN agent
    """
    def __init__(self, state_size, action_size, hidden_size=64):
        """
        Initialize the network
        
        Args:
            state_size: Number of state dimensions
            action_size: Number of actions
            hidden_size: Size of hidden layer
        """
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions
    """
    def __init__(self, capacity=10000):
        """
        Initialize the buffer
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*experiences)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """
        Get the current size of the buffer
        
        Returns:
            int: Number of experiences in the buffer
        """
        return len(self.buffer)

class DQNAgent(Agent):
    """
    Deep Q-Network Agent that learns to navigate the Frozen Lake environment
    
    This agent implements the DQN algorithm, which uses a neural network to
    approximate the Q-function and experience replay to improve training stability.
    """
    
    def __init__(self, action_space_size, state_space_size, 
                learning_rate=0.001, discount_factor=0.99, 
                exploration_rate=1.0, min_exploration_rate=0.01,
                exploration_decay_rate=0.0005, batch_size=64,
                update_target_every=200):
        """
        Initialize the DQN agent
        
        Args:
            action_space_size: Number of possible actions
            state_space_size: Number of possible states
            learning_rate: How quickly the model learns
            discount_factor: Importance of future rewards
            exploration_rate: Initial exploration rate
            min_exploration_rate: Minimum exploration rate
            exploration_decay_rate: Rate at which exploration decreases
            batch_size: Number of samples to learn from each time
            update_target_every: How often to update the target network
        """
        self.action_space_size = action_space_size
        self.state_space_size = state_space_size
        
        # DQN parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        
        # Use one-hot encoding for discrete states
        self.state_size = state_space_size
        
        # Initialize neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(self.state_size, action_space_size).to(self.device)
        self.target_net = DQNNetwork(self.state_size, action_space_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training tracking
        self.training_episodes = 0
        self.successful_episodes = 0
        self.steps = 0
    
    def state_to_tensor(self, state):
        """
        Convert a state to a tensor for the neural network
        
        Args:
            state: The state (integer)
            
        Returns:
            state_tensor: The tensor representation of the state
        """
        # Create one-hot encoded representation
        one_hot = np.zeros(self.state_size, dtype=np.float32)
        one_hot[state] = 1.0
        # Convert to tensor
        state_tensor = torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0)
        return state_tensor
    
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
        
        # Exploitation: choose the best action according to the policy network
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()
    
    def learn(self, state, action, reward, next_state, done):
        """
        Add experience to replay buffer and learn from a batch
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Add experience to replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Increment steps
        self.steps += 1
        
        # Only learn if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Create state tensors
        state_batch = torch.zeros(self.batch_size, self.state_size, dtype=torch.float32).to(self.device)
        next_state_batch = torch.zeros(self.batch_size, self.state_size, dtype=torch.float32).to(self.device)
        
        # Fill one-hot encodings
        for i, s in enumerate(states):
            state_batch[i, s] = 1.0
        
        for i, s in enumerate(next_states):
            next_state_batch[i, s] = 1.0
        
        # Convert action, reward, and done
        action_batch = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        done_batch = torch.tensor([float(d) for d in dones], dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next Q values using the target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1, keepdim=True)[0]
            
        # Compute target Q values
        target_q_values = reward_batch + (1 - done_batch) * self.discount_factor * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update target network periodically
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Update exploration rate and performance tracking at the end of an episode
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
        # Nothing to reset for DQN between episodes
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
    
    def save_model(self, filename='dqn_model.pth'):
        """
        Save the model to a file
        
        Args:
            filename: Filename to save the model to
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'exploration_rate': self.exploration_rate,
            'steps': self.steps,
            'training_episodes': self.training_episodes,
            'successful_episodes': self.successful_episodes
        }, filename)
    
    def load_model(self, filename='dqn_model.pth'):
        """
        Load the model from a file
        
        Args:
            filename: Filename to load the model from
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        if not os.path.exists(filename):
            return False
        
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.exploration_rate = checkpoint['exploration_rate']
        self.steps = checkpoint['steps']
        self.training_episodes = checkpoint['training_episodes']
        self.successful_episodes = checkpoint['successful_episodes']
        return True
    
    def get_q_values(self, state):
        """
        Get the Q-values for a given state
        
        Args:
            state: State to get Q-values for
            
        Returns:
            numpy.array: Array of Q-values for each action
        """
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            q_values = self.policy_net(state_tensor)
            return q_values.cpu().numpy()[0] 