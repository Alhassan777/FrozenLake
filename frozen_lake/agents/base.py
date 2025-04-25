"""
Base class for Frozen Lake agents
"""
from abc import ABC, abstractmethod

class Agent(ABC):
    """
    Abstract base class for agents that can interact with the Frozen Lake environment
    """
    
    @abstractmethod
    def get_action(self, state):
        """
        Determine the next action to take based on the current state
        
        Args:
            state: The current state of the environment
            
        Returns:
            int: The action to take (0-3)
        """
        pass
    
    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        """
        Learn from an interaction with the environment
        
        Args:
            state: The current state before taking the action
            action: The action taken
            reward: The reward received after taking the action
            next_state: The state after taking the action
            done: Whether the episode is done after this step
        """
        pass
    
    @abstractmethod
    def reset(self):
        """
        Reset the agent for a new episode
        """
        pass 