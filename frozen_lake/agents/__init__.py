"""
Agent module for Frozen Lake environment
"""

from frozen_lake.agents.base import Agent
from frozen_lake.agents.q_learning import QLearningAgent
from frozen_lake.agents.sarsa import SarsaAgent
from frozen_lake.agents.dqn import DQNAgent
from frozen_lake.agents.trainer import AgentTrainer

__all__ = ['Agent', 'QLearningAgent', 'SarsaAgent', 'DQNAgent', 'AgentTrainer'] 