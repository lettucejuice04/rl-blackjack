"""Training utilities for Blackjack agents."""
from typing import List
from .env import BlackjackEnv
from .agent import QLearningAgent


def train(agent: QLearningAgent, env: BlackjackEnv, episodes: int = 1000) -> List[float]:
    rewards = []
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
        rewards.append(ep_reward)
    return rewards


def train_bet(agent, env: BlackjackEnv, episodes: int = 1000) -> List[float]:
    """Train a bet-aware agent. Agent must implement `choose(state) -> (bet, action)` and `update(state, bet, action, reward, next_state, done)`.
    Returns list of episode returns (bankroll change per episode, scaled by chosen bet each hand).
    """
    rewards = []
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            bet, action = agent.choose(state)
            next_state, reward, done, _ = env.step(action)
            # reward is per-unit; agent update expects reward as unit reward
            agent.update(state, bet, action, reward, next_state, done)
            state = next_state
            ep_reward += reward * bet
        rewards.append(ep_reward)
    return rewards
