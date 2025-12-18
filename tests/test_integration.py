from src.blackjack.env import BlackjackEnv
from src.blackjack.agent import QLearningAgent
from src.blackjack.train import train


def test_short_train_run():
    env = BlackjackEnv(n_decks=1, seed=0)
    agent = QLearningAgent(lr=0.1, gamma=0.99, epsilon=0.2, seed=0)
    rewards = train(agent, env, episodes=200)
    assert isinstance(rewards, list)
    assert len(rewards) == 200
