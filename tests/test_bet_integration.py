from src.blackjack.env import BlackjackEnv
from src.blackjack.agent import BetAwareQLearningAgent
from src.blackjack.train import train_bet


def test_short_bet_training_run():
    env = BlackjackEnv(n_decks=1, seed=0)
    agent = BetAwareQLearningAgent(bets=(1, 2), lr=0.1, gamma=0.99, epsilon=0.2, seed=0)
    rewards = train_bet(agent, env, episodes=100)
    assert isinstance(rewards, list)
    assert len(rewards) == 100
