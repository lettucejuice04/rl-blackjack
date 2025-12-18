from src.blackjack.agent import QLearningAgent, BetAwareQLearningAgent


def test_q_update_changes_value():
    agent = QLearningAgent(lr=1.0, gamma=0.0, epsilon=0.0, seed=1)
    state = (12, 5, False, 0)
    next_state = (20, 10, False, 0)
    # initial Q is 0
    agent.update(state, 1, 1.0, next_state, done=True)
    assert agent.Q[(state, 1)] == 1.0

    # if not done, target uses next state's best value
    agent = QLearningAgent(lr=0.5, gamma=1.0, epsilon=0.0, seed=1)
    agent.Q[((next_state), 0)] = 2.0
    agent.update(state, 0, 0.0, next_state, done=False)
    assert agent.Q[(state, 0)] > 0.0


def test_bet_agent_update_changes_value():
    agent = BetAwareQLearningAgent(bets=(1, 2), lr=1.0, gamma=0.0, epsilon=0.0, seed=2)
    state = (12, 5, False, 0)
    next_state = (20, 10, False, 0)
    agent.update(state, 2, 1, 1.0, next_state, done=True)
    assert agent.Q[(state, 2, 1)] == 2.0  # reward scaled by bet on terminal

    agent = BetAwareQLearningAgent(bets=(1, 2), lr=0.5, gamma=1.0, epsilon=0.0, seed=2)
    agent.Q[(next_state, 1, 0)] = 3.0
    agent.update(state, 1, 0, 0.0, next_state, done=False)
    assert agent.Q[(state, 1, 0)] > 0.0
