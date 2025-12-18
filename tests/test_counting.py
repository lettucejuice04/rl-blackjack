from src.blackjack.env import BlackjackEnv
from scripts.simulate_counting import count_bet, flat_bet, simulate_strategy


def test_count_bet_increases_with_tc():
    f = count_bet(base=1, ramp=1)
    assert f(0.0) == 1
    assert f(1.2) >= 1
    assert f(3.9) >= 3


def test_simulate_returns_history_length():
    env = BlackjackEnv(n_decks=1, seed=0)
    hist = simulate_strategy(env, lambda s: 0, flat_bet(base=1), hands=10, seed=0)
    # history includes initial bankroll + entries per hand
    assert len(hist) == 11
    assert isinstance(hist[0], float)
