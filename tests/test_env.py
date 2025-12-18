import pytest
from src.blackjack.env import BlackjackEnv, _hi_lo_value


def test_hi_lo_values():
    assert _hi_lo_value(2) == 1
    assert _hi_lo_value(5) == 1
    assert _hi_lo_value(7) == 0
    assert _hi_lo_value(10) == -1
    assert _hi_lo_value(1) == -1


def test_running_count_with_injected_deck():
    # deck is popped from the end: we will arrange so draws are predictable
    # deck order: last items are drawn first; to make top-of-deck be [2,3,10,1,...]
    deck = [4] * 52  # filler
    # place our desired top 4 cards at the end (drawn first)
    custom_top = [2, 3, 10, 1]
    deck[-4:] = custom_top

    env = BlackjackEnv(n_decks=1, seed=0, deck=deck)
    obs = env.reset()
    # player gets two cards and dealer's showing card is the next drawn
    # reset counts player's two and dealer[0]
    expected_running = _hi_lo_value(env.player[0]) + _hi_lo_value(env.player[1]) + _hi_lo_value(env.dealer[0])
    assert env.running_count == expected_running

    # If we hit, the drawn card should affect the count
    old_count = env.running_count
    # force a hit
    obs, reward, done, _ = env.step(1)
    assert env.running_count != old_count


def test_bust_and_stick_behaviour():
    # simple deterministic test: choose a deck that causes player to bust on hit
    deck = [10, 10, 10, 9, 9, 9]
    env = BlackjackEnv(deck=deck)
    env.reset()
    # If player hits, likely to bust quickly with high cards; we run until done
    done = False
    while not done:
        obs, reward, done, _ = env.step(1)
        if done:
            assert reward in (-1.0, 0.0, 1.0)
