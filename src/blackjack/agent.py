"""Simple tabular Q-learning agent that includes true count bin in state."""
from __future__ import annotations

import pickle
import random
from collections import defaultdict
from typing import Tuple

State = Tuple[int, int, bool, int]  # (player_sum, dealer_show, usable_ace, true_count_bin)


class QLearningAgent:
    def __init__(self, lr: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1, seed: int = 0):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.rand = random.Random(seed)
        self.Q = defaultdict(float)  # key: (state, action)

    def _key(self, state: State, action: int):
        return (state, action)

    def choose_action(self, state: State) -> int:
        if self.rand.random() < self.epsilon:
            return self.rand.choice([0, 1])
        # greedy
        q0 = self.Q[self._key(state, 0)]
        q1 = self.Q[self._key(state, 1)]
        return 0 if q0 >= q1 else 1

    def update(self, state: State, action: int, reward: float, next_state: State, done: bool):
        key = self._key(state, action)
        current = self.Q[key]
        if done:
            target = reward
        else:
            best_next = max(self.Q[self._key(next_state, 0)], self.Q[self._key(next_state, 1)])
            target = reward + self.gamma * best_next
        self.Q[key] = current + self.lr * (target - current)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.Q = defaultdict(float, data)


class BetAwareQLearningAgent:
    """Q-learning agent that selects a bet size and an action.

    Q keys are (state, bet, action) and rewards are assumed to be per-unit
    so total reward scales by the chosen bet.
    """

    def __init__(self, bets=(1, 2, 4, 8), lr: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1, seed: int = 0):
        self.bets = tuple(bets)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.rand = random.Random(seed)
        self.Q = defaultdict(float)  # key: (state, bet, action)

    def _key(self, state: State, bet: int, action: int):
        return (state, bet, action)

    def choose(self, state: State):
        """Return (bet, action)."""
        if self.rand.random() < self.epsilon:
            bet = self.rand.choice(self.bets)
            action = self.rand.choice([0, 1])
            return bet, action
        # greedy: find best (bet,action)
        best = None
        best_val = float("-inf")
        for b in self.bets:
            for a in (0, 1):
                v = self.Q[self._key(state, b, a)]
                if v > best_val:
                    best_val = v
                    best = (b, a)
        if best is None:
            return self.bets[0], 0
        return best

    def update(self, state: State, bet: int, action: int, reward: float, next_state: State, done: bool):
        key = self._key(state, bet, action)
        current = self.Q[key]
        if done:
            target = reward * bet
        else:
            # best next Q across bets/actions
            best_next = max(self.Q[self._key(next_state, b, a)] for b in self.bets for a in (0, 1))
            target = reward * bet + self.gamma * best_next
        self.Q[key] = current + self.lr * (target - current)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.Q = defaultdict(float, data)
