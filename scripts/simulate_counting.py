"""Simulate betting strategies that use card-counting vs flat betting.

Usage (example):
    python scripts/simulate_counting.py --hands 10000 --seed 42
"""
from __future__ import annotations

import argparse
import os
import sys
# ensure project root is on sys.path so `src` is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
from typing import Callable, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.blackjack.env import BlackjackEnv


Policy = Callable[[tuple], int]  # state -> action (0=stick,1=hit)
BetFn = Callable[[float], int]  # true_count -> bet units


def basic_policy(state: tuple) -> int:
    player_sum, dealer_show, usable, tc = state
    # ultra-simple strategy: stand on 17+, else hit
    return 0 if player_sum >= 17 else 1


def flat_bet(base: int = 1) -> BetFn:
    def f(tc: float) -> int:
        return base
    return f


def count_bet(base: int = 1, ramp: int = 1) -> BetFn:
    """Bet increases linearly with positive true count.

    Example: bet = base * max(1, int(floor(true_count))) * ramp
    """

    def f(tc: float) -> int:
        mult = max(1, int(tc))
        return base * mult * ramp

    return f


def simulate_strategy(env: BlackjackEnv, policy: Policy, bet_fn: BetFn, hands: int = 1000, seed: int = None) -> List[float]:
    rng = random.Random(seed)
    bankroll = 0.0
    history: List[float] = [bankroll]

    for i in range(hands):
        env.seed(rng.randrange(2**31 - 1))
        state = env.reset()
        # choose bet based on raw true count at start of hand
        tc = env.true_count()
        bet = bet_fn(tc)

        done = False
        while not done:
            action = policy(state)
            state, reward, done, _ = env.step(action)
        # reward is -1/0/1 per unit; scale by bet
        bankroll += reward * bet
        history.append(bankroll)
    return history


def plot_histories(histories: dict, out: str):
    plt.figure(figsize=(8, 4))
    for label, h in histories.items():
        plt.plot(h, label=label, alpha=0.8)
    plt.xlabel("Hand")
    plt.ylabel("Bankroll (units)")
    plt.legend()
    plt.title("Counting vs Flat Betting: Bankroll over hands")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out)
    print(f"Saved plot to: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="plots/count_vs_flat.png")
    args = parser.parse_args()

    env = BlackjackEnv(n_decks=6, seed=args.seed)

    flat = simulate_strategy(env, basic_policy, flat_bet(base=1), hands=args.hands, seed=args.seed)
    count = simulate_strategy(env, basic_policy, count_bet(base=1, ramp=1), hands=args.hands, seed=args.seed)

    histories = {"flat": flat, "count": count}
    plot_histories(histories, args.out)


if __name__ == "__main__":
    main()
