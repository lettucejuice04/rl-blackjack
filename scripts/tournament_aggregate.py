"""Aggregate tournament runs across multiple seeds and plot mean ± std.

Usage example:
    python scripts/tournament_aggregate.py --repeats 5 --train-episodes 1000 --eval-hands 5000 --seed 42
"""
from __future__ import annotations

import os
import sys
import argparse
import random
from typing import List

# ensure `src` is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.blackjack.env import BlackjackEnv
from src.blackjack.agent import BetAwareQLearningAgent
from src.blackjack.train import train_bet
from scripts.simulate_counting import basic_policy, flat_bet, count_bet, simulate_strategy
from scripts.tournament import evaluate_learned_agent


def _pad_histories(histories: List[List[float]]) -> np.ndarray:
    # pad with last value to make equal lengths
    max_len = max(len(h) for h in histories)
    arr = np.zeros((len(histories), max_len), dtype=float)
    for i, h in enumerate(histories):
        arr[i, : len(h)] = h
        if len(h) < max_len:
            arr[i, len(h) :] = h[-1]
    return arr


def run_aggregate(repeats: int = 5, train_episodes: int = 2000, eval_hands: int = 10000, seed: int = 42, out: str = "plots/tournament_agg.png") -> None:
    rng = random.Random(seed)

    learned_runs: List[List[float]] = []
    flat_runs: List[List[float]] = []
    count_runs: List[List[float]] = []

    for r in range(repeats):
        s = seed + r
        print(f"Run {r+1}/{repeats} (seed={s})")
        env = BlackjackEnv(n_decks=6, seed=s)
        agent = BetAwareQLearningAgent(bets=(1, 2, 4, 8), lr=0.1, gamma=0.99, epsilon=0.1, seed=s)
        train_bet(agent, env, episodes=train_episodes)
        learned = evaluate_learned_agent(agent, env, hands=eval_hands, seed=s)
        flat = simulate_strategy(BlackjackEnv(n_decks=6, seed=s), basic_policy, flat_bet(base=1), hands=eval_hands, seed=s)
        count = simulate_strategy(BlackjackEnv(n_decks=6, seed=s), basic_policy, count_bet(base=1, ramp=2), hands=eval_hands, seed=s)

        learned_runs.append(learned)
        flat_runs.append(flat)
        count_runs.append(count)

    # convert to arrays (shape: runs x time)
    learned_arr = _pad_histories(learned_runs)
    flat_arr = _pad_histories(flat_runs)
    count_arr = _pad_histories(count_runs)

    # compute mean and std across runs
    learned_mean = learned_arr.mean(axis=0)
    learned_std = learned_arr.std(axis=0)
    flat_mean = flat_arr.mean(axis=0)
    flat_std = flat_arr.std(axis=0)
    count_mean = count_arr.mean(axis=0)
    count_std = count_arr.std(axis=0)

    # plot
    x = np.arange(len(learned_mean))
    plt.figure(figsize=(8, 4))
    plt.plot(x, learned_mean, label="learned bet-aware (mean)")
    plt.fill_between(x, learned_mean - learned_std, learned_mean + learned_std, alpha=0.2)

    plt.plot(x, flat_mean, label="flat-bet (mean)")
    plt.fill_between(x, flat_mean - flat_std, flat_mean + flat_std, alpha=0.2)

    plt.plot(x, count_mean, label="count-bet (mean)")
    plt.fill_between(x, count_mean - count_std, count_mean + count_std, alpha=0.2)

    plt.xlabel("Hand")
    plt.ylabel("Bankroll (units)")
    plt.title(f"Aggregate Tournament (repeats={repeats}, train={train_episodes}, eval_hands={eval_hands})")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out)
    print(f"Saved aggregated tournament plot to: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--train-episodes", type=int, default=2000)
    parser.add_argument("--eval-hands", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="plots/tournament_agg.png")
    args = parser.parse_args()
    run_aggregate(repeats=args.repeats, train_episodes=args.train_episodes, eval_hands=args.eval_hands, seed=args.seed, out=args.out)


if __name__ == "__main__":
    main()
