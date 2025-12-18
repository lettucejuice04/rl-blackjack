"""Train bet-aware agents and compare against flat and count-based strategies.

Produces a plot at `plots/tournament.png` showing bankroll over hands.

Usage example:
    python scripts/tournament.py --train-episodes 2000 --eval-hands 20000 --seed 42
"""
from __future__ import annotations

import os
import sys
import argparse
import random
from typing import Callable, List

# make sure `src` package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.blackjack.env import BlackjackEnv
from src.blackjack.agent import BetAwareQLearningAgent
from src.blackjack.train import train_bet
from scripts.simulate_counting import basic_policy, flat_bet, count_bet, simulate_strategy


def evaluate_learned_agent(agent: BetAwareQLearningAgent, env: BlackjackEnv, hands: int = 1000, seed: int = None) -> List[float]:
    """Evaluate a trained bet-aware agent greedily (epsilon=0) over `hands` hands."""
    rng = random.Random(seed)
    bankroll = 0.0
    history = [bankroll]

    # ensure greedy
    old_eps = agent.epsilon
    agent.epsilon = 0.0

    for i in range(hands):
        env.seed(rng.randrange(2**31 - 1))
        state = env.reset()
        bet, action = agent.choose(state)
        done = False
        while not done:
            state, reward, done, _ = env.step(action)
            if not done:
                # agent could adapt to state during the hand (we allow it)
                _, action = agent.choose(state)
        bankroll += reward * bet
        history.append(bankroll)

    agent.epsilon = old_eps
    return history


def run_tournament(train_episodes: int = 2000, eval_hands: int = 20000, seed: int = 42, out: str = "plots/tournament.png") -> None:
    rng = random.Random(seed)

    # Train bet-aware agent
    env = BlackjackEnv(n_decks=6, seed=seed)
    agent = BetAwareQLearningAgent(bets=(1, 2, 4, 8), lr=0.1, gamma=0.99, epsilon=0.1, seed=seed)
    print(f"Training bet-aware agent for {train_episodes} episodes...")
    train_bet(agent, env, episodes=train_episodes)

    # Evaluate learned agent
    print("Evaluating learned agent...")
    learned_hist = evaluate_learned_agent(agent, env, hands=eval_hands, seed=seed)

    # Baselines
    print("Evaluating flat and count baselines...")
    flat_hist = simulate_strategy(BlackjackEnv(n_decks=6, seed=seed), basic_policy, flat_bet(base=1), hands=eval_hands, seed=seed)
    count_hist = simulate_strategy(BlackjackEnv(n_decks=6, seed=seed), basic_policy, count_bet(base=1, ramp=2), hands=eval_hands, seed=seed)

    # Plot results
    plt.figure(figsize=(8, 4))
    plt.plot(learned_hist, label="learned bet-aware")
    plt.plot(flat_hist, label="flat-bet")
    plt.plot(count_hist, label="count-bet (ramp=2)")
    plt.xlabel("Hand")
    plt.ylabel("Bankroll (units)")
    plt.title("Tournament: learned bet-aware vs baselines")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out)
    print(f"Saved tournament plot to: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=int, default=2000)
    parser.add_argument("--eval-hands", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="plots/tournament.png")
    args = parser.parse_args()
    run_tournament(train_episodes=args.train_episodes, eval_hands=args.eval_hands, seed=args.seed, out=args.out)


if __name__ == "__main__":
    main()
