"""Run a short training session and save learning curve plot."""
import os
import sys
# ensure project root (workspace) is on sys.path so `src` is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.blackjack.env import BlackjackEnv
from src.blackjack.agent import QLearningAgent
from src.blackjack.train import train


def moving_average(x, w=50):
    if len(x) < w:
        return x
    import numpy as np
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    env = BlackjackEnv(n_decks=2, seed=0)
    agent = QLearningAgent(lr=0.1, gamma=0.99, epsilon=0.1, seed=0)
    episodes = 2000
    rewards = train(agent, env, episodes=episodes)

    ma = moving_average(rewards, w=100)

    os.makedirs("plots", exist_ok=True)
    out = os.path.join("plots", "learning_curve.png")

    plt.figure(figsize=(8, 4))
    plt.plot(rewards, alpha=0.3, label="episode reward")
    plt.plot(range(len(ma)), ma, color="tab:orange", label="100-ep MA")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Q-learning on Blackjack (tabular, count-aware)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    print(f"Saved learning curve to: {out}")


if __name__ == "__main__":
    main()
