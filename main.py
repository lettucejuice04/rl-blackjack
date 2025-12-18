"""Entry point: quick script to train and evaluate a Q-learning agent on Blackjack."""
from src.blackjack.env import BlackjackEnv
from src.blackjack.agent import QLearningAgent
from src.blackjack.train import train


def main():
    env = BlackjackEnv(n_decks=2, seed=42)
    agent = QLearningAgent(lr=0.1, gamma=0.99, epsilon=0.05, seed=0)
    print("Training for 2000 episodes (this is lightweight demo)")
    rewards = train(agent, env, episodes=2000)
    avg = sum(rewards[-100:]) / min(100, len(rewards))
    print(f"Average reward (last 100 eps): {avg:.4f}")
    agent.save("q_agent.pkl")


if __name__ == "__main__":
    main()
