"""Generate a short HTML report with numeric summaries and embedded plots.

This script runs a small aggregate tournament (or uses provided results) to compute
numeric summaries (mean/std final bankroll) and produces `reports/summary.html`.
"""
from __future__ import annotations

import os
import sys
import argparse
import json
from typing import List, Tuple

# make sure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scripts.tournament_aggregate import run_aggregate  # reuse plotting routine if needed
from scripts.tournament import evaluate_learned_agent
from src.blackjack.env import BlackjackEnv
from src.blackjack.agent import BetAwareQLearningAgent
from src.blackjack.train import train_bet
from scripts.simulate_counting import basic_policy, flat_bet, count_bet, simulate_strategy

REPORT_DIR = "reports"
REPORT_FILE = os.path.join(REPORT_DIR, "summary.html")


def _pad_histories(histories: List[List[float]]) -> np.ndarray:
    max_len = max(len(h) for h in histories)
    arr = np.zeros((len(histories), max_len), dtype=float)
    for i, h in enumerate(histories):
        arr[i, : len(h)] = h
        if len(h) < max_len:
            arr[i, len(h) :] = h[-1]
    return arr


def compute_aggregate_stats(repeats: int, train_episodes: int, eval_hands: int, seed: int) -> Tuple[dict, str]:
    """Run aggregate experiments and return numeric summary and path to aggregate plot."""
    learned_runs = []
    flat_runs = []
    count_runs = []

    for r in range(repeats):
        s = seed + r
        env = BlackjackEnv(n_decks=6, seed=s)
        agent = BetAwareQLearningAgent(bets=(1, 2, 4, 8), lr=0.1, gamma=0.99, epsilon=0.1, seed=s)
        train_bet(agent, env, episodes=train_episodes)
        learned = evaluate_learned_agent(agent, env, hands=eval_hands, seed=s)
        flat = simulate_strategy(BlackjackEnv(n_decks=6, seed=s), basic_policy, flat_bet(base=1), hands=eval_hands, seed=s)
        count = simulate_strategy(BlackjackEnv(n_decks=6, seed=s), basic_policy, count_bet(base=1, ramp=2), hands=eval_hands, seed=s)

        learned_runs.append(learned)
        flat_runs.append(flat)
        count_runs.append(count)

    l_arr = _pad_histories(learned_runs)
    f_arr = _pad_histories(flat_runs)
    c_arr = _pad_histories(count_runs)

    stats = {
        "learned_final_mean": float(l_arr[:, -1].mean()),
        "learned_final_std": float(l_arr[:, -1].std()),
        "learned_final_median": float(np.median(l_arr[:, -1])),
        "flat_final_mean": float(f_arr[:, -1].mean()),
        "flat_final_std": float(f_arr[:, -1].std()),
        "count_final_mean": float(c_arr[:, -1].mean()),
        "count_final_std": float(c_arr[:, -1].std()),
        "repeats": repeats,
        "train_episodes": train_episodes,
        "eval_hands": eval_hands,
        "seed": seed,
    }

    # produce aggregate plot (reuse tournament_aggregate behavior)
    from scripts.tournament_aggregate import run_aggregate as _run_aggregate_plot

    out_plot = f"plots/tournament_agg_{repeats}x{train_episodes}_{eval_hands}h.png"
    _run_aggregate_plot(repeats=repeats, train_episodes=train_episodes, eval_hands=eval_hands, seed=seed, out=out_plot)

    return stats, out_plot


def render_html(stats: dict, plot_path: str, project_lead: str = "lettucejuice04") -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>RL-blackjack: Aggregate Tournament Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th, td {{ padding: 8px 12px; border: 1px solid #ddd; text-align: left; }}
  </style>
</head>
<body>
  <h1>RL-blackjack — Aggregate Tournament Report</h1>
  <p><strong>Project lead:</strong> {project_lead} &nbsp;|&nbsp; <em>Assisted by GitHub Copilot (Raptor mini)</em></p>

  <h2>Experiment setup</h2>
  <ul>
    <li>Repeats: {stats['repeats']}</li>
    <li>Train episodes (per repeat): {stats['train_episodes']}</li>
    <li>Evaluation hands (per repeat): {stats['eval_hands']}</li>
    <li>Random seed base: {stats['seed']}</li>
  </ul>

  <h2>Final bankroll (per run) — summary</h2>
  <table>
    <thead><tr><th>Strategy</th><th>Final mean</th><th>Final std</th><th>Median</th></tr></thead>
    <tbody>
      <tr><td>Learned bet-aware</td><td>{stats['learned_final_mean']:.2f}</td><td>{stats['learned_final_std']:.2f}</td><td>{stats['learned_final_median']:.2f}</td></tr>
      <tr><td>Flat-bet baseline</td><td>{stats['flat_final_mean']:.2f}</td><td>{stats['flat_final_std']:.2f}</td><td>-</td></tr>
      <tr><td>Count-based ramp</td><td>{stats['count_final_mean']:.2f}</td><td>{stats['count_final_std']:.2f}</td><td>-</td></tr>
    </tbody>
  </table>

  <h2>Aggregate plot</h2>
  <p>The plot below shows mean bankroll (with shaded ±1 std) across runs for each strategy.</p>
  <img src="../{plot_path}" alt="Aggregate tournament plot" style="max-width:100%;height:auto;">

  <hr/>
  <p>Generated by a short automated report generator. For reproducibility, see <code>scripts/tournament_aggregate.py</code> and <code>scripts/generate_report.py</code>.</p>
</body>
</html>
"""

    out = REPORT_FILE
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--train-episodes", type=int, default=2000)
    parser.add_argument("--eval-hands", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lead", type=str, default="lettucejuice04")
    args = parser.parse_args()

    stats, plot = compute_aggregate_stats(args.repeats, args.train_episodes, args.eval_hands, args.seed)
    report_path = render_html(stats, plot, project_lead=args.lead)
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
