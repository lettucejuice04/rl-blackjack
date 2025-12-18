from pathlib import Path
from scripts.tournament_aggregate import run_aggregate


def test_aggregate_creates_plot(tmp_path):
    out = tmp_path / "tournament_agg.png"
    # use small numbers to keep test time reasonable
    run_aggregate(repeats=2, train_episodes=20, eval_hands=50, seed=0, out=str(out))
    assert out.exists()
    assert out.stat().st_size > 0
