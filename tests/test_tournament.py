import os
from scripts.tournament import run_tournament


def test_run_tournament_creates_plot(tmp_path):
    out = tmp_path / "tournament_test.png"
    # use small counts to keep test fast
    run_tournament(train_episodes=50, eval_hands=200, seed=0, out=str(out))
    assert out.exists()
    assert out.stat().st_size > 0
