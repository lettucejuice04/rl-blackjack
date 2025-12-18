# RL-blackjack

This project contains a reinforcement learning agent to learn Blackjack and experiment with card-counting strategies using deep RL.

## Quickstart

1. Create and activate the venv (from project root `C:\Projects\RL-blackjack`):

PowerShell:
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\.venv\Scripts\Activate.ps1
# or simply:
.\.venv\Scripts\Activate.ps1
```

cmd.exe:
```cmd
.\.venv\Scripts\activate.bat
```

Git Bash / Unix-like shell:
```bash
source .venv/Scripts/activate
```

2. Install project dependencies
```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

3. Suggested steps
- Add environments and an initial training script using `stable-baselines3` (PPO or DQN are good starting points).
- Add logging to TensorBoard for monitoring training progress.
- Implement environment wrappers to provide observations (counts, deck state) suitable for learning.

## Notes
- Keep source code outside of `.venv/` and add `.venv/` to `.gitignore`.
- If you have an NVIDIA GPU and want faster training, install a CUDA-enabled PyTorch build matching your CUDA driver.
