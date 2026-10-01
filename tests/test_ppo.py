import subprocess
import sys

import torch

from cornichon_rl.ppo import compute_gae


def reference_gae(rewards, values, last_value, dones, gamma, lam):
    T, N = rewards.shape
    adv = torch.zeros(T, N)
    for n in range(N):
        future = 0.0
        for t in reversed(range(T)):
            next_value = last_value[n] if t == T - 1 else values[t + 1, n]
            if dones[t, n]:
                next_value, future = 0.0, 0.0
            delta = rewards[t, n] + gamma * next_value - values[t, n]
            future = delta + gamma * lam * future
            adv[t, n] = future
    return adv


def test_gae_matches_reference():
    torch.manual_seed(0)
    T, N = 17, 5
    rewards, values, last = torch.randn(T, N), torch.randn(T, N), torch.randn(N)
    dones = (torch.rand(T, N) < 0.2).float()
    advantages, returns = compute_gae(rewards, values, last, dones, 0.99, 0.95)
    assert torch.allclose(advantages, reference_gae(rewards, values, last, dones, 0.99, 0.95), atol=1e-5)
    assert torch.allclose(returns, advantages + values)


def test_smoke_training_writes_checkpoints_and_resumes(tmp_path):
    common = [sys.executable, "-m", "cornichon_rl.ppo", "--config", "configs/smoke.yaml", "--name", "t", f"out_dir={tmp_path}"]
    subprocess.run(common, check=True, timeout=600)
    first = torch.load(tmp_path / "t" / "latest.pt", weights_only=False)
    assert first["update"] == 4 and (tmp_path / "t" / "best.pt").exists()
    subprocess.run(common + ["--resume", "ppo.total_steps=1536"], check=True, timeout=600)
    assert torch.load(tmp_path / "t" / "latest.pt", weights_only=False)["update"] == 6


def test_recurrent_smoke_training_and_resume(tmp_path):
    common = [sys.executable, "-m", "cornichon_rl.ppo", "--config", "configs/smoke.yaml", "--name", "r",
              f"out_dir={tmp_path}", "model.recurrent=true"]
    subprocess.run(common, check=True, timeout=600)
    subprocess.run(common + ["--resume", "ppo.total_steps=1536"], check=True, timeout=600)
    checkpoint = torch.load(tmp_path / "r" / "latest.pt", weights_only=False)
    assert checkpoint["update"] == 6 and checkpoint["config"]["model"]["recurrent"] is True
    assert any("lstm" in key for key in checkpoint["model"])
