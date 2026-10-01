"""PPO on the real Cornichon game, with a difficulty curriculum and held-out evaluation.

  python -m cornichon_rl.ppo --config configs/default.yaml --name ppo_v1
  python -m cornichon_rl.ppo --config configs/default.yaml --name ppo_v1 --resume   # continue RUN_DIR/latest.pt
  python -m cornichon_rl.ppo --config configs/smoke.yaml ppo.total_steps=4096 wandb_mode=disabled
"""

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import wandb

from .config import load_config, to_dict
from .curriculum import Curriculum, eval_seeds
from .env import VecCornichon
from .evaluate import evaluate
from .model import ActorCritic, to_tensors


def compute_gae(rewards, values, last_value, dones, gamma, lam):
    """Advantages and returns for a (T, N) rollout.

    dones[t] marks that the episode ended after step t (terminated or truncated). Truncated steps must already
    carry gamma * V(final observation) in their reward, so the cut is bootstrapped rather than treated as a death.
    """
    horizon = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    for t in reversed(range(horizon)):
        next_value = last_value if t == horizon - 1 else values[t + 1]
        alive = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * alive - values[t]
        running = delta + gamma * lam * alive * running
        advantages[t] = running
    return advantages, advantages + values


def ppo_update(model, optimizer, batch, config):
    """A few epochs of clipped PPO over one rollout. Returns averaged diagnostics."""
    size = batch["actions"].shape[0]
    minibatch = size // config.minibatches
    stats = {"policy": [], "value": [], "entropy": [], "total": [], "approx_kl": [], "clip_frac": []}
    for _ in range(config.epochs):
        order = torch.randperm(size, device=batch["actions"].device)
        for start in range(0, minibatch * config.minibatches, minibatch):
            idx = order[start : start + minibatch]
            dists, value = model.distributions({"grid": batch["grid"][idx], "state": batch["state"][idx]})
            actions = batch["actions"][idx]
            log_prob = sum(d.log_prob(actions[:, i]) for i, d in enumerate(dists))
            entropy = sum(d.entropy() for d in dists).mean()

            advantages = batch["advantages"][idx]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            log_ratio = log_prob - batch["log_probs"][idx]
            ratio = log_ratio.exp()
            policy_loss = -torch.min(ratio * advantages, ratio.clamp(1 - config.clip, 1 + config.clip) * advantages).mean()
            value_loss = 0.5 * (value - batch["returns"][idx]).pow(2).mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                stats["policy"].append(policy_loss.item())
                stats["value"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
                stats["total"].append(loss.item())
                stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                stats["clip_frac"].append(((ratio - 1).abs() > config.clip).float().mean().item())
    return {key: float(np.mean(values)) for key, values in stats.items()}


def explained_variance(predicted, target):
    variance = target.var()
    return float("nan") if variance == 0 else float(1 - (target - predicted).var() / variance)


def episode_metrics(episodes, prefix):
    if not episodes:
        return {}
    metrics = {f"{prefix}/episodes": len(episodes)}
    for key in ("return", "length", "success", "death", "timeout", "progress", "mobs_killed", "damage", "collected", "score"):
        metrics[f"{prefix}/{key}"] = float(np.mean([float(e[key]) for e in episodes]))
    for part in episodes[0]["parts"]:
        metrics[f"reward_parts/{part}"] = float(np.mean([e["parts"].get(part, 0.0) for e in episodes]))
    return metrics


def save_checkpoint(path, model, optimizer, update, env_steps, curriculum, config, wandb_id, best):
    tmp = Path(str(path) + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "env_steps": env_steps,
            "curriculum": curriculum.state_dict(),
            "config": to_dict(config),
            "wandb_id": wandb_id,
            "best": best,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
        },
        tmp,
    )
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--name", help="run name; checkpoints go to out_dir/name")
    parser.add_argument("--resume", action="store_true", help="continue from out_dir/name/latest.pt")
    parser.add_argument("overrides", nargs="*", help="key.sub=value, e.g. ppo.lr=1e-4 env.num_envs=32")
    args = parser.parse_intermixed_args()

    config = load_config(args.config, args.overrides)
    if args.name:
        config.name = args.name
    run_dir = Path(config.out_dir) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")

    torch.set_num_threads(config.torch_threads)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    curriculum = Curriculum(config.curriculum, np.random.default_rng(config.seed))
    model = ActorCritic(config.env.view_height, config.env.view_width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ppo.lr, eps=1e-5)

    update, env_steps, wandb_id = 0, 0, None
    best = {"difficulty": 0, "success": -1.0}
    if args.resume:
        checkpoint = torch.load(run_dir / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        curriculum.load_state_dict(checkpoint["curriculum"])
        update, env_steps, wandb_id, best = checkpoint["update"], checkpoint["env_steps"], checkpoint["wandb_id"], checkpoint["best"]
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        print(f"resumed {run_dir}/latest.pt at update {update}, {env_steps} env steps, difficulty {curriculum.difficulty}")
    with open(run_dir / "config.json", "w") as f:
        json.dump(to_dict(config), f, indent=1)

    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.name,
        id=wandb_id,
        resume="allow",
        mode=config.wandb_mode,
        dir=str(run_dir),
        config=to_dict(config),
    )
    wandb_id = run.id

    ppo, env_cfg = config.ppo, config.env
    n, horizon = env_cfg.num_envs, ppo.horizon
    total_updates = ppo.total_steps // (n * horizon)
    env = VecCornichon(
        n,
        lambda _env: curriculum.sample(),
        envs_per_service=env_cfg.envs_per_service,
        max_steps=env_cfg.max_steps,
        repeat=env_cfg.repeat,
        reward=config.reward_config(),
        view_width=env_cfg.view_width,
        view_height=env_cfg.view_height,
    )
    grid_shape = env.observation_space["grid"].shape
    state_shape = env.observation_space["state"].shape
    buffers = {
        "grid": torch.zeros((horizon, n, *grid_shape), device=device),
        "state": torch.zeros((horizon, n, *state_shape), device=device),
        "actions": torch.zeros((horizon, n, 5), dtype=torch.long, device=device),
        "log_probs": torch.zeros((horizon, n), device=device),
        "values": torch.zeros((horizon, n), device=device),
        "rewards": torch.zeros((horizon, n), device=device),
        "dones": torch.zeros((horizon, n), device=device),
    }
    recent = deque(maxlen=200)

    try:
        obs = to_tensors(env.reset(), device)
        while update < total_updates:
            update += 1
            if ppo.anneal_lr:
                optimizer.param_groups[0]["lr"] = ppo.lr * (1 - (update - 1) / total_updates)
            started = time.time()
            finished = []
            model.eval()
            for t in range(horizon):
                action, log_prob, value = model.act(obs)
                next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                reward = torch.as_tensor(reward, device=device)
                if info["final_obs"]:
                    envs = list(info["final_obs"])
                    final = {k: np.stack([info["final_obs"][e][k] for e in envs]) for k in ("grid", "state")}
                    with torch.no_grad():
                        _, final_value = model(to_tensors(final, device))
                    reward[envs] += ppo.gamma * final_value
                buffers["grid"][t] = obs["grid"]
                buffers["state"][t] = obs["state"]
                buffers["actions"][t] = action
                buffers["log_probs"][t] = log_prob
                buffers["values"][t] = value
                buffers["rewards"][t] = reward
                buffers["dones"][t] = torch.as_tensor(terminated | truncated, device=device, dtype=torch.float32)
                obs = to_tensors(next_obs, device)
                for episode in info["episodes"]:
                    finished.append(episode)
                    recent.append(episode)
                    if curriculum.record(episode["difficulty"], episode["success"]):
                        print(f"update {update}: curriculum promoted to difficulty {curriculum.difficulty}", flush=True)
            rollout_seconds = time.time() - started
            env_steps += n * horizon

            with torch.no_grad():
                _, last_value = model(obs)
            advantages, returns = compute_gae(
                buffers["rewards"], buffers["values"], last_value, buffers["dones"], ppo.gamma, ppo.gae_lambda
            )
            batch = {
                "grid": buffers["grid"].flatten(0, 1),
                "state": buffers["state"].flatten(0, 1),
                "actions": buffers["actions"].flatten(0, 1),
                "log_probs": buffers["log_probs"].flatten(0, 1),
                "advantages": advantages.flatten(),
                "returns": returns.flatten(),
            }
            model.train()
            losses = ppo_update(model, optimizer, batch, ppo)

            metrics = {f"loss/{k}": v for k, v in losses.items() if k in ("policy", "value", "entropy", "total")}
            metrics.update({
                "ppo/approx_kl": losses["approx_kl"],
                "ppo/clip_frac": losses["clip_frac"],
                "ppo/explained_variance": explained_variance(buffers["values"].flatten(), returns.flatten()),
                "ppo/lr": optimizer.param_groups[0]["lr"],
                "ppo/advantage_std": float(advantages.std()),
                "curriculum/difficulty": curriculum.difficulty,
                "curriculum/success_window": curriculum.success_rate(),
                "system/env_steps": env_steps,
                "system/update": update,
                "system/sps": n * horizon / (time.time() - started),
                "system/rollout_sps": n * horizon / rollout_seconds,
            })
            metrics.update(episode_metrics(finished, "rollout"))
            metrics.update({k.replace("rollout/", "rollout_window/"): v for k, v in episode_metrics(list(recent), "rollout").items() if k.startswith("rollout/")})

            if update % config.eval.every_updates == 0 or update == total_updates:
                model.eval()
                summary, _ = evaluate(
                    model, curriculum.difficulty, eval_seeds(config.eval.episodes), env_cfg,
                    config.reward_config(), device, greedy=config.eval.greedy,
                )
                metrics.update({f"eval/{k}": v for k, v in summary.items()})
                score = (summary["difficulty"], summary["success"])
                if score > (best["difficulty"], best["success"]):
                    best = {"difficulty": summary["difficulty"], "success": summary["success"], "update": update}
                    save_checkpoint(run_dir / "best.pt", model, optimizer, update, env_steps, curriculum, config, wandb_id, best)
                print(f"update {update} eval: {json.dumps(summary)}", flush=True)

            wandb.log(metrics, step=env_steps)
            print(
                f"update {update}/{total_updates} steps {env_steps} sps {metrics['system/sps']:.0f} "
                f"difficulty {curriculum.difficulty} success(window) {curriculum.success_rate():.2f} "
                f"return {metrics.get('rollout/return', float('nan')):.2f} kl {losses['approx_kl']:.4f}",
                flush=True,
            )
            if update % config.checkpoint_every == 0 or update == total_updates:
                save_checkpoint(run_dir / "latest.pt", model, optimizer, update, env_steps, curriculum, config, wandb_id, best)
    finally:
        env.close()
        run.finish()


if __name__ == "__main__":
    main()
