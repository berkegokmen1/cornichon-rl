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
from .evaluate import evaluate, sweep
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


def _minibatches(model, batch, count):
    """Yield (logits, values, flat index) per minibatch. A recurrent model needs whole sequences, so it splits by env;
    a feed-forward model splits individual steps."""
    steps, envs = batch["starts"].shape
    obs = {"grid": batch["grid"], "state": batch["state"]}
    if model.recurrent:
        order = torch.randperm(envs, device=batch["starts"].device)
        size = envs // count
        for start in range(0, size * count, size):
            idx = order[start : start + size]
            state = (batch["state0"][0][idx], batch["state0"][1][idx])
            logits, values = model.forward_sequence({k: v[:, idx] for k, v in obs.items()}, state, batch["starts"][:, idx])
            flat = (torch.arange(steps, device=idx.device).unsqueeze(1) * envs + idx.unsqueeze(0)).flatten()
            yield logits, values, flat
    else:
        flat_obs = {k: v.flatten(0, 1) for k, v in obs.items()}
        order = torch.randperm(steps * envs, device=batch["starts"].device)
        size = steps * envs // count
        for start in range(0, size * count, size):
            idx = order[start : start + size]
            logits, values, _ = model({k: v[idx] for k, v in flat_obs.items()})
            yield logits, values, idx


def ppo_update(model, optimizer, batch, config):
    """A few epochs of clipped PPO over one (T, N) rollout. Returns averaged diagnostics."""
    flat = {k: batch[k].flatten(0, 1) for k in ("actions", "log_probs", "advantages", "returns")}
    stats = {"policy": [], "value": [], "entropy": [], "total": [], "approx_kl": [], "clip_frac": []}
    for _ in range(config.epochs):
        for logits, value, idx in _minibatches(model, batch, config.minibatches):
            dists = [torch.distributions.Categorical(logits=l) for l in logits]
            actions = flat["actions"][idx]
            log_prob = sum(d.log_prob(actions[:, i]) for i, d in enumerate(dists))
            entropy = sum(d.entropy() for d in dists).mean()

            advantages = flat["advantages"][idx]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            log_ratio = log_prob - flat["log_probs"][idx]
            ratio = log_ratio.exp()
            policy_loss = -torch.min(ratio * advantages, ratio.clamp(1 - config.clip, 1 + config.clip) * advantages).mean()
            value_loss = 0.5 * (value - flat["returns"][idx]).pow(2).mean()
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
    for key in ("return", "length", "time_limit", "success", "death", "timeout", "progress", "mobs_killed", "damage_dealt", "damage", "collected", "score"):
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
    model = ActorCritic(
        config.env.view_height, config.env.view_width, hidden=config.model.hidden, recurrent=config.model.recurrent
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ppo.lr, eps=1e-5)

    update, env_steps, wandb_id = 0, 0, None
    best = {"difficulty": 0, "success": -1.0}
    if args.resume:
        checkpoint = torch.load(run_dir / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        curriculum.load_state_dict(checkpoint["curriculum"])
        update, env_steps, wandb_id, best = checkpoint["update"], checkpoint["env_steps"], checkpoint["wandb_id"], checkpoint["best"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())  # map_location moved it to the GPU
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
    env = VecCornichon.from_config(n, lambda _env: curriculum.sample(), env_cfg, config.reward_config())
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
        "starts": torch.zeros((horizon, n), device=device),  # env begins a new episode at this step (resets memory)
    }
    recent = deque(maxlen=200)

    try:
        obs = to_tensors(env.reset(), device)
        memory = model.initial_state(n, device)
        starts = torch.ones(n, device=device)
        while update < total_updates:
            update += 1
            if ppo.anneal_lr:
                optimizer.param_groups[0]["lr"] = ppo.lr * (1 - (update - 1) / total_updates)
            started = time.time()
            finished = []
            model.eval()
            memory0 = None if memory is None else (memory[0].clone(), memory[1].clone())
            for t in range(horizon):
                buffers["starts"][t] = starts
                action, log_prob, value, memory = model.act(obs, memory, starts)
                next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                reward = torch.as_tensor(reward, device=device)
                if info["final_obs"]:
                    envs = list(info["final_obs"])
                    final = {k: np.stack([info["final_obs"][e][k] for e in envs]) for k in ("grid", "state")}
                    final_memory = None if memory is None else (memory[0][envs], memory[1][envs])
                    with torch.no_grad():
                        _, final_value, _ = model(to_tensors(final, device), final_memory, torch.zeros(len(envs), device=device))
                    reward[envs] += ppo.gamma * final_value
                buffers["grid"][t] = obs["grid"]
                buffers["state"][t] = obs["state"]
                buffers["actions"][t] = action
                buffers["log_probs"][t] = log_prob
                buffers["values"][t] = value
                buffers["rewards"][t] = reward
                buffers["dones"][t] = torch.as_tensor(terminated | truncated, device=device, dtype=torch.float32)
                starts = buffers["dones"][t]
                obs = to_tensors(next_obs, device)
                for episode in info["episodes"]:
                    finished.append(episode)
                    recent.append(episode)
                    if curriculum.record(episode["difficulty"], episode["success"]):
                        print(f"update {update}: curriculum promoted to difficulty {curriculum.difficulty}", flush=True)
            rollout_seconds = time.time() - started
            env_steps += n * horizon

            with torch.no_grad():
                _, last_value, _ = model(obs, memory, starts)
            advantages, returns = compute_gae(
                buffers["rewards"], buffers["values"], last_value, buffers["dones"], ppo.gamma, ppo.gae_lambda
            )
            batch = {
                key: buffers[key] for key in ("grid", "state", "actions", "log_probs", "starts")
            }
            batch.update({"advantages": advantages, "returns": returns, "state0": memory0})
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

            if update % config.eval.sweep_every_updates == 0 or update == total_updates:
                model.eval()
                top = min(config.curriculum.end, curriculum.difficulty + 1)
                by_difficulty = sweep(
                    model, range(config.curriculum.start, top + 1), eval_seeds(config.eval.sweep_episodes), env_cfg,
                    config.reward_config(), device, greedy=config.eval.greedy,
                )
                for d, summary in by_difficulty.items():
                    for key in ("success", "death", "timeout", "progress", "length"):
                        metrics[f"eval_sweep/d{d:02d}/{key}"] = summary[key]
                print(f"update {update} sweep success: " + " ".join(f"d{d}={s['success']:.2f}" for d, s in by_difficulty.items()), flush=True)

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
