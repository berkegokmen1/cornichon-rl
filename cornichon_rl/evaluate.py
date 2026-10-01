"""Play held-out levels and report completion. Also the random-policy baseline.

  python -m cornichon_rl.evaluate --checkpoint RUN_DIR/best.pt --difficulties 1 2 3 --episodes 100
  python -m cornichon_rl.evaluate --policy random --difficulties 1 2 3
"""

import argparse
import json

import numpy as np
import torch

from .config import EnvConfig
from .curriculum import EVAL_SEED_START, eval_seeds
from .env import VecCornichon
from .model import ActorCritic, to_tensors
from .reward import RewardConfig
from .service import ACTION_NVEC

FILLER_SEED = EVAL_SEED_START - 1  # keeps idle envs busy once every evaluation seed has been handed out


def evaluate(policy, difficulty, seeds, env_config, reward=RewardConfig(), device="cpu", greedy=True, max_envs=32):
    """Plays each seed once at `difficulty`. policy is an ActorCritic or "random". Returns (summary, episodes)."""
    queue = list(seeds)
    wanted = set(seeds)

    def next_level(_env):
        return difficulty, (queue.pop(0) if queue else FILLER_SEED)

    env = VecCornichon(
        min(max_envs, len(seeds)),
        next_level,
        envs_per_service=env_config.envs_per_service,
        max_steps=env_config.max_steps,
        repeat=env_config.repeat,
        reward=reward,
        view_width=env_config.view_width,
        view_height=env_config.view_height,
    )
    rng = np.random.default_rng(0)
    episodes = {}
    try:
        obs = env.reset()
        while len(episodes) < len(wanted):
            if policy == "random":
                actions = np.stack([rng.integers(n, size=env.num_envs) for n in ACTION_NVEC], 1)
            else:
                actions = policy.act(to_tensors(obs, device), greedy=greedy)[0].cpu().numpy()
            obs, _, _, _, info = env.step(actions)
            for episode in info["episodes"]:
                if episode["seed"] in wanted:
                    episodes.setdefault(episode["seed"], episode)
    finally:
        env.close()
    episodes = [episodes[s] for s in seeds]
    summary = {
        "difficulty": difficulty,
        "episodes": len(episodes),
        "success": float(np.mean([e["success"] for e in episodes])),
        "death": float(np.mean([e["death"] for e in episodes])),
        "timeout": float(np.mean([e["timeout"] for e in episodes])),
        "return": float(np.mean([e["return"] for e in episodes])),
        "length": float(np.mean([e["length"] for e in episodes])),
        "mobs_killed": float(np.mean([e["mobs_killed"] for e in episodes])),
        "score": float(np.mean([e["score"] for e in episodes])),
    }
    return summary, episodes


def load_policy(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    env_config = EnvConfig(**checkpoint["config"]["env"])
    model = ActorCritic(env_config.view_height, env_config.view_width).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, env_config, RewardConfig.from_dict(checkpoint["config"]["reward"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint")
    parser.add_argument("--policy", choices=["checkpoint", "random"], default="checkpoint")
    parser.add_argument("--difficulties", type=int, nargs="+", default=[1])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the argmax")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", help="write per-difficulty summaries and every episode as JSON here")
    args = parser.parse_args()
    torch.set_num_threads(1)

    if args.policy == "random":
        policy, env_config, reward = "random", EnvConfig(), RewardConfig()
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required unless --policy random")
        policy, env_config, reward = load_policy(args.checkpoint, args.device)

    results = []
    for difficulty in args.difficulties:
        summary, episodes = evaluate(
            policy, difficulty, eval_seeds(args.episodes), env_config, reward, args.device, greedy=not args.stochastic
        )
        summary["policy"] = args.policy if args.policy == "random" else args.checkpoint
        if args.policy != "random":
            summary["mode"] = "stochastic" if args.stochastic else "greedy"
        print(json.dumps(summary), flush=True)
        results.append({"summary": summary, "episodes": episodes})
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
