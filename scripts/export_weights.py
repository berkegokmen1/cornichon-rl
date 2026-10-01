#!/usr/bin/env python3
"""Turn a training checkpoint into a small release file: model weights + the config needed to rebuild the policy,
without the optimizer and RNG state. Writes NAME.pt and a model card NAME.json (where it came from and how it
scored), which evaluate.py / render.py load like any checkpoint.

  python scripts/export_weights.py RUN_DIR/latest.pt weights/NAME --run ppo_v4_combat --wandb 7u3j82aa \
      --benchmark bench/NAME.json
"""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("out", help="output path without extension")
    parser.add_argument("--run", required=True, help="training run name (EXPERIMENTS.md)")
    parser.add_argument("--wandb", default="", help="W&B run id")
    parser.add_argument("--commit", default="", help="commit the run was trained at")
    parser.add_argument("--notes", default="")
    parser.add_argument("--benchmark", help="evaluate.py --out JSON to copy into the model card")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": checkpoint["model"], "config": checkpoint["config"]}, out.with_suffix(".pt"))
    card = {
        "run": args.run,
        "wandb": f"berkegokmen1/cornichon-rl/runs/{args.wandb}" if args.wandb else "",
        "commit": args.commit,
        "update": checkpoint.get("update"),
        "env_steps": checkpoint.get("env_steps"),
        "curriculum_difficulty": checkpoint.get("curriculum", {}).get("difficulty"),
        "grid_channels": checkpoint["model"]["grid.0.weight"].shape[1],
        "state_size": checkpoint["model"]["state.0.weight"].shape[1],
        "notes": args.notes,
    }
    if args.benchmark:
        results = json.loads(Path(args.benchmark).read_text())
        card["benchmark"] = {
            "what": "held-out levels (seeds from 1,000,000), sampled actions, evaluate.py --stochastic",
            "per_difficulty": [
                {k: r["summary"][k] for k in ("difficulty", "episodes", "success", "death", "timeout", "mobs_killed", "kill_fraction")}
                for r in results
            ],
        }
    out.with_suffix(".json").write_text(json.dumps(card, indent=1) + "\n")
    print(f"wrote {out.with_suffix('.pt')} ({out.with_suffix('.pt').stat().st_size / 1e6:.1f} MB) and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
