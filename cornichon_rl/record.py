"""Film a policy playing the actual game: real sprites, the game's camera and HUD, 60 fps mp4.

Runs the recorder (recorder/, the training simulator inside a libGDX window) and drives it with the policy exactly
like evaluation does. Needs Java, ffmpeg and a display; with no $DISPLAY it starts one with xvfb-run.

  python -m cornichon_rl.record --checkpoint weights/cornichon_fighter.pt --difficulty 2 --seeds 1000000 --out-dir videos/
"""

import argparse
import contextlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch

from .curriculum import eval_seeds
from .env import observe, time_limit
from .evaluate import load_policy
from .model import to_tensors
from .service import RECORDER, SimService


@contextlib.contextmanager
def display(width, height):
    """$DISPLAY if there is one, else a private Xvfb server for as long as the block runs. (Not xvfb-run: it merges
    the program's stderr into stdout, which carries the protocol.)"""
    if os.environ.get("DISPLAY"):
        yield os.environ["DISPLAY"]
        return
    if not shutil.which("Xvfb"):
        raise RuntimeError("no $DISPLAY and no Xvfb (sudo apt install xvfb)")
    read, write = os.pipe()
    server = subprocess.Popen(
        ["Xvfb", "-displayfd", str(write), "-screen", "0", f"{width}x{height}x24", "-nolisten", "tcp"],
        pass_fds=(write,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.close(write)
    try:
        with os.fdopen(read) as f:
            number = f.readline().strip()  # Xvfb picks a free display and writes its number here
        if not number:
            raise RuntimeError("Xvfb did not start")
        yield f":{number}"
    finally:
        server.terminate()
        server.wait()


def film(model, env_config, seed, difficulty, out, device="cpu", greedy=False, width=1280, height=720):
    """Play one level and write it to `out` (.mp4). Returns the final simulator row."""
    if not RECORDER.exists():
        raise FileNotFoundError(f"{RECORDER} missing; build it with ./gradlew :recorder:installDist")
    with display(width, height) as screen:
        return _film(model, env_config, seed, difficulty, out, device, greedy, width, height, screen)


def _film(model, env_config, seed, difficulty, out, device, greedy, width, height, screen):
    service = SimService(
        1, env_config.view_width, env_config.view_height, launcher=RECORDER,
        extra_args=(f"--out={Path(out).resolve()}", f"--width={width}", f"--height={height}"),
        env={**os.environ, "DISPLAY": screen},
    )
    try:
        row = service.reset([0], [seed], [difficulty])[0]
        memory, starts = model.initial_state(1, device), torch.ones(1, device=device)
        limit = time_limit(row["door_distance"], env_config.max_steps, env_config.time_base, env_config.time_per_tile)
        for step in range(limit):
            if row["completed"] or row["dead"]:
                break
            obs = observe(row, step, limit, env_config.view_height, env_config.view_width)
            action, _, _, memory = model.act(to_tensors({k: v[None] for k, v in obs.items()}, device), memory, starts, greedy=greedy)
            starts = torch.zeros(1, device=device)
            service.send_step([action[0].cpu().numpy()], env_config.repeat)
            row = service.read_step()[0]
    finally:
        service.close()  # finalizes the mp4
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=eval_seeds(3))
    parser.add_argument("--greedy", action="store_true", help="argmax actions (default: sample, as in evaluation)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--out-dir", default="videos")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, env_config, _ = load_policy(args.checkpoint, "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(args.checkpoint).stem
    for seed in args.seeds:
        out = out_dir / f"{name}_d{args.difficulty}_seed{seed}.mp4"
        row = film(model, env_config, seed, args.difficulty, out, greedy=args.greedy, width=args.width, height=args.height)
        outcome = "door" if row["completed"] else "died" if row["dead"] else "timeout"
        print(json.dumps({"video": str(out), "seed": seed, "difficulty": args.difficulty, "outcome": outcome,
                          "decisions": row["frames"] // env_config.repeat, "mobs_killed": row["mobs_killed"],
                          "mobs_total": row["mobs_total"], "health": row["health"]}), flush=True)


if __name__ == "__main__":
    main()
