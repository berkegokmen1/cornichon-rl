"""Record a policy playing held-out mazes as a GIF (top-down view of the whole level, drawn from the simulator).

  python -m cornichon_rl.render --checkpoint RUN_DIR/best.pt --seeds 1000000 1000001 --out demo.gif
"""

import argparse
import base64

import numpy as np
import torch
from PIL import Image, ImageDraw

from .curriculum import eval_seeds
from .env import observe, time_limit
from .evaluate import load_policy
from .model import to_tensors
from .service import SimService

CELL = 24
COLORS = {  # bit -> RGB, drawn in this order (later wins)
    1: (70, 70, 80),  # wall
    2: (200, 200, 210),  # spikes
    32: (220, 60, 60),  # health potion
    64: (60, 110, 230),  # mana potion
    4: (60, 170, 70),  # mob
    8: (150, 70, 200),  # wizard
    16: (255, 140, 0),  # fireball
    128: (150, 95, 40),  # door
}


def draw(snapshot, caption):
    w, h = snapshot["width"], snapshot["height"]
    grid = np.frombuffer(base64.b64decode(snapshot["grid"]), np.uint8).reshape(h, w)
    image = Image.new("RGB", (w * CELL, h * CELL + 22), (245, 243, 236))
    pen = ImageDraw.Draw(image)
    for bit, color in COLORS.items():
        for r, c in np.argwhere(grid & bit):
            pen.rectangle([c * CELL, r * CELL, (c + 1) * CELL - 1, (r + 1) * CELL - 1], fill=color)

    def at(x, y):
        return (x + 0.5) * CELL, (snapshot["top_y"] - y + 0.5) * CELL

    px, py = at(*snapshot["player"])
    pen.rectangle([px - 4, py - 6, px + 4, py + 6], fill=(40, 160, 60), outline=(0, 0, 0))  # the pickle
    sx, sy = at(*snapshot["sphere"])
    pen.ellipse([sx - 4, sy - 4, sx + 4, sy + 4], fill=(170, 60, 220))
    pen.text((4, h * CELL + 5), caption, fill=(20, 20, 20))
    return image


def record(model, env_config, seed, difficulty, device, greedy=True, max_steps=None):
    service = SimService(1, env_config.view_width, env_config.view_height)
    frames = []
    try:
        row = service.reset([0], [seed], [difficulty])[0]
        memory, starts = model.initial_state(1, device), torch.ones(1, device=device)
        limit = max_steps or time_limit(row["door_distance"], env_config.max_steps, env_config.time_base, env_config.time_per_tile)
        for step in range(limit):
            outcome = "door!" if row["completed"] else "died" if row["dead"] else ""
            caption = f"seed {seed}  difficulty {difficulty}  step {step}  HP {row['health']:.0f}  {outcome}"
            frames.append(draw(service.snapshot(0), caption))
            if row["completed"] or row["dead"]:
                break
            obs = observe(row, step, limit, env_config.view_height, env_config.view_width)
            obs = {k: v[None] for k, v in obs.items()}
            action, _, _, memory = model.act(to_tensors(obs, device), memory, starts, greedy=greedy)
            action, starts = action[0].cpu().numpy(), torch.zeros(1, device=device)
            service.send_step([action], env_config.repeat)
            row = service.read_step()[0]
    finally:
        service.close()
    return frames, row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=eval_seeds(4))
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--fps", type=int, default=15, help="15 = real time (one frame per decision)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, env_config, _ = load_policy(args.checkpoint, "cpu")
    frames = []
    for seed in args.seeds:
        clip, row = record(model, env_config, seed, args.difficulty, "cpu", greedy=not args.stochastic)
        frames += clip + [clip[-1]] * args.fps  # hold the last frame for a second
        print(f"seed {seed}: {'success' if row['completed'] else 'death' if row['dead'] else 'timeout'} in {len(clip) - 1} steps")
    size = max(f.size for f in frames)
    frames = [f if f.size == size else _pad(f, size) for f in frames]
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=1000 // args.fps, loop=0, optimize=True)
    print(f"wrote {args.out} ({len(frames)} frames)")


def _pad(frame, size):
    canvas = Image.new("RGB", size, (245, 243, 236))
    canvas.paste(frame, (0, 0))
    return canvas


if __name__ == "__main__":
    main()
