#!/usr/bin/env python3
"""Why do episodes time out? Plays a checkpoint on held-out mazes, records every step, and classifies each timeout:

  stuck_below_climb  last 150 steps within a 3x3-tile box, and the shortest path's next tile is upward
  stuck_other        same box, path does not go up there (blocked sideways, corner, mob)
  wandering          moving, but ends no closer to the door than its best point

For stuck episodes it also reports how often the sphere sat on the player's head (blocks jumps) and how often the
policy pressed jump.

  python scripts/diagnose_timeouts.py --checkpoint RUN_DIR/best.pt --difficulty 2 --episodes 100
"""

import argparse
import base64
import json
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cornichon_rl.curriculum import eval_seeds  # noqa: E402
from cornichon_rl.env import observe, time_limit  # noqa: E402
from cornichon_rl.evaluate import load_policy  # noqa: E402
from cornichon_rl.model import to_tensors  # noqa: E402
from cornichon_rl.service import SimService  # noqa: E402

TAIL = 150


def door_field(snapshot):
    h, w = snapshot["height"], snapshot["width"]
    grid = np.frombuffer(base64.b64decode(snapshot["grid"]), np.uint8).reshape(h, w)
    wall = (grid & 1) > 0
    door = tuple(np.argwhere(grid & 128)[0])
    dist = np.full((h, w), np.inf)
    dist[door] = 0
    queue = deque([door])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not wall[rr, cc] and dist[rr, cc] == np.inf:
                dist[rr, cc] = dist[r, c] + 1
                queue.append((rr, cc))
    return dist


def path_goes_up(dist, top_y, x, y):
    r, c = top_y - round(y), round(x)
    if not (0 <= r < dist.shape[0] and 0 <= c < dist.shape[1]) or not np.isfinite(dist[r, c]):
        return False
    best = min(((dist[r + dr, c + dc], dr) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= r + dr < dist.shape[0] and 0 <= c + dc < dist.shape[1]), default=(np.inf, 0))
    return best[0] < dist[r, c] and best[1] == -1


def play(model, env_config, difficulty, seeds, batch=32):
    traces = []
    for start in range(0, len(seeds), batch):
        chunk = seeds[start : start + batch]
        service = SimService(len(chunk), env_config.view_width, env_config.view_height)
        rows = service.reset(range(len(chunk)), chunk, [difficulty] * len(chunk))
        snaps = [service.snapshot(i) for i in range(len(chunk))]
        limits = [time_limit(r["door_distance"], env_config.max_steps, env_config.time_base, env_config.time_per_tile) for r in rows]
        steps = [[] for _ in chunk]
        done = [False] * len(chunk)
        memory, starts = model.initial_state(len(chunk), "cpu"), torch.ones(len(chunk))
        for t in range(max(limits)):
            obs = [observe(r, t, lim, env_config.view_height, env_config.view_width) for r, lim in zip(rows, limits)]
            obs = {k: np.stack([o[k] for o in obs]) for k in ("grid", "state")}
            actions, _, _, memory = model.act(to_tensors(obs, "cpu"), memory, starts, greedy=True)
            actions, starts = actions.numpy(), torch.zeros(len(chunk))
            service.send_step(actions, env_config.repeat)
            new_rows = service.read_step()
            for i, (row, action) in enumerate(zip(new_rows, actions)):
                if done[i]:
                    continue
                steps[i].append({"x": row["x"], "y": row["y"], "d": row["door_distance"], "jump": int(action[1]),
                                 "sphere": (row["state"][5] * 5, row["state"][6] * 5), "grounded": row["state"][4] > 0.5})
                if row["completed"] or row["dead"] or t + 1 >= limits[i]:
                    done[i] = True
                    outcome = "success" if row["completed"] else "death" if row["dead"] else "timeout"
                    traces.append({"seed": chunk[i], "outcome": outcome, "steps": steps[i], "snapshot": snaps[i]})
            rows = new_rows
            if all(done):
                break
        service.close()
    return traces


def classify(trace):
    steps = trace["steps"]
    tail = steps[-TAIL:]
    xs, ys = [s["x"] for s in tail], [s["y"] for s in tail]
    distances = [s["d"] for s in steps if s["d"] < 2**31 - 1]
    info = {"best_distance": min(distances) if distances else None, "end_distance": steps[-1]["d"]}
    if max(xs) - min(xs) < 3 and max(ys) - min(ys) < 3:
        dist = door_field(trace["snapshot"])
        up = path_goes_up(dist, trace["snapshot"]["top_y"], np.median(xs), np.median(ys))
        on_head = np.mean([abs(s["sphere"][0]) < 0.3 and 0.3 < s["sphere"][1] < 0.9 for s in tail])
        info.update({"kind": "stuck_below_climb" if up else "stuck_other", "sphere_on_head": float(on_head),
                     "jump_rate": float(np.mean([s["jump"] for s in tail])),
                     "grounded_rate": float(np.mean([s["grounded"] for s in tail]))})
    else:
        info["kind"] = "wandering"
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--difficulty", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, env_config, _ = load_policy(args.checkpoint, "cpu")
    traces = play(model, env_config, args.difficulty, eval_seeds(args.episodes))
    outcomes = Counter(t["outcome"] for t in traces)
    timeouts = [dict(seed=t["seed"], **classify(t)) for t in traces if t["outcome"] == "timeout"]
    kinds = Counter(t["kind"] for t in timeouts)
    print("outcomes:", dict(outcomes))
    print("timeouts by kind:", dict(kinds))
    for kind in kinds:
        group = [t for t in timeouts if t["kind"] == kind]
        line = {k: round(float(np.mean([g[k] for g in group])), 2) for k in ("sphere_on_head", "jump_rate", "grounded_rate") if k in group[0]}
        print(f"  {kind}: n={len(group)} {line} end_distance(median)={np.median([g['end_distance'] for g in group]):.0f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"outcomes": dict(outcomes), "timeouts": timeouts}, f, indent=1)


if __name__ == "__main__":
    main()
