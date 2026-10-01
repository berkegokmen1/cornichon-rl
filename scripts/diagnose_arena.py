#!/usr/bin/env python3
"""Why do arena levels time out? Plays a checkpoint on held-out arenas (door closed) and, for every timeout, lists
the mobs still alive: kind (melee mob or wizard) and shortest-path distance from the player.

  python scripts/diagnose_arena.py --checkpoint RUN_DIR/best.pt --monsters 2 --episodes 60
"""

import argparse
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cornichon_rl.curriculum import eval_seeds  # noqa: E402
from cornichon_rl.env import observe  # noqa: E402
from cornichon_rl.evaluate import load_policy  # noqa: E402
from cornichon_rl.model import to_tensors  # noqa: E402
from cornichon_rl.service import SimService, grid_cells  # noqa: E402

MOB, WIZARD, WALL = 4, 8, 1


def distances_from(cells, start):
    h, w = cells.shape
    dist = np.full((h, w), np.inf)
    dist[start] = 0
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not cells[rr, cc] & WALL and dist[rr, cc] == np.inf:
                dist[rr, cc] = dist[r, c] + 1
                queue.append((rr, cc))
    return dist


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--maze", type=int, default=1)
    parser.add_argument("--monsters", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=60)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, env_config, _ = load_policy(args.checkpoint, "cpu")
    seeds = eval_seeds(args.episodes)
    service = SimService(len(seeds), env_config.view_width, env_config.view_height)
    rows = service.reset(range(len(seeds)), seeds, [args.maze] * len(seeds), [args.monsters] * len(seeds), [True] * len(seeds))
    limit = env_config.arena_steps
    memory, starts = model.initial_state(len(seeds), "cpu"), torch.ones(len(seeds))
    done = [r["mobs_total"] == 0 for r in rows]
    outcome = ["no mobs" if d else None for d in done]
    first_kill_step, last_kill_step = [None] * len(seeds), [None] * len(seeds)
    for t in range(limit):
        obs = [observe(r, t, limit, env_config.view_height, env_config.view_width) for r in rows]
        obs = {k: np.stack([o[k] for o in obs]) for k in ("grid", "state")}
        actions, _, _, memory = model.act(to_tensors(obs, "cpu"), memory, starts, greedy=False)
        starts = torch.zeros(len(seeds))
        service.send_step(actions.numpy(), env_config.repeat)
        new = service.read_step()
        for i, (old, row) in enumerate(zip(rows, new)):
            if done[i]:
                continue
            if row["mobs_killed"] > old["mobs_killed"]:
                first_kill_step[i] = first_kill_step[i] or t + 1
                last_kill_step[i] = t + 1
            if row["dead"] or row["mobs_left"] == 0:
                done[i], outcome[i] = True, "death" if row["dead"] else "cleared"
        rows = new
        if all(done):
            break
    survivors, distances, per_episode = Counter(), [], []
    for i, row in enumerate(rows):
        if outcome[i] is not None:
            continue
        outcome[i] = "timeout"
        snap = service.snapshot(i)
        cells = grid_cells(snap["grid"], snap["height"], snap["width"])
        player = (snap["top_y"] - round(snap["player"][1]), round(snap["player"][0]))
        dist = distances_from(cells, player)
        kinds = []
        for r, c in np.argwhere(cells & (MOB | WIZARD)):
            kind = "wizard" if cells[r, c] & WIZARD else "melee"
            survivors[kind] += 1
            distances.append(dist[r, c])
            kinds.append(kind)
        per_episode.append((row["mobs_left"], row["mobs_total"], last_kill_step[i]))
    service.close()
    print("outcomes:", dict(Counter(outcome)))
    print("surviving mobs at timeout by kind:", dict(survivors))
    finite = [d for d in distances if np.isfinite(d)]
    print(f"path distance player -> survivor: median {np.median(finite):.0f}, unreachable {len(distances) - len(finite)} of {len(distances)}")
    left = [p[0] for p in per_episode]
    print(f"mobs left per timed-out arena: median {np.median(left):.0f} (of median {np.median([p[1] for p in per_episode]):.0f})")
    idle = [limit - p[2] for p in per_episode if p[2]]
    print(f"steps since the last kill when time ran out: median {np.median(idle):.0f}")


if __name__ == "__main__":
    main()
