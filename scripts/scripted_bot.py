#!/usr/bin/env python3
"""Can the levels be finished through the agent's action interface? A hand-written bot that follows the shortest
grid path to the door (jumping when the path goes up) plays held-out mazes. Upper bound check, not a baseline to beat.

  python scripts/scripted_bot.py --difficulty 1 --episodes 30
"""

import argparse
import base64
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cornichon_rl.curriculum import eval_seeds  # noqa: E402
from cornichon_rl.service import SimService  # noqa: E402


def door_distances(snapshot):
    h, w = snapshot["height"], snapshot["width"]
    grid = np.frombuffer(base64.b64decode(snapshot["grid"]), np.uint8).reshape(h, w)
    wall = (grid & 1) > 0
    door = np.argwhere(grid & 128)[0]
    dist = np.full((h, w), np.inf)
    dist[tuple(door)] = 0
    queue = deque([tuple(door)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not wall[rr, cc] and dist[rr, cc] == np.inf:
                dist[rr, cc] = dist[r, c] + 1
                queue.append((rr, cc))
    return dist, wall


def choose(dist, wall, top_y, x, y, grounded, lookahead=3):
    r, c = top_y - round(y), round(x)
    h, w = dist.shape
    # follow the gradient a few cells ahead
    target = (r, c)
    for _ in range(lookahead):
        best = target
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = best[0] + dr, best[1] + dc
            if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] < dist[best]:
                best = (rr, cc)
        if best == target:
            break
        target = best
    move = 0
    if target[1] > c:
        move = 2
    elif target[1] < c:
        move = 1
    blocked = move and 0 <= c + (1 if move == 2 else -1) < w and wall[r, c + (1 if move == 2 else -1)]
    jump = int(grounded and (target[0] < r or blocked))
    return [move, jump, 0, 0, 1]  # sphere held up: idle, it rests on the head and blocks every jump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=1500)
    args = parser.parse_args()
    service = SimService(1)
    outcomes = []
    for seed in eval_seeds(args.episodes):
        row = service.reset([0], [seed], [args.difficulty])[0]
        snap = service.snapshot(0)
        dist, wall = door_distances(snap)
        start = row["door_distance"]
        for step in range(args.max_steps):
            action = choose(dist, wall, snap["top_y"], row["x"], row["y"], row["state"][4] > 0.5)
            service.send_step([action], 4)
            row = service.read_step()[0]
            if row["completed"] or row["dead"]:
                break
        outcome = "success" if row["completed"] else "death" if row["dead"] else "timeout"
        outcomes.append(outcome)
        print(json.dumps({"seed": seed, "outcome": outcome, "steps": step + 1, "start_distance": start,
                          "end_distance": row["door_distance"], "health": row["health"]}), flush=True)
    service.close()
    print({k: outcomes.count(k) / len(outcomes) for k in ("success", "death", "timeout")})


if __name__ == "__main__":
    main()
