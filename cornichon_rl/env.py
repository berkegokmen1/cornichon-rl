"""Cornichon as an RL environment.

Observation (dict):
  grid  (8, H, W) 0/1 channels around the player (GRID_CHANNELS), player at the centre cell, row 0 at the top.
  state (17,)     Simulation.state() (velocities, health, mana, grounded, sphere offset/velocity, buff, door offset,
                  path distance, difficulty, sub-cell position) + fraction of the time limit used.
Action: MultiDiscrete(ACTION_NVEC), each decision held for `repeat` game frames (60 fps; repeat 4 -> 15 Hz).
Episode: one level. Ends on reaching the door (success), death, or at the time limit (truncation): time_base +
time_per_tile decisions per tile of shortest path from the start to the door, capped at max_steps, so bigger
mazes (higher difficulty) get proportionally more time.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .reward import UNREACHABLE, RewardConfig, step_reward
from .service import ACTION_NVEC, GRID_CHANNELS, SimService, decode_grid

STATE_SIZE = 17


def observation_space(view_height, view_width):
    return spaces.Dict({
        "grid": spaces.Box(0.0, 1.0, (len(GRID_CHANNELS), view_height, view_width), np.float32),
        "state": spaces.Box(-10.0, 10.0, (STATE_SIZE,), np.float32),
    })


def time_limit(start_distance, max_steps, time_base, time_per_tile):
    """Decisions allowed for one level. time_per_tile 0 gives every level max_steps."""
    if time_per_tile <= 0 or start_distance >= UNREACHABLE:
        return max_steps
    return int(min(max_steps, time_base + time_per_tile * start_distance))


def observe(row, steps, limit, height, width):
    state = np.empty(STATE_SIZE, np.float32)
    state[:-1] = np.clip(row["state"], -10.0, 10.0)
    state[-1] = steps / limit
    return {"grid": decode_grid(row, height, width), "state": state}


def episode_summary(row, steps, limit, episode_return, parts, start_distance):
    return {
        "return": episode_return,
        "length": steps,
        "time_limit": limit,
        "success": bool(row["completed"]),
        "death": bool(row["dead"]),
        "timeout": not (row["completed"] or row["dead"]),
        "difficulty": row["difficulty"],
        "seed": row["seed"],
        "mobs_killed": row["mobs_killed"],
        "damage": row["damage"],
        "collected": row["collected"],
        "score": row["score"],
        "door_distance": row["door_distance"],
        # tiles of shortest path to the door gained over the episode (0 when the path was unknown at either end)
        "progress": start_distance - row["door_distance"] if max(start_distance, row["door_distance"]) < UNREACHABLE else 0,
        "parts": dict(parts),
    }


class VecCornichon:
    """num_envs levels spread over several simulator processes that step in parallel; resets finished levels.

    next_level(env_index) -> (difficulty, seed) chooses the level for each new episode.
    """

    def __init__(
        self,
        num_envs,
        next_level,
        envs_per_service=16,
        max_steps=1500,
        time_base=300,
        time_per_tile=6,
        repeat=4,
        reward=RewardConfig(),
        view_width=31,
        view_height=21,
    ):
        self.num_envs = num_envs
        self.next_level = next_level
        self.max_steps = max_steps
        self.time_base = time_base
        self.time_per_tile = time_per_tile
        self.repeat = repeat
        self.reward = reward
        self.height, self.width = view_height, view_width
        sizes = [min(envs_per_service, num_envs - start) for start in range(0, num_envs, envs_per_service)]
        self.services = [SimService(n, view_width, view_height) for n in sizes]
        self.slices = []
        start = 0
        for n in sizes:
            self.slices.append(slice(start, start + n))
            start += n
        self.observation_space = observation_space(view_height, view_width)
        self.rows = [None] * num_envs
        self.steps = np.zeros(num_envs, dtype=int)
        self.returns = np.zeros(num_envs)
        self.parts = [{} for _ in range(num_envs)]
        self.start_distance = [0] * num_envs
        self.limits = np.full(num_envs, max_steps)

    @classmethod
    def from_config(cls, num_envs, next_level, env_config, reward=RewardConfig()):
        return cls(
            num_envs,
            next_level,
            envs_per_service=env_config.envs_per_service,
            max_steps=env_config.max_steps,
            time_base=env_config.time_base,
            time_per_tile=env_config.time_per_tile,
            repeat=env_config.repeat,
            reward=reward,
            view_width=env_config.view_width,
            view_height=env_config.view_height,
        )

    def _locate(self, env):
        for service, part in zip(self.services, self.slices):
            if part.start <= env < part.stop:
                return service, env - part.start
        raise IndexError(env)

    def _reset_envs(self, envs):
        by_service = {}
        for env in envs:
            service, local = self._locate(env)
            difficulty, seed = self.next_level(env)
            by_service.setdefault(id(service), (service, []))[1].append((env, local, difficulty, seed))
        for service, items in by_service.values():
            rows = service.reset([i[1] for i in items], [i[3] for i in items], [i[2] for i in items])
            for (env, *_), row in zip(items, rows):
                self.rows[env] = row
                self.steps[env] = 0
                self.returns[env] = 0.0
                self.parts[env] = {}
                self.start_distance[env] = row["door_distance"]
                self.limits[env] = time_limit(row["door_distance"], self.max_steps, self.time_base, self.time_per_tile)

    def reset(self):
        self._reset_envs(range(self.num_envs))
        return self._batch_obs()

    def _obs(self, env):
        return observe(self.rows[env], self.steps[env], self.limits[env], self.height, self.width)

    def _batch_obs(self):
        obs = [self._obs(i) for i in range(self.num_envs)]
        return {key: np.stack([o[key] for o in obs]) for key in ("grid", "state")}

    def step(self, actions):
        """Returns obs (after resets), reward, terminated, truncated, info.

        info["episodes"] lists summaries of episodes that ended; info["final_obs"] maps env -> last observation of
        a truncated episode, which the learner bootstraps from.
        """
        actions = np.asarray(actions)
        for service, part in zip(self.services, self.slices):
            service.send_step(actions[part], self.repeat)
        new_rows = []
        for service in self.services:
            new_rows.extend(service.read_step())

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        episodes, final_obs, finished = [], {}, []
        for env, row in enumerate(new_rows):
            reward, parts = step_reward(self.rows[env], row, self.reward)
            self.rows[env] = row
            self.steps[env] += 1
            self.returns[env] += reward
            for key, value in parts.items():
                self.parts[env][key] = self.parts[env].get(key, 0.0) + value
            rewards[env] = reward
            terminated[env] = row["completed"] or row["dead"]
            truncated[env] = not terminated[env] and self.steps[env] >= self.limits[env]
            if terminated[env] or truncated[env]:
                episodes.append(episode_summary(
                    row, int(self.steps[env]), int(self.limits[env]), float(self.returns[env]), self.parts[env], self.start_distance[env]
                ))
                if truncated[env]:
                    final_obs[env] = self._obs(env)
                finished.append(env)
        if finished:
            self._reset_envs(finished)
        return self._batch_obs(), rewards, terminated, truncated, {"episodes": episodes, "final_obs": final_obs}

    def snapshot(self, env):
        service, local = self._locate(env)
        return service.snapshot(local)

    def close(self):
        for service in self.services:
            service.close()


class CornichonEnv(gym.Env):
    """Single-level Gymnasium env. reset(seed=s) plays maze s; options={"difficulty": d} overrides difficulty."""

    metadata = {"render_modes": []}

    def __init__(
        self, difficulty=1, max_steps=1500, time_base=300, time_per_tile=6, repeat=4, reward=RewardConfig(),
        view_width=31, view_height=21,
    ):
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.time_base, self.time_per_tile = time_base, time_per_tile
        self.limit = max_steps
        self.repeat = repeat
        self.reward = reward
        self.height, self.width = view_height, view_width
        self.service = SimService(1, view_width, view_height)
        self.action_space = spaces.MultiDiscrete(ACTION_NVEC)
        self.observation_space = observation_space(view_height, view_width)
        self.row = None
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        level_seed = seed if seed is not None else int(self.np_random.integers(100_000))
        difficulty = (options or {}).get("difficulty", self.difficulty)
        self.row = self.service.reset([0], [level_seed], [difficulty])[0]
        self.steps = 0
        self.limit = time_limit(self.row["door_distance"], self.max_steps, self.time_base, self.time_per_tile)
        info = {"seed": level_seed, "difficulty": difficulty, "time_limit": self.limit}
        return observe(self.row, 0, self.limit, self.height, self.width), info

    def step(self, action):
        if not self.action_space.contains(np.asarray(action)):
            raise ValueError(f"invalid action {action}")
        self.service.send_step([action], self.repeat)
        row = self.service.read_step()[0]
        reward, parts = step_reward(self.row, row, self.reward)
        self.row = row
        self.steps += 1
        terminated = bool(row["completed"] or row["dead"])
        truncated = not terminated and self.steps >= self.limit
        info = {"reward_parts": parts, "completed": row["completed"], "dead": row["dead"], "score": row["score"]}
        return observe(row, self.steps, self.limit, self.height, self.width), reward, terminated, truncated, info

    def close(self):
        self.service.close()
