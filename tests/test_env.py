import numpy as np
from gymnasium.utils.env_checker import check_env

from cornichon_rl import CornichonEnv, VecCornichon


def test_gymnasium_checker():
    env = CornichonEnv()
    check_env(env, skip_render_check=True)
    env.close()


def test_observations_stay_in_space_and_episodes_truncate():
    levels = iter(range(1000))
    env = VecCornichon(4, lambda _i: (1, next(levels)), envs_per_service=2, max_steps=30)
    rng = np.random.default_rng(0)
    obs = env.reset()
    episodes = []
    for _ in range(65):
        actions = np.stack([rng.integers(n, size=4) for n in (3, 2, 2, 3, 3)], 1)
        obs, reward, terminated, truncated, info = env.step(actions)
        for i in range(4):
            assert env.observation_space.contains({"grid": obs["grid"][i], "state": obs["state"][i]})
        assert set(info["final_obs"]) == set(np.flatnonzero(truncated))
        episodes += info["episodes"]
    env.close()
    assert len(episodes) >= 8
    assert all(e["length"] <= 30 for e in episodes)
    assert all(e["success"] + e["death"] + e["timeout"] == 1 for e in episodes)
    for e in episodes:
        assert abs(sum(e["parts"].values()) - e["return"]) < 1e-6


def test_time_limit_scales_with_maze_size():
    from cornichon_rl.env import time_limit
    from cornichon_rl.reward import UNREACHABLE

    assert time_limit(55, 1500, 300, 6) == 630
    assert time_limit(175, 1500, 300, 6) == 1350
    assert time_limit(400, 1500, 300, 6) == 1500  # capped
    assert time_limit(UNREACHABLE, 1500, 300, 6) == 1500
    assert time_limit(55, 750, 300, 0) == 750  # fixed limit when time_per_tile is 0


def test_episodes_end_at_their_own_time_limit():
    seeds = iter(range(1000))
    env = VecCornichon(2, lambda _i: (1, next(seeds)), envs_per_service=2, max_steps=10_000, time_base=5, time_per_tile=1)
    env.reset()
    expected = [5 + d for d in env.start_distance]
    episodes = []
    while len(episodes) < 2:
        episodes += env.step([[0, 0, 0, 0, 0]] * 2)[4]["episodes"]
    env.close()
    for e in episodes:
        assert e["timeout"] and e["length"] == e["time_limit"] and e["time_limit"] in expected
