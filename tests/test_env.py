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
