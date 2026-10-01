from cornichon_rl.reward import UNREACHABLE, RewardConfig, step_reward


def row(**values):
    base = {"completed": False, "dead": False, "mobs_killed": 0, "damage": 0.0, "collected": 0, "door_distance": 30}
    base.update(values)
    return base


def test_path_progress_telescopes():
    config = RewardConfig(step=0.0)
    distances = [30, 29, 31, 25, 25, 10]
    total = sum(step_reward(row(door_distance=a), row(door_distance=b), config)[0] for a, b in zip(distances, distances[1:]))
    assert abs(total - config.path_progress * (30 - 10)) < 1e-9


def test_no_shaping_when_path_unknown():
    _, parts = step_reward(row(door_distance=UNREACHABLE), row(door_distance=5), RewardConfig())
    assert parts["path_progress"] == 0.0


def test_events_pay_once_per_occurrence():
    config = RewardConfig()
    _, parts = step_reward(row(mobs_killed=1, damage=10.0), row(mobs_killed=2, damage=25.0, completed=True), config)
    assert parts["mob_killed"] == config.mob_killed
    assert parts["damage"] == config.damage_per_hp * 15.0
    assert parts["level_complete"] == config.level_complete


def test_death_outweighs_any_single_step_bonus():
    config = RewardConfig()
    assert config.death + config.mob_killed + config.collected < 0
