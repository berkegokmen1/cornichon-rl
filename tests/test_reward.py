from cornichon_rl.reward import UNREACHABLE, RewardConfig, step_reward


def row(**values):
    base = {"completed": False, "dead": False, "mobs_killed": 0, "damage": 0.0, "damage_dealt": 0, "collected": 0, "door_distance": 30}
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


def test_sphere_hits_pay_on_the_way_to_a_kill():
    config = RewardConfig()
    _, parts = step_reward(row(damage_dealt=50), row(damage_dealt=75), config)
    assert parts["damage_dealt"] == config.damage_dealt_per_hp * 25
    kill = config.mob_killed + 100 * config.damage_dealt_per_hp
    assert kill * 9 > config.level_complete  # clearing a difficulty-2 maze is worth more than the door, as in the game


def test_hunt_progress_only_in_arena_and_not_on_kill_steps():
    config = RewardConfig(hunt_progress=0.1)
    arena = dict(door_closed=True, mobs_left=3)
    _, parts = step_reward(row(mob_distance=20, **arena), row(mob_distance=15, **arena), config)
    assert abs(parts["hunt_progress"] - 0.5) < 1e-9
    _, parts = step_reward(row(mob_distance=2, **arena), row(mob_distance=30, door_closed=True, mobs_left=2), config)
    assert parts["hunt_progress"] == 0.0  # a kill switched the target
    _, parts = step_reward(row(mob_distance=20), row(mob_distance=15), config)
    assert parts["hunt_progress"] == 0.0  # not an arena
