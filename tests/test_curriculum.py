import numpy as np

from cornichon_rl.curriculum import EVAL_SEED_START, Curriculum, CurriculumConfig, eval_seeds


def test_promotes_after_a_full_successful_window():
    curriculum = Curriculum(CurriculumConfig(window=10, promote_at=0.8, replay=0.0), np.random.default_rng(0))
    for i in range(9):
        assert not curriculum.record(episode(1))
    assert curriculum.record(episode(1))
    assert curriculum.difficulty == 2 and curriculum.success_rate() == 0.0


def test_replayed_easier_episodes_do_not_count():
    curriculum = Curriculum(CurriculumConfig(window=5, replay=0.5), np.random.default_rng(0))
    curriculum.difficulty = 3
    for _ in range(20):
        curriculum.record(episode(2))
    assert curriculum.difficulty == 3


def test_training_and_eval_seeds_disjoint():
    curriculum = Curriculum(CurriculumConfig(), np.random.default_rng(0))
    train = {curriculum.sample()[1] for _ in range(5000)}
    assert max(train) < EVAL_SEED_START <= min(eval_seeds(100))


def test_state_round_trip():
    a = Curriculum(CurriculumConfig(window=5), np.random.default_rng(3))
    a.record(episode(1))
    b = Curriculum(CurriculumConfig(window=5), np.random.default_rng(99))
    b.load_state_dict(a.state_dict())
    assert [a.sample() for _ in range(5)] == [b.sample() for _ in range(5)]


def test_sweep_reports_every_difficulty():
    from cornichon_rl.config import EnvConfig
    from cornichon_rl.evaluate import sweep

    levels = {d: [(d, s) for s in eval_seeds(2)] for d in (1, 2, 3)}
    result = sweep("random", levels, EnvConfig(max_steps=20, envs_per_service=4))
    assert sorted(result) == [1, 2, 3]
    assert all(r["episodes"] == 2 and r["success"] + r["death"] + r["timeout"] == 1 for r in result.values())


def episode(difficulty, success=True, monsters=None, arena=False):
    return {"difficulty": difficulty, "success": success, "arena": arena,
            "monster_difficulty": difficulty if monsters is None else monsters}


def test_monster_practice_levels_never_promote():
    curriculum = Curriculum(CurriculumConfig(window=5, monster_practice=1.0), np.random.default_rng(0))
    spec = curriculum.sample()
    assert spec.monsters > spec.difficulty
    for _ in range(20):
        curriculum.record(episode(1, monsters=3))
    assert curriculum.difficulty == 1
    for _ in range(5):
        curriculum.record(episode(1))
    assert curriculum.difficulty == 2


def test_arena_curriculum_promotes_on_mob_density():
    config = CurriculumConfig(arena=True, arena_maze=1, start=2, window=5, replay=0.0)
    curriculum = Curriculum(config, np.random.default_rng(0))
    spec = curriculum.sample()
    assert spec.arena and spec.difficulty == 1 and spec.monsters == 2
    for _ in range(5):
        curriculum.record(episode(2))  # a real level does not count in arena mode
    assert curriculum.difficulty == 2
    for _ in range(5):
        curriculum.record(episode(1, monsters=2, arena=True))
    assert curriculum.difficulty == 3
