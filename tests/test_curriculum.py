import numpy as np

from cornichon_rl.curriculum import EVAL_SEED_START, Curriculum, CurriculumConfig, eval_seeds


def test_promotes_after_a_full_successful_window():
    curriculum = Curriculum(CurriculumConfig(window=10, promote_at=0.8, replay=0.0), np.random.default_rng(0))
    for i in range(9):
        assert not curriculum.record(1, True)
    assert curriculum.record(1, True)
    assert curriculum.difficulty == 2 and curriculum.success_rate() == 0.0


def test_replayed_easier_episodes_do_not_count():
    curriculum = Curriculum(CurriculumConfig(window=5, replay=0.5), np.random.default_rng(0))
    curriculum.difficulty = 3
    for _ in range(20):
        curriculum.record(2, True)
    assert curriculum.difficulty == 3


def test_training_and_eval_seeds_disjoint():
    curriculum = Curriculum(CurriculumConfig(), np.random.default_rng(0))
    train = {curriculum.sample()[1] for _ in range(5000)}
    assert max(train) < EVAL_SEED_START <= min(eval_seeds(100))


def test_state_round_trip():
    a = Curriculum(CurriculumConfig(window=5), np.random.default_rng(3))
    a.record(1, True)
    b = Curriculum(CurriculumConfig(window=5), np.random.default_rng(99))
    b.load_state_dict(a.state_dict())
    assert [a.sample() for _ in range(5)] == [b.sample() for _ in range(5)]


def test_sweep_reports_every_difficulty():
    from cornichon_rl.config import EnvConfig
    from cornichon_rl.evaluate import sweep

    result = sweep("random", [1, 2, 3], eval_seeds(2), EnvConfig(max_steps=20, envs_per_service=4))
    assert sorted(result) == [1, 2, 3]
    assert all(r["episodes"] == 2 and r["success"] + r["death"] + r["timeout"] == 1 for r in result.values())
