"""Which level (difficulty, seed) each training episode gets.

Training and evaluation seeds come from disjoint ranges, so evaluation always plays mazes the policy has never
seen. The difficulty rises one step at a time once the rolling success rate at the current difficulty is high
enough; a fraction of episodes replays easier difficulties so earlier levels are not forgotten.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

# Evaluation seeds start here; training seeds are drawn from [0, train_seed_count), which must stay below it.
EVAL_SEED_START = 1_000_000


@dataclass
class CurriculumConfig:
    start: int = 1  # the game starts at difficulty 1 and ends after difficulty 10
    end: int = 10
    promote_at: float = 0.8  # rolling success needed to move up
    window: int = 200  # episodes at the current difficulty that the success rate is computed over
    replay: float = 0.2  # fraction of episodes at an easier difficulty, once there is one
    train_seed_count: int = 100_000
    # Combat practice: this fraction of training levels keeps the maze size but gets the mob density of 1..monster_boost
    # difficulties higher, so fighting is needed long before the large levels. Promotion counts only real game levels.
    monster_practice: float = 0.0
    monster_boost: int = 3


class Curriculum:
    def __init__(self, config, rng):
        if config.train_seed_count > EVAL_SEED_START:
            raise ValueError("training seeds would overlap evaluation seeds")
        self.config = config
        self.rng = rng
        self.difficulty = config.start
        self.recent = deque(maxlen=config.window)

    def sample(self):
        """(difficulty, seed, monster_difficulty) for a new training episode."""
        difficulty = self.difficulty
        if difficulty > self.config.start and self.rng.random() < self.config.replay:
            difficulty = int(self.rng.integers(self.config.start, self.difficulty))
        seed = int(self.rng.integers(self.config.train_seed_count))
        monsters = difficulty
        if self.rng.random() < self.config.monster_practice:
            monsters = min(10, difficulty + int(self.rng.integers(1, self.config.monster_boost + 1)))
        return difficulty, seed, monsters

    def record(self, difficulty, success, monster_difficulty=None):
        """Count a finished episode; returns True when this promoted the curriculum. Practice levels (denser mobs
        than the game) do not count."""
        if difficulty != self.difficulty or (monster_difficulty is not None and monster_difficulty != difficulty):
            return False
        self.recent.append(bool(success))
        if (
            self.difficulty < self.config.end
            and len(self.recent) == self.recent.maxlen
            and self.success_rate() >= self.config.promote_at
        ):
            self.difficulty += 1
            self.recent.clear()
            return True
        return False

    def success_rate(self):
        return float(np.mean(self.recent)) if self.recent else 0.0

    def state_dict(self):
        return {"difficulty": self.difficulty, "recent": list(self.recent), "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.difficulty = state["difficulty"]
        self.recent = deque(state["recent"], maxlen=self.config.window)
        self.rng.bit_generator.state = state["rng"]


def eval_seeds(count, offset=0):
    return [EVAL_SEED_START + offset + i for i in range(count)]
