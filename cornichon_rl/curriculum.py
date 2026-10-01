"""Which level (difficulty, seed) each training episode gets.

Training and evaluation seeds come from disjoint ranges, so evaluation always plays mazes the policy has never
seen. The difficulty rises one step at a time once the rolling success rate at the current difficulty is high
enough; a fraction of episodes replays easier difficulties so earlier levels are not forgotten.
"""

from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

# Evaluation seeds start here; training seeds are drawn from [0, train_seed_count), which must stay below it.
EVAL_SEED_START = 1_000_000


class LevelSpec(NamedTuple):
    difficulty: int  # maze size (the game's level number)
    seed: int
    monsters: int | None = None  # mob/potion density; None = the game's (= difficulty)
    arena: bool = False  # door closed: the level ends when every mob is dead (combat practice)

    @classmethod
    def of(cls, level):
        spec = level if isinstance(level, LevelSpec) else cls(*level)
        return spec if spec.monsters is not None else spec._replace(monsters=spec.difficulty)


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
    # Arena mode (combat pretraining): every level is a small maze (arena_maze) with the door closed; the curriculum
    # difficulty is the mob density, and success means killing every mob.
    arena: bool = False
    arena_maze: int = 1


class Curriculum:
    def __init__(self, config, rng):
        if config.train_seed_count > EVAL_SEED_START:
            raise ValueError("training seeds would overlap evaluation seeds")
        self.config = config
        self.rng = rng
        self.difficulty = config.start
        self.recent = deque(maxlen=config.window)

    def sample(self):
        """LevelSpec for a new training episode."""
        difficulty = self.difficulty
        if difficulty > self.config.start and self.rng.random() < self.config.replay:
            difficulty = int(self.rng.integers(self.config.start, self.difficulty))
        seed = int(self.rng.integers(self.config.train_seed_count))
        if self.config.arena:
            return LevelSpec(self.config.arena_maze, seed, difficulty, arena=True)
        monsters = difficulty
        if self.rng.random() < self.config.monster_practice:
            monsters = min(10, difficulty + int(self.rng.integers(1, self.config.monster_boost + 1)))
        return LevelSpec(difficulty, seed, monsters)

    def level_difficulty(self, episode):
        """Which curriculum difficulty an episode belongs to, or None if it does not count (practice levels)."""
        if self.config.arena:
            return episode["monster_difficulty"] if episode["arena"] else None
        if episode["arena"] or episode["monster_difficulty"] != episode["difficulty"]:
            return None
        return episode["difficulty"]

    def levels(self, difficulty, seeds):
        """Evaluation levels at a curriculum difficulty: real game levels, or arenas in arena mode."""
        if self.config.arena:
            return [LevelSpec(self.config.arena_maze, s, difficulty, arena=True) for s in seeds]
        return [LevelSpec(difficulty, s, difficulty) for s in seeds]

    def record(self, episode):
        """Count a finished episode; returns True when this promoted the curriculum."""
        if self.level_difficulty(episode) != self.difficulty:
            return False
        self.recent.append(bool(episode["success"]))
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
