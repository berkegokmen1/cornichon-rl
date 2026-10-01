"""Training reward, computed in Python from the cumulative event counters the simulator reports.

The game's own score (Scores.java) is reported for logging but never trained on.
"""

from dataclasses import dataclass, fields

UNREACHABLE = 2**31 - 1  # Simulation.UNREACHABLE: the player's cell has no path to the door


@dataclass(frozen=True)
class RewardConfig:
    level_complete: float = 10.0
    death: float = -2.0
    # The game's own score values a kill at 100 points and a finished level at 25 (Scores.java). With kills worth 0.5
    # the agent learned to run past every mob (0.2 kills/episode of ~9 at difficulty 2) and most deaths were mobs.
    mob_killed: float = 2.0
    damage_dealt_per_hp: float = 0.01  # sphere hits; a 100 HP mob pays 1 over its 4 hits, so ~3 per kill in total
    # Taking hits must cost more than killing the mob pays: a skeleton hit (20 HP) is -0.6 against ~3 per kill.
    damage_per_hp: float = -0.03
    collected: float = 0.1  # health or mana potion
    step: float = -0.001  # per agent decision
    # Progress shaping: + per tile the shortest path to the door gets shorter, - per tile it gets longer. It
    # telescopes over an episode to path_progress * (start distance - end distance), so it cannot be farmed.
    path_progress: float = 0.2

    @classmethod
    def from_dict(cls, values):
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown reward keys {sorted(unknown)}")
        return cls(**values)


def step_reward(previous, current, config):
    """Reward for one agent step and its parts. previous/current are simulator rows (dicts) before and after."""
    parts = {
        "level_complete": config.level_complete * float(current["completed"]),
        "death": config.death * float(current["dead"]),
        "mob_killed": config.mob_killed * (current["mobs_killed"] - previous["mobs_killed"]),
        "damage_dealt": config.damage_dealt_per_hp * (current["damage_dealt"] - previous["damage_dealt"]),
        "damage": config.damage_per_hp * (current["damage"] - previous["damage"]),
        "collected": config.collected * (current["collected"] - previous["collected"]),
        "step": config.step,
        "path_progress": 0.0,
    }
    before, after = previous["door_distance"], current["door_distance"]
    if before != UNREACHABLE and after != UNREACHABLE:
        parts["path_progress"] = config.path_progress * (before - after)
    return sum(parts.values()), parts
