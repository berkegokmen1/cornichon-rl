# Reward design

Python owns the reward (`cornichon_rl/reward.py`); the simulator only reports cumulative counters. The game's own
score is logged (`rollout/score`) but never trained on.

| Term | Default | Note |
|---|---:|---|
| level complete | +10 | the objective |
| death | −5 | on top of the damage that caused it |
| mob killed | +0.5 | per kill (sphere hits) |
| damage taken | −0.02 / HP | mobs, spikes, fireballs; 100 HP = −2 |
| potion collected | +0.1 | |
| step | −0.001 | per decision; −1.5 over a full timeout |
| path progress | +0.05 / tile | shortest 4-connected path to the door (BFS over bricks, ignores gravity). Telescopes to 0.05 × (start − end distance), so it cannot be farmed; skipped when the player's cell has no path |

`reward_parts/*` in W&B shows each term's per-episode sum. Change any term with `reward.<name>=value`.
