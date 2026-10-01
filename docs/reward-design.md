# Reward design

Python owns the reward (`cornichon_rl/reward.py`); the simulator only reports cumulative counters. The game's own
score is logged (`rollout/score`) but never trained on.

| Term | Default | Note |
|---|---:|---|
| level complete | +10 | the objective |
| death | −2 | on top of the damage that caused it. −5 in `ppo_curriculum_v1` collapsed into "never move, never die" (99% timeouts) |
| mob killed | +2 | per kill. Was +0.5 up to v2: the agent ran past every mob (0.2 kills of ~9 per d2 episode) and mobs caused most deaths. The game itself scores a kill at 100 and a level at 25 |
| sphere damage dealt | +0.01 / HP | each hit pays on the way to a kill (100 HP mob, 25 per hit, 50 buffed): ~3 per kill in total |
| damage taken | −0.01 / HP | mobs, spikes, fireballs; 100 HP = −1 |
| potion collected | +0.1 | |
| step | −0.001 | per decision; −1.5 over a full timeout |
| path progress | +0.2 / tile | shortest 4-connected path to the door (BFS over bricks, ignores gravity). Telescopes to 0.2 × (start − end distance), so it cannot be farmed (a level starts 40–80 tiles away); skipped when the player's cell has no path |

`reward_parts/*` in W&B shows each term's per-episode sum. Change any term with `reward.<name>=value`.
