# Experiments

Every run and check, newest last. Times are US Pacific. "Held-out" = evaluation mazes (seeds ≥ 1,000,000) never
used in training. Runs live in `/data/local/berke/cornichon-rl/runs/<name>/` on the node listed (node-local disk);
logs in `/data/local/berke/cornichon-rl/logs/<name>.log`. W&B: `berkegokmen1/cornichon-rl/runs/<id>`.

## Runs

| run | started (PT) | node:GPU | commit | what changed | result | status |
|---|---|---|---|---|---|---|
| `ppo_curriculum_v1` · `pest3dv3` | 09-30 18:50 | d5:0 | pre-`9bc2f2e` (uncommitted) | first PPO on the real game. CNN policy, reward: complete +10, death −5, damage −0.02/HP, kill +0.5, path +0.05/tile, 1500-step limit | **collapsed into "never move, never die"**: held-out d1 0% success, 98–99% timeouts, 1–2% deaths (update 50–150) | stopped at update 174 (berke OK'd) |
| `ppo_curriculum_v2` · `vxxkua23` | 09-30 19:02 | d5:0 | `f1e8f97`, resumed on `a5b7571` | reward rebalance: death −2, damage −0.01/HP, path +0.2/tile; 750-step limit. Resumed at update 320 with the path-scaled time limit (300 + 6/tile) and per-difficulty sweeps | **finished 50M steps.** Promoted d2 at update 561, d3 at 2531. Final held-out d3: **54%** success (27% death, 19% timeout). Sweep: d1 90%, d2 65%, d3 60%, d4 30%. Kills ≈ 0.2/episode | done (best.pt = d3 56%) |
| `ppo_v3_lstm_combat` · `lx8yo48j` | 09-30 21:42 | d1:9 | `7448bf8` | duplicate of v3 on d1 | 900 steps/s (d1 load 178) | stopped at update 28 (berke OK'd) |
| `ppo_v3` · `sogcneji` | 09-30 21:47 | cthulhu1:4 | `7448bf8` | LSTM policy; kill +2, sphere damage +0.01/HP; eval with sampled actions | promoted d2 at update 332 (v2: 561). Held-out d2 ~45–52%, **deaths 36–45%**, kills 0.2–0.4/episode: the combat reward alone did not make it fight. Sweep at 600: d1 95%, d2 55%, d3 35% | running |
| `ppo_v4_combat` · `7u3j82aa` | 09-30 22:36 | cthulhu1:5 | `e65bf7e` | v3 + sphere and fireball-direction grid channels; half of training levels get mobs of 1–3 difficulties higher; damage −0.03/HP | update 400: held-out d1 67% (v3 at 400: ~55%, v2: ~33%), kills 0.4/episode, damage dealt ~80 HP/episode (≈ 3 hits, spread over mobs) | running |
| `arena_v1` | 09-30 23:47 | d5:0 | `e0ea71b` | phase 1 of two-phase training: door closed, kill every mob to finish; mob density curriculum 2 → 10 on difficulty-1 mazes | — | launching |

## Checks and findings

| when (PT) | what | result |
|---|---|---|
| ~09-30 18:40 | random policy, 100 held-out mazes each at d1/d5/d10 | 0% success at every difficulty; 80/85/100% death |
| ~09-30 19:00 | `scripts/scripted_bot.py`, BFS path follower, 30 held-out d1 mazes | 10% success (each in ~110 steps): levels are completable; the bot itself gets stuck at ledges |
| ~09-30 19:00 | jump height | **idle sphere rests 0.62 above the player's head and blocks jumps** (0.2 tiles vs 4.9). Moving it up/aside first restores the full jump. Same code as the desktop game |
| ~09-30 19:00 | ground contact | any brick contact counts as ground, side walls included, so the player can re-jump while touching a wall |
| ~09-30 19:45 | level stats per difficulty (20 mazes each) | d1 29×17, path 55, 6 mobs; d5 45×29, path 105, 19 mobs; d10 65×45, path 175, **131 mobs, 0 potions** (mob chance `1/(6 − d/2)` = 1) |
| ~09-30 21:00 | `scripts/diagnose_timeouts.py`, v2 best.pt, 100 held-out d2 | 57% success; 23 of 26 timeouts are the **greedy** policy frozen in a loop (sphere never on the head) |
| ~09-30 21:15 | same, greedy vs sampled actions | greedy 57/17/26 (success/death/timeout) vs sampled 60/27/13: sampling breaks loops; then mobs kill it. Kills 0.15–0.2/episode |
| ~09-30 23:10 | why kills stay low | sphere reach ≈ 1.6 tiles sideways (spring pulls it back), touching a mob hurts, a kill needs 4 separate hits (25 vs 100 HP, knock-back after each), the 50-damage buff costs 70 mana at 10 per potion |
| ~09-30 22:30 | simulator install | a stale core jar went live for ~3.5 min; no run hit an eval in that window. Installs now go through `scripts/install_simulator.sh` (tests both protocol formats before swapping) |

## Two-phase plan (09-30 23:30, berke's idea)

1. **Arena** (`configs/arena.yaml`): door closed, the level ends only when every mob is dead (+5 clear bonus) or at
   600 decisions; −0.003 per step while mobs are alive (hiding is not free); kill +2, sphere damage +0.01/HP,
   damage −0.03/HP. Curriculum = mob density 2 → 10 on difficulty-1 mazes, promote at 70% cleared.
2. **Game**: `configs/default.yaml` with `init_from=<arena best.pt>`; door and path rewards are added on top of the
   combat terms (not swapped), and half the levels stay monster-dense, so fighting keeps paying.
3. Compare against `ppo_v4_combat`, same features without pretraining. Watch `rollout/kill_fraction` in phase 2:
   if it drops back toward v2's, phase 2 is forgetting combat.
