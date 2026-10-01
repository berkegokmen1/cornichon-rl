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
| `ppo_v3` · `sogcneji` | 09-30 21:47 | cthulhu1:4 | `7448bf8` | LSTM policy; kill +2, sphere damage +0.01/HP; eval with sampled actions | **finished 50M.** Promoted d2 at 332, d3 at 2039, **d4 at 2879** (furthest of any run). Final held-out d4 48% (49% death, 3% timeout), kills ~1.0/episode (v2: 0.2). Final sweep: d1 95%, d2 90%, d3 70%, d4 35%, d5 30%. Early on kills stayed 0.2–0.4 and deaths 36–45% | done |
| `ppo_v4_combat` · `7u3j82aa` | 09-30 22:36 | cthulhu1:5 | `e65bf7e` | v3 + sphere and fireball-direction grid channels; half of training levels get mobs of 1–3 difficulties higher; damage −0.03/HP | **finished 50M.** Promoted d2 at 681, d3 at 2216. Final held-out d3 **72%** (25% death), kills ~1.2/episode. Final sweep: d1 90%, d2 85%, d3 70%, d4 45%. Best pure-game run; the control for two-phase training | done |
| `arena_v1` · `i2q758lx` | 09-30 23:47 | d5:0 | `e0ea71b` | phase 1 of two-phase training: door closed, kill every mob to finish; mob density curriculum 2 → 10 on difficulty-1 mazes | **finished 30M, never left density 2.** Held-out: 20–25% cleared, 55–59% timeout, 17–24% death; **5.6 kills/arena = 70% of the mobs** (game runs: ~1.2). Promotion needs 70% cleared | done |
| `game_from_arena_v1` | 10-01 07:20 | d5:0 | `57234aa` | phase 2: `configs/default.yaml` (= v4 setup) initialized from `arena_v1/best.pt`. Control: `ppo_v4_combat` | update 50: training success 66% at d1 (v4: ~update 400). Promoted d2 at 422 (v4: 681). **Same-update vs v4 at d2 (updates 800–1100): success 57–64% vs 43–57%, deaths 10–18% vs 32–42%, kills 4.6–4.7 vs 0.5–0.8 per episode.** Arena combat carries over to the game. Sweep at 1000: d1 80%, d2 50%, d3 40% | running |
| `arena_v2` | 10-01 07:21 | cthulhu1:4 | `c346caa` | arena_v1 + **mob compass** (path distance and direction to the nearest living mob in the state) + hunt_progress +0.1/tile. Fresh weights (state size changed) | update 800: 3% cleared, 46% of mobs killed (= arena_v1 at 800). Peaked at update 900–1200: **54–55% of mobs killed** (arena_v1 at that point: ~50%), 7–12% cleared, 22–29% death; flat from 900 on | **stopped at update ~1180** (berke: stuck, start phase 2). best.pt = update 1200 (12% cleared) |
| `game_from_arena_v2` | 10-01 10:41 | cthulhu1:5 | `f574937` | phase 2 from `arena_v2/best.pt`: the game agent gets the mob compass and the compass-trained arena fighter. Compare with `game_from_arena_v1` (no compass) and `ppo_v4_combat` (no arena) | — | running |
| `fight_from_arena_v1` · `uiwet2ss` | 10-01 11:08 | cthulhu1:4 | `d1976f2` | best recipe so far (phase 2 from `arena_v1/best.pt`, = `game_from_arena_v1`) + **door bonus × fraction of mobs killed** (`configs/fight_through.yaml`, +10 for a full clear on top of the door's 10): berke wants it to fight through to the door, not jump over mobs. Compass inputs added with zero weights (`load_widened`), so it starts as arena_v1. Control: `game_from_arena_v1` | — | running |
| `fight_from_arena_v1_s2` · `urew7ecu` | 10-01 11:08 | d2:0 | `d1976f2` | same, seed 2 (seed spread) | — | running |

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
| 10-01 07:30 | `scripts/diagnose_arena.py`, arena_v1 best.pt, 60 held-out density-2 arenas | 10 cleared / 20 death / 30 timeout. Timed-out arenas: median **3 of 8 mobs left**, all reachable, median **40 tiles** of path away; median **305 steps since the last kill**. It fights well up close, then **stops hunting**: survivors are outside its 31×21 view and nothing points to them (the door has a compass in the state; mobs do not) |
| 10-01 11:40 | **release benchmark**: 200 held-out mazes per difficulty d1–d6, sampled actions, `ppo_v3` final, `ppo_v4_combat` best + final, `game_from_arena_v1` update 1960. Results in `/data/local/berke/cornichon-rl/bench/` (cthulhu1) | success d1–d6: v4 best **91/82/71/42/40/16**, v4 final 94/77/64/44/38/16, v3 92/80/73/44/35/16, arena-pretrained 90/70/59/41/28/14. Arena-pretrained kills 3.7–9.1/level (v4: 0.8–1.7), deaths 8–49% (v4: 8–76%), but timeouts 18–41% from d2 on (v4: 0–10%): it fights instead of finishing. The 20-level in-training sweeps had hidden this |
| 10-01 12:00 | released `weights/cornichon_ppo_v4` (= `ppo_v4_combat/best.pt`) and `weights/cornichon_fighter` (= `game_from_arena_v1` update 1960, to be replaced by its final checkpoint) with `scripts/export_weights.py`; README "Pretrained agents" section, GIFs in `docs/media/` (held-out d2 seed 1000003, both reach the door) | |

## Two-phase plan (09-30 23:30, berke's idea)

1. **Arena** (`configs/arena.yaml`): door closed, the level ends only when every mob is dead (+5 clear bonus) or at
   600 decisions; −0.003 per step while mobs are alive (hiding is not free); kill +2, sphere damage +0.01/HP,
   damage −0.03/HP. Curriculum = mob density 2 → 10 on difficulty-1 mazes, promote at 70% cleared.
2. **Game**: `configs/default.yaml` with `init_from=<arena best.pt>`; door and path rewards are added on top of the
   combat terms (not swapped), and half the levels stay monster-dense, so fighting keeps paying.
3. Compare against `ppo_v4_combat`, same features without pretraining. Watch `rollout/kill_fraction` in phase 2:
   if it drops back toward v2's, phase 2 is forgetting combat.
