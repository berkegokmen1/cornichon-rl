# Architecture

## Simulator: the game's own code, headless

`headless/` depends on `core/` and runs the real game logic. Changes made to `core` so it runs without a GL
context (none of them change gameplay in the desktop game):

| Change | Why |
|---|---|
| `Textures` loads lazily and returns `null` when `Gdx.gl == null`; `Player`/`Skeleton`/`Slime` use the shared textures instead of `new Texture(...)` per instance | Texture creation needs GL. Per-instance textures also leaked one GPU texture per mob per level in the game |
| `Sphere` creates its `Sprite` only when a texture exists | `Sprite` needs a texture |
| Frame logic moved from `GameScreen.render` to `Level.tick()` | Physics step, body cleanup, sphere buff timer, wizard fire, mob movement: the game and the simulator now run the same tick |
| `PlayerController.update(ControlInput, delta)`; keyboard fills a `ControlInput` | The old static key maps would be shared by every simulated level in one JVM |
| `Level(difficulty, ..., seed)`, `Map(difficulty, seed)`, seeded `LevelWriter`/`Maze` | Same seed → same maze, mobs and potions |
| `Level.nextLevel()` sets `completed` and returns when there is no `Cornichon` game | Reaching the door ends the episode instead of switching screens |
| `Level.addDying*` ignores duplicates | Two contacts in one step destroyed a body twice, which crashes Box2D |
| `CornichonListener`: fixture-A branch used `isDead()` (never set) instead of `checkDeath()` | Mobs hit on that side could never die. **Gameplay fix** |
| Debug `System.out.println` in the listener removed | Spam |
| `LevelWriter.initMap` calls `maze.updateGrid()` | The grid was only filled as a side effect of `maze.draw()` (a debug print) |

Not ported: the MongoDB leaderboard (`Database.java` is an in-memory stub since the Codex branch).

## Protocol

`HeadlessService` holds N independent `Simulation`s and talks JSON lines over stdin/stdout (so it exits with its
parent; game prints go to stderr). Ops: `reset`, `step` (all envs), `snapshot` (whole-level grid), `close`.
Python runs several services (default 16 levels each) and steps them concurrently.

## Observation, action

* `grid` 10×21×31 (channels: wall, spikes, mob, wizard, projectile, health potion, mana potion, door, sphere,
  fireball flying right), centred on the player. The last two arrived in v4 (`--grid-bytes=2`); older checkpoints
  read the first 8 channels and older clients still get one byte per cell.
* `state` 17: player velocity, health, mana, grounded, sphere offset and velocity, buffed, door offset, shortest-path
  distance to door, difficulty, sub-cell position, fraction of time limit used.
* Action `MultiDiscrete([3,2,2,3,3])`: move none/left/right, jump (a key press: first frame only), spell
  (buff costs 70 mana), sphere x none/left/right, sphere y none/up/down. Held for 4 frames → 15 decisions/s.
* Episode = one level: door → success, health ≤ 0 → death, time limit → truncated (bootstrapped in PPO). The limit is
  300 + 6 decisions per tile of shortest path from the start (capped at 1500), so bigger mazes get more time.

## Game mechanics that matter for learning

* The sphere is pulled toward the player and its velocity is zeroed on every frame no sphere key is held, so when
  idle it rests on the player's head (0.62 above) and **blocks jumps** (0.2 tiles instead of 4.9). Moving it aside
  or up first restores the full jump. Same code as the desktop game.
* Any brick contact counts as ground, side walls included, so the player can jump again while touching a wall.
* `scripts/scripted_bot.py` follows the BFS path with the sphere held up: a solvability check, not a strong player.

## Curriculum and seeds

Difficulty 1 → 10 (the game's levels). Promote when the last 200 episodes at the current difficulty reach 80%
success; 20% of episodes replay easier difficulties. Training seeds ∈ [0, 100000); evaluation seeds ≥ 1,000,000,
so evaluation always plays unseen mazes. Every 200 updates a sweep evaluates 20 such mazes at every difficulty up to
current + 1.

What difficulty changes (`LevelWriter`, mean over 20 mazes):

| difficulty | maze | path to door | mobs | wizards | spikes | potions |
|---|---|---|---|---|---|---|
| 1 | 29×17 | 55 | 6 | 2 | 2 | 4 |
| 3 | 37×25 | 81 | 12 | 4 | 4 | 7 |
| 5 | 45×29 | 105 | 19 | 7 | 7 | 8 |
| 7 | 53×37 | 133 | 38 | 13 | 13 | 8 |
| 9 | 61×41 | 163 | 67 | 25 | 23 | 7 |
| 10 | 65×45 | 175 | 131 | 53 | 46 | 0 |

**Combat practice (training only, v4):** `monster_practice` of training levels keep their maze but get the mob and
potion density of 1–3 difficulties higher (`LevelWriter(difficulty, monsterDifficulty, seed)`), so fighting pays off
long before the big levels. These episodes never count toward promotion; evaluation always uses the real game.

Difficulty 10 places a mob on every free floor spot (mob chance `1/(6 − d/2)` = 1), so no potions: the original
game's final level.
