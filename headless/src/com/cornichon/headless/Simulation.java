package com.cornichon.headless;

import com.badlogic.gdx.math.Vector2;
import com.cornichon.controllers.ControlInput;
import com.cornichon.controllers.PlayerController;
import com.cornichon.models.construction.Level;
import com.cornichon.models.entities.Entity;
import com.cornichon.models.entities.aliveEntities.Mob;
import com.cornichon.models.entities.aliveEntities.Player;
import com.cornichon.models.entities.aliveEntities.Sphere;
import com.cornichon.models.entities.aliveEntities.Wizard;
import com.cornichon.models.entities.collectibles.HealthPotion;
import com.cornichon.models.entities.collectibles.ManaPotion;
import com.cornichon.models.entities.helpers.Collectible;
import com.cornichon.models.entities.projectiles.Projectile;
import com.cornichon.utils.Constants;
import com.cornichon.views.helpers.DrawableValues;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Locale;

/**
 * One Cornichon level running the real game code (Level, Box2D world, contact listener, PlayerController)
 * without a renderer. Frames are the game's own fixed 1/60 s ticks; one agent step repeats an action for
 * several frames.
 */
final class Simulation {

  static final float FRAME = 1f / 60f;

  // One bit per grid channel; Python unpacks them in the same order (cornichon_rl/service.py).
  static final int WALL = 1, SPIKES = 1 << 1, MOB = 1 << 2, WIZARD = 1 << 3;
  static final int PROJECTILE = 1 << 4, HEALTH_POTION = 1 << 5, MANA_POTION = 1 << 6, DOOR = 1 << 7;
  // Second byte, sent only with --grid-bytes=2 (older clients get the first byte, unchanged).
  static final int SPHERE = 1 << 8, PROJECTILE_RIGHTWARD = 1 << 9;

  static final int STATE_SIZE = 16;
  static final int UNREACHABLE = Integer.MAX_VALUE;

  private final int viewWidth;
  private final int viewHeight;
  private final int gridBytes;
  private final ControlInput input = new ControlInput();

  private Level level;
  private PlayerController controller;
  private int[][] map;
  private int[][] doorDistance;

  private long seed;
  private int difficulty;
  private int monsterDifficulty;
  private int frames;
  private int mobsAtStart;
  private final List<Mob> mobs = new ArrayList<Mob>(); // every mob of the level, dead ones included
  private int collectiblesAtStart;
  private float damageTaken;

  Simulation(int viewWidth, int viewHeight, int gridBytes) {
    this.viewWidth = viewWidth;
    this.viewHeight = viewHeight;
    this.gridBytes = gridBytes;
  }

  void reset(long seed, int difficulty, int monsterDifficulty) {
    if (level != null) {
      level.dispose();
    }
    this.seed = seed;
    this.difficulty = difficulty;
    this.monsterDifficulty = monsterDifficulty;
    level = new Level(difficulty, monsterDifficulty, Constants.PLAYER_HEALTH, seed);
    controller = new PlayerController(level);
    map = level.getMap().getMapIntArr();
    doorDistance = distancesToDoor();
    frames = 0;
    damageTaken = 0;
    mobsAtStart = countMobs();
    mobs.clear();
    for (Entity e : level.getEntities()) if (e instanceof Mob) mobs.add((Mob) e);
    collectiblesAtStart = countCollectibles();
  }

  /**
   * move: 0 none, 1 left, 2 right. jump/spell: 0/1. sphereX: 0 none, 1 left, 2 right. sphereY: 0 none, 1 up, 2 down.
   * Jump is a key press, so it only fires on the first frame of the step.
   */
  void step(int move, int jump, int spell, int sphereX, int sphereY, int repeat) {
    if (isDone()) {
      return;
    }
    Player player = level.getPlayer();
    for (int frame = 0; frame < repeat; frame++) {
      input.clear();
      input.left = move == 1;
      input.right = move == 2;
      input.jump = jump == 1 && frame == 0;
      input.spell = spell == 1;
      input.sphereLeft = sphereX == 1;
      input.sphereRight = sphereX == 2;
      input.sphereUp = sphereY == 1;
      input.sphereDown = sphereY == 2;

      float healthBefore = player.getHealth();
      controller.update(input, FRAME);
      level.tick();
      frames++;
      damageTaken += Math.max(0f, healthBefore - player.getHealth());
      if (isDone()) {
        break;
      }
    }
  }

  boolean isDone() {
    return level.isCompleted() || level.getPlayer().isDead();
  }

  void dispose() {
    if (level != null) {
      level.dispose();
      level = null;
    }
  }

  /** The step result as one JSON object. The grid is viewHeight x viewWidth bytes, row 0 at the top, base64. */
  String toJson() {
    Player player = level.getPlayer();
    StringBuilder json = new StringBuilder(4096);
    json.append("{\"grid\":\"").append(java.util.Base64.getEncoder().encodeToString(encode(view()))).append("\",\"state\":[");
    float[] state = state();
    for (int i = 0; i < state.length; i++) {
      if (i > 0) json.append(',');
      json.append(String.format(Locale.US, "%.4f", state[i]));
    }
    json.append(String.format(
      Locale.US,
      "],\"completed\":%b,\"dead\":%b,\"frames\":%d,\"mobs_killed\":%d,\"collected\":%d,\"damage\":%.1f," +
      "\"damage_dealt\":%d,\"health\":%.1f,\"score\":%d,\"door_distance\":%d,\"x\":%.3f,\"y\":%.3f,\"seed\":%d,\"difficulty\":%d,\"monster_difficulty\":%d}",
      level.isCompleted(),
      player.isDead(),
      frames,
      mobsAtStart - countMobs(),
      collectiblesAtStart - countCollectibles(),
      damageTaken,
      damageDealt(),
      player.getHealth(),
      level.getLatestScore(),
      playerDoorDistance(),
      player.getBody().getPosition().x,
      player.getBody().getPosition().y,
      seed,
      difficulty,
      monsterDifficulty
    ));
    return json.toString();
  }

  /** Whole level as a grid (for pictures and debugging), plus where the player and sphere are. */
  String snapshotJson() {
    int rows = map.length, cols = map[0].length;
    int maxRow = 0, maxCol = 0;
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        if (isBrick(map[r][c])) {
          maxRow = Math.max(maxRow, r);
          maxCol = Math.max(maxCol, c);
        }
      }
    }
    int height = maxRow + 1, width = maxCol + 1;
    int[] grid = new int[height * width];
    for (int r = 0; r < height; r++) {
      for (int c = 0; c < width; c++) {
        grid[r * width + c] = staticCell(c, rows - 1 - r);
      }
    }
    addEntities(grid, width, height, 0, rows - 1);
    Vector2 p = level.getPlayer().getBody().getPosition();
    Vector2 s = level.getSphere().getBody().getPosition();
    return String.format(
      Locale.US,
      "{\"grid\":\"%s\",\"width\":%d,\"height\":%d,\"top_y\":%d,\"player\":[%.3f,%.3f],\"sphere\":[%.3f,%.3f]}",
      java.util.Base64.getEncoder().encodeToString(encode(grid)),
      width,
      height,
      rows - 1,
      p.x,
      p.y,
      s.x,
      s.y
    );
  }

  /** One byte per cell, or two (little-endian) with --grid-bytes=2. */
  private byte[] encode(int[] grid) {
    byte[] out = new byte[grid.length * gridBytes];
    for (int i = 0; i < grid.length; i++) {
      out[i * gridBytes] = (byte) grid[i];
      if (gridBytes == 2) out[i * 2 + 1] = (byte) (grid[i] >> 8);
    }
    return out;
  }

  private int[] view() {
    Vector2 p = level.getPlayer().getBody().getPosition();
    int left = Math.round(p.x) - viewWidth / 2;
    int top = Math.round(p.y) + viewHeight / 2;
    int[] grid = new int[viewWidth * viewHeight];
    for (int r = 0; r < viewHeight; r++) {
      for (int c = 0; c < viewWidth; c++) {
        grid[r * viewWidth + c] = staticCell(left + c, top - r);
      }
    }
    addEntities(grid, viewWidth, viewHeight, left, top);
    return grid;
  }

  /** Bricks and spikes never move, so they come from the generated map. Outside the map counts as wall. */
  private int staticCell(int x, int y) {
    int row = map.length - 1 - y;
    if (row < 0 || row >= map.length || x < 0 || x >= map[0].length) {
      return WALL;
    }
    int value = map[row][x];
    if (isBrick(value)) return WALL;
    if (value == DrawableValues.SPIKES) return SPIKES;
    return 0;
  }

  /** Mobs, potions, projectiles and the door, at the cell their body centre rounds to. */
  private void addEntities(int[] grid, int width, int height, int left, int top) {
    for (Entity e : level.getEntities()) {
      int bit;
      if (e instanceof Wizard) bit = WIZARD; else if (e instanceof Mob) bit = MOB; else if (
        e instanceof HealthPotion
      ) bit = HEALTH_POTION; else if (e instanceof ManaPotion) bit = MANA_POTION; else continue;
      mark(grid, width, height, left, top, e.getBody().getPosition(), bit);
    }
    for (Projectile p : level.getProjectiles()) {
      int bits = PROJECTILE | (p.getBody().getLinearVelocity().x > 0 ? PROJECTILE_RIGHTWARD : 0);
      mark(grid, width, height, left, top, p.getBody().getPosition(), bits);
    }
    mark(grid, width, height, left, top, level.getDoor().getBody().getPosition(), DOOR);
    mark(grid, width, height, left, top, level.getSphere().getBody().getPosition(), SPHERE);
  }

  private static void mark(int[] grid, int width, int height, int left, int top, Vector2 at, int bit) {
    int c = Math.round(at.x) - left;
    int r = top - Math.round(at.y);
    if (r >= 0 && r < height && c >= 0 && c < width) {
      grid[r * width + c] |= bit;
    }
  }

  private float[] state() {
    Player player = level.getPlayer();
    Sphere sphere = level.getSphere();
    Vector2 p = player.getBody().getPosition();
    Vector2 v = player.getBody().getLinearVelocity();
    Vector2 s = sphere.getBody().getPosition();
    Vector2 sv = sphere.getBody().getLinearVelocity();
    Vector2 door = level.getDoor().getBody().getPosition();
    int distance = playerDoorDistance();
    return new float[] {
      v.x / Player.SPEED,
      v.y / 10f,
      player.getHealth() / 100f,
      player.getMana() / 100f,
      level.getListener().getGroundContacts() > 0 ? 1f : 0f,
      (s.x - p.x) / 5f,
      (s.y - p.y) / 5f,
      sv.x / 10f,
      sv.y / 10f,
      sphere.getBuffed() ? 1f : 0f,
      (door.x - p.x) / 30f,
      (door.y - p.y) / 30f,
      distance == UNREACHABLE ? 1f : Math.min(distance / 100f, 1f),
      difficulty / 10f,
      p.x - Math.round(p.x),
      p.y - Math.round(p.y),
    };
  }

  /** Shortest 4-connected path through non-brick cells from every map cell to the door (ignores gravity). */
  private int[][] distancesToDoor() {
    int rows = map.length, cols = map[0].length;
    int[][] distance = new int[rows][cols];
    for (int[] row : distance) Arrays.fill(row, UNREACHABLE);
    Vector2 door = level.getDoor().getPosition();
    int startRow = rows - 1 - Math.round(door.y), startCol = Math.round(door.x);
    ArrayDeque<int[]> queue = new ArrayDeque<int[]>();
    distance[startRow][startCol] = 0;
    queue.add(new int[] { startRow, startCol });
    int[][] moves = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
    while (!queue.isEmpty()) {
      int[] cell = queue.poll();
      for (int[] m : moves) {
        int r = cell[0] + m[0], c = cell[1] + m[1];
        if (r < 0 || r >= rows || c < 0 || c >= cols) continue;
        if (isBrick(map[r][c]) || distance[r][c] != UNREACHABLE) continue;
        distance[r][c] = distance[cell[0]][cell[1]] + 1;
        queue.add(new int[] { r, c });
      }
    }
    return distance;
  }

  /** Door distance at the player's cell; if that cell rounds into a brick, the best neighbour plus one. */
  private int playerDoorDistance() {
    Vector2 p = level.getPlayer().getBody().getPosition();
    int row = map.length - 1 - Math.round(p.y), col = Math.round(p.x);
    int best = UNREACHABLE;
    for (int dr = -1; dr <= 1; dr++) {
      for (int dc = -1; dc <= 1; dc++) {
        int r = row + dr, c = col + dc;
        if (r < 0 || r >= map.length || c < 0 || c >= map[0].length) continue;
        int d = doorDistance[r][c];
        if (d == UNREACHABLE) continue;
        best = Math.min(best, dr == 0 && dc == 0 ? d : d + 1);
      }
    }
    return best;
  }

  private static boolean isBrick(int value) {
    return value == DrawableValues.BRICK || value == DrawableValues.BRICK_PLATFORM;
  }

  /** Sphere damage dealt to mobs so far (a dead mob counts its full health once). */
  private int damageDealt() {
    int total = 0;
    for (Mob m : mobs) total += Constants.MOB_HEALTH_GENERAL - Math.max(0, m.getHealth());
    return total;
  }

  private int countMobs() {
    int n = 0;
    for (Entity e : level.getEntities()) if (e instanceof Mob) n++;
    return n;
  }

  private int countCollectibles() {
    int n = 0;
    for (Entity e : level.getEntities()) if (e instanceof Collectible) n++;
    return n;
  }
}
