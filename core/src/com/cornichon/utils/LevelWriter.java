package com.cornichon.utils;

import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.utils.Array;
import com.cornichon.models.construction.background.BackgroundBrick;
import com.cornichon.models.entities.Entity;
import com.cornichon.views.helpers.DrawableValues;
import java.util.Random;

public final class LevelWriter {

  public Maze maze;
  private int difficulty;
  private int monsterDifficulty; // sets mob and potion density; equals difficulty in the game
  private Random random;

  public LevelWriter(int difficulty) {
    this(difficulty, new Random().nextLong());
  }

  public LevelWriter(int difficulty, long seed) {
    this(difficulty, difficulty, seed);
  }

  /** Maze size from difficulty, mob/potion density from monsterDifficulty (RL training uses denser mazes). */
  public LevelWriter(int difficulty, int monsterDifficulty, long seed) {
    int d = difficulty + 6;
    this.difficulty = difficulty;
    this.monsterDifficulty = monsterDifficulty;
    this.random = new Random(seed ^ 0x9e3779b97f4a7c15L);
    this.maze = new Maze(d, d * 7 / 10, seed);
  }

  public void initMap(int[][] map) {
    this.maze.solve();
    this.maze.updateGrid(); // fills the grid; the game used to get this as a side effect of maze.draw()
    final char[][] mazeArr = maze.getGridArr();

    for (int r = 0; r < mazeArr.length; r += 1) {
      for (int c = 0; c < mazeArr[0].length; c += 1) {
        char current = mazeArr[r][c];
        if (current == 'X') {
          map[2 * c][r] = DrawableValues.BRICK;
          if (c < mazeArr[0].length - 1) {
            if (mazeArr[r][c + 1] == 'X') {
              map[2 * c + 1][r] = DrawableValues.BRICK;
            }
          }
        }
      }
    }
  }

  public void placePlayer(int[][] map) {
    map[2][1] = DrawableValues.PLAYER;
  }

  public void placeMobsAndCollectibles(int[][] map) {
    int[] collectibles = { DrawableValues.POTION_HEALTH, DrawableValues.POTION_MANA };
    int[] mobs = {
      DrawableValues.SKELETON, // Skeleton x3 chance
      DrawableValues.SKELETON,
      DrawableValues.SKELETON,
      DrawableValues.SLIME, // Slime x3 chance
      DrawableValues.SLIME,
      DrawableValues.SLIME,
      DrawableValues.SPIKES, // Spikes x2
      DrawableValues.SPIKES,
      DrawableValues.WIZARD, // Wizard x2
      DrawableValues.WIZARD,
    };

    int mobCount = 0;
    int collectibleCount = 0;

    int collectibleTestModifier = 10;
    int mobTestModifier = 6;

      for (int r = 1; r < map.length; r += 1) {
        for (int c = 1; c < map[0].length - 1; c += 1) {
          if (map[r][c] == DrawableValues.BRICK && map[r - 1][c] != DrawableValues.BRICK) {
            if (map[r][c - 1] == DrawableValues.BRICK && map[r][c + 1] == DrawableValues.BRICK) {
              if (map[r - 1][c] == DrawableValues.AIR) {
                if (mobCount > 5) mobTestModifier = 12;
                if (mobCount > 7) mobTestModifier = 16;

                if (collectibleCount > 3) collectibleTestModifier = 12;
                if (collectibleCount > 5) collectibleTestModifier = 16;

                boolean testCollectible = random.nextInt(monsterDifficulty / 2 + collectibleTestModifier) == 0; // the chance of placing decreases with the difficulty
                boolean testMob = random.nextInt(mobTestModifier - monsterDifficulty / 2) == 0; // the chance of placing increases with the difficulty

                if (testMob) {
                  int mobValue = mobs[random.nextInt(mobs.length)];

                  if (mobValue == DrawableValues.WIZARD) {
                    // Empty left and right of the wizard
                    if (map[r - 1][c - 1] != DrawableValues.BRICK) {
                      map[r - 1][c - 1] = DrawableValues.AIR;
                    }
                    map[r - 1][c] = DrawableValues.WIZARD;
                    c += 1;
                  } else {
                    map[r - 1][c] = mobValue;
                  }
                } else if (testCollectible) {
                  map[r - 1][c] = collectibles[random.nextInt(collectibles.length)];
                }
              }
            }
          }
        }
      }
  }

  public void placeExitDoor(int[][] map) {
    final char[][] mazeArr = maze.getGridArr();
    map[mazeArr[0].length * 2 - 3][mazeArr.length - 2] = DrawableValues.DOOR_CLOSED;
  }

  public void fillBackground(int[][] map, Array<Entity> entities) {
    final char[][] mazeArr = maze.getGridArr();

    for (int r = 0; r < mazeArr[0].length * 2 - 1; r += 1) {
      for (int c = 0; c < mazeArr.length; c += 1) {
        if (map[r][c] != 1) {
          entities.add(new BackgroundBrick(new Vector2(c, map.length - r - 1)));
        }
      }
    }
  }
}
