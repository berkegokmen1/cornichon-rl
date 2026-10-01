package com.cornichon.views.maps;

import com.cornichon.utils.LevelWriter;

public class Map {

  private int[][] map;
  private LevelWriter levelWriter;

  public Map(int difficulty) {
    this.map = new int[70][100];
    this.levelWriter = new LevelWriter(difficulty);
  }

  /** Same level every time for the same (difficulty, seed). */
  public Map(int difficulty, long seed) {
    this(difficulty, difficulty, seed);
  }

  public Map(int difficulty, int monsterDifficulty, long seed) {
    this.map = new int[70][100];
    this.levelWriter = new LevelWriter(difficulty, monsterDifficulty, seed);
  }

  public void processMap() {
    levelWriter.initMap(map);
    levelWriter.placeMobsAndCollectibles(map);
    levelWriter.placePlayer(map);
    levelWriter.placeExitDoor(map);
  }

  public int[][] getMapIntArr() {
    return this.map;
  }
}
