package com.cornichon.utils;

import com.badlogic.gdx.utils.Array;

/** Offline-only leaderboard. Historical remote MongoDB is intentionally unsupported. */
public final class Database {
  private static final Array<String> entries = new Array<String>();
  private Database() {}
  public static void initDatabase() { }
  public static void insert(final String name, final int score) { entries.add(name); entries.add(String.valueOf(score)); }
  public static Array<String> read() { return new Array<String>(entries); }
}
