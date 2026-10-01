package com.cornichon.views.screens;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input.Keys;
import com.badlogic.gdx.Screen;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.scenes.scene2d.ui.ImageButton;
import com.cornichon.Cornichon;
import com.cornichon.controllers.PlayerController;
import com.cornichon.models.construction.Level;
import com.cornichon.views.LevelRenderer;
import com.cornichon.views.PauseRenderer;

public class GameScreen implements Screen {

  private Level level;
  private LevelRenderer renderer;
  private PauseRenderer pauseRenderer;
  private Cornichon game;
  public ImageButton soundButton;

  /** controllers */
  private PlayerController playerController;
  private int difficulty;
  private int lastScore;
  private float lastHealth;

  public GameScreen(Cornichon game, int difficulty, int lastScore, float lastHealth) {
    this.game = game;
    this.difficulty = difficulty;
    this.lastScore = lastScore;
    this.lastHealth = lastHealth;
  }

  @Override
  public void show() {
    this.level = new Level(this.difficulty, this.lastScore, this.lastHealth, game);
    this.renderer = new LevelRenderer(level, false);
    this.pauseRenderer = new PauseRenderer(game.batch);
    this.playerController = new PlayerController(this.level);
    Gdx.input.setInputProcessor(playerController);
  }

  @Override
  public void render(float delta) {
    Gdx.gl.glClearColor(0.1f, 0.1f, 0.1f, 1);
    Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);

    if (!game.getPaused()) {
      playerController.update(delta);
      level.tick();
      renderer.render();

      if (level.getPlayer().isDead()) {
        game.setScreen(new GameOverScreen(game, level.getLatestScore()));
      }
    } else {
      pauseRenderer.render();

      if (Gdx.input.isKeyJustPressed(Keys.R)) {
        this.resume();
      } else if (Gdx.input.isKeyJustPressed(Keys.M)) {
        this.dispose();
      }
    }
  }

  @Override
  public void resize(int width, int height) {
    renderer.setSize(width, height);
  }

  @Override
  public void pause() {
    game.setPaused(true);
  }

  @Override
  public void resume() {
    game.setPaused(false);
    Gdx.input.setInputProcessor(playerController);
  }

  @Override
  public void hide() {
    Gdx.input.setInputProcessor(null);
  }

  @Override
  public void dispose() {
    Gdx.input.setInputProcessor(null);
    game.setScreen(new MainMenuScreen(game));
  }
}
