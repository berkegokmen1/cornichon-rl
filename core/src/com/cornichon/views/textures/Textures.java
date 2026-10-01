package com.cornichon.views.textures;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.Texture;

public class Textures {

  /** Loads a texture, or returns null when there is no GL context (the headless RL simulator). */
  private static Texture load(String path) {
    if (Gdx.gl == null) {
      return null;
    }
    return new Texture(Gdx.files.internal(path));
  }

  public static final Texture BACKGROUND_BRICK = load("images/background_brick_block.jpg");
  public static final Texture BRICK = load("images/brick_block.jpg");
  public static final Texture PLAYER_WALKING = load("images/walking.png");
  public static final Texture PLAYER_WALKING2 = load("images/walking2.png");
  public static final Texture PLAYER_JUMPING = load("images/jumping.png");
  public static final Texture PLAYER_IDLE = load("images/idle.png");
  public static final Texture PLAYER_WALKINGLEFT = load("images/walkingLEFT.png");
  public static final Texture PLAYER_WALKING2LEFT = load("images/walking2LEFT.png");
  public static final Texture PLAYER_IDLELEFT = load("images/idleLEFT.png");
  public static final Texture DOOR_CLOSED = load("images/doorclosed.png");
  public static final Texture DOOR_OPENED = load("images/dooropened.png");
  public static final Texture SPHERE = load("images/purpleBall.png");
  public static final Texture GREEN_SLIME = load("images/greenSlime.png");
  public static final Texture BLUE_SLIME = load("images/blueSlime.png");
  public static final Texture YELLOW_SLIME = load("images/yellowSlime.png");
  public static final Texture SPHERE_BUFFED = load("images/sphere_buffed.png");

  public static final Texture SKELETON_IDLE = load("images/skeleton_idle.png");
  public static final Texture SKELETON_LEFT1 = load("images/SKELETON_walking.png");
  public static final Texture SKELETON_LEFT2 = load("images/skeleton_left2.png");
  public static final Texture SKELETON_RIGHT1 = load("images/skeleton_right1.png");
  public static final Texture SKELETON_RIGHT2 = load("images/skeleton_right2.png");

  public static final Texture SOUND_ON = load("images/ON.png");
  public static final Texture SOUND_OFF = load("images/SOUND_OFF.png");
  public static final Texture MUSIC_ON = load("images/musicON.png");
  public static final Texture MUSIC_OFF = load("images/musicOFF.png");

  public static final Texture POTIONS_HEALTH = load("images/health_potion.png");
  public static final Texture POTIONS_MANA = load("images/mana_potion.png");

  public static final Texture SPIKES = load("images/spikes.png");

  public static final Texture FIREBALL = load("images/fireball.png");

  public static final Texture WIZARD = load("images/wizard.png");
}
