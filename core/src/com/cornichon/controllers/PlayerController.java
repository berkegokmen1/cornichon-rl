package com.cornichon.controllers;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input.Keys;
import com.badlogic.gdx.math.Vector2;
import com.cornichon.models.construction.Level;
import com.cornichon.models.entities.aliveEntities.Player;
import com.cornichon.models.entities.aliveEntities.Sphere;
import com.cornichon.models.entities.helpers.State;
import com.cornichon.utils.CornichonListener;
import com.cornichon.views.textures.Textures;

public class PlayerController extends GeneralController {

  private Player player;
  private Sphere sphere;
  private Level level;
  private CornichonListener listener;

  // Static so a key held through the door is still held in the next level, as before.
  private static final ControlInput keyboard = new ControlInput();

  public PlayerController(Level level) {
    this.level = level;
    this.player = this.level.getPlayer();
    this.sphere = this.player.getSphere();
    this.listener = this.level.getListener();
  }

  /** The main update method **/

  public void update(float delta) {
    keyboard.jump = Gdx.input.isKeyJustPressed(Keys.SPACE);
    this.update(keyboard, delta);
  }

  public void update(ControlInput input, float delta) {
    this.processInput(input);
    this.player.update(delta);
  }

  /** Change playerplayer's state and parameters based on input controls **/

  private void processInput(ControlInput input) {
    if (input.jump && listener.getGroundContacts() > 0) {
      player.getBody().applyForceToCenter(0, 600f, true);
      player.setTexture(Textures.PLAYER_JUMPING);
      player.setState(State.JUMPING);
    }

    if (input.left) {
      // left is pressed
      player.setFacingLeft(true);
      player.setState(State.WALKING);

      player.getBody().setLinearVelocity(new Vector2(-Player.SPEED, player.getBody().getLinearVelocity().y));
    }

    if (input.right) {
      // right is pressed
      player.setFacingLeft(false);
      player.setState(State.WALKING);

      player.getBody().setLinearVelocity(new Vector2(Player.SPEED, player.getBody().getLinearVelocity().y));
    }

    if (
      (input.left && input.right) || (!input.left && !(input.right))
    ) {
      player.setState(State.IDLE);

      // horizontal speed is 0
      player.getBody().setLinearVelocity(new Vector2(0, player.getBody().getLinearVelocity().y));
    }

    //Buff Spell
    if (input.spell) {
      if (player.getMana() >= 70) {
        player.setMana(player.getMana() - 70);
        player.getSphere().setBuffed(true);
      }
    }

    // Sphere's actions

    if (input.sphereLeft) {
      sphere.setFacingLeft(true);
      sphere.setState(State.WALKING);
      sphere.rotate(30f);
      // sphere.getBody().setLinearVelocity(new Vector2(-Sphere.SPEED,
      // sphere.getBody().getLinearVelocity().y));
      if (Math.abs(sphere.getBody().getLinearVelocity().x) <= sphere.getMaxSpeed()) sphere
        .getBody()
        .applyLinearImpulse(new Vector2(-2f, 0), sphere.getBody().getPosition(), true);
    }

    if (input.sphereRight) {
      // right is pressed
      sphere.setFacingLeft(false);
      sphere.setState(State.WALKING);
      sphere.rotate(-30f);
      // sphere.getBody().setLinearVelocity(new Vector2(Sphere.SPEED,
      // sphere.getBody().getLinearVelocity().y));
      if (Math.abs(sphere.getBody().getLinearVelocity().x) <= sphere.getMaxSpeed()) sphere
        .getBody()
        .applyLinearImpulse(new Vector2(2f, 0), sphere.getBody().getPosition(), true);
    }

    if (input.sphereUp) {
      sphere.setFacingLeft(false);
      sphere.setState(State.JUMPING);
      // sphere.getBody().setLinearVelocity(new Vector2(sphere.getBody().getLinearVelocity().x, Sphere.SPEED));
      if (Math.abs(sphere.getBody().getLinearVelocity().x) <= sphere.getMaxSpeed()) sphere
        .getBody()
        .applyLinearImpulse(new Vector2(0, 2f), sphere.getBody().getPosition(), true);
    }

    if (input.sphereDown) {
      sphere.setFacingLeft(false);
      sphere.setState(State.FALLING);
      // sphere.getBody().setLinearVelocity(new Vector2(sphere.getBody().getLinearVelocity().x, -Sphere.SPEED));
      if (Math.abs(sphere.getBody().getLinearVelocity().x) <= sphere.getMaxSpeed()) sphere
        .getBody()
        .applyLinearImpulse(new Vector2(0, -2f), sphere.getBody().getPosition(), true);
    }

    if (
      (input.sphereLeft && input.sphereRight) ||
      (!input.sphereLeft && !(input.sphereRight))
    ) {
      sphere.setState(State.IDLE);
      sphere.getBody().setLinearVelocity(new Vector2(0, sphere.getBody().getLinearVelocity().y));
    }

    if (
      (input.sphereUp && input.sphereDown) ||
      ((!input.sphereUp) && (!input.sphereDown))
    ) {
      sphere.setState(State.IDLE);
      sphere.getBody().setLinearVelocity(new Vector2(sphere.getBody().getLinearVelocity().x, 0));
    }

    sphere
      .getBody()
      .applyForceToCenter(
        new Vector2(
          30 * (player.getBody().getPosition().x - sphere.getBody().getPosition().x),
          40 * (player.getBody().getPosition().y - sphere.getBody().getPosition().y)
        ),
        true
      );
  }

  @Override
  public boolean keyDown(int keycode) {
    if (keycode == Keys.A) keyboard.left = true;
    if (keycode == Keys.D) keyboard.right = true;

    if (keycode == Keys.Z) keyboard.spell = true;

    if (keycode == Keys.LEFT) keyboard.sphereLeft = true;
    if (keycode == Keys.RIGHT) keyboard.sphereRight = true;
    if (keycode == Keys.UP) keyboard.sphereUp = true;
    if (keycode == Keys.DOWN) keyboard.sphereDown = true;

    return true;
  }

  @Override
  public boolean keyUp(int keycode) {
    if (keycode == Keys.A) keyboard.left = false;
    if (keycode == Keys.D) keyboard.right = false;
    if (keycode == Keys.Z) keyboard.spell = false;

    if (keycode == Keys.LEFT) keyboard.sphereLeft = false;
    if (keycode == Keys.RIGHT) keyboard.sphereRight = false;
    if (keycode == Keys.UP) keyboard.sphereUp = false;
    if (keycode == Keys.DOWN) keyboard.sphereDown = false;

    return true;
  }
}
