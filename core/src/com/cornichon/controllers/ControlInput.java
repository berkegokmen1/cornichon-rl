package com.cornichon.controllers;

/** Buttons held during one frame. Filled from the keyboard in the game, from agent actions in the headless simulator. */
public class ControlInput {

  public boolean left;
  public boolean right;
  /** Jump fires on the press, not while held, so this is true for a single frame. */
  public boolean jump;
  public boolean spell;

  public boolean sphereLeft;
  public boolean sphereRight;
  public boolean sphereUp;
  public boolean sphereDown;

  public void clear() {
    left = right = jump = spell = false;
    sphereLeft = sphereRight = sphereUp = sphereDown = false;
  }
}
