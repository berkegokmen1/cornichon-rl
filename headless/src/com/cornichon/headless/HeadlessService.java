package com.cornichon.headless;

import com.badlogic.gdx.physics.box2d.Box2D;
import com.badlogic.gdx.utils.JsonReader;
import com.badlogic.gdx.utils.JsonValue;
import java.io.BufferedReader;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

/**
 * Batched headless Cornichon simulator. Reads one JSON request per line on stdin, writes one JSON response
 * per line on stdout, and exits when stdin closes, so it can never outlive the Python process that started it.
 *
 * <pre>
 * {"op":"reset","ids":[0,3],"seeds":[17,42],"difficulties":[1,1]}  -> {"envs":[...]} for those ids
 *   optional "monster_difficulties":[4,1] (mob density) and "door_closed":[true,false] (combat practice);
 *   start with --grid-bytes=2 for sphere/fireball channels
 * {"op":"step","actions":[[m,j,s,sx,sy], ...one per env],"repeat":4} -> {"envs":[...]} for all envs
 * {"op":"snapshot","id":0}                                           -> whole-level grid for env 0
 * {"op":"close"}                                                     -> {"ok":true}, then exit
 * </pre>
 *
 * Errors come back as {"error":"..."} and leave the service running.
 */
public final class HeadlessService {

  public static void main(String[] args) throws Exception {
    int envs = 1, viewWidth = 31, viewHeight = 21, gridBytes = 1;
    for (String arg : args) {
      if (arg.startsWith("--envs=")) envs = Integer.parseInt(arg.substring(7));
      if (arg.startsWith("--view-width=")) viewWidth = Integer.parseInt(arg.substring(13));
      if (arg.startsWith("--view-height=")) viewHeight = Integer.parseInt(arg.substring(14));
      if (arg.startsWith("--grid-bytes=")) gridBytes = Integer.parseInt(arg.substring(13));
    }

    // stdout carries the protocol; anything the game code prints goes to stderr instead.
    PrintStream protocol = new PrintStream(new FileOutputStream(FileDescriptor.out), false, "UTF-8");
    System.setOut(System.err);
    Box2D.init();

    Simulation[] sims = new Simulation[envs];
    for (int i = 0; i < envs; i++) {
      sims[i] = new Simulation(viewWidth, viewHeight, gridBytes);
      sims[i].reset(i, 1, 1, false);
    }
    protocol.println(String.format(
      "{\"ready\":true,\"envs\":%d,\"view_width\":%d,\"view_height\":%d,\"state_size\":%d}",
      envs, viewWidth, viewHeight, Simulation.STATE_SIZE
    ));
    protocol.flush();

    BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
    JsonReader reader = new JsonReader();
    String line;
    while ((line = in.readLine()) != null) {
      String response;
      boolean close = false;
      try {
        JsonValue request = reader.parse(line);
        String op = request.getString("op");
        if (op.equals("reset")) {
          int[] ids = request.get("ids").asIntArray();
          long[] seeds = request.get("seeds").asLongArray();
          int[] difficulties = request.get("difficulties").asIntArray();
          // optional: denser mobs than the maze size implies (training only); defaults to the game's density
          int[] monsters = request.has("monster_difficulties") ? request.get("monster_difficulties").asIntArray() : difficulties;
          // optional: the door does nothing (combat practice; the level ends when every mob is dead)
          boolean[] closed = new boolean[ids.length];
          if (request.has("door_closed")) closed = request.get("door_closed").asBooleanArray();
          for (int i = 0; i < ids.length; i++) sims[ids[i]].reset(seeds[i], difficulties[i], monsters[i], closed[i]);
          response = envsJson(sims, ids);
        } else if (op.equals("step")) {
          JsonValue actions = request.get("actions");
          int repeat = request.getInt("repeat", 4);
          if (actions.size != envs) throw new IllegalArgumentException("need " + envs + " actions, got " + actions.size);
          int i = 0;
          for (JsonValue a = actions.child; a != null; a = a.next, i++) {
            int[] x = a.asIntArray();
            sims[i].step(x[0], x[1], x[2], x[3], x[4], repeat);
          }
          response = envsJson(sims, null);
        } else if (op.equals("snapshot")) {
          response = sims[request.getInt("id")].snapshotJson();
        } else if (op.equals("close")) {
          response = "{\"ok\":true}";
          close = true;
        } else {
          throw new IllegalArgumentException("unknown op " + op);
        }
      } catch (Exception e) {
        e.printStackTrace();
        response = "{\"error\":\"" + escape(String.valueOf(e)) + "\"}";
      }
      protocol.println(response);
      protocol.flush();
      if (close) break;
    }
    for (Simulation sim : sims) sim.dispose();
  }

  private static String escape(String text) {
    StringBuilder out = new StringBuilder();
    for (char c : text.toCharArray()) {
      if (c == '"' || c == '\\') out.append('\\').append(c); else if (c < 0x20) out.append(' '); else out.append(c);
    }
    return out.toString();
  }

  private static String envsJson(Simulation[] sims, int[] ids) {
    StringBuilder json = new StringBuilder("{\"envs\":[");
    int n = ids == null ? sims.length : ids.length;
    for (int i = 0; i < n; i++) {
      if (i > 0) json.append(',');
      json.append(sims[ids == null ? i : ids[i]].toJson());
    }
    return json.append("]}").toString();
  }
}
