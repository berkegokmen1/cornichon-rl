package com.cornichon.headless;

import com.badlogic.gdx.ApplicationListener;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.backends.lwjgl.LwjglApplication;
import com.badlogic.gdx.backends.lwjgl.LwjglApplicationConfiguration;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.physics.box2d.Box2D;
import com.badlogic.gdx.utils.BufferUtils;
import com.badlogic.gdx.utils.JsonReader;
import com.badlogic.gdx.utils.JsonValue;
import com.cornichon.views.LevelRenderer;
import java.io.BufferedReader;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

/**
 * The headless simulator's protocol (see HeadlessService) for one level, run inside a real libGDX window: after
 * every 1/60 s game frame the level is drawn with the game's own LevelRenderer (sprites, camera, health and mana
 * bars) and the frame is piped to ffmpeg, so a policy can be filmed playing the actual game. Python drives it
 * exactly like the training simulator (cornichon_rl/record.py). Needs a display; on a server use xvfb-run.
 *
 *   recorder --out=episode.mp4 [--width=1280 --height=720 --view-width=31 --view-height=21]
 *
 * The video starts at the first reset request and is finalized on close (or when stdin closes).
 */
public final class Recorder implements ApplicationListener {

  private final String out;
  private final int width, height, viewWidth, viewHeight;
  private final PrintStream protocol;

  private Simulation sim;
  private LevelRenderer renderer;
  private BufferedReader in;
  private final JsonReader reader = new JsonReader();
  private Process ffmpeg;
  private OutputStream video;
  private ByteBuffer pixels;
  private byte[] frame;

  private Recorder(String out, int width, int height, int viewWidth, int viewHeight, PrintStream protocol) {
    this.out = out;
    this.width = width;
    this.height = height;
    this.viewWidth = viewWidth;
    this.viewHeight = viewHeight;
    this.protocol = protocol;
  }

  public static void main(String[] args) throws Exception {
    String out = "episode.mp4";
    int width = 1280, height = 720, viewWidth = 31, viewHeight = 21;
    for (String arg : args) {
      if (arg.startsWith("--out=")) out = arg.substring(6);
      if (arg.startsWith("--width=")) width = Integer.parseInt(arg.substring(8));
      if (arg.startsWith("--height=")) height = Integer.parseInt(arg.substring(9));
      if (arg.startsWith("--view-width=")) viewWidth = Integer.parseInt(arg.substring(13));
      if (arg.startsWith("--view-height=")) viewHeight = Integer.parseInt(arg.substring(14));
    }
    PrintStream protocol = new PrintStream(new FileOutputStream(FileDescriptor.out), false, "UTF-8");
    System.setOut(System.err);

    LwjglApplicationConfiguration config = new LwjglApplicationConfiguration();
    config.title = "Cornichon recorder";
    config.width = width;
    config.height = height;
    config.resizable = false;
    config.vSyncEnabled = false;
    config.foregroundFPS = 0; // never sleep: frames are paced by the protocol, not the clock
    config.backgroundFPS = 0;
    config.forceExit = true;
    new LwjglApplication(new Recorder(out, width, height, viewWidth, viewHeight, protocol), config);
  }

  @Override
  public void create() {
    Box2D.init();
    sim = new Simulation(viewWidth, viewHeight, 2);
    sim.reset(0, 1, 1, false);
    pixels = BufferUtils.newByteBuffer(width * height * 4);
    frame = new byte[width * height * 4];
    in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
    protocol.println(String.format(
      "{\"ready\":true,\"envs\":1,\"view_width\":%d,\"view_height\":%d,\"state_size\":%d}",
      viewWidth, viewHeight, Simulation.STATE_SIZE
    ));
    protocol.flush();
  }

  /** One protocol request per libGDX frame; everything runs on the GL thread. */
  @Override
  public void render() {
    String line;
    try {
      line = in.readLine();
    } catch (IOException e) {
      line = null;
    }
    if (line == null) {
      finishVideo();
      Gdx.app.exit();
      return;
    }
    String response;
    boolean close = false;
    try {
      JsonValue request = reader.parse(line);
      String op = request.getString("op");
      close = op.equals("close");
      if (close) {
        finishVideo();
      }
      response = HeadlessService.handle(new Simulation[] {sim}, request);
      if (op.equals("reset")) {
        renderer = new LevelRenderer(sim.level(), false);
        startVideo();
        sim.onFrame = this::capture;
        capture(); // the level as it starts
      }
    } catch (Exception e) {
      e.printStackTrace();
      response = "{\"error\":\"" + HeadlessService.escape(String.valueOf(e)) + "\"}";
    }
    protocol.println(response);
    protocol.flush();
    if (close) {
      Gdx.app.exit();
    }
  }

  private void startVideo() throws IOException {
    if (ffmpeg != null) {
      return; // one video per process; later resets continue it
    }
    ffmpeg = new ProcessBuilder(
      "ffmpeg", "-y", "-loglevel", "error",
      "-f", "rawvideo", "-pix_fmt", "rgba", "-s", width + "x" + height, "-r", "60", "-i", "-",
      // OpenGL rows run bottom-up; hold the last frame for a second so the ending is visible
      "-vf", "vflip,tpad=stop_mode=clone:stop_duration=1",
      "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out
    ).redirectErrorStream(true).redirectOutput(ProcessBuilder.Redirect.INHERIT).start();
    video = ffmpeg.getOutputStream();
  }

  /** Draw the current game frame the way GameScreen does and send it to ffmpeg. */
  private void capture() {
    if (video == null) {
      return;
    }
    Gdx.gl.glViewport(0, 0, width, height);
    Gdx.gl.glClearColor(0.1f, 0.1f, 0.1f, 1);
    Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);
    renderer.render();
    pixels.clear();
    Gdx.gl.glPixelStorei(GL20.GL_PACK_ALIGNMENT, 1);
    Gdx.gl.glReadPixels(0, 0, width, height, GL20.GL_RGBA, GL20.GL_UNSIGNED_BYTE, pixels);
    pixels.get(frame);
    try {
      video.write(frame);
    } catch (IOException e) {
      throw new RuntimeException("ffmpeg stopped taking frames", e);
    }
  }

  private void finishVideo() {
    if (ffmpeg == null) {
      return;
    }
    try {
      video.close();
      ffmpeg.waitFor();
    } catch (IOException | InterruptedException e) {
      e.printStackTrace();
    }
    ffmpeg = null;
    video = null;
  }

  @Override
  public void resize(int width, int height) {}

  @Override
  public void pause() {}

  @Override
  public void resume() {}

  @Override
  public void dispose() {
    finishVideo();
    sim.dispose();
  }
}
