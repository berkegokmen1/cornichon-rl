"""Client for the headless Java simulator (headless/src/com/cornichon/headless/HeadlessService.java).

One Java process holds several independent Cornichon levels. Requests and responses are JSON lines over the
process's stdin/stdout, so the simulator exits as soon as this side closes the pipe or dies.
"""

import base64
import json
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LAUNCHER = REPO / "headless" / "build" / "install" / "headless" / "bin" / "headless"

# Bit order of the grid bytes, matching Simulation.java.
GRID_CHANNELS = (
    "wall", "spikes", "mob", "wizard", "projectile", "health_potion", "mana_potion", "door",
    "sphere", "projectile_rightward",  # second grid byte; models from before v4 use only the first 8 channels
)

# Action heads: move (none/left/right), jump, spell, sphere x (none/left/right), sphere y (none/up/down).
ACTION_NVEC = (3, 2, 2, 3, 3)
ACTION_NAMES = ("move", "jump", "spell", "sphere_x", "sphere_y")


class SimulatorError(RuntimeError):
    pass


class SimService:
    def __init__(self, num_envs, view_width=31, view_height=21, launcher=LAUNCHER):
        if not Path(launcher).exists():
            raise FileNotFoundError(f"{launcher} missing; build it with ./gradlew :headless:installDist")
        self.num_envs = num_envs
        self.proc = subprocess.Popen(
            [str(launcher), f"--envs={num_envs}", f"--view-width={view_width}", f"--view-height={view_height}", "--grid-bytes=2"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=1 << 20,
        )
        hello = self._read()
        self.view_width = hello["view_width"]
        self.view_height = hello["view_height"]
        self.state_size = hello["state_size"]

    def _send(self, request):
        self.proc.stdin.write(json.dumps(request).encode() + b"\n")
        self.proc.stdin.flush()

    def _read(self):
        line = self.proc.stdout.readline()
        if not line:
            raise SimulatorError(f"simulator exited (code {self.proc.poll()})")
        response = json.loads(line)
        if "error" in response:
            raise SimulatorError(response["error"])
        return response

    def call(self, request):
        self._send(request)
        return self._read()

    def reset(self, ids, seeds, difficulties, monster_difficulties=None):
        """monster_difficulties: mob/potion density per level (training only); defaults to each level's difficulty."""
        request = {
            "op": "reset",
            "ids": [int(i) for i in ids],
            "seeds": [int(s) for s in seeds],
            "difficulties": [int(d) for d in difficulties],
        }
        if monster_difficulties is not None:
            request["monster_difficulties"] = [int(m) for m in monster_difficulties]
        return self.call(request)["envs"]

    # step is split in two so several services can simulate at the same time (see VecCornichon.step).
    def send_step(self, actions, repeat):
        self._send({"op": "step", "actions": np.asarray(actions, dtype=int).tolist(), "repeat": int(repeat)})

    def read_step(self):
        return self._read()["envs"]

    def snapshot(self, env_id=0):
        return self.call({"op": "snapshot", "id": int(env_id)})

    def close(self):
        if self.proc.poll() is None:
            try:
                self.call({"op": "close"})
                self.proc.wait(timeout=10)
            except (OSError, SimulatorError, subprocess.TimeoutExpired):
                self.proc.kill()
        for stream in (self.proc.stdin, self.proc.stdout):
            if stream:
                stream.close()


def grid_cells(encoded, height, width):
    """Base64 grid (two little-endian bytes per cell) -> (height, width) int bitmask, bits in GRID_CHANNELS order."""
    return np.frombuffer(base64.b64decode(encoded), dtype="<u2").reshape(height, width).astype(np.int32)


def decode_grid(row, height, width):
    """Base64 bitmask grid -> float32 array (channels, height, width), channel order GRID_CHANNELS."""
    cells = grid_cells(row["grid"], height, width)
    shifts = np.arange(len(GRID_CHANNELS)).reshape(-1, 1, 1)
    return ((cells[None] >> shifts) & 1).astype(np.float32)
