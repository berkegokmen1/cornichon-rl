# Cornichon RL

PPO agent that plays **the real Cornichon game**: the same `core` code as `./gradlew desktop:run` (maze generator,
Box2D world, contact listener, mobs, wizards' fireballs, potions, sphere buff), run headless with no GL context.

```
Python (cornichon_rl/)                            Java (headless/)
  ppo.py  ── VecCornichon (env.py) ── SimService ──stdin/stdout JSON lines──  HeadlessService
              obs/reward/resets       (service.py)                             └─ Simulation × N
                                                                                  └─ core: Level.tick(), PlayerController
```

## Setup (per node: JDK + a venv on local disk)

```bash
sudo apt-get install -y openjdk-17-jdk-headless
GRADLE_USER_HOME=/data/local/berke/cache/gradle ./gradlew :headless:installDist --no-daemon
scripts/install_simulator.sh     # later rebuilds: tests the new simulator, then swaps it in under running trainings
uv venv --python 3.12 /opt/berke/envs/cornichon-rl
uv pip install --python /opt/berke/envs/cornichon-rl/bin/python -e '.[dev]' \
  --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
/opt/berke/envs/cornichon-rl/bin/python -m pytest
```

Venvs: `d4` → `/data/local/berke/envs/cornichon-rl`, `d5` → `/opt/berke/envs/cornichon-rl`. Use a node-local
`GRADLE_USER_HOME`: `~/.gradle` is on NFS and VS Code's Gradle daemon holds its lock.

## Train, evaluate

```bash
scripts/train.sh 0 ppo_v1                      # tmux session ppo_v1, configs/default.yaml, physical GPU 0
scripts/train.sh 0 ppo_v1 --resume             # continue /data/local/berke/cornichon-rl/runs/ppo_v1/latest.pt
python -m cornichon_rl.ppo --config configs/smoke.yaml        # 20-second CPU plumbing check
python -m cornichon_rl.evaluate --checkpoint RUN_DIR/best.pt --difficulties 1 2 3 --episodes 100
python -m cornichon_rl.evaluate --policy random --difficulties 1 5 10                # baseline
python -m cornichon_rl.render --checkpoint RUN_DIR/best.pt --out demo.gif          # GIF of held-out mazes
python scripts/scripted_bot.py --difficulty 1                                       # solvability check
python scripts/create_wandb_workspace.py      # curated W&B view (berkegokmen1/cornichon-rl)
```

Any config key can be overridden on the command line: `ppo.lr=1e-4 env.num_envs=128 reward.death=-10`.

See `docs/architecture.md` (simulator, observation, actions), `docs/reward-design.md`, `docs/cluster-training.md`.
